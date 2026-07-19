"""``StandardFactorEvaluator``: the default implementation of the 8 mandatory sections.

Design ``tmp/design/factor_eval_contract_v0.1.md`` §10 step 2. Every section is a
cheap reduction of the eval-IR (:mod:`analytics.eval.ir`) — the IR is built ONCE,
vectorized, and nothing here touches the panel again.

WHAT IS REAL AND WHAT IS SKIPPED (the point of the contract is that you can tell):

    predictive_power    real  — IC / ICIR / hypothesis win rate / Newey-West t
    return_risk         real  — quantile NAVs / net long-short / monotonicity
    stability_cost      real  — turnover / rank autocorrelation / cost gradient
    purity              real ONLY when EvalContext.known_factors is supplied,
                              Skipped otherwise (no anchors = no purity claim). When
                              real it also carries the Incremental verdict axis's
                              fact: the factor's IC AFTER residualizing it on the
                              WHOLE book jointly (incremental_ic_ir).
    oos_generalization  real ONLY when EvalConfig.oos_split is set,
                              Skipped otherwise ("no OOS evidence")
    execution_capacity  Skipped unless EvalContext.execution_capacity carries
                              facts MEASURED ELSEWHERE — this evaluator does not
                              run the I5b/I5f machinery and will not pretend to
    data_coverage       real
    caveats             real

Because ``execution_capacity`` is Skipped by default, the Tradable axis is
NOT_ASSESSED, and because ``purity`` is Skipped without a book the Incremental axis
is NOT_ASSESSED too — so a default run CANNOT reach Adopt (design §6, v0.5: Adopt
needs all three axes PASS) and tops out at Watch. That is deliberate: an untested
execution path is not evidence of tradability, and a factor judged with no known
book has not shown it adds anything. Both are disclosed in the section reasons,
not hidden.

NEVER FABRICATE, NEVER SILENTLY OMIT: a metric that cannot be computed is NaN or
an explicit Skipped(reason), never a plausible-looking number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from analytics.eval.config import EvalConfig
from analytics.eval.evaluator import FactorEvaluator
from analytics.eval.report import FactorEvalReport
from analytics.eval.verdict import VerdictThresholds
from analytics.eval.ir import (
    DECAY_LAGS,
    EvalContext,
    StandardEvalIR,
    build_eval_ir,
    cross_section_corr,
)
from analytics.eval.sections import Section, Skipped
from analytics.eval.stats import (
    DEFAULT_CONFIDENCE,
    as_float,
    effective_sample_size,
    half_life,
    hypothesis_win_rate,
    information_ratio_ci,
    mean_ci,
    newey_west_t,
    sortino,
    spearman,
)
from analytics.factor import ic_summary
from analytics.performance import performance_summary
from analytics.quantstats_adapter import quantstats_performance
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.spec import FactorSpec

#: The synthetic-leg caveat the project pays for every time it forgets it (I5d).
LONG_SHORT_NOTE = (
    "QN-Q1 is a SYNTHETIC LONG-ONLY LEG DIFFERENCE (the top bucket's mean return "
    "minus the bottom bucket's), NOT a dollar-neutral executed portfolio: no "
    "short book, no borrow, no financing, and the two legs are never netted into "
    "one order. Costs are charged as turnover x fee_rate on each leg. Sharpe / "
    "Sortino / maxDD / vol below are computed on the HYPOTHESIS-ALIGNED "
    "(expected_ic_sign x) base-cost spread, so a -1 factor reads in its own "
    "direction. Buckets are formed from the FACTOR cross-section alone; a bucket "
    "mean covers only the members whose forward return exists."
)


def _ic_span_days(ic: pd.Series) -> float:
    """Calendar days spanned by the SETTLED part of the IC series (gate part B).

    Only periods that actually produced a finite IC count: a period that never
    settled did not extend the evidence, so it must not extend the span either
    (the same reason ``settled_rebalances`` counts finite ICs). NaN when the
    index is not dates or fewer than two periods settled — the verdict reports
    that as UNKNOWN and FAILS the gate rather than passing on a guess.
    """
    settled = ic.dropna()
    if len(settled) < 2:
        return float("nan")
    try:
        dates = pd.DatetimeIndex(settled.index)
    except (TypeError, ValueError):
        return float("nan")
    span = dates.max() - dates.min()
    return float(span.days)


def _shift_within_symbol(series: pd.Series, lag: int) -> pd.Series:
    """The panel value as of ``lag`` periods ago, never reaching across symbols."""
    return series.groupby(level=SYMBOL_LEVEL, sort=False).shift(lag)


def _mean(series: pd.Series) -> float:
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _residualize_cross_section(y: pd.Series, x: pd.Series) -> pd.Series:
    """Per-date cross-sectional OLS residual of ``y`` on ``x`` — NO date loop.

    Single regressor, so the per-date beta is ``cov_t(x, y) / var_t(x)`` and the
    whole thing is grouped transforms. A date whose regressor has no variance
    yields NaN (the regression is undefined there) rather than a passthrough that
    would look like a successful orthogonalization.
    """
    frame = (
        pd.DataFrame({"y": y, "x": x})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    grouped = frame.groupby(level=DATE_LEVEL, sort=False)
    yc = frame["y"] - grouped["y"].transform("mean")
    xc = frame["x"] - grouped["x"].transform("mean")
    cov = (xc * yc).groupby(level=DATE_LEVEL, sort=False).transform("sum")
    var = (xc * xc).groupby(level=DATE_LEVEL, sort=False).transform("sum")
    beta = cov / var.where(var > 0)
    return (yc - beta * xc).rename("residual")


def _residualize_on_book(factor: pd.Series, anchors: pd.DataFrame) -> pd.Series:
    """Per-date residual of ``factor`` on the WHOLE anchor book — NO date loop.

    This is the Incremental axis's engine: the residual of the factor after ALL
    known factors have been projected out per date, so its IC vs forward returns is
    what the factor adds BEYOND the book (design §6, v0.5). Multi-regressor OLS is
    done by sequential (modified Gram-Schmidt / Frisch-Waugh) orthogonalization —
    each anchor is first residualized on the anchors already processed, then the
    running factor residual is residualized on it. Projecting onto an orthogonal
    basis of the book's span is EXACTLY the multi-regressor residual, and every
    step reuses the vectorized single-regressor :func:`_residualize_cross_section`.

    The only loops are over the ANCHOR COLUMNS (a small constant, like
    ``DECAY_LAGS``) — NOT over dates/periods, so this stays loop-free in the sense
    design §8 cares about. A perfect collinearity leaves a zero residual, whose IC
    is then NaN (an honest "undefined"), never a fabricated number.
    """
    residual = factor
    basis: list[pd.Series] = []
    for column in anchors.columns:
        component = anchors[column].astype(float)
        for prior in basis:
            component = _residualize_cross_section(component, prior)
        basis.append(component)
        residual = _residualize_cross_section(residual, component)
    return residual


class StandardFactorEvaluator(FactorEvaluator):
    """Default evaluator: reduce the vectorized eval-IR into the 8 sections.

    Stateless — every section reads only the IR it is handed, so one instance may
    evaluate any number of factors.
    """

    # -- the IR seam --------------------------------------------------------

    def build_ir(
        self,
        factor_panel: pd.Series | pd.DataFrame,
        spec: FactorSpec,
        cfg: EvalConfig,
        ctx: object | None = None,
    ) -> StandardEvalIR:
        """Build the design §8 four objects once (see :func:`analytics.eval.ir.build_eval_ir`)."""
        if ctx is not None and not isinstance(ctx, EvalContext):
            raise TypeError(
                f"StandardFactorEvaluator expects an analytics.eval.ir.EvalContext "
                f"(price_panel / forward_returns / known_factors / ...); got "
                f"{type(ctx).__name__}."
            )
        return build_eval_ir(factor_panel, spec, cfg, ctx)

    def evaluate_with_ir(
        self,
        factor_panel: pd.Series | pd.DataFrame,
        spec: FactorSpec,
        cfg: EvalConfig,
        ctx: object | None = None,
        thresholds: VerdictThresholds | None = None,
    ) -> tuple[FactorEvalReport, StandardEvalIR]:
        """Like :meth:`evaluate`, but also return the IR (for dashboard rendering).

        Mirrors the frozen ABC ``evaluate`` template step for step. It exists
        because the contract ``evaluate`` returns ONLY the report (its signature is
        frozen), whereas :func:`analytics.eval.figures.render_factor_dashboard`
        needs the IR's ``ic`` / ``quantile_returns`` series. Building the IR once
        here avoids a second ``build_ir`` pass, and reusing the identical
        assemble -> validate -> with_verdict sequence keeps the two entry points in
        lockstep (the report is byte-identical to ``evaluate``'s).
        """
        ir = self.build_ir(factor_panel, spec, cfg, ctx)
        sections = [getattr(self, name)(ir) for name in self.SECTION_ORDER]
        report = FactorEvalReport.assemble(spec, cfg, sections, thresholds=thresholds)
        report.validate_all_mandatory_present()
        return report.with_verdict(thresholds), ir

    # -- 2. Predictive Power ------------------------------------------------

    def predictive_power(self, ir: StandardEvalIR) -> Section | Skipped:
        """Rank IC / ICIR / hypothesis win rate / Newey-West t / IC decay."""
        ic = ir.ic
        if ic.notna().sum() == 0:
            return Skipped(
                "predictive_power",
                reason=(
                    f"no evaluation period produced a finite rank IC across "
                    f"{ir.n_rebalances} period(s): every cross-section was empty, "
                    f"smaller than 2 usable (factor, forward-return) pairs, or had "
                    f"zero variance on one side. There is nothing to summarize."
                ),
            )
        sign = ir.spec.expected_ic_sign
        rank_summary = ic_summary(ic)
        pearson_summary = ic_summary(ir.ic_pearson)
        nw = newey_west_t(ic)
        # N_eff-based CIs (design §6, v0.6). ic_ir_ci_low is what the Predictive
        # verdict axis gates on (the LOWER bound in the expected direction), NOT the
        # point. IC-mean carries a CI too, reported for the reader but not gated on.
        ir_ci = information_ratio_ci(ic, confidence=DEFAULT_CONFIDENCE)
        mean_interval = mean_ci(ic, confidence=DEFAULT_CONFIDENCE)

        payload: dict[str, object] = {
            "expected_ic_sign": sign,
            "ic_mean": rank_summary["ic_mean"],
            "ic_mean_se": mean_interval["se"],
            "ic_mean_ci_low": mean_interval["ci_low"],
            "ic_mean_ci_high": mean_interval["ci_high"],
            "ic_std": as_float(ic.dropna().std(ddof=1)) if ic.notna().sum() > 1 else float("nan"),
            "ic_ir": rank_summary["ic_ir"],
            "ic_ir_se": ir_ci["se"],
            "ic_ir_ci_low": ir_ci["ci_low"],
            "ic_ir_ci_high": ir_ci["ci_high"],
            "ic_ir_ci_confidence": ir_ci["confidence"],
            "ic_ir_ci_n_eff": ir_ci["n_eff"],
            "ic_win_rate": hypothesis_win_rate(ic, sign),
            "ic_nw_t": nw["t"],
            "ic_t_iid_unadjusted": nw["t_iid"],
            "ic_nw_bandwidth_lags": int(nw["lags"]),
            "ic_periods_finite": int(ic.notna().sum()),
            "ic_periods_nan_dropped": int(nw["n_dropped"]),
            "ic_pearson_mean": pearson_summary["ic_mean"],
            "ic_pearson_ir": pearson_summary["ic_ir"],
        }
        for lag in DECAY_LAGS:
            stale = cross_section_corr(
                _shift_within_symbol(ir.factor, lag),
                ir.forward_returns,
                rank=True,
                dates=ir.dates,
            )
            payload[f"ic_decay_stale_lag_{lag}_mean"] = _mean(stale)

        note = (
            "Rank IC (Spearman) is the primary; the Pearson IC is secondary "
            "(design §A). ic_win_rate is the share of finite periods whose IC "
            "carries the EXPECTED sign, so 0.5 is a coin flip. "
            "ic_nw_t is the Newey-West (Bartlett kernel, automatic bandwidth "
            "floor(4*(n/100)^(2/9))) autocorrelation-corrected t of the mean IC; "
            "ic_t_iid_unadjusted is the naive t that assumes independent periods "
            "and OVERSTATES significance on an autocorrelated IC series — the two "
            "are shown together so the gap is visible. The autocovariances are "
            "computed ON THE TIME GRID: a NaN period (ic_periods_nan_dropped) "
            "breaks the lag pair rather than being dropped first and silently "
            "bridged, so a gap never manufactures an autocorrelation that is not "
            "there. "
            "ic_decay_stale_lag_k_mean is the mean rank IC of the factor AS OF k "
            "periods ago against the SAME horizon-h forward return, i.e. how much "
            "predictive power a stale signal retains (lag 0 is ic_mean). It is NOT "
            "the IC at a longer forward horizon — that would need a second R_h. "
            "ic_ir_ci_low/high is the ICIR's N_eff-based 95% confidence interval "
            "(Lo Sharpe SE with N->N_eff); the Predictive verdict axis gates on the "
            "LOWER bound in the expected direction, not the point, so a wide interval "
            "from a noisy or autocorrelated IC series cannot buy a PASS."
        )
        return Section("predictive_power", payload=payload, note=note)

    # -- 3. Return & Risk ---------------------------------------------------

    def return_risk(self, ir: StandardEvalIR) -> Section | Skipped:
        """Quantile NAVs / net long-short by cost / monotonicity / risk metrics."""
        quantiles = ir.quantile_returns
        if quantiles.empty or quantiles.notna().to_numpy().sum() == 0:
            return Skipped(
                "return_risk",
                reason=(
                    f"no (date, bucket) cell has a realized forward return across "
                    f"{ir.n_rebalances} period(s) and {ir.cfg.n_quantiles} buckets, "
                    f"so there is no quantile return matrix to reduce."
                ),
            )
        sign = ir.spec.expected_ic_sign
        top, bottom = ir.cfg.long_short
        buckets = list(quantiles.columns)

        bucket_mean = {int(q): as_float(quantiles[q].mean()) for q in buckets}
        bucket_nav = {
            int(q): as_float((1.0 + quantiles[q].dropna()).prod()) for q in buckets
        }
        bucket_periods = {int(q): int(quantiles[q].notna().sum()) for q in buckets}

        gross = (quantiles[top] - quantiles[bottom]).rename("spread")
        turnover = ir.quantile_turnover
        leg_turnover = (
            turnover[top].reindex(gross.index).fillna(0.0)
            + turnover[bottom].reindex(gross.index).fillna(0.0)
        )
        net_by_cost: dict[float, float] = {}
        cumulative_by_cost: dict[float, float] = {}
        base_net = None
        for multiplier in ir.cfg.cost_scenarios:
            net = gross - ir.ctx.fee_rate * multiplier * leg_turnover
            net_by_cost[float(multiplier)] = as_float(net.mean())
            cumulative_by_cost[float(multiplier)] = as_float(
                (1.0 + net.dropna()).prod() - 1.0
            )
            if abs(multiplier - 1.0) < 1e-9:
                base_net = net

        payload: dict[str, object] = {
            "long_short_legs": f"Q{top} - Q{bottom}",
            "quantile_mean_return": bucket_mean,
            "quantile_final_nav": bucket_nav,
            "quantile_periods_with_return": bucket_periods,
            # RAW Spearman of bucket index vs bucket mean return; the verdict is
            # the ONE place that multiplies by expected_ic_sign.
            "monotonicity_spearman": spearman(
                [float(q) for q in buckets], [bucket_mean[int(q)] for q in buckets]
            ),
            "gross_long_short_mean": as_float(gross.mean()),
            "net_long_short_by_cost": net_by_cost,
            "net_long_short_cumulative_by_cost": cumulative_by_cost,
            "fee_rate_base": ir.ctx.fee_rate,
            "periods_per_year": ir.periods_per_year,
        }
        # N_eff-based CI of the base-cost QN-Q1 spread (design §6, v0.6). REPORTED
        # for the reader (b); the Tradable axis still gates on the POINT base spread
        # > 0 (the CI on this leg difference is point-only for now — c names only
        # ICIR / incremental ICIR as the lower-CI magnitude criteria).
        if base_net is not None:
            base_ci = mean_ci(sign * base_net, confidence=DEFAULT_CONFIDENCE)
            payload["net_long_short_base_se"] = base_ci["se"]
            payload["net_long_short_base_ci_low"] = base_ci["ci_low"]
            payload["net_long_short_base_ci_high"] = base_ci["ci_high"]
            payload["net_long_short_base_ci_note"] = (
                "hypothesis-aligned base-cost spread CI (N_eff-based); REPORTED, not "
                "gated — the Tradable axis PASS reads the POINT base spread > 0."
            )
        if ir.periods_per_year_is_default:
            payload["periods_per_year_basis"] = (
                f"DEFAULT {ir.periods_per_year}: rebalance label "
                f"{ir.cfg.rebalance!r} is not a recognized frequency, so the "
                f"annualization below is a fallback, not a derived fact."
            )

        if base_net is not None and base_net.notna().sum() >= 2:
            aligned = (sign * base_net).dropna()
            nav = (1.0 + aligned).cumprod()
            simple = performance_summary(nav, periods_per_year=ir.periods_per_year)
            payload["aligned_spread_annual_return"] = simple["annual_return"]
            payload["aligned_spread_sharpe"] = simple["sharpe"]
            payload["aligned_spread_volatility"] = simple["volatility"]
            payload["aligned_spread_max_drawdown"] = simple["max_drawdown"]
            payload["aligned_spread_sortino"] = sortino(aligned, ir.periods_per_year)
            payload["aligned_spread_final_nav"] = as_float(nav.iloc[-1])
            # report-only cross-check; the simple numbers above stay authoritative
            # and the backend is disclosed (INV-007), never silently faked.
            cross = quantstats_performance(
                aligned,
                periods_per_year=ir.periods_per_year,
                simple_fallback={
                    "cagr": simple["annual_return"],
                    "sharpe": simple["sharpe"],
                    "volatility": simple["volatility"],
                    "max_drawdown": simple["max_drawdown"],
                },
            )
            payload["quantstats_cross_check"] = cross

        return Section("return_risk", payload=payload, note=LONG_SHORT_NOTE)

    # -- 4. Stability & Cost ------------------------------------------------

    def stability_cost(self, ir: StandardEvalIR) -> Section | Skipped:
        """Turnover / factor rank autocorrelation -> half-life / cost gradient."""
        if ir.quantile_labels.empty:
            return Skipped(
                "stability_cost",
                reason=(
                    "no (date, symbol) row carries a finite factor value, so there "
                    "is no bucket membership to measure turnover on and no "
                    "cross-section to autocorrelate."
                ),
            )
        top, bottom = ir.cfg.long_short
        turnover = ir.quantile_turnover

        payload: dict[str, object] = {
            "turnover_mean_by_quantile": {
                int(q): as_float(turnover[q].mean()) for q in turnover.columns
            },
            "turnover_mean_long_short_legs": as_float(
                (turnover[top] + turnover[bottom]).mean()
            ),
            "fee_rate_base": ir.ctx.fee_rate,
        }

        autocorr: dict[int, float] = {}
        for lag in DECAY_LAGS:
            rho = cross_section_corr(
                ir.factor, _shift_within_symbol(ir.factor, lag), rank=True, dates=ir.dates
            )
            autocorr[lag] = _mean(rho)
        payload["factor_rank_autocorr_by_lag"] = autocorr
        payload["half_life_periods"] = half_life(autocorr.get(1, float("nan")))

        # The cost GRADIENT (design §4: base / 2x / 4x, never a single point). Read
        # off the same net spreads return_risk reports, so the two cannot disagree.
        gross = ir.quantile_returns[top] - ir.quantile_returns[bottom]
        leg_turnover = (
            turnover[top].reindex(gross.index).fillna(0.0)
            + turnover[bottom].reindex(gross.index).fillna(0.0)
        )
        base_cost = float("nan")
        drag: dict[float, float] = {}
        for multiplier in ir.cfg.cost_scenarios:
            cost = as_float((ir.ctx.fee_rate * multiplier * leg_turnover).mean())
            drag[float(multiplier)] = cost
            if abs(multiplier - 1.0) < 1e-9:
                base_cost = cost
        payload["mean_cost_per_period_by_scenario"] = drag
        payload["extra_cost_vs_base_by_scenario"] = {
            m: as_float(c - base_cost) for m, c in drag.items()
        }
        payload["annualized_cost_drag_by_scenario"] = {
            m: as_float(c * ir.periods_per_year) for m, c in drag.items()
        }

        note = (
            "Turnover is sum_i |w_i(t) - w_i(t-1)| for an equal-weight bucket "
            "(both sides of the trade), and turnover x fee_rate is the period's "
            "cost — the project's convention. The FIRST period is charged for "
            "establishing the book. factor_rank_autocorr_by_lag is the mean "
            "per-date rank correlation of the factor with itself k periods "
            "earlier; half_life_periods = log(0.5)/log(rho_1), defined only for "
            "0 < rho_1 < 1 (a flipping or non-decaying signal has no half-life and "
            "reports NaN). The cost gradient re-charges the SAME trades at k x the "
            "fee: turnover and gross returns are identical across scenarios, only "
            "the cost line moves."
        )
        return Section("stability_cost", payload=payload, note=note)

    # -- 5. Purity ----------------------------------------------------------

    def purity(self, ir: StandardEvalIR) -> Section | Skipped:
        """Correlation with known anchors / orthogonalized IC. VIF is NOT computed."""
        known = ir.ctx.known_factors
        if known is not None and not isinstance(known, pd.DataFrame):
            # A wrong TYPE is a caller bug, not an absent anchor set: reporting it
            # as "was not supplied" would be a misleading diagnostic.
            raise TypeError(
                f"EvalContext.known_factors must be a MultiIndex(date, symbol) "
                f"DataFrame of anchor factors (one column each); got "
                f"{type(known).__name__}."
            )
        if known is None or len(known.columns) == 0:
            return Skipped(
                "purity",
                reason=(
                    "EvalContext.known_factors was not supplied, so there is no "
                    "anchor panel to correlate against, orthogonalize on, or "
                    "compute a VIF from. Purity cannot be claimed from the factor "
                    "and its forward returns alone. Pass an already-processed "
                    "MultiIndex(date, symbol) frame of anchor factors (the project's "
                    "independently validated value_ep / value_bp / volatility_20, "
                    "plus classic size / momentum) to populate this section. "
                    f"Already stripped upstream per EvalConfig: "
                    f"neutralization={list(ir.cfg.neutralization)!r} "
                    f"({ir.cfg.industry_level}), winsorize={ir.cfg.winsorize!r}, "
                    f"standardize={ir.cfg.standardize!r}."
                ),
            )
        anchors = pd.DataFrame(known)
        raw_ic = _mean(ir.ic)
        payload: dict[str, object] = {
            "family": ir.spec.family,
            "neutralization_applied_upstream": list(ir.cfg.neutralization),
            "industry_level": ir.cfg.industry_level,
            "anchor_factors": [str(c) for c in anchors.columns],
            "ic_mean_raw": raw_ic,
        }
        correlations: dict[str, float] = {}
        orthogonal_ic: dict[str, float] = {}
        for column in anchors.columns:
            anchor = anchors[column].astype(float)
            correlations[str(column)] = _mean(
                cross_section_corr(ir.factor, anchor, rank=True, dates=ir.dates)
            )
            residual = _residualize_cross_section(ir.factor, anchor)
            orthogonal_ic[str(column)] = _mean(
                cross_section_corr(
                    residual, ir.forward_returns, rank=True, dates=ir.dates
                )
            )
        payload["mean_rank_corr_with_anchor"] = correlations
        payload["ic_mean_orthogonalized_vs_anchor"] = orthogonal_ic

        # -- the Incremental axis's fact (design §6, v0.5) -------------------
        # The orthogonalized IC vs the WHOLE book (multi-regressor residual), NOT
        # one anchor at a time: "does the factor add anything the book does not
        # already have?" The verdict reads incremental_ic_ir exactly as it reads
        # ic_ir for the Predictive axis; RAW, so the hypothesis is applied there.
        book_residual = _residualize_on_book(ir.factor, anchors)
        book_ortho_ic = cross_section_corr(
            book_residual, ir.forward_returns, rank=True, dates=ir.dates
        )
        book_summary = ic_summary(book_ortho_ic)
        # The CI is computed on the ORTHOGONALIZED IC series' OWN N_eff (design §6,
        # v0.6): the Incremental axis gates on incremental_ic_ir_ci_low, so the
        # residual's own autocorrelation — not the raw IC's — sets the interval.
        book_ir_ci = information_ratio_ci(book_ortho_ic, confidence=DEFAULT_CONFIDENCE)
        payload["known_factors_supplied"] = True
        payload["ic_mean_orthogonalized_vs_book"] = book_summary["ic_mean"]
        payload["incremental_ic_mean"] = book_summary["ic_mean"]
        payload["incremental_ic_ir"] = book_summary["ic_ir"]
        payload["incremental_ic_ir_se"] = book_ir_ci["se"]
        payload["incremental_ic_ir_ci_low"] = book_ir_ci["ci_low"]
        payload["incremental_ic_ir_ci_high"] = book_ir_ci["ci_high"]
        payload["incremental_ic_ir_ci_n_eff"] = book_ir_ci["n_eff"]

        payload["vif"] = None
        payload["vif_status"] = (
            "NOT COMPUTED — a VIF needs a MULTI-regressor per-date cross-sectional "
            "design; this section orthogonalizes against ONE anchor at a time "
            "(vectorized single-regressor OLS). Read the pairwise correlations "
            "together: several highly correlated anchors would each individually "
            "look survivable here while being collectively collinear."
        )
        note = (
            "mean_rank_corr_with_anchor is the mean per-date cross-sectional rank "
            "correlation with each anchor. ic_mean_orthogonalized_vs_anchor is the "
            "mean rank IC of the factor AFTER removing ONE anchor per date by "
            "cross-sectional OLS (residual vs the same R_h); compare it with "
            "ic_mean_raw — a collapse means the anchor explained the signal. "
            "incremental_ic_ir / incremental_ic_mean drive the INCREMENTAL VERDICT "
            "AXIS: they are the ICIR and mean rank IC of the factor residualized on "
            "the WHOLE book jointly (multi-regressor, per date) — what the factor "
            "adds BEYOND the known set. A value ~ 0 means the factor is redundant "
            "with the book (a hard Reject on the Incremental axis), however strong "
            "its RAW IC. Anchors are taken as supplied; this evaluator does not "
            "verify that they were processed the same way as the factor."
        )
        return Section("purity", payload=payload, note=note)

    # -- 6. OOS & Generalization (the project's core lesson) ----------------

    def oos_generalization(self, ir: StandardEvalIR) -> Section | Skipped:
        """Train/test sign consistency by REALIZED date. Drives the Adopt verdict."""
        cfg = ir.cfg
        if cfg.oos_split is None:
            return Skipped(
                "oos_generalization",
                reason=(
                    "cfg.oos_split not set — no OOS evidence. This run measured the "
                    "factor on ONE window with no holdout, so generalization is NOT "
                    "established and the verdict cannot reach Adopt. The project's "
                    "standing lesson (P3-3/P3-4: IC signs flipped train->test in "
                    "almost every cell) is that in-sample strength does not "
                    "extrapolate."
                ),
            )
        try:
            split = pd.Timestamp(cfg.oos_split)
        except (TypeError, ValueError) as exc:
            # Loud, not a silent downgrade: a typo'd split must never be reported
            # as "we chose not to do OOS".
            raise ValueError(
                f"EvalConfig.oos_split={cfg.oos_split!r} is not a parseable date, so "
                f"the train/test boundary is undefined. Silently degrading to 'no "
                f"OOS split' would report a CONFIG TYPO as a deliberate choice."
            ) from exc

        sign = ir.spec.expected_ic_sign
        realized = pd.to_datetime(ir.realized_date, errors="coerce")
        is_test = realized >= split
        is_train = realized < split
        unrealized = realized.isna()

        train_ic = ir.ic[is_train.to_numpy()]
        test_ic = ir.ic[is_test.to_numpy()]
        train_mean, test_mean = _mean(train_ic), _mean(test_ic)

        payload: dict[str, object] = {
            "oos_split": str(cfg.oos_split),
            "split_basis": "realized date (t+h), never the signal date",
            "train_periods_settled": int(train_ic.notna().sum()),
            "test_periods_settled": int(test_ic.notna().sum()),
            "periods_never_realized": int(unrealized.sum()),
            "expected_ic_sign": sign,
            "train_ic_mean": train_mean,
            "test_ic_mean": test_mean,
            "independent_cells_declared": len(cfg.independent_cells),
            "independent_cells_evaluated": 0,
        }

        available = train_ic.notna().sum() > 0 and test_ic.notna().sum() > 0
        payload["oos_available"] = bool(available)

        # The holdout, split in two by realized date: the contract's Adopt rule
        # wants the expected sign in BOTH holdout subperiods (design §6), not just
        # on the holdout average, which one lucky stretch could carry.
        test_dates = list(ir.dates[is_test.to_numpy()])
        first_mean = second_mean = float("nan")
        if len(test_dates) >= 2:
            middle = len(test_dates) // 2
            first_mean = _mean(test_ic.reindex(test_dates[:middle]))
            second_mean = _mean(test_ic.reindex(test_dates[middle:]))
        payload["holdout_subperiod_1_ic_mean"] = first_mean
        payload["holdout_subperiod_2_ic_mean"] = second_mean

        sign_consistent = bool(
            available
            and math.isfinite(first_mean)
            and math.isfinite(second_mean)
            and sign * first_mean > 0
            and sign * second_mean > 0
        )
        sign_flipped = bool(
            available and math.isfinite(test_mean) and sign * test_mean < 0
        )
        test_monotonicity = float("nan")
        if available:
            test_quantiles = ir.quantile_returns[is_test.to_numpy()]
            bucket_means = [as_float(test_quantiles[q].mean()) for q in test_quantiles.columns]
            test_monotonicity = spearman(
                [float(q) for q in test_quantiles.columns], bucket_means
            )

        payload["sign_consistent"] = sign_consistent
        payload["sign_flipped"] = sign_flipped
        payload["test_monotonicity_spearman"] = test_monotonicity
        payload["test_monotonicity_aligned"] = as_float(sign * test_monotonicity)
        # The contract's monotonicity_reversed means an INDEPENDENT-CELL reversal
        # (verdict.py: "# independent-cell reversal (I5e)", and its Reject reason
        # literally reads "reversed on an independent cell"). This evaluator scores
        # ONE cell and has just reported independent_cells_evaluated=0, so writing
        # a SAME-CELL reversal into that key would make the report claim an
        # independent check ran and failed when none ran at all — a fabricated
        # provenance in the very payload that discloses zero independent cells.
        # Left False: unknown is not evidence, and the contract is explicit that an
        # absent fact must never manufacture a hard Reject. The same-cell number is
        # reported above as test_monotonicity_spearman, under its own honest name.
        payload["monotonicity_reversed"] = False
        payload["monotonicity_reversed_status"] = (
            "NOT ASSESSED — this key means a reversal on an INDEPENDENT cell "
            "(different universe / non-overlapping window; the I5e signature: "
            "CSI500 +1.0 -> CSI300 -0.5). This evaluator scores ONE cell and cannot "
            "produce that evidence; a declared independent universe must be "
            "evaluated as its OWN run and compared across runs. The same-cell "
            "holdout figure is test_monotonicity_spearman "
            f"({test_monotonicity:.4f}), aligned to the hypothesis "
            f"({sign * test_monotonicity:+.4f}) — read it as a diagnostic, NOT as "
            f"independent-cell evidence."
        )

        note = (
            "Periods are assigned to train/test by the REALIZED date (t+h), never "
            "the signal date: a signal given at t is only out-of-sample once t+h "
            "has happened (P3-3 fixed exactly this, and the fix moved the numbers "
            "materially). sign_consistent requires the EXPECTED IC sign in BOTH "
            "halves of the holdout, not merely on its average. sign_flipped means "
            "the holdout mean IC CONTRADICTS the stated hypothesis — a hard "
            "Reject. This evaluator scores ONE cell: independent_cells is a HUMAN "
            "declaration it can neither run nor verify, so "
            "independent_cells_evaluated is always 0 here and "
            "monotonicity_reversed (which the contract defines as an "
            "INDEPENDENT-cell reversal) is therefore always left False — see "
            "monotonicity_reversed_status. A holdout split inside ONE "
            "window/universe is NOT independent generalization: a declared "
            "independent universe must be evaluated as its own run."
        )
        return Section("oos_generalization", payload=payload, note=note)

    # -- 7. Execution & Capacity --------------------------------------------

    def execution_capacity(self, ir: StandardEvalIR) -> Section | Skipped:
        """I5b fill feasibility / I5f capacity — reported ONLY if measured elsewhere."""
        facts = ir.ctx.execution_capacity
        if facts is None:
            return Skipped(
                "execution_capacity",
                reason=(
                    f"NOT WIRED: this evaluator reduces the eval-IR (a factor panel "
                    f"and its forward returns) and never runs the backtest engine, so "
                    f"it observes neither I5b raw stk_limit fill feasibility "
                    f"(cfg.limit_feasibility={ir.cfg.limit_feasibility}) nor the I5f "
                    f"single-minute capacity diagnostic "
                    f"(cfg.capacity_notional={ir.cfg.capacity_notional}, "
                    f"max_participation_rate={ir.cfg.max_participation_rate}). Both "
                    f"need minute bars, raw stk_limit rows and the event engine — "
                    f"none of which is an IR input. A capacity number invented from "
                    f"the factor panel alone would be a fiction. Supply facts "
                    f"MEASURED by a P-I5b / P-I5f run via "
                    f"EvalContext.execution_capacity to populate this section. "
                    f"CONSEQUENCE: tradability is UNKNOWN, so this run cannot reach "
                    f"the Adopt verdict — an untested execution path is not evidence "
                    f"of tradability."
                ),
            )
        if not isinstance(facts, Mapping):
            raise TypeError(
                f"EvalContext.execution_capacity must be a mapping of measured "
                f"execution facts (recognized: 'tradable', 'capacity_sufficient'); "
                f"got {type(facts).__name__}."
            )
        payload: dict[str, object] = dict(facts)
        payload["source"] = (
            "MEASURED OUTSIDE this evaluator and supplied via "
            "EvalContext.execution_capacity; these numbers are reported, not "
            "produced here."
        )
        for key in ("tradable", "capacity_sufficient"):
            if not isinstance(facts.get(key), bool):
                # Left absent/unknown on purpose: report.extract_verdict_inputs
                # reads a non-bool as None, i.e. unknown, which can never earn an
                # Adopt. Say so rather than let a missing key look measured.
                payload[f"{key}_status"] = (
                    f"UNKNOWN — EvalContext.execution_capacity carries no boolean "
                    f"{key!r}; the verdict treats it as not established."
                )
        note = (
            "This section is a PASS-THROUGH of facts measured by the execution "
            "machinery (I5b raw stk_limit fill gating / I5f single-minute capacity "
            "at a target notional). The evaluator does not verify them, does not "
            "re-run the engine, and cannot detect a stale or mismatched run."
        )
        return Section("execution_capacity", payload=payload, note=note)

    # -- 8. Data & Coverage -------------------------------------------------

    def data_coverage(self, ir: StandardEvalIR) -> Section | Skipped:
        """Coverage / dropped symbols / NaN rates / cross-section sizes / sample size."""
        factor = ir.factor
        total = int(len(factor))
        symbols = factor.index.get_level_values(SYMBOL_LEVEL)
        evaluated = pd.Index(pd.unique(symbols))

        finite = np.isfinite(factor.to_numpy(dtype=float))
        warmup = ir.warmup_mask.to_numpy(dtype=bool)
        post_warmup = int((~warmup).sum())

        # The verdict's THREE-PART sample gate (design §6, v0.3) reads
        # settled_rebalances + effective_samples + span_days. All three come from
        # the IR's IC series, computed ONCE here — a raw count is not a sample
        # size under a daily rebalance.
        ess = effective_sample_size(ir.ic)
        span = _ic_span_days(ir.ic)

        payload: dict[str, object] = {
            # -- the sample gate's three facts --
            "settled_rebalances": ir.settled_rebalances,
            "effective_samples": as_float(ess["n_eff"]),
            "span_days": span,
            # how N_eff was reached, so the gate's arithmetic is auditable rather
            # than a bare number the reader must trust.
            "effective_samples_lags": as_float(ess["lags"]),
            "effective_samples_lags_nw_floor": as_float(ess["lags_nw"]),
            "effective_samples_sum_rho": as_float(ess["sum_rho"]),
            "effective_samples_denominator": as_float(ess["denominator"]),
            "effective_samples_note": (
                str(ess["status"])
                or (
                    "N_eff = N / (1 + 2*sum_k rho_k) over the IC series, lags "
                    "truncated at the Newey-West floor then extended while rho>0, "
                    "clamped to [1, N]."
                )
            ),
            "evaluation_periods": ir.n_rebalances,
            "declared_rebalance": ir.cfg.rebalance,
            "median_period_gap_days": as_float(ir.median_period_gap_days),
            "rebalance_grid_check": ir.rebalance_grid_check,
            "symbols_evaluated": int(len(evaluated)),
            "panel_rows": total,
            "universe_is_pit": ir.cfg.universe_is_pit,
            "forward_return_source": ir.forward_return_source,
            "factor_nan_rate": as_float(1.0 - finite.sum() / total) if total else float("nan"),
            "factor_nan_rate_excluding_warmup": (
                as_float(1.0 - (finite & ~warmup).sum() / post_warmup)
                if post_warmup
                else float("nan")
            ),
            "warmup_rows_excluded": int(warmup.sum()),
            "min_history_bars": ir.spec.min_history_bars,
            "forward_return_nan_rate": as_float(
                1.0 - np.isfinite(ir.forward_returns.to_numpy(dtype=float)).sum() / total
            )
            if total
            else float("nan"),
            "cross_section_size_mean": as_float(ir.cross_section_size.mean()),
            "cross_section_size_min": as_float(ir.cross_section_size.min()),
            "cross_section_size_median": as_float(ir.cross_section_size.median()),
            "cross_section_size_max": as_float(ir.cross_section_size.max()),
            "periods_with_empty_cross_section": int((ir.cross_section_size == 0).sum()),
            "input_fields_declared": list(ir.spec.input_fields),
        }

        declared = ir.ctx.universe_symbols
        if declared:
            dropped = sorted(set(map(str, declared)) - set(map(str, evaluated)))
            payload["universe_symbols_declared"] = len(declared)
            payload["dropped_symbols_count"] = len(dropped)
            payload["dropped_symbols_share"] = as_float(len(dropped) / len(declared))
            payload["dropped_symbols_examples"] = dropped[:5]
        else:
            payload["universe_symbols_declared"] = 0
            payload["dropped_symbols_count"] = None
            payload["dropped_symbols_status"] = (
                "NOT ASSESSED — EvalContext.universe_symbols was not supplied, so "
                "the evaluator cannot tell which universe members never reached the "
                "factor panel. A silently dropped name is a coverage bias (I5d "
                "dropped 103 of 995 minute-uncovered constituents and had to say so)."
            )

        note = (
            "settled_rebalances counts periods with a FINITE rank IC — the periods "
            "that actually produced evidence; the last h periods have no realized "
            "forward return and never settle. ONE ROW OF THE FACTOR PANEL IS ONE "
            "EVALUATION PERIOD: this evaluator does NOT resample. It instead CHECKS "
            "the supplied spacing against declared_rebalance and refuses to run on a "
            "grid that contradicts it (see rebalance_grid_check), because "
            "min_rebalances — the gate that decides INSUFFICIENT-DATA — would "
            "otherwise pass on a sample that does not exist at the declared "
            "frequency. A 'NOT CHECKED' outcome means the declaration is UNVERIFIED "
            "and the sample size below is simply the rows supplied. "
            + (
                "universe_is_pit=False: constituents are NOT point-in-time, so these "
                "results carry survivorship bias."
                if not ir.cfg.universe_is_pit
                else "universe_is_pit=True: constituents are point-in-time."
            )
        )
        return Section("data_coverage", payload=payload, note=note)

    # -- 9. Caveats & Provenance --------------------------------------------

    def caveats(self, ir: StandardEvalIR) -> Section | Skipped:
        """Post-hoc / exploratory / tuned / multiple testing / sample size."""
        cfg, spec = ir.cfg, ir.spec
        items: list[str] = []
        if cfg.is_exploratory:
            items.append(
                "EXPLORATORY: this is not a return claim and must not be read as one."
            )
        if cfg.post_hoc_selected:
            items.append(
                "POST-HOC SELECTED: the factor was chosen after seeing results on "
                "(some of) this data, so this run quantifies it — it does not "
                "CONFIRM it. Independent confirmation needs a window and/or universe "
                "that took no part in the screening (P3-6 -> P3-7)."
            )
        if cfg.tuned:
            items.append(
                "TUNED: at least one parameter was fitted, so in-sample fit is "
                "optimistic by construction."
            )
        if not cfg.universe_is_pit:
            items.append(
                "NON-PIT UNIVERSE: constituents are not point-in-time — survivorship "
                "bias is present and not corrected."
            )
        if cfg.oos_split is None:
            items.append(
                "NO OOS SPLIT: a single window with no holdout. The project's own "
                "record (P3-2's single-year outperformance did not survive P3-3/P3-4) "
                "is that this cannot be extrapolated."
            )
        if len(cfg.independent_cells) == 0:
            items.append(
                "NO INDEPENDENT CELL DECLARED: even a passing OOS split inside ONE "
                "window/universe is not independent generalization (I5d held on "
                "CSI500 and reversed on CSI300 in I5e)."
            )
        items.append(
            "SYNTHETIC LONG-SHORT: the QN-Q1 spread is a long-only leg difference, "
            "not a dollar-neutral executed portfolio."
        )
        if ir.settled_rebalances < 24:
            items.append(
                f"SMALL SAMPLE: {ir.settled_rebalances} settled rebalance(s). The "
                f"project has repeatedly seen ~21-period cells reverse each other."
            )

        payload: dict[str, object] = {
            "caveats": items,
            "is_exploratory": cfg.is_exploratory,
            "post_hoc_selected": cfg.post_hoc_selected,
            "tuned": cfg.tuned,
            "universe_is_pit": cfg.universe_is_pit,
            "n_factors_screened": cfg.n_factors_screened,
            "independent_cells_declared": len(cfg.independent_cells),
            "settled_rebalances": ir.settled_rebalances,
            "factor_version": spec.version,
            "data_snapshot_id": cfg.data_snapshot_id,
            "window": f"{cfg.start} .. {cfg.end}",
        }
        screened = cfg.n_factors_screened
        if screened:
            payload["bonferroni_alpha_for_5pct_family"] = as_float(0.05 / screened)
            payload["multiple_testing_note"] = (
                f"{screened} factor(s) were screened to arrive at this one. A "
                f"nominal 5% test over {screened} candidates expects "
                f"{0.05 * screened:.2f} false positive(s); the Bonferroni-corrected "
                f"per-factor alpha is {0.05 / screened:.5f}. The reported "
                f"Newey-West t is NOT corrected for this."
            )
        else:
            payload["multiple_testing_note"] = (
                "n_factors_screened is None: the multiple-testing background is "
                "UNDECLARED, so the reported significance cannot be discounted for "
                "how many candidates were looked at. Absence of a declaration is not "
                "evidence that only one factor was tried."
            )
        return Section("caveats", payload=payload)


__all__ = ["LONG_SHORT_NOTE", "StandardFactorEvaluator"]
