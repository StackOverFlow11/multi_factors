"""run-eval-ridge-minute-return: the ninth real factor evaluation (PR-K).

Reproduces the FIFTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, reportId 4957417, §3) — the
"量岭分钟收益" factor — as a first-class
:class:`~factors.compute.intraday_derived.RidgeMinuteReturnFactor` and runs it through the
FROZEN :class:`~analytics.eval.StandardFactorEvaluator` on REAL cached A-share data
(CSI500, PIT membership) — the same contract-driven loop PR-C .. PR-J used.

This is the FIFTH reproduction from the same report and it opens a FOURTH statistic
family. The four prior ones covered a COUNT (PR-F, weak), a TIMING moment (PR-H, null) and
two PRICE LEVELS (PR-I / PR-J, both passed). This factor is a RETURN — and it is the
report's ONLY NEGATIVE peak/ridge/valley factor, so the run also tests whether the SIGN
transfers along with the family, not just the magnitude.

The factor is a DAILY signal derived DIRECTLY from the 1min cache (see
``data.clean.intraday_ridge_return.compute_ridge_minute_return``): PIT-truncate each day at
14:50, classify every visible minute with the taxonomy REUSED from PR-F
(``data.clean.intraday_volume_prv``: same-slot strictly-prior μ+σ eruptive test; a RIDGE is
an eruptive minute that is not an isolated peak), form each minute's return against the
previous VISIBLE bar of the SAME day, sum those returns over the day's ridge bars, and sum
the daily sums over the trailing 20 VALID trading days. It is executed CLOSE-TO-CLOSE
(daily default), so ``is_intraday=False``. The eval CELL is identical to PR-C..PR-J, so
this run is directly comparable to its siblings — same classification, different statistic.

RIDGE SCARCITY IS MEASURED, NOT ASSUMED, exactly as in PR-J. Ridge bars are structurally
rare (a minute must erupt AND fail the isolation test), and this factor narrows them
further: only ridge bars that CARRY A VALID RETURN count, so the day's first visible bar
drops out even when it is a ridge. The runner therefore collects the realized per-day
distribution of BOTH counts plus the day-validity rate across the whole universe and logs
them, so the return-guard attrition and any coverage regression against PR-J are visible as
numbers rather than hidden behind a threshold.

CACHE-ONLY: every input is read from the persistent tushare cache
(``artifacts/cache/tushare/v1``). The minute read is provably live-call-free (the minute
store has no fetch closure — a miss simply yields no rows); the daily / universe /
covariate endpoints go through the shared read-through cache, which on a fully-warmed cache
does zero gap fetches (disclosed via the run-log cache-stats line). Forward returns are
computed ONLY at the evaluator/analytics boundary from ``ctx.price_panel`` — the factor
computation never sees a future return.

The evaluator is run TWICE: once with NO known-factor book (the Incremental axis is
NOT_ASSESSED) and once with the project's independently-confirmed book (value_ep /
value_bp / volatility_20) so the Incremental axis measures whether the accumulated
ridge-minute return adds alpha BEYOND value / low-vol. That check matters here because a
trailing sum of returns is a plausible cousin of a REVERSAL signal, and the with-book run
is the one that says whether this is a genuinely new bet.

⚠️ THIS FACTOR'S PRE-REGISTERED SIGN IS -1, which triggers a KNOWN DEFECT in the frozen
evaluation layer: its ``aligned_spread_*`` fields compute ``sign * (gross - cost)``, which
for a negative sign becomes ``-gross + cost`` — i.e. the trading cost is ADDED BACK rather
than deducted. The frozen layer is deliberately NOT patched here. Any reading of this run's
tradability must use ``net_long_short_by_cost`` (which is sign-agnostic and correct) and
must treat ``aligned_spread_*`` as unreliable. :func:`extract_metrics` therefore surfaces
the cost-scenario net spreads explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.eval import (
    EvalConfig,
    EvalContext,
    FactorEvalReport,
    StandardFactorEvaluator,
)
from analytics.eval.figures import render_factor_dashboard
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import READ_COLUMNS
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_ridge_return import (
    RIDGE_RETURN_LOOKBACK_DAYS,
    RIDGE_RETURN_MIN_RIDGE_BARS,
    compute_ridge_minute_return,
)
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
)
from data.clean.schema import CORE_COLUMNS, DATE_LEVEL
from factors.compute.intraday_derived import RidgeMinuteReturnFactor
from factors.spec import FactorSpec
from qt.config import RootConfig, load_config
from qt.pipeline import (
    _build_cache,
    _build_universe,
    _load_panel,
    _log_run_cache_stats,
    _make_logger,
    _maybe_enrich_covariates,
    _maybe_enrich_value,
    _process_factors,
)

_LOGGER_NAME = "qt.eval_ridge_minute_return"
_REPORT_STEM = "eval_ridge_minute_return"

# Percentiles reported for the realized ridge-bar distribution (the scarcity disclosure).
_RIDGE_PCTL = (0, 10, 25, 50, 75, 90, 100)

# The counterfactual floor the coverage disclosure also reports, so PR-K's ridge coverage
# is directly comparable to PR-J's (which used 20 for its VALLEY leg).
_COMPARISON_FLOOR = 20


# --------------------------------------------------------------------------- #
# Ridge-scarcity coverage (measured, never assumed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RidgeReturnCoverage:
    """Realized ridge-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the days
    the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible bars,
    including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The headline
    ``validity_rate`` is taken over ``classifiable_days``, because a day with no baseline
    fails for a PR-F warm-up reason rather than a ridge-scarcity one and would otherwise
    make the ridge gate look worse than it is — both denominators are reported so the
    reader can check that framing.

    TWO ridge counts are tracked, because this factor gates on the narrower one: total
    ``ridge_bars`` and the ``ridge_return_bars`` subset that carries a valid minute return
    (the day's first visible bar is excluded by the within-day lag even when it is a
    ridge). Reporting both makes the return-guard attrition visible instead of implicit.
    The gate-failure counts are NOT mutually exclusive and are reported for shape, not as
    a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    ridge_return_percentiles: tuple[tuple[int, float], ...]
    ridge_return_mean: float
    ridge_bars_mean: float
    ridge_bars_median: float
    days_below_ridge_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive at the higher floor PR-J used for its
    # VALLEY leg. Quantifies exactly what the scarcity-driven threshold buys.
    valid_days_at_comparison_floor: int
    # The gates this run actually applied, so the disclosure can never describe the module
    # defaults while the run used something else.
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE
    comparison_floor: int = _COMPARISON_FLOOR

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    @property
    def return_guard_attrition(self) -> float:
        """Share of ridge bars LOST to the return guard (mean over classifiable days)."""
        if not np.isfinite(self.ridge_bars_mean) or self.ridge_bars_mean <= 0.0:
            return float("nan")
        return 1.0 - self.ridge_return_mean / self.ridge_bars_mean

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.ridge_return_percentiles)
        return (
            f"ridge scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"ridge_return_bars[{pctl} mean={self.ridge_return_mean:.1f}] "
            f"ridge_bars_mean={self.ridge_bars_mean:.1f} "
            f"ridge_bars_median={self.ridge_bars_median:.0f} "
            f"return_guard_attrition={self.return_guard_attrition:.1%} "
            f"below_ridge_gate({self.min_ridge_bars})={self.days_below_ridge_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_floor_were_{self.comparison_floor}="
            f"{self.valid_days_at_comparison_floor}"
        )


def summarize_ridge_return_coverage(
    frames: list[pd.DataFrame],
    *,
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    comparison_floor: int = _COMPARISON_FLOOR,
) -> RidgeReturnCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The floors must be the ones the RUN applied, not the module defaults — otherwise the
    disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        comparison_floor=comparison_floor,
    )
    empty = tuple((p, float("nan")) for p in _RIDGE_PCTL)
    if not frames:
        return RidgeReturnCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            ridge_return_percentiles=empty,
            ridge_return_mean=float("nan"),
            ridge_bars_mean=float("nan"),
            ridge_bars_median=float("nan"),
            days_below_ridge_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_comparison_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the
    # percentiles towards zero for a reason that has nothing to do with ridge scarcity.
    scored = classifiable >= min_classifiable
    ridge_ret = diag.loc[scored, "ridge_return_bars"].to_numpy(dtype=float)
    ridge_all = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    # The counterfactual raises the ridge floor, leaving every other gate exactly as it was.
    at_comparison = valid & (
        diag["ridge_return_bars"].to_numpy(dtype=float) >= comparison_floor
    )
    return RidgeReturnCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        ridge_return_percentiles=(
            tuple((p, float(np.percentile(ridge_ret, p))) for p in _RIDGE_PCTL)
            if ridge_ret.size
            else empty
        ),
        ridge_return_mean=float(ridge_ret.mean()) if ridge_ret.size else float("nan"),
        ridge_bars_mean=float(ridge_all.mean()) if ridge_all.size else float("nan"),
        ridge_bars_median=float(np.median(ridge_all)) if ridge_all.size else float("nan"),
        days_below_ridge_gate=int((ridge_ret < min_ridge_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_comparison_floor=int(at_comparison.sum()),
        **gates,
    )


# --------------------------------------------------------------------------- #
# Minute loading (cache-only, per-symbol -> memory-bounded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _RidgeReturnMinuteLoad:
    """Diagnostics from the cache-only per-symbol minute read + aggregation."""

    factor: pd.Series  # MultiIndex(date, symbol) raw ridge-minute-return sum
    requested: int
    covered: tuple[str, ...]  # symbols that produced >= 1 finite factor value
    empty_symbols: tuple[str, ...]  # requested but no cached minute / no value
    raw_rows: int
    live_calls: int  # provably 0 (store read has no fetch closure)
    ridge_coverage: RidgeReturnCoverage


def _load_ridge_minute_return_panel(
    cfg: RootConfig,
    symbols: list[str],
    spec: FactorSpec,
    logger,
    *,
    lookback_days: int,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    min_valid_days: int,
    min_classifiable: int,
    min_ridge_bars: int,
) -> _RidgeReturnMinuteLoad:
    """Compute the raw ridge-minute-return panel per symbol from the minute cache.

    Memory-bounded: one symbol's minute history is read, aggregated to its daily factor
    series, and discarded before the next — the multi-year all-symbol minute panel is NEVER
    materialized. Read-only: :meth:`IntradayParquetStore.read_range` has no fetch closure,
    so ``stk_mins`` live calls are provably zero (a symbol with no cached minute simply
    yields no rows and is disclosed as empty). The classification (reused from PR-F) and
    the return reduction run entirely inside ``compute_ridge_minute_return`` on 1min bars,
    which read ``volume`` and ``close``. The per-day bar-count diagnostics are collected
    alongside so the ridge scarcity can be REPORTED, not assumed; collecting them does not
    change the factor.
    """
    root = cfg.data.cache.root_dir
    store = IntradayParquetStore(root)
    start = pd.Timestamp(cfg.data.start).normalize()
    end = pd.Timestamp(cfg.data.end).normalize() + pd.Timedelta("23:59:59")

    series: list[pd.Series] = []
    covered: list[str] = []
    empty: list[str] = []
    diagnostics: list[pd.DataFrame] = []
    raw_rows = 0
    for i, sym in enumerate(symbols):
        part = store.read_range(INTRADAY_ENDPOINT, sym, RAW_INTRADAY_FREQ, start, end)
        if part.empty:
            empty.append(sym)
            continue
        raw_rows += len(part)
        bars = normalize_intraday_bars(
            part.rename(columns={"bar_end": "time"})[READ_COLUMNS],
            freq=RAW_INTRADAY_FREQ,
        )
        s = compute_ridge_minute_return(
            bars,
            lookback_days=lookback_days,
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_ridge_bars=min_ridge_bars,
            name=spec.factor_id,
            diagnostics_out=diagnostics,
        )
        if s.notna().any():
            series.append(s)
            covered.append(sym)
        else:
            empty.append(sym)
        if (i + 1) % 100 == 0:
            logger.info(
                "minute aggregation: %d/%d symbols processed (%d with a value)",
                i + 1, len(symbols), len(covered),
            )

    coverage = summarize_ridge_return_coverage(
        diagnostics,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
    )
    if not series:
        raise ValueError(
            "run-eval-ridge-minute-return blocked: no requested symbol produced a cached "
            f"ridge-minute-return value over [{cfg.data.start}, {cfg.data.end}]. The "
            "minute cache is required (this runner never warms it); check coverage."
        )
    factor = pd.concat(series).sort_index()
    logger.info(
        "minute aggregation (cache-only): %d/%d symbols with a value, %d raw 1min rows "
        "read, %d factor rows, stk_mins_live_calls=0",
        len(covered), len(symbols), raw_rows, len(factor),
    )
    logger.info("%s", coverage.render())
    return _RidgeReturnMinuteLoad(
        factor=factor,
        requested=len(symbols),
        covered=tuple(covered),
        empty_symbols=tuple(empty),
        raw_rows=raw_rows,
        live_calls=0,
        ridge_coverage=coverage,
    )


# --------------------------------------------------------------------------- #
# Evaluation core (network-free seam: given panels, run the two evaluations)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _RunReports:
    """The two evaluation reports (no-book / with-book) + their file paths."""

    no_book: FactorEvalReport
    with_book: FactorEvalReport
    no_book_md: Path
    no_book_json: Path
    with_book_md: Path
    with_book_json: Path
    no_book_dashboard: Path
    with_book_dashboard: Path


def evaluate_two_runs(
    factor_panel: pd.Series | pd.DataFrame,
    spec: FactorSpec,
    eval_cfg: EvalConfig,
    price_panel: pd.DataFrame,
    book: pd.DataFrame,
    *,
    universe_symbols: tuple[str, ...],
    fee_rate: float,
    report_dir: Path,
    stem: str = _REPORT_STEM,
) -> _RunReports:
    """Run the StandardFactorEvaluator TWICE (no book / with book) and write reports.

    This is the network-free seam: given the PROCESSED factor panel, the qfq price panel
    (for forward returns), and the PROCESSED known-factor book, it does the two
    ``evaluate`` calls and writes ``{stem}_no_book`` / ``{stem}_with_book`` as Markdown +
    JSON. Run 1 omits ``known_factors`` (Incremental NOT_ASSESSED); run 2 supplies the book
    (Incremental measured).
    """
    evaluator = StandardFactorEvaluator()
    report_dir.mkdir(parents=True, exist_ok=True)

    ctx_no_book = EvalContext(
        price_panel=price_panel,
        universe_symbols=universe_symbols,
        fee_rate=fee_rate,
    )
    # evaluate_with_ir yields the SAME report as evaluate() plus the IR the research-style
    # dashboard needs (per-period IC + quantile return series).
    report_no_book, ir_no_book = evaluator.evaluate_with_ir(
        factor_panel, spec, eval_cfg, ctx_no_book
    )

    ctx_with_book = EvalContext(
        price_panel=price_panel,
        universe_symbols=universe_symbols,
        fee_rate=fee_rate,
        known_factors=book,
    )
    report_with_book, ir_with_book = evaluator.evaluate_with_ir(
        factor_panel, spec, eval_cfg, ctx_with_book
    )

    nb_md, nb_json = _write_report(report_no_book, report_dir, f"{stem}_no_book")
    wb_md, wb_json = _write_report(report_with_book, report_dir, f"{stem}_with_book")
    nb_png = render_factor_dashboard(
        report_no_book, ir_no_book, report_dir / f"{stem}_no_book_dashboard.png"
    )
    wb_png = render_factor_dashboard(
        report_with_book, ir_with_book, report_dir / f"{stem}_with_book_dashboard.png"
    )
    return _RunReports(
        no_book=report_no_book,
        with_book=report_with_book,
        no_book_md=nb_md,
        no_book_json=nb_json,
        with_book_md=wb_md,
        with_book_json=wb_json,
        no_book_dashboard=nb_png,
        with_book_dashboard=wb_png,
    )


def _write_report(report: FactorEvalReport, report_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write ``report`` as deterministic Markdown + JSON; return the two paths."""
    md_path = report_dir / f"{stem}.md"
    json_path = report_dir / f"{stem}.json"
    md_path.write_text(report.render(), encoding="utf-8")
    json_path.write_text(report.to_json(), encoding="utf-8")
    return md_path, json_path


# --------------------------------------------------------------------------- #
# Metric extraction (for the CLI line + the handoff)
# --------------------------------------------------------------------------- #
def _section_payload(report: FactorEvalReport, name: str) -> dict:
    section = report.by_name().get(name)
    return dict(getattr(section, "payload", {}) or {})


def extract_metrics(report: FactorEvalReport) -> dict:
    """Pull the headline verdict + gated metrics out of a finished report.

    Beyond the fields PR-C..PR-J surfaced, this also extracts the TRADABILITY comparison
    quantities the PR-K analysis needs alongside PR-I / PR-J: the long-short leg turnover,
    the net long-short spread at EACH cost scenario, the lag-1 rank autocorrelation with
    its implied half-life, and the cross-section size.

    ⚠️ ``net_long_short_by_cost`` is deliberately the tradability field surfaced here. The
    frozen layer's ``aligned_spread_*`` computes ``sign * (gross - cost)``, which for this
    factor's pre-registered sign of -1 becomes ``-gross + cost`` — costs are added back
    rather than deducted, so those fields are UNRELIABLE for a negative-sign factor. The
    frozen layer is not patched; the correct, sign-agnostic net spreads are reported
    instead.
    """
    verdict = report.require_verdict()
    pred = _section_payload(report, "predictive_power")
    coverage = _section_payload(report, "data_coverage")
    incr = _section_payload(report, "purity")
    ret_risk = _section_payload(report, "return_risk")
    stability = _section_payload(report, "stability_cost")
    autocorr = dict(stability.get("factor_rank_autocorr_by_lag", {}) or {})
    return {
        "deployment": verdict.verdict,
        "predictive": verdict.predictive.verdict,
        "incremental": verdict.incremental.verdict,
        "tradable": verdict.tradable.verdict,
        "ic_mean": pred.get("ic_mean"),
        "ic_ir": pred.get("ic_ir"),
        "ic_ir_ci_low": pred.get("ic_ir_ci_low"),
        "ic_ir_ci_high": pred.get("ic_ir_ci_high"),
        "ic_ir_ci_n_eff": pred.get("ic_ir_ci_n_eff"),
        "ic_win_rate": pred.get("ic_win_rate"),
        "ic_nw_t": pred.get("ic_nw_t"),
        "settled_rebalances": coverage.get("settled_rebalances"),
        "effective_samples": coverage.get("effective_samples"),
        "span_days": coverage.get("span_days"),
        "cross_section_size_mean": coverage.get("cross_section_size_mean"),
        "cross_section_size_median": coverage.get("cross_section_size_median"),
        "incremental_ic_ir": incr.get("incremental_ic_ir"),
        "incremental_ic_ir_ci_low": incr.get("incremental_ic_ir_ci_low"),
        "incremental_ic_ir_ci_high": incr.get("incremental_ic_ir_ci_high"),
        "incremental_ic_ir_ci_n_eff": incr.get("incremental_ic_ir_ci_n_eff"),
        "incremental_ic_mean": incr.get("incremental_ic_mean"),
        "monotonicity_spearman": ret_risk.get("monotonicity_spearman"),
        "gross_long_short_mean": ret_risk.get("gross_long_short_mean"),
        # The correct tradability read for a negative-sign factor (see the docstring).
        "net_long_short_by_cost": dict(ret_risk.get("net_long_short_by_cost", {}) or {}),
        "long_short_turnover": stability.get("turnover_mean_long_short_legs"),
        "rank_autocorr_lag1": autocorr.get(1),
        "half_life_periods": stability.get("half_life_periods"),
    }


# --------------------------------------------------------------------------- #
# EvalConfig construction (HONEST provenance of what the runner actually did)
# --------------------------------------------------------------------------- #
def _build_eval_config(cfg: RootConfig) -> EvalConfig:
    """Build the per-run EvalConfig, declaring EXACTLY what the pipeline applied.

    The declarations must not overstate: this codebase's winsorize step is a P0 no-op, so
    ``winsorize`` is declared None (nothing was clipped) even though the config may toggle
    it. z-score + industry/size neutralization ARE applied, so they are declared.
    ``oos_split`` (from the config's ``oos`` block) makes the OOS section run so the
    Predictive axis can be assessed. ``is_exploratory=True``: this is a reproduction on a
    shorter window / narrower neutralization than the report, not a return claim (it caps
    the deployment label at Watch).
    """
    if cfg.oos is None:
        raise ValueError(
            "run-eval-ridge-minute-return requires an 'oos' section (split_date) so the "
            "Predictive axis has an out-of-sample split to assess; add e.g. "
            "oos: {split_date: '2024-01-01'}."
        )
    return EvalConfig(
        universe=cfg.universe.index_code or cfg.universe.type,
        universe_is_pit=cfg.universe.type == "index",
        start=cfg.data.start,
        end=cfg.data.end,
        is_exploratory=True,
        post_hoc_selected=False,
        rebalance="daily",
        n_quantiles=int(cfg.analytics.quantiles),
        cost_scenarios=(1.0, 2.0, 4.0),
        oos_split=cfg.oos.split_date,
        # winsorize is a P0 no-op in this codebase -> declare None (nothing clipped).
        winsorize=None,
        standardize="zscore" if cfg.processing.standardize.enabled else None,
        neutralization=("industry", "size") if cfg.processing.neutralize.enabled else (),
        industry_level=cfg.processing.neutralize.industry_level,
        tuned=False,
        # We evaluated ONE pre-registered factor whose sign came from the report (not a
        # screen of our own); the report's own factor screen is a caveat noted in the run's
        # prose, not our multiple-testing background.
        n_factors_screened=1,
        data_snapshot_id=cfg.data.cache.root_dir,
    )


# --------------------------------------------------------------------------- #
# Result container + the full glue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RidgeMinuteReturnEvalResult:
    """Immutable summary of one run-eval-ridge-minute-return run."""

    config: RootConfig
    spec: FactorSpec
    requested_symbols: int
    covered_symbols: int
    empty_symbols: int
    factor_rows: int
    minute_raw_rows: int
    minute_live_calls: int
    ridge_coverage: RidgeReturnCoverage
    no_book_metrics: dict
    with_book_metrics: dict
    reports: _RunReports
    log_path: Path
    elapsed: float


def _check_preconditions(cfg: RootConfig) -> None:
    """Fail readably if the config cannot drive a real, cache-only CSI500 eval."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-eval-ridge-minute-return needs data.source='tushare' (real cached "
            f"A-share data); got {cfg.data.source!r}."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-eval-ridge-minute-return needs data.cache.enabled=true (it reads the "
            "persistent tushare cache and never warms live)."
        )
    if cfg.universe.type != "index":
        raise ValueError(
            "run-eval-ridge-minute-return needs universe.type='index' (PIT membership, "
            f"e.g. 000905.SH for CSI500); got {cfg.universe.type!r}."
        )
    if not cfg.processing.neutralize.enabled:
        raise ValueError(
            "run-eval-ridge-minute-return expects processing.neutralize.enabled=true "
            "(industry + size neutralization, matching the report's neutral column and "
            "the EvalConfig declaration)."
        )


def run_eval_ridge_minute_return(config_path: str) -> RidgeMinuteReturnEvalResult:
    """Run the two real ridge-minute-return evaluations (cache-only) + reports."""
    cfg = load_config(config_path)
    _check_preconditions(cfg)

    log_path = Path(cfg.output.log_dir) / f"{_REPORT_STEM}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    factor = RidgeMinuteReturnFactor(lookback_days=RIDGE_RETURN_LOOKBACK_DAYS)
    spec = factor.spec
    eval_cfg = _build_eval_config(cfg)
    logger.info(
        "eval config: %s rebalance=daily oos_split=%s", eval_cfg.universe, eval_cfg.oos_split
    )

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)

    # Book (value_ep / value_bp / volatility_20): enrich, compute, process. The value
    # factors need daily_basic pe/pb; volatility_20 needs close.
    book_factors = _build_book_factors()
    panel = _maybe_enrich_value(cfg, panel, symbols, book_factors, logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)  # daily/universe/covariate gap-fetches (warm -> 0)

    panel_dates = pd.Index(
        pd.unique(panel.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    )
    book_raw = pd.concat(
        [f.compute(panel).rename(f.name) for f in book_factors], axis=1
    )
    book_processed = _process_factors(cfg, book_raw, panel)

    # Ridge-minute-return factor: cache-only per-symbol aggregation -> raw -> process.
    load = _load_ridge_minute_return_panel(
        cfg, symbols, spec, logger,
        lookback_days=RIDGE_RETURN_LOOKBACK_DAYS,
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        min_valid_days=VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=VOLUME_PRV_MIN_CLASSIFIABLE,
        min_ridge_bars=RIDGE_RETURN_MIN_RIDGE_BARS,
    )
    # A minute date with no daily bar cannot have a forward return; keep the factor on the
    # daily trading grid so the analytics boundary has a price for every date.
    factor_raw = load.factor[
        load.factor.index.get_level_values(DATE_LEVEL).isin(panel_dates)
    ]
    factor_processed = _process_factors(cfg, factor_raw.to_frame(spec.factor_id), panel)
    factor_series = factor_processed[spec.factor_id]

    price_panel = panel[CORE_COLUMNS]
    reports = evaluate_two_runs(
        factor_series,
        spec,
        eval_cfg,
        price_panel,
        book_processed,
        universe_symbols=tuple(symbols),
        fee_rate=float(cfg.cost.fee_rate),
        report_dir=Path(cfg.output.report_dir),
    )

    no_book_metrics = extract_metrics(reports.no_book)
    with_book_metrics = extract_metrics(reports.with_book)
    logger.info(
        "verdict no-book: %s (predictive=%s); with-book: %s (incremental=%s)",
        no_book_metrics["deployment"], no_book_metrics["predictive"],
        with_book_metrics["deployment"], with_book_metrics["incremental"],
    )
    # The tradability read for a NEGATIVE-sign factor: aligned_spread_* is unreliable here
    # (frozen-layer defect, see the module docstring), so log the net spreads explicitly.
    logger.info(
        "net long-short by cost (aligned_spread_* is UNRELIABLE for sign=-1): %s",
        no_book_metrics["net_long_short_by_cost"],
    )

    return RidgeMinuteReturnEvalResult(
        config=cfg,
        spec=spec,
        requested_symbols=load.requested,
        covered_symbols=len(load.covered),
        empty_symbols=len(load.empty_symbols),
        factor_rows=int(len(factor_series)),
        minute_raw_rows=load.raw_rows,
        minute_live_calls=load.live_calls,
        ridge_coverage=load.ridge_coverage,
        no_book_metrics=no_book_metrics,
        with_book_metrics=with_book_metrics,
        reports=reports,
        log_path=log_path,
        elapsed=time.monotonic() - started,
    )


def _build_book_factors() -> list:
    """Instantiate the confirmed book (value_ep / value_bp / volatility_20)."""
    from factors.compute.candidates import ValueFactor, VolatilityFactor

    return [
        ValueFactor("value_ep"),
        ValueFactor("value_bp"),
        VolatilityFactor(window=20),
    ]


__all__ = [
    "RidgeMinuteReturnEvalResult",
    "RidgeReturnCoverage",
    "evaluate_two_runs",
    "extract_metrics",
    "run_eval_ridge_minute_return",
    "summarize_ridge_return_coverage",
]
