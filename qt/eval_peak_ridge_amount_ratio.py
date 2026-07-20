"""run-eval-peak-ridge-amount-ratio: the eleventh real factor evaluation (PR-M).

Reproduces the SEVENTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, reportId 4957417, §7.2) — the
"峰岭成交比" factor — as a first-class
:class:`~factors.compute.intraday_derived.PeakRidgeAmountRatioFactor` and runs it through
the FROZEN :class:`~analytics.eval.StandardFactorEvaluator` on REAL cached A-share data
(CSI500, PIT membership) — the same contract-driven loop PR-C .. PR-L used.

THIS IS THE CLOSING FACTOR of the reproduction loop, and it is the one that closes the
FAMILY MAP. The nine prior reproductions covered a COUNT (PR-F volume_peak_count, weak), a
TIMING moment (PR-H peak_interval_kurtosis, null), two price RATIOS (PR-I
valley_relative_vwap, PR-J valley_ridge_vwap_ratio, both PASSING), a price POSITION (PR-L
valley_price_quantile, PASSING and strongest) and a RETURN (PR-K ridge_minute_return, sign
transferred but Reject). Every signal found so far has been a PRICE signal. This factor
carries NO price information whatsoever — both legs are pure traded VALUE — so it is the
one test that distinguishes "only price information survives in this taxonomy" from "the
peak/ridge behavioural split itself carries alpha". A PASS widens the finding to the
taxonomy; a REJECT sharpens it to prices. Either outcome is a legitimate result and is
reported as such.

The factor is a DAILY signal derived DIRECTLY from the 1min cache (see
``data.clean.intraday_amount_ratio.compute_peak_ridge_amount_ratio``): PIT-truncate each
day at 14:50, classify every visible minute with the taxonomy REUSED from PR-F
(``data.clean.intraday_volume_prv``: same-slot strictly-prior μ+σ eruptive test; a PEAK is
an ISOLATED eruption, a RIDGE is an eruptive minute that is not isolated), total each valid
day's peak and ridge traded amount, and divide the trailing-20-VALID-day SUM of the peak
leg by that of the ridge leg. NOTE THE AGGREGATION: the report specifies a RATIO OF SUMS
here ("计算 20 日量峰总成交额与量岭总成交额，二者做比"), NOT the mean of daily ratios that
§7.1 (PR-J) specifies — the two forms are followed as written in each section. It is
executed CLOSE-TO-CLOSE (daily default), so ``is_intraday=False``. The eval CELL is
identical to PR-C..PR-L, so this run is directly comparable to its siblings.

PEAK SCARCITY IS MEASURED, NOT ASSUMED — and the asymmetry RUNS THE OTHER WAY THAN PR-J's.
There, ridges were the scarce leg; here PEAKS are, because a peak must erupt AND be
ISOLATED. The valid-day gate therefore asks for >= 5 tradable peak bars against the ridge
leg's >= 10. The runner collects the REALIZED per-day peak-bar distribution, the day
validity rate across the whole universe, and the COUNTERFACTUAL valid-day count at a peak
floor of 10, so the cost of the lowered threshold is a number rather than an assumption.

CACHE-ONLY: every input is read from the persistent tushare cache
(``artifacts/cache/tushare/v1``). The minute read is provably live-call-free (the minute
store has no fetch closure — a miss simply yields no rows); the daily / universe /
covariate endpoints go through the shared read-through cache, which on a fully-warmed
cache does zero gap fetches (disclosed via the run-log cache-stats line). Forward returns
are computed ONLY at the evaluator/analytics boundary from ``ctx.price_panel`` — the factor
computation never sees a future return.

The evaluator is run TWICE: once with NO known-factor book (the Incremental axis is
NOT_ASSESSED) and once with the project's independently-confirmed book (value_ep /
value_bp / volatility_20) so the Incremental axis measures whether the ratio adds alpha
BEYOND value / low-vol. The with-book run carries a different weight here than for the
price factors: a traded-VALUE mix has no obvious kinship with a value or low-vol bet, so an
Incremental PASS would be a genuinely new exposure rather than a repackaged one — and an
Incremental failure would be the more surprising outcome.
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
from data.clean.intraday_amount_ratio import (
    PEAK_RIDGE_LOOKBACK_DAYS,
    PEAK_RIDGE_MIN_PEAK_BARS,
    PEAK_RIDGE_MIN_RIDGE_BARS,
    compute_peak_ridge_amount_ratio,
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
from factors.compute.intraday_derived import PeakRidgeAmountRatioFactor
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

_LOGGER_NAME = "qt.eval_peak_ridge_amount_ratio"
_REPORT_STEM = "eval_peak_ridge_amount_ratio"

# Percentiles reported for the realized peak-bar distribution (the scarcity disclosure).
_PEAK_PCTL = (0, 10, 25, 50, 75, 90, 100)

# The counterfactual peak floor the task card asks to quantify: how many days would still
# be valid if the PEAK leg were held to the RIDGE leg's floor instead of its own.
_COUNTERFACTUAL_PEAK_FLOOR = PEAK_RIDGE_MIN_RIDGE_BARS


# --------------------------------------------------------------------------- #
# Peak-scarcity coverage (measured, never assumed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeakCoverage:
    """Realized peak-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the days
    the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible bars,
    including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The headline
    ``validity_rate`` is taken over ``classifiable_days``, because a day with no baseline
    fails for a PR-F warm-up reason rather than a peak-scarcity one and would otherwise
    make the peak gate look worse than it is — both denominators are reported so the reader
    can check that framing. The gate-failure counts are NOT mutually exclusive (a thin day
    can fail several gates at once) and are reported for shape, not as a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    peak_percentiles: tuple[tuple[int, float], ...]
    peak_mean: float
    ridge_median: float
    days_below_peak_gate: int
    days_below_ridge_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive if the PEAK leg were held to the RIDGE
    # floor. Quantifies exactly what the lowered threshold buys.
    valid_days_at_ridge_floor: int
    # The gates this run actually applied, so the disclosure can never describe the module
    # defaults while the run used something else.
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE
    counterfactual_peak_floor: int = _COUNTERFACTUAL_PEAK_FLOOR

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.peak_percentiles)
        return (
            f"peak scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"peak_bars[{pctl} mean={self.peak_mean:.1f}] "
            f"ridge_bars_median={self.ridge_median:.0f} "
            f"below_peak_gate({self.min_peak_bars})={self.days_below_peak_gate} "
            f"below_ridge_gate({self.min_ridge_bars})={self.days_below_ridge_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_peak_floor_were_{self.counterfactual_peak_floor}="
            f"{self.valid_days_at_ridge_floor}"
        )


def summarize_peak_coverage(
    frames: list[pd.DataFrame],
    *,
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS,
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    counterfactual_peak_floor: int = _COUNTERFACTUAL_PEAK_FLOOR,
) -> PeakCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The three floors must be the ones the RUN applied, not the module defaults — otherwise
    the disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_peak_bars=min_peak_bars,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        counterfactual_peak_floor=counterfactual_peak_floor,
    )
    empty = tuple((p, float("nan")) for p in _PEAK_PCTL)
    if not frames:
        return PeakCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            peak_percentiles=empty,
            peak_mean=float("nan"),
            ridge_median=float("nan"),
            days_below_peak_gate=0,
            days_below_ridge_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_ridge_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the percentiles
    # towards zero for a reason that has nothing to do with peak scarcity.
    scored = classifiable >= min_classifiable
    peak = diag.loc[scored, "peak_bars"].to_numpy(dtype=float)
    ridge = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    # The counterfactual raises the PEAK floor, leaving every other gate exactly as it was.
    at_ridge_floor = valid & (
        diag["peak_bars"].to_numpy(dtype=float) >= counterfactual_peak_floor
    )
    return PeakCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        peak_percentiles=(
            tuple((p, float(np.percentile(peak, p))) for p in _PEAK_PCTL)
            if peak.size
            else empty
        ),
        peak_mean=float(peak.mean()) if peak.size else float("nan"),
        ridge_median=float(np.median(ridge)) if ridge.size else float("nan"),
        days_below_peak_gate=int((peak < min_peak_bars).sum()),
        days_below_ridge_gate=int((ridge < min_ridge_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_ridge_floor=int(at_ridge_floor.sum()),
        **gates,
    )


# --------------------------------------------------------------------------- #
# Minute loading (cache-only, per-symbol -> memory-bounded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PeakRidgeMinuteLoad:
    """Diagnostics from the cache-only per-symbol minute read + aggregation."""

    factor: pd.Series  # MultiIndex(date, symbol) raw peak/ridge amount ratio
    requested: int
    covered: tuple[str, ...]  # symbols that produced >= 1 finite factor value
    empty_symbols: tuple[str, ...]  # requested but no cached minute / no value
    raw_rows: int
    live_calls: int  # provably 0 (store read has no fetch closure)
    peak_coverage: PeakCoverage


def _load_peak_ridge_amount_ratio_panel(
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
    min_peak_bars: int,
    min_ridge_bars: int,
) -> _PeakRidgeMinuteLoad:
    """Compute the raw peak/ridge amount-ratio panel per symbol from the minute cache.

    Memory-bounded: one symbol's minute history is read, aggregated to its daily factor
    series, and discarded before the next — the multi-year all-symbol minute panel is NEVER
    materialized. Read-only: :meth:`IntradayParquetStore.read_range` has no fetch closure,
    so ``stk_mins`` live calls are provably zero (a symbol with no cached minute simply
    yields no rows and is disclosed as empty). The classification (reused from PR-F) and
    the amount-ratio reduction run entirely inside ``compute_peak_ridge_amount_ratio`` on
    1min bars, which read ``volume`` and ``amount``. The per-day bar-count diagnostics are
    collected alongside so the peak scarcity can be REPORTED, not assumed; collecting them
    does not change the factor.
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
        s = compute_peak_ridge_amount_ratio(
            bars,
            lookback_days=lookback_days,
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_peak_bars=min_peak_bars,
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

    coverage = summarize_peak_coverage(
        diagnostics,
        min_peak_bars=min_peak_bars,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
    )
    if not series:
        raise ValueError(
            "run-eval-peak-ridge-amount-ratio blocked: no requested symbol produced a "
            f"cached peak/ridge amount-ratio value over [{cfg.data.start}, "
            f"{cfg.data.end}]. The minute cache is required (this runner never warms it); "
            "check coverage."
        )
    factor = pd.concat(series).sort_index()
    logger.info(
        "minute aggregation (cache-only): %d/%d symbols with a value, %d raw 1min rows "
        "read, %d factor rows, stk_mins_live_calls=0",
        len(covered), len(symbols), raw_rows, len(factor),
    )
    logger.info("%s", coverage.render())
    return _PeakRidgeMinuteLoad(
        factor=factor,
        requested=len(symbols),
        covered=tuple(covered),
        empty_symbols=tuple(empty),
        raw_rows=raw_rows,
        live_calls=0,
        peak_coverage=coverage,
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


def _write_report(
    report: FactorEvalReport, report_dir: Path, stem: str
) -> tuple[Path, Path]:
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

    Surfaces the same comparison quantities PR-K / PR-L extracted (turnover, net long-short
    by cost scenario, lag-1 rank autocorrelation + half-life, cross-section size) so PR-M
    sits in ONE table with all nine siblings — which matters more for this run than any
    other, since the closing deliverable of the loop is the full ten-factor comparison.
    ``ic_pearson_mean`` rides alongside the rank ``ic_mean`` to keep testing the PR-K
    review's regularity, that a divergence between the two predicts the monotonicity gate
    failing.

    ``aligned_spread_*`` is NOT unreliable here: the frozen layer's cost-sign defect only
    mis-signs negative-sign factors, and this factor's pre-registered sign is +1.
    ``net_long_short_by_cost`` is still surfaced as the primary read, for comparability.
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
        "ic_pearson_mean": pred.get("ic_pearson_mean"),
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
        "net_long_short_by_cost": dict(ret_risk.get("net_long_short_by_cost", {}) or {}),
        # sign=+1 -> the aligned spreads are NOT mis-signed for this factor.
        "aligned_spread_by_cost": dict(ret_risk.get("aligned_spread_by_cost", {}) or {}),
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
            "run-eval-peak-ridge-amount-ratio requires an 'oos' section (split_date) so "
            "the Predictive axis has an out-of-sample split to assess; add e.g. "
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
class PeakRidgeAmountRatioEvalResult:
    """Immutable summary of one run-eval-peak-ridge-amount-ratio run."""

    config: RootConfig
    spec: FactorSpec
    requested_symbols: int
    covered_symbols: int
    empty_symbols: int
    factor_rows: int
    minute_raw_rows: int
    minute_live_calls: int
    peak_coverage: PeakCoverage
    no_book_metrics: dict
    with_book_metrics: dict
    reports: _RunReports
    log_path: Path
    elapsed: float


def _check_preconditions(cfg: RootConfig) -> None:
    """Fail readably if the config cannot drive a real, cache-only CSI500 eval."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-eval-peak-ridge-amount-ratio needs data.source='tushare' (real cached "
            f"A-share data); got {cfg.data.source!r}."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-eval-peak-ridge-amount-ratio needs data.cache.enabled=true (it reads the "
            "persistent tushare cache and never warms live)."
        )
    if cfg.universe.type != "index":
        raise ValueError(
            "run-eval-peak-ridge-amount-ratio needs universe.type='index' (PIT "
            f"membership, e.g. 000905.SH for CSI500); got {cfg.universe.type!r}."
        )
    if not cfg.processing.neutralize.enabled:
        raise ValueError(
            "run-eval-peak-ridge-amount-ratio expects processing.neutralize.enabled=true "
            "(industry + size neutralization, matching the report's neutral column and "
            "the EvalConfig declaration)."
        )


def run_eval_peak_ridge_amount_ratio(config_path: str) -> PeakRidgeAmountRatioEvalResult:
    """Run the two real peak/ridge amount-ratio evaluations (cache-only) + reports."""
    cfg = load_config(config_path)
    _check_preconditions(cfg)

    log_path = Path(cfg.output.log_dir) / f"{_REPORT_STEM}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    factor = PeakRidgeAmountRatioFactor(lookback_days=PEAK_RIDGE_LOOKBACK_DAYS)
    spec = factor.spec
    eval_cfg = _build_eval_config(cfg)
    logger.info(
        "eval config: %s rebalance=daily oos_split=%s",
        eval_cfg.universe, eval_cfg.oos_split,
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

    # Peak/ridge amount-ratio factor: cache-only per-symbol aggregation -> raw -> process.
    load = _load_peak_ridge_amount_ratio_panel(
        cfg, symbols, spec, logger,
        lookback_days=PEAK_RIDGE_LOOKBACK_DAYS,
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        min_valid_days=VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=VOLUME_PRV_MIN_CLASSIFIABLE,
        min_peak_bars=PEAK_RIDGE_MIN_PEAK_BARS,
        min_ridge_bars=PEAK_RIDGE_MIN_RIDGE_BARS,
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

    return PeakRidgeAmountRatioEvalResult(
        config=cfg,
        spec=spec,
        requested_symbols=load.requested,
        covered_symbols=len(load.covered),
        empty_symbols=len(load.empty_symbols),
        factor_rows=int(len(factor_series)),
        minute_raw_rows=load.raw_rows,
        minute_live_calls=load.live_calls,
        peak_coverage=load.peak_coverage,
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
    "PeakCoverage",
    "PeakRidgeAmountRatioEvalResult",
    "evaluate_two_runs",
    "extract_metrics",
    "run_eval_peak_ridge_amount_ratio",
    "summarize_peak_coverage",
]
