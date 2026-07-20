"""run-eval-valley-ridge-vwap-ratio: the eighth real factor evaluation (PR-J).

Reproduces the FOURTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, reportId 4957417, §7.1) — the
"谷岭加权价格比" factor — as a first-class
:class:`~factors.compute.intraday_derived.ValleyRidgeVwapRatioFactor` and runs it through
the FROZEN :class:`~analytics.eval.StandardFactorEvaluator` on REAL cached A-share data
(CSI500, PIT membership) — the same contract-driven loop PR-C .. PR-I used.

This is the FOURTH reproduction from the same report and the direct ROBUSTNESS TEST of
PR-I. PR-I (valley VWAP / whole-visible-day VWAP) is the strongest factor of this loop —
both assessed axes PASS. PR-J keeps everything else fixed and swaps only the DENOMINATOR
to the RIDGE VWAP, contrasting the two behavioural groups head-on. If the price-level
signal survives the swap it is a property of the FAMILY; if it does not, PR-I depended on
its specific denominator. Either outcome is a legitimate result and is reported as such.

The factor is a DAILY signal derived DIRECTLY from the 1min cache (see
``data.clean.intraday_valley_ridge_vwap.compute_valley_ridge_vwap_ratio``): PIT-truncate
each day at 14:50, classify every visible minute with the taxonomy REUSED from PR-F
(``data.clean.intraday_volume_prv``: same-slot strictly-prior μ+σ eruptive test; a VALLEY
is a classifiable non-eruptive minute, a RIDGE is an eruptive minute that is not an
isolated peak), take each valid day's ratio of the valley VWAP to the ridge VWAP (both
via the Σamount/Σvolume identity), and average that ratio over the trailing 20 VALID
trading days. It is executed CLOSE-TO-CLOSE (daily default), so ``is_intraday=False``.
The eval CELL is identical to PR-C..PR-I, so this run is directly comparable to its
siblings — same classification, same VWAP identity, different denominator.

RIDGE SCARCITY IS MEASURED, NOT ASSUMED. Ridge bars are structurally far rarer than
valley bars (a minute must erupt AND fail the isolation test), which is why the valid-day
gate asks for >= 10 tradable ridge bars against the valley leg's >= 20. The runner
therefore collects the REALIZED per-day ridge-bar distribution and the day-validity rate
across the whole universe and logs them, so a coverage regression against PR-I is visible
as a number rather than hidden behind a lower threshold.

CACHE-ONLY: every input is read from the persistent tushare cache
(``artifacts/cache/tushare/v1``). The minute read is provably live-call-free (the
minute store has no fetch closure — a miss simply yields no rows); the daily /
universe / covariate endpoints go through the shared read-through cache, which on a
fully-warmed cache does zero gap fetches (disclosed via the run-log cache-stats line).
Forward returns are computed ONLY at the evaluator/analytics boundary from
``ctx.price_panel`` — the factor computation never sees a future return.

The evaluator is run TWICE: once with NO known-factor book (the Incremental axis is
NOT_ASSESSED) and once with the project's independently-confirmed book (value_ep /
value_bp / volatility_20) so the Incremental axis measures whether the ratio adds alpha
BEYOND value / low-vol. As with PR-I that check matters more than usual: a relative PRICE
LEVEL is a plausible cousin of a value signal, so the with-book run is the one that says
whether this is a new bet.
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
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from data.clean.intraday_valley_ridge_vwap import (
    VALLEY_RIDGE_LOOKBACK_DAYS,
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
    compute_valley_ridge_vwap_ratio,
)
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
)
from data.clean.schema import CORE_COLUMNS, DATE_LEVEL
from factors.compute.intraday_derived import ValleyRidgeVwapRatioFactor
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

_LOGGER_NAME = "qt.eval_valley_ridge_vwap_ratio"
_REPORT_STEM = "eval_valley_ridge_vwap_ratio"

# Percentiles reported for the realized ridge-bar distribution (the scarcity disclosure).
_RIDGE_PCTL = (0, 10, 25, 50, 75, 90, 100)


# --------------------------------------------------------------------------- #
# Ridge-scarcity coverage (measured, never assumed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RidgeCoverage:
    """Realized ridge-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the
    days the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible
    bars, including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The
    headline ``validity_rate`` is taken over ``classifiable_days``, because a day with
    no baseline fails for a PR-F warm-up reason rather than a ridge-scarcity one and
    would otherwise make the ridge gate look worse than it is — both denominators are
    reported so the reader can check that framing. The gate-failure counts are NOT
    mutually exclusive (a thin day can fail several gates at once) and are reported for
    shape, not as a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    ridge_percentiles: tuple[tuple[int, float], ...]
    ridge_mean: float
    valley_median: float
    days_below_ridge_gate: int
    days_below_valley_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive if the ridge leg were held to the
    # VALLEY floor. Quantifies exactly what the lowered threshold buys.
    valid_days_at_valley_floor: int
    # The gates this run actually applied, so the disclosure can never describe the
    # module defaults while the run used something else.
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.ridge_percentiles)
        return (
            f"ridge scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"ridge_bars[{pctl} mean={self.ridge_mean:.1f}] "
            f"valley_bars_median={self.valley_median:.0f} "
            f"below_ridge_gate({self.min_ridge_bars})="
            f"{self.days_below_ridge_gate} "
            f"below_valley_gate({self.min_valley_bars})="
            f"{self.days_below_valley_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_ridge_floor_were_{self.min_valley_bars}="
            f"{self.valid_days_at_valley_floor}"
        )


def summarize_ridge_coverage(
    frames: list[pd.DataFrame],
    *,
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS,
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> RidgeCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The three floors must be the ones the RUN applied, not the module defaults —
    otherwise the disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_ridge_bars=min_ridge_bars,
        min_valley_bars=min_valley_bars,
        min_classifiable=min_classifiable,
    )
    empty = tuple((p, float("nan")) for p in _RIDGE_PCTL)
    if not frames:
        return RidgeCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            ridge_percentiles=empty,
            ridge_mean=float("nan"),
            valley_median=float("nan"),
            days_below_ridge_gate=0,
            days_below_valley_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_valley_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the
    # percentiles towards zero for a reason that has nothing to do with ridge scarcity.
    scored = classifiable >= min_classifiable
    ridge = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    valley = diag.loc[scored, "valley_bars"].to_numpy(dtype=float)
    # The counterfactual holds the ridge leg to the valley floor, leaving every other
    # gate exactly as it was.
    at_valley_floor = valid & (
        diag["ridge_bars"].to_numpy(dtype=float) >= min_valley_bars
    )
    return RidgeCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        ridge_percentiles=(
            tuple((p, float(np.percentile(ridge, p))) for p in _RIDGE_PCTL)
            if ridge.size
            else empty
        ),
        ridge_mean=float(ridge.mean()) if ridge.size else float("nan"),
        valley_median=float(np.median(valley)) if valley.size else float("nan"),
        days_below_ridge_gate=int((ridge < min_ridge_bars).sum()),
        days_below_valley_gate=int((valley < min_valley_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_valley_floor=int(at_valley_floor.sum()),
        **gates,
    )


# --------------------------------------------------------------------------- #
# Minute loading (cache-only, per-symbol -> memory-bounded)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ValleyRidgeMinuteLoad:
    """Diagnostics from the cache-only per-symbol minute read + aggregation."""

    factor: pd.Series  # MultiIndex(date, symbol) raw valley/ridge VWAP ratio
    requested: int
    covered: tuple[str, ...]  # symbols that produced >= 1 finite factor value
    empty_symbols: tuple[str, ...]  # requested but no cached minute / no value
    raw_rows: int
    live_calls: int  # provably 0 (store read has no fetch closure)
    ridge_coverage: RidgeCoverage


def _load_valley_ridge_vwap_ratio_panel(
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
    min_valley_bars: int,
    min_ridge_bars: int,
) -> _ValleyRidgeMinuteLoad:
    """Compute the raw valley/ridge VWAP-ratio panel per symbol from the minute cache.

    Memory-bounded: one symbol's minute history is read, aggregated to its daily factor
    series, and discarded before the next — the multi-year all-symbol minute panel is
    NEVER materialized. Read-only: :meth:`IntradayParquetStore.read_range` has no fetch
    closure, so ``stk_mins`` live calls are provably zero (a symbol with no cached
    minute simply yields no rows and is disclosed as empty). The classification (reused
    from PR-F) and the VWAP-ratio reduction run entirely inside
    ``compute_valley_ridge_vwap_ratio`` on 1min bars, which read ``volume`` and
    ``amount``. The per-day bar-count diagnostics are collected alongside so the ridge
    scarcity can be REPORTED, not assumed; collecting them does not change the factor.
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
        s = compute_valley_ridge_vwap_ratio(
            bars,
            lookback_days=lookback_days,
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_valley_bars=min_valley_bars,
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

    coverage = summarize_ridge_coverage(
        diagnostics,
        min_ridge_bars=min_ridge_bars,
        min_valley_bars=min_valley_bars,
        min_classifiable=min_classifiable,
    )
    if not series:
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio blocked: no requested symbol produced a "
            f"cached valley/ridge VWAP-ratio value over [{cfg.data.start}, "
            f"{cfg.data.end}]. The minute cache is required (this runner never warms "
            "it); check coverage."
        )
    factor = pd.concat(series).sort_index()
    logger.info(
        "minute aggregation (cache-only): %d/%d symbols with a value, %d raw 1min "
        "rows read, %d factor rows, stk_mins_live_calls=0",
        len(covered), len(symbols), raw_rows, len(factor),
    )
    logger.info("%s", coverage.render())
    return _ValleyRidgeMinuteLoad(
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

    This is the network-free seam: given the PROCESSED factor panel, the qfq price
    panel (for forward returns), and the PROCESSED known-factor book, it does the two
    ``evaluate`` calls and writes ``{stem}_no_book`` / ``{stem}_with_book`` as
    Markdown + JSON. Run 1 omits ``known_factors`` (Incremental NOT_ASSESSED); run 2
    supplies the book (Incremental measured).
    """
    evaluator = StandardFactorEvaluator()
    report_dir.mkdir(parents=True, exist_ok=True)

    ctx_no_book = EvalContext(
        price_panel=price_panel,
        universe_symbols=universe_symbols,
        fee_rate=fee_rate,
    )
    # evaluate_with_ir yields the SAME report as evaluate() plus the IR the
    # research-style dashboard needs (per-period IC + quantile return series).
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
    """Pull the headline verdict + gated metrics out of a finished report."""
    verdict = report.require_verdict()
    pred = _section_payload(report, "predictive_power")
    purity = _section_payload(report, "data_coverage")
    incr = _section_payload(report, "purity")
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
        "settled_rebalances": purity.get("settled_rebalances"),
        "effective_samples": purity.get("effective_samples"),
        "span_days": purity.get("span_days"),
        "incremental_ic_ir": incr.get("incremental_ic_ir"),
        "incremental_ic_ir_ci_low": incr.get("incremental_ic_ir_ci_low"),
        "incremental_ic_ir_ci_high": incr.get("incremental_ic_ir_ci_high"),
        "incremental_ic_ir_ci_n_eff": incr.get("incremental_ic_ir_ci_n_eff"),
        "incremental_ic_mean": incr.get("incremental_ic_mean"),
    }


# --------------------------------------------------------------------------- #
# EvalConfig construction (HONEST provenance of what the runner actually did)
# --------------------------------------------------------------------------- #
def _build_eval_config(cfg: RootConfig) -> EvalConfig:
    """Build the per-run EvalConfig, declaring EXACTLY what the pipeline applied.

    The declarations must not overstate: this codebase's winsorize step is a P0 no-op,
    so ``winsorize`` is declared None (nothing was clipped) even though the config may
    toggle it. z-score + industry/size neutralization ARE applied, so they are
    declared. ``oos_split`` (from the config's ``oos`` block) makes the OOS section run
    so the Predictive axis can be assessed. ``is_exploratory=True``: this is a
    reproduction on a shorter window / narrower neutralization than the report, not a
    return claim (it caps the deployment label at Watch).
    """
    if cfg.oos is None:
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio requires an 'oos' section (split_date) so "
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
        # screen of our own); the report's own factor screen is a caveat noted in the
        # run's prose, not our multiple-testing background.
        n_factors_screened=1,
        data_snapshot_id=cfg.data.cache.root_dir,
    )


# --------------------------------------------------------------------------- #
# Result container + the full glue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValleyRidgeVwapRatioEvalResult:
    """Immutable summary of one run-eval-valley-ridge-vwap-ratio run."""

    config: RootConfig
    spec: FactorSpec
    requested_symbols: int
    covered_symbols: int
    empty_symbols: int
    factor_rows: int
    minute_raw_rows: int
    minute_live_calls: int
    ridge_coverage: RidgeCoverage
    no_book_metrics: dict
    with_book_metrics: dict
    reports: _RunReports
    log_path: Path
    elapsed: float


def _check_preconditions(cfg: RootConfig) -> None:
    """Fail readably if the config cannot drive a real, cache-only CSI500 eval."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio needs data.source='tushare' (real cached "
            f"A-share data); got {cfg.data.source!r}."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio needs data.cache.enabled=true (it reads "
            "the persistent tushare cache and never warms live)."
        )
    if cfg.universe.type != "index":
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio needs universe.type='index' (PIT "
            f"membership, e.g. 000905.SH for CSI500); got {cfg.universe.type!r}."
        )
    if not cfg.processing.neutralize.enabled:
        raise ValueError(
            "run-eval-valley-ridge-vwap-ratio expects "
            "processing.neutralize.enabled=true (industry + size neutralization, "
            "matching the report's neutral column and the EvalConfig declaration)."
        )


def run_eval_valley_ridge_vwap_ratio(config_path: str) -> ValleyRidgeVwapRatioEvalResult:
    """Run the two real valley/ridge VWAP-ratio evaluations (cache-only) + reports."""
    cfg = load_config(config_path)
    _check_preconditions(cfg)

    log_path = Path(cfg.output.log_dir) / f"{_REPORT_STEM}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    factor = ValleyRidgeVwapRatioFactor(lookback_days=VALLEY_RIDGE_LOOKBACK_DAYS)
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

    # Valley/ridge VWAP-ratio factor: cache-only per-symbol aggregation -> raw -> process.
    load = _load_valley_ridge_vwap_ratio_panel(
        cfg, symbols, spec, logger,
        lookback_days=VALLEY_RIDGE_LOOKBACK_DAYS,
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        min_valid_days=VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=VOLUME_PRV_MIN_CLASSIFIABLE,
        min_valley_bars=VALLEY_RIDGE_MIN_VALLEY_BARS,
        min_ridge_bars=VALLEY_RIDGE_MIN_RIDGE_BARS,
    )
    # A minute date with no daily bar cannot have a forward return; keep the factor on
    # the daily trading grid so the analytics boundary has a price for every date.
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

    return ValleyRidgeVwapRatioEvalResult(
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
    "RidgeCoverage",
    "ValleyRidgeVwapRatioEvalResult",
    "evaluate_two_runs",
    "extract_metrics",
    "run_eval_valley_ridge_vwap_ratio",
    "summarize_ridge_coverage",
]
