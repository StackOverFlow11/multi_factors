"""run-eval-intraday-amp-cut: the fifth real factor evaluation (PR-G).

Reproduces the SECOND factor of the Kaiyuan market-microstructure series #30 (开源证券
《高频振幅因子的内部切割——市场微观结构系列（30）》, reportId 4988549) — the "日内振幅切割因子"
(§3, thought (2)) — as a first-class
:class:`~factors.compute.intraday_derived.IntradayAmpCutFactor` and runs it through the
FROZEN :class:`~analytics.eval.StandardFactorEvaluator` on REAL cached A-share data
(CSI500, PIT membership) — the same contract-driven loop PR-C / PR-D / PR-E / PR-F used.

DISTINCT FROM PR-D (``minute_ideal_amp``, the report's flagship, thought (1)) which pools
the trailing 10 days' minutes into ONE set and cuts by minute CLOSE PRICE: this factor
cuts EACH DAY independently by the 1-MINUTE RETURN, produces a daily ``V_day`` series,
takes its trailing-10-valid-day ``V_mean`` / ``V_std``, and combines them CROSS-
SECTIONALLY (z-score each per date, average). See ``data.clean.intraday_amp_cut``.

The factor is a DAILY signal executed CLOSE-TO-CLOSE (daily default), so
``is_intraday=False`` (the reasoning is documented on the factor's spec).

CACHE-ONLY: every input is read from the persistent tushare cache
(``artifacts/cache/tushare/v1``). The minute read is provably live-call-free (the minute
store has no fetch closure — a miss simply yields no rows); the daily / universe /
covariate endpoints go through the shared read-through cache, which on a fully-warmed
cache does zero gap fetches (disclosed via the run-log cache-stats line). Forward returns
are computed ONLY at the evaluator/analytics boundary from ``ctx.price_panel`` — the
factor computation never sees a future return.

The cross-sectional standardization (report step 4) needs EVERY symbol's ``(V_mean,
V_std)`` present before it can z-score a date. The loader therefore streams one symbol at
a time through ``compute_amp_cut_stats`` (memory-bounded), assembles the full-universe
two-column stats panel, then calls ``combine_amp_cut_cross_section`` ONCE — the CSI500
covered set is the cross-section (PINNED, disclosed on the factor spec).

The evaluator is run TWICE: once with NO known-factor book (the Incremental axis is
NOT_ASSESSED) and once with the project's independently-confirmed book (value_ep /
value_bp / volatility_20) so the Incremental axis measures whether the amplitude-cut
factor adds alpha BEYOND value / low-vol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

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
from data.clean.intraday_amp_cut import (
    AMP_CUT_LAMBDA,
    AMP_CUT_LOOKBACK_DAYS,
    AMP_CUT_MIN_CROSS_SECTION,
    AMP_CUT_MIN_DAY_MINUTES,
    AMP_CUT_MIN_VALID_DAYS,
    V_MEAN_COL,
    V_STD_COL,
    combine_amp_cut_cross_section,
    compute_amp_cut_stats,
)
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from data.clean.schema import CORE_COLUMNS, DATE_LEVEL
from factors.compute.intraday_derived import IntradayAmpCutFactor
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

_LOGGER_NAME = "qt.eval_intraday_amp_cut"
_REPORT_STEM = "eval_intraday_amp_cut"


# --------------------------------------------------------------------------- #
# Minute loading (cache-only, per-symbol stats -> ONE cross-sectional combine)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _AmpCutMinuteLoad:
    """Diagnostics from the cache-only per-symbol stats read + cross-sectional combine."""

    factor: pd.Series  # MultiIndex(date, symbol) final combined intraday_amp_cut
    stats_rows: int  # rows in the assembled (V_mean, V_std) panel
    requested: int
    covered: tuple[str, ...]  # symbols that produced >= 1 finite (V_mean, V_std) pair
    empty_symbols: tuple[str, ...]  # requested but no cached minute / no finite stat
    raw_rows: int
    live_calls: int  # provably 0 (store read has no fetch closure)


def _load_amp_cut_panel(
    cfg: RootConfig,
    symbols: list[str],
    spec: FactorSpec,
    logger,
    *,
    lookback_days: int,
    lam: float,
    min_day_minutes: int,
    min_valid_days: int,
    min_cross_section: int,
) -> _AmpCutMinuteLoad:
    """Compute the final amp-cut factor from the minute cache (stats loop + combine).

    Memory-bounded: one symbol's minute history is read, aggregated to its daily
    ``(V_mean, V_std)`` stats, and discarded before the next — the multi-year all-symbol
    minute panel is NEVER materialized. Read-only: :meth:`IntradayParquetStore.read_range`
    has no fetch closure, so ``stk_mins`` live calls are provably zero (a symbol with no
    cached minute simply yields no rows and is disclosed as empty). The per-day cut and
    trailing aggregation run inside ``compute_amp_cut_stats``; the CROSS-SECTIONAL z-score
    combine (report step 4) is applied ONCE to the assembled full-universe stats panel.
    """
    root = cfg.data.cache.root_dir
    store = IntradayParquetStore(root)
    start = pd.Timestamp(cfg.data.start).normalize()
    end = pd.Timestamp(cfg.data.end).normalize() + pd.Timedelta("23:59:59")

    stats_frames: list[pd.DataFrame] = []
    covered: list[str] = []
    empty: list[str] = []
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
        stats = compute_amp_cut_stats(
            bars,
            lookback_days=lookback_days,
            lam=lam,
            min_day_minutes=min_day_minutes,
            min_valid_days=min_valid_days,
        )
        # A symbol counts as covered iff it produced >= 1 finite (V_mean, V_std) pair
        # (those are the only rows that can enter a date's cross-section).
        if not stats.empty and stats[[V_MEAN_COL, V_STD_COL]].notna().all(axis=1).any():
            stats_frames.append(stats)
            covered.append(sym)
        else:
            empty.append(sym)
        if (i + 1) % 100 == 0:
            logger.info(
                "minute aggregation: %d/%d symbols processed (%d with a value)",
                i + 1, len(symbols), len(covered),
            )

    if not stats_frames:
        raise ValueError(
            "run-eval-intraday-amp-cut blocked: no requested symbol produced a cached "
            f"intraday-amp-cut stat over [{cfg.data.start}, {cfg.data.end}]. The minute "
            "cache is required (this runner never warms it); check coverage."
        )
    stats = pd.concat(stats_frames).sort_index()
    # ONE cross-sectional combine over the full covered universe (report step 4); the
    # CSI500 covered set is the cross-section (PINNED, disclosed on the spec).
    factor = combine_amp_cut_cross_section(
        stats, min_cross_section=min_cross_section, name=spec.factor_id
    )
    logger.info(
        "minute aggregation (cache-only): %d/%d symbols with a stat, %d raw 1min rows "
        "read, %d stats rows, %d factor rows, stk_mins_live_calls=0",
        len(covered), len(symbols), raw_rows, len(stats), len(factor),
    )
    return _AmpCutMinuteLoad(
        factor=factor,
        stats_rows=int(len(stats)),
        requested=len(symbols),
        covered=tuple(covered),
        empty_symbols=tuple(empty),
        raw_rows=raw_rows,
        live_calls=0,
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
            "run-eval-intraday-amp-cut requires an 'oos' section (split_date) so the "
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
        # screen of our own); the report's own factor screen is a caveat noted in the
        # run's prose, not our multiple-testing background.
        n_factors_screened=1,
        data_snapshot_id=cfg.data.cache.root_dir,
    )


# --------------------------------------------------------------------------- #
# Result container + the full glue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AmpCutEvalResult:
    """Immutable summary of one run-eval-intraday-amp-cut run."""

    config: RootConfig
    spec: FactorSpec
    requested_symbols: int
    covered_symbols: int
    empty_symbols: int
    factor_rows: int
    stats_rows: int
    minute_raw_rows: int
    minute_live_calls: int
    no_book_metrics: dict
    with_book_metrics: dict
    reports: _RunReports
    log_path: Path
    elapsed: float


def _check_preconditions(cfg: RootConfig) -> None:
    """Fail readably if the config cannot drive a real, cache-only CSI500 eval."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-eval-intraday-amp-cut needs data.source='tushare' (real cached A-share "
            f"data); got {cfg.data.source!r}."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-eval-intraday-amp-cut needs data.cache.enabled=true (it reads the "
            "persistent tushare cache and never warms live)."
        )
    if cfg.universe.type != "index":
        raise ValueError(
            "run-eval-intraday-amp-cut needs universe.type='index' (PIT membership, "
            f"e.g. 000905.SH for CSI500); got {cfg.universe.type!r}."
        )
    if not cfg.processing.neutralize.enabled:
        raise ValueError(
            "run-eval-intraday-amp-cut expects processing.neutralize.enabled=true "
            "(industry + size neutralization, matching the report's neutral column and "
            "the EvalConfig declaration)."
        )


def run_eval_intraday_amp_cut(config_path: str) -> AmpCutEvalResult:
    """Run the two real intraday-amp-cut evaluations (cache-only) + reports."""
    cfg = load_config(config_path)
    _check_preconditions(cfg)

    log_path = Path(cfg.output.log_dir) / f"{_REPORT_STEM}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    factor = IntradayAmpCutFactor(lookback_days=AMP_CUT_LOOKBACK_DAYS)
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

    # Amp-cut factor: cache-only per-symbol stats + ONE cross-sectional combine -> process.
    load = _load_amp_cut_panel(
        cfg, symbols, spec, logger,
        lookback_days=AMP_CUT_LOOKBACK_DAYS,
        lam=AMP_CUT_LAMBDA,
        min_day_minutes=AMP_CUT_MIN_DAY_MINUTES,
        min_valid_days=AMP_CUT_MIN_VALID_DAYS,
        min_cross_section=AMP_CUT_MIN_CROSS_SECTION,
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

    return AmpCutEvalResult(
        config=cfg,
        spec=spec,
        requested_symbols=load.requested,
        covered_symbols=len(load.covered),
        empty_symbols=len(load.empty_symbols),
        factor_rows=int(len(factor_series)),
        stats_rows=load.stats_rows,
        minute_raw_rows=load.raw_rows,
        minute_live_calls=load.live_calls,
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
    "AmpCutEvalResult",
    "evaluate_two_runs",
    "extract_metrics",
    "run_eval_intraday_amp_cut",
]
