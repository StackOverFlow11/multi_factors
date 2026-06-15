"""run-phase-i5a-intraday: an architecture smoke for the intraday tail event model.

This runner proves the shared event-driven backtest engine
(:class:`runtime.backtest.engine.BacktestEngine`) can drive an
:class:`runtime.backtest.event_models.IntradayTailEventModel` end-to-end on REAL
SH/SZ data — without a new research alpha and without touching the daily path.

Pipeline:

    build daily panel + universe (existing P4 read-through cache)
    -> load required 1min bars from the EXISTING intraday cache (read-only; a
       cache miss is a loud blocker, NEVER a silent warm -> zero stk_mins calls)
    -> deterministic PIT-safe score = the I3 ``intraday_ret_0930_1450`` feature
       (only bars with available_time <= 14:50 enter it)
    -> engine + IntradayTailEventModel: decision 14:50, execute at the first valid
       1min close in [14:51, 14:56:59], exec-to-exec holding returns
    -> NAV / feasibility / holdings / event logs -> markdown report.

It is intentionally small: the score is a single already-PIT-safe feature solely
to exercise the event framework, not a performance claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.cache.intervals import subtract_intervals
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import TushareIntradayCache
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_aggregate import asof_daily_features
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from portfolio.construct import TopNEqualWeight
from qt.config import RootConfig, load_config
from qt.pipeline import (
    _FrameScores,
    _build_cache,
    _build_universe,
    _load_panel,
    _make_logger,
)
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import IntradayExecutionConfig

_SCORE_FEATURE = "intraday_ret"  # the I3 feature family used as the smoke score
_LOGGER_NAME = "qt.intraday_tail_framework"


@dataclass(frozen=True)
class I5aResult:
    """Immutable summary of one I5a intraday-tail smoke run."""

    config: RootConfig
    event_order: str
    exec_cfg: IntradayExecutionConfig
    score_feature: str
    requested_symbols: int
    covered_symbols: int
    uncovered_symbols: tuple[str, ...]
    minute_live_calls: int
    nav_table: pd.DataFrame
    event_log: pd.DataFrame
    feasibility_log: pd.DataFrame
    holdings_log: pd.DataFrame
    blocked_fill_counts: dict[str, int]
    report_path: Path
    log_path: Path


def _exec_cfg_from(cfg: RootConfig) -> IntradayExecutionConfig:
    ic = cfg.intraday
    assert ic is not None  # guarded by the caller
    return IntradayExecutionConfig(
        decision_time=ic.decision_time,
        data_lag=ic.data_lag,
        execution_model=ic.execution_model,
        execution_window=tuple(ic.execution_window),
    )


def _load_minute_bars_cache_only(
    cfg: RootConfig, symbols: list[str], logger
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """Read 1min bars for ``symbols`` over the window from the cache ONLY.

    Returns ``(bars, covered_symbols, uncovered_symbols, live_calls)``. A symbol
    whose window is not fully covered by the intraday ledger is EXCLUDED (never
    warmed); the read uses a fetch closure that raises, so any planned gap is a
    loud error and ``live_calls`` is provably zero on the happy path.
    """
    root = cfg.data.cache.root_dir
    ledger = IntradayCoverageLedger(root)
    store = IntradayParquetStore(root)
    cache = TushareIntradayCache(store, ledger)

    req_start = pd.Timestamp(cfg.data.start).normalize()
    req_end = pd.Timestamp(cfg.data.end).normalize()
    covered: list[str] = []
    uncovered: list[str] = []
    for sym in symbols:
        gaps = subtract_intervals(
            req_start,
            req_end,
            ledger.covered_day_intervals(INTRADAY_ENDPOINT, sym, RAW_INTRADAY_FREQ),
        )
        (uncovered if gaps else covered).append(sym)

    require_cov = cfg.intraday is not None and cfg.intraday.require_cache_coverage
    if require_cov and not covered:
        raise ValueError(
            "intraday smoke blocked: NONE of the requested symbols are fully "
            f"covered in the minute cache for [{cfg.data.start}, {cfg.data.end}]. "
            "Shrink/move the window to covered SH/SZ data — this runner refuses to "
            "warm missing minute history."
        )
    if uncovered:
        logger.info(
            "intraday cache: %d/%d symbols covered; %d excluded (uncovered)",
            len(covered), len(symbols), len(uncovered),
        )

    def _no_warm(sym: str, start_dt: str, end_dt: str):
        raise RuntimeError(
            f"minute cache miss for {sym} [{start_dt}, {end_dt}]; the I5a runner "
            "is read-only and refuses to warm the intraday cache."
        )

    start_dt = f"{cfg.data.start} 00:00:00"
    end_dt = f"{cfg.data.end} 23:59:59"
    read = cache.stk_mins_1min(covered, start_dt, end_dt, _no_warm, freq=RAW_INTRADAY_FREQ)
    live_calls = int(cache.stats().get(INTRADAY_ENDPOINT, 0))
    if read.empty:
        raise ValueError(
            "intraday smoke blocked: the covered symbols returned no cached 1min "
            "bars for the window (unexpected — check the coverage ledger)."
        )
    bars = normalize_intraday_bars(
        read, freq=RAW_INTRADAY_FREQ, data_lag=cfg.intraday.data_lag
    )
    return bars, covered, uncovered, live_calls


def _score_panel(cfg: RootConfig, bars: pd.DataFrame, logger) -> tuple[pd.Series, str]:
    """Deterministic PIT-safe score = the I3 intraday return feature.

    ``asof_daily_features`` keeps only bars with ``available_time <= decision_time``
    before aggregating to (date, symbol), so the score for date T is known at T's
    14:50 cutoff — exactly the information a tail decision may use. Returns the
    score Series (named ``score``) and the source feature column name.
    """
    ic = cfg.intraday
    assert ic is not None
    feats = asof_daily_features(
        bars, decision_time=ic.decision_time, session_open=ic.session_open
    )
    col = next((c for c in feats.columns if c.startswith(_SCORE_FEATURE)), None)
    if col is None:
        raise ValueError(
            f"no {_SCORE_FEATURE!r} feature column produced by asof_daily_features "
            f"(got {list(feats.columns)})."
        )
    logger.info("intraday score: feature=%s, %d (date,symbol) rows", col, len(feats))
    return feats[col].rename("score"), col


def run_phase_i5a_intraday(config_path: str) -> I5aResult:
    """Run the I5a intraday-tail architecture smoke and write its report."""
    cfg = load_config(config_path)
    _check_i5a_preconditions(cfg)

    log_dir = Path(cfg.output.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase_i5a_intraday_tail_framework.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)  # daily qfq panel: selection + calendar

    exec_cfg = _exec_cfg_from(cfg)
    bars, covered, uncovered, live_calls = _load_minute_bars_cache_only(cfg, symbols, logger)
    score_series, score_feature = _score_panel(cfg, bars, logger)
    scores = _FrameScores(score_series)

    model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=exec_cfg)
    execution = SimExecution(fee_rate=cfg.cost.fee_rate)
    engine = BacktestEngine(
        model=model,
        universe=universe,
        scores=scores,
        constructor=TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only),
        execution=execution,
        selection_panel=panel,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    nav_table = engine.run()
    logger.info("intraday backtest: %d settled periods", len(nav_table))

    blocked = model.blocked_fills()
    blocked_counts: dict[str, int] = {}
    for f in blocked:
        blocked_counts[f.reason or "unknown"] = blocked_counts.get(f.reason or "unknown", 0) + 1

    report_path = Path(cfg.output.report_dir) / "phase_i5a_intraday_tail_framework.md"
    result = I5aResult(
        config=cfg,
        event_order=cfg.backtest.event_order,
        exec_cfg=exec_cfg,
        score_feature=score_feature,
        requested_symbols=len(symbols),
        covered_symbols=len(covered),
        uncovered_symbols=tuple(uncovered),
        minute_live_calls=live_calls,
        nav_table=nav_table,
        event_log=engine.event_log(),
        feasibility_log=engine.feasibility_log(),
        holdings_log=engine.holdings_log(),
        blocked_fill_counts=blocked_counts,
        report_path=report_path,
        log_path=log_path,
    )
    _write_report(result, elapsed=time.monotonic() - started)
    logger.info("report: %s", report_path)
    return result


def _check_i5a_preconditions(cfg: RootConfig) -> None:
    """Guards: real tushare source, intraday enabled, intraday event order, cache on."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-phase-i5a-intraday is a REAL-data smoke and requires "
            f"data.source='tushare' (got {cfg.data.source!r})."
        )
    if cfg.intraday is None or not cfg.intraday.enabled:
        raise ValueError(
            "run-phase-i5a-intraday requires an 'intraday' section with enabled=true."
        )
    if cfg.backtest.event_order != "intraday_tail_rebalance":
        raise ValueError(
            "run-phase-i5a-intraday requires backtest.event_order="
            f"'intraday_tail_rebalance' (got {cfg.backtest.event_order!r})."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-phase-i5a-intraday requires data.cache.enabled=true (the minute "
            "bars are read from the persistent intraday cache)."
        )


def _write_report(result: I5aResult, *, elapsed: float) -> None:
    """Write the I5a architecture-smoke markdown report (auditable event basis)."""
    cfg = result.config
    ec = result.exec_cfg
    path = result.report_path
    path.parent.mkdir(parents=True, exist_ok=True)

    nav = result.nav_table
    lines: list[str] = []
    lines.append("# Phase I5a — Intraday Tail-Rebalance Event Framework (architecture smoke)")
    lines.append("")
    lines.append(
        "**This is an architecture/framework run, NOT a research result.** The "
        "score is a single PIT-safe I3 feature used solely to exercise the shared "
        "event-driven backtest engine with an intraday tail event model."
    )
    lines.append("")
    lines.append("## Event model")
    lines.append("")
    lines.append("- event model: `IntradayTailEventModel`")
    lines.append(f"- backtest.event_order: `{result.event_order}`")
    lines.append(f"- decision_time (signal cutoff): `{ec.decision_time}`")
    lines.append(f"- data_lag (available_time = bar_end + lag): `{ec.data_lag}`")
    lines.append(f"- execution_model: `{ec.execution_model}`")
    lines.append(
        f"- execution_window: `[{ec.execution_window[0]}, {ec.execution_window[1]}]`"
    )
    lines.append(f"- score feature: `{result.score_feature}`")
    lines.append("")
    lines.append("**Returns are execution-to-execution, NOT close-to-close.** Holding "
                 "return of a period = exec_price(next) / exec_price(this) - 1, priced "
                 "at the first valid 1min close in the execution window.")
    lines.append("")
    lines.append("## Minute-cache coverage & data provenance")
    lines.append("")
    lines.append(f"- window: `{cfg.data.start}` → `{cfg.data.end}`")
    lines.append(f"- universe: `{cfg.universe.type}` `{cfg.universe.index_code or ''}`")
    lines.append(
        f"- requested symbols: {result.requested_symbols}; "
        f"minute-cache covered: {result.covered_symbols}; "
        f"excluded (uncovered): {len(result.uncovered_symbols)}"
    )
    lines.append(
        f"- **stk_mins live API calls during this run: {result.minute_live_calls}** "
        "(read-only; a cache miss is a hard blocker, never a silent warm)"
    )
    if result.uncovered_symbols:
        shown = ", ".join(result.uncovered_symbols[:10])
        more = "" if len(result.uncovered_symbols) <= 10 else f" (+{len(result.uncovered_symbols)-10} more)"
        lines.append(f"- excluded symbols: {shown}{more}")
    lines.append("")
    lines.append("## Event table (decision / execution / exit anchors)")
    lines.append("")
    if result.event_log.empty:
        lines.append("_no settled periods_")
    else:
        lines.append("| date | decision_ts | execution_ts | exit_date | next_decision_ts |")
        lines.append("|---|---|---|---|---|")
        for date, row in result.event_log.iterrows():
            lines.append(
                f"| {pd.Timestamp(date).date()} | {row['decision_ts']} | "
                f"{row['execution_ts']} | {pd.Timestamp(row['exit_date']).date()} | "
                f"{row['next_decision_ts']} |"
            )
    lines.append("")
    lines.append("## NAV / turnover / cost / cash")
    lines.append("")
    if nav.empty:
        lines.append("_no settled periods_")
    else:
        lines.append("| date | nav | net_return | gross_return | cost | turnover |")
        lines.append("|---|---|---|---|---|---|")
        for date, row in nav.iterrows():
            lines.append(
                f"| {pd.Timestamp(date).date()} | {row['nav']:.6f} | "
                f"{row['net_return']:.6f} | {row['gross_return']:.6f} | "
                f"{row['cost']:.6f} | {row['turnover']:.6f} |"
            )
        lines.append("")
        lines.append(
            f"- final NAV: {nav['nav'].iloc[-1]:.6f}; "
            f"avg turnover: {nav['turnover'].mean():.6f}; "
            f"total cost: {nav['cost'].sum():.6f}; cash_return: {cfg.backtest.cash_return}"
        )
        lines.append(
            "- turnover/cost count the ACHIEVED book after feasible fills, not the "
            "desired target."
        )
    lines.append("")
    lines.append("## Blocked fills (by reason)")
    lines.append("")
    if not result.blocked_fill_counts:
        lines.append("_none_")
    else:
        for reason, n in sorted(result.blocked_fill_counts.items()):
            lines.append(f"- `{reason}`: {n}")
    lines.append(
        "\nA blocked entry/exit bar excludes that symbol from the period (it earns "
        "nothing) and is NEVER replaced by a daily close."
    )
    lines.append("")
    lines.append("## Achieved holdings (sample)")
    lines.append("")
    h = result.holdings_log
    if h.empty:
        lines.append("_none_")
    else:
        lines.append("| date | symbol | weight | rank |")
        lines.append("|---|---|---|---|")
        for _, row in h.head(15).iterrows():
            lines.append(
                f"| {pd.Timestamp(row['date']).date()} | {row['symbol']} | "
                f"{row['weight']:.6f} | {int(row['rank'])} |"
            )
        if len(h) > 15:
            lines.append(f"| … | … | … | … | ({len(h)} rows total) |")
    lines.append("")
    lines.append("## Limitations (explicit)")
    lines.append("")
    lines.append(
        "- **Execution-time feasibility is the minimum I5a rule**: a missing/NaN "
        "execution bar blocks BOTH directions. Price-limit feasibility at execution "
        "time is NOT applied — the daily panel carries only daily-close-derived "
        "limit flags, which the I5a contract forbids using for execution. Raw "
        "`stk_limit` vs execution-minute-close is future work."
    )
    lines.append(
        "- **ST / suspension availability**: suspended stocks have no minute bars, "
        "so they are blocked by the missing-bar rule; explicit ST status is NOT "
        "consulted at execution time in this smoke."
    )
    lines.append(
        "- The score is a single PIT-safe intraday feature to prove the framework; "
        "this is not a performance claim and no parameters were tuned."
    )
    lines.append("")
    lines.append(f"_elapsed: {elapsed:.1f}s_")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
