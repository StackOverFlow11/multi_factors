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

import numpy as np
import pandas as pd

from data.cache.intervals import subtract_intervals
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import TushareIntradayCache
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_aggregate import asof_daily_features, mmp_valid_minute_counts
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from data.feed.tushare_flags import TushareFlagsFeed
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
from runtime.intraday_liquidity import (
    LiquidityDiagnostics,
    build_liquidity_diagnostics,
)

_LOGGER_NAME = "qt.intraday_tail_framework"
_DEFAULT_REPORT_NAME = "phase_i5a_intraday_tail_framework"


def _report_basename(cfg: RootConfig) -> str:
    """Report/log basename: configurable so I5b never clobbers the I5a artifact."""
    return cfg.output.intraday_report_name or _DEFAULT_REPORT_NAME


def _report_heading(
    score_feature: str, price_limit_check: bool, title_override: str | None = None
) -> tuple[str, str]:
    """(H1 title, intro paragraph) — names the actual study (I5a / I5b / I5c).

    The intro is keyed on the actual run (MMP factor study / execution-hardening /
    architecture smoke); ``title_override`` (config ``intraday_report_title``) wins
    for the H1 so a reused runner never keeps a stale phase label — the same
    stale-wording trap fixed for the P3-7/P3-8 reports.
    """
    if score_feature == "mmp_ew":
        title = "# Phase I5c — MMP Minute Factor Study"
        intro = (
            "**This is an EXPLORATORY minute-factor study, NOT a performance claim "
            "and NOT a broad factor search.** It runs ONE user-proposed minute "
            "factor — Minute Microstructure Pressure (MMP) — as a PIT-safe daily "
            "score through the I5a intraday tail event model with I5b raw "
            "`stk_limit` execution feasibility ON. No parameters were tuned from "
            "performance and no learned/IC weights are used."
        )
    elif price_limit_check:
        title = (
            "# Phase I5b — Intraday Execution-Time Price-Limit Feasibility "
            "(execution hardening)"
        )
        intro = (
            "**This is an execution-feasibility hardening run, NOT a research "
            "result.** It extends the I5a intraday tail event model with "
            "direction-aware raw `stk_limit` blocking at the execution minute; the "
            "score is still a single PIT-safe I3 feature used only to exercise the "
            "shared event-driven backtest engine."
        )
    else:
        title = (
            "# Phase I5a — Intraday Tail-Rebalance Event Framework (architecture smoke)"
        )
        intro = (
            "**This is an architecture/framework run, NOT a research result.** The "
            "score is a single PIT-safe I3 feature used solely to exercise the "
            "shared event-driven backtest engine with an intraday tail event model."
        )
    if title_override:
        t = title_override.strip()
        title = t if t.startswith("#") else f"# {t}"
    return title, intro


@dataclass(frozen=True)
class I5aResult:
    """Immutable summary of one I5a intraday-tail smoke run."""

    config: RootConfig
    event_order: str
    exec_cfg: IntradayExecutionConfig
    score_feature: str        # the resolved feature COLUMN (e.g. intraday_mmp20_ew_0930_1450)
    score_feature_key: str    # the config key (e.g. "ret", "mmp_ew") that selected it
    requested_symbols: int
    covered_symbols: int
    uncovered_symbols: tuple[str, ...]
    minute_live_calls: int
    nav_table: pd.DataFrame
    event_log: pd.DataFrame
    feasibility_log: pd.DataFrame
    holdings_log: pd.DataFrame
    blocked_fill_counts: dict[str, int]
    # per rebalance date -> (earliest, latest) ACTUAL execution-bar time among the
    # non-blocked fills (so a 14:52 fill when 14:51 was missing is auditable, vs
    # the planned execution_ts). Empty tuple key absent -> no fill that date.
    actual_exec_by_date: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]
    # I5b execution-time price-limit feasibility diagnostics.
    price_limit_check: bool
    limit_coverage: dict[str, int]
    stk_limit_gap_fetches: int
    up_limit_blocked_buys: int
    down_limit_blocked_sells: int
    missing_limit_rows: int
    # Where the VWAP gate diverges from a close-based one: the minute CLOSED at a
    # limit but traded through it, so the fill stands. Reported because it is the
    # only place the two gates disagree, and a silent divergence in what does or
    # does not trade is exactly what this layer exists to prevent.
    opened_limit_up_minutes: int
    opened_limit_down_minutes: int
    # Holding-period returns dropped for want of a usable adj_factor at an anchor.
    # Never defaulted to 1.0 (that would reintroduce the ex-date bias), so a
    # non-zero count is a coverage fact the reader must see.
    missing_adj_factor_pairs: int
    # I5c MMP factor diagnostics (report-only). Empty/None unless score is mmp_ew.
    score_coverage: dict[str, int]              # {rows, valid, nan} over the daily score panel
    minute_count_summary: dict[str, float] | None  # valid-MMP-minutes-per-(date,symbol) distribution
    factor_diagnostics: tuple[dict, ...]        # per settled rebalance: n / stats / Spearman IC
    # I5f execution liquidity diagnostics (report-only). None unless enabled.
    liquidity_diagnostics: LiquidityDiagnostics | None
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
        execution_price_basis=ic.execution_price_basis,
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
    if require_cov and uncovered:
        # require_cache_coverage=true must NOT silently drop uncovered names: that
        # would bias the realized universe. Fail loudly so the operator shrinks the
        # window (or opts into lenient mode). Matches the goal: coverage missing ->
        # change the window or stop, never warm.
        shown = ", ".join(uncovered[:10])
        more = "" if len(uncovered) <= 10 else f" (+{len(uncovered) - 10} more)"
        raise ValueError(
            "intraday smoke blocked: require_cache_coverage=true but "
            f"{len(uncovered)}/{len(symbols)} requested symbols are NOT fully covered "
            f"in the minute cache for [{cfg.data.start}, {cfg.data.end}]: {shown}{more}. "
            "Shrink/move the window to fully-covered SH/SZ data, or set "
            "intraday.require_cache_coverage=false to drop the uncovered names "
            "(disclosed). This runner refuses to warm missing minute history."
        )
    if uncovered:
        logger.info(
            "intraday cache: %d/%d symbols covered; %d excluded (uncovered, "
            "require_cache_coverage=false)",
            len(covered), len(symbols), len(uncovered),
        )
    if not covered:
        raise ValueError(
            "intraday smoke blocked: no requested symbol is covered in the minute "
            f"cache for [{cfg.data.start}, {cfg.data.end}]."
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


_STK_LIMIT_ENDPOINT = "stk_limit"


def _load_price_limits(
    cfg: RootConfig, covered: list[str], cache, logger
) -> tuple[pd.DataFrame | None, int]:
    """Load raw ``stk_limit`` rows for I5b execution-time limit feasibility.

    Returns ``(limits_df, stk_limit_gap_fetches)``. The limits flow through the
    SAME P4 read-through cache as the daily endpoints (no new endpoint); a
    fully-covered window costs zero gap fetches. The cache stores RAW
    ``up_limit`` / ``down_limit`` only — the model compares them to the RAW price
    that EXECUTES at the selected minute (the bar VWAP under the default basis),
    never qfq / daily close.
    """
    if cfg.intraday is None or not cfg.intraday.price_limit_check:
        return None, 0
    if not cfg.data.external_secret_file:
        raise ValueError(
            "intraday.price_limit_check=true requires data.external_secret_file "
            "(token for the raw stk_limit endpoint)."
        )
    before = int(cache.stats().get(_STK_LIMIT_ENDPOINT, 0)) if cache is not None else 0
    feed = TushareFlagsFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    limits = feed.limits(covered, cfg.data.start, cfg.data.end)
    after = int(cache.stats().get(_STK_LIMIT_ENDPOINT, 0)) if cache is not None else 0
    gap_fetches = after - before
    logger.info(
        "intraday price limits: %d raw stk_limit rows for %d covered symbols; "
        "stk_limit cache gap_fetches=%d (read-through P4 cache; raw up/down only)",
        len(limits), len(covered), gap_fetches,
    )
    return limits, gap_fetches


def _score_panel(cfg: RootConfig, bars: pd.DataFrame, logger) -> tuple[pd.Series, str]:
    """PIT-safe daily score = the configured I3 intraday feature (I5c).

    ``asof_daily_features`` keeps only bars with ``available_time <= decision_time``
    before aggregating to (date, symbol), so the score for date T is known at T's
    14:50 cutoff — exactly the information a tail decision may use. The feature is
    selected by ``intraday.score_feature`` (default ``ret`` reproduces I5a/I5b;
    ``mmp_ew`` is the exploratory MMP factor); only that one feature is requested
    and its returned column is used EXACTLY (no prefix matching). Returns the score
    Series (named ``score``) and the source feature column name.
    """
    ic = cfg.intraday
    assert ic is not None
    feats = asof_daily_features(
        bars,
        decision_time=ic.decision_time,
        session_open=ic.session_open,
        features=[ic.score_feature],
    )
    if feats.shape[1] != 1:
        raise ValueError(
            f"expected exactly one feature column for score_feature="
            f"{ic.score_feature!r}, got {list(feats.columns)}."
        )
    col = feats.columns[0]
    logger.info(
        "intraday score: feature_key=%s, column=%s, %d (date,symbol) rows",
        ic.score_feature, col, len(feats),
    )
    return feats[col].rename("score"), col


# Minimum cross-sectional sample for a meaningful Spearman IC; below it the IC is
# reported as NaN with the sample size, never silently computed on too few names.
_MIN_IC_SAMPLE = 5


def _score_coverage(score_series: pd.Series) -> dict[str, int]:
    """{rows, valid, nan} over the daily score panel (report-only)."""
    rows = int(score_series.shape[0])
    nan = int(score_series.isna().sum())
    return {"rows": rows, "valid": rows - nan, "nan": nan}


def _minute_count_summary(cfg: RootConfig, bars: pd.DataFrame) -> dict[str, float] | None:
    """Distribution of valid-MMP minutes per (date, symbol) (report-only).

    Reuses :func:`mmp_valid_minute_counts` (single MMP source of truth) under the
    SAME PIT cutoff. Returns None when there is nothing to summarize.
    """
    ic = cfg.intraday
    assert ic is not None
    counts = mmp_valid_minute_counts(
        bars, decision_time=ic.decision_time, session_open=ic.session_open
    )
    if counts.empty:
        return None
    arr = counts.to_numpy(dtype=float)
    return {
        "groups": float(counts.shape[0]),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _factor_diagnostics(
    score_series: pd.Series,
    model: IntradayTailEventModel,
    nav_table: pd.DataFrame,
    covered: list[str],
) -> list[dict]:
    """Per settled-rebalance score stats + Spearman IC vs exec-to-exec return.

    REPORT-ONLY: the IC correlates the decision-date score with the SAME period's
    execution-to-execution return the event model realizes (``holding_returns``),
    over the covered, non-NaN-scored cross-section. Returns are read here purely
    for analytics — never fed back into the factor/alpha layer (invariant #1).
    """
    covered_set = {str(s) for s in covered}
    settled = {pd.Timestamp(d).normalize() for d in nav_table.index}
    periods = [
        p for p in model.holding_periods()
        if pd.Timestamp(p.date).normalize() in settled
    ]
    rows: list[dict] = []
    for p in periods:
        d = pd.Timestamp(p.date).normalize()
        try:
            cross = score_series.xs(d, level="date")
        except KeyError:
            continue
        cross = cross[[s for s in cross.index if str(s) in covered_set]].dropna()
        n = int(cross.shape[0])
        if n:
            arr = cross.to_numpy(dtype=float)
            stats = {
                "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }
        else:
            stats = {k: float("nan") for k in ("mean", "std", "p10", "p50", "p90")}
        rets = model.holding_returns(p, list(cross.index))  # exec-to-exec; omits blocked
        joined = pd.concat(
            [cross.rename("score"), rets.rename("ret")], axis=1
        ).dropna()
        ic_n = int(joined.shape[0])
        ic_val = (
            float(joined["score"].corr(joined["ret"], method="spearman"))
            if ic_n >= _MIN_IC_SAMPLE else float("nan")
        )
        rows.append({"date": d, "n_scored": n, **stats, "ic": ic_val, "ic_n": ic_n})
    return rows


def run_phase_i5a_intraday(config_path: str) -> I5aResult:
    """Run the I5a intraday-tail architecture smoke and write its report."""
    cfg = load_config(config_path)
    _check_i5a_preconditions(cfg)

    basename = _report_basename(cfg)
    log_dir = Path(cfg.output.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{basename}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)  # daily qfq panel: selection + calendar

    exec_cfg = _exec_cfg_from(cfg)
    bars, covered, uncovered, live_calls = _load_minute_bars_cache_only(cfg, symbols, logger)
    score_series, score_feature = _score_panel(cfg, bars, logger)
    scores = _FrameScores(score_series)

    price_limits, stk_limit_gap_fetches = _load_price_limits(cfg, covered, cache, logger)
    ic = cfg.intraday
    assert ic is not None  # guarded by _check_i5a_preconditions
    model = IntradayTailEventModel(
        calendar_panel=panel,
        bars=bars,
        cfg=exec_cfg,
        price_limits=price_limits,
        price_limit_check=ic.price_limit_check,
        limit_tolerance=ic.limit_tolerance,
        require_price_limit_coverage=ic.require_price_limit_coverage,
    )
    execution = SimExecution(fee_rate=cfg.cost.fee_rate)
    ld_cfg = ic.liquidity_diagnostics
    engine = BacktestEngine(
        model=model,
        universe=universe,
        scores=scores,
        constructor=TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only),
        execution=execution,
        selection_panel=panel,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
        # I5f: record the per-rebalance (target, current) plan ONLY when the
        # report-only liquidity diagnostics are enabled. Default-off keeps the
        # engine loop byte-identical (no plan rows, no behaviour change).
        record_rebalance_plan=ld_cfg.enabled,
    )
    nav_table = engine.run()
    logger.info("intraday backtest: %d settled periods", len(nav_table))

    # I5f execution liquidity diagnostics — REPORT-ONLY, computed AFTER settlement
    # from logs the backtest already produced; it never altered fills or NAV.
    liquidity_diagnostics = None
    if ld_cfg.enabled:
        liquidity_diagnostics = build_liquidity_diagnostics(
            plan_log=engine.rebalance_plan_log(),
            fills=model.fills(),
            up_blocked_buy_keys=model.up_limit_blocked_buy_keys(),
            down_blocked_sell_keys=model.down_limit_blocked_sell_keys(),
            bars=bars,
            portfolio_notional=ld_cfg.portfolio_notional,
            max_participation_rate=ld_cfg.max_participation_rate,
        )
        logger.info(
            "intraday liquidity diagnostics (report-only): %d desired trades, "
            "%d feasibility-blocked, %d missing-capacity, %d inspected, %d below 1.0x "
            "capacity (did NOT alter fills/NAV)",
            liquidity_diagnostics.total_desired_trades,
            liquidity_diagnostics.feasibility_blocked,
            liquidity_diagnostics.missing_capacity_rows,
            liquidity_diagnostics.inspected,
            liquidity_diagnostics.below_capacity,
        )

    blocked = model.blocked_fills()
    blocked_counts: dict[str, int] = {}
    for f in blocked:
        blocked_counts[f.reason or "unknown"] = blocked_counts.get(f.reason or "unknown", 0) + 1

    # ACTUAL execution-bar time range per rebalance date (non-blocked fills): a fill
    # later than the planned execution_ts (e.g. 14:52 when 14:51 was missing) is
    # auditable here, distinct from the planned anchor in the event log.
    actual_exec: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}
    rebalance_dates = {pd.Timestamp(d).normalize() for d in nav_table.index}
    for f in model.fills():
        if f.blocked or f.exec_time is None:
            continue
        d = pd.Timestamp(f.date).normalize()
        if d not in rebalance_dates:
            continue  # exit-only anchors are not rebalance rows
        t = pd.Timestamp(f.exec_time)
        lo, hi = actual_exec.get(d, (t, t))
        actual_exec[d] = (min(lo, t), max(hi, t))

    # I5c factor diagnostics (report-only): score coverage, valid-MMP-minute
    # distribution (mmp only), and per-rebalance score stats + Spearman IC.
    score_coverage = _score_coverage(score_series)
    minute_count_summary = (
        _minute_count_summary(cfg, bars) if ic.score_feature == "mmp_ew" else None
    )
    factor_diag = _factor_diagnostics(score_series, model, nav_table, covered)

    report_path = Path(cfg.output.report_dir) / f"{basename}.md"
    result = I5aResult(
        config=cfg,
        event_order=cfg.backtest.event_order,
        exec_cfg=exec_cfg,
        score_feature=score_feature,
        score_feature_key=ic.score_feature,
        requested_symbols=len(symbols),
        covered_symbols=len(covered),
        uncovered_symbols=tuple(uncovered),
        minute_live_calls=live_calls,
        nav_table=nav_table,
        event_log=engine.event_log(),
        feasibility_log=engine.feasibility_log(),
        holdings_log=engine.holdings_log(),
        blocked_fill_counts=blocked_counts,
        actual_exec_by_date=actual_exec,
        price_limit_check=model.price_limit_check_enabled(),
        limit_coverage=model.limit_coverage(),
        stk_limit_gap_fetches=stk_limit_gap_fetches,
        up_limit_blocked_buys=model.up_limit_blocked_buys(),
        down_limit_blocked_sells=model.down_limit_blocked_sells(),
        missing_limit_rows=model.missing_limit_rows(),
        opened_limit_up_minutes=model.opened_limit_up_minutes(),
        opened_limit_down_minutes=model.opened_limit_down_minutes(),
        missing_adj_factor_pairs=model.missing_adj_factor_pairs(),
        score_coverage=score_coverage,
        minute_count_summary=minute_count_summary,
        factor_diagnostics=tuple(factor_diag),
        liquidity_diagnostics=liquidity_diagnostics,
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


def _append_factor_section(lines: list[str], result: I5aResult) -> None:
    """Score-feature disclosure + factor diagnostics (report-only, I5c §4/§5)."""
    if result.score_feature_key == "mmp_ew":
        lines.append("## MMP minute factor (definition & PIT)")
        lines.append("")
        lines.append(
            "Per 1min bar `t`: `mid=(high+low)/2`; `S=(close-mid)/mid`; "
            "`V=sqrt(volume/median(vol[t-20:t]))`; `B=|close-open|/(high-low+eps)`; "
            "`R=(high-low)/(mean(hl[t-20:t])+eps)`; **`MMP_t = S*V*B*R`** "
            "(`eps=1e-6`)."
        )
        lines.append(
            "- **daily score** `intraday_mmp20_ew_0930_1450` = the EQUAL-WEIGHT mean "
            "of valid `MMP_t` over the bars visible at the cutoff "
            "(`[session_open, decision_time]`). The volume term lives inside "
            "`MMP_t`; the daily aggregation is NOT additionally volume-weighted."
        )
        lines.append(
            "- **rolling baselines** use ONLY the prior 20 bars `t-20..t-1` within "
            "the SAME `(symbol, trade_date)` session — never bar `t`, never a later "
            "bar, never the prior day's tail. The first 20 bars of each session "
            "have NaN `MMP`."
        )
        lines.append(
            "- **PIT**: a bar enters only if `available_time <= trade_date + "
            "decision_time`; the filter runs on per-bar timestamps BEFORE daily "
            "grouping, so post-cutoff / late-available bars cannot move the score."
        )
        mc = result.minute_count_summary
        if mc is not None:
            lines.append(
                f"- **valid-MMP minutes per (date,symbol)**: groups "
                f"{int(mc['groups'])}; min {int(mc['min'])} / p10 {int(mc['p10'])} / "
                f"p50 {int(mc['p50'])} / p90 {int(mc['p90'])} / max {int(mc['max'])} "
                f"(mean {mc['mean']:.1f})."
            )
        lines.append("")
    sc = result.score_coverage
    lines.append("## Score coverage & factor diagnostics (report-only)")
    lines.append("")
    lines.append(
        f"- daily score panel: {sc['rows']} (date,symbol) rows; "
        f"valid {sc['valid']}; NaN {sc['nan']}."
    )
    diag = result.factor_diagnostics
    if not diag:
        lines.append("- _no settled rebalance to diagnose._")
        lines.append("")
        return
    lines.append(
        "- per settled rebalance — cross-sectional score stats over covered, "
        "non-NaN-scored names, and **Spearman IC** of the decision-date score vs "
        "the SAME period's execution-to-execution return "
        f"(report-only; min sample {_MIN_IC_SAMPLE}, else NaN):"
    )
    lines.append("")
    lines.append("| date | n_scored | mean | std | p10 | p50 | p90 | IC(spearman) | IC_n |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in diag:
        ic_str = "NaN" if pd.isna(r["ic"]) else f"{r['ic']:.4f}"
        lines.append(
            f"| {pd.Timestamp(r['date']).date()} | {r['n_scored']} | "
            f"{r['mean']:.6f} | {r['std']:.6f} | {r['p10']:.6f} | {r['p50']:.6f} | "
            f"{r['p90']:.6f} | {ic_str} | {r['ic_n']} |"
        )
    ics = [r["ic"] for r in diag if not pd.isna(r["ic"])]
    if ics:
        lines.append("")
        lines.append(
            f"- mean IC across {len(ics)} settled period(s) with sufficient sample: "
            f"{float(np.mean(ics)):.4f}. **Report-only, NOT a performance claim** — "
            "a 1–3 period intraday smoke is far too small to infer factor quality; "
            "returns are read here for analytics only and never feed the factor/alpha."
        )
    lines.append("")


def _append_liquidity_section(lines: list[str], result: I5aResult) -> None:
    """Report-only execution liquidity diagnostics (I5f). No-op if disabled."""
    ld = result.liquidity_diagnostics
    if ld is None:
        return
    lines.append("## Execution liquidity diagnostics (I5f, report-only)")
    lines.append("")
    lines.append(
        "**Report-only: these diagnostics did NOT alter fills, can_buy/can_sell, "
        "blocked reasons, target weights, achieved holdings, turnover, cost, or NAV.** "
        "They size each desired rebalance trade against the SELECTED execution-minute "
        "1min bar's traded `amount` (RMB) at a participation cap — no future bar, no "
        "daily amount/volume/close, no EOD proxy."
    )
    lines.append("")
    lines.append(
        f"- desired trade notional = `|target_weight - current_weight| * "
        f"portfolio_notional`; portfolio_notional = `{ld.portfolio_notional:,.0f}` RMB "
        "(illustrative diagnostic sizing, NOT a performance claim)."
    )
    lines.append(
        f"- bar capacity = `execution_minute_amount * max_participation_rate`; "
        f"max_participation_rate = `{ld.max_participation_rate}`."
    )
    lines.append(
        "- capacity_ratio = bar_capacity / desired_notional; `>= 1` ⇒ the capped bar "
        "covers the trade, `< 1` ⇒ potentially liquidity-constrained."
    )
    lines.append("")
    lines.append(f"- total desired trades inspected: {ld.total_desired_trades}")
    lines.append(
        f"- excluded — existing execution feasibility already blocked them "
        f"(missing bar / missing price / raw stk_limit; original reason kept, NOT "
        f"reclassified as liquidity): {ld.feasibility_blocked}"
    )
    lines.append(f"- missing capacity-data rows (amount missing/NaN/≤0): {ld.missing_capacity_rows}")
    lines.append(f"- trades with a usable capacity ratio: {ld.inspected}")
    lines.append(f"- trades below 100% capacity (ratio < 1.0): {ld.below_capacity}")
    rs = ld.ratio_stats
    if ld.inspected > 0:
        lines.append(
            f"- capacity ratio distribution — min `{rs['min']:.3f}` / p10 "
            f"`{rs['p10']:.3f}` / median `{rs['median']:.3f}` / p90 `{rs['p90']:.3f}`."
        )
    else:
        lines.append("- capacity ratio distribution: _no inspected trade with a usable ratio._")
    lines.append("")
    if ld.top_constrained:
        lines.append(
            "Top constrained trades (lowest capacity ratio first; report-only):"
        )
        lines.append("")
        lines.append(
            "| date | symbol | direction | desired_notional | bar_capacity_notional | capacity_ratio |"
        )
        lines.append("|---|---|---|---|---|---|")
        for t in ld.top_constrained:
            lines.append(
                f"| {pd.Timestamp(t.date).date()} | {t.symbol} | {t.direction} | "
                f"{t.desired_notional:,.0f} | {t.bar_capacity_notional:,.0f} | "
                f"{t.capacity_ratio:.3f} |"
            )
    else:
        lines.append("_no constrained trade to list._")
    lines.append("")
    lines.append(
        "- **No alpha / performance / tradability claim** is made from this "
        "diagnostic: it only flags where a single execution minute may be too thin "
        "for the sized trade; partial-fill / volume-cap enforcement is explicitly "
        "out of scope here."
    )
    lines.append("")


def limit_basis_lines(execution_price_basis: str) -> list[str]:
    """Report prose stating what the I5b price-limit gate ACTUALLY compares.

    Extracted so it is testable. The gate reads the price that EXECUTES — which
    since PR #75 is the bar VWAP by default, not the bar close — and an earlier
    revision of this text kept claiming "the raw 1min close" after the gate input
    had changed. A report that misstates the check it performed is worse than no
    report, so the wording is derived from the active basis and pinned by test.
    """
    return [
        f"- **comparison basis**: the price that actually EXECUTES — the selected "
        f"execution minute on the `{execution_price_basis}` basis — vs the "
        f"symbol/date **raw** `stk_limit` band. NOT qfq, NOT the daily close, NOT "
        f"a daily-close-derived limit flag. (The intraday cache stores unadjusted "
        f"bars and `stk_limit` is raw, and a bar VWAP is raw amount over raw "
        f"volume, so the comparison is RAW-vs-RAW either way.)",
        "- **why the executed price and not the bar close**: a limit-up minute has "
        "two shapes. LOCKED (封死涨停) — every print is at the limit, so the VWAP "
        "equals it up to rounding and the buy must be blocked. OPENED (盘中打开) — "
        "some prints landed below the limit, which is direct evidence a fill was "
        "achievable, so the buy must go through. The executed price separates "
        "them; the bar close misclassifies both edges.",
    ]


def _write_report(result: I5aResult, *, elapsed: float) -> None:
    """Write the I5a architecture-smoke markdown report (auditable event basis)."""
    cfg = result.config
    ec = result.exec_cfg
    path = result.report_path
    path.parent.mkdir(parents=True, exist_ok=True)

    nav = result.nav_table
    title, intro = _report_heading(
        result.score_feature_key, result.price_limit_check,
        cfg.output.intraday_report_title,
    )
    lines: list[str] = []
    lines.append(title)
    lines.append("")
    lines.append(intro)
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
    lines.append(
        f"- execution_price_basis: `{ec.execution_price_basis}` "
        "(`bar_vwap` = the selected 1min bar's amount/volume, RAW unadjusted; "
        "`bar_close` = that bar's single closing tick)"
    )
    lines.append(
        f"- score feature: `{result.score_feature}` "
        f"(key=`{result.score_feature_key}`)"
    )
    lines.append("")
    lines.append("**Returns are execution-to-execution, NOT close-to-close.** Holding "
                 "return of a period = exec_price(next) / exec_price(this) - 1, priced "
                 "at the first valid 1min bar in the execution window on the "
                 f"`{ec.execution_price_basis}` basis.")
    lines.append("")
    lines.append(
        "**Returns are corporate-action adjusted.** The cached minute bars are raw, "
        "so a holding period spanning an ex-dividend or split date would otherwise "
        "book the mechanical price drop as a loss. The return divides it out as "
        "`(raw_exit * adj_factor(exit)) / (raw_entry * adj_factor(entry)) - 1`; the "
        "per-symbol anchor cancels in the ratio, so nothing is re-derived. Fills "
        "still PAY the raw execution price and the price-limit gate stays raw — only "
        "the measured return is adjusted. Periods dropped for want of a usable "
        f"adj_factor at an anchor: {result.missing_adj_factor_pairs} "
        "(never defaulted to 1.0, which would reintroduce the bias)."
    )
    lines.append("")
    _append_factor_section(lines, result)
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
    lines.append(
        "`exec_ts(planned)` is the window start; `exec_bar(actual)` is the actual "
        "fill-bar time range used across the held names (e.g. `14:52:00` would show "
        "if 14:51 was missing). `exit_exec_ts` is the planned exit fill time at "
        "`exit_date` — the holding period runs `exec_ts -> exit_exec_ts`."
    )
    lines.append("")
    if result.event_log.empty:
        lines.append("_no settled periods_")
    else:
        lines.append(
            "| date | decision_ts | exec_ts(planned) | exec_bar(actual) | "
            "exit_date | exit_exec_ts |"
        )
        lines.append("|---|---|---|---|---|---|")
        for date, row in result.event_log.iterrows():
            actual = result.actual_exec_by_date.get(pd.Timestamp(date).normalize())
            if actual is None:
                actual_str = "—"
            else:
                lo, hi = actual
                actual_str = (
                    f"{lo.time()}" if lo == hi else f"{lo.time()}–{hi.time()}"
                )
            lines.append(
                f"| {pd.Timestamp(date).date()} | {row['decision_ts']} | "
                f"{row['execution_ts']} | {actual_str} | "
                f"{pd.Timestamp(row['exit_date']).date()} | {row['exit_execution_ts']} |"
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
    lines.append("## Execution-time price-limit feasibility (I5b)")
    lines.append("")
    if not result.price_limit_check:
        lines.append(
            "- **disabled** (`intraday.price_limit_check=false`): feasibility is the "
            "base bar-exists rule only (missing/NaN execution bar blocks both "
            "directions). No raw `stk_limit` comparison was performed."
        )
    else:
        cov = result.limit_coverage
        lines.append(
            "- **enabled** (`intraday.price_limit_check=true`): a buy is blocked at "
            "the raw upper limit and a sell at the raw lower limit, directionally."
        )
        lines.extend(limit_basis_lines(ec.execution_price_basis))
        lines.append(
            f"- limit tolerance (raw-price equality band): "
            f"`{cfg.intraday.limit_tolerance}`; "
            f"require_price_limit_coverage: `{cfg.intraday.require_price_limit_coverage}`"
        )
        lines.append(
            f"- limit coverage over rebalance anchors: required "
            f"{cov.get('required', 0)} (rebalance date, symbol) pairs; present "
            f"{cov.get('present', 0)}; missing {cov.get('missing', 0)}"
        )
        lines.append(
            f"- **stk_limit cache gap-fetches this run: {result.stk_limit_gap_fetches}** "
            "(read through the existing P4 cache — only uncovered date ranges or the "
            "recent-tail refresh window hit the API; never a minute/stk_mins fetch. A "
            "window whose tail is older than refresh_recent_days costs 0; a recent "
            "window re-pulls its tail by policy.)"
        )
        lines.append(
            f"- buys blocked by a raw up-limit: {result.up_limit_blocked_buys}; "
            f"sells blocked by a raw down-limit: {result.down_limit_blocked_sells}"
        )
        lines.append(
            f"- minutes that CLOSED at a limit but traded through it, so the fill "
            f"was allowed: {result.opened_limit_up_minutes} up / "
            f"{result.opened_limit_down_minutes} down. This is the entire set on "
            f"which this gate and a close-based gate disagree; a close-based gate "
            f"would have blocked these."
        )
        lines.append(
            f"- evaluated pairs with no usable raw limit row (unchecked): "
            f"{result.missing_limit_rows} "
            "(lenient mode counts/discloses these and falls back to the bar-exists "
            "rule; strict mode fails before any result — never a silent passed check)"
        )
    lines.append("")
    _append_liquidity_section(lines, result)
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
    if result.price_limit_check:
        lines.append(
            f"- **Execution-time feasibility (I5b)**: a missing/NaN execution bar "
            f"blocks BOTH directions; on top of that, raw `stk_limit` blocks buys at "
            f"the upper limit and sells at the lower limit, comparing the price that "
            f"actually EXECUTES — the selected execution minute on the "
            f"`{ec.execution_price_basis}` basis — to raw limits (see the section "
            f"above). Remaining gap: only the price-limit and bar-existence "
            f"constraints are modeled; partial-fill / liquidity / volume caps at the "
            f"execution minute are not."
        )
    else:
        lines.append(
            "- **Execution-time feasibility is the base bar-exists rule**: a "
            "missing/NaN execution bar blocks BOTH directions. Raw `stk_limit` "
            "price-limit feasibility is available (`intraday.price_limit_check`) but "
            "DISABLED in this config, so no limit comparison was applied."
        )
    lines.append(
        "- **ST / suspension availability**: suspended stocks have no minute bars, "
        "so they are blocked by the missing-bar rule; explicit ST status is NOT "
        "consulted at execution time in this smoke."
    )
    if result.score_feature_key == "mmp_ew":
        lines.append(
            "- **EXPLORATORY single-factor study, NOT a performance claim**: ONE "
            "user-proposed minute factor (MMP) on one short SSE50 window. No "
            "parameter was tuned from performance, no robustness matrix, no learned "
            "/ IC-weighted alpha. The IC above is report-only over 1–3 periods — far "
            "too small to infer factor quality; final NAV is reported, not claimed."
        )
    else:
        lines.append(
            "- The score is a single PIT-safe intraday feature to prove the "
            "framework; this is not a performance claim and no parameters were tuned."
        )
    lines.append("")
    lines.append(f"_elapsed: {elapsed:.1f}s_")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
