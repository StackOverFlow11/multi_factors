"""run-phase-i5d-intraday-groups: a 5-quantile grouped intraday-tail backtest of
the I5c MMP daily score (EXPLORATORY factor analysis — NOT a performance claim).

On each monthly rebalance date the cross-section is ranked by the PIT-safe daily
score ``intraday_mmp20_ew_0930_1450`` (the I5c Minute Microstructure Pressure
factor, visible at the 14:50 cutoff) and split into ``analytics.quantiles``
EQUAL-COUNT groups (Q1 = lowest score, QN = highest). Each group is run as its own
long-only equal-weight portfolio through the SAME event-driven machinery the daily
and I5a/I5b intraday paths use — a fresh :class:`BacktestEngine` +
:class:`IntradayTailEventModel` + ``SimExecution(fee_rate=...)`` per group — so
14:51 execution pricing, exec-to-exec holding returns, raw ``stk_limit`` execution
feasibility (I5b), turnover, cost and cash are modelled consistently and NEVER
fall back to a daily close-to-close return.

Two engineering constraints drive the design:

  * **memory** — a 5-year CSI500 minute history is enormous, but the monthly event
    model only needs minute bars on the ANCHOR dates (each rebalance/exit date).
    :func:`_load_anchor_minute_bars` reads ONLY those days from the persistent
    intraday cache (read-only; a miss never warms the cache → zero ``stk_mins``
    live calls), so the run never materialises the full-window minute panel.
  * **compute** — the execution-price matrix is a pure function of
    ``(bars, anchors, symbols, cfg)``; it is built ONCE and shared (immutable)
    across the N fresh per-group models, while each model keeps its OWN mutable
    feasibility diagnostics (no cross-group contamination).

The MMP score, the factor math, the daily backtest, and the execution feasibility
are all UNCHANGED — only the grouping + per-group orchestration is new.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache.intervals import subtract_intervals
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import READ_COLUMNS
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from portfolio.base import PortfolioConstructor
from qt.config import RootConfig, load_config

# Reuse the stable I5a/I5b/I5c helpers rather than copy-paste them (goal §1: avoid
# drift). These are import-stable private helpers: run preconditions, the exec
# config builder, the configured-feature daily score, and the raw stk_limit loader.
from qt.intraday_groups import EqualWeightAll, GroupScores, assign_quantile_buckets
from qt.intraday_tail_framework import (
    _check_i5a_preconditions,
    _exec_cfg_from,
    _load_price_limits,
    _score_panel,
)
from qt.pipeline import _build_cache, _build_universe, _load_panel, _make_logger
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.backtest.events import monthly_anchor_pairs, trading_calendar
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import IntradayExecutionConfig

_LOGGER_NAME = "qt.intraday_group_backtest"
_DEFAULT_REPORT_NAME = "phase_i5d_mmp_quintile_5y"
_DEFAULT_REPORT_TITLE = "Phase I5d — MMP Quintile 5Y Group Backtest"

# Monthly rebalance -> 12 periods/year for annualization.
_PERIODS_PER_YEAR = 12.0


# --------------------------------------------------------------------------- #
# Per-group performance metrics (monthly net-return series -> annualized stats)
# --------------------------------------------------------------------------- #
def _annualized_return(nav: pd.DataFrame) -> float:
    """CAGR from the settled NAV path (monthly compounding)."""
    if nav.empty:
        return float("nan")
    final = float(nav["nav"].iloc[-1])
    n = len(nav)
    if n <= 0 or final <= 0:
        return float("nan")
    return final ** (_PERIODS_PER_YEAR / n) - 1.0


def _annualized_vol(nav: pd.DataFrame) -> float:
    """Annualized volatility of the monthly net returns."""
    if len(nav) < 2:
        return float("nan")
    return float(nav["net_return"].std(ddof=1) * np.sqrt(_PERIODS_PER_YEAR))


def _sharpe(nav: pd.DataFrame) -> float:
    """Annualized Sharpe (rf=0) = mean(net)*12 / (std(net)*sqrt(12))."""
    sigma = _annualized_vol(nav)
    if not np.isfinite(sigma) or sigma == 0.0:
        return float("nan")
    mu = float(nav["net_return"].mean()) * _PERIODS_PER_YEAR
    return mu / sigma


def _max_drawdown(nav: pd.DataFrame) -> float:
    """Worst peak-to-trough drawdown of the NAV path (incl. the 1.0 start)."""
    if nav.empty:
        return float("nan")
    path = np.concatenate([[1.0], nav["nav"].to_numpy(dtype=float)])
    running_max = np.maximum.accumulate(path)
    return float(np.min(path / running_max - 1.0))


def _avg_holdings(holdings_log: pd.DataFrame) -> float:
    """Mean number of achieved holdings per settled rebalance date."""
    if holdings_log is None or holdings_log.empty:
        return 0.0
    return float(holdings_log.groupby("date").size().mean())


def _group_metrics(nav: pd.DataFrame, holdings_log: pd.DataFrame) -> dict[str, float]:
    """Headline per-group metrics from one group's NAV table + holdings log."""
    final_nav = float(nav["nav"].iloc[-1]) if not nav.empty else float("nan")
    return {
        "final_nav": final_nav,
        "annual_return": _annualized_return(nav),
        "volatility": _annualized_vol(nav),
        "sharpe": _sharpe(nav),
        "max_drawdown": _max_drawdown(nav),
        "mean_turnover": float(nav["turnover"].mean()) if not nav.empty else float("nan"),
        "total_cost": float(nav["cost"].sum()) if not nav.empty else 0.0,
        "avg_holdings": _avg_holdings(holdings_log),
        "n_periods": int(len(nav)),
    }


# --------------------------------------------------------------------------- #
# Minute loading (cache-only, anchor-date-sliced -> memory-conscious)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _MinuteLoad:
    """Diagnostics from the cache-only, anchor-date-sliced minute read."""

    bars: pd.DataFrame
    covered: list[str]
    uncovered: list[str]
    raw_rows: int
    normalized_rows: int
    anchor_dates: int
    live_calls: int


def _coverage_partition(
    cfg: RootConfig, symbols: list[str], root: str
) -> tuple[list[str], list[str]]:
    """Split ``symbols`` into (fully covered, uncovered) over the data window.

    A symbol is covered only if the intraday ledger leaves NO uncovered trading-day
    gap across ``[data.start, data.end]`` — exactly the I5a rule, so the realized
    universe is auditable and never silently shrunk by a partial gap.
    """
    ledger = IntradayCoverageLedger(root)
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
    return covered, uncovered


def _load_anchor_minute_bars(
    cfg: RootConfig,
    symbols: list[str],
    anchor_dates: list[pd.Timestamp],
    logger,
) -> _MinuteLoad:
    """Read 1min bars for ONLY the ``anchor_dates`` from the cache (read-only).

    Memory-conscious: the monthly event model prices only the rebalance/exit
    anchors, so we read one trading day at a time from the month-partitioned store
    and keep just those rows — the full multi-year minute history is NEVER
    materialised. No fetch closure exists here, so ``stk_mins`` live calls are
    provably zero (a miss simply yields no rows for that day/symbol). Uncovered
    symbols are handled exactly as I5a: in ``require_cache_coverage`` mode a single
    uncovered name is a loud blocker; otherwise they are excluded and disclosed.
    """
    root = cfg.data.cache.root_dir
    covered, uncovered = _coverage_partition(cfg, symbols, root)

    require_cov = cfg.intraday is not None and cfg.intraday.require_cache_coverage
    if require_cov and uncovered:
        shown = ", ".join(uncovered[:10])
        more = "" if len(uncovered) <= 10 else f" (+{len(uncovered) - 10} more)"
        raise ValueError(
            "i5d grouped backtest blocked: require_cache_coverage=true but "
            f"{len(uncovered)}/{len(symbols)} requested symbols are NOT fully "
            f"covered in the minute cache for [{cfg.data.start}, {cfg.data.end}]: "
            f"{shown}{more}. Set intraday.require_cache_coverage=false to drop the "
            "uncovered names (disclosed), or shrink the window to fully-covered "
            "data. This runner refuses to warm missing minute history."
        )
    if uncovered:
        logger.info(
            "intraday cache: %d/%d symbols fully covered; %d excluded (uncovered, "
            "require_cache_coverage=false)",
            len(covered), len(symbols), len(uncovered),
        )
    if not covered:
        raise ValueError(
            "i5d grouped backtest blocked: no requested symbol is fully covered in "
            f"the minute cache for [{cfg.data.start}, {cfg.data.end}]."
        )

    store = IntradayParquetStore(root)
    frames: list[pd.DataFrame] = []
    raw_rows = 0
    for anchor in anchor_dates:
        day = pd.Timestamp(anchor).normalize()
        day_end = day + pd.Timedelta("23:59:59")
        for sym in covered:
            part = store.read_range(
                INTRADAY_ENDPOINT, sym, RAW_INTRADAY_FREQ, day, day_end
            )
            if not part.empty:
                hit = part.rename(columns={"bar_end": "time"})
                frames.append(hit[READ_COLUMNS])
                raw_rows += len(part)
    if not frames:
        raise ValueError(
            "i5d grouped backtest blocked: the covered symbols returned no cached "
            "1min bars on the anchor dates (unexpected — check the coverage ledger)."
        )
    read = pd.concat(frames, ignore_index=True)
    bars = normalize_intraday_bars(
        read, freq=RAW_INTRADAY_FREQ, data_lag=cfg.intraday.data_lag
    )
    logger.info(
        "intraday minute load (anchor-sliced): %d anchor dates, %d covered symbols, "
        "%d raw rows, %d normalized rows, stk_mins_live_calls=0",
        len(anchor_dates), len(covered), raw_rows, len(read),
    )
    return _MinuteLoad(
        bars=bars,
        covered=covered,
        uncovered=uncovered,
        raw_rows=raw_rows,
        normalized_rows=int(len(read)),
        anchor_dates=len(anchor_dates),
        live_calls=0,
    )


# --------------------------------------------------------------------------- #
# Group assignment (single source of truth, shared by all N group engines)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _GroupAssignment:
    """Per-date bucket assignment + per-date diagnostics (immutable)."""

    by_date: dict[pd.Timestamp, dict[str, int]]
    per_date_rows: tuple[dict, ...]


def _build_group_assignment(
    score_series: pd.Series,
    universe,
    panel: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    covered: set[str],
    n_groups: int,
) -> _GroupAssignment:
    """Rank the MMP cross-section into N equal-count buckets on each rebalance date.

    Uses ONLY the daily PIT universe (``universe.tradable``) ∩ covered names ∩
    valid MMP scores — no forward return ever enters. Returns the per-date
    ``{symbol: group}`` map and the per-date diagnostics (scored count, group
    sizes, score distribution).
    """
    by_date: dict[pd.Timestamp, dict[str, int]] = {}
    rows: list[dict] = []
    for d in rebalance_dates:
        norm = pd.Timestamp(d).normalize()
        tradable = [s for s in universe.tradable(d, panel) if str(s) in covered]
        try:
            cross = score_series.xs(norm, level="date")
        except KeyError:
            cross = pd.Series(dtype=float)
        cross = cross.reindex([str(s) for s in tradable]).dropna()
        assignment = assign_quantile_buckets(cross, n_groups)
        by_date[norm] = assignment
        sizes = [0] * n_groups
        for g in assignment.values():
            sizes[g - 1] += 1
        if len(cross) > 0:
            arr = cross.to_numpy(dtype=float)
            stats = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }
        else:
            stats = {k: float("nan") for k in ("mean", "std", "p10", "p50", "p90")}
        rows.append(
            {"date": norm, "n_scored": int(len(cross)), "sizes": tuple(sizes), **stats}
        )
    return _GroupAssignment(by_date=by_date, per_date_rows=tuple(rows))


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroupRunResult:
    """One quantile group's settled run + diagnostics (immutable)."""

    group: int
    nav_table: pd.DataFrame
    holdings_log: pd.DataFrame
    metrics: dict[str, float]
    up_limit_blocked_buys: int
    down_limit_blocked_sells: int
    missing_limit_rows: int
    # The only set on which the executed-price gate and a close-based gate differ:
    # the minute closed at a limit but traded through it, so the fill stands.
    opened_limit_up_minutes: int
    opened_limit_down_minutes: int
    # Holding-period returns dropped for want of a usable adj_factor at an anchor;
    # never defaulted to 1.0, so a non-zero count must reach the reader.
    missing_adj_factor_pairs: int
    blocked_fill_reasons: dict[str, int]


@dataclass(frozen=True)
class I5dResult:
    """Immutable summary of the I5d grouped backtest run."""

    config: RootConfig
    n_groups: int
    score_feature: str
    score_feature_key: str
    requested_symbols: int
    covered_symbols: int
    uncovered_symbols: tuple[str, ...]
    anchor_dates: int
    raw_rows: int
    normalized_rows: int
    minute_live_calls: int
    rebalance_count: int
    groups: tuple[GroupRunResult, ...]
    spread_per_period: pd.Series
    spread_cumulative: pd.Series
    spread_summary: dict[str, float]
    monotonicity: dict[str, float]
    per_date_rows: tuple[dict, ...]
    score_coverage: dict[str, int]
    price_limit_check: bool
    limit_coverage: dict[str, int]
    stk_limit_gap_fetches: int
    figure_paths: dict[str, Path]
    report_path: Path
    log_path: Path
    elapsed: float


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _report_basename(cfg: RootConfig) -> str:
    return cfg.output.intraday_report_name or _DEFAULT_REPORT_NAME


def _run_one_group(
    group: int,
    assignment: _GroupAssignment,
    *,
    cfg: RootConfig,
    panel: pd.DataFrame,
    bars: pd.DataFrame,
    universe,
    exec_cfg: IntradayExecutionConfig,
    price_limits: pd.DataFrame | None,
    shared_prices: tuple[pd.DataFrame, list] | None,
    constructor: PortfolioConstructor,
) -> tuple[GroupRunResult, tuple[pd.DataFrame, list]]:
    """Run ONE quantile group through a fresh engine/model/execution.

    A FRESH :class:`IntradayTailEventModel` and ``SimExecution`` per group (no
    shared mutable state); the immutable execution-price matrix is built by the
    first group and reused by the rest (``shared_prices``). Returns the group
    result and the shared prices (so the caller can thread them to the next group).
    """
    ic = cfg.intraday
    assert ic is not None
    model = IntradayTailEventModel(
        calendar_panel=panel,
        bars=bars,
        cfg=exec_cfg,
        price_limits=price_limits,
        price_limit_check=ic.price_limit_check,
        limit_tolerance=ic.limit_tolerance,
        require_price_limit_coverage=ic.require_price_limit_coverage,
        precomputed_prices=shared_prices,
    )
    if shared_prices is None:
        # First group builds the matrix; reuse it (immutable) for the rest.
        shared_prices = (model.execution_prices(), model.fills())

    execution = SimExecution(fee_rate=cfg.cost.fee_rate)
    engine = BacktestEngine(
        model=model,
        universe=universe,
        scores=GroupScores(assignment.by_date, group),
        constructor=constructor,
        execution=execution,
        selection_panel=panel,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    nav = engine.run()
    holdings = engine.holdings_log()
    blocked: dict[str, int] = {}
    for f in model.blocked_fills():
        key = f.reason or "unknown"
        blocked[key] = blocked.get(key, 0) + 1
    result = GroupRunResult(
        group=group,
        nav_table=nav,
        holdings_log=holdings,
        metrics=_group_metrics(nav, holdings),
        up_limit_blocked_buys=model.up_limit_blocked_buys(),
        down_limit_blocked_sells=model.down_limit_blocked_sells(),
        missing_limit_rows=model.missing_limit_rows(),
        opened_limit_up_minutes=model.opened_limit_up_minutes(),
        opened_limit_down_minutes=model.opened_limit_down_minutes(),
        missing_adj_factor_pairs=model.missing_adj_factor_pairs(),
        blocked_fill_reasons=blocked,
    )
    return result, shared_prices


def _spread_and_monotonicity(
    groups: tuple[GroupRunResult, ...], n_groups: int
) -> tuple[pd.Series, pd.Series, dict[str, float], dict[str, float]]:
    """Synthetic QN−Q1 spread series + group monotonicity diagnostics (report-only).

    The spread is the per-period difference of the high-group and low-group NET
    returns and its CUMULATIVE SUM — a synthetic long-only leg difference, NOT a
    separately executed dollar-neutral portfolio. Monotonicity is the Spearman
    rank correlation of the group index (1..N) vs the group annual return / final
    NAV.
    """
    by_group = {g.group: g for g in groups}
    low, high = by_group.get(1), by_group.get(n_groups)
    if low is None or high is None or low.nav_table.empty or high.nav_table.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, {}, {}
    aligned = pd.concat(
        [high.nav_table["net_return"].rename("high"),
         low.nav_table["net_return"].rename("low")],
        axis=1,
    ).dropna()
    per_period = (aligned["high"] - aligned["low"]).rename("spread")
    cumulative = per_period.cumsum().rename("cum_spread")
    summary = {
        "mean_per_period": float(per_period.mean()) if len(per_period) else float("nan"),
        "total": float(cumulative.iloc[-1]) if len(cumulative) else float("nan"),
        "n_periods": int(len(per_period)),
    }
    idx = pd.Series(
        [g for g in sorted(by_group)], dtype=float, index=sorted(by_group)
    )
    annual = pd.Series(
        {g: by_group[g].metrics["annual_return"] for g in sorted(by_group)}
    )
    final_nav = pd.Series(
        {g: by_group[g].metrics["final_nav"] for g in sorted(by_group)}
    )
    mono = {
        "annual_spearman": float(idx.corr(annual, method="spearman")),
        "final_nav_spearman": float(idx.corr(final_nav, method="spearman")),
    }
    return per_period, cumulative, summary, mono


def run_phase_i5d_intraday_groups(config_path: str) -> I5dResult:
    """Run the I5d MMP quintile grouped intraday-tail backtest and write its report."""
    cfg = load_config(config_path)
    _check_i5a_preconditions(cfg)  # real tushare, intraday enabled, event order, cache
    ic = cfg.intraday
    assert ic is not None
    if ic.score_feature != "mmp_ew":
        raise ValueError(
            "run-phase-i5d-intraday-groups is the MMP quintile study and requires "
            f"intraday.score_feature='mmp_ew' (got {ic.score_feature!r})."
        )
    n_groups = int(cfg.analytics.quantiles)
    if n_groups < 2:
        raise ValueError(
            f"analytics.quantiles must be >= 2 for a grouped backtest; got {n_groups}."
        )

    basename = _report_basename(cfg)
    log_dir = Path(cfg.output.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{basename}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)  # daily qfq selection panel + calendar

    # Monthly anchors from the daily calendar; minute bars only on those days.
    pairs = monthly_anchor_pairs(trading_calendar(panel))
    rebalance_dates = [pd.Timestamp(p[0]).normalize() for p in pairs]
    anchor_dates = sorted({pd.Timestamp(d).normalize() for pair in pairs for d in pair})
    logger.info(
        "schedule: %d monthly rebalances, %d anchor dates", len(pairs), len(anchor_dates)
    )

    load = _load_anchor_minute_bars(cfg, symbols, anchor_dates, logger)
    score_series, score_feature = _score_panel(cfg, load.bars, logger)
    score_coverage = {
        "rows": int(score_series.shape[0]),
        "valid": int(score_series.notna().sum()),
        "nan": int(score_series.isna().sum()),
    }

    price_limits, stk_limit_gap_fetches = _load_price_limits(
        cfg, load.covered, cache, logger
    )
    exec_cfg = _exec_cfg_from(cfg)

    assignment = _build_group_assignment(
        score_series, universe, panel, rebalance_dates, set(load.covered), n_groups
    )

    constructor = EqualWeightAll(long_only=cfg.portfolio.long_only)
    group_results: list[GroupRunResult] = []
    shared_prices: tuple[pd.DataFrame, list] | None = None
    limit_coverage: dict[str, int] = {}
    for group in range(1, n_groups + 1):
        result, shared_prices = _run_one_group(
            group,
            assignment,
            cfg=cfg,
            panel=panel,
            bars=load.bars,
            universe=universe,
            exec_cfg=exec_cfg,
            price_limits=price_limits,
            shared_prices=shared_prices,
            constructor=constructor,
        )
        group_results.append(result)
        logger.info(
            "group Q%d: periods=%d final_nav=%.6f up_blocked=%d down_blocked=%d",
            group, result.metrics["n_periods"], result.metrics["final_nav"],
            result.up_limit_blocked_buys, result.down_limit_blocked_sells,
        )
    # limit coverage is identical across groups (derived from the shared prices);
    # read it once from a fresh model so the diagnostic is reported a single time.
    if ic.price_limit_check:
        cov_model = IntradayTailEventModel(
            calendar_panel=panel, bars=load.bars, cfg=exec_cfg,
            price_limits=price_limits, price_limit_check=True,
            limit_tolerance=ic.limit_tolerance,
            require_price_limit_coverage=ic.require_price_limit_coverage,
            precomputed_prices=shared_prices,
        )
        limit_coverage = cov_model.limit_coverage()

    per_period, cumulative, spread_summary, monotonicity = _spread_and_monotonicity(
        tuple(group_results), n_groups
    )

    report_path = Path(cfg.output.report_dir) / f"{basename}.md"
    figure_dir = Path(cfg.output.report_dir) / f"{basename}_figures"
    result = I5dResult(
        config=cfg,
        n_groups=n_groups,
        score_feature=score_feature,
        score_feature_key=ic.score_feature,
        requested_symbols=len(symbols),
        covered_symbols=len(load.covered),
        uncovered_symbols=tuple(load.uncovered),
        anchor_dates=load.anchor_dates,
        raw_rows=load.raw_rows,
        normalized_rows=load.normalized_rows,
        minute_live_calls=load.live_calls,
        rebalance_count=len(pairs),
        groups=tuple(group_results),
        spread_per_period=per_period,
        spread_cumulative=cumulative,
        spread_summary=spread_summary,
        monotonicity=monotonicity,
        per_date_rows=assignment.per_date_rows,
        score_coverage=score_coverage,
        price_limit_check=ic.price_limit_check,
        limit_coverage=limit_coverage,
        stk_limit_gap_fetches=stk_limit_gap_fetches,
        figure_paths=_write_figures(figure_dir, tuple(group_results), per_period,
                                    cumulative, n_groups),
        report_path=report_path,
        log_path=log_path,
        elapsed=time.monotonic() - started,
    )
    _write_report(result)
    logger.info("report: %s", report_path)
    return result


# Report + figure rendering live in a sibling module for cohesion.
from qt.intraday_group_report import _write_figures, _write_report  # noqa: E402
