"""Phase 2-1: a small-scale REAL-data (tushare) reproducibility baseline.

This is NOT a new strategy and adds NO new factor or parameter search. It runs
the *existing* P0/P1 spine on the real tushare path end-to-end over a small
universe + short window (designed to finish in ~10-30 min) and emits a richer
diagnostic report (``artifacts/reports/phase2_real_baseline.md``) so the real
path can be eyeballed and reproduced:

    data window · PIT membership summary · ann_date as-of coverage ·
    tradability filter hits · rebalance dates · per-period holdings ·
    turnover / cost · IC / quantile returns · performance · P2 downgrades.

It reuses :mod:`qt.pipeline`'s step helpers verbatim (same universe / panel /
factor / neutralization / backtest), so "baseline == the real pipeline".
Per-period holdings are the ACHIEVED book recorded by the driver during the run
(``BacktestDriver.holdings_log()``, post execution-feasibility); the other extra
reporting (filter hits, ann_date coverage) is read-only diagnostics. Nothing here
changes the strategy or sees forward returns at the factor stage.

Guard: the baseline REQUIRES ``data.source='tushare'``. Running it on the demo
source is a category error (no PIT / ann_date / tradability meaning) and raises
a readable error rather than silently producing a meaningless "real" report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha.equal_weight import EqualWeightAlpha
from analytics.performance import performance_summary
from factors.compute.financial import SUPPORTED_FIELDS as SUPPORTED_FINANCIAL_FIELDS
from factors.compute.financial import FinancialFactor
from portfolio.construct import TopNEqualWeight
from qt.config import RootConfig, load_config
from qt.pipeline import (
    _FrameScores,
    _build_factors,
    _build_scores,
    _build_universe,
    _collect_downgrades,
    _compute_factor_panel,
    _factor_analytics,
    _standard_analytics,
    _load_panel,
    _make_logger,
    _maybe_enrich_covariates,
    _maybe_enrich_financials,
    _maybe_enrich_listing,
    _periods_per_year,
    _process_factors,
)
from qt.reports import write_phase2_baseline_summary
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution
from universe.index_universe import PITIndexUniverse

_LOGGER_NAME = "qt.run_phase2_baseline"

# Tradability reasons reported (mirrors universe.filters.apply_tradable_filters).
_FILTER_REASONS = ("missing_close", "suspended", "is_st", "at_up_limit", "at_down_limit")


@dataclass(frozen=True)
class Phase2Result:
    """Immutable summary of one phase2 real-baseline run (what the report consumes)."""

    config: RootConfig
    elapsed_seconds: float
    # data window
    first_trade_date: pd.Timestamp | None
    last_trade_date: pd.Timestamp | None
    trade_days: int
    panel_rows: int
    panel_symbols: int
    # universe / PIT membership
    universe_summary: dict
    # min_listing_days list_date coverage (known vs disclosed data gap, this run)
    list_date_known: int
    list_date_total: int
    # PIT SW industry coverage for neutralization (NaN if neutralize off)
    industry_pit_coverage: float
    # ann_date financial coverage PER FIELD (P3-1):
    # financial_coverage[field] = {"is_factor": bool, "overall": float,
    #                              "by_rebalance": DataFrame}
    # is_factor distinguishes a TRADED financial factor from a pure diagnostic
    # (a no-financial-factor run still reports the default field as diagnostic).
    financial_coverage: dict[str, dict]
    # tradability filter hits
    tradability_hits: pd.DataFrame
    # execution feasibility (direction-aware fills)
    feasibility_log: pd.DataFrame
    # rebalance + holdings (rebalance_dates are the SETTLED dates == nav_table.index)
    rebalance_dates: tuple[pd.Timestamp, ...]
    candidate_rebalance_dates: tuple[pd.Timestamp, ...]
    skipped_terminal_dates: tuple[pd.Timestamp, ...]
    holdings: pd.DataFrame
    # turnover / cost / performance
    # primary factor = first enabled; factor_names = ALL enabled (P3-1).
    factor_name: str
    factor_names: tuple[str, ...]
    # per-factor + combo-score simple analytics (P3-1):
    # per_factor[name] = {ic_mean, ic_ir, quantile_returns, coverage};
    # combo_analytics  = {ic_mean, ic_ir, quantile_returns} on the traded score.
    per_factor: dict[str, dict]
    combo_analytics: dict
    nav_table: pd.DataFrame
    avg_turnover: float
    cost_drag: float
    ic_mean: float
    ic_ir: float
    quantile_returns: pd.DataFrame
    performance: dict
    # P2-4 standard-analytics cross-check (report-only; never alters trading)
    std_performance: dict
    std_factor: dict
    # disclosure
    downgrades: tuple[str, ...]
    # paths
    report_path: Path
    log_path: Path


# --------------------------------------------------------------------------- #
# Pure collectors (network-free; unit-tested with synthetic inputs).
# --------------------------------------------------------------------------- #
def summarize_universe(universe, universe_type: str, start, end) -> dict:
    """Summarize the (PIT or static) universe for the report.

    For a :class:`PITIndexUniverse` the feed deliberately loads snapshots from
    BEFORE ``start`` (a ~370-day pre-start lookback) so the as-of membership at the
    window start resolves correctly. This summary therefore separates two things:

      * LOADED snapshots — everything fetched, incl. the pre-start lookback (so the
        reader sees exactly what was pulled);
      * IN-WINDOW membership — the snapshots dated within ``[start, end]`` plus the
        as-of anchor active at ``start`` (the lookback snapshot the early dates
        resolve to). ``distinct_names_in_window`` / size / churn are reported over
        this in-window view, so "distinct names over window" means what the
        backtest actually saw, not names that only existed before the window.

    Static universes carry no point-in-time membership, recorded explicitly.
    """
    if isinstance(universe, PITIndexUniverse):
        snaps = universe.membership_snapshots()
        loaded = sorted(snaps)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        in_window = [d for d in loaded if start_ts <= d <= end_ts]
        prior = [d for d in loaded if d <= start_ts]
        anchor = max(prior) if prior else None
        # names seen during the window = anchor membership (held into the window)
        # ∪ every in-window snapshot's membership.
        active = ([anchor] if anchor is not None else []) + in_window
        sizes = [len(snaps[d]) for d in active]
        distinct = sorted({s for d in active for s in snaps[d]})
        churn_in: list[int] = []
        churn_out: list[int] = []
        # churn is reported over in-window transitions only (membership changes
        # that actually occur inside the backtest window).
        for prev, cur in zip(in_window, in_window[1:]):
            prev_set, cur_set = set(snaps[prev]), set(snaps[cur])
            churn_in.append(len(cur_set - prev_set))
            churn_out.append(len(prev_set - cur_set))
        return {
            "pit": True,
            "type": "index",
            "n_loaded_snapshots": len(loaded),
            "loaded_first": loaded[0] if loaded else None,
            "loaded_last": loaded[-1] if loaded else None,
            "n_window_snapshots": len(in_window),
            "anchor_snapshot": anchor,
            "distinct_names_in_window": len(distinct),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "avg_churn_in": (sum(churn_in) / len(churn_in)) if churn_in else 0.0,
            "avg_churn_out": (sum(churn_out) / len(churn_out)) if churn_out else 0.0,
        }
    return {
        "pit": False,
        "type": universe_type,
        "distinct_names_in_window": len(getattr(universe, "_symbols", []) or []),
    }


def tradability_hit_stats(
    universe,
    panel: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    filters: dict,
) -> pd.DataFrame:
    """Count, summed over rebalance dates, how many members each filter knocks out.

    For every rebalance date we take the universe members, look at that day's
    cross-section, and tally the reason a name is dropped. The reasons are
    EXCLUSIVE and use the SAME first-match-wins order as
    :func:`universe.filters.apply_tradable_filters` (missing close always; then
    suspended; then ST; then at-limit — each gated by its toggle and the flag
    column being present). A name flagged by several filters is counted once, in
    the first matching bucket, so ``sum(hits) == candidates - tradable`` exactly
    and the funnel is auditable. Returns a frame indexed by reason with a ``hits``
    column plus ``candidates`` / ``tradable`` totals in ``.attrs``.
    """
    counts = {r: 0 for r in _FILTER_REASONS}
    n_candidates = 0
    n_tradable = 0
    for date in rebalance_dates:
        target = pd.Timestamp(date).normalize()
        members = list(universe.members(target))
        try:
            cross = panel.xs(target, level="date")
        except KeyError:
            cross = panel.iloc[0:0]
        for sym in members:
            n_candidates += 1
            if sym not in cross.index:
                counts["missing_close"] += 1
                continue
            row = cross.loc[sym]
            if pd.isna(row["close"]):
                counts["missing_close"] += 1
                continue
            # First-match-wins, mirroring apply_tradable_filters (exclusive buckets).
            if filters.get("suspended") and bool(row.get("suspended", False)):
                counts["suspended"] += 1
                continue
            if filters.get("st") and bool(row.get("is_st", False)):
                counts["is_st"] += 1
                continue
            if filters.get("limit_up_down") and bool(row.get("at_up_limit", False)):
                counts["at_up_limit"] += 1
                continue
            if filters.get("limit_up_down") and bool(row.get("at_down_limit", False)):
                counts["at_down_limit"] += 1
                continue
            n_tradable += 1
    frame = pd.DataFrame(
        {"hits": [counts[r] for r in _FILTER_REASONS]},
        index=list(_FILTER_REASONS),
    )
    frame.index.name = "reason"
    frame.attrs["candidates"] = n_candidates
    frame.attrs["tradable"] = n_tradable
    return frame


def financial_coverage_at_dates(
    aligned_col: pd.Series,
    universe,
    panel: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Per-rebalance-date as-of financial coverage among that date's members.

    ``aligned_col`` is the ann_date as-of aligned financial series (NaN where no
    report had been disclosed by that date). The denominator (``n_members``) is the
    universe members PRESENT IN THE PANEL that date (taken from ``panel`` so the
    contract is explicit, not implied by the aligned series' index); of those, we
    count how many carry a non-NaN figure. Returns ``[date, n_members, n_covered,
    coverage]`` — a data-quality lens on how well the financial path is populated,
    NOT a strategy signal.
    """
    rows: list[dict] = []
    for date in rebalance_dates:
        target = pd.Timestamp(date).normalize()
        members = list(universe.members(target))
        try:
            cross_syms = set(panel.xs(target, level="date").index)
        except KeyError:
            cross_syms = set()
        n_members = 0
        n_covered = 0
        for sym in members:
            if str(sym) not in cross_syms:
                continue  # not present in the panel that day -> not a denominator member
            n_members += 1
            val = aligned_col.get((target, str(sym)), float("nan"))
            if pd.notna(val):
                n_covered += 1
        coverage = (n_covered / n_members) if n_members else float("nan")
        rows.append(
            {
                "date": target,
                "n_members": n_members,
                "n_covered": n_covered,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows, columns=["date", "n_members", "n_covered", "coverage"])


def _phase2_downgrades(
    cfg: RootConfig, financial_coverage: dict[str, dict]
) -> tuple[str, ...]:
    """Base downgrades + baseline-specific disclosures (per-field aware, P3-1)."""
    traded = sorted(f for f, info in financial_coverage.items() if info.get("is_factor"))
    diag_only = sorted(
        f for f, info in financial_coverage.items() if not info.get("is_factor")
    )
    if traded:
        fin_item = (
            f"Financial ann_date coverage is reported PER FIELD: {traded} are TRADED "
            "financial factors in this run (ann_date PIT-aligned, fetched in one "
            "pass); their coverage tables are a data-quality lens on the same "
            "columns the strategy consumes."
        )
    else:
        fin_item = (
            f"Financial ann_date coverage is a DATA-QUALITY DIAGNOSTIC on "
            f"{diag_only} (how well disclosed reports populate the universe "
            "as-of); it is NOT the alpha factor in this baseline (the factor is the "
            "configured price factor). No financial signal is traded here."
        )
    extra = (
        fin_item,
        "Per-period holdings are the ACHIEVED book recorded by the backtest driver "
        "(post execution-feasibility), NOT the constructor's desired target: a "
        "blocked sell shows the carried name and a blocked buy is absent.",
        "This is a SMALL-SCALE baseline (a small index over a short window) for "
        "reproducibility/plumbing validation, NOT a performance claim; numbers are "
        "not optimized and must not be read as a strategy result.",
    )
    return _collect_downgrades(cfg) + extra


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_phase2_baseline(config_path: str) -> Phase2Result:
    """Run the small-scale real-data baseline and write the phase2 report.

    Raises ``ValueError`` (readable) if pointed at the demo source — the baseline
    is meaningless without real PIT / ann_date / tradability data. All writes land
    under the configured ``output`` dirs (the report is git-ignored under
    ``artifacts/``; the tushare token is never read into the report).
    """
    cfg = load_config(config_path)
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-phase2-baseline is a REAL-data experiment and requires "
            "data.source='tushare'. The demo/offline path carries no PIT, "
            "ann_date, or tradability meaning, so a 'baseline' there would be a "
            f"category error. Got data.source={cfg.data.source!r}."
        )

    t0 = time.perf_counter()
    log_path = Path(cfg.output.log_dir) / "run_phase2_baseline.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    logger.info("phase2 baseline start: project=%s", cfg.project.name)

    # --- reuse the exact P0/P1 spine ------------------------------------- #
    universe, symbols = _build_universe(cfg, logger)
    panel = _load_panel(cfg, symbols, logger)
    factors = _build_factors(cfg)
    primary = factors[0]
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger)
    factor_panel = _compute_factor_panel(cfg, panel, factors, logger)
    processed = _process_factors(cfg, factor_panel, panel)
    score_panel = _build_scores(processed, EqualWeightAlpha())
    scores = _FrameScores(score_panel)

    constructor = TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only)
    execution = SimExecution(fee_rate=cfg.cost.fee_rate)
    driver = BacktestDriver(
        universe=universe,
        scores=scores,
        constructor=constructor,
        execution=execution,
        prices=panel,
        rebalance=cfg.backtest.rebalance,
        fee_rate=cfg.cost.fee_rate,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    candidate_dates = driver.rebalance_dates()
    nav_table = driver.run()
    feasibility_log = driver.feasibility_log()
    # The diagnostics (coverage / filter hits / holdings) MUST key off the dates
    # that were actually held + settled — i.e. nav_table.index — NOT the candidate
    # rebalance dates. The driver skips a terminal rebalance with no forward
    # holding period (BT-003), so the last candidate date has no NAV/turnover row;
    # reporting holdings for it would be a phantom period.
    settled_dates = list(nav_table.index)
    settled_set = set(settled_dates)
    skipped_dates = [d for d in candidate_dates if d not in settled_set]
    logger.info("backtest: %d candidate rebalance dates, %d settled rows (skipped %d)",
                len(candidate_dates), len(nav_table), len(skipped_dates))

    per_factor, combo = _factor_analytics(cfg, panel, factor_panel, score_panel)
    first = per_factor[primary.name]
    ic_mean, ic_ir, q_returns = first["ic_mean"], first["ic_ir"], first["quantile_returns"]
    perf = (
        performance_summary(
            nav_table["nav"], periods_per_year=_periods_per_year(cfg.backtest.rebalance)
        )
        if not nav_table.empty
        else {k: float("nan") for k in ("annual_return", "max_drawdown", "volatility", "sharpe")}
    )
    std_performance, std_factor = _standard_analytics(
        cfg, panel, factor_panel, primary.name, nav_table, ic_mean, ic_ir, perf, logger
    )

    # --- diagnostics (read-only) ----------------------------------------- #
    # ann_date coverage PER financial field (P3-1). Traded financial factors are
    # already on the panel (one batched fetch); a run with NO financial factor
    # still reports the default field as a pure diagnostic (extra fetch, not traded).
    factor_fields = [f.name for f in factors if isinstance(f, FinancialFactor)]
    if factor_fields:
        diag_fields = list(factor_fields)
        diag_panel = panel
    else:
        diag_fields = [SUPPORTED_FINANCIAL_FIELDS[0]]
        logger.info(
            "diagnostics: fetching '%s' for ann_date coverage report ONLY "
            "(NOT an active factor; the strategy factors are %s)",
            diag_fields[0], [f.name for f in factors],
        )
        diag_panel = _maybe_enrich_financials(
            cfg, panel, symbols, [FinancialFactor(diag_fields[0])], logger
        )
    financial_coverage: dict[str, dict] = {}
    for field in diag_fields:
        col = diag_panel[field]
        financial_coverage[field] = {
            "is_factor": field in factor_fields,
            "overall": float(col.notna().mean()) if len(col) else float("nan"),
            "by_rebalance": financial_coverage_at_dates(
                col, universe, panel, settled_dates
            ),
        }

    hits = tradability_hit_stats(
        universe, panel, settled_dates, cfg.universe.filters.model_dump()
    )
    # ACHIEVED holdings (post-feasibility) from the driver — the actual book held,
    # NOT the constructor's desired target (which differs once a trade is blocked).
    holdings = driver.holdings_log()

    # list_date coverage for the min_listing_days disclosure (how many names had a
    # known listing date this run vs a disclosed data gap).
    if "list_date" in panel.columns:
        _ld = panel["list_date"].groupby(level="symbol").first()
        list_date_known = int(_ld.notna().sum())
        list_date_total = int(len(_ld))
    else:
        list_date_known = 0
        list_date_total = 0

    # PIT SW industry coverage (fraction of panel rows with an as-of industry).
    if cfg.processing.neutralize.enabled and "industry" in panel.columns:
        industry_pit_coverage = float(panel["industry"].notna().mean())
    else:
        industry_pit_coverage = float("nan")

    dates = panel.index.get_level_values("date")
    result = Phase2Result(
        config=cfg,
        elapsed_seconds=time.perf_counter() - t0,
        first_trade_date=dates.min() if len(dates) else None,
        last_trade_date=dates.max() if len(dates) else None,
        trade_days=int(pd.Series(dates).nunique()),
        panel_rows=len(panel),
        panel_symbols=int(panel.index.get_level_values("symbol").nunique()),
        universe_summary=summarize_universe(
            universe, cfg.universe.type, cfg.data.start, cfg.data.end
        ),
        list_date_known=list_date_known,
        list_date_total=list_date_total,
        industry_pit_coverage=industry_pit_coverage,
        financial_coverage=financial_coverage,
        tradability_hits=hits,
        feasibility_log=feasibility_log,
        rebalance_dates=tuple(settled_dates),
        candidate_rebalance_dates=tuple(candidate_dates),
        skipped_terminal_dates=tuple(skipped_dates),
        holdings=holdings,
        factor_name=primary.name,
        factor_names=tuple(f.name for f in factors),
        per_factor=per_factor,
        combo_analytics=combo,
        nav_table=nav_table,
        avg_turnover=float(nav_table["turnover"].mean()) if not nav_table.empty else 0.0,
        cost_drag=float(nav_table["cost"].sum()) if not nav_table.empty else 0.0,
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        quantile_returns=q_returns,
        performance=perf,
        std_performance=std_performance,
        std_factor=std_factor,
        downgrades=_phase2_downgrades(cfg, financial_coverage),
        report_path=Path(cfg.output.report_dir)
        / (cfg.output.baseline_report_name or "phase2_real_baseline.md"),
        log_path=log_path,
    )
    write_phase2_baseline_summary(result)
    logger.info(
        "phase2 baseline done: ic_mean=%.4f annual_return=%.4f report=%s (%.1fs)",
        ic_mean, perf.get("annual_return", float("nan")), result.report_path,
        result.elapsed_seconds,
    )
    return result
