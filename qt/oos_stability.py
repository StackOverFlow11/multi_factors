"""Phase 3-3: out-of-sample stability validation (equal_weight vs ic_weighted).

A REPORT-ONLY validation layer, not a new strategy: on ONE shared data load
(same universe, same factor panel, same processing / portfolio / execution
rules) it runs the backtest twice — once under :class:`EqualWeightAlpha`, once
under :class:`RollingICWeightAlpha` — and splits every diagnostic at
``oos.split_date``:

    train = [data.start, split)       test = [split, data.end]

Evaluation is WALK-FORWARD (rolling subperiod): the ic-weighted model derives
each date's weights from observations realized by that date only (t + h <= d,
P3-2, locked by tests), so no test-period forward return can reach any
train-period computation — the split here is an ACCOUNTING boundary for the
statistics, not a new training mode. Nothing in portfolio / execution / factor
math changes.

The report (``phase3_oos_stability.md``) carries: the exact split dates,
per-subperiod performance for both models, per-series IC stability (mean / IR /
hit rate / cross-period sign consistency) for every raw factor and both combo
scores, the ic-weighted weight-stability diagnostics (per-rebalance weights,
sign flips, fallback counts + reasons), and the explicit small-sample caveat.
Guard: requires the real tushare source and an ``oos`` config section.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha.equal_weight import EqualWeightAlpha
from analytics.factor import compute_ic, forward_returns
from analytics.performance import performance_summary
from portfolio.construct import TopNEqualWeight
from qt.config import RootConfig, load_config
from qt.pipeline import (
    _FrameScores,
    _alpha_disclosure,
    _build_alpha,
    _build_cache,
    _build_factors,
    _build_scores,
    _build_universe,
    _collect_downgrades,
    _compute_factor_panel,
    _load_panel,
    _log_run_cache_stats,
    _make_logger,
    _maybe_enrich_covariates,
    _maybe_enrich_financials,
    _maybe_enrich_listing,
    _maybe_enrich_value,
    _periods_per_year,
    _process_factors,
)
from qt.reports import render_oos_stability, write_oos_stability_summary
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution

__all__ = ["OOSResult", "run_phase3_oos", "render_oos_stability",
           "split_nav_by_holding", "subperiod_perf", "ic_period_stats",
           "sign_consistent", "weight_sign_flips", "fallback_reason_counts"]

_LOGGER_NAME = "qt.run_phase3_oos"
_PERF_NAN = {
    "annual_return": float("nan"), "volatility": float("nan"),
    "sharpe": float("nan"), "max_drawdown": float("nan"),
    "avg_turnover": float("nan"), "n_rebalances": 0,
}


@dataclass(frozen=True)
class OOSResult:
    """Immutable summary of one OOS stability run (what the report consumes)."""

    config: RootConfig
    elapsed_seconds: float
    # split boundaries (actual panel dates, disclosed verbatim in the report)
    split_date: pd.Timestamp
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None
    n_train_days: int
    n_test_days: int
    # rebalance dates whose HOLDING WINDOW straddles the split (excluded from
    # both subperiods' performance and disclosed in the report).
    boundary_dates: tuple[pd.Timestamp, ...]
    factor_names: tuple[str, ...]
    # performance[model][period] -> {annual_return, volatility, sharpe,
    # max_drawdown, avg_turnover, n_rebalances}; model in (equal_weight,
    # ic_weighted); period in (train, test).
    performance: dict[str, dict[str, dict]]
    # ic_stats[series][period] -> {ic_mean, ic_ir, hit_rate, n}; series = every
    # raw factor + combo_equal_weight + combo_ic_weighted.
    ic_stats: dict[str, dict[str, dict]]
    # sign_consistency[series] -> True iff train/test mean ICs share a nonzero sign.
    sign_consistency: dict[str, bool]
    # ic_weighted weight stability
    weights_at_rebalances: pd.DataFrame
    sign_flips: dict[str, int]
    n_scored: int
    n_fallback: int
    fallback_reasons: dict[str, int]
    alpha_summary: dict
    downgrades: tuple[str, ...]
    report_path: Path
    log_path: Path


# --------------------------------------------------------------------------- #
# Pure subperiod statistics (network-free; unit-tested with synthetic inputs).
# --------------------------------------------------------------------------- #
def split_nav_by_holding(
    nav_table: pd.DataFrame, split: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    """(train_rows, test_rows, boundary_dates) sliced by the HOLDING WINDOW.

    A driver nav row is INDEXED by its rebalance (signal) date, but its return
    covers the holding window [index[i], index[i+1]] — slicing by the row index
    alone would credit a straddling holding period's post-split return to the
    train period. Calendar-pure subperiods therefore require:

      train    <=> holding END on/before the split (return fully realized
                   pre-split);
      test     <=> holding START on/after the split;
      boundary <=> the holding window straddles the split (start < split < end,
                   or the end is unknown — the LAST row's end is the skipped
                   terminal candidate, not recoverable here — with a pre-split
                   start). Boundary rows are EXCLUDED from both subperiods and
                   disclosed in the report.
    """
    if nav_table is None or nav_table.empty:
        empty = nav_table if nav_table is not None else pd.DataFrame()
        return empty, empty, []
    split = pd.Timestamp(split)
    starts = [pd.Timestamp(d) for d in nav_table.index]
    ends: list[pd.Timestamp | None] = starts[1:] + [None]
    train_mask: list[bool] = []
    test_mask: list[bool] = []
    boundary: list[pd.Timestamp] = []
    for start, end in zip(starts, ends):
        if start >= split:
            train_mask.append(False)
            test_mask.append(True)
        elif end is not None and end <= split:
            train_mask.append(True)
            test_mask.append(False)
        else:  # straddles the split, or unknown end with a pre-split start
            train_mask.append(False)
            test_mask.append(False)
            boundary.append(start)
    return nav_table[train_mask], nav_table[test_mask], boundary


def subperiod_perf(nav_slice: pd.DataFrame, periods_per_year: int = 12) -> dict:
    """Performance of one pre-sliced nav segment — period-local compounding.

    The segment nav is REBASED to 1.0 from its own net returns, so a drawdown
    or annualization never bleeds across the split. Empty slices return NaN
    metrics (never crash). Slicing itself lives in
    :func:`split_nav_by_holding` (holding-window aware).
    """
    if nav_slice is None or nav_slice.empty:
        return dict(_PERF_NAN)
    local_nav = (1.0 + nav_slice["net_return"]).cumprod()
    perf = performance_summary(local_nav, periods_per_year=periods_per_year)
    return {
        **perf,
        "avg_turnover": float(nav_slice["turnover"].mean()),
        "n_rebalances": int(len(nav_slice)),
    }


def ic_period_stats(
    ic: pd.Series, split: pd.Timestamp, horizon: int = 1
) -> dict[str, dict]:
    """{'train': {...}, 'test': {...}} stats of a per-date IC series.

    The IC at factor date t uses the h-day forward return REALIZED at trading
    position pos(t) + h, so (mirroring the nav slicing):
      train <=> realization date strictly BEFORE the split;
      test  <=> factor date on/after the split;
      straddlers (t < split <= realization) are excluded from both.
    Each period reports the NaN-dropped mean, IR (mean/std, ddof=1), hit rate
    (share of positive ICs) and observation count.
    """
    split = pd.Timestamp(split)
    idx = list(ic.index)
    n_idx = len(idx)
    realization = [
        idx[pos + horizon] if pos + horizon < n_idx else None
        for pos in range(n_idx)
    ]
    train_mask = [
        r is not None and pd.Timestamp(r) < split for r in realization
    ]
    test_mask = [pd.Timestamp(t) >= split for t in idx]
    out: dict[str, dict] = {}
    for name, mask in (("train", train_mask), ("test", test_mask)):
        clean = ic[mask].dropna()
        n = int(len(clean))
        mean = float(clean.mean()) if n else float("nan")
        std = float(clean.std(ddof=1)) if n > 1 else float("nan")
        ir = mean / std if (math.isfinite(std) and std != 0) else float("nan")
        hit = float((clean > 0).mean()) if n else float("nan")
        out[name] = {"ic_mean": mean, "ic_ir": float(ir), "hit_rate": hit, "n": n}
    return out


def sign_consistent(stats: dict[str, dict]) -> bool:
    """True iff the train and test mean ICs share the SAME nonzero finite sign."""
    tr = float(stats.get("train", {}).get("ic_mean", float("nan")))
    te = float(stats.get("test", {}).get("ic_mean", float("nan")))
    if not (math.isfinite(tr) and math.isfinite(te)) or tr == 0.0 or te == 0.0:
        return False
    return (tr > 0) == (te > 0)


def weight_sign_flips(weights_log: pd.DataFrame) -> dict[str, int]:
    """Per-factor count of sign changes between CONSECUTIVE TRAINED rows.

    Fallback rows carry all-positive equal weights, so including them would
    inject artificial flips — they are excluded here and disclosed separately
    (fallback count / reasons).
    """
    if weights_log is None or weights_log.empty:
        return {}
    trained = weights_log.loc[~weights_log["fallback"].astype(bool)]
    factor_cols = [c for c in weights_log.columns if c != "fallback"]
    flips: dict[str, int] = {}
    for col in factor_cols:
        signs = trained[col].apply(
            lambda v: 0 if (pd.isna(v) or v == 0) else (1 if v > 0 else -1)
        )
        signs = signs[signs != 0]
        flips[col] = int((signs.diff().fillna(0) != 0).sum())
    return flips


def fallback_reason_counts(fallback_log: dict) -> dict[str, int]:
    """Aggregate the per-date fallback reasons into reason -> count (None = trained)."""
    counts: dict[str, int] = {}
    for reason in fallback_log.values():
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _oos_downgrades(cfg: RootConfig) -> tuple[str, ...]:
    """Base downgrades + the OOS-validation caveats (INV-007)."""
    extra = (
        "OOS VALIDATION SEMANTICS: this run evaluates BOTH equal_weight and "
        "ic_weighted on one shared data load; the split is an ACCOUNTING boundary "
        "for subperiod statistics. Weight training is walk-forward (a pair enters "
        "date d's weights only once realized, t + horizon <= d), so no test-period "
        "forward return reaches any train-period computation; freezing weights at "
        "the split is NOT used (that would be a new alpha mode, out of scope). "
        "Subperiod PERFORMANCE is sliced by the HOLDING WINDOW (train rows end "
        "on/before the split, test rows start on/after it; a straddling rebalance "
        "is excluded from both and disclosed) and IC stats by the realization "
        "date — never by the row's signal date alone.",
        "This is a SMALL-SAMPLE stability check — each run/cell covers ONE index "
        "universe over ONE window (~22 rebalances per train+test pair) — NOT a "
        "return claim and NOT a tuned result: subperiod metrics carry wide "
        "uncertainty and must not be read as expected performance. (A robustness "
        "matrix spans multiple universes × windows; see its MATRIX SCOPE line.)",
    )
    return _collect_downgrades(cfg) + extra


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_backtest_for(
    cfg: RootConfig, panel, universe, score_panel, fee_rate: float | None = None
) -> pd.DataFrame:
    """One backtest run for a given score panel (same rules for both models).

    ``fee_rate=None`` (the default, and the only value the OOS/robustness paths
    pass) uses ``cfg.cost.fee_rate`` — behaviour-preserving, locked by a test.
    The P3-6 cost scenarios pass an explicit scaled fee; scores/fills never see
    the fee, so a scenario changes ONLY the cost line, never the trades.
    """
    fee = cfg.cost.fee_rate if fee_rate is None else float(fee_rate)
    driver = BacktestDriver(
        universe=universe,
        scores=_FrameScores(score_panel),
        constructor=TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only),
        execution=SimExecution(fee_rate=fee),
        prices=panel,
        rebalance=cfg.backtest.rebalance,
        fee_rate=fee,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    return driver.run()


def check_oos_preconditions(cfg: RootConfig, runner: str = "run-phase3-oos") -> None:
    """Shared guards for the OOS cell computation (single run AND matrix, P3-4).

    Real tushare source only; an ``oos`` section (the split date); and the
    alpha section MUST be ic_weighted — anything else would silently run
    equal_weight twice and label one leg ic_weighted (a fake comparison).
    """
    if cfg.data.source != "tushare":
        raise ValueError(
            f"{runner} is a REAL-data validation and requires "
            "data.source='tushare'; the demo path carries no PIT/ann_date meaning. "
            f"Got data.source={cfg.data.source!r}."
        )
    if cfg.oos is None:
        raise ValueError(
            f"{runner} requires an 'oos' config section with split_date "
            "(train = [data.start, split), test = [split, data.end])."
        )
    if cfg.alpha.model != "ic_weighted":
        raise ValueError(
            f"{runner} compares equal_weight vs ic_weighted: the config's "
            "alpha section must set model='ic_weighted' (it carries the "
            "ic-weighted leg's params; the equal-weight control is built "
            f"internally). Got alpha.model={cfg.alpha.model!r} — running that "
            "would silently label an equal_weight leg as ic_weighted."
        )


def run_phase3_oos(config_path: str) -> OOSResult:
    """Run the OOS stability validation and write the phase3 OOS report.

    Requires ``data.source='tushare'`` (a stability check on offline demo data
    would be a category error) and an ``oos`` config section (the split date).
    All writes land under the configured output dirs; no secret is echoed.
    """
    cfg = load_config(config_path)
    check_oos_preconditions(cfg, runner="run-phase3-oos")
    log_path = Path(cfg.output.log_dir) / "run_phase3_oos.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    result = _run_oos_cell(cfg, logger, log_path)
    write_oos_stability_summary(result)
    logger.info(
        "phase3 oos done: eq test annual=%.4f ic test annual=%.4f report=%s (%.1fs)",
        result.performance["equal_weight"]["test"].get("annual_return", float("nan")),
        result.performance["ic_weighted"]["test"].get("annual_return", float("nan")),
        result.report_path, result.elapsed_seconds,
    )
    return result


def _run_oos_cell(
    cfg: RootConfig, logger, log_path: Path
) -> OOSResult:
    """Compute ONE OOS cell (the P3-3 core; no report write).

    Shared verbatim by the single run (:func:`run_phase3_oos`) and the P3-4
    robustness matrix, so every matrix cell gets the same holding-window
    slicing, realization-date IC slicing and walk-forward semantics. The
    caller is responsible for guards (:func:`check_oos_preconditions`) and for
    writing whatever report it owns.
    """
    split = pd.Timestamp(cfg.oos.split_date)
    t0 = time.perf_counter()
    logger.info("phase3 oos start: project=%s split=%s", cfg.project.name, split.date())

    # --- ONE shared data load / factor panel (identical inputs for both) ----- #
    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    factors = _build_factors(cfg)
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_value(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)
    factor_panel = _compute_factor_panel(cfg, panel, factors, logger)
    processed = _process_factors(cfg, factor_panel, panel)

    horizon = int(cfg.analytics.forward_return_periods[0])
    fwd_col = forward_returns(panel, periods=(horizon,))[f"forward_return_{horizon}d"]

    # --- both alphas on the SAME processed panel ----------------------------- #
    eq_alpha = EqualWeightAlpha()
    eq_scores = _build_scores(processed, eq_alpha)
    ic_alpha = _build_alpha(cfg)  # config carries the ic_weighted params
    ic_scores = _build_scores(processed, ic_alpha, fwd_col)
    alpha_summary, weights_log = _alpha_disclosure(cfg, ic_alpha)
    logger.info("alpha (ic_weighted leg): %s", alpha_summary)

    nav_eq = _run_backtest_for(cfg, panel, universe, eq_scores)
    nav_ic = _run_backtest_for(cfg, panel, universe, ic_scores)
    logger.info("backtests: equal_weight %d rows, ic_weighted %d rows",
                len(nav_eq), len(nav_ic))

    # --- subperiod statistics (holding-window aware slicing) ------------------ #
    ppy = _periods_per_year(cfg.backtest.rebalance)
    eq_train, eq_test, eq_boundary = split_nav_by_holding(nav_eq, split)
    ic_train, ic_test, ic_boundary = split_nav_by_holding(nav_ic, split)
    # both legs share the driver's rebalance calendar -> identical boundary rows
    boundary_dates = tuple(sorted(set(eq_boundary) | set(ic_boundary)))
    performance = {
        "equal_weight": {
            "train": subperiod_perf(eq_train, periods_per_year=ppy),
            "test": subperiod_perf(eq_test, periods_per_year=ppy),
        },
        "ic_weighted": {
            "train": subperiod_perf(ic_train, periods_per_year=ppy),
            "test": subperiod_perf(ic_test, periods_per_year=ppy),
        },
    }
    ic_series: dict[str, pd.Series] = {
        name: compute_ic(factor_panel[name], fwd_col) for name in factor_panel.columns
    }
    ic_series["combo_equal_weight"] = compute_ic(eq_scores, fwd_col)
    ic_series["combo_ic_weighted"] = compute_ic(ic_scores, fwd_col)
    ic_stats = {
        name: ic_period_stats(s, split, horizon=horizon)
        for name, s in ic_series.items()
    }
    consistency = {name: sign_consistent(stats) for name, stats in ic_stats.items()}

    # weight stability at the SETTLED rebalance dates of the ic run
    settled = list(nav_ic.index)
    if weights_log is not None and not weights_log.empty:
        wanted = [d for d in settled if d in weights_log.index]
        weights_at_reb = weights_log.loc[wanted]
        n_scored = int(len(weights_log))
        n_fallback = int(weights_log["fallback"].sum())
    else:  # pragma: no cover - ic leg always logs
        weights_at_reb = pd.DataFrame()
        n_scored = n_fallback = 0
    flips = weight_sign_flips(weights_at_reb)
    reasons = fallback_reason_counts(
        ic_alpha.fallback_log() if hasattr(ic_alpha, "fallback_log") else {}
    )

    dates = panel.index.get_level_values("date")
    train_dates = sorted({d for d in dates if d < split})
    test_dates = sorted({d for d in dates if d >= split})
    result = OOSResult(
        config=cfg,
        elapsed_seconds=time.perf_counter() - t0,
        split_date=split,
        train_start=train_dates[0] if train_dates else None,
        train_end=train_dates[-1] if train_dates else None,
        test_start=test_dates[0] if test_dates else None,
        test_end=test_dates[-1] if test_dates else None,
        n_train_days=len(train_dates),
        n_test_days=len(test_dates),
        boundary_dates=boundary_dates,
        factor_names=tuple(f.name for f in factors),
        performance=performance,
        ic_stats=ic_stats,
        sign_consistency=consistency,
        weights_at_rebalances=weights_at_reb,
        sign_flips=flips,
        n_scored=n_scored,
        n_fallback=n_fallback,
        fallback_reasons=reasons,
        alpha_summary=alpha_summary,
        downgrades=_oos_downgrades(cfg),
        report_path=Path(cfg.output.report_dir) / "phase3_oos_stability.md",
        log_path=log_path,
    )
    logger.info(
        "oos cell done: eq test annual=%.4f ic test annual=%.4f (%.1fs)",
        performance["equal_weight"]["test"].get("annual_return", float("nan")),
        performance["ic_weighted"]["test"].get("annual_return", float("nan")),
        result.elapsed_seconds,
    )
    return result
