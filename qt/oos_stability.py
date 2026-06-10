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
    _build_factors,
    _build_scores,
    _build_universe,
    _collect_downgrades,
    _compute_factor_panel,
    _load_panel,
    _make_logger,
    _maybe_enrich_covariates,
    _maybe_enrich_financials,
    _maybe_enrich_listing,
    _periods_per_year,
    _process_factors,
)
from qt.reports import render_oos_stability, write_oos_stability_summary
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution

__all__ = ["OOSResult", "run_phase3_oos", "render_oos_stability",
           "subperiod_perf", "ic_period_stats", "sign_consistent",
           "weight_sign_flips", "fallback_reason_counts"]

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
def subperiod_perf(
    nav_table: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    periods_per_year: int = 12,
) -> dict:
    """Performance over the nav rows in [start, end) — period-local compounding.

    The subperiod nav is REBASED to 1.0 from the slice's own net returns, so a
    drawdown or annualization never bleeds across the split. ``end`` is
    exclusive (a rebalance ON the split date belongs to the test period).
    Empty slices return NaN metrics (never crash).
    """
    if nav_table is None or nav_table.empty:
        return dict(_PERF_NAN)
    sliced = nav_table
    if start is not None:
        sliced = sliced[sliced.index >= pd.Timestamp(start)]
    if end is not None:
        sliced = sliced[sliced.index < pd.Timestamp(end)]
    if sliced.empty:
        return dict(_PERF_NAN)
    local_nav = (1.0 + sliced["net_return"]).cumprod()
    perf = performance_summary(local_nav, periods_per_year=periods_per_year)
    return {
        **perf,
        "avg_turnover": float(sliced["turnover"].mean()),
        "n_rebalances": int(len(sliced)),
    }


def ic_period_stats(ic: pd.Series, split: pd.Timestamp) -> dict[str, dict]:
    """{'train': {...}, 'test': {...}} stats of a per-date IC series.

    Each period reports the NaN-dropped mean, IR (mean/std, ddof=1), hit rate
    (share of positive ICs) and observation count. Train is strictly before the
    split; test is on/after it.
    """
    split = pd.Timestamp(split)
    out: dict[str, dict] = {}
    for name, mask in (("train", ic.index < split), ("test", ic.index >= split)):
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
        "the split is NOT used (that would be a new alpha mode, out of scope).",
        "This is a SMALL-SAMPLE stability check (one index, two years, ~22 "
        "rebalances), NOT a return claim and NOT a tuned result: subperiod "
        "metrics carry wide uncertainty and must not be read as expected "
        "performance.",
    )
    return _collect_downgrades(cfg) + extra


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_backtest_for(cfg: RootConfig, panel, universe, score_panel) -> pd.DataFrame:
    """One backtest run for a given score panel (same rules for both models)."""
    driver = BacktestDriver(
        universe=universe,
        scores=_FrameScores(score_panel),
        constructor=TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only),
        execution=SimExecution(fee_rate=cfg.cost.fee_rate),
        prices=panel,
        rebalance=cfg.backtest.rebalance,
        fee_rate=cfg.cost.fee_rate,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    return driver.run()


def run_phase3_oos(config_path: str) -> OOSResult:
    """Run the OOS stability validation and write the phase3 OOS report.

    Requires ``data.source='tushare'`` (a stability check on offline demo data
    would be a category error) and an ``oos`` config section (the split date).
    All writes land under the configured output dirs; no secret is echoed.
    """
    cfg = load_config(config_path)
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-phase3-oos is a REAL-data validation and requires "
            "data.source='tushare'; the demo path carries no PIT/ann_date meaning. "
            f"Got data.source={cfg.data.source!r}."
        )
    if cfg.oos is None:
        raise ValueError(
            "run-phase3-oos requires an 'oos' config section with split_date "
            "(train = [data.start, split), test = [split, data.end])."
        )
    split = pd.Timestamp(cfg.oos.split_date)

    t0 = time.perf_counter()
    log_path = Path(cfg.output.log_dir) / "run_phase3_oos.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    logger.info("phase3 oos start: project=%s split=%s", cfg.project.name, split.date())

    # --- ONE shared data load / factor panel (identical inputs for both) ----- #
    universe, symbols = _build_universe(cfg, logger)
    panel = _load_panel(cfg, symbols, logger)
    factors = _build_factors(cfg)
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger)
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

    # --- subperiod statistics ------------------------------------------------ #
    ppy = _periods_per_year(cfg.backtest.rebalance)
    performance = {
        "equal_weight": {
            "train": subperiod_perf(nav_eq, end=split, periods_per_year=ppy),
            "test": subperiod_perf(nav_eq, start=split, periods_per_year=ppy),
        },
        "ic_weighted": {
            "train": subperiod_perf(nav_ic, end=split, periods_per_year=ppy),
            "test": subperiod_perf(nav_ic, start=split, periods_per_year=ppy),
        },
    }
    ic_series: dict[str, pd.Series] = {
        name: compute_ic(factor_panel[name], fwd_col) for name in factor_panel.columns
    }
    ic_series["combo_equal_weight"] = compute_ic(eq_scores, fwd_col)
    ic_series["combo_ic_weighted"] = compute_ic(ic_scores, fwd_col)
    ic_stats = {name: ic_period_stats(s, split) for name, s in ic_series.items()}
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
    write_oos_stability_summary(result)
    logger.info(
        "phase3 oos done: eq test annual=%.4f ic test annual=%.4f report=%s (%.1fs)",
        performance["equal_weight"]["test"].get("annual_return", float("nan")),
        performance["ic_weighted"]["test"].get("annual_return", float("nan")),
        result.report_path, result.elapsed_seconds,
    )
    return result
