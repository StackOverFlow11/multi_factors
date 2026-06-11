"""Phase 3-6: value+lowvol subset re-check + cost sensitivity (EXPLORATORY).

A REPORT-ONLY comparison layer on top of the P3-4 robustness matrix: on every
(universe, window) cell it loads the data ONCE, computes the FULL raw factor
panel once (the same call sequence as the P3-3/P3-4 cell core, so raw-factor
ICs double as a no-drift cross-check against the P3-5 report), then compares
configured FACTOR GROUPS head-to-head:

  * each group is re-processed INDEPENDENTLY from the shared raw factor panel
    (``drop_missing`` applies to the group's own columns — exactly as if the
    group were the configured factor set, never the full pack's missing-mask);
  * each group runs the SAME equal_weight vs walk-forward ic_weighted
    comparison with the SAME holding-window OOS slicing as P3-3/P3-4;
  * each (group, model) backtest is repeated under every COST SCENARIO
    (``cost.fee_rate`` × multiplier). Scores and fills never see the fee, so a
    scenario changes ONLY the cost line — trades and turnover are identical
    across scenarios (locked by tests).

No new alpha model, no tuning, no change to portfolio / execution / factor
math; ``_run_oos_cell`` itself is untouched. HONESTY: the value+lowvol subset
was chosen AFTER seeing the P3-5 results on these SAME windows — the report
discloses this POST-HOC selection and is NOT a return claim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha.equal_weight import EqualWeightAlpha
from analytics.factor import compute_ic, forward_returns
from qt.config import RootConfig, load_config
from qt.oos_stability import (
    _run_backtest_for,
    check_oos_preconditions,
    fallback_reason_counts,
    ic_period_stats,
    sign_consistent,
    split_nav_by_holding,
    subperiod_perf,
    weight_sign_flips,
)
from qt.pipeline import (
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
    _maybe_enrich_value,
    _periods_per_year,
    _process_factors,
)
from qt.reports import render_subset_validation, write_subset_validation_summary
from qt.robustness import cell_label, derive_cell_config, iter_cells, skipped_cell_labels

__all__ = ["SubsetCellResult", "SubsetValidationResult", "run_phase3_subset",
           "render_subset_validation", "check_subset_preconditions",
           "process_group", "subperiod_cost", "summarize_subset_matrix",
           "sample_class", "independent_verdict", "summarize_by_sample"]

_LOGGER_NAME = "qt.run_phase3_subset"


@dataclass(frozen=True)
class SubsetCellResult:
    """Immutable summary of one subset-validation cell (what the report consumes)."""

    split_date: pd.Timestamp
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None
    n_train_days: int
    n_test_days: int
    boundary_dates: tuple
    # full raw factor panel columns (group-independent)
    factor_names: tuple[str, ...]
    # raw per-column IC stats — the no-drift hook vs the P3-5 report
    raw_ic_stats: dict[str, dict]
    raw_sign_consistency: dict[str, bool]
    # groups[group_label] = {factors, performance[scenario][model][period],
    #   combo_ic_stats, combo_sign_consistency, n_scored, n_fallback,
    #   fallback_reasons, sign_flips}
    groups: dict[str, dict]
    downgrades: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class SubsetValidationResult:
    """Immutable summary of one subset-validation matrix run."""

    config: RootConfig
    elapsed_seconds: float
    base_scenario: str
    # scenario label -> effective fee_rate (cost.fee_rate x multiplier)
    scenario_fees: dict[str, float]
    cells: dict[str, SubsetCellResult]
    cell_runtimes: dict[str, float]
    skipped_cells: tuple[str, ...]
    summary: dict
    report_path: Path
    log_path: Path
    # ---- P3-7 independent-sample dimension (defaults keep P3-6 results valid) --
    # cell label -> "independent" | "screened"
    cell_samples: dict[str, str] = field(default_factory=dict)
    # sample class -> a summarize_subset_matrix() dict over THAT class only
    sample_summaries: dict[str, dict] = field(default_factory=dict)
    # independent cell label -> independent_verdict() dict
    verdicts: dict[str, dict] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure helpers (network-free; unit-tested)
# --------------------------------------------------------------------------- #
def subperiod_cost(nav_slice: pd.DataFrame, periods_per_year: int = 12) -> dict:
    """Cost metrics of one pre-sliced nav segment.

    ``total_cost`` sums the per-rebalance cost column; ``cost_drag_annual`` is
    the simple arithmetic annualization (mean per-rebalance cost × rebalances
    per year — a drag estimate, not a compounded figure). Empty slices return
    NaN metrics (never crash), mirroring :func:`qt.oos_stability.subperiod_perf`.
    """
    if nav_slice is None or nav_slice.empty:
        return {"total_cost": float("nan"), "cost_drag_annual": float("nan")}
    cost = nav_slice["cost"]
    return {
        "total_cost": float(cost.sum()),
        "cost_drag_annual": float(cost.mean()) * periods_per_year,
    }


def process_group(
    cfg: RootConfig,
    factor_panel: pd.DataFrame,
    panel: pd.DataFrame,
    group_factors: list[str],
) -> pd.DataFrame:
    """Process ONE group's columns exactly as if they were the configured set.

    Subsets the raw factor panel to the group's columns FIRST, then runs the
    unchanged :func:`qt.pipeline._process_factors` — so ``drop_missing``
    requires completeness only across the GROUP's factors (a row killed by an
    excluded column's NaN survives here), and a group listing every column
    reproduces the P3-4/P3-5 processing bitwise (locked by tests).
    """
    missing = [f for f in group_factors if f not in factor_panel.columns]
    if missing:
        raise ValueError(
            f"subset group factor(s) {missing} are not raw factor-panel columns "
            f"(available: {list(factor_panel.columns)})."
        )
    return _process_factors(cfg, factor_panel[list(group_factors)], panel)


def summarize_subset_matrix(cells: dict[str, dict], base_scenario: str) -> dict:
    """Aggregate per-cell per-group findings into the cross-cell summary.

    ``cells`` maps cell label -> {"groups": {group_label: group dict}} (see
    :class:`SubsetCellResult`). Counts are attributed strictly per cell AND per
    group (never pooled — pooling would let one cell or one group mask
    another). ``ic_beats_eq_test_base`` compares test annual returns at the
    BASE scenario only (the cost ladder is reported separately per scenario).
    """
    group_labels: list[str] = []
    for cell in cells.values():
        for name in cell["groups"]:
            if name not in group_labels:
                group_labels.append(name)

    groups_summary: dict[str, dict] = {}
    for glabel in group_labels:
        present = {
            label: cell["groups"][glabel]
            for label, cell in cells.items()
            if glabel in cell["groups"]
        }
        ic_beats = 0
        for g in present.values():
            perf = g["performance"].get(base_scenario) or {}
            eq = float(
                (perf.get("equal_weight") or {}).get("test", {}).get(
                    "annual_return", float("nan"))
            )
            ic = float(
                (perf.get("ic_weighted") or {}).get("test", {}).get(
                    "annual_return", float("nan"))
            )
            if math.isfinite(eq) and math.isfinite(ic) and ic > eq:
                ic_beats += 1

        combo: dict[str, dict] = {}
        for series in ("combo_equal_weight", "combo_ic_weighted"):
            test_by_cell = {
                label: float(g["combo_ic_stats"][series]["test"].get(
                    "ic_mean", float("nan")))
                for label, g in present.items()
                if series in g["combo_ic_stats"]
            }
            combo[series] = {
                "test_ic_positive": sum(
                    1 for v in test_by_cell.values() if math.isfinite(v) and v > 0
                ),
                "sign_consistent": sum(
                    1 for g in present.values()
                    if bool(g["combo_sign_consistency"].get(series))
                ),
                "test_ic_by_cell": test_by_cell,
            }

        scenario_labels: list[str] = []
        for g in present.values():
            for scn in g["performance"]:
                if scn not in scenario_labels:
                    scenario_labels.append(scn)
        ic_test_annual_by_scenario = {
            scn: {
                label: float(
                    (g["performance"].get(scn) or {}).get("ic_weighted", {}).get(
                        "test", {}).get("annual_return", float("nan"))
                )
                for label, g in present.items()
                if scn in g["performance"]
            }
            for scn in scenario_labels
        }
        groups_summary[glabel] = {
            "n_cells": len(present),
            "ic_beats_eq_test_base": ic_beats,
            "combo": combo,
            "ic_test_annual_by_scenario": ic_test_annual_by_scenario,
        }
    return {"n_cells": len(cells), "groups": groups_summary}


def sample_class(cfg: RootConfig, universe: str, window_label: str) -> str:
    """``"independent"`` iff the cell is declared in ``independent_cells``.

    Everything else — including every cell of a pre-P3-7 config — is
    ``"screened"``: the conservative default, because independence is a human
    declaration (the machine cannot know which data took part in screening),
    and an undeclared cell must never silently count as a holdout.
    """
    sv = cfg.subset_validation
    if sv is None:
        return "screened"
    declared = {(c.universe, c.window) for c in sv.independent_cells}
    return "independent" if (universe, window_label) in declared else "screened"


def independent_verdict(
    raw_ic_stats: dict[str, dict],
    hypotheses: dict[str, str],
    n_settled: int,
    min_rebalances: int,
) -> dict:
    """Factual sign check of the pre-declared hypotheses on ONE independent cell.

    A hypothesis HOLDS iff the factor's mean IC carries the expected sign in
    BOTH subperiods of the holdout window (both postdate the screening, so
    both must agree for a clean confirmation). A NaN or missing IC never
    holds. Sample sufficiency gates everything: fewer settled rebalances
    (train + test) than ``min_rebalances`` yields ``INSUFFICIENT-DATA`` with
    the size disclosed — the per-factor table is still reported for
    transparency. This is a SIGN check on ICs, never a return claim.

    Statuses: ``SUPPORTED`` (all hold) / ``PARTIAL`` (some) / ``NOT SUPPORTED``
    (none) / ``INSUFFICIENT-DATA`` / ``NO-HYPOTHESES``.
    """
    factors: dict[str, dict] = {}
    n_holds = 0
    for name, expected in hypotheses.items():
        stats = raw_ic_stats.get(name) or {}
        train = float((stats.get("train") or {}).get("ic_mean", float("nan")))
        test = float((stats.get("test") or {}).get("ic_mean", float("nan")))

        def _matches(value: float) -> bool:
            if not math.isfinite(value) or value == 0.0:
                return False
            return value > 0 if expected == "positive" else value < 0

        holds_train = _matches(train)
        holds_test = _matches(test)
        holds = holds_train and holds_test
        n_holds += int(holds)
        factors[name] = {
            "expected": expected,
            "train_ic": train,
            "test_ic": test,
            "holds_train": holds_train,
            "holds_test": holds_test,
            "holds": holds,
        }

    n_hyp = len(hypotheses)
    if n_settled < min_rebalances:
        status = "INSUFFICIENT-DATA"
        reason = (
            f"only {n_settled} settled rebalances (train+test) in this cell, "
            f"below the configured minimum of {min_rebalances} — too few periods "
            "to read the sign check as evidence either way."
        )
    elif n_hyp == 0:
        status = "NO-HYPOTHESES"
        reason = "no hypotheses configured; nothing to verdict."
    elif n_holds == n_hyp:
        status = "SUPPORTED"
        reason = f"all {n_hyp} hypothesis factors carry the expected IC sign in BOTH subperiods."
    elif n_holds == 0:
        status = "NOT SUPPORTED"
        reason = f"0/{n_hyp} hypothesis factors carry the expected IC sign in both subperiods."
    else:
        status = "PARTIAL"
        reason = (
            f"{n_holds}/{n_hyp} hypothesis factors carry the expected IC sign in "
            "both subperiods."
        )
    return {
        "status": status,
        "reason": reason,
        "n_settled": int(n_settled),
        "min_rebalances": int(min_rebalances),
        "n_holds": n_holds,
        "n_hypotheses": n_hyp,
        "factors": factors,
    }


def summarize_by_sample(
    cells: dict[str, dict], cell_samples: dict[str, str], base_scenario: str
) -> dict[str, dict]:
    """Per-sample-class cross-cell summaries (screened and independent NEVER mix).

    Each class gets its own :func:`summarize_subset_matrix` over ONLY its
    cells, so no screened number can leak into the independent summary (or
    vice versa) — the conclusions gate of the P3-7 /goal. Unlabeled cells
    default to ``screened`` (the conservative class).
    """
    by_class: dict[str, dict[str, dict]] = {}
    for label, cell in cells.items():
        cls = cell_samples.get(label, "screened")
        by_class.setdefault(cls, {})[label] = cell
    return {
        cls: summarize_subset_matrix(class_cells, base_scenario=base_scenario)
        for cls, class_cells in by_class.items()
    }


def check_subset_preconditions(
    cfg: RootConfig, runner: str = "run-phase3-subset"
) -> None:
    """Guards: the shared OOS preconditions + the two P3-6 config sections."""
    check_oos_preconditions(cfg, runner=runner)
    if cfg.subset_validation is None:
        raise ValueError(
            f"{runner} requires a 'subset_validation' config section "
            "(factor groups + cost scenarios)."
        )
    if cfg.robustness is None:
        raise ValueError(
            f"{runner} requires a 'robustness' config section (the matrix shape: "
            "universes + windows; optional skip_cells)."
        )


def _subset_downgrades(cfg: RootConfig) -> tuple[str, ...]:
    """Base downgrades + the P3-6 comparison caveats (INV-007)."""
    extra = (
        "SUBSET VALIDATION SEMANTICS: every factor group is re-processed "
        "INDEPENDENTLY from one shared raw factor panel — drop_missing applies "
        "to the group's own columns, exactly as if the group were the configured "
        "factor set (a group listing every column reproduces the P3-4/P3-5 "
        "processing bitwise; locked by tests). Raw-factor ICs are per-column and "
        "group-independent, so the per-cell raw IC table doubles as a no-drift "
        "cross-check against the P3-5 candidate-pack report. Cost scenarios "
        "scale cost.fee_rate ONLY: scores and fills never see the fee, so trades "
        "and turnover are IDENTICAL across scenarios — only the cost line (and "
        "net return) changes (locked by tests). cost_drag_annual is the simple "
        "arithmetic annualization (mean per-rebalance cost x rebalances/year).",
        "POST-HOC SELECTION: the value+lowvol subset was chosen AFTER seeing the "
        "P3-5 results on these SAME universes x windows. This matrix quantifies "
        "RELATIVE robustness (subset vs full pack vs legacy trio on equal "
        "footing) and cost sensitivity; it is NOT independent confirmation of "
        "the value/low-vol signal — that needs genuinely new windows/universes. "
        "NOT a return claim.",
        "This remains a SMALL-SAMPLE comparison — each cell covers one index "
        "universe over one window (~22 rebalances per train+test pair) and the "
        "windows overlap across cells; subperiod metrics carry wide uncertainty "
        "and must not be read as expected performance. NOT a tuned result.",
    )
    return _collect_downgrades(cfg) + extra


# --------------------------------------------------------------------------- #
# Cell computation
# --------------------------------------------------------------------------- #
def _run_subset_cell(cfg: RootConfig, logger) -> SubsetCellResult:
    """Compute ONE subset-validation cell (no report write).

    The shared-load call sequence is IDENTICAL to the P3-3/P3-4 cell core
    (:func:`qt.oos_stability._run_oos_cell`), so the raw factor panel — and
    therefore the per-column raw ICs — cannot drift from the P3-5 run. Groups
    and cost scenarios only ADD comparisons on top of that shared load.
    """
    split = pd.Timestamp(cfg.oos.split_date)
    t0 = time.perf_counter()
    logger.info("subset cell start: project=%s split=%s", cfg.project.name, split.date())

    # --- ONE shared data load / raw factor panel (same sequence as _run_oos_cell)
    universe, symbols = _build_universe(cfg, logger)
    panel = _load_panel(cfg, symbols, logger)
    factors = _build_factors(cfg)
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_value(cfg, panel, symbols, factors, logger)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger)
    factor_panel = _compute_factor_panel(cfg, panel, factors, logger)

    horizon = int(cfg.analytics.forward_return_periods[0])
    fwd_col = forward_returns(panel, periods=(horizon,))[f"forward_return_{horizon}d"]
    ppy = _periods_per_year(cfg.backtest.rebalance)

    # --- raw per-column ICs (group-independent; no-drift hook vs P3-5) -------- #
    raw_ic_stats: dict[str, dict] = {}
    raw_consistency: dict[str, bool] = {}
    for name in factor_panel.columns:
        stats = ic_period_stats(
            compute_ic(factor_panel[name], fwd_col), split, horizon=horizon
        )
        raw_ic_stats[name] = stats
        raw_consistency[name] = sign_consistent(stats)

    # --- per group: re-process -> both alphas -> per-scenario backtests ------- #
    scenarios = list(cfg.subset_validation.cost_scenarios)
    groups: dict[str, dict] = {}
    all_boundary: set = set()
    for group in cfg.subset_validation.groups:
        processed = process_group(cfg, factor_panel, panel, group.factors)
        eq_scores = _build_scores(processed, EqualWeightAlpha())
        ic_alpha = _build_alpha(cfg)  # fresh walk-forward instance per group
        ic_scores = _build_scores(processed, ic_alpha, fwd_col)
        alpha_summary, weights_log = _alpha_disclosure(cfg, ic_alpha)
        logger.info("group %s: %d factors, alpha %s",
                    group.label, len(group.factors), alpha_summary)

        combo_ic_stats = {
            "combo_equal_weight": ic_period_stats(
                compute_ic(eq_scores, fwd_col), split, horizon=horizon),
            "combo_ic_weighted": ic_period_stats(
                compute_ic(ic_scores, fwd_col), split, horizon=horizon),
        }
        combo_consistency = {
            name: sign_consistent(stats) for name, stats in combo_ic_stats.items()
        }

        performance: dict[str, dict] = {}
        nav_ic_last: pd.DataFrame | None = None
        for scn in scenarios:
            fee = cfg.cost.fee_rate * scn.fee_multiplier
            nav_eq = _run_backtest_for(cfg, panel, universe, eq_scores, fee_rate=fee)
            nav_ic = _run_backtest_for(cfg, panel, universe, ic_scores, fee_rate=fee)
            nav_ic_last = nav_ic
            eq_train, eq_test, eq_b = split_nav_by_holding(nav_eq, split)
            ic_train, ic_test, ic_b = split_nav_by_holding(nav_ic, split)
            all_boundary.update(eq_b)
            all_boundary.update(ic_b)
            performance[scn.label] = {
                "equal_weight": {
                    "train": {**subperiod_perf(eq_train, periods_per_year=ppy),
                              **subperiod_cost(eq_train, periods_per_year=ppy)},
                    "test": {**subperiod_perf(eq_test, periods_per_year=ppy),
                             **subperiod_cost(eq_test, periods_per_year=ppy)},
                },
                "ic_weighted": {
                    "train": {**subperiod_perf(ic_train, periods_per_year=ppy),
                              **subperiod_cost(ic_train, periods_per_year=ppy)},
                    "test": {**subperiod_perf(ic_test, periods_per_year=ppy),
                             **subperiod_cost(ic_test, periods_per_year=ppy)},
                },
            }
            logger.info(
                "group %s scenario %s (fee %.4f): eq test annual=%.4f "
                "ic test annual=%.4f",
                group.label, scn.label, fee,
                performance[scn.label]["equal_weight"]["test"].get(
                    "annual_return", float("nan")),
                performance[scn.label]["ic_weighted"]["test"].get(
                    "annual_return", float("nan")),
            )

        # weight stability at the settled rebalances of the ic leg (the
        # rebalance calendar is fee-independent — any scenario's nav works).
        settled = list(nav_ic_last.index) if nav_ic_last is not None else []
        if weights_log is not None and not weights_log.empty:
            wanted = [d for d in settled if d in weights_log.index]
            weights_at_reb = weights_log.loc[wanted]
            n_scored = int(len(weights_log))
            n_fallback = int(weights_log["fallback"].sum())
        else:  # pragma: no cover - the ic leg always logs
            weights_at_reb = pd.DataFrame()
            n_scored = n_fallback = 0
        groups[group.label] = {
            "factors": tuple(group.factors),
            "performance": performance,
            "combo_ic_stats": combo_ic_stats,
            "combo_sign_consistency": combo_consistency,
            "n_scored": n_scored,
            "n_fallback": n_fallback,
            "fallback_reasons": fallback_reason_counts(
                ic_alpha.fallback_log() if hasattr(ic_alpha, "fallback_log") else {}
            ),
            "sign_flips": weight_sign_flips(weights_at_reb),
        }

    dates = panel.index.get_level_values("date")
    train_dates = sorted({d for d in dates if d < split})
    test_dates = sorted({d for d in dates if d >= split})
    result = SubsetCellResult(
        split_date=split,
        train_start=train_dates[0] if train_dates else None,
        train_end=train_dates[-1] if train_dates else None,
        test_start=test_dates[0] if test_dates else None,
        test_end=test_dates[-1] if test_dates else None,
        n_train_days=len(train_dates),
        n_test_days=len(test_dates),
        boundary_dates=tuple(sorted(all_boundary)),
        factor_names=tuple(f.name for f in factors),
        raw_ic_stats=raw_ic_stats,
        raw_sign_consistency=raw_consistency,
        groups=groups,
        downgrades=_subset_downgrades(cfg),
        elapsed_seconds=time.perf_counter() - t0,
    )
    logger.info("subset cell done: %d groups x %d scenarios (%.1fs)",
                len(groups), len(scenarios), result.elapsed_seconds)
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_phase3_subset(config_path: str) -> SubsetValidationResult:
    """Run the subset-validation matrix and write the phase3 subset report.

    Guards: the shared OOS preconditions (tushare source, ``oos`` section,
    ic_weighted alpha) plus the ``subset_validation`` and ``robustness``
    sections. Cells run sequentially (the tushare SDK is rate-limited); the
    data is loaded once per cell and shared by every group × scenario.
    """
    cfg = load_config(config_path)
    check_subset_preconditions(cfg, runner="run-phase3-subset")

    t0 = time.perf_counter()
    log_path = Path(cfg.output.log_dir) / "run_phase3_subset.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    scenarios = cfg.subset_validation.cost_scenarios
    base_scenario = next(s.label for s in scenarios if s.fee_multiplier == 1.0)
    scenario_fees = {
        s.label: cfg.cost.fee_rate * s.fee_multiplier for s in scenarios
    }
    cells_to_run = list(iter_cells(cfg))
    skipped = skipped_cell_labels(cfg)
    logger.info(
        "phase3 subset start: %d cells, %d groups, %d scenarios, %d skipped (%s)",
        len(cells_to_run), len(cfg.subset_validation.groups), len(scenarios),
        len(skipped), ", ".join(skipped) or "none",
    )

    cells: dict[str, SubsetCellResult] = {}
    runtimes: dict[str, float] = {}
    for universe_code, window in cells_to_run:
        label = cell_label(universe_code, window)
        cell_cfg = derive_cell_config(cfg, universe_code, window)
        logger.info("cell %s: start (window %s -> %s, split %s)",
                    label, window.start, window.end, window.split)
        cell_t0 = time.perf_counter()
        cells[label] = _run_subset_cell(cell_cfg, logger)
        runtimes[label] = time.perf_counter() - cell_t0
        logger.info("cell %s: done in %.1fs", label, runtimes[label])

    plain = {label: {"groups": c.groups} for label, c in cells.items()}
    cell_samples = {
        cell_label(u, w): sample_class(cfg, u, w.label) for u, w in cells_to_run
    }
    verdicts: dict[str, dict] = {}
    for label, cell in cells.items():
        if cell_samples.get(label) != "independent":
            continue
        # the rebalance calendar is group/scenario-independent: read the settled
        # train+test counts off the first group's base-scenario ic_weighted leg.
        first_group = next(iter(cell.groups.values()))
        base_perf = first_group["performance"][base_scenario]["ic_weighted"]
        n_settled = int(base_perf["train"].get("n_rebalances", 0)) + int(
            base_perf["test"].get("n_rebalances", 0)
        )
        verdicts[label] = independent_verdict(
            cell.raw_ic_stats,
            dict(cfg.subset_validation.hypotheses),
            n_settled=n_settled,
            min_rebalances=cfg.subset_validation.min_rebalances,
        )
        logger.info("independent verdict %s: %s (%s)",
                    label, verdicts[label]["status"], verdicts[label]["reason"])
    result = SubsetValidationResult(
        config=cfg,
        elapsed_seconds=time.perf_counter() - t0,
        base_scenario=base_scenario,
        scenario_fees=scenario_fees,
        cells=cells,
        cell_runtimes=runtimes,
        skipped_cells=skipped,
        summary=summarize_subset_matrix(plain, base_scenario=base_scenario),
        report_path=Path(cfg.output.report_dir) / "phase3_subset_validation.md",
        log_path=log_path,
        cell_samples=cell_samples,
        sample_summaries=summarize_by_sample(
            plain, cell_samples, base_scenario=base_scenario
        ),
        verdicts=verdicts,
    )
    write_subset_validation_summary(result)
    logger.info(
        "phase3 subset done: %d cells (%d independent), report=%s (%.1fs)",
        result.summary["n_cells"],
        sum(1 for v in cell_samples.values() if v == "independent"),
        result.report_path, result.elapsed_seconds,
    )
    return result
