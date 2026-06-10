"""Phase 3-4: robustness matrix — the P3-3 OOS check over universes × windows.

A REPORT-ONLY validation layer on top of :mod:`qt.oos_stability`: the SAME
three factors and the SAME equal_weight-vs-ic_weighted comparison are re-run
on every (universe, window) cell declared in the ``robustness`` config section,
to answer whether the P3-2/P3-3 conclusions were an SSE50 small-sample
accident. No new factor, no new alpha, no change to portfolio / execution /
factor math.

Every cell reuses the P3-3 cell core VERBATIM (:func:`qt.oos_stability._run_oos_cell`):
holding-window performance slicing, realization-date IC slicing, walk-forward
weights, and the shared guards (:func:`qt.oos_stability.check_oos_preconditions`,
incl. the ic_weighted fake-comparison guard). Per-cell configs are derived from
the base config with ONLY the cell identity swapped in (universe.index_code,
data.start/end, oos.split_date, data.output_name — the last so cells never
overwrite each other's panel parquet).

``robustness.skip_cells`` may drop named cells (runtime budget — a wide
universe × long fold can take hours through the rate-limited SDK); skipped
cells are DISCLOSED in the report, never silently absent. The report carries
per-cell diagnostics plus a cross-cell summary matrix (which findings hold
across cells and which do not) and the explicit not-a-return-claim caveat.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


from qt.config import RobustnessWindowCfg, RootConfig, load_config
from qt.oos_stability import OOSResult, _run_oos_cell, check_oos_preconditions
from qt.pipeline import _make_logger
from qt.reports import render_robustness_matrix, write_robustness_matrix_summary

_LOGGER_NAME = "qt.run_phase3_robustness"

__all__ = ["RobustnessResult", "run_phase3_robustness", "render_robustness_matrix",
           "cell_label", "iter_cells", "derive_cell_config", "summarize_matrix"]


@dataclass(frozen=True)
class RobustnessResult:
    """Immutable summary of one robustness-matrix run (what the report consumes)."""

    config: RootConfig
    elapsed_seconds: float
    # cell label ("<index_code>|<window label>") -> the cell's OOSResult
    cells: dict[str, OOSResult]
    cell_runtimes: dict[str, float]
    skipped_cells: tuple[str, ...]
    # cross-cell aggregation (see summarize_matrix)
    summary: dict
    report_path: Path
    log_path: Path


# --------------------------------------------------------------------------- #
# Cell enumeration / derivation (pure; unit-tested)
# --------------------------------------------------------------------------- #
def cell_label(universe: str, window: RobustnessWindowCfg) -> str:
    """Canonical cell id: '<index_code>|<window label>'."""
    return f"{universe}|{window.label}"


def iter_cells(cfg: RootConfig) -> Iterator[tuple[str, RobustnessWindowCfg]]:
    """Yield the (universe, window) cells to RUN, config order, skips removed."""
    rob = cfg.robustness
    if rob is None:
        return
    skipped = {(s.universe, s.window) for s in rob.skip_cells}
    for universe in rob.universes:
        for window in rob.windows:
            if (universe, window.label) in skipped:
                continue
            yield universe, window


def skipped_cell_labels(cfg: RootConfig) -> tuple[str, ...]:
    """Labels of the explicitly skipped cells (disclosed in the report)."""
    rob = cfg.robustness
    if rob is None:
        return ()
    by_label = {w.label: w for w in rob.windows}
    return tuple(
        cell_label(s.universe, by_label[s.window]) for s in rob.skip_cells
    )


def derive_cell_config(
    cfg: RootConfig, universe: str, window: RobustnessWindowCfg
) -> RootConfig:
    """One cell's config: the base verbatim with ONLY the cell identity swapped.

    Rebuilt through ``RootConfig(**dict)`` so every pydantic validation
    (date order, split-inside-window, ...) re-runs on the derived values.
    ``data.output_name`` gets a per-cell suffix so cells never overwrite each
    other's panel parquet.
    """
    raw = cfg.model_dump()
    raw["universe"]["index_code"] = universe
    raw["data"]["start"] = window.start
    raw["data"]["end"] = window.end
    raw["oos"] = {"split_date": window.split}
    safe_universe = universe.replace(".", "_").lower()
    raw["data"]["output_name"] = (
        f"{cfg.data.output_name}_{safe_universe}_{window.label.replace('-', '_')}"
    )
    return RootConfig(**raw)


# --------------------------------------------------------------------------- #
# Cross-cell aggregation (pure; unit-tested)
# --------------------------------------------------------------------------- #
def summarize_matrix(cells: dict[str, dict]) -> dict:
    """Aggregate per-cell findings into the cross-cell stability summary.

    ``cells`` maps cell label -> a dict (or OOSResult-like mapping) carrying
    ``performance`` / ``ic_stats`` / ``sign_consistency`` / ``sign_flips`` /
    ``n_scored`` / ``n_fallback``. Counts are attributed strictly per cell
    (never pooled across cells — pooling would let a big cell mask a small
    one). Returns::

        {"n_cells", "ic_beats_eq_test": <#cells where ic test annual > eq>,
         "series": {name: {"n_cells", "test_ic_positive", "sign_consistent",
                            "test_ic_by_cell": {label: mean}}}}
    """
    series_names: list[str] = []
    for cell in cells.values():
        for name in cell["ic_stats"]:
            if name not in series_names:
                series_names.append(name)

    series_summary: dict[str, dict] = {}
    for name in series_names:
        present = {
            label: cell for label, cell in cells.items() if name in cell["ic_stats"]
        }
        test_by_cell = {
            label: float(cell["ic_stats"][name]["test"].get("ic_mean", float("nan")))
            for label, cell in present.items()
        }
        series_summary[name] = {
            "n_cells": len(present),
            "test_ic_positive": sum(
                1 for v in test_by_cell.values() if math.isfinite(v) and v > 0
            ),
            "sign_consistent": sum(
                1 for cell in present.values()
                if bool(cell["sign_consistency"].get(name))
            ),
            "test_ic_by_cell": test_by_cell,
        }

    ic_beats = 0
    for cell in cells.values():
        eq = float(
            cell["performance"]["equal_weight"]["test"].get("annual_return", float("nan"))
        )
        ic = float(
            cell["performance"]["ic_weighted"]["test"].get("annual_return", float("nan"))
        )
        if math.isfinite(eq) and math.isfinite(ic) and ic > eq:
            ic_beats += 1
    return {
        "n_cells": len(cells),
        "ic_beats_eq_test": ic_beats,
        "series": series_summary,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_phase3_robustness(config_path: str) -> RobustnessResult:
    """Run every matrix cell through the P3-3 core and write the matrix report.

    Guards are the single-run preconditions (tushare source, ic_weighted alpha;
    the base ``oos`` section validates the base config) plus a ``robustness``
    section. Cells run sequentially (the tushare SDK is rate-limited anyway);
    each cell's runtime is recorded and disclosed.
    """
    cfg = load_config(config_path)
    check_oos_preconditions(cfg, runner="run-phase3-robustness")
    if cfg.robustness is None:
        raise ValueError(
            "run-phase3-robustness requires a 'robustness' config section "
            "(universes + windows; optional skip_cells)."
        )

    t0 = time.perf_counter()
    log_path = Path(cfg.output.log_dir) / "run_phase3_robustness.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    cells_to_run = list(iter_cells(cfg))
    skipped = skipped_cell_labels(cfg)
    logger.info(
        "phase3 robustness start: %d cells to run, %d skipped (%s)",
        len(cells_to_run), len(skipped), ", ".join(skipped) or "none",
    )

    cells: dict[str, OOSResult] = {}
    runtimes: dict[str, float] = {}
    for universe, window in cells_to_run:
        label = cell_label(universe, window)
        cell_cfg = derive_cell_config(cfg, universe, window)
        logger.info("cell %s: start (window %s -> %s, split %s)",
                    label, window.start, window.end, window.split)
        cell_t0 = time.perf_counter()
        cells[label] = _run_oos_cell(cell_cfg, logger, log_path)
        runtimes[label] = time.perf_counter() - cell_t0
        logger.info("cell %s: done in %.1fs", label, runtimes[label])

    plain = {
        label: {
            "performance": c.performance,
            "ic_stats": c.ic_stats,
            "sign_consistency": c.sign_consistency,
            "sign_flips": c.sign_flips,
            "n_scored": c.n_scored,
            "n_fallback": c.n_fallback,
        }
        for label, c in cells.items()
    }
    result = RobustnessResult(
        config=cfg,
        elapsed_seconds=time.perf_counter() - t0,
        cells=cells,
        cell_runtimes=runtimes,
        skipped_cells=skipped,
        summary=summarize_matrix(plain),
        report_path=Path(cfg.output.report_dir) / "phase3_robustness_matrix.md",
        log_path=log_path,
    )
    write_robustness_matrix_summary(result)
    logger.info(
        "phase3 robustness done: %d cells, ic beats eq in %d, report=%s (%.1fs)",
        result.summary["n_cells"], result.summary["ic_beats_eq_test"],
        result.report_path, result.elapsed_seconds,
    )
    return result
