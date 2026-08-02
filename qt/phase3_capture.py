"""D6b phase-3 legacy capture + reconciliation harness (PR-1).

D6b switches the three phase-3 validation runners (``qt.oos_stability`` /
``qt.robustness`` / ``qt.subset_validation``) from the legacy factor-sourcing
path (``factor.compute(panel)`` inside ``qt.pipeline._compute_factor_panel``)
to the factor service (``qt.pipeline._serve_factor_panel``). Before any runner
is touched, this module captures what the LEGACY code produces, so the switch
can later be reconciled against bytes that were frozen beforehand (the same
"freeze first, reconcile against the frozen copy" discipline as D5's
``qt.exec_baseline_freeze``).

Two capture legs
----------------
PANEL leg (:func:`capture_panels`): for every cell of a phase-3 config, replay
the runner's EXACT data-load sequence (``_build_universe`` -> ``_load_panel``
-> the four enrichments — one shared cache threaded through all of them, the
D6a-3-fixed call shape), then compute the factor panel TWICE in one process:

* legacy — ``factor.compute(panel)`` concatenated, exactly what
  ``_compute_factor_panel`` computes (its write/log side effects excluded);
* served — ``qt.factor_source.factor_values`` over the same enriched panel,
  exactly what ``_serve_factor_panel`` serves.

Both panels land as parquet (the legacy one is the D6b reference) plus a
per-cell reconcile JSON (max abs diff / NaN-mask mismatch / index-set
difference). The load sequence MIRRORS ``qt.oos_stability._run_oos_cell`` and
``qt.subset_validation._run_subset_cell``; if those change, this changes.

RUNNER leg (:func:`capture_runner`): call ``run_phase3_oos`` /
``run_phase3_robustness`` / ``run_phase3_subset`` and export the returned
result object as full-precision JSON — every leaf of performance / ic_stats /
sign_consistency / sign_flips / fallback counts / verdicts. The ONLY excluded
fields are the explicit :data:`EXCLUDED_RESULT_FIELDS` list (wall-clock
timings, the config object which carries the secret-file path, and output
paths — locations, not values).

Reconcilers
-----------
:func:`compare_json` deep-compares two captured JSON trees leaf-by-leaf with
0.0 float tolerance (NaN equals NaN); the difference list is ALLOWED to be
empty — a non-empty list is a finding, never something to explain away
silently (design §六.5: an uncatalogued difference is a failure).
:func:`compare_panels` does the same at the factor-panel level.

CLI (what the L0/L1 capture runs invoke)::

    python -m qt.phase3_capture run --runner oos --config CFG --out result.json
    python -m qt.phase3_capture capture-panels --config CFG --out-dir DIR
    python -m qt.phase3_capture compare-json A.json B.json [--out report.json]
    python -m qt.phase3_capture compare-panels A.parquet B.parquet [--out R.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qt.config import RootConfig, load_config
from qt.factor_source import factor_values, open_factor_value_store
from qt.pipeline import (
    _build_cache,
    _build_factors,
    _build_universe,
    _factor_params_by_id,
    _load_panel,
    _log_run_cache_stats,
    _make_logger,
    _maybe_enrich_covariates,
    _maybe_enrich_financials,
    _maybe_enrich_listing,
    _maybe_enrich_value,
)

__all__ = [
    "EXCLUDED_RESULT_FIELDS",
    "capture_panels",
    "capture_runner",
    "compare_json",
    "compare_panels",
    "iter_cell_configs",
    "legacy_factor_panel",
    "load_cell_panel",
    "served_factor_panel",
    "to_jsonable",
]

_LOGGER_NAME = "qt.phase3_capture"

#: Result-object fields EXCLUDED from the runner-leg capture. Timings are not
#: values; the config object carries the secret-file path; report/log paths are
#: locations, not values. Everything else is captured leaf-by-leaf — if a field
#: ever needs adding here, that is a disclosure decision, not a convenience.
EXCLUDED_RESULT_FIELDS = frozenset({
    "config",
    "elapsed_seconds",
    "cell_runtimes",
    "report_path",
    "log_path",
})

#: Runner name -> (module, entry point). Imported lazily inside
#: :func:`capture_runner` so the pure comparators stay import-light.
_RUNNERS = {
    "oos": ("qt.oos_stability", "run_phase3_oos"),
    "robustness": ("qt.robustness", "run_phase3_robustness"),
    "subset": ("qt.subset_validation", "run_phase3_subset"),
}


# --------------------------------------------------------------------------- #
# Runner-leg serialization (full precision; network-free; unit-tested)
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Reduce a runner result object to a JSON tree, leaf by leaf.

    Dataclass fields in :data:`EXCLUDED_RESULT_FIELDS` are dropped (the ONLY
    exclusions). Floats pass through untouched — Python's ``repr`` round-trips
    them exactly, and ``json`` emits ``NaN``/``Infinity`` tokens that
    ``json.load`` reads back, so the capture is full-precision by construction.
    DataFrames become ``{"__dataframe__", "columns", "index", "data"}`` with
    the index as row tuples (Timestamps as ISO strings); Timestamps become ISO
    strings (``NaT`` -> ``"NaT"``); Paths become strings.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name not in EXCLUDED_RESULT_FIELDS
        }
    if isinstance(obj, pd.DataFrame):
        return {
            "__dataframe__": True,
            "columns": [str(c) for c in obj.columns],
            "index": [to_jsonable(idx) for idx in obj.index],
            "data": to_jsonable(obj.to_numpy(dtype=object).tolist()),
        }
    if isinstance(obj, pd.Series):
        return {
            "__series__": True,
            "name": str(obj.name),
            "index": [to_jsonable(idx) for idx in obj.index],
            "data": to_jsonable(obj.tolist()),
        }
    if isinstance(obj, pd.Timestamp):
        return "NaT" if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def _flatten(tree: Any, prefix: str, out: dict[str, Any]) -> None:
    """Flatten a JSON tree into {dotted path -> leaf}; lists index as ``[i]``."""
    if isinstance(tree, dict):
        if not tree:
            out[prefix] = {}
        for key, value in tree.items():
            _flatten(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(tree, list):
        if not tree:
            out[prefix] = []
        for i, value in enumerate(tree):
            _flatten(value, f"{prefix}[{i}]", out)
    else:
        out[prefix] = tree


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json(left: Any, right: Any) -> dict:
    """Deep leaf-wise compare of two captured JSON trees (0.0 float tolerance).

    Floats compare EXACTLY (``==``) — the capture is full-precision, so any
    movement is a real movement; two NaNs at the same path count as equal.
    Returns ``{"n_diffs": int, "diffs": [{"path", "kind", "left", "right"}]}``
    with ``kind`` in ``missing_left`` / ``missing_right`` / ``type_mismatch`` /
    ``value_mismatch``. An empty ``diffs`` list is the expected reconciliation
    outcome; a non-empty one is a finding to catalogue, not to round away.
    """
    flat_left: dict[str, Any] = {}
    flat_right: dict[str, Any] = {}
    _flatten(left, "", flat_left)
    _flatten(right, "", flat_right)
    diffs: list[dict] = []
    for path in sorted(set(flat_left) | set(flat_right)):
        if path not in flat_left:
            diffs.append({"path": path, "kind": "missing_left",
                          "left": None, "right": flat_right[path]})
            continue
        if path not in flat_right:
            diffs.append({"path": path, "kind": "missing_right",
                          "left": flat_left[path], "right": None})
            continue
        lv, rv = flat_left[path], flat_right[path]
        if _is_number(lv) and _is_number(rv):
            lf, rf = float(lv), float(rv)
            both_nan = math.isnan(lf) and math.isnan(rf)
            if not both_nan and lf != rf:
                diffs.append({"path": path, "kind": "value_mismatch",
                              "left": lv, "right": rv})
            continue
        if type(lv) is not type(rv):
            diffs.append({"path": path, "kind": "type_mismatch",
                          "left": str(lv), "right": str(rv)})
            continue
        if lv != rv:
            diffs.append({"path": path, "kind": "value_mismatch",
                          "left": lv, "right": rv})
    return {"n_diffs": len(diffs), "diffs": diffs}


# --------------------------------------------------------------------------- #
# Panel-leg comparison (network-free; unit-tested)
# --------------------------------------------------------------------------- #
def compare_panels(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Cell-by-cell compare of two factor panels (legacy vs served, or L0 vs L1).

    Aligned on the COMMON (index, column) grid; cells compare with exact float
    equality (two NaNs equal). Reports the index/column set differences
    explicitly — a panel whose GRID moved is a different finding from one whose
    VALUES moved, and the report must say which it is.
    """
    left_only_index = left.index.difference(right.index)
    right_only_index = right.index.difference(left.index)
    left_only_cols = [c for c in left.columns if c not in right.columns]
    right_only_cols = [c for c in right.columns if c not in left.columns]
    common_cols = [c for c in left.columns if c in right.columns]
    common_index = left.index.intersection(right.index)

    per_column: dict[str, dict] = {}
    max_abs_diff_overall = 0.0
    for col in common_cols:
        lcol = left.loc[common_index, col].astype(float)
        rcol = right.loc[common_index, col].astype(float)
        lnan, rnan = lcol.isna(), rcol.isna()
        nan_mask_mismatch = int((lnan != rnan).sum())
        both = ~(lnan | rnan)
        diff = (lcol[both] - rcol[both]).abs()
        n_value_mismatch = int((diff > 0).sum())
        max_abs = float(diff.max()) if len(diff) else 0.0
        max_abs_diff_overall = max(max_abs_diff_overall, max_abs)
        per_column[str(col)] = {
            "n_compared": int(both.sum()),
            "n_value_mismatch": n_value_mismatch,
            "nan_mask_mismatch": nan_mask_mismatch,
            "max_abs_diff": max_abs,
        }
    return {
        "n_rows_left": int(len(left)),
        "n_rows_right": int(len(right)),
        "n_common_rows": int(len(common_index)),
        "left_only_index": [str(i) for i in left_only_index[:20]],
        "right_only_index": [str(i) for i in right_only_index[:20]],
        "n_left_only_index": int(len(left_only_index)),
        "n_right_only_index": int(len(right_only_index)),
        "left_only_columns": left_only_cols,
        "right_only_columns": right_only_cols,
        "max_abs_diff": max_abs_diff_overall,
        "per_column": per_column,
    }


# --------------------------------------------------------------------------- #
# Panel-leg capture (replays the runner load sequence; hits the data source)
# --------------------------------------------------------------------------- #
def iter_cell_configs(cfg: RootConfig) -> list[tuple[str, RootConfig]]:
    """(label, per-cell config) pairs mirroring what the matrix runners run.

    A config with a ``robustness`` section enumerates its cells through
    ``qt.robustness.iter_cells`` + ``derive_cell_config`` (skips honoured); a
    single-cell config (the OOS one) yields itself once, labelled by
    ``data.output_name``.
    """
    if cfg.robustness is None:
        return [(cfg.data.output_name, cfg)]
    from qt.robustness import cell_label, derive_cell_config, iter_cells

    return [
        (cell_label(universe, window), derive_cell_config(cfg, universe, window))
        for universe, window in iter_cells(cfg)
    ]


def load_cell_panel(cfg: RootConfig, logger: logging.Logger):
    """Replay the phase-3 runners' data-load sequence EXACTLY (D6a-3-fixed shape).

    Mirrors ``qt.oos_stability._run_oos_cell`` / ``qt.subset_validation.
    _run_subset_cell``: one shared cache from ``_build_cache`` threaded through
    universe build, panel load, and ALL FOUR enrichments. Returns
    ``(universe, symbols, panel, factors)``; factor panels are NOT computed
    here (that is the legacy/served pair's job, so both see the same panel).
    """
    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    factors = _build_factors(cfg)
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger, cache)
    panel = _maybe_enrich_value(cfg, panel, symbols, factors, logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)
    return universe, symbols, panel, factors


def legacy_factor_panel(panel: pd.DataFrame, factors: list) -> pd.DataFrame:
    """EXACTLY what ``qt.pipeline._compute_factor_panel`` computes (no write/log).

    Kept as the verbatim three lines rather than a call into the pipeline
    helper so the capture has NO side effect on ``factors/factors.parquet`` —
    a capture that overwrites the artifact it references would be the
    compare-with-a-copy-of-itself failure mode.
    """
    columns = [factor.compute(panel).rename(factor.name) for factor in factors]
    return pd.concat(columns, axis=1)


def served_factor_panel(
    cfg: RootConfig, panel: pd.DataFrame, factors: list, symbols: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """EXACTLY what ``qt.pipeline._serve_factor_panel`` serves (no write/log).

    Same wiring (store policy, close view x close-to-close pairing, params
    keyed by built-factor id) minus the parquet write — same no-side-effect
    reason as :func:`legacy_factor_panel`.
    """
    with open_factor_value_store(cfg, logger) as store:
        served = factor_values(
            factors,
            panel,
            symbols,
            store=store,
            params_by_id=_factor_params_by_id(cfg, factors),
        )
    logger.info(
        "served factor panel: %d rows x %d columns (%d served, %d footprint "
        "rows reduced away)",
        len(served.frame), served.frame.shape[1], served.served_rows,
        served.footprint_rows_dropped,
    )
    return served.frame


def capture_panels(config_path: str, out_dir: str) -> dict:
    """Panel-leg capture for every cell of ``config_path`` into ``out_dir``.

    Per cell ``<label>`` writes ``<label>__legacy.parquet`` (the D6b
    reference), ``<label>__served.parquet`` and
    ``<label>__panel_reconcile.json``; a run-level ``_summary.json`` ties them
    together. ``|`` in matrix cell labels becomes ``_`` in filenames.
    """
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out / "capture_panels.log", name=_LOGGER_NAME)
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells": {},
    }
    for label, cell_cfg in iter_cell_configs(cfg):
        safe = label.replace("|", "_")
        logger.info("panel capture cell %s: start", label)
        _, symbols, panel, factors = load_cell_panel(cell_cfg, logger)
        legacy = legacy_factor_panel(panel, factors)
        served = served_factor_panel(cell_cfg, panel, factors, symbols, logger)
        legacy_path = out / f"{safe}__legacy.parquet"
        served_path = out / f"{safe}__served.parquet"
        legacy.to_parquet(legacy_path)
        served.to_parquet(served_path)
        reconcile = compare_panels(legacy, served)
        reconcile_path = out / f"{safe}__panel_reconcile.json"
        reconcile_path.write_text(
            json.dumps(reconcile, indent=1, allow_nan=True), encoding="utf-8"
        )
        summary["cells"][label] = {
            "legacy_parquet": str(legacy_path),
            "served_parquet": str(served_path),
            "reconcile_json": str(reconcile_path),
            "max_abs_diff": reconcile["max_abs_diff"],
        }
        logger.info("panel capture cell %s: max_abs_diff=%r", label,
                    reconcile["max_abs_diff"])
    (out / "_summary.json").write_text(
        json.dumps(summary, indent=1, allow_nan=True), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# Runner-leg capture
# --------------------------------------------------------------------------- #
def capture_runner(runner: str, config_path: str, out_path: str) -> dict:
    """Run one phase-3 runner and export its result object as full-precision JSON.

    The runner's own writes (report / log / panel parquet) still happen — this
    is a capture OF the legacy runner, not a reimplementation of it. Returns
    the payload dict that was written.
    """
    if runner not in _RUNNERS:
        raise ValueError(f"unknown runner {runner!r}; known: {sorted(_RUNNERS)}")
    module_name, entry = _RUNNERS[runner]
    import importlib

    run = getattr(importlib.import_module(module_name), entry)
    result = run(config_path)
    payload = {
        "runner": runner,
        "config_path": str(config_path),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_fields": sorted(EXCLUDED_RESULT_FIELDS),
        "result": to_jsonable(result),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, allow_nan=True), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    payload = capture_runner(args.runner, args.config, args.out)
    leaves: dict[str, Any] = {}
    _flatten(payload["result"], "", leaves)
    print(f"OK capture runner={args.runner}: {len(leaves)} leaves -> {args.out}")
    return 0


def _cmd_capture_panels(args: argparse.Namespace) -> int:
    summary = capture_panels(args.config, args.out_dir)
    worst = max(
        (cell["max_abs_diff"] for cell in summary["cells"].values()), default=0.0
    )
    print(f"OK capture-panels: {len(summary['cells'])} cell(s), "
          f"worst max_abs_diff={worst!r} -> {args.out_dir}")
    return 0


def _cmd_compare_json(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    report = compare_json(left.get("result", left), right.get("result", right))
    text = json.dumps(report, indent=1, allow_nan=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"compare-json: n_diffs={report['n_diffs']}")
    return 0 if report["n_diffs"] == 0 else 1


def _cmd_compare_panels(args: argparse.Namespace) -> int:
    left = pd.read_parquet(args.left)
    right = pd.read_parquet(args.right)
    report = compare_panels(left, right)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=1, allow_nan=True), encoding="utf-8"
        )
    print(f"compare-panels: max_abs_diff={report['max_abs_diff']!r} "
          f"left_only_index={report['n_left_only_index']} "
          f"right_only_index={report['n_right_only_index']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qt.phase3_capture",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Runner-leg capture (runs the legacy runner).")
    p_run.add_argument("--runner", required=True, choices=sorted(_RUNNERS))
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_panels = sub.add_parser("capture-panels",
                              help="Panel-leg capture (legacy vs served, per cell).")
    p_panels.add_argument("--config", required=True)
    p_panels.add_argument("--out-dir", required=True)
    p_panels.set_defaults(func=_cmd_capture_panels)

    p_cj = sub.add_parser("compare-json", help="Leaf-wise compare of two captures.")
    p_cj.add_argument("left")
    p_cj.add_argument("right")
    p_cj.add_argument("--out", default=None)
    p_cj.set_defaults(func=_cmd_compare_json)

    p_cp = sub.add_parser("compare-panels", help="Cell-wise compare of two panels.")
    p_cp.add_argument("left")
    p_cp.add_argument("right")
    p_cp.add_argument("--out", default=None)
    p_cp.set_defaults(func=_cmd_compare_panels)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
