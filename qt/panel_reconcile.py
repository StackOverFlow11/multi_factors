"""D2 cell-by-cell reconciliation of the migrated engine vs the D1 frozen baseline.

Design v3.2 §五 leg 4 (R11), pulled forward into D2 by the task card: rebuild
all 14 closing factors' RAW panels on the CURRENT tree (the ``qt.panel_freeze``
recipes route through the ``data.clean`` shims, so they exercise the migrated
``factors.compute.minute`` math automatically), write them to a SEPARATE
directory, and compare each panel CELL BY CELL against the frozen D1 baseline:

* the (date, symbol) index sets must be IDENTICAL;
* the NaN sets must be IDENTICAL (a NaN-set change would be an undeclared
  NaN-policy change — D2 declares none);
* on the jointly-finite cells the relative difference
  ``|new - old| / max(|old|, |new|)`` must be <= 1e-12 (the float-reordering
  budget; any excess means STOP AND FIX THE ENGINE, never widen the budget).

Anti-empty-reconciliation provenance (the ``compare_postmerge.py`` lesson):

* the frozen side is NEVER regenerated here — it is read from
  ``artifacts/refactor_baseline/panels`` exactly as D1 froze it, and each
  frozen file's canonical content hash is first checked against the D1
  ``manifest.json`` (a stale/clobbered baseline fails loudly before any
  comparison is trusted);
* the comparison reads BOTH sides from disk through two independent file
  reads — the freshly built panel is written to
  ``artifacts/refactor_baseline/panels_d2`` first and read back, so no shared
  in-memory object can make the equality vacuous.

Run: ``python -m qt.panel_reconcile`` (deliberately NOT registered in qt/cli;
``--resume`` reads back already-written panels_d2 files after checking that
they were produced at the SAME git SHA recorded in ``manifest_d2.json``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.clean.schema import DATE_LEVEL
from qt.panel_freeze import (
    DEFAULT_CONFIG,
    PANEL_STORE_NAME,
    atomic_write_parquet,
    atomic_write_text,
    canonical_content_hash,
    minute_recipes,
    read_frozen_panel,
    _git_head_sha,
)

DEFAULT_BASELINE_ROOT = "artifacts/refactor_baseline"
D2_SUBDIR = "panels_d2"
RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CellComparison:
    """One factor's cell-by-cell comparison result (no secrets)."""

    factor_id: str
    rows: int
    index_equal: bool
    nan_only_in_frozen: int
    nan_only_in_new: int
    n_joint_finite: int
    n_cells_beyond_tol: int
    max_rel_diff: float
    max_abs_diff: float
    hashes_equal: bool

    @property
    def ok(self) -> bool:
        return (
            self.index_equal
            and self.nan_only_in_frozen == 0
            and self.nan_only_in_new == 0
            and self.n_cells_beyond_tol == 0
        )


def compare_panels(frozen: pd.Series, new: pd.Series, factor_id: str) -> CellComparison:
    """Cell-by-cell comparison per the module docstring's three rules."""
    index_equal = frozen.index.equals(new.index)
    if not index_equal:
        return CellComparison(
            factor_id=factor_id,
            rows=int(len(frozen)),
            index_equal=False,
            nan_only_in_frozen=-1,
            nan_only_in_new=-1,
            n_joint_finite=0,
            n_cells_beyond_tol=-1,
            max_rel_diff=float("nan"),
            max_abs_diff=float("nan"),
            hashes_equal=False,
        )
    a = frozen.to_numpy(dtype=float)
    b = new.to_numpy(dtype=float)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    joint = ~nan_a & ~nan_b
    diff = np.abs(a[joint] - b[joint])
    denom = np.maximum(np.abs(a[joint]), np.abs(b[joint]))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(denom > 0.0, diff / denom, 0.0)
    beyond = int((rel > RELATIVE_TOLERANCE).sum())
    return CellComparison(
        factor_id=factor_id,
        rows=int(len(frozen)),
        index_equal=True,
        nan_only_in_frozen=int((nan_a & ~nan_b).sum()),
        nan_only_in_new=int((nan_b & ~nan_a).sum()),
        n_joint_finite=int(joint.sum()),
        n_cells_beyond_tol=beyond,
        max_rel_diff=float(rel.max()) if rel.size else 0.0,
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        hashes_equal=canonical_content_hash(frozen) == canonical_content_hash(new),
    )


def _verify_frozen_against_manifest(baseline_root: Path) -> dict[str, str]:
    """Check every frozen panel's canonical hash against the D1 manifest.

    Returns {factor_id: canonical_hash}. Any mismatch raises — a clobbered or
    regenerated baseline must never be silently reconciled against.
    """
    manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    frozen_hashes: dict[str, str] = {}
    for row in manifest["rows"]:
        factor_id = row["factor_id"]
        path = baseline_root / "panels" / row["file"]
        frozen = read_frozen_panel(path, factor_id)
        actual = canonical_content_hash(frozen)
        if actual != row["canonical_sha256"]:
            raise RuntimeError(
                f"frozen baseline {row['file']} canonical hash {actual} does not "
                f"match the D1 manifest ({row['canonical_sha256']}) — the baseline "
                "has been altered since the freeze; refusing to reconcile."
            )
        frozen_hashes[factor_id] = actual
    return frozen_hashes


def render_report(rows: list[CellComparison], header: dict) -> str:
    lines = ["# D2 panel reconciliation vs the D1 frozen baseline", ""]
    for key in sorted(header):
        lines.append(f"- **{key}**: {header[key]}")
    lines.append("")
    cols = (
        "factor_id | rows | index_equal | nan_only_frozen | nan_only_new | "
        "joint_finite | cells_beyond_1e-12 | max_rel_diff | max_abs_diff | "
        "hash_equal | verdict"
    ).split(" | ")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append(
            f"| {r.factor_id} | {r.rows} | {r.index_equal} | "
            f"{r.nan_only_in_frozen} | {r.nan_only_in_new} | {r.n_joint_finite} | "
            f"{r.n_cells_beyond_tol} | {r.max_rel_diff!r} | {r.max_abs_diff!r} | "
            f"{r.hashes_equal} | {'OK' if r.ok else 'MISMATCH'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_panel_reconcile(
    config_path: str = DEFAULT_CONFIG,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
    *,
    resume: bool = False,
) -> list[CellComparison]:
    """Rebuild the 14 RAW panels on the current tree and reconcile vs the freeze."""
    from qt.config import load_config
    from qt.eval_jump_amount_corr import _build_book_factors, _check_preconditions
    from qt.pipeline import (
        _build_cache,
        _build_universe,
        _load_panel,
        _log_run_cache_stats,
        _make_logger,
        _maybe_enrich_covariates,
        _maybe_enrich_value,
    )

    started = time.monotonic()
    root = Path(baseline_root)
    d2_dir = root / D2_SUBDIR
    d2_dir.mkdir(parents=True, exist_ok=True)
    head_sha = _git_head_sha()

    # provenance for --resume: a panels_d2 file may only be reused if it was
    # produced at the SAME git SHA (else it might carry an older engine).
    manifest_d2_path = root / "manifest_d2.json"
    prior: dict = {}
    if manifest_d2_path.exists():
        prior = json.loads(manifest_d2_path.read_text(encoding="utf-8"))
    if resume and prior and prior.get("producing_git_sha") != head_sha:
        raise RuntimeError(
            f"--resume refused: existing panels_d2 were produced at "
            f"{prior.get('producing_git_sha')} but HEAD is {head_sha}; delete "
            f"{d2_dir} to rebuild from scratch."
        )

    frozen_hashes = _verify_frozen_against_manifest(root)

    cfg = load_config(config_path)
    _check_preconditions(cfg)
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"output_name": PANEL_STORE_NAME})}
    )
    log_path = Path(cfg.output.log_dir) / "panel_reconcile.log"
    logger = _make_logger(log_path, name="qt.panel_reconcile")
    logger.info("panel reconcile: config=%s baseline_root=%s sha=%s",
                config_path, root, head_sha)

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    book_factors = _build_book_factors()
    panel = _maybe_enrich_value(cfg, panel, symbols, book_factors, logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)
    panel_dates = pd.Index(
        pd.unique(panel.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    )

    results: list[CellComparison] = []
    d2_rows: list[dict] = []

    def _reconcile_one(factor_id: str, raw: pd.Series | None) -> None:
        target = d2_dir / f"{factor_id}.parquet"
        if raw is None:  # resume path: reuse the already-written d2 panel
            logger.info("resume: reading back %s", target.name)
        else:
            atomic_write_parquet(raw.rename(factor_id), target)
        new = read_frozen_panel(target, factor_id)          # file read #1
        frozen = read_frozen_panel(root / "panels" / f"{factor_id}.parquet", factor_id)  # #2
        comp = compare_panels(frozen, new, factor_id)
        results.append(comp)
        d2_rows.append(
            {"factor_id": factor_id, "canonical_sha256": canonical_content_hash(new),
             "frozen_sha256": frozen_hashes[factor_id], "rows": comp.rows}
        )
        logger.info(
            "reconciled %s: ok=%s max_rel=%.3e nan_frozen_only=%d nan_new_only=%d",
            factor_id, comp.ok, comp.max_rel_diff,
            comp.nan_only_in_frozen, comp.nan_only_in_new,
        )

    # book factors (raw, pre-processing) — same call shape as the freeze
    for factor in book_factors:
        target = d2_dir / f"{factor.name}.parquet"
        if resume and target.exists():
            _reconcile_one(factor.name, None)
        else:
            _reconcile_one(factor.name, factor.compute(panel).rename(factor.name))

    # minute factors — the freeze recipes, now routed through the D2 shims
    live_total = 0
    for recipe in minute_recipes():
        target = d2_dir / f"{recipe.factor_id}.parquet"
        if resume and target.exists():
            _reconcile_one(recipe.factor_id, None)
            continue
        load = recipe.build(cfg, symbols, panel, logger)
        live = int(load.live_calls)
        live_total += live
        if live != 0:
            raise RuntimeError(
                f"{recipe.factor_id}: stk_mins_live_calls={live} != 0 — the "
                "reconciliation is cache-only by contract."
            )
        raw = load.factor[
            load.factor.index.get_level_values(DATE_LEVEL).isin(panel_dates)
        ].rename(recipe.factor_id)
        _reconcile_one(recipe.factor_id, raw)

    header = {
        "producing_git_sha": head_sha,
        "config": str(config_path),
        "baseline_root": str(root),
        "relative_tolerance": repr(RELATIVE_TOLERANCE),
        "stk_mins_live_calls": live_total,
        "panels": len(results),
        "all_ok": all(r.ok for r in results),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(
        json.dumps({"producing_git_sha": head_sha, "rows": d2_rows, "header": header},
                   indent=2, sort_keys=True),
        manifest_d2_path,
    )
    atomic_write_text(render_report(results, header), root / "reconcile_d2.md")
    logger.info("panel reconcile complete: all_ok=%s (%ss)",
                header["all_ok"], header["elapsed_seconds"])
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse already-written panels_d2 files (same-SHA guarded)",
    )
    args = parser.parse_args(argv)
    results = run_panel_reconcile(
        args.config, args.baseline_root, resume=args.resume
    )
    for r in results:
        print(
            f"{'OK      ' if r.ok else 'MISMATCH'} {r.factor_id}: rows={r.rows} "
            f"max_rel={r.max_rel_diff:.3e} max_abs={r.max_abs_diff:.3e} "
            f"nan_frozen_only={r.nan_only_in_frozen} nan_new_only={r.nan_only_in_new} "
            f"hash_equal={r.hashes_equal}"
        )
    ok = all(r.ok for r in results)
    print(f"reconcile: {'ALL OK' if ok else 'MISMATCH — stop and fix the engine'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
