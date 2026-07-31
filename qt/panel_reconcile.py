"""D2 cell-by-cell panel reconciliation — VERIFY-ONLY since D5 C6.

D2 rewrote the factor math into ``factors.compute.minute``. To show the rewrite
changed no value, all 14 closing factors' RAW panels were rebuilt on the D2 tree
into ``artifacts/refactor_baseline/panels_d2`` and compared CELL BY CELL against
the D1 frozen baseline:

* the (date, symbol) index sets must be IDENTICAL;
* the NaN sets must be IDENTICAL (a NaN-set change would be an undeclared
  NaN-policy change — D2 declares none);
* on the jointly-finite cells the relative difference
  ``|new - old| / max(|old|, |new|)`` must be <= 1e-12 (the float-reordering
  budget; any excess means STOP AND FIX THE ENGINE, never widen the budget).

It passed 14/14 at ``max_rel_diff = 0.0`` — bit-identical, canonical hashes
equal. Both sides are now frozen.

REGENERATION IS RETIRED (owner ruling 2026-07-28, D5 C6)
---------------------------------------------------------
The rebuild ran the ``qt.panel_freeze`` recipes, i.e. the eleven legacy eval
runners' private ``_load_*_panel`` loaders, which C6 deletes. Retired rather
than left to break on first use.

What survives is the part that never needed the runners: **the comparison
itself**. ``--verify`` re-derives the D2 verdict from the frozen bytes of both
sides — it re-reads the two panel sets from disk and re-runs
:func:`compare_panels` over all 14 factors. That is a real check, not a
restatement of a recorded one: if either frozen tree drifted, the cells no
longer match and the run goes red.

Anti-empty-reconciliation provenance (the ``compare_postmerge.py`` lesson):

* neither side is ever regenerated here;
* both sides are read from disk through two INDEPENDENT file reads, so no
  shared in-memory object can make the equality vacuous;
* before any cell is compared, both sides are checked against manifests — the
  D2 panels against the git-tracked D1 document (they are expected to be
  bit-identical to it, which IS the D2 result) and against their own
  ``manifest_d2.json``.

Run (deliberately NOT registered in qt/cli, as before)::

    python -m qt.panel_reconcile --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qt.frozen_manifest_doc import parse_doc_manifest
from qt.panel_freeze import (
    D1_MANIFEST_DOC,
    DEFAULT_CONFIG,
    PANEL_STORE_NAME,
    RegenerationRetiredError,
    canonical_content_hash,
    read_frozen_panel,
    repo_root_for,
    retirement_message,
    verify_frozen_panels,
)

__all__ = [
    "CellComparison",
    "D2_SUBDIR",
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_CONFIG",
    "PANEL_STORE_NAME",
    "RELATIVE_TOLERANCE",
    "ReconcileVerification",
    "compare_panels",
    "main",
    "render_report",
    "run_panel_reconcile",
    "verify_d2_reconciliation",
]

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


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReconcileVerification:
    """Outcome of re-deriving the D2 verdict from the two frozen panel sets."""

    baseline_root: Path
    comparisons: tuple[CellComparison, ...]
    problems: tuple[str, ...]
    d1_ok: bool
    recorded_all_ok: bool | None

    @property
    def ok(self) -> bool:
        return (
            self.d1_ok
            and not self.problems
            and bool(self.comparisons)
            and all(c.ok for c in self.comparisons)
        )


def verify_d2_reconciliation(
    baseline_root: Path | str = DEFAULT_BASELINE_ROOT,
    doc_path: Path | str | None = None,
) -> ReconcileVerification:
    """Re-derive the D2 cell-by-cell verdict from the frozen bytes of both sides.

    Order matters. The D1 side is verified FIRST, against the git-tracked
    document (delegated to :func:`qt.panel_freeze.verify_frozen_panels`, so
    there is one implementation of "is this the frozen baseline?"). Comparing
    two panel sets while neither has been authenticated would report agreement
    between two unknowns.

    The D2 side is then checked against the SAME git-tracked hashes. That is not
    a shortcut: "bit-identical to D1" is precisely what the D2 reconciliation
    concluded, so the D1 table IS the D2 expectation, and it is the only one of
    the two that lives in git.

    ``doc_path`` overrides which manifest document supplies the expectations;
    it exists so the tests can build a small frozen tree with its own document
    instead of being forced through the 14 real factors.
    """
    baseline_root = Path(baseline_root)
    doc = Path(doc_path) if doc_path is not None else _doc_path(baseline_root)
    problems: list[str] = []

    d1 = verify_frozen_panels(baseline_root, doc)
    for problem in d1.problems:
        problems.append(f"D1 side: {problem}")
    for panel in d1.panels:
        for problem in panel.problems:
            problems.append(f"D1 side {panel.factor_id}: {problem}")

    expected = parse_doc_manifest(doc)
    d2_dir = baseline_root / D2_SUBDIR
    if not d2_dir.is_dir():
        problems.append(f"D2 panel directory not found: {d2_dir}")
        return ReconcileVerification(
            baseline_root=baseline_root,
            comparisons=(),
            problems=tuple(problems),
            d1_ok=d1.ok,
            recorded_all_ok=None,
        )

    on_disk = {path.stem for path in d2_dir.glob("*.parquet")}
    for extra in sorted(on_disk - set(expected)):
        problems.append(f"unregistered D2 panel on disk: {extra}.parquet")

    recorded_rows, recorded_all_ok = _read_manifest_d2(d2_dir.parent, problems)

    comparisons: list[CellComparison] = []
    for factor_id in sorted(expected):
        d2_path = d2_dir / f"{factor_id}.parquet"
        d1_path = baseline_root / "panels" / f"{factor_id}.parquet"
        if not d2_path.exists() or not d1_path.exists():
            problems.append(
                f"{factor_id}: cannot compare — missing "
                f"{'D2' if not d2_path.exists() else 'D1'} panel on disk."
            )
            continue
        # TWO independent file reads; no shared in-memory object.
        new = read_frozen_panel(d2_path, factor_id)
        frozen = read_frozen_panel(d1_path, factor_id)
        comparison = compare_panels(frozen, new, factor_id)
        comparisons.append(comparison)

        d2_hash = canonical_content_hash(new)
        if d2_hash != expected[factor_id].canonical_sha256:
            problems.append(
                f"{factor_id}: D2 panel canonical hash {d2_hash} != the git-tracked "
                f"{expected[factor_id].canonical_sha256}. D2 concluded the rewrite "
                "was bit-identical, so these must agree."
            )
        recorded = recorded_rows.get(factor_id)
        if recorded is None:
            problems.append(f"{factor_id}: no row in manifest_d2.json")
        else:
            if recorded.get("canonical_sha256") != d2_hash:
                problems.append(
                    f"{factor_id}: manifest_d2.json records canonical "
                    f"{recorded.get('canonical_sha256')}, panel gives {d2_hash}"
                )
            if recorded.get("rows") != comparison.rows:
                problems.append(
                    f"{factor_id}: manifest_d2.json records rows "
                    f"{recorded.get('rows')!r}, panel has {comparison.rows}"
                )

    if recorded_all_ok is False:
        problems.append(
            "manifest_d2.json records all_ok=False — the frozen D2 reconciliation "
            "did not pass, and nothing downstream may treat it as if it had."
        )
    return ReconcileVerification(
        baseline_root=baseline_root,
        comparisons=tuple(comparisons),
        problems=tuple(problems),
        d1_ok=d1.ok,
        recorded_all_ok=recorded_all_ok,
    )


def _doc_path(baseline_root: Path) -> Path:
    return repo_root_for(baseline_root) / D1_MANIFEST_DOC


def _read_manifest_d2(
    root: Path, problems: list[str]
) -> tuple[dict[str, dict], bool | None]:
    path = root / "manifest_d2.json"
    if not path.exists():
        problems.append(f"D2 manifest missing: {path}")
        return {}, None
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["factor_id"]): row for row in document.get("rows", [])}
    return rows, document.get("header", {}).get("all_ok")


# --------------------------------------------------------------------------- #
# Retired regeneration entry point
# --------------------------------------------------------------------------- #
def run_panel_reconcile(*args, **kwargs):
    """RETIRED (D5 C6). Always raises :class:`RegenerationRetiredError`.

    Kept as a stub so a stale caller gets the retirement explanation rather than
    an ``AttributeError`` that reads like a packaging accident.
    """
    raise RegenerationRetiredError(
        retirement_message(
            "python -m qt.panel_reconcile",
            "the D2 rebuilt panel set (`panels_d2`), which is frozen-forever",
            "The rebuild ran the qt.panel_freeze recipes, i.e. the eleven legacy "
            "eval runners' private `_load_*_panel` loaders, which C6 deletes.",
            "the COMPARISON never needed those loaders, so it still runs — it "
            "re-reads both frozen panel sets from disk and re-derives the "
            "cell-by-cell D2 verdict.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m qt.panel_reconcile --verify``.

    ``--resume`` is still accepted by the parser so the old documented command
    produces the retirement explanation rather than ``unrecognized arguments``,
    which reads like a version mismatch.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--doc",
        default=None,
        help="manifest document supplying the expected hashes "
        f"(default: {D1_MANIFEST_DOC})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive the D2 cell-by-cell verdict from the frozen bytes",
    )
    args = parser.parse_args(argv)

    if not args.verify:
        try:
            run_panel_reconcile()
        except RegenerationRetiredError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = verify_d2_reconciliation(args.baseline_root, args.doc)
    for comparison in result.comparisons:
        print(
            f"{'OK      ' if comparison.ok else 'MISMATCH'} {comparison.factor_id}: "
            f"rows={comparison.rows} max_rel={comparison.max_rel_diff:.3e} "
            f"max_abs={comparison.max_abs_diff:.3e} "
            f"nan_frozen_only={comparison.nan_only_in_frozen} "
            f"nan_new_only={comparison.nan_only_in_new} "
            f"hash_equal={comparison.hashes_equal}"
        )
    for problem in result.problems:
        print(f"  PROBLEM {problem}")
    print(
        f"D1 side verified: {result.d1_ok}; recorded all_ok: {result.recorded_all_ok}; "
        f"{len(result.comparisons)} factors re-compared"
    )
    print(
        "panel reconcile verification: "
        + ("OK" if result.ok else "FAILED — stop and investigate the frozen trees")
    )
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
