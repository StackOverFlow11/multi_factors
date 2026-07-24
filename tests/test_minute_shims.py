"""D2 shim discipline tests (design v3.2 §6.4 / R14).

Two independent locks:

1. **Purity** — each FULLY-migrated module (the 10 pure ``data/clean``
   intraday factor modules + ``factors/compute/intraday_derived.py``) must
   contain ONLY a module docstring, ``from ... import ...`` statements and an
   ``__all__`` assignment. Any function/class/constant defined in a shim would
   be a resurrected second definition point — exactly the drift §6.4 forbids.
   R14 scope note: ``intraday_schema`` and the ``intraday_aggregate`` generic
   core keep REAL code and are deliberately NOT purity-tested (mixed modules);
   testing them would force the implementer to quietly relax the check.

2. **Re-export identity** — every public name a shim re-exports must be the
   SAME OBJECT as the one defined in ``factors.compute.minute`` (``is``, not
   ``==``). This is what makes "the pre-existing property tests exercise the
   NEW engine" a proven fact rather than an assumption: the shim path and the
   new path cannot diverge because there is only one object.

Mutation evidence (run for this commit, recorded in the acceptance report):
adding a stray ``def _local(): pass`` to ``data/clean/intraday_amplitude.py``
makes ``test_fully_migrated_shims_are_import_only`` fail (rc=1); reverting it
passes (rc=0). Rebinding a shim name to a wrapper function makes
``test_shim_reexports_are_identical_objects`` fail.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# The FULLY-migrated modules (R14: the only ones the purity test may cover).
_PURE_SHIMS = [
    "data/clean/intraday_amount_ratio.py",
    "data/clean/intraday_amp_anomaly.py",
    "data/clean/intraday_amp_cut.py",
    "data/clean/intraday_amplitude.py",
    "data/clean/intraday_peak_interval.py",
    "data/clean/intraday_ridge_return.py",
    "data/clean/intraday_valley_quantile.py",
    "data/clean/intraday_valley_ridge_vwap.py",
    "data/clean/intraday_valley_vwap.py",
    "data/clean/intraday_volume_prv.py",
    "factors/compute/intraday_derived.py",
]

# shim module -> the factors.compute.minute module(s) its names must resolve to.
_SHIM_SOURCES = {
    "data.clean.intraday_amount_ratio": ["factors.compute.minute.peak_ridge_amount_ratio"],
    "data.clean.intraday_amp_anomaly": ["factors.compute.minute.amp_marginal_anomaly_vol"],
    "data.clean.intraday_amp_cut": ["factors.compute.minute.intraday_amp_cut"],
    "data.clean.intraday_amplitude": ["factors.compute.minute.minute_ideal_amplitude"],
    "data.clean.intraday_peak_interval": ["factors.compute.minute.peak_interval_kurtosis"],
    "data.clean.intraday_ridge_return": ["factors.compute.minute.ridge_minute_return"],
    "data.clean.intraday_valley_quantile": ["factors.compute.minute.valley_price_quantile"],
    "data.clean.intraday_valley_ridge_vwap": [
        "factors.compute.minute.valley_ridge_vwap_ratio"
    ],
    "data.clean.intraday_valley_vwap": ["factors.compute.minute.valley_relative_vwap"],
    "data.clean.intraday_volume_prv": [
        "factors.compute.minute.volume_peak_count",
        "factors.compute.minute.primitives",
    ],
    "factors.compute.intraday_derived": [
        "factors.compute.minute.amp_marginal_anomaly_vol",
        "factors.compute.minute.intraday_amp_cut",
        "factors.compute.minute.jump_amount_corr",
        "factors.compute.minute.minute_ideal_amplitude",
        "factors.compute.minute.peak_interval_kurtosis",
        "factors.compute.minute.peak_ridge_amount_ratio",
        "factors.compute.minute.ridge_minute_return",
        "factors.compute.minute.valley_price_quantile",
        "factors.compute.minute.valley_relative_vwap",
        "factors.compute.minute.valley_ridge_vwap_ratio",
        "factors.compute.minute.volume_peak_count",
    ],
}

# Factor math re-exported by the MIXED aggregate module (kept real code; its
# re-exports must still be identity-equal to the migrated definitions).
_AGGREGATE_REEXPORTS = {
    "compute_jump_amount_corr": "factors.compute.minute.jump_amount_corr",
    "JUMP_LOOKBACK_DAYS": "factors.compute.minute.jump_amount_corr",
    "JUMP_MIN_PAIRS": "factors.compute.minute.jump_amount_corr",
    "JUMP_Z": "factors.compute.minute.jump_amount_corr",
    "compute_minute_mmp": "factors.compute.minute.mmp",
    "mmp_valid_minute_counts": "factors.compute.minute.mmp",
    "MMP_LOOKBACK": "factors.compute.minute.mmp",
    "DEFAULT_EPSILON": "factors.compute.minute.mmp",
}


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _is_all_assignment(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )


@pytest.mark.parametrize("rel_path", _PURE_SHIMS)
def test_fully_migrated_shims_are_import_only(rel_path):
    """Every fully-migrated module is docstring + imports + ``__all__`` — nothing else."""
    tree = ast.parse((_REPO / rel_path).read_text(encoding="utf-8"))
    for i, node in enumerate(tree.body):
        if i == 0 and _is_docstring(node):
            continue
        if isinstance(node, ast.ImportFrom):
            continue
        if _is_all_assignment(node):
            continue
        raise AssertionError(
            f"{rel_path} is a D2 re-export shim but contains a "
            f"{type(node).__name__} at line {node.lineno} — shims may only "
            f"import and declare __all__ (design v3.2 §6.4)."
        )


@pytest.mark.parametrize("shim_name", sorted(_SHIM_SOURCES))
def test_shim_reexports_are_identical_objects(shim_name):
    """Each shim ``__all__`` name IS the object defined in factors.compute.minute."""
    shim = importlib.import_module(shim_name)
    sources = [importlib.import_module(m) for m in _SHIM_SOURCES[shim_name]]
    assert shim.__all__, f"{shim_name} must declare a non-empty __all__"
    for name in shim.__all__:
        obj = getattr(shim, name)
        homes = [src for src in sources if getattr(src, name, None) is obj]
        assert homes, (
            f"{shim_name}.{name} is not identical to any declared source object — "
            f"the shim has drifted from the single definition point."
        )


def test_aggregate_reexports_are_identical_objects():
    """The MIXED aggregate module's factor-math re-exports point at the new homes."""
    agg = importlib.import_module("data.clean.intraday_aggregate")
    for name, src_name in _AGGREGATE_REEXPORTS.items():
        src = importlib.import_module(src_name)
        assert getattr(agg, name) is getattr(src, name), (
            f"data.clean.intraday_aggregate.{name} is not the object defined in "
            f"{src_name} — the re-export has drifted."
        )


def test_purity_scope_is_exactly_the_fully_migratable_set():
    """R14: the purity list covers the 10 pure data/clean factor modules + the
    surface shim, and deliberately EXCLUDES the mixed/kept-real modules."""
    covered = set(_PURE_SHIMS)
    assert "data/clean/intraday_schema.py" not in covered
    assert "data/clean/intraday_aggregate.py" not in covered
    assert len(covered) == 11
