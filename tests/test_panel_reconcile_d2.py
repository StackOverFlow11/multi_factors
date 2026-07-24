"""Teeth tests for the D2 cell-by-cell panel comparator (network-free).

A reconciliation whose comparator cannot fail is the ``compare_postmerge.py``
failure mode; these tests feed the comparator engineered defects and assert it
CONVICTS each one — and that the one legitimate difference class (float
reordering within 1e-12) passes without being confused with hash equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qt.panel_reconcile import RELATIVE_TOLERANCE, compare_panels


def _series(values, dates=None, symbols=("AAA", "BBB")):
    dates = dates or ["2024-01-02", "2024-01-03"]
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), list(symbols)], names=["date", "symbol"]
    )
    return pd.Series(list(values), index=index, dtype=float, name="f")


def test_identical_panels_reconcile_exactly():
    a = _series([1.0, 2.0, np.nan, 4.0])
    comp = compare_panels(a, a.copy(), "f")
    assert comp.ok and comp.max_rel_diff == 0.0 and comp.hashes_equal
    assert comp.nan_only_in_frozen == 0 and comp.nan_only_in_new == 0


def test_float_reordering_within_budget_passes_but_hash_differs():
    a = _series([1.0, 2.0, np.nan, 4.0])
    b = a.copy()
    b.iloc[0] = 1.0 * (1.0 + 1e-14)  # sub-budget drift (legit reordering scale)
    comp = compare_panels(a, b, "f")
    assert comp.ok  # within the 1e-12 budget
    assert comp.max_rel_diff > 0.0
    assert not comp.hashes_equal  # the hash still SEES it (no false comfort)


def test_value_drift_beyond_budget_is_convicted():
    a = _series([1.0, 2.0, np.nan, 4.0])
    b = a.copy()
    b.iloc[3] = 4.0 * (1.0 + 1e-9)
    comp = compare_panels(a, b, "f")
    assert not comp.ok
    assert comp.n_cells_beyond_tol == 1
    assert comp.max_rel_diff > RELATIVE_TOLERANCE


def test_nan_set_change_is_convicted_in_both_directions():
    a = _series([1.0, 2.0, np.nan, 4.0])
    b = _series([1.0, np.nan, np.nan, 4.0])
    comp = compare_panels(a, b, "f")
    assert not comp.ok and comp.nan_only_in_new == 1 and comp.nan_only_in_frozen == 0
    comp_rev = compare_panels(b, a, "f")
    assert not comp_rev.ok and comp_rev.nan_only_in_frozen == 1


def test_index_mismatch_is_convicted_before_any_value_math():
    a = _series([1.0, 2.0, 3.0, 4.0])
    b = _series([1.0, 2.0, 3.0, 4.0], dates=["2024-01-02", "2024-01-04"])
    comp = compare_panels(a, b, "f")
    assert not comp.ok and not comp.index_equal


def test_sign_flip_at_zero_magnitude_is_within_denominator_rule():
    # denominator = max(|a|,|b|): a 0.0 vs 0.0 cell contributes rel 0 (0/0 -> 0),
    # while 0.0 vs 1e-30 is rel 1.0 and convicted — tiny absolute fabrications
    # near zero cannot hide behind a relative rule.
    a = _series([0.0, 2.0, 3.0, 4.0])
    b = a.copy()
    comp = compare_panels(a, b, "f")
    assert comp.ok
    b2 = a.copy()
    b2.iloc[0] = 1e-30
    comp2 = compare_panels(a, b2, "f")
    assert not comp2.ok and comp2.max_rel_diff == pytest.approx(1.0)
