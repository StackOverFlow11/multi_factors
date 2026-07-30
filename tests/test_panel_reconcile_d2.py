"""Teeth tests for the D2 panel reconciliation (network-free, VERIFY-ONLY era).

Two halves:

* the cell-by-cell comparator — a reconciliation whose comparator cannot fail is
  the ``compare_postmerge.py`` failure mode, so it is fed engineered defects and
  must CONVICT each one, while the one legitimate difference class (float
  reordering within 1e-12) passes without being confused with hash equality;
* the verify-only entry point, which re-derives the frozen D2 verdict from the
  two frozen panel sets. Its tests always establish a GREEN tree first, so a red
  result is attributable to the tampering rather than to a comparator that is
  red on everything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.panel_freeze import RegenerationRetiredError, canonical_content_hash
from qt.panel_reconcile import (
    RELATIVE_TOLERANCE,
    compare_panels,
    main,
    run_panel_reconcile,
    verify_d2_reconciliation,
)
from tests.fixtures.frozen_baseline import (
    build_frozen_tree,
    make_panel,
    patch_manifest,
    rewrite_panel,
)


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


# --------------------------------------------------------------------------- #
# Retired regeneration entry point
# --------------------------------------------------------------------------- #
def test_the_rebuild_raises_instead_of_quietly_doing_nothing():
    with pytest.raises(RegenerationRetiredError, match="RETIRED"):
        run_panel_reconcile()


def test_the_retirement_message_says_the_comparison_survives():
    """The three retired tools do NOT verify the same kind of thing, so each
    states its own. Claiming a check it does not perform is the defect this
    repository keeps catching."""
    with pytest.raises(RegenerationRetiredError) as caught:
        run_panel_reconcile()
    text = str(caught.value)
    assert "python -m qt.panel_reconcile --verify" in text
    assert "COMPARISON never needed those loaders" in text


def test_the_old_resume_command_still_parses_so_it_gets_the_explanation(capsys):
    assert main(["--resume"]) == 1
    assert "RETIRED" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Verification: re-deriving the frozen D2 verdict
# --------------------------------------------------------------------------- #
def test_two_intact_frozen_sides_re_derive_the_passing_verdict(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    result = verify_d2_reconciliation(tmp_path, doc)
    assert result.ok and result.d1_ok and result.recorded_all_ok is True
    assert len(result.comparisons) == 2
    assert all(c.max_rel_diff == 0.0 and c.hashes_equal for c in result.comparisons)


def test_a_drifted_d2_panel_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    assert verify_d2_reconciliation(tmp_path, doc).ok  # green control

    panel = make_panel("alpha_20")
    panel.iloc[0] = float(panel.iloc[0]) + 1.0
    rewrite_panel(tmp_path, "alpha_20", panel, d2=True)

    result = verify_d2_reconciliation(tmp_path, doc)
    assert not result.ok
    convicted = [c for c in result.comparisons if not c.ok]
    assert [c.factor_id for c in convicted] == ["alpha_20"]
    assert any("D2 panel canonical hash" in p for p in result.problems)


def test_a_drifted_d1_side_is_convicted_before_the_cells_are_believed(tmp_path: Path):
    """Comparing two panel sets while neither has been authenticated would report
    agreement between two unknowns — so a D1 side that no longer matches git
    fails the run even though the two sides still agree with EACH OTHER."""
    doc = build_frozen_tree(tmp_path, with_d2=True)
    panel = make_panel("alpha_20")
    panel.iloc[0] = float(panel.iloc[0]) + 1.0
    rewrite_panel(tmp_path, "alpha_20", panel)
    rewrite_panel(tmp_path, "alpha_20", panel, d2=True)

    result = verify_d2_reconciliation(tmp_path, doc)
    assert all(c.ok for c in result.comparisons)  # the two sides DO agree
    assert not result.d1_ok and not result.ok
    assert any(p.startswith("D1 side") for p in result.problems)


def test_a_nan_set_change_in_the_d2_side_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    panel = make_panel("alpha_20")
    panel.iloc[1] = np.nan
    rewrite_panel(tmp_path, "alpha_20", panel, d2=True)
    result = verify_d2_reconciliation(tmp_path, doc)
    convicted = [c for c in result.comparisons if not c.ok]
    assert convicted and convicted[0].nan_only_in_new == 1
    assert not result.ok


def test_a_missing_d2_manifest_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    (tmp_path / "manifest_d2.json").unlink()
    result = verify_d2_reconciliation(tmp_path, doc)
    assert not result.ok
    assert any("D2 manifest missing" in p for p in result.problems)


def test_a_recorded_failure_is_never_treated_as_a_pass(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    patch_manifest(tmp_path, "manifest_d2.json", lambda d: d["header"].update(all_ok=False))
    result = verify_d2_reconciliation(tmp_path, doc)
    assert not result.ok
    assert any("all_ok=False" in p for p in result.problems)


def test_a_d2_manifest_row_that_disagrees_with_its_panel_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)

    def _lie(document):
        for row in document["rows"]:
            if row["factor_id"] == "alpha_20":
                row["canonical_sha256"] = "f" * 64

    patch_manifest(tmp_path, "manifest_d2.json", _lie)
    result = verify_d2_reconciliation(tmp_path, doc)
    assert not result.ok
    assert any("manifest_d2.json records canonical" in p for p in result.problems)


def test_an_absent_d2_directory_cannot_pass_by_having_nothing_to_compare(tmp_path: Path):
    """Zero comparisons is not a passing reconciliation."""
    doc = build_frozen_tree(tmp_path, with_d2=False)
    result = verify_d2_reconciliation(tmp_path, doc)
    assert not result.ok and result.comparisons == ()


def test_verify_writes_nothing_into_the_frozen_trees(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert verify_d2_reconciliation(tmp_path, doc).ok
    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert after == before


def test_the_two_sides_are_read_from_disk_independently(tmp_path: Path):
    """Both sides come from their own file read; no shared in-memory object can
    make the equality vacuous. Shown by making the two files differ: if one read
    were reused for both, this could not be detected."""
    doc = build_frozen_tree(tmp_path, with_d2=True)
    panel = make_panel("book_x", offset=1.0)
    panel.iloc[2] = float(panel.iloc[2]) + 7.0
    rewrite_panel(tmp_path, "book_x", panel, d2=True)
    assert canonical_content_hash(panel) != canonical_content_hash(make_panel("book_x", offset=1.0))

    result = verify_d2_reconciliation(tmp_path, doc)
    convicted = [c for c in result.comparisons if not c.ok]
    assert [c.factor_id for c in convicted] == ["book_x"]
