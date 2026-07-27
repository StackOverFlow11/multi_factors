"""Unit tests for qt.factor_eval_reconcile (D5 C4, commit 3).

All tests are network-free and cache-free: the classification rules are pure
functions driven by synthetic frames/dicts, and the hard gate is exercised
against stub baselines. The real-cache orchestration (run_*_mode) is covered
by the real three-factor reconciliation run, not here.
"""

from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from qt.factor_eval_reconcile import (
    ReconciliationError,
    check_new_pair_consistency,
    classify_anchor_row,
    classify_panel_differences,
    diff_report_json,
    diff_report_md,
    frozen_panel_path,
    require_baseline_verified,
)

# --------------------------------------------------------------------------- #
# Hard gate
# --------------------------------------------------------------------------- #
def _stub_baseline(ok: int, problems: list[str], file_count: int = 77):
    return types.SimpleNamespace(
        verify_all=lambda: (ok, problems), file_count=file_count
    )


def test_hard_gate_passes_on_full_verification():
    require_baseline_verified(_stub_baseline(77, []))


def test_hard_gate_missing_baseline_bytes_is_a_hard_error_not_a_skip():
    with pytest.raises(ReconciliationError, match="did NOT verify"):
        require_baseline_verified(_stub_baseline(76, ["x.json: missing"]))


def test_hard_gate_raises_when_count_disagrees_without_problems():
    with pytest.raises(ReconciliationError):
        require_baseline_verified(_stub_baseline(76, []))


# --------------------------------------------------------------------------- #
# reports mode — JSON leaf diff
# --------------------------------------------------------------------------- #
def _frozen_like() -> dict:
    return {
        "schema_version": "0.1",
        "criteria_source": "default",
        "eval_config": {"universe": "000905.SH", "start": "2021-07-01"},
        "spec": {"factor_id": "volume_peak_count_20", "version": "1.0"},
        "verdict": {"verdict": "Watch"},
        "sections": [{"name": "ic", "payload": {"ic": 0.0177, "icir": 0.21}}],
    }


def _with_registered_additions(new: dict) -> dict:
    new["eval_config"].update(
        {"view": "decision", "return_basis": "exec_to_exec", "book_view": None}
    )
    new["eval_contract_version"] = "1.1"
    new["spec"].update(
        {
            "requires": "['PanelField(...)']",
            "adjustment": "returns_invariant",
            "overnight_boundary": "uses_prior_close",
            "lookback_depth": 20,
        }
    )
    return new


def test_json_registered_additions_only_passes():
    new = _with_registered_additions(_frozen_like())
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    assert {d.classification for d in result.diffs} == {"registered_addition"}


def test_json_unregistered_addition_fails():
    new = _with_registered_additions(_frozen_like())
    new["surprise"] = 1
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_addition")[0].path == "surprise"


def test_json_unregistered_removal_fails():
    new = _with_registered_additions(_frozen_like())
    del new["criteria_source"]
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_removal")[0].path == "criteria_source"


def test_json_numeric_change_beyond_tolerance_fails_when_strict():
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["ic"] = 0.0199
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_change")[0].path.endswith("ic")


def test_json_numeric_within_tolerance_passes_and_records_max():
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["ic"] = 0.0177 * (1 + 1e-12)
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    assert result.max_numeric_rel_diff > 0.0


def test_json_decision_book_numeric_drift_is_reported_not_gated():
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["ic"] = 0.0201  # the decision-view book moved it
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=False, correction_expected=False
    )
    assert result.ok
    assert result.by_class("book_view_effect")


def test_json_jump_correction_effect_accepted_only_with_corrections_block():
    old = _frozen_like()
    old["spec"]["factor_id"] = "jump_amount_corr_20"
    new = _with_registered_additions(_frozen_like())
    new["spec"]["factor_id"] = "jump_amount_corr_20"
    new["spec"]["version"] = "1.1"
    new["sections"][0]["payload"]["ic"] = -0.030539  # restated by the cutoff fix
    with_block = dict(new)
    with_block["corrections"] = [{"defect": "...", "to_version": "1.1"}]
    result = diff_report_json(
        old, with_block, name="t", strict=True, correction_expected=True
    )
    assert result.ok, result.diffs
    classes = {d.classification for d in result.diffs}
    assert "registered_correction_effect" in classes
    # WITHOUT the structured carrier the very same drift is unexplained -> fail.
    result_no_block = diff_report_json(
        old, new, name="t", strict=True, correction_expected=True
    )
    assert not result_no_block.ok


def test_json_spec_version_change_is_unregistered_without_correction():
    new = _with_registered_additions(_frozen_like())
    new["spec"]["version"] = "1.1"
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert any(
        d.path == "spec.version" and d.classification == "unregistered_change"
        for d in result.diffs
    )


def test_new_pair_eval_config_may_differ_only_in_book_view():
    no_book = {"eval_config": {"universe": "X", "book_view": None}}
    with_book = {"eval_config": {"universe": "X", "book_view": "decision"}}
    assert check_new_pair_consistency(no_book, with_book) == []
    with_book["eval_config"]["oos_split"] = "2025-01-01"
    problems = check_new_pair_consistency(no_book, with_book)
    assert len(problems) == 1
    assert problems[0].path == "eval_config.oos_split"


# --------------------------------------------------------------------------- #
# reports mode — Markdown line diff
# --------------------------------------------------------------------------- #
_MD_OLD = "# Factor Evaluation — x (v1.0)\n\n## 0. Header & Provenance\n\n- factor_id: x\n- family: microstructure\n"


def test_md_registered_provenance_lines_pass():
    new = _MD_OLD + "- evaluation contract: v1.1, ...\n- requires (endpoint inputs): a.b\n"
    result = diff_report_md(_MD_OLD, new, name="t", correction_expected=False)
    assert result.ok, result.diffs


def test_md_unregistered_new_line_fails():
    new = _MD_OLD + "- verdict secretly changed: Adopt\n"
    result = diff_report_md(_MD_OLD, new, name="t", correction_expected=False)
    assert not result.ok
    assert result.by_class("unregistered_addition")


def test_md_old_line_vanishing_fails():
    new = _MD_OLD.replace("- family: microstructure\n", "")
    result = diff_report_md(_MD_OLD, new, name="t", correction_expected=False)
    assert not result.ok
    assert result.by_class("unregistered_removal")


# --------------------------------------------------------------------------- #
# panels mode — cell classification
# --------------------------------------------------------------------------- #
def _frozen_panel(rows: list[tuple[str, str, float]], fid: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", fid])


def _new_series(rows: list[tuple[str, str, float]]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for d, s, _ in rows], names=["date", "symbol"]
    )
    return pd.Series([v for _, _, v in rows], index=idx)


NAN = float("nan")


def test_panels_identical_is_clean():
    frozen = _frozen_panel(
        [("2021-07-01", "A", NAN), ("2021-07-02", "A", 1.5), ("2021-07-02", "B", 2.0)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", NAN), ("2021-07-02", "A", 1.5), ("2021-07-02", "B", 2.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert result.ok, result.diffs
    assert result.equal == 1 and result.within_tolerance == 2


def test_panels_bounded_trim_fix_class_on_first_lookback_rows():
    frozen = _frozen_panel(
        [("2021-07-01", "SPARSE", NAN), ("2021-07-02", "SPARSE", NAN)], "f"
    )
    new = _new_series([("2021-07-01", "SPARSE", NAN), ("2021-07-02", "SPARSE", 0.7)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert result.ok, result.diffs
    assert len(result.by_class("per_symbol_trim_fix")) == 1


def test_panels_bounded_nan_to_finite_outside_first_rows_is_unclassified():
    frozen = _frozen_panel(
        [("2021-07-01", "A", NAN), ("2021-07-02", "A", NAN), ("2021-07-03", "A", NAN)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", NAN), ("2021-07-02", "A", NAN), ("2021-07-03", "A", 0.7)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert not result.ok
    assert len(result.by_class("unclassified_nan_to_finite")) == 1


def test_panels_pooled_saturation_class_with_monthly_decrease():
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", NAN), ("2021-07-02", "A", NAN),
            ("2021-08-02", "A", NAN), ("2021-10-29", "A", NAN),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "A", 1.0), ("2021-07-02", "A", 2.0),
            ("2021-08-02", "A", 3.0), ("2021-10-29", "A", NAN),
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.ok, result.diffs
    assert len(result.by_class("saturation_vs_anchor_truncation")) == 3
    assert result.saturation_by_month == {"2021-07": 2, "2021-08": 1}
    assert result.saturation_monotonic


def test_panels_pooled_nan_to_finite_after_early_region_is_unclassified():
    frozen = _frozen_panel([("2022-01-04", "A", NAN)], "f")
    new = _new_series([("2022-01-04", "A", 1.0)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert not result.ok
    assert len(result.by_class("unclassified_nan_to_finite")) == 1


def test_panels_saturation_non_monotonic_monthly_counts_fail():
    frozen = _frozen_panel(
        [("2021-07-01", "A", NAN), ("2021-08-02", "A", NAN), ("2021-08-03", "A", NAN)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 1.0), ("2021-08-02", "A", 2.0), ("2021-08-03", "A", 3.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert not result.ok  # 2021-08 (2) > 2021-07 (1) violates 按月递减至零
    assert not result.saturation_monotonic


def test_panels_frozen_finite_new_nan_never_allowed():
    frozen = _frozen_panel([("2021-07-01", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", NAN)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert not result.ok
    assert len(result.by_class("unclassified_frozen_finite_new_nan")) == 1


def test_panels_finite_vs_finite_beyond_tolerance_never_allowed():
    frozen = _frozen_panel([("2021-07-01", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", 1.0 + 1e-6)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1
    assert result.max_rel_diff > 0


def test_panels_extra_rows_allowed_only_as_nan_footprint():
    frozen = _frozen_panel([("2021-07-01", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", 1.0), ("2021-07-01", "NEWSYM", NAN)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert result.ok, result.diffs
    assert result.nan_footprint_rows == 1
    new_bad = _new_series([("2021-07-01", "A", 1.0), ("2021-07-01", "NEWSYM", 3.0)])
    result_bad = classify_panel_differences(
        new_bad, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert not result_bad.ok
    assert len(result_bad.by_class("unregistered_new_finite_row")) == 1


def test_frozen_panel_path_routes_jump_to_the_cutoff_reference():
    root = Path("/repo")
    jump = frozen_panel_path("jump_amount_corr_20", root)
    assert jump.parent.name == "panels" and jump.parent.parent.name == "pr_c_cutoff_fix"
    other = frozen_panel_path("volume_peak_count_20", root)
    assert other.parent.parent.name == "refactor_baseline" and "pr_c_cutoff_fix" not in str(other)


# --------------------------------------------------------------------------- #
# anchors mode — row classification
# --------------------------------------------------------------------------- #
def test_anchor_jump_mismatch_fails_even_in_early_region():
    row = classify_anchor_row(
        factor_id="jump_amount_corr_20", cls="warmup_end",
        date=pd.Timestamp("2021-07-02"), symbol="000537.SZ",
        hand=0.46, service=0.47, is_pooled=False, tol=1e-12,
    )
    assert row.classification == "failed"


def test_anchor_jump_match_is_ok():
    row = classify_anchor_row(
        factor_id="jump_amount_corr_20", cls="random",
        date=pd.Timestamp("2024-05-30"), symbol="002690.SZ",
        hand=0.3831735019222186, service=0.3831735019222186, is_pooled=False, tol=1e-12,
    )
    assert row.classification == "ok" and row.rel_diff == 0.0


def test_anchor_pooled_early_mismatch_is_the_saturation_class():
    row = classify_anchor_row(
        factor_id="volume_peak_count_20", cls="warmup_end",
        date=pd.Timestamp("2021-07-28"), symbol="600827.SH",
        hand=150.0, service=151.0, is_pooled=True, tol=1e-12,
    )
    assert row.classification == "saturation_vs_anchor_truncation"


def test_anchor_pooled_late_mismatch_fails():
    row = classify_anchor_row(
        factor_id="volume_peak_count_20", cls="random",
        date=pd.Timestamp("2023-05-18"), symbol="600867.SH",
        hand=1.0, service=1.1, is_pooled=True, tol=1e-12,
    )
    assert row.classification == "failed"
