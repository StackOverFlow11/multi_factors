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


def test_json_numeric_change_on_aggregate_path_is_the_warmup_class():
    # Aggregate metrics legitimately move: they aggregate a panel whose
    # warmup cells moved (the panels leg is the value gate). Registered as
    # warmup_aggregate_effect, REPORTED not gated.
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["ic"] = 0.0199
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    assert result.by_class("warmup_aggregate_effect")[0].path.endswith("ic")


def test_json_numeric_change_outside_aggregate_paths_fails():
    new = _with_registered_additions(_frozen_like())
    new["n_periods"] = 999  # top-level numeric: not an aggregate path
    old = _frozen_like()
    old["n_periods"] = 1205
    result = diff_report_json(
        old, new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_change")[0].path == "n_periods"


def test_json_verdict_label_flip_carries_no_digit_and_fails():
    new = _with_registered_additions(_frozen_like())
    new["verdict"]["verdict"] = "Adopt"  # pure label flip, no digits
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_change")[0].path == "verdict.verdict"


def test_json_requires_list_leaves_are_registered_by_prefix():
    new = _with_registered_additions(_frozen_like())
    new["spec"]["requires"] = ["PanelField(field='high')", "PanelField(field='low')"]
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    paths = {d.path for d in result.by_class("registered_addition")}
    assert "spec.requires[0]" in paths and "spec.requires[1]" in paths


def test_json_sanity_report_stem_rename_is_registered():
    old = _frozen_like()
    old["sections"][0]["payload"]["sanity_report"] = (
        "artifacts/reports/eval_x_exec_basis_sanity.md"
    )
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["sanity_report"] = (
        "artifacts/reports/factor_eval_x_20_exec_basis_sanity.md"
    )
    result = diff_report_json(
        old, new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    assert result.by_class("registered_sanity_stem_rename")


def test_json_exec_price_artifact_reused_false_to_true_is_registered():
    old = _frozen_like()
    old["sections"][0]["payload"]["exec_price_artifact_reused"] = False
    new = _with_registered_additions(_frozen_like())
    new["sections"][0]["payload"]["exec_price_artifact_reused"] = True
    result = diff_report_json(
        old, new, name="t", strict=True, correction_expected=False
    )
    assert result.ok, result.diffs
    assert result.by_class("registered_run_order_artifact")
    # The REVERSE direction is not a run-order artifact: it must fail.
    result_rev = diff_report_json(
        new, old, name="t", strict=True, correction_expected=False
    )
    assert not result_rev.ok


def test_json_jump_description_and_factor_version_are_correction_effects():
    old = _frozen_like()
    old["spec"]["factor_id"] = "jump_amount_corr_20"
    old["spec"]["description"] = "old (pre-cutoff) description"
    old["sections"].append({"name": "provenance", "payload": {"factor_version": "1.0"}})
    new = _with_registered_additions(_frozen_like())
    new["spec"]["factor_id"] = "jump_amount_corr_20"
    new["spec"]["description"] = "new description with the 14:50 truncation"
    new["sections"].append({"name": "provenance", "payload": {"factor_version": "1.1"}})
    new["corrections"] = [{"defect": "...", "to_version": "1.1"}]
    result = diff_report_json(
        old, new, name="t", strict=True, correction_expected=True
    )
    assert result.ok, result.diffs
    changed = {d.path for d in result.by_class("registered_correction_effect")}
    assert "spec.description" in changed
    assert "sections[1].payload.factor_version" in changed


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


def test_md_same_key_value_change_pairs_into_a_change_not_a_removal():
    old = _MD_OLD + "- ic_mean: 0.018587\n- settled_rebalances: 1190\n"
    new = _MD_OLD + "- ic_mean: 0.018475\n- settled_rebalances: 1209\n"
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert result.ok, result.diffs
    # No phantom removal/addition pairs: both rows are single CHANGES.
    assert len(result.diffs) == 2
    assert {d.classification for d in result.diffs} == {"warmup_aggregate_effect"}
    assert all(d.old is not None and d.new is not None for d in result.diffs)


def test_md_paired_label_flip_without_digits_fails():
    old = _MD_OLD + "- verdict: Watch\n"
    new = _MD_OLD + "- verdict: Adopt\n"
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert not result.ok
    assert result.by_class("unregistered_change")


def test_md_paired_change_in_decision_book_is_book_view_effect():
    old = _MD_OLD + "- incremental_ic_ir: 0.120357\n"
    new = _MD_OLD + "- incremental_ic_ir: 0.131000\n"
    result = diff_report_md(old, new, name="t", strict=False, correction_expected=False)
    assert result.ok
    assert result.by_class("book_view_effect")


def test_md_correction_expected_turns_paired_changes_into_correction_effects():
    old = _MD_OLD + "- ic_mean: -0.030840\n"
    new = _MD_OLD + "- ic_mean: -0.030539\n"
    result = diff_report_md(old, new, name="t", correction_expected=True)
    assert result.ok
    assert result.by_class("registered_correction_effect")


def test_md_sanity_report_line_rename_is_registered():
    old = _MD_OLD + "- sanity_report: artifacts/reports/eval_x_exec_basis_sanity.md\n"
    new = _MD_OLD + "- sanity_report: artifacts/reports/factor_eval_x_20_exec_basis_sanity.md\n"
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert result.ok, result.diffs
    assert result.by_class("registered_sanity_stem_rename")


def test_md_prose_line_with_number_before_the_colon_pairs_via_normalized_key():
    # The changed number PRECEDES the first colon: the exact head can never
    # pair these. The digit-normalized fallback pairs them (real instance:
    # the with_book incremental-FAIL bullet in minute_ideal_amp/volume_peak).
    old = (
        _MD_OLD
        + "- [incremental FAIL] orthogonalized ICIR +0.120 is ~ 0 (redundant with the book):"
          " after residualizing on the known-factor book the factor adds no signal.\n"
    )
    new = (
        _MD_OLD
        + "- [incremental FAIL] orthogonalized ICIR +0.113 is ~ 0 (redundant with the book):"
          " after residualizing on the known-factor book the factor adds no signal.\n"
    )
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert result.ok, result.diffs
    assert len(result.diffs) == 1
    assert result.diffs[0].classification == "warmup_aggregate_effect"
    assert result.diffs[0].old is not None and result.diffs[0].new is not None


def test_md_long_prose_head_within_cap_still_pairs():
    # The incremental-axis reason line is ~103 chars before its first colon —
    # it must remain PAIRABLE (the cap exists only for pathological lines).
    old = (
        _MD_OLD
        + "- [incremental PASS] orthogonalized ICIR lower CI bound (N_eff-based,"
          " expected direction) +0.198 > 0.15: the factor convincingly adds a signal.\n"
    )
    new = (
        _MD_OLD
        + "- [incremental PASS] orthogonalized ICIR lower CI bound (N_eff-based,"
          " expected direction) +0.195 > 0.15: the factor convincingly adds a signal.\n"
    )
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert result.ok, result.diffs
    assert len(result.diffs) == 1
    assert result.diffs[0].classification == "warmup_aggregate_effect"


def test_md_prose_WORD_change_does_not_pair_and_fails():
    old = (
        _MD_OLD
        + "- [incremental FAIL] orthogonalized ICIR +0.120 is ~ 0 (redundant with the book):"
          " after residualizing the factor adds no signal.\n"
    )
    new = (
        _MD_OLD
        + "- [incremental FAIL] orthogonalized ICIR +0.113 is clearly above (redundant with the book):"
          " after residualizing the factor adds no signal.\n"
    )
    result = diff_report_md(old, new, name="t", correction_expected=False)
    assert not result.ok
    assert result.by_class("unregistered_addition")
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


def test_panels_bounded_warmup_class_on_first_lookback_minus_1_rows():
    # lookback_depth=3 -> the first w-1 = 2 trading dates are the warmup
    # boundary. A NaN -> finite cell on the 2nd date is the left-extension
    # warmup effect (the old runner's partial pool fills in).
    frozen = _frozen_panel(
        [
            ("2021-07-01", "SPARSE", NAN), ("2021-07-02", "SPARSE", NAN),
            ("2021-07-03", "SPARSE", 0.5),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "SPARSE", NAN), ("2021-07-02", "SPARSE", 0.7),
            ("2021-07-03", "SPARSE", 0.5),
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    warmup = result.by_class("warmup_left_extension")
    assert len(warmup) == 1
    assert result.warmup_by_direction == {"nan_to_finite": 1}


def test_panels_bounded_warmup_finite_to_finite_inside_boundary():
    # Partial pool -> full pool: both sides finite, inside the first w-1
    # dates. Registered (this is the 4554/17109-cell form from the first run).
    frozen = _frozen_panel(
        [("2021-07-01", "A", 0.10), ("2021-07-02", "A", 0.20), ("2021-07-03", "A", 0.30)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 0.11), ("2021-07-02", "A", 0.25), ("2021-07-03", "A", 0.30)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert result.warmup_by_direction == {"finite_to_finite": 2}


def test_panels_bounded_finite_to_finite_OUTSIDE_warmup_boundary_fails():
    # REVERSE direction: the very same finite->finite change one date later
    # is a real regression and must FAIL.
    frozen = _frozen_panel(
        [("2021-07-01", "A", 0.10), ("2021-07-02", "A", 0.20), ("2021-07-03", "A", 0.30)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 0.10), ("2021-07-02", "A", 0.20), ("2021-07-03", "A", 0.35)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


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


def test_panels_bounded_new_only_finite_row_inside_warmup_is_registered():
    # B's frozen grid skips 2021-07-02 (a grid gap, as in the real volume_peak
    # new-only rows); the served panel fills it. 07-02 sits inside the grid's
    # first w-1 = 2 trading dates -> registered warmup, not an unregistered row.
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0),
            ("2021-07-01", "B", 0.5), ("2021-07-03", "B", 0.6),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0),
            ("2021-07-01", "B", 0.5), ("2021-07-02", "B", 0.55), ("2021-07-03", "B", 0.6),
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert result.warmup_by_direction == {"new_only_finite": 1}


def test_panels_bounded_new_only_finite_row_outside_warmup_fails():
    frozen = _frozen_panel([("2021-07-01", "A", 0.5)], "f")
    new = _new_series([("2021-07-01", "A", 0.5), ("2021-09-01", "A", 0.4)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert not result.ok
    assert len(result.by_class("unregistered_new_finite_row")) == 1


def test_panels_pooled_warmup_class_all_directions_with_monthly_decrease():
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", NAN), ("2021-07-02", "A", 1.0),
            ("2021-08-02", "A", NAN),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.5),  # nan->finite + finite->finite
            ("2021-08-02", "A", 3.0), ("2021-07-05", "A", 2.0),  # new-only finite row
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_left_extension")) == 4
    assert result.warmup_by_direction == {
        "nan_to_finite": 2, "finite_to_finite": 1, "new_only_finite": 1
    }
    assert result.warmup_by_month == {"2021-07": 3, "2021-08": 1}
    assert result.warmup_monotonic


def test_panels_pooled_warmup_after_early_region_is_unclassified():
    frozen = _frozen_panel([("2022-01-04", "A", NAN)], "f")
    new = _new_series([("2022-01-04", "A", 1.0)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert not result.ok
    assert len(result.by_class("unclassified_nan_to_finite")) == 1


def test_panels_pooled_warmup_non_monotonic_monthly_counts_fail():
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
    assert not result.warmup_monotonic


def test_panels_float_reordering_tail_within_bounds_passes():
    # Scattered cells at rel 3e-12 (the measured jump tail: 1.0e-12..2.9e-12).
    # lookback_depth=1 disables the bounded warmup boundary so the tail
    # classes are what is being exercised.
    frozen = _frozen_panel(
        [("2022-11-03", "A", 1.0), ("2023-01-10", "B", 2.0)], "f"
    )
    new = _new_series(
        [("2022-11-03", "A", 1.0 * (1 + 3e-12)), ("2023-01-10", "B", 2.0 * (1 - 3e-12))]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert result.ok, result.diffs
    assert len(result.by_class("float_reordering_tail")) == 2


def test_panels_float_tail_beyond_rel_bound_fails():
    frozen = _frozen_panel([("2022-11-03", "A", 1.0)], "f")
    new = _new_series([("2022-11-03", "A", 1.0 * (1 + 1e-11))])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_float_tail_beyond_cell_cap_fails():
    n = 102  # one above the registered cap (101)
    frozen = _frozen_panel([(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0) for i in range(n)], "f")
    new = _new_series(
        [(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0 * (1 + 3e-12)) for i in range(n)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert not result.ok
    assert len(result.by_class("float_reordering_tail")) == n


def test_panels_threshold_flip_tail_within_bounds_passes():
    # Count flips by EXACTLY +/-1 at small rel (the measured volume_peak
    # cluster: 600623.SH, 20 consecutive emits, sigma noise x integer volume).
    rows = [(f"2023-06-{d:02d}", "600623.SH", 150.0) for d in (14, 15, 16)]
    frozen = _frozen_panel(rows, "f")
    new = _new_series([(d, s, v - 1.0) for d, s, v in rows])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.ok, result.diffs
    assert len(result.by_class("threshold_flip_tail")) == 3


def test_panels_threshold_flip_amplitude_above_one_fails():
    frozen = _frozen_panel([("2023-06-15", "600623.SH", 150.0)], "f")
    new = _new_series([("2023-06-15", "600623.SH", 148.0)])  # delta = 2
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_threshold_flip_beyond_cell_cap_fails():
    n = 26  # one above the registered cap (25)
    frozen = _frozen_panel([(f"2023-06-{(i % 28) + 1:02d}", f"S{i}", 150.0) for i in range(n)], "f")
    new = _new_series([(f"2023-06-{(i % 28) + 1:02d}", f"S{i}", 149.0) for i in range(n)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert not result.ok
    assert len(result.by_class("threshold_flip_tail")) == n


def test_panels_frozen_finite_new_nan_never_allowed():
    frozen = _frozen_panel([("2021-07-01", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", NAN)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=2
    )
    assert not result.ok
    assert len(result.by_class("unclassified_frozen_finite_new_nan")) == 1


def test_panels_finite_vs_finite_beyond_tolerance_never_allowed():
    # lookback_depth=1 -> EMPTY bounded warmup boundary, so no warmup excuse.
    frozen = _frozen_panel([("2021-07-01", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", 1.0 + 1e-6)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
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


def test_anchor_pooled_early_mismatch_is_the_warmup_class():
    row = classify_anchor_row(
        factor_id="volume_peak_count_20", cls="warmup_end",
        date=pd.Timestamp("2021-07-28"), symbol="600827.SH",
        hand=150.0, service=151.0, is_pooled=True, tol=1e-12,
    )
    assert row.classification == "warmup_left_extension"


def test_anchor_pooled_late_mismatch_fails():
    row = classify_anchor_row(
        factor_id="volume_peak_count_20", cls="random",
        date=pd.Timestamp("2023-05-18"), symbol="600867.SH",
        hand=1.0, service=1.1, is_pooled=True, tol=1e-12,
    )
    assert row.classification == "failed"


def test_anchor_bounded_mismatch_inside_warmup_dates_is_the_warmup_class():
    warmup = frozenset(pd.Timestamp(d) for d in ("2021-07-01", "2021-07-02", "2021-07-05"))
    row = classify_anchor_row(
        factor_id="minute_ideal_amp_10", cls="warmup_end",
        date=pd.Timestamp("2021-07-02"), symbol="000537.SZ",
        hand=0.46, service=0.47, is_pooled=False, tol=1e-12, warmup_dates=warmup,
    )
    assert row.classification == "warmup_left_extension"


def test_anchor_bounded_mismatch_outside_warmup_dates_fails():
    warmup = frozenset(pd.Timestamp(d) for d in ("2021-07-01", "2021-07-02", "2021-07-05"))
    row = classify_anchor_row(
        factor_id="minute_ideal_amp_10", cls="random",
        date=pd.Timestamp("2021-07-06"), symbol="000537.SZ",
        hand=0.46, service=0.47, is_pooled=False, tol=1e-12, warmup_dates=warmup,
    )
    assert row.classification == "failed"


def test_anchor_jump_mismatch_inside_warmup_is_warmup_not_definition_failure():
    # The truncation-carried signal lives on jump's NON-warmup rows; a warmup
    # row differs by loading geometry (hand side anchored at 2021-07-01).
    warmup = frozenset(pd.Timestamp(d) for d in ("2021-07-01", "2021-07-02"))
    row = classify_anchor_row(
        factor_id="jump_amount_corr_20", cls="warmup_end",
        date=pd.Timestamp("2021-07-02"), symbol="000537.SZ",
        hand=0.46, service=0.35, is_pooled=False, tol=1e-12, warmup_dates=warmup,
    )
    assert row.classification == "warmup_left_extension"
