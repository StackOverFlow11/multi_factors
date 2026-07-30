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
    REGISTERED_EXTRA_SECTIONS,
    REGISTERED_SPEC_DESCRIPTION_REWRITES,
    SPARSE_VALID_DAY_TAIL_SYMBOLS,
    ReconciliationError,
    check_new_pair_consistency,
    classify_anchor_row,
    classify_panel_differences,
    diff_report_json,
    diff_report_md,
    frozen_panel_path,
    require_baseline_verified,
    require_report_inputs,
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
# reports mode — registered add-Section additions (§七之四, D5 C4b vpq)
# --------------------------------------------------------------------------- #
def _with_neutralization_section(new: dict) -> dict:
    """The unified runner's vpq artifact: one extra add-Section the frozen lacks."""
    new = dict(new)
    new["sections"] = [
        *new["sections"],
        {
            "name": "neutralization_coverage",
            "note": "neutralization (T-1 rev20): raw_rows=100 rev_paired=90 "
            "residual_rows=80 dates=10/10 cross_section min/med/max=11/12.0/13 "
            "mean_spearman(raw,rev20)=-0.1234",
            "payload": {
                "raw_rows": 100,
                "rev_rows": 90,
                "residual_rows": 80,
                "dates_total": 10,
                "dates_residualized": 10,
                "cross_section_min": 11,
                "cross_section_median": 12.0,
                "cross_section_max": 13,
                "raw_rev_spearman_mean": -0.1234,
            },
        },
    ]
    return new


def test_json_registered_section_addition_passes_for_vpq():
    new = _with_neutralization_section(_frozen_like())
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False,
        registered_sections=("neutralization_coverage",),
    )
    assert result.ok, result.diffs
    classes = {d.classification for d in result.diffs}
    assert classes == {"registered_section_addition"}
    assert all(d.path.startswith("sections[1]") for d in result.diffs)


def test_json_extra_section_without_registration_fails():
    new = _with_neutralization_section(_frozen_like())
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_addition")


def test_json_section_addition_is_matched_by_NAME_not_index():
    """A DIFFERENT section at the registered index stays unregistered."""
    new = _with_neutralization_section(_frozen_like())
    new["sections"][1]["name"] = "some_other_coverage"
    result = diff_report_json(
        _frozen_like(), new, name="t", strict=True, correction_expected=False,
        registered_sections=("neutralization_coverage",),
    )
    assert not result.ok
    # every leaf of the foreign section is unregistered, incl. the payload
    assert len(result.by_class("unregistered_addition")) >= 10


_NEUTRALIZATION_MD = (
    "## + neutralization_coverage\n"
    "\n"
    "neutralization (T-1 rev20): raw_rows=100 rev_paired=90 residual_rows=80 "
    "dates=10/10 cross_section min/med/max=11/12.0/13 "
    "mean_spearman(raw,rev20)=-0.1234\n"
    "\n"
    "- cross_section_max: 13\n"
    "- cross_section_median: 12.0\n"
    "- cross_section_min: 11\n"
    "- dates_residualized: 10\n"
    "- dates_total: 10\n"
    "- raw_rev_spearman_mean: -0.1234\n"
    "- raw_rows: 100\n"
    "- residual_rows: 80\n"
    "- rev_rows: 90\n"
)


def test_md_registered_section_lines_pass_for_vpq():
    from qt.factor_eval_reconcile import _registered_section_md_prefixes

    prefixes = _registered_section_md_prefixes(
        "valley_price_quantile_20", ("neutralization_coverage",)
    )
    result = diff_report_md(
        _MD_OLD, _MD_OLD + _NEUTRALIZATION_MD, name="t", correction_expected=False,
        registered_section_lines=prefixes,
    )
    assert result.ok, result.diffs
    assert {d.classification for d in result.diffs} == {"registered_section_addition"}


def test_md_section_lines_without_registration_fail():
    result = diff_report_md(
        _MD_OLD, _MD_OLD + _NEUTRALIZATION_MD, name="t", correction_expected=False
    )
    assert not result.ok
    assert result.by_class("unregistered_addition")


def test_md_section_prefixes_are_derived_from_the_dataclass_fields():
    """A NeutralizationCoverage field rename breaks the registration loudly."""
    from dataclasses import fields as dc_fields

    from qt.factor_eval_disclosures import NeutralizationCoverage
    from qt.factor_eval_reconcile import _registered_section_md_prefixes

    prefixes = _registered_section_md_prefixes(
        "valley_price_quantile_20", ("neutralization_coverage",)
    )
    for f in dc_fields(NeutralizationCoverage):
        assert f"- {f.name}:" in prefixes
    # the note prefix really is the render() format's head (not a stale copy)
    cov = NeutralizationCoverage(
        raw_rows=1, rev_rows=1, residual_rows=1, dates_total=1,
        dates_residualized=1, cross_section_min=1, cross_section_median=1.0,
        cross_section_max=1, raw_rev_spearman_mean=0.0,
    )
    assert cov.render().startswith("neutralization (T-1 rev20):")


def test_md_section_prefixes_reject_an_unknown_section():
    from qt.factor_eval_reconcile import _registered_section_md_prefixes

    with pytest.raises(ValueError, match="no MD rendering is registered"):
        _registered_section_md_prefixes(
            "valley_price_quantile_20", ("bogus_coverage",)
        )


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


# --------------------------------------------------------------------------- #
# Bounded warmup CELL CEILING (D5 C4a review LOW-1): the class is capped at
# (lookback_depth - 1) x |frozen symbols| — a structural bound (every class
# cell is a distinct (date, symbol) pair in warmup_dates x frozen_symbols),
# so an overshoot means the boundary/counting logic itself is broken.
# --------------------------------------------------------------------------- #
def test_panels_bounded_warmup_ceiling_is_the_structural_derivation():
    # 3 symbols, lookback_depth=4 -> ceiling = 3 warmup dates x 3 symbols = 9.
    frozen = _frozen_panel(
        [(f"2021-07-0{d}", s, 1.0) for d in (1, 2, 3, 4, 5) for s in ("A", "B", "C")],
        "f",
    )
    new = _new_series(
        [(f"2021-07-0{d}", s, 1.0) for d in (1, 2, 3, 4, 5) for s in ("A", "B", "C")]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=4
    )
    assert result.ok, result.diffs
    assert result.warmup_max_cells == 3 * 3


def test_panels_bounded_warmup_cells_exactly_at_the_ceiling_pass():
    # lookback_depth=3 -> 2 warmup dates; 2 symbols -> ceiling 4. All four
    # warmup cells move (finite -> finite): AT the ceiling, still registered.
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", 0.10), ("2021-07-01", "B", 0.50),
            ("2021-07-02", "A", 0.20), ("2021-07-02", "B", 0.60),
            ("2021-07-03", "A", 0.30), ("2021-07-03", "B", 0.70),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "A", 0.11), ("2021-07-01", "B", 0.55),
            ("2021-07-02", "A", 0.25), ("2021-07-02", "B", 0.65),
            ("2021-07-03", "A", 0.30), ("2021-07-03", "B", 0.70),
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_left_extension")) == 4
    assert result.warmup_max_cells == 4


def test_panels_bounded_warmup_cells_beyond_the_ceiling_fail():
    # REVERSE (LOW-1): a frozen panel with DUPLICATE (date, symbol) rows
    # emits one warmup cell per duplicate — 4 cells against the structural
    # ceiling of 2 warmup dates x 1 symbol = 2. An overshoot of a bound the
    # class cannot structurally exceed means the counting is broken -> FAIL.
    # The individual cells still classify as warmup (the ceiling is a COUNT
    # gate on the class, not a per-cell reclassification).
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", 0.10), ("2021-07-01", "A", 0.10),  # duplicate
            ("2021-07-02", "A", 0.20), ("2021-07-02", "A", 0.20),  # duplicate
            ("2021-07-03", "A", 0.30),
        ],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 0.15), ("2021-07-02", "A", 0.25), ("2021-07-03", "A", 0.30)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert len(result.by_class("warmup_left_extension")) == 4
    assert result.warmup_max_cells == 2
    assert not result.ok


def test_panels_pooled_warmup_class_has_no_cell_ceiling():
    # Pooled factors: NO count ceiling — the monthly monotonicity gate is
    # the mechanism (a large but non-increasing count still passes).
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", 1.0), ("2021-07-01", "B", 1.0),
            ("2021-08-02", "A", 1.0),
        ],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 2.0), ("2021-07-01", "B", 2.0), ("2021-08-02", "A", 2.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_left_extension")) == 3
    assert result.warmup_max_cells is None


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
    # REVERSE (lead ruling 1): the structurally anchored exempt month (July,
    # where the frozen panel's FIRST finite value lives) is exempt, but
    # months AFTER it must still be non-increasing — {07:1, 08:1, 09:2}
    # rises after the exempt month and must FAIL.
    frozen = _frozen_panel(
        [
            ("2021-07-01", "A", NAN),
            ("2021-07-15", "B", 9.9),  # first finite frozen value -> exempt 2021-07
            ("2021-08-02", "A", NAN),
            ("2021-09-01", "A", NAN), ("2021-09-02", "A", NAN),
        ],
        "f",
    )
    new = _new_series(
        [
            ("2021-07-01", "A", 1.0),
            ("2021-07-15", "B", 9.9),
            ("2021-08-02", "A", 2.0),
            ("2021-09-01", "A", 3.0), ("2021-09-02", "A", 4.0),
        ]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.warmup_exempt_month == "2021-07"
    assert not result.ok  # 2021-09 (2) > 2021-08 (1) violates 按月递减至零
    assert not result.warmup_monotonic


def test_panels_pooled_warmup_zero_diff_exempt_month_does_not_reset_the_gate():
    # REVIEW PROBE, reversed (LOW-1): the exemption is anchored to the
    # STRUCTURAL partial month (the frozen panel's first-finite-value month),
    # NOT to the first month that HAPPENS to have warmup diffs. Here July
    # (the structural partial month) has ZERO diffs and the counts then rise
    # {08:2, 09:5} — the positional "first month WITH diffs is exempt"
    # reading let exactly this through; the structural anchor must FAIL it.
    frozen = _frozen_panel(
        [("2021-07-29", "B", 9.9)]  # first finite frozen value -> exempt 2021-07
        + [(f"2021-08-{d + 1:02d}", f"S{i}", NAN) for i, d in enumerate(range(2))]
        + [(f"2021-09-{d + 1:02d}", f"S{i}", NAN) for i, d in enumerate(range(5))],
        "f",
    )
    new = _new_series(
        [("2021-07-29", "B", 9.9)]
        + [(f"2021-08-{d + 1:02d}", f"S{i}", 1.0) for i, d in enumerate(range(2))]
        + [(f"2021-09-{d + 1:02d}", f"S{i}", 1.0) for i, d in enumerate(range(5))]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.warmup_exempt_month == "2021-07"
    assert result.warmup_by_month == {"2021-08": 2, "2021-09": 5}
    assert not result.warmup_monotonic
    assert not result.ok


def test_panels_pooled_warmup_all_nan_frozen_panel_gets_no_exemption():
    # DEGENERATE: a frozen panel with NO finite value at all has no
    # structural partial month to anchor to -> no exemption (conservative):
    # {07:1, 08:2} rises from the very first month and must FAIL.
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
    assert result.warmup_exempt_month is None
    assert result.warmup_by_month == {"2021-07": 1, "2021-08": 2}
    assert not result.warmup_monotonic
    assert not result.ok


def test_panels_pooled_warmup_partial_first_month_is_exempt_from_monotonicity():
    # FORWARD (lead ruling 1): the exempt month is the month of the frozen
    # panel's FIRST FINITE VALUE — the structural partial month in which
    # residual/value existence starts (vpq: the frozen panel's first values
    # exist only from ~07-29) — it is exempt from the monotonicity check, so
    # {07:5, 08:10, 09:3} passes even though 2021-08 > 2021-07.
    rows = (
        [("2021-07-29", "B", 9.9)]  # first finite frozen value -> exempt 2021-07
        + [(f"2021-07-{d + 1:02d}", f"S{i}", NAN) for i, d in enumerate(range(5))]
        + [(f"2021-08-{d + 1:02d}", f"S{i}", NAN) for i, d in enumerate(range(10))]
        + [(f"2021-09-{d + 1:02d}", f"S{i}", NAN) for i, d in enumerate(range(3))]
    )
    frozen = _frozen_panel(rows, "f")
    new = _new_series([(d, s, v if pd.notna(v) else 1.0) for d, s, v in rows])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=20
    )
    assert result.ok, result.diffs
    assert result.warmup_exempt_month == "2021-07"
    assert result.warmup_by_month == {"2021-07": 5, "2021-08": 10, "2021-09": 3}
    assert result.warmup_monotonic


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


# --------------------------------------------------------------------------- #
# panels mode — threshold_flip_contamination (catalogue §七之五, lead ruling 2)
# --------------------------------------------------------------------------- #
def test_panels_threshold_flip_contamination_within_bounds_passes():
    # FORWARD: the measured vpq shape — the direct symbol (600623.SH) at
    # 1.6e-04 and cross-sectionally contaminated symbols at <= 5.5e-07, all
    # inside [2023-06-01, 2023-07-14], on a CROSS-SECTIONAL factor.
    rows = [
        ("2023-06-01", "600623.SH", 1.0), ("2023-07-14", "600623.SH", 1.0),
        ("2023-06-15", "S1", 2.0), ("2023-07-14", "S2", 3.0),
    ]
    deltas = [1.6e-04, -1.6e-04, 5.5e-07, -5.5e-07]
    frozen = _frozen_panel(rows, "f")
    new = _new_series([(d, s, v + dv) for (d, s, v), dv in zip(rows, deltas)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert result.ok, result.diffs
    assert len(result.by_class("threshold_flip_contamination")) == 4


def test_panels_contamination_direct_symbol_above_2e_04_fails():
    # REVERSE (bound 1 of 4): the direct symbol overshoots 2e-04 inside the
    # window -> UNCLASSIFIED, never a fall-through to the generic tails.
    frozen = _frozen_panel([("2023-06-15", "600623.SH", 1.0)], "f")
    new = _new_series([("2023-06-15", "600623.SH", 1.0 + 3e-04)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1
    assert not result.by_class("threshold_flip_contamination")


def test_panels_contamination_other_symbol_above_1e_06_fails():
    # REVERSE (bound 2 of 4): a contaminated symbol overshoots 1e-06 inside
    # the window.
    frozen = _frozen_panel([("2023-07-01", "S1", 1.0)], "f")
    new = _new_series([("2023-07-01", "S1", 1.0 + 2e-06)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_contamination_outside_window_fails():
    # REVERSE (bound 3 of 4): the same magnitude one day AFTER the window
    # (2023-07-15) is a new fact, not the adjudicated mechanism.
    frozen = _frozen_panel([("2023-07-15", "600623.SH", 1.0)], "f")
    new = _new_series([("2023-07-15", "600623.SH", 1.0 + 1.6e-04)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_contamination_beyond_cell_cap_fails():
    # REVERSE (bound 4 of 4): 20,020 in-window cells within the abs bounds
    # (1001 symbols x 20 dates) exceed the 20,000 cap.
    symbols = [f"S{i}" for i in range(1001)]
    dates = [f"2023-06-{d + 1:02d}" for d in range(20)]
    rows = [(d, s, 1.0) for d in dates for s in symbols]
    frozen = _frozen_panel(rows, "f")
    new = _new_series([(d, s, 1.0 + 5e-07) for d, s, _v in rows])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("threshold_flip_contamination")) == len(rows)


def test_panels_contamination_not_available_for_bars_only_factor():
    # REVERSE: the class is CROSS-SECTIONAL-ONLY — the same in-window cells
    # on a bars-only factor (is_cross_sectional=False) are unclassified.
    frozen = _frozen_panel([("2023-06-15", "600623.SH", 1.0)], "f")
    new = _new_series([("2023-06-15", "600623.SH", 1.0 + 1.6e-04)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


# --------------------------------------------------------------------------- #
# panels mode — float-tail abs floor + tiered cap (lead ruling 3)
# --------------------------------------------------------------------------- #
def test_panels_float_dust_abs_floor_passes_regardless_of_rel():
    # FORWARD: a near-zero cross-sectional OLS residual — abs 5e-13 (machine
    # precision on a ~1e-6 residual) but rel ~5e-07 >> 5e-12. The abs floor
    # (|diff| <= 1e-12) classes it as float dust on ANY factor kind.
    frozen = _frozen_panel([("2022-11-03", "A", 1e-06)], "f")
    new = _new_series([("2022-11-03", "A", 1e-06 + 5e-13)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert result.ok, result.diffs
    assert len(result.by_class("float_reordering_tail")) == 1


def test_panels_float_dust_above_abs_floor_with_rel_overage_fails():
    # REVERSE: abs 2e-12 > the 1e-12 floor AND rel ~2e-06 > 5e-12 — neither
    # criterion catches it, so it is unclassified (the floor is not a
    # tolerance widening).
    frozen = _frozen_panel([("2022-11-03", "A", 1e-06)], "f")
    new = _new_series([("2022-11-03", "A", 1e-06 + 2e-12)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_float_tail_cross_sectional_cap_allows_up_to_1000():
    # FORWARD: 500 float-tail cells pass on a cross-sectional factor (the
    # bars-only 101 cap would fail them — the cap is tiered, measured vpq
    # 707 + headroom).
    n = 500
    frozen = _frozen_panel([(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0) for i in range(n)], "f")
    new = _new_series(
        [(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0 * (1 + 3e-12)) for i in range(n)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert result.ok, result.diffs
    assert len(result.by_class("float_reordering_tail")) == n


def test_panels_float_tail_cross_sectional_cap_1001_fails():
    # REVERSE: 1001 float-tail cells — one above the cross-sectional cap.
    n = 1001
    frozen = _frozen_panel([(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0) for i in range(n)], "f")
    new = _new_series(
        [(f"2022-11-{(i % 28) + 1:02d}", f"S{i}", 1.0 * (1 + 3e-12)) for i in range(n)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("float_reordering_tail")) == n


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


# --------------------------------------------------------------------------- #
# D5 C5 F4 — the ABSOLUTE float-dust predicate runs BEFORE every region branch.
#
# Machine-precision dust exists uniformly across the grid; when the region
# branches ran first, the few dust cells that happened to land in the warmup
# region were counted as warmup cells and their monthly counts failed the
# pooled non-increasing gate on pure noise (measured intraday_amp_cut:
# 2021-08/09/10 = 8/3/9 cells, every one |diff| <= 1e-12).
# --------------------------------------------------------------------------- #
def test_panels_float_dust_inside_the_warmup_boundary_is_dust_not_warmup():
    # A 1e-13 absolute move on the grid's first w-1 dates: dust, whatever
    # region it sits in.
    frozen = _frozen_panel(
        [("2021-07-01", "A", 1e-5), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 1e-5 + 1e-13), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert len(result.by_class("float_reordering_tail")) == 1
    assert result.by_class("warmup_left_extension") == []


def test_panels_a_real_move_inside_the_warmup_boundary_is_still_warmup():
    # REVERSE of the above: raise the SAME cell's move above the dust floor
    # and it goes back to the warmup class. The reorder must not have turned
    # the warmup class off.
    frozen = _frozen_panel(
        [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 1.05), ("2021-07-02", "A", 1.0), ("2021-07-03", "A", 1.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_left_extension")) == 1
    assert result.by_class("float_reordering_tail") == []


def test_panels_pooled_monthly_gate_is_not_failed_by_float_dust():
    # The measured F4 shape: a big genuine warmup month, then later months
    # carrying ONLY dust. Under the old ordering those dust cells were warmup
    # cells and 1 -> 2 across months failed the non-increasing gate.
    rows_f = [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0)]
    rows_f += [(f"2021-08-{d:02d}", "A", 1e-5) for d in (2, 3)]
    rows_f += [(f"2021-09-{d:02d}", "A", 1e-5) for d in (1, 2, 3)]
    rows_f += [("2021-12-01", "A", 1.0)]
    frozen = _frozen_panel(rows_f, "f")
    # July: the structurally exempt month (first finite value) + a real warmup
    # move on 07-02; Aug/Sep: dust only, in a RISING count (2 then 3).
    rows_n = [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.4)]
    rows_n += [(f"2021-08-{d:02d}", "A", 1e-5 + 1e-13) for d in (2, 3)]
    rows_n += [(f"2021-09-{d:02d}", "A", 1e-5 + 1e-13) for d in (1, 2, 3)]
    rows_n += [("2021-12-01", "A", 1.0)]
    new = _new_series(rows_n)
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.ok, result.diffs
    assert result.warmup_by_month == {"2021-07": 1}
    assert result.warmup_monotonic is True
    assert len(result.by_class("float_reordering_tail")) == 5


def test_panels_pooled_monthly_gate_still_fails_on_real_rising_counts():
    # REVERSE: the same shape with the Aug/Sep cells moved ABOVE the dust
    # floor. The gate must still fire — the reorder removed noise from the
    # counts, it did not weaken the gate.
    rows_f = [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0)]
    rows_f += [(f"2021-08-{d:02d}", "A", 1.0) for d in (2, 3)]
    rows_f += [(f"2021-09-{d:02d}", "A", 1.0) for d in (1, 2, 3)]
    rows_f += [("2021-12-01", "A", 1.0)]
    frozen = _frozen_panel(rows_f, "f")
    rows_n = [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.4)]
    rows_n += [(f"2021-08-{d:02d}", "A", 1.2) for d in (2, 3)]
    rows_n += [(f"2021-09-{d:02d}", "A", 1.2) for d in (1, 2, 3)]
    rows_n += [("2021-12-01", "A", 1.0)]
    new = _new_series(rows_n)
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert not result.ok
    assert result.warmup_by_month == {"2021-07": 1, "2021-08": 2, "2021-09": 3}
    assert result.warmup_monotonic is False


def test_panels_float_dust_inside_the_contamination_window_is_dust():
    # The reorder also precedes the contamination window (measured: all 17 of
    # intraday_amp_cut's "contamination" cells were dust and now say so).
    frozen = _frozen_panel(
        [("2023-06-15", "600623.SH", 1e-5), ("2023-06-16", "600623.SH", 1.0)], "f"
    )
    new = _new_series(
        [("2023-06-15", "600623.SH", 1e-5 + 1e-13), ("2023-06-16", "600623.SH", 1.0)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1,
        is_cross_sectional=True,
    )
    assert result.ok, result.diffs
    assert len(result.by_class("float_reordering_tail")) == 1
    assert result.by_class("threshold_flip_contamination") == []


# --------------------------------------------------------------------------- #
# D5 C5 F2 — threshold_flip_contamination, BARS-ONLY arm (per-factor REL bound)
# --------------------------------------------------------------------------- #
def _flip_cells(factor_id: str, rel: float, n: int, *, symbol="600623.SH"):
    """n cells on ``symbol`` inside the registered flip window, at ~``rel``."""
    dates = pd.date_range("2023-06-15", periods=n, freq="D").strftime("%Y-%m-%d")
    frozen = _frozen_panel([(d, symbol, 1.0) for d in dates], factor_id)
    new = _new_series([(d, symbol, 1.0 / (1.0 - rel)) for d in dates])
    return new, frozen


def test_panels_bars_only_flip_contamination_within_the_per_factor_bound():
    new, frozen = _flip_cells("peak_interval_kurtosis_20", 2.9e-03, 3)
    result = classify_panel_differences(
        new, frozen, factor_id="peak_interval_kurtosis_20", is_pooled=True,
        lookback_depth=1,
    )
    assert result.ok, result.diffs
    assert len(result.by_class("threshold_flip_contamination")) == 3


def test_panels_bars_only_flip_above_the_per_factor_bound_fails():
    # REVERSE: one order of magnitude past kurtosis's 5e-3 bound.
    new, frozen = _flip_cells("peak_interval_kurtosis_20", 5e-02, 3)
    result = classify_panel_differences(
        new, frozen, factor_id="peak_interval_kurtosis_20", is_pooled=True,
        lookback_depth=1,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 3


def test_panels_bars_only_flip_bound_is_PER_FACTOR():
    # The same 1e-3 cell passes for kurtosis (bound 5e-3) and FAILS for
    # valley_relative_vwap (bound 1e-5) — a single shared bound calibrated on
    # either factor would be wrong about the other.
    kurt_new, kurt_frozen = _flip_cells("peak_interval_kurtosis_20", 1e-03, 2)
    kurt = classify_panel_differences(
        kurt_new, kurt_frozen, factor_id="peak_interval_kurtosis_20",
        is_pooled=True, lookback_depth=1,
    )
    vwap_new, vwap_frozen = _flip_cells("valley_relative_vwap_20", 1e-03, 2)
    vwap = classify_panel_differences(
        vwap_new, vwap_frozen, factor_id="valley_relative_vwap_20",
        is_pooled=True, lookback_depth=1,
    )
    assert kurt.ok and len(kurt.by_class("threshold_flip_contamination")) == 2
    assert not vwap.ok
    assert len(vwap.by_class("unclassified_finite_vs_finite")) == 2


def test_panels_bars_only_flip_is_not_available_to_an_unregistered_factor():
    # REVERSE: jump lives in the same window with the same magnitude and is
    # NOT in the registry -> it never gets the class.
    new, frozen = _flip_cells("jump_amount_corr_20", 2.9e-03, 3)
    result = classify_panel_differences(
        new, frozen, factor_id="jump_amount_corr_20", is_pooled=False,
        lookback_depth=1,
    )
    assert not result.ok
    assert result.by_class("threshold_flip_contamination") == []
    assert len(result.by_class("unclassified_finite_vs_finite")) == 3


def test_panels_bars_only_flip_is_only_the_directly_affected_symbol():
    # REVERSE: a bars-only factor has no per-date OLS to spread the flip, so
    # another symbol moving inside the window is a NEW FACT, not contamination.
    new, frozen = _flip_cells("peak_interval_kurtosis_20", 2.9e-03, 3, symbol="600000.SH")
    result = classify_panel_differences(
        new, frozen, factor_id="peak_interval_kurtosis_20", is_pooled=True,
        lookback_depth=1,
    )
    assert not result.ok
    assert result.by_class("threshold_flip_contamination") == []


def test_panels_bars_only_flip_outside_the_window_fails():
    # REVERSE: same symbol, same magnitude, one day past the window's end.
    frozen = _frozen_panel([("2023-07-17", "600623.SH", 1.0)], "peak_interval_kurtosis_20")
    new = _new_series([("2023-07-17", "600623.SH", 1.0029)])
    result = classify_panel_differences(
        new, frozen, factor_id="peak_interval_kurtosis_20", is_pooled=True,
        lookback_depth=1,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_bars_only_flip_cell_cap():
    at_cap_new, at_cap_frozen = _flip_cells("peak_interval_kurtosis_20", 2.9e-03, 25)
    at_cap = classify_panel_differences(
        at_cap_new, at_cap_frozen, factor_id="peak_interval_kurtosis_20",
        is_pooled=True, lookback_depth=1,
    )
    over_new, over_frozen = _flip_cells("peak_interval_kurtosis_20", 2.9e-03, 26)
    over = classify_panel_differences(
        over_new, over_frozen, factor_id="peak_interval_kurtosis_20",
        is_pooled=True, lookback_depth=1,
    )
    assert at_cap.ok and len(at_cap.by_class("threshold_flip_contamination")) == 25
    # every cell still CLASSIFIES; the cap is a count gate on the class
    assert len(over.by_class("threshold_flip_contamination")) == 26
    assert not over.ok


def test_panels_bars_only_bound_never_applies_to_a_cross_sectional_factor():
    # Exclusivity: a cross-sectional factor uses the ABSOLUTE arm even if its
    # id is in the bars-only registry. rel 2.9e-3 on a value of 1.0 is
    # |diff| 2.9e-3, past the direct-symbol 2e-4 absolute bound -> FAIL.
    new, frozen = _flip_cells("peak_interval_kurtosis_20", 2.9e-03, 3)
    result = classify_panel_differences(
        new, frozen, factor_id="peak_interval_kurtosis_20", is_pooled=True,
        lookback_depth=1, is_cross_sectional=True,
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 3


# --------------------------------------------------------------------------- #
# D5 C5 F3 (1) — frozen-finite -> new-NaN joins the warmup direction set, for
# VALID-DAY POOLED factors inside the early region only.
# --------------------------------------------------------------------------- #
def test_panels_pooled_frozen_finite_new_nan_in_the_early_region_is_warmup():
    frozen = _frozen_panel(
        [("2021-08-20", "688276.SH", 0.42), ("2021-12-01", "688276.SH", 0.5)], "f"
    )
    new = _new_series(
        [("2021-08-20", "688276.SH", NAN), ("2021-12-01", "688276.SH", 0.5)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.ok, result.diffs
    assert result.warmup_by_direction == {"frozen_finite_new_nan": 1}
    assert result.warmup_by_month == {"2021-08": 1}


def test_panels_bounded_frozen_finite_new_nan_is_never_warmup():
    # REVERSE: a bounded factor has no valid-day counting gate that could
    # legitimately drop a value — inside its own warmup boundary or not.
    frozen = _frozen_panel(
        [("2021-08-20", "688276.SH", 0.42), ("2021-12-01", "688276.SH", 0.5)], "f"
    )
    new = _new_series(
        [("2021-08-20", "688276.SH", NAN), ("2021-12-01", "688276.SH", 0.5)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=40
    )
    assert not result.ok
    assert len(result.by_class("unclassified_frozen_finite_new_nan")) == 1


def test_panels_pooled_frozen_finite_new_nan_outside_the_early_region_fails():
    # REVERSE: the same disappearance one month past the early region.
    frozen = _frozen_panel(
        [("2021-11-20", "688276.SH", 0.42), ("2021-12-01", "688276.SH", 0.5)], "f"
    )
    new = _new_series(
        [("2021-11-20", "688276.SH", NAN), ("2021-12-01", "688276.SH", 0.5)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert not result.ok
    assert len(result.by_class("unclassified_frozen_finite_new_nan")) == 1


# --------------------------------------------------------------------------- #
# D5 C5 F3 (2) — warmup_sparse_valid_day_tail
# --------------------------------------------------------------------------- #
def _sparse_tail_cells(n: int, *, symbol="000402.SZ", start="2021-11-01", value=-0.5):
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    frozen = _frozen_panel([(d, symbol, 0.5) for d in dates], "f")
    new = _new_series([(d, symbol, value) for d in dates])
    return new, frozen


def test_panels_sparse_valid_day_tail_is_registered_for_pooled_factors():
    # Amplitude is deliberately unbounded inside the class: the measured
    # ridge_minute_return cells include a sign flip (rel 1.96).
    new, frozen = _sparse_tail_cells(3)
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_sparse_valid_day_tail")) == 3


def test_panels_sparse_valid_day_tail_is_not_available_to_bounded_factors():
    new, frozen = _sparse_tail_cells(3)
    result = classify_panel_differences(
        # lookback_depth=1 -> no warmup dates at all, so the only class that
        # could take these cells is the sparse tail, and a bounded factor
        # must not get it.
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=1
    )
    assert not result.ok
    assert result.by_class("warmup_sparse_valid_day_tail") == []
    assert len(result.by_class("unclassified_finite_vs_finite")) == 3


def test_panels_sparse_valid_day_tail_symbol_whitelist_has_teeth():
    # REVERSE: a name that is not on the sparse whitelist, same window.
    new, frozen = _sparse_tail_cells(3, symbol="600519.SH")
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 3


def test_panels_sparse_valid_day_tail_window_has_teeth():
    # REVERSE, re-cut for the re-derived boundary: the window now ends
    # 2021-11-15 (a Monday), so the first date OUTSIDE it is 2021-11-16.
    frozen = _frozen_panel([("2021-11-16", "000402.SZ", 0.5)], "f")
    new = _new_series([("2021-11-16", "000402.SZ", -0.5)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert not result.ok
    assert len(result.by_class("unclassified_finite_vs_finite")) == 1


def test_panels_sparse_valid_day_tail_cell_cap():
    at_cap_new, at_cap_frozen = _sparse_tail_cells(20)
    over_new, over_frozen = _sparse_tail_cells(21)
    at_cap = classify_panel_differences(
        at_cap_new, at_cap_frozen, factor_id="f", is_pooled=True, lookback_depth=40,
        sparse_tail_hi=pd.Timestamp("2021-12-31"),
    )
    over = classify_panel_differences(
        over_new, over_frozen, factor_id="f", is_pooled=True, lookback_depth=40,
        sparse_tail_hi=pd.Timestamp("2021-12-31"),
    )
    assert at_cap.ok and len(at_cap.by_class("warmup_sparse_valid_day_tail")) == 20
    assert len(over.by_class("warmup_sparse_valid_day_tail")) == 21
    assert not over.ok


def test_panels_sparse_valid_day_tail_does_not_enter_the_monthly_warmup_counts():
    # It sits OUTSIDE the early region by construction; feeding it into the
    # monthly non-increasing machinery would fail that gate by definition.
    new, frozen = _sparse_tail_cells(3)
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.warmup_by_month == {}
    assert result.warmup_by_direction == {}


# --------------------------------------------------------------------------- #
# D5 C4a review NIT-1 — per-class max rel in the run summary
# --------------------------------------------------------------------------- #
def test_max_rel_by_class_separates_a_registered_headline_from_the_gate():
    # The headline max_rel_diff is taken before any bucketing; here it is
    # 0.5 and belongs ENTIRELY to the warmup class, while the only other
    # differing cell is dust. Printed alone the headline reads like an
    # ungated tolerance.
    frozen = _frozen_panel(
        [("2021-07-01", "A", 1.0), ("2021-07-02", "A", 1.0), ("2021-09-01", "A", 1e-5)],
        "f",
    )
    new = _new_series(
        [("2021-07-01", "A", 2.0), ("2021-07-02", "A", 1.0), ("2021-09-01", "A", 1e-5 + 1e-13)]
    )
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.ok, result.diffs
    assert result.max_rel_diff == pytest.approx(0.5)
    by_class = result.max_rel_by_class()
    assert by_class["warmup_left_extension"] == pytest.approx(0.5)
    # ... while the other differing cell is seven orders of magnitude smaller
    # (its RATIO is 1e-8 — it qualifies on the absolute arm, which is exactly
    # why reading the headline as "the tolerance" is wrong).
    assert by_class["float_reordering_tail"] < 1e-6


def test_max_rel_by_class_reports_inf_for_one_sided_cells():
    # A NaN -> finite cell has no ratio; reporting 0.0 would read as "these
    # cells agree".
    frozen = _frozen_panel([("2021-07-01", "A", NAN), ("2021-07-02", "A", 1.0)], "f")
    new = _new_series([("2021-07-01", "A", 0.7), ("2021-07-02", "A", 1.0)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=False, lookback_depth=3
    )
    assert result.max_rel_by_class()["warmup_left_extension"] == float("inf")


# --------------------------------------------------------------------------- #
# D5 C5 F5 — the reports leg names its missing inputs instead of dying on the
# first one it happens to open.
# --------------------------------------------------------------------------- #
def _write_report_set(report_dir: Path, factor_id: str, *, bookclose: bool):
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"factor_eval_{factor_id}"
    names = [
        f"{stem}_exec_no_book.json", f"{stem}_exec_no_book.md",
        f"{stem}_exec_with_book.json", f"{stem}_exec_with_book.md",
    ]
    if bookclose:
        names += [
            f"{stem}_exec_with_book_bookclose.json",
            f"{stem}_exec_with_book_bookclose.md",
        ]
    for name in names:
        (report_dir / name).write_text("{}", encoding="utf-8")
    return stem


def test_report_inputs_complete_decision_set_passes_with_or_without_bookclose(tmp_path):
    for bookclose in (False, True):
        d = tmp_path / f"bc_{bookclose}"
        _write_report_set(d, "jump_amount_corr_20", bookclose=bookclose)
        require_report_inputs(d, "jump_amount_corr_20", "config/x.yaml")


def test_report_inputs_names_every_missing_decision_artifact(tmp_path):
    stem = _write_report_set(tmp_path, "jump_amount_corr_20", bookclose=True)
    (tmp_path / f"{stem}_exec_with_book.json").unlink()
    (tmp_path / f"{stem}_exec_with_book.md").unlink()
    with pytest.raises(ReconciliationError) as excinfo:
        require_report_inputs(tmp_path, "jump_amount_corr_20", "config/x.yaml")
    message = str(excinfo.value)
    # BOTH missing files are named (the bare FileNotFoundError named one) ...
    assert f"{stem}_exec_with_book.json" in message
    assert f"{stem}_exec_with_book.md" in message
    # ... and the message says what to run, with the config it was given.
    assert "--book-mode decision" in message
    assert "config/x.yaml" in message


def test_report_inputs_reject_a_half_written_bookclose_pair(tmp_path):
    stem = _write_report_set(tmp_path, "jump_amount_corr_20", bookclose=True)
    (tmp_path / f"{stem}_exec_with_book_bookclose.md").unlink()
    with pytest.raises(ReconciliationError, match="HALF-written"):
        require_report_inputs(tmp_path, "jump_amount_corr_20", "config/x.yaml")


def test_panels_sparse_valid_day_tail_last_registered_day_is_inside():
    # The re-derived window's LAST day must be inside it (the boundary moved
    # to 2021-11-15 because one real cell sat there); paired with the reverse
    # test above, the two pin the edge from both sides.
    frozen = _frozen_panel([("2021-11-15", "000402.SZ", 0.5)], "f")
    new = _new_series([("2021-11-15", "000402.SZ", -0.5)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_sparse_valid_day_tail")) == 1


#: The re-derived membership, written out INDEPENDENTLY of the constant.
#: Parametrizing over ``SPARSE_VALID_DAY_TAIL_SYMBOLS`` itself cannot detect a
#: name being dropped — the case simply disappears and the suite still passes
#: (measured: removing 600906.SH took the run from 9 passed to 8 passed, green
#: both times). A test whose expectations are read from the thing under test
#: is the shape this repo has been bitten by repeatedly.
_EXPECTED_SPARSE_TAIL_SYMBOLS = (
    "000034.SZ", "000402.SZ", "000999.SZ", "002281.SZ", "002375.SZ",
    "002653.SZ", "300857.SZ", "600906.SH", "688183.SH",
)


def test_the_sparse_tail_whitelist_is_exactly_the_re_derived_membership():
    assert SPARSE_VALID_DAY_TAIL_SYMBOLS == frozenset(_EXPECTED_SPARSE_TAIL_SYMBOLS)
    assert len(_EXPECTED_SPARSE_TAIL_SYMBOLS) == 9


@pytest.mark.parametrize("symbol", _EXPECTED_SPARSE_TAIL_SYMBOLS)
def test_panels_sparse_tail_whitelist_members_are_each_accepted(symbol):
    # Every registered name is really usable (a typo'd ticker would sit in the
    # constant looking authoritative while never matching anything).
    frozen = _frozen_panel([("2021-11-02", symbol, 0.5)], "f")
    new = _new_series([("2021-11-02", symbol, -0.5)])
    result = classify_panel_differences(
        new, frozen, factor_id="f", is_pooled=True, lookback_depth=40
    )
    assert result.ok, result.diffs
    assert len(result.by_class("warmup_sparse_valid_day_tail")) == 1


# --------------------------------------------------------------------------- #
# The sparse-tail class's NECESSARY-CONDITION GUARD, checked against the real
# frozen panels. The claim it guards is stated ONCE, on
# ``SPARSE_VALID_DAY_TAIL_SYMBOLS`` in qt/factor_eval_reconcile.py; this banner
# deliberately does not restate it, because the same sentence written in four
# places is how three of the four got corrected and the fourth kept asserting
# a median gate fifteen lines above code that reads p25.
# --------------------------------------------------------------------------- #
_SPARSE_TAIL_FACTORS = (
    "ridge_minute_return_20",
    "valley_ridge_vwap_ratio_20",
    "peak_ridge_amount_ratio_20",
)
#: The emission window the percentiles are taken over: the early region plus
#: the November cluster the class covers.
_DENSITY_LO, _DENSITY_HI = "2021-07-01", "2021-11-30"
_FROZEN_PANELS = Path("artifacts/refactor_baseline/panels")

requires_frozen_panels = pytest.mark.skipif(
    not all((_FROZEN_PANELS / f"{f}.parquet").exists() for f in _SPARSE_TAIL_FACTORS),
    reason="frozen D1 panels not on disk (artifacts/ is gitignored)",
)


def _emission_density(factor_id: str) -> tuple[dict[str, int], float]:
    """(symbol -> emitted days, the factor's p25) over the density window.

    The threshold is the 25th percentile, NOT the median. A median gate is
    nearly vacuous by construction — half the universe passes it — so a guard
    built on it could hardly fail. Measured, all in-class pairs sit below p25
    (worst percentile rank 21.7%, 15 of 18 below 3%), so tightening costs
    nothing here and turns an almost-unfailable check into one with teeth.
    """
    frame = pd.read_parquet(_FROZEN_PANELS / f"{factor_id}.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[(frame["date"] >= _DENSITY_LO) & (frame["date"] <= _DENSITY_HI)]
    emitted = window.groupby("symbol")[factor_id].apply(lambda s: int(s.notna().sum()))
    return emitted.to_dict(), float(emitted.quantile(0.25))


@requires_frozen_panels
def test_every_sparse_tail_name_is_sparse_on_some_affected_factor():
    """Each whitelisted name must satisfy the NECESSARY condition somewhere.

    This is a guard on the enumeration, not a rule that produces it: the
    predicate admits roughly 69x as many (factor, symbol) pairs as the class
    covers, so passing it says only that no member contradicts the story.
    A name at or above p25 on all three affected factors WOULD contradict it —
    the class would then be admitting that name for some reason nobody has
    stated. The ruling on that is explicit: STOP, do not widen.
    """
    density = {f: _emission_density(f) for f in _SPARSE_TAIL_FACTORS}
    offenders = {}
    for symbol in sorted(SPARSE_VALID_DAY_TAIL_SYMBOLS):
        seen = {
            f: (emitted.get(symbol), median)
            for f, (emitted, median) in density.items()
            if symbol in emitted
        }
        if not any(e < m for e, m in seen.values()):
            offenders[symbol] = seen
    assert not offenders, (
        "whitelisted name(s) are NOT sparse on any affected factor, so the "
        f"class is not describing the mechanism it claims: {offenders}"
    )


@requires_frozen_panels
def test_the_density_criterion_predicts_where_600906_appears_and_where_it_does_not():
    """The asymmetry, with its reach: only the NEGATIVE direction holds.

    600906.SH is below p25 on peak_ridge and above the median on ridge — and it
    shows up in the former's November cluster and not in the latter's. What
    that supports is "dense => absent" (the control group volume_peak_count_20
    corroborates it: zero cells on all nine names). It does NOT support
    "sparse => present", which is false by roughly 69x. It is also ONE data
    point, dense by a single day.
    """
    peak_emitted, peak_p25 = _emission_density("peak_ridge_amount_ratio_20")
    ridge_emitted, _ridge_p25 = _emission_density("ridge_minute_return_20")
    assert peak_emitted["600906.SH"] < peak_p25
    # The dense side is stated against the MEDIAN and clears it by ONE DAY
    # (42 vs 41) — a single data point, recorded with its reach rather than
    # leaned on. Only the negative direction of the prediction holds
    # (dense => absent); sparse => present is false, by 69x.
    assert ridge_emitted["600906.SH"] >= 41


# --------------------------------------------------------------------------- #
# D5 C5 phase B — the two drifts the reports leg had never shown anyone,
# because until phase B it had never produced a judgement for these factors.
#
# A: the diagnostics-sink coverage disclosures arriving as an appended section.
# B: the D2 provenance rewrite of spec.description, registered as EXACT PAIRS.
# --------------------------------------------------------------------------- #
def _json_with_sections(names: list[str]) -> dict:
    doc = _frozen_like()
    doc["sections"] = [{"name": n, "status": "ok", "payload": {"v": 1}} for n in names]
    return doc


def test_registered_disclosure_sections_are_additions_for_each_factor():
    for factor_id, section in (
        ("valley_ridge_vwap_ratio_20", "ridge_scarcity_coverage"),
        ("ridge_minute_return_20", "ridge_scarcity_coverage"),
        ("peak_ridge_amount_ratio_20", "peak_scarcity_coverage"),
        ("valley_price_quantile_20", "neutralization_coverage"),
    ):
        old = _json_with_sections(["ic", "quantiles"])
        new = _with_registered_additions(_json_with_sections(["ic", "quantiles", section]))
        result = diff_report_json(
            old, new, name=factor_id, strict=True, correction_expected=False,
            registered_sections=REGISTERED_EXTRA_SECTIONS[factor_id],
        )
        assert result.ok, (factor_id, [d for d in result.diffs if "unregistered" in d.classification])
        assert result.by_class("registered_section_addition")


def test_an_unregistered_disclosure_section_still_fails():
    # REVERSE: a section that is not in this factor's registered tuple.
    old = _json_with_sections(["ic", "quantiles"])
    new = _with_registered_additions(
        _json_with_sections(["ic", "quantiles", "some_other_coverage"])
    )
    result = diff_report_json(
        old, new, name="ridge_minute_return_20", strict=True, correction_expected=False,
        registered_sections=REGISTERED_EXTRA_SECTIONS["ridge_minute_return_20"],
    )
    assert not result.ok
    assert result.by_class("unregistered_addition")


def test_a_section_registered_for_ANOTHER_factor_is_not_accepted_here():
    # The map is per factor: ridge's disclosure must not pass for a factor
    # whose registered tuple names a different section.
    old = _json_with_sections(["ic", "quantiles"])
    new = _with_registered_additions(
        _json_with_sections(["ic", "quantiles", "ridge_scarcity_coverage"])
    )
    result = diff_report_json(
        old, new, name="valley_price_quantile_20", strict=True,
        correction_expected=False,
        registered_sections=REGISTERED_EXTRA_SECTIONS["valley_price_quantile_20"],
    )
    assert not result.ok


def test_a_section_INSERTED_mid_list_is_detected_not_absorbed():
    """The check the lead asked for: index displacement must not pass silently.

    Leaf paths are indexed, so a section inserted before the end shifts every
    later one. If that were absorbed, "the other sections all match" would be
    an illusion. Measured on the real artifacts the disclosures are APPENDED
    (sections 8 -> 9, first eight names pairwise identical), and this test pins
    what happens if that ever stops being true.
    """
    old = _json_with_sections(["ic", "quantiles", "verdict_inputs"])
    new = _with_registered_additions(
        _json_with_sections(["ic", "ridge_scarcity_coverage", "quantiles", "verdict_inputs"])
    )
    result = diff_report_json(
        old, new, name="ridge_minute_return_20", strict=True, correction_expected=False,
        registered_sections=REGISTERED_EXTRA_SECTIONS["ridge_minute_return_20"],
    )
    assert not result.ok
    # Pin the MECHANISM, not just the failure: leaves are compared position by
    # position, so a displaced section shows up as a changed `sections[k].name`.
    # Asserting only "something was unregistered" passes for the wrong reason —
    # the trailing extra index alone would satisfy it (measured: registering
    # name changes left that weaker assertion green).
    displaced_names = [
        d for d in result.diffs
        if d.path.endswith(".name")
        and d.classification.startswith("unregistered")
        and d.old is not None
        and d.new is not None
    ]
    assert displaced_names, (
        "a section inserted mid-list must surface as a changed sections[k].name; "
        f"got {[(d.path, d.classification) for d in result.diffs]}"
    )


def test_registered_d2_description_rewrite_is_accepted_as_an_exact_pair():
    for factor_id, (old_text, new_text) in REGISTERED_SPEC_DESCRIPTION_REWRITES.items():
        old = _frozen_like()
        old["spec"]["description"] = old_text
        new = _with_registered_additions(_frozen_like())
        new["spec"]["description"] = new_text
        result = diff_report_json(
            old, new, name=factor_id, strict=True, correction_expected=False,
            description_rewrite=REGISTERED_SPEC_DESCRIPTION_REWRITES[factor_id],
        )
        assert result.ok, (factor_id, result.diffs)
        assert len(result.by_class("registered_d2_provenance_rewrite")) == 1


def test_a_DIFFERENT_description_rewrite_on_a_registered_factor_fails():
    # REVERSE: the registered OLD text but some other new text. This is the
    # case the exact pair exists for — the day a factor's stated meaning
    # really changes must not ride in on a provenance registration.
    factor_id = "peak_interval_kurtosis_20"
    old_text, _ = REGISTERED_SPEC_DESCRIPTION_REWRITES[factor_id]
    old = _frozen_like()
    old["spec"]["description"] = old_text
    new = _with_registered_additions(_frozen_like())
    new["spec"]["description"] = "Now computes something else entirely."
    result = diff_report_json(
        old, new, name=factor_id, strict=True, correction_expected=False,
        description_rewrite=REGISTERED_SPEC_DESCRIPTION_REWRITES[factor_id],
    )
    assert not result.ok
    assert len(result.by_class("unregistered_change")) == 1


def test_an_unregistered_factors_description_rewrite_fails():
    # REVERSE: a factor with no registered pair at all (the default None).
    old = _frozen_like()
    old["spec"]["description"] = "A"
    new = _with_registered_additions(_frozen_like())
    new["spec"]["description"] = "B"
    result = diff_report_json(
        old, new, name="jump_amount_corr_20", strict=True, correction_expected=False,
    )
    assert not result.ok
    assert len(result.by_class("unregistered_change")) == 1


@requires_frozen_panels
def test_the_registered_description_pairs_match_the_frozen_artifacts_exactly():
    """The transcribed constants must equal the bytes on disk, character for
    character. A pair that is one character off would leave the leg failing
    while the registration looked right in review."""
    from qt.exec_baseline_freeze import (
        DEFAULT_FROZEN_ROOT,
        DEFAULT_MANIFEST,
        FrozenExecBaseline,
    )
    from qt.factor_eval_reconcile import _FACTOR_TO_REPORT_NAME

    repo = Path(".").resolve()
    baseline = FrozenExecBaseline(repo / DEFAULT_FROZEN_ROOT, repo / DEFAULT_MANIFEST)
    for factor_id, (old_text, _new_text) in REGISTERED_SPEC_DESCRIPTION_REWRITES.items():
        frozen = baseline.report_json(_FACTOR_TO_REPORT_NAME[factor_id], "no_book")
        assert frozen["spec"]["description"] == old_text, factor_id


def test_md_prefixes_are_derived_per_FACTOR_not_per_section_name():
    """`ridge_scarcity_coverage` is published by two factors with DIFFERENT
    payloads (RidgeCoverage vs RidgeReturnCoverage), so a section-name-only
    lookup would hand one of them the other's field list."""
    from qt.factor_eval_reconcile import _registered_section_md_prefixes

    a = _registered_section_md_prefixes(
        "valley_ridge_vwap_ratio_20", ("ridge_scarcity_coverage",)
    )
    b = _registered_section_md_prefixes(
        "ridge_minute_return_20", ("ridge_scarcity_coverage",)
    )
    assert a[:2] == b[:2] == ("## + ridge_scarcity_coverage", "ridge scarcity:")
    assert set(a) != set(b), "the two payloads must not produce the same fields"
    assert "- valley_median:" in a and "- valley_median:" not in b
    assert "- ridge_return_mean:" in b and "- ridge_return_mean:" not in a


def test_md_prefixes_reject_a_factor_that_does_not_publish_that_section():
    from qt.factor_eval_reconcile import _registered_section_md_prefixes

    with pytest.raises(ValueError, match="does not publish the add-Section"):
        _registered_section_md_prefixes(
            "jump_amount_corr_20", ("ridge_scarcity_coverage",)
        )


@pytest.mark.parametrize(
    "section_name, payload_factory",
    [
        ("ridge_scarcity_coverage", "ridge"),
        ("peak_scarcity_coverage", "peak"),
        ("neutralization_coverage", "neutralization"),
    ],
)
def test_the_registered_note_prefix_matches_what_render_actually_emits(
    section_name, payload_factory
):
    """Pin the note prefixes by ASSERTION against a real rendering.

    The constants exist so the gate is not coupled to the renderer's format by
    construction; that only works if something checks they still agree.
    """
    from qt.factor_eval_reconcile import _SECTION_NOTE_PREFIXES

    frame = pd.DataFrame(
        {
            "classifiable_bars": [240, 240],
            "valley_bars": [200, 200],
            "ridge_bars": [25, 30],
            "peak_bars": [12, 14],
            "ridge_return_bars": [25, 30],
            "valid": [True, True],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-07-01"), "A"), (pd.Timestamp("2021-07-02"), "A")],
            names=["date", "symbol"],
        ),
    )
    if payload_factory == "ridge":
        from qt.factor_eval_disclosures import summarize_ridge_coverage as fn

        coverage = fn([frame])
    elif payload_factory == "peak":
        from qt.factor_eval_disclosures import summarize_peak_coverage as fn

        coverage = fn([frame])
    else:
        from qt.factor_eval_disclosures import summarize_neutralization as fn

        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-07-01"), f"S{i}") for i in range(12)],
            names=["date", "symbol"],
        )
        series = pd.Series(range(12), index=idx, dtype=float)
        coverage = fn(series, series, series, min_cross_section=10)
    assert coverage.render().startswith(_SECTION_NOTE_PREFIXES[section_name])


@pytest.mark.parametrize(
    "factor_id, section_name",
    [
        ("valley_ridge_vwap_ratio_20", "ridge_scarcity_coverage"),
        ("ridge_minute_return_20", "ridge_scarcity_coverage"),
        ("peak_ridge_amount_ratio_20", "peak_scarcity_coverage"),
        ("valley_price_quantile_20", "neutralization_coverage"),
    ],
)
def test_md_prefixes_cover_EVERY_key_the_real_section_payload_carries(
    factor_id, section_name
):
    """The registration must cover the payload a real run writes, key for key.

    Deriving the prefixes from ``dataclasses.fields`` alone missed the two
    COMPUTED properties ``to_section`` adds on top of ``asdict``, so three
    rendered Markdown lines stayed unregistered and the reports leg kept
    failing with the JSON side already green.

    ⚠️ REACH, measured: this test does NOT guard
    ``DERIVED_PAYLOAD_PROPERTIES`` itself. Both sides now read that tuple, so
    deleting an entry from it leaves this test GREEN by construction (measured:
    dropping ``validity_rate`` -> 4 passed). What catches that is the REAL
    reports leg (all three Markdown grids fail) and
    ``test_derived_payload_properties_are_exactly_the_coverage_properties``
    below, which pins the tuple against the payload classes themselves. This
    test's own job is narrower: a payload key with no dataclass field AND no
    entry in that tuple fails here rather than on a two-hour run.
    """
    from qt.factor_eval_disclosures import (
        NEUTRALIZATION_SECTION_NAME,
        disclosure_binding_for,
        summarize_neutralization,
        to_section,
    )
    from qt.factor_eval_reconcile import _registered_section_md_prefixes
    from factors import registry as factor_registry

    frame = pd.DataFrame(
        {
            "classifiable_bars": [240, 240],
            "valley_bars": [200, 200],
            "ridge_bars": [25, 30],
            "peak_bars": [12, 14],
            "ridge_return_bars": [25, 30],
            "valid": [True, True],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-07-01"), "A"), (pd.Timestamp("2021-07-02"), "A")],
            names=["date", "symbol"],
        ),
    )
    if section_name == NEUTRALIZATION_SECTION_NAME:
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2021-07-01"), f"S{i}") for i in range(12)],
            names=["date", "symbol"],
        )
        series = pd.Series(range(12), index=idx, dtype=float)
        coverage = summarize_neutralization(series, series, series, min_cross_section=10)
    else:
        binding = disclosure_binding_for(factor_registry.build(factor_id))
        coverage = binding.summarize([frame])

    payload_keys = set(to_section(section_name, coverage).payload)
    prefixes = set(_registered_section_md_prefixes(factor_id, (section_name,)))
    missing = {k for k in payload_keys if f"- {k}:" not in prefixes}
    assert not missing, (
        f"{factor_id}/{section_name}: payload keys with no registered Markdown "
        f"prefix -> their rendered lines would fail the reports leg: {sorted(missing)}"
    )


def test_derived_payload_properties_are_exactly_the_coverage_properties():
    """Pin ``DERIVED_PAYLOAD_PROPERTIES`` to the payload classes themselves.

    Both ``to_section`` and the reconcile prefix derivation read that tuple, so
    every test comparing one against the other is equal BY CONSTRUCTION and
    stays green when an entry is deleted (measured). The tuple therefore needs
    an anchor that is not itself derived from it: the union of ``@property``
    names declared on the coverage dataclasses. Delete an entry and this goes
    red; add a property to a coverage class without registering it and this
    goes red too — which is the direction that would otherwise ship an
    unregistered Markdown line.
    """
    from qt.factor_eval_disclosures import (
        DERIVED_PAYLOAD_PROPERTIES,
        NeutralizationCoverage,
        PeakCoverage,
        RidgeCoverage,
        RidgeReturnCoverage,
    )

    declared: set[str] = set()
    for cls in (RidgeCoverage, RidgeReturnCoverage, PeakCoverage, NeutralizationCoverage):
        declared |= {
            name
            for name, attr in vars(cls).items()
            if isinstance(attr, property) and not name.startswith("_")
        }
    # ``render`` is a method, not a property, and is carried as the section's
    # note rather than as a payload key — the set below is payload-bearing only.
    assert set(DERIVED_PAYLOAD_PROPERTIES) == declared, (
        "DERIVED_PAYLOAD_PROPERTIES must equal the coverage classes' property "
        f"names; registered={sorted(DERIVED_PAYLOAD_PROPERTIES)} "
        f"declared={sorted(declared)}"
    )
