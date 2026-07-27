"""The moved coverage disclosures: summarizers, add-Section bridge, gate constants.

The summarizer tests MOVED here from the three legacy runner test files (the
old files are untouched and still pass through the runners' re-exports);
catalogue BUG 6 closure: the two colliding
``test_summarize_ridge_coverage_handles_no_frames`` definitions are renamed
with factor prefixes in this home. PR-M's ``summarize_peak_coverage`` had ZERO
tests (catalogue BUG 3) — the counterfactual + empty-frames pair its three
siblings all have is NET-NEW here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, MANDATORY_SECTIONS, Section
from analytics.eval.render import canonical_sections
from analytics.eval.report import FactorEvalReport
from data.clean.intraday_amount_ratio import (
    PEAK_RIDGE_MIN_PEAK_BARS,
    PEAK_RIDGE_MIN_RIDGE_BARS,
)
from data.clean.intraday_ridge_return import RIDGE_RETURN_MIN_RIDGE_BARS
from data.clean.intraday_valley_ridge_vwap import (
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
)
from factors import registry as factor_registry
from factors.compute.minute import (
    peak_ridge_amount_ratio as peak_module,
)
from factors.compute.minute import (
    ridge_minute_return as ridge_return_module,
)
from factors.compute.minute import (
    valley_ridge_vwap_ratio as valley_ridge_module,
)
from qt.exec_basis_eval import _with_extra_sections
from qt.factor_eval_disclosures import (
    NeutralizationCoverage,
    disclosure_binding_for,
    summarize_neutralization,
    summarize_peak_coverage,
    summarize_ridge_coverage,
    summarize_ridge_return_coverage,
    to_section,
)


def _diag(columns: dict, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.DatetimeIndex(
            pd.bdate_range("2022-01-03", periods=days), name="trade_date"
        ),
    )


# --------------------------------------------------------------------------- #
# Moved: valley/ridge VWAP-ratio ridge-scarcity summarizer (PR-J)
# --------------------------------------------------------------------------- #
def test_valley_ridge_summarize_counterfactual_at_the_valley_floor():
    """The disclosure quantifies exactly what the LOWERED ridge floor buys."""
    diag = _diag(
        {
            "classifiable_bars": [240, 240, 240, 240],
            "valley_bars": [200, 200, 200, 200],
            "ridge_bars": [4, 12, 25, 30],
            # as the factor would mark them under the default floor of 10
            "valid": [False, True, True, True],
        },
        4,
    )
    cov = summarize_ridge_coverage([diag])
    assert cov.symbol_days == 4
    assert cov.classifiable_days == 4  # every day clears PR-F's classifiable floor
    assert cov.valid_days == 3
    # holding the ridge leg to the VALLEY floor (20) would keep only the 25 / 30 days
    assert cov.valid_days_at_valley_floor == 2
    assert cov.days_below_ridge_gate == 1
    assert cov.days_below_valley_gate == 0
    assert cov.ridge_mean == pytest.approx((4 + 12 + 25 + 30) / 4)
    # defaults are the PINNED production floors when the caller does not override
    assert cov.min_ridge_bars == VALLEY_RIDGE_MIN_RIDGE_BARS == 10
    assert cov.min_valley_bars == VALLEY_RIDGE_MIN_VALLEY_BARS == 20
    assert f"below_ridge_gate({VALLEY_RIDGE_MIN_RIDGE_BARS})" in cov.render()


def test_valley_ridge_summarize_handles_no_frames():
    cov = summarize_ridge_coverage([])
    assert cov.symbol_days == 0
    assert cov.classifiable_days == 0
    assert cov.valid_days == 0
    assert np.isnan(cov.validity_rate)
    assert cov.render()  # renders without dividing by zero


# --------------------------------------------------------------------------- #
# Moved: ridge-minute-return ridge-scarcity summarizer (PR-K)
# --------------------------------------------------------------------------- #
def test_ridge_return_summarize_reports_the_return_guard_attrition():
    """Ridge bars lost to the within-day lag must be VISIBLE, not silently absorbed."""
    diag = _diag(
        {
            "classifiable_bars": [240, 240],
            "ridge_bars": [10, 30],
            # one ridge on each day is the day's first visible bar -> no return
            "ridge_return_bars": [9, 29],
            "valid": [False, True],
        },
        2,
    )
    cov = summarize_ridge_return_coverage([diag])
    assert cov.ridge_bars_mean == pytest.approx(20.0)
    assert cov.ridge_return_mean == pytest.approx(19.0)
    assert cov.return_guard_attrition == pytest.approx(1.0 - 19.0 / 20.0)
    # the gate is applied to the RETURN-carrying count, so the 9-ridge day falls below 10
    assert cov.days_below_ridge_gate == 1
    assert "return_guard_attrition" in cov.render()


def test_ridge_return_summarize_counterfactual_at_the_comparison_floor():
    """The disclosure quantifies exactly what the scarcity floor buys (vs PR-J's 20)."""
    diag = _diag(
        {
            "classifiable_bars": [240, 240, 240, 240],
            "ridge_bars": [4, 12, 25, 30],
            "ridge_return_bars": [4, 12, 25, 30],
            # as the factor would mark them under the default floor of 10
            "valid": [False, True, True, True],
        },
        4,
    )
    cov = summarize_ridge_return_coverage([diag])
    assert cov.symbol_days == 4
    assert cov.classifiable_days == 4  # every day clears PR-F's classifiable floor
    assert cov.valid_days == 3
    # raising the ridge floor to PR-J's 20 would keep only the 25 / 30 days
    assert cov.valid_days_at_comparison_floor == 2
    assert cov.days_below_ridge_gate == 1
    assert cov.ridge_return_mean == pytest.approx((4 + 12 + 25 + 30) / 4)
    # defaults are the PINNED production floors when the caller does not override
    assert cov.min_ridge_bars == RIDGE_RETURN_MIN_RIDGE_BARS == 10
    assert f"below_ridge_gate({RIDGE_RETURN_MIN_RIDGE_BARS})" in cov.render()


def test_ridge_return_summarize_handles_no_frames():
    cov = summarize_ridge_return_coverage([])
    assert cov.symbol_days == 0
    assert cov.classifiable_days == 0
    assert cov.valid_days == 0
    assert np.isnan(cov.validity_rate)
    assert np.isnan(cov.return_guard_attrition)
    assert cov.render()  # renders without dividing by zero


# --------------------------------------------------------------------------- #
# NET-NEW (catalogue BUG 3): peak/ridge amount-ratio peak-scarcity summarizer
# --------------------------------------------------------------------------- #
def test_peak_summarize_counterfactual_at_the_ridge_floor():
    """The disclosure quantifies exactly what the LOWERED peak floor buys."""
    diag = _diag(
        {
            "classifiable_bars": [240, 240, 240, 240],
            "peak_bars": [2, 7, 12, 30],
            "ridge_bars": [30, 30, 30, 30],
            # as the factor would mark them under the default PEAK floor of 5
            "valid": [False, True, True, True],
        },
        4,
    )
    cov = summarize_peak_coverage([diag])
    assert cov.symbol_days == 4
    assert cov.classifiable_days == 4  # every day clears PR-F's classifiable floor
    assert cov.valid_days == 3
    # raising the PEAK leg to the RIDGE floor (10) would keep only the 12 / 30 days
    assert cov.valid_days_at_ridge_floor == 2
    assert cov.days_below_peak_gate == 1
    assert cov.days_below_ridge_gate == 0
    assert cov.peak_mean == pytest.approx((2 + 7 + 12 + 30) / 4)
    assert cov.ridge_median == pytest.approx(30.0)
    # defaults are the PINNED production floors when the caller does not override
    assert cov.min_peak_bars == PEAK_RIDGE_MIN_PEAK_BARS == 5
    assert cov.min_ridge_bars == PEAK_RIDGE_MIN_RIDGE_BARS == 10
    assert f"below_peak_gate({PEAK_RIDGE_MIN_PEAK_BARS})" in cov.render()
    assert f"below_ridge_gate({PEAK_RIDGE_MIN_RIDGE_BARS})" in cov.render()


def test_peak_summarize_handles_no_frames():
    cov = summarize_peak_coverage([])
    assert cov.symbol_days == 0
    assert cov.classifiable_days == 0
    assert cov.valid_days == 0
    assert np.isnan(cov.validity_rate)
    assert cov.render()  # renders without dividing by zero


# --------------------------------------------------------------------------- #
# Moved: reversal-neutralization summarizer (PR-L)
# --------------------------------------------------------------------------- #
def _panel(dates, syms, values, name):
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=idx, name=name)


def test_neutralization_summarize_counts_missing_reversal_rows():
    dates = pd.bdate_range("2023-01-02", periods=2)
    syms = ["A", "B", "C"]
    raw = _panel(dates, syms, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "q")
    rev = _panel(dates, syms, [[1.0, 2.0, 3.0], [1.0, np.nan, np.nan]], "r")
    resid = _panel(dates, syms, [[0.0, 0.0, 0.0], [np.nan] * 3], "f")
    cov = summarize_neutralization(raw, rev, resid, min_cross_section=3)
    assert cov.raw_rows == 6
    assert cov.rev_rows == 4        # only four rows had BOTH
    assert cov.residual_rows == 3
    assert cov.dates_total == 2
    assert cov.dates_residualized == 1
    assert cov.cross_section_min == 1 and cov.cross_section_max == 3


def test_neutralization_summarize_handles_an_all_missing_reversal():
    dates = pd.bdate_range("2023-01-02", periods=1)
    syms = ["A", "B"]
    raw = _panel(dates, syms, [[0.1, 0.2]], "q")
    rev = _panel(dates, syms, [[np.nan, np.nan]], "r")
    resid = _panel(dates, syms, [[np.nan, np.nan]], "f")
    cov = summarize_neutralization(raw, rev, resid, min_cross_section=3)
    assert cov.rev_rows == 0
    assert cov.residual_rows == 0
    assert cov.dates_residualized == 0


def test_neutralization_render_matches_the_former_cli_inline_line():
    """render() is the line the CLI used to inline (catalogue section 3 one-site normalization) — pinned."""
    cov = NeutralizationCoverage(
        raw_rows=6, rev_rows=4, residual_rows=3, dates_total=2, dates_residualized=1,
        cross_section_min=1, cross_section_median=2.0, cross_section_max=3,
        raw_rev_spearman_mean=0.123456,
    )
    assert cov.render() == (
        "neutralization (T-1 rev20): raw_rows=6 rev_paired=4 residual_rows=3 "
        "dates=1/2 cross_section min/med/max=1/2.0/3 "
        "mean_spearman(raw,rev20)=+0.1235"
    )


# --------------------------------------------------------------------------- #
# Gate-constant single source: summarizer defaults == the compute gates
# --------------------------------------------------------------------------- #
def _defaults(fn) -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_summarizer_defaults_are_the_gates_the_binding_applies():
    """The disclosure must report the floors the RUN applied, not a hand copy.

    The minute binding (``factors/compute/minute/binding.py``) calls every
    compute function with all gate parameters at their module defaults, so the
    summarizer defaults and the compute-signature defaults must agree — and
    both must be the SAME constants the factor modules define (not a second
    literal that can drift, the #76/#78 lesson).
    """
    ridge = _defaults(summarize_ridge_coverage)
    compute = _defaults(valley_ridge_module.compute_valley_ridge_vwap_ratio)
    for key in ("min_ridge_bars", "min_valley_bars", "min_classifiable"):
        assert ridge[key] == compute[key]

    ret = _defaults(summarize_ridge_return_coverage)
    compute = _defaults(ridge_return_module.compute_ridge_minute_return)
    for key in ("min_ridge_bars", "min_classifiable"):
        assert ret[key] == compute[key]

    peak = _defaults(summarize_peak_coverage)
    compute = _defaults(peak_module.compute_peak_ridge_amount_ratio)
    for key in ("min_peak_bars", "min_ridge_bars", "min_classifiable"):
        assert peak[key] == compute[key]

    # the counterfactual floors: PR-K's comparison floor is PR-J's valley floor
    # (a documented cross-factor anchor, hardcoded as 20 with the reason on it);
    # PR-M's is DERIVED from the ridge gate, never a hardcoded duplicate.
    assert ret["comparison_floor"] == 20 == VALLEY_RIDGE_MIN_VALLEY_BARS
    assert peak["counterfactual_peak_floor"] == PEAK_RIDGE_MIN_RIDGE_BARS


def test_disclosure_binding_covers_exactly_the_three_publishing_factors():
    for fid in (
        "valley_ridge_vwap_ratio_20",
        "ridge_minute_return_20",
        "peak_ridge_amount_ratio_20",
    ):
        binding = disclosure_binding_for(factor_registry.build(fid))
        assert binding is not None, fid
        assert binding.section_name.endswith("_coverage")
    # every other factor publishes NO per-day disclosure — stated, not inferred
    assert disclosure_binding_for(factor_registry.build("jump_amount_corr_20")) is None
    assert disclosure_binding_for(factor_registry.build("value_ep")) is None


# --------------------------------------------------------------------------- #
# to_section: the add-Section bridge (contract section 3.6)
# --------------------------------------------------------------------------- #
def _verdicted_report() -> FactorEvalReport:
    spec = factor_registry.build("jump_amount_corr_20").spec
    cfg = EvalConfig(
        universe="000905.SH",
        universe_is_pit=True,
        start="2021-07-01",
        end="2026-06-30",
        is_exploratory=True,
        post_hoc_selected=False,
        oos_split="2024-01-01",
        view="decision",
        return_basis="exec_to_exec",
    )
    sections = [Section(name=n, payload={}) for n in MANDATORY_SECTIONS]
    return FactorEvalReport.assemble(spec, cfg, sections).with_verdict()


def test_to_section_packs_fields_properties_and_the_render_note():
    cov = summarize_ridge_coverage([])
    section = to_section("ridge_scarcity_coverage", cov)
    assert section.name == "ridge_scarcity_coverage"
    # dataclass fields land in the payload...
    assert section.payload["symbol_days"] == 0
    assert section.payload["min_ridge_bars"] == VALLEY_RIDGE_MIN_RIDGE_BARS
    # ...the derived property rides along...
    assert "validity_rate" in section.payload
    # ...and the note is the one-line render, so artifact and log cannot disagree
    assert section.note == cov.render()
    # a disclosure without that property must not invent it
    peak = to_section("peak_scarcity_coverage", summarize_peak_coverage([]))
    assert "return_guard_attrition" not in peak.payload
    assert "validity_rate" in peak.payload


def test_extra_section_never_moves_the_verdict_or_mandatory_sections():
    base = _verdicted_report()
    extra = to_section("ridge_scarcity_coverage", summarize_ridge_coverage([]))
    augmented = FactorEvalReport.assemble(
        base.spec, base.cfg, [*base.sections, extra], thresholds=base.thresholds
    ).with_verdict()
    # the mandatory sections are the ORIGINALS, in order...
    assert augmented.sections[: len(base.sections)] == base.sections
    # ...the verdict is bit-identical (it reads only the mandatory payloads)...
    assert augmented.verdict == base.verdict
    # ...and canonical order is mandatory first, extras sorted by name after
    ordered = [name for name, _ in canonical_sections(augmented, MANDATORY_SECTIONS)]
    assert ordered == [*MANDATORY_SECTIONS, "ridge_scarcity_coverage"]


def test_exec_basis_augmentation_seam_preserves_report_and_verdict():
    base = _verdicted_report()
    extra = to_section("peak_scarcity_coverage", summarize_peak_coverage([]))
    augmented = _with_extra_sections(base, [extra])
    assert augmented.sections[: len(base.sections)] == base.sections
    assert augmented.verdict == base.verdict
    assert augmented.by_name()["peak_scarcity_coverage"] is extra or (
        augmented.by_name()["peak_scarcity_coverage"] == extra
    )


def test_an_extra_section_may_never_shadow_a_mandatory_name():
    base = _verdicted_report()
    bad = Section(name="caveats", payload={"smuggled": True})
    with pytest.raises(ValueError, match="duplicate report section"):
        FactorEvalReport.assemble(
            base.spec, base.cfg, [*base.sections, bad], thresholds=base.thresholds
        )
