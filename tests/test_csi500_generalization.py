"""P3-8: CSI500 independent generalization config + report-name (network-free).

Locks the P3-8 contract on top of the unchanged P3-7 machinery:
  * the CSI500 config validates and labels its cells correctly (SSE50|2022-2024
    = screened anchor; SSE50|2024-2026 and 000905.SH|2024-2026 = independent
    holdouts; CSI500|2022-2024 skipped + disclosed);
  * ``output.subset_report_name`` lets a subset-validation config own its
    report filename (default None keeps the historical
    ``phase3_subset_validation.md`` — behaviour-preserving, the
    ``baseline_report_name`` precedent), so a P3-8 run no longer clobbers the
    accepted P3-7 artifact;
  * the P3-6/P3-7 configs keep validating unchanged (their own test files lock
    the report shape; nothing else in the machinery moves).
"""

from __future__ import annotations

from pathlib import Path

from qt.config import load_config

_CSI500_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_csi500_generalization.yaml"
)
_P37_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_independent_validation.yaml"
)


def test_csi500_config_validates_with_expected_cells():
    cfg = load_config(_CSI500_CONFIG)
    assert "000905.SH" in cfg.robustness.universes
    assert "000016.SH" in cfg.robustness.universes
    skipped = {(s.universe, s.window) for s in cfg.robustness.skip_cells}
    assert ("000905.SH", "2022-2024") in skipped  # runtime budget, disclosed
    sv = cfg.subset_validation
    indep = {(c.universe, c.window) for c in sv.independent_cells}
    assert ("000905.SH", "2024-2026") in indep  # THE new cell
    assert ("000016.SH", "2024-2026") in indep  # P3-7 reproducibility anchor
    assert ("000016.SH", "2022-2024") not in indep  # screened anchor
    # same hypotheses / groups / scenarios as P3-7 (no tuning, no new factors)
    assert sv.hypotheses == {
        "value_ep": "positive", "value_bp": "positive", "volatility_20": "negative",
    }
    assert [g.label for g in sv.groups] == [
        "legacy_trio", "full_pack", "value_lowvol", "value_lowvol_liq",
    ]
    assert {s.label: s.fee_multiplier for s in sv.cost_scenarios} == {
        "base": 1.0, "2x": 2.0, "high_cost": 4.0,
    }


def test_csi500_cells_sample_classes():
    from qt.subset_validation import sample_class

    cfg = load_config(_CSI500_CONFIG)
    assert sample_class(cfg, "000016.SH", "2022-2024") == "screened"
    assert sample_class(cfg, "000016.SH", "2024-2026") == "independent"
    assert sample_class(cfg, "000905.SH", "2024-2026") == "independent"


def test_subset_report_name_default_preserves_old_filename():
    """Configs without subset_report_name keep the historical report path."""
    from qt.subset_validation import subset_report_filename

    p36 = load_config(str(
        Path(__file__).resolve().parents[1]
        / "config" / "phase3_real_subset_costs.yaml"
    ))
    p37 = load_config(_P37_CONFIG)
    assert subset_report_filename(p36) == "phase3_subset_validation.md"
    assert subset_report_filename(p37) == "phase3_subset_validation.md"


def test_csi500_config_owns_its_report_filename():
    """The P3-8 run must not clobber the accepted P3-7 artifact."""
    from qt.subset_validation import subset_report_filename

    cfg = load_config(_CSI500_CONFIG)
    name = subset_report_filename(cfg)
    assert name == "phase3_csi500_generalization.md"
    assert name != "phase3_subset_validation.md"


def test_csi500_config_sets_its_report_title():
    cfg = load_config(_CSI500_CONFIG)
    assert cfg.output.subset_report_title is not None
    assert "Phase 3-8" in cfg.output.subset_report_title
    assert "CSI500" in cfg.output.subset_report_title


def test_p36_p37_configs_leave_report_title_unset():
    """The pre-P3-8 configs must keep the renderer's sample-aware default title."""
    p36 = load_config(str(
        Path(__file__).resolve().parents[1]
        / "config" / "phase3_real_subset_costs.yaml"
    ))
    p37 = load_config(_P37_CONFIG)
    assert p36.output.subset_report_title is None
    assert p37.output.subset_report_title is None


def _csi500_synthetic_result():
    """A P3-7-shaped synthetic result re-pointed at the CSI500 config (which
    sets subset_report_title) — exercises the config-driven H1 title."""
    import dataclasses

    from tests.test_independent_validation import _synthetic_independent_result

    return dataclasses.replace(
        _synthetic_independent_result(), config=load_config(_CSI500_CONFIG)
    )


def test_csi500_report_title_is_p38_not_p37():
    """The generated CSI500 report's first line must name THIS study, not the
    default P3-7 phase label (the review finding: hardcoded P3-7 title)."""
    from qt.reports import render_subset_validation

    md = render_subset_validation(_csi500_synthetic_result())
    first_line = md.split("\n", 1)[0]
    assert first_line.startswith("# Phase 3-8")
    assert "CSI500" in first_line
    assert "Phase 3-7" not in first_line
    # the body framing stays sample-aware (screened vs independent) — still correct
    assert "independent holdout" in md.lower() and "screened" in md.lower()
    assert "## Independent holdout verdict" in md


def test_default_subset_title_preserved_without_config_override():
    """A result whose config sets no subset_report_title keeps the sample-aware
    default (P3-7 for an independent run) — regression guard for P3-6/P3-7."""
    from qt.reports import render_subset_validation
    from tests.test_independent_validation import _synthetic_independent_result

    md = render_subset_validation(_synthetic_independent_result())
    assert md.split("\n", 1)[0].startswith("# Phase 3-7")
