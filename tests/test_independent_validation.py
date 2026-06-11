"""P3-7: genuinely independent sample validation for value/lowvol (network-free).

Locks the independent-validation contract on top of the P3-6 subset layer:
  * ``subset_validation.independent_cells`` / ``hypotheses`` / ``min_rebalances``
    validate (cells must reference declared robustness cells and must not be
    skip-listed; hypotheses must reference ENABLED factors with a
    positive/negative literal; min_rebalances > 0); the old P3-6 config still
    validates with defaults (no behaviour change — group/cost logic untouched);
  * every run cell gets an explicit sample class (independent holdout vs
    screened/post-hoc);
  * cross-cell summaries are computed PER SAMPLE CLASS — screened numbers can
    never leak into the independent summary or vice versa;
  * the independent verdict is a factual sign check per pre-declared
    hypothesis (holds = expected IC sign in BOTH subperiods), with a
    sample-sufficiency gate (INSUFFICIENT-DATA discloses size and reason);
  * the report carries a sample column, separate per-class summary sections,
    and an independent-verdict section that contains ONLY independent cells.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import yaml

from qt.config import ConfigError, load_config

_INDEP_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_independent_validation.yaml"
)
_P36_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_subset_costs.yaml"
)


def _mutate(tmp_path, name="i.yaml", subset_patch=None, drop=(), **root_patch):
    raw = yaml.safe_load(Path(_INDEP_CONFIG).read_text(encoding="utf-8"))
    for key in drop:
        raw.pop(key, None)
    for key, value in (subset_patch or {}).items():
        raw["subset_validation"][key] = value
    for key, value in root_patch.items():
        raw[key] = value
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_independent_config_validates():
    cfg = load_config(_INDEP_CONFIG)
    sv = cfg.subset_validation
    cells = [(c.universe, c.window) for c in sv.independent_cells]
    assert ("000016.SH", "2024-2026") in cells
    assert ("000300.SH", "2024-2026") in cells
    assert ("000016.SH", "2022-2024") not in cells  # the screened anchor
    assert sv.hypotheses == {
        "value_ep": "positive", "value_bp": "positive", "volatility_20": "negative",
    }
    assert sv.min_rebalances == 8


def test_old_p36_config_still_validates_with_defaults():
    """The P3-6 config carries no P3-7 keys and must keep validating unchanged."""
    cfg = load_config(_P36_CONFIG)
    sv = cfg.subset_validation
    assert sv.independent_cells == []
    assert sv.hypotheses == {}
    assert sv.min_rebalances == 8  # default exists but is inert without cells


def test_independent_cell_unknown_universe_rejected(tmp_path):
    bad = [{"universe": "999999.SH", "window": "2024-2026"}]
    with pytest.raises(ConfigError, match="999999.SH"):
        load_config(_mutate(tmp_path, subset_patch={"independent_cells": bad}))


def test_independent_cell_unknown_window_rejected(tmp_path):
    bad = [{"universe": "000016.SH", "window": "2030-2032"}]
    with pytest.raises(ConfigError, match="2030-2032"):
        load_config(_mutate(tmp_path, subset_patch={"independent_cells": bad}))


def test_independent_cell_cannot_be_skip_listed(tmp_path):
    """Claiming to independently validate a cell that never runs is a config bug."""
    bad = [{"universe": "000300.SH", "window": "2022-2024"}]  # in skip_cells
    with pytest.raises(ConfigError, match="skip"):
        load_config(_mutate(tmp_path, subset_patch={"independent_cells": bad}))


def test_independent_cells_require_robustness_section(tmp_path):
    with pytest.raises(ConfigError, match="robustness"):
        load_config(_mutate(tmp_path, drop=("robustness",)))


def test_hypotheses_must_reference_enabled_factors(tmp_path):
    with pytest.raises(ConfigError, match="no_such_factor"):
        load_config(_mutate(
            tmp_path, subset_patch={"hypotheses": {"no_such_factor": "positive"}}
        ))


def test_hypothesis_sign_literal_enforced(tmp_path):
    with pytest.raises(ConfigError, match="positive|negative"):
        load_config(_mutate(
            tmp_path, subset_patch={"hypotheses": {"value_ep": "up"}}
        ))


def test_min_rebalances_must_be_positive(tmp_path):
    with pytest.raises(ConfigError, match="positive"):
        load_config(_mutate(tmp_path, subset_patch={"min_rebalances": 0}))


# --------------------------------------------------------------------------- #
# sample-class labeling
# --------------------------------------------------------------------------- #
def test_sample_class_labels_cells():
    from qt.subset_validation import sample_class

    cfg = load_config(_INDEP_CONFIG)
    assert sample_class(cfg, "000016.SH", "2022-2024") == "screened"
    assert sample_class(cfg, "000016.SH", "2024-2026") == "independent"
    assert sample_class(cfg, "000300.SH", "2024-2026") == "independent"


def test_sample_class_defaults_to_screened_for_old_config():
    from qt.subset_validation import sample_class

    cfg = load_config(_P36_CONFIG)
    assert sample_class(cfg, "000016.SH", "2022-2024") == "screened"


# --------------------------------------------------------------------------- #
# independent verdict (pure)
# --------------------------------------------------------------------------- #
def _ic(train, test):
    return {"train": {"ic_mean": train, "ic_ir": 0.1, "hit_rate": 0.5, "n": 100},
            "test": {"ic_mean": test, "ic_ir": 0.1, "hit_rate": 0.5, "n": 100}}


_HYP = {"value_ep": "positive", "value_bp": "positive", "volatility_20": "negative"}


def test_verdict_supported_when_all_hypotheses_hold():
    from qt.subset_validation import independent_verdict

    raw = {"value_ep": _ic(0.03, 0.05), "value_bp": _ic(0.02, 0.04),
           "volatility_20": _ic(-0.05, -0.06), "momentum_20": _ic(-0.01, 0.01)}
    v = independent_verdict(raw, _HYP, n_settled=20, min_rebalances=8)
    assert v["status"] == "SUPPORTED"
    assert v["n_holds"] == 3 and v["n_hypotheses"] == 3
    assert v["factors"]["volatility_20"]["holds"] is True
    assert "momentum_20" not in v["factors"]  # only hypothesis factors judged


def test_verdict_partial_and_not_supported():
    from qt.subset_validation import independent_verdict

    raw_partial = {"value_ep": _ic(0.03, 0.05), "value_bp": _ic(0.02, -0.01),
                   "volatility_20": _ic(-0.05, -0.06)}
    v = independent_verdict(raw_partial, _HYP, n_settled=20, min_rebalances=8)
    assert v["status"] == "PARTIAL" and v["n_holds"] == 2
    assert v["factors"]["value_bp"]["holds_train"] is True
    assert v["factors"]["value_bp"]["holds_test"] is False

    raw_none = {"value_ep": _ic(-0.03, -0.05), "value_bp": _ic(-0.02, -0.01),
                "volatility_20": _ic(0.05, 0.06)}
    v2 = independent_verdict(raw_none, _HYP, n_settled=20, min_rebalances=8)
    assert v2["status"] == "NOT SUPPORTED" and v2["n_holds"] == 0


def test_verdict_insufficient_data_overrides_sign_check():
    from qt.subset_validation import independent_verdict

    raw = {"value_ep": _ic(0.03, 0.05), "value_bp": _ic(0.02, 0.04),
           "volatility_20": _ic(-0.05, -0.06)}
    v = independent_verdict(raw, _HYP, n_settled=5, min_rebalances=8)
    assert v["status"] == "INSUFFICIENT-DATA"
    assert v["n_settled"] == 5 and v["min_rebalances"] == 8
    assert "5" in v["reason"] and "8" in v["reason"]
    # the factor table is still reported for transparency
    assert v["factors"]["value_ep"]["holds"] is True


def test_verdict_nan_ic_never_holds():
    from qt.subset_validation import independent_verdict

    raw = {"value_ep": _ic(float("nan"), 0.05), "value_bp": _ic(0.02, 0.04),
           "volatility_20": _ic(-0.05, -0.06)}
    v = independent_verdict(raw, _HYP, n_settled=20, min_rebalances=8)
    assert v["factors"]["value_ep"]["holds"] is False
    assert v["status"] == "PARTIAL"


def test_verdict_missing_hypothesis_factor_never_holds():
    from qt.subset_validation import independent_verdict

    raw = {"value_bp": _ic(0.02, 0.04), "volatility_20": _ic(-0.05, -0.06)}
    v = independent_verdict(raw, _HYP, n_settled=20, min_rebalances=8)
    assert v["factors"]["value_ep"]["holds"] is False
    assert math.isnan(v["factors"]["value_ep"]["train_ic"])
    assert v["status"] == "PARTIAL"


# --------------------------------------------------------------------------- #
# per-class summaries — screened and independent never mix
# --------------------------------------------------------------------------- #
def test_summaries_by_sample_class_never_mix():
    from qt.subset_validation import summarize_by_sample
    from tests.test_subset_validation import _fake_group

    cells = {
        "S|old": {"groups": {"g": _fake_group(
            eq_test=-0.05, ic_test_by_scn={"base": -0.02},
            combo_ic_test=0.0111, consistent=True)}},
        "I|new": {"groups": {"g": _fake_group(
            eq_test=-0.01, ic_test_by_scn={"base": -0.07},
            combo_ic_test=-0.0222, consistent=False)}},
    }
    samples = {"S|old": "screened", "I|new": "independent"}
    by_class = summarize_by_sample(cells, samples, base_scenario="base")
    assert set(by_class) == {"screened", "independent"}
    scr = by_class["screened"]["groups"]["g"]
    ind = by_class["independent"]["groups"]["g"]
    # per-class cell counts and per-cell attributions are disjoint
    assert by_class["screened"]["n_cells"] == 1
    assert by_class["independent"]["n_cells"] == 1
    assert list(scr["combo"]["combo_ic_weighted"]["test_ic_by_cell"]) == ["S|old"]
    assert list(ind["combo"]["combo_ic_weighted"]["test_ic_by_cell"]) == ["I|new"]
    # the screened cell's IC value never appears under independent (and v.v.)
    assert scr["combo"]["combo_ic_weighted"]["test_ic_by_cell"]["S|old"] == 0.0111
    assert ind["combo"]["combo_ic_weighted"]["test_ic_by_cell"]["I|new"] == -0.0222


# --------------------------------------------------------------------------- #
# report rendering — sample column, split sections, verdict separation
# --------------------------------------------------------------------------- #
def _synthetic_independent_result():
    import pandas as pd

    from qt.subset_validation import (
        independent_verdict,
        summarize_by_sample,
        summarize_subset_matrix,
    )
    from tests.test_subset_validation import _synthetic_subset_result

    base = _synthetic_subset_result()
    screened_label = "000016.SH|2022-2024"
    indep_label = "000016.SH|2024-2026"
    screened_cell = base.cells[screened_label]
    indep_raw = {
        "value_ep": _ic(0.03, 0.05),
        "value_bp": _ic(0.02, 0.04),
        "volatility_20": _ic(-0.05, -0.06),
        "momentum_20": _ic(-0.01, 0.01),
    }
    indep_cell = dataclasses.replace(
        screened_cell,
        split_date=pd.Timestamp("2025-07-01"),
        train_start=pd.Timestamp("2024-07-01"),
        train_end=pd.Timestamp("2025-06-30"),
        test_start=pd.Timestamp("2025-07-01"),
        test_end=pd.Timestamp("2026-05-29"),
        raw_ic_stats=indep_raw,
        raw_sign_consistency={k: True for k in indep_raw},
    )
    cells = {screened_label: screened_cell, indep_label: indep_cell}
    samples = {screened_label: "screened", indep_label: "independent"}
    plain = {label: {"groups": c.groups} for label, c in cells.items()}
    verdict = independent_verdict(indep_raw, _HYP, n_settled=21, min_rebalances=8)
    return dataclasses.replace(
        base,
        cells=cells,
        cell_runtimes={screened_label: 900.0, indep_label: 1100.0},
        summary=summarize_subset_matrix(plain, base_scenario="base"),
        cell_samples=samples,
        sample_summaries=summarize_by_sample(plain, samples, base_scenario="base"),
        verdicts={indep_label: verdict},
    )


def test_render_labels_cells_and_splits_summary_sections():
    from qt.reports import render_subset_validation

    md = render_subset_validation(_synthetic_independent_result())
    assert "independent" in md and "screened" in md  # sample column values
    # per-class cross-cell sections, separated — never one mixed table
    assert "Independent holdout cells" in md
    assert "Screened (post-hoc) cells" in md
    assert "never averaged" in md.lower()


def test_render_verdict_section_contains_only_independent_cells():
    from qt.reports import render_subset_validation

    md = render_subset_validation(_synthetic_independent_result())
    assert "## Independent holdout verdict" in md
    verdict_block = md.split("## Independent holdout verdict")[1].split("\n## ")[0]
    assert "000016.SH|2024-2026" in verdict_block
    assert "000016.SH|2022-2024" not in verdict_block  # screened cell stays out
    assert "SUPPORTED" in verdict_block
    # sample size + hypothesis criterion disclosed
    assert "rebalances" in verdict_block.lower()
    assert "value_ep" in verdict_block and "volatility_20" in verdict_block
    assert "BOTH subperiods" in verdict_block


def test_render_insufficient_data_verdict_disclosed():
    from qt.reports import render_subset_validation
    from qt.subset_validation import independent_verdict

    result = _synthetic_independent_result()
    short = independent_verdict(
        result.cells["000016.SH|2024-2026"].raw_ic_stats,
        _HYP, n_settled=5, min_rebalances=8,
    )
    result = dataclasses.replace(result, verdicts={"000016.SH|2024-2026": short})
    md = render_subset_validation(result)
    assert "INSUFFICIENT-DATA" in md
    block = md.split("## Independent holdout verdict")[1].split("\n## ")[0]
    assert "5" in block and "8" in block  # n_settled vs threshold disclosed


def test_render_without_independent_cells_keeps_p36_shape():
    """A P3-6-era result (defaults) renders exactly the old report shape."""
    from qt.reports import render_subset_validation
    from tests.test_subset_validation import _synthetic_subset_result

    md = render_subset_validation(_synthetic_subset_result())
    assert "## Independent holdout verdict" not in md
    assert "## Cross-cell summary by group" in md


def test_render_independent_result_leaks_no_secret():
    from qt.reports import render_subset_validation

    result = _synthetic_independent_result()
    md = render_subset_validation(result)
    assert result.config.data.external_secret_file not in md
    assert "token" not in md.lower()
