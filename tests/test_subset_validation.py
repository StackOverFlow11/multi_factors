"""P3-6: value+lowvol subset re-check + cost sensitivity (network-free).

Locks the subset-validation contract:
  * the ``subset_validation`` config section validates (non-empty groups with
    unique labels, factors referencing ENABLED config factors only, positive
    fee multipliers, a mandatory multiplier-1.0 base scenario);
  * groups are re-processed independently from the shared raw factor panel
    (drop_missing is PER GROUP — never the full pack's mask) and a group with
    all columns reproduces the P3-4/P3-5 processing bitwise (no drift);
  * cost scenarios scale ONLY the fee: trades / turnover / gross returns are
    identical across scenarios, the cost line scales linearly;
  * ``_run_backtest_for``'s new ``fee_rate`` parameter is default-preserving;
  * cross-cell aggregation attributes strictly per cell and per group;
  * the report discloses groups, scenarios, skipped cells, the POST-HOC
    selection caveat and the not-a-return-claim caveat, and leaks no secret.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qt.config import ConfigError, load_config

_SUBSET_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_subset_costs.yaml"
)


def _mutate_subset(tmp_path, name="s.yaml", **patch):
    """Copy the real subset config, patch subset_validation subkeys."""
    raw = yaml.safe_load(Path(_SUBSET_CONFIG).read_text(encoding="utf-8"))
    for key, value in patch.items():
        raw["subset_validation"][key] = value
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def _mutate_root(tmp_path, name="r.yaml", drop=(), **patch):
    """Copy the real subset config, patch/drop TOP-LEVEL sections."""
    raw = yaml.safe_load(Path(_SUBSET_CONFIG).read_text(encoding="utf-8"))
    for key in drop:
        raw.pop(key, None)
    for key, value in patch.items():
        raw[key] = value
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_subset_config_validates():
    cfg = load_config(_SUBSET_CONFIG)
    sv = cfg.subset_validation
    assert sv is not None
    labels = [g.label for g in sv.groups]
    assert "legacy_trio" in labels and "full_pack" in labels
    assert "value_lowvol" in labels and "value_lowvol_liq" in labels
    by_label = {g.label: g for g in sv.groups}
    assert by_label["legacy_trio"].factors == ["momentum_20", "roe", "netprofit_yoy"]
    assert by_label["value_lowvol"].factors == ["value_ep", "value_bp", "volatility_20"]
    assert len(by_label["full_pack"].factors) == 11
    scn = {s.label: s.fee_multiplier for s in sv.cost_scenarios}
    assert scn == {"base": 1.0, "2x": 2.0, "high_cost": 4.0}
    # the matrix shape + alpha guard prerequisites are in place
    assert cfg.robustness is not None
    assert cfg.alpha.model == "ic_weighted"


def test_subset_config_default_scenarios_is_single_base(tmp_path):
    raw = yaml.safe_load(Path(_SUBSET_CONFIG).read_text(encoding="utf-8"))
    raw["subset_validation"].pop("cost_scenarios")
    p = tmp_path / "default_scn.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(str(p))
    scn = cfg.subset_validation.cost_scenarios
    assert len(scn) == 1
    assert scn[0].fee_multiplier == 1.0


def test_subset_config_rejects_empty_groups(tmp_path):
    with pytest.raises(ConfigError, match="at least one group"):
        load_config(_mutate_subset(tmp_path, groups=[]))


def test_subset_config_rejects_duplicate_group_labels(tmp_path):
    groups = [
        {"label": "g", "factors": ["momentum_20"]},
        {"label": "g", "factors": ["roe"]},
    ]
    with pytest.raises(ConfigError, match="[Dd]uplicate"):
        load_config(_mutate_subset(tmp_path, groups=groups))


def test_subset_config_rejects_empty_group_factors(tmp_path):
    groups = [{"label": "g", "factors": []}]
    with pytest.raises(ConfigError, match="at least one factor"):
        load_config(_mutate_subset(tmp_path, groups=groups))


def test_subset_config_rejects_duplicate_factor_in_group(tmp_path):
    groups = [{"label": "g", "factors": ["momentum_20", "momentum_20"]}]
    with pytest.raises(ConfigError, match="[Dd]uplicate"):
        load_config(_mutate_subset(tmp_path, groups=groups))


def test_subset_config_rejects_unknown_group_factor(tmp_path):
    groups = [{"label": "g", "factors": ["no_such_factor"]}]
    with pytest.raises(ConfigError, match="no_such_factor"):
        load_config(_mutate_subset(tmp_path, groups=groups))


def test_subset_config_rejects_disabled_group_factor(tmp_path):
    """A group factor must be ENABLED — a disabled one has no raw panel column."""
    raw = yaml.safe_load(Path(_SUBSET_CONFIG).read_text(encoding="utf-8"))
    raw["factors"].append(
        {"name": "momentum_5", "enabled": False, "params": {"window": 5}}
    )
    raw["subset_validation"]["groups"] = [
        {"label": "g", "factors": ["momentum_5"]}
    ]
    p = tmp_path / "disabled.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="momentum_5"):
        load_config(str(p))


def test_subset_config_rejects_nonpositive_multiplier(tmp_path):
    scenarios = [
        {"label": "base", "fee_multiplier": 1.0},
        {"label": "free", "fee_multiplier": 0.0},
    ]
    with pytest.raises(ConfigError, match="positive"):
        load_config(_mutate_subset(tmp_path, cost_scenarios=scenarios))


def test_subset_config_requires_base_scenario(tmp_path):
    """Cost drag needs an anchor: one scenario MUST have fee_multiplier == 1.0."""
    scenarios = [{"label": "2x", "fee_multiplier": 2.0}]
    with pytest.raises(ConfigError, match="1.0"):
        load_config(_mutate_subset(tmp_path, cost_scenarios=scenarios))


def test_subset_config_rejects_duplicate_scenario_labels(tmp_path):
    scenarios = [
        {"label": "base", "fee_multiplier": 1.0},
        {"label": "base", "fee_multiplier": 2.0},
    ]
    with pytest.raises(ConfigError, match="[Dd]uplicate"):
        load_config(_mutate_subset(tmp_path, cost_scenarios=scenarios))


# --------------------------------------------------------------------------- #
# runner guards (no network — every guard fires before any feed is built)
# --------------------------------------------------------------------------- #
def test_subset_runner_rejects_demo_source(tmp_path):
    from qt.subset_validation import run_phase3_subset

    path = _mutate_root(
        tmp_path, data={**yaml.safe_load(Path(_SUBSET_CONFIG).read_text())["data"],
                        "source": "demo"},
        universe={"type": "static", "symbols": ["000001.SZ"], "index_code": None,
                  "min_listing_days": 0,
                  "filters": {"missing_close": True, "suspended": False,
                              "st": False, "limit_up_down": False}},
    )
    with pytest.raises(ValueError, match="tushare|REAL"):
        run_phase3_subset(path)


def test_subset_runner_requires_subset_section(tmp_path):
    from qt.subset_validation import run_phase3_subset

    with pytest.raises(ValueError, match="subset_validation"):
        run_phase3_subset(_mutate_root(tmp_path, drop=("subset_validation",)))


def test_subset_runner_requires_robustness_section(tmp_path):
    from qt.subset_validation import run_phase3_subset

    with pytest.raises(ValueError, match="robustness"):
        run_phase3_subset(_mutate_root(tmp_path, drop=("robustness",)))


def test_subset_runner_rejects_non_ic_weighted_alpha(tmp_path):
    from qt.subset_validation import run_phase3_subset

    with pytest.raises(ValueError, match="ic_weighted"):
        run_phase3_subset(
            _mutate_root(tmp_path, alpha={"model": "equal_weight", "params": {}})
        )


# --------------------------------------------------------------------------- #
# pure cost statistics
# --------------------------------------------------------------------------- #
def test_subperiod_cost_totals_and_annualizes():
    from qt.subset_validation import subperiod_cost

    nav = pd.DataFrame(
        {"cost": [0.001, 0.003], "turnover": [1.0, 0.5],
         "net_return": [0.01, -0.02]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    out = subperiod_cost(nav, periods_per_year=12)
    assert out["total_cost"] == pytest.approx(0.004)
    assert out["cost_drag_annual"] == pytest.approx(0.002 * 12)


def test_subperiod_cost_empty_is_nan():
    from qt.subset_validation import subperiod_cost

    out = subperiod_cost(pd.DataFrame(), periods_per_year=12)
    assert math.isnan(out["total_cost"]) and math.isnan(out["cost_drag_annual"])


# --------------------------------------------------------------------------- #
# fee-parameterized backtest: default-preserving + cost-line-only scenarios
# --------------------------------------------------------------------------- #
def _mini_backtest(example_config_path, demo_panel, fee_rate=None, *, explicit=True):
    from qt.oos_stability import _run_backtest_for
    from tests.fixtures.panel_factory import SYMBOLS
    from universe.static import StaticUniverse

    cfg = load_config(example_config_path)
    universe = StaticUniverse(list(SYMBOLS), {"missing_close": True})
    scores = demo_panel["close"].rename("score")
    if explicit:
        return _run_backtest_for(cfg, demo_panel, universe, scores, fee_rate=fee_rate)
    return _run_backtest_for(cfg, demo_panel, universe, scores)


def test_backtest_fee_rate_default_preserves_behaviour(example_config_path, demo_panel):
    """fee_rate=None (and the old call shape) == the config's fee, bitwise."""
    cfg = load_config(example_config_path)
    old_shape = _mini_backtest(example_config_path, demo_panel, explicit=False)
    none_arg = _mini_backtest(example_config_path, demo_panel, fee_rate=None)
    explicit = _mini_backtest(example_config_path, demo_panel,
                              fee_rate=cfg.cost.fee_rate)
    pd.testing.assert_frame_equal(old_shape, none_arg)
    pd.testing.assert_frame_equal(old_shape, explicit)
    assert len(old_shape) >= 1  # the mini panel settles at least one rebalance


def test_cost_scenarios_change_cost_line_only(example_config_path, demo_panel):
    """Scores/fills never see the fee: 2x fee => identical trades & gross,
    exactly doubled cost, net = gross - cost."""
    nav1 = _mini_backtest(example_config_path, demo_panel, fee_rate=0.001)
    nav2 = _mini_backtest(example_config_path, demo_panel, fee_rate=0.002)
    pd.testing.assert_index_equal(nav1.index, nav2.index)
    pd.testing.assert_series_equal(nav1["turnover"], nav2["turnover"])
    pd.testing.assert_series_equal(nav1["gross_return"], nav2["gross_return"])
    assert nav2["cost"].to_numpy() == pytest.approx(2.0 * nav1["cost"].to_numpy())
    assert nav1["cost"].abs().sum() > 0  # the first rebalance actually paid cost
    for nav in (nav1, nav2):
        assert nav["net_return"].to_numpy() == pytest.approx(
            (nav["gross_return"] - nav["cost"]).to_numpy()
        )


# --------------------------------------------------------------------------- #
# per-group processing semantics
# --------------------------------------------------------------------------- #
def _processing_cfg(tmp_path, example_config_path):
    """A demo config with drop_missing+zscore on, neutralize off."""
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    raw["processing"] = {
        "drop_missing": True,
        "standardize": {"enabled": True, "method": "zscore"},
        "winsorize": {"enabled": False, "method": "mad", "n": 3.0},
        "neutralize": {"enabled": False, "industry_col": "industry",
                       "size_col": "market_cap", "industry_level": "L1"},
    }
    p = tmp_path / "proc.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(p))


def _two_col_factor_panel():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-05", "2024-01-08"]),
         ["000001.SZ", "000002.SZ", "000003.SZ"]],
        names=["date", "symbol"],
    )
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, name="alpha_a")
    b = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0], index=idx, name="beta_b")
    return pd.concat([a, b], axis=1)


def test_group_processing_drop_missing_is_per_group(tmp_path, example_config_path):
    """drop_missing applies to the GROUP's columns only: a row killed by the
    full pack's NaN survives in a group that excludes the NaN column."""
    from qt.subset_validation import process_group

    cfg = _processing_cfg(tmp_path, example_config_path)
    fp = _two_col_factor_panel()
    panel = pd.DataFrame(index=fp.index)  # no covariates needed (neutralize off)
    nan_row = (pd.Timestamp("2024-01-05"), "000003.SZ")

    full = process_group(cfg, fp, panel, ["alpha_a", "beta_b"])
    only_a = process_group(cfg, fp, panel, ["alpha_a"])
    assert nan_row not in full.index          # full pack: NaN in beta_b kills the row
    assert nan_row in only_a.index            # alpha_a-only group keeps it
    assert list(only_a.columns) == ["alpha_a"]


def test_group_with_all_columns_matches_oos_processing(tmp_path, example_config_path):
    """A group listing every column reproduces the P3-4/P3-5 processing bitwise
    (the full_pack leg of the comparison IS the old pipeline — no drift)."""
    from qt.pipeline import _process_factors
    from qt.subset_validation import process_group

    cfg = _processing_cfg(tmp_path, example_config_path)
    fp = _two_col_factor_panel()
    panel = pd.DataFrame(index=fp.index)
    via_group = process_group(cfg, fp, panel, ["alpha_a", "beta_b"])
    via_oos = _process_factors(cfg, fp, panel)
    pd.testing.assert_frame_equal(via_group, via_oos)


def test_process_group_rejects_missing_column(tmp_path, example_config_path):
    from qt.subset_validation import process_group

    cfg = _processing_cfg(tmp_path, example_config_path)
    fp = _two_col_factor_panel()
    with pytest.raises(ValueError, match="not_a_col"):
        process_group(cfg, fp, pd.DataFrame(index=fp.index), ["not_a_col"])


# --------------------------------------------------------------------------- #
# cross-cell aggregation — strictly per cell and per group
# --------------------------------------------------------------------------- #
def _fake_perf(annual):
    return {"annual_return": annual, "volatility": 0.15, "sharpe": -0.2,
            "max_drawdown": -0.1, "avg_turnover": 0.8, "n_rebalances": 11,
            "total_cost": 0.01, "cost_drag_annual": 0.012}


def _fake_group(*, eq_test, ic_test_by_scn, combo_ic_test, consistent):
    performance = {
        scn: {
            "equal_weight": {"train": _fake_perf(-0.06), "test": _fake_perf(eq_test)},
            "ic_weighted": {"train": _fake_perf(-0.03), "test": _fake_perf(annual)},
        }
        for scn, annual in ic_test_by_scn.items()
    }
    stats = {
        "train": {"ic_mean": 0.01 if consistent else -0.01,
                  "ic_ir": 0.1, "hit_rate": 0.5, "n": 100},
        "test": {"ic_mean": combo_ic_test, "ic_ir": 0.1, "hit_rate": 0.5, "n": 100},
    }
    return {
        "factors": ("value_ep", "volatility_20"),
        "performance": performance,
        "combo_ic_stats": {"combo_equal_weight": stats, "combo_ic_weighted": stats},
        "combo_sign_consistency": {"combo_equal_weight": consistent,
                                   "combo_ic_weighted": consistent},
        "n_scored": 200, "n_fallback": 20, "fallback_reasons": {},
        "sign_flips": {"value_ep": 1},
    }


def test_summarize_subset_attributes_per_cell_and_group():
    from qt.subset_validation import summarize_subset_matrix

    cells = {
        "A|w1": {"groups": {
            "good": _fake_group(eq_test=-0.05, ic_test_by_scn={"base": -0.02, "2x": -0.04},
                                combo_ic_test=0.03, consistent=True),
            "bad": _fake_group(eq_test=-0.02, ic_test_by_scn={"base": -0.05, "2x": -0.07},
                               combo_ic_test=-0.01, consistent=False),
        }},
        "B|w1": {"groups": {
            "good": _fake_group(eq_test=-0.05, ic_test_by_scn={"base": -0.08, "2x": -0.09},
                                combo_ic_test=-0.02, consistent=False),
            "bad": _fake_group(eq_test=-0.06, ic_test_by_scn={"base": -0.01, "2x": -0.03},
                               combo_ic_test=0.02, consistent=True),
        }},
    }
    summary = summarize_subset_matrix(cells, base_scenario="base")
    assert summary["n_cells"] == 2
    good = summary["groups"]["good"]
    bad = summary["groups"]["bad"]
    # ic beats eq on TEST annual at the BASE scenario, attributed per cell
    assert good["ic_beats_eq_test_base"] == 1   # only cell A
    assert bad["ic_beats_eq_test_base"] == 1    # only cell B
    # combo IC positivity / sign consistency counted per group per cell
    assert good["combo"]["combo_ic_weighted"]["test_ic_positive"] == 1
    assert good["combo"]["combo_ic_weighted"]["sign_consistent"] == 1
    assert bad["combo"]["combo_ic_weighted"]["test_ic_positive"] == 1
    # per-scenario test annuals keyed by cell (cost sensitivity readout)
    assert good["ic_test_annual_by_scenario"]["2x"]["A|w1"] == pytest.approx(-0.04)
    assert good["ic_test_annual_by_scenario"]["base"]["B|w1"] == pytest.approx(-0.08)


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #
def _synthetic_subset_result():
    from qt.subset_validation import (
        SubsetCellResult,
        SubsetValidationResult,
        summarize_subset_matrix,
    )

    group_a = _fake_group(eq_test=-0.05,
                          ic_test_by_scn={"base": -0.02, "2x": -0.03, "high_cost": -0.05},
                          combo_ic_test=0.03, consistent=True)
    group_b = _fake_group(eq_test=-0.02,
                          ic_test_by_scn={"base": -0.05, "2x": -0.06, "high_cost": -0.08},
                          combo_ic_test=-0.01, consistent=False)
    raw_stats = {
        "value_ep": {"train": {"ic_mean": 0.04, "ic_ir": 0.3, "hit_rate": 0.55, "n": 200},
                     "test": {"ic_mean": 0.05, "ic_ir": 0.4, "hit_rate": 0.56, "n": 210}},
        "momentum_20": {"train": {"ic_mean": -0.02, "ic_ir": -0.1, "hit_rate": 0.48, "n": 200},
                        "test": {"ic_mean": 0.01, "ic_ir": 0.05, "hit_rate": 0.51, "n": 210}},
    }
    cell = SubsetCellResult(
        split_date=pd.Timestamp("2023-07-01"),
        train_start=pd.Timestamp("2022-07-01"), train_end=pd.Timestamp("2023-06-30"),
        test_start=pd.Timestamp("2023-07-03"), test_end=pd.Timestamp("2024-06-28"),
        n_train_days=240, n_test_days=238,
        boundary_dates=(pd.Timestamp("2023-06-30"),),
        factor_names=("momentum_20", "value_ep"),
        raw_ic_stats=raw_stats,
        raw_sign_consistency={"value_ep": True, "momentum_20": False},
        groups={"value_lowvol": group_a, "legacy_trio": group_b},
        downgrades=("DATA PATH = REAL tushare: PIT index membership (000016.SH); ...",
                    "shared-disclosure line"),
        elapsed_seconds=900.0,
    )
    cells = {"000016.SH|2022-2024": cell}
    plain = {label: {"groups": c.groups} for label, c in cells.items()}
    cfg = load_config(_SUBSET_CONFIG)
    return SubsetValidationResult(
        config=cfg,
        elapsed_seconds=1000.0,
        base_scenario="base",
        scenario_fees={"base": 0.001, "2x": 0.002, "high_cost": 0.004},
        cells=cells,
        cell_runtimes={"000016.SH|2022-2024": 900.0},
        skipped_cells=("000300.SH|2020-2022",),
        summary=summarize_subset_matrix(plain, base_scenario="base"),
        report_path=Path("artifacts/reports/phase3_subset_validation.md"),
        log_path=Path("artifacts/logs/run_phase3_subset.log"),
    )


def test_render_subset_report_discloses_groups_and_scenarios():
    from qt.reports import render_subset_validation

    md = render_subset_validation(_synthetic_subset_result())
    # factor groups disclosed with their exact factor lists
    assert "value_lowvol" in md and "legacy_trio" in md
    assert "value_ep" in md and "volatility_20" in md
    # cost scenarios disclosed with multipliers / effective fees
    assert "base" in md and "2x" in md and "high_cost" in md
    assert "0.004" in md  # the high-cost effective fee
    # cost metrics present
    assert "cost drag" in md.lower()
    # trades are cost-invariant note
    assert "identical across" in md.lower() or "cost line" in md.lower()
    # skipped cells + boundary disclosure survive
    assert "000300.SH|2020-2022" in md and "skipped" in md.lower()
    assert "boundary" in md.lower()
    # raw-factor IC no-drift hook is present
    assert "no-drift" in md.lower() or "per-column" in md.lower()


def test_render_subset_report_carries_honesty_caveats():
    from qt.reports import render_subset_validation

    md = render_subset_validation(_synthetic_subset_result())
    assert "POST-HOC" in md            # subset chosen after seeing P3-5 results
    assert "NOT a return claim" in md
    assert "MATRIX SCOPE" in md
    # union-of-cells downgrades (shared line not duplicated)
    caveats = md.split("## DOWNGRADES")[1]
    assert caveats.count("shared-disclosure line") == 1


def test_render_subset_report_leaks_no_secret():
    from qt.reports import render_subset_validation

    result = _synthetic_subset_result()
    md = render_subset_validation(result)
    assert result.config.data.external_secret_file not in md
    assert "token" not in md.lower()


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_cli_has_run_phase3_subset_with_readable_guard(tmp_path, capsys):
    from qt.cli import main

    raw = yaml.safe_load(Path(_SUBSET_CONFIG).read_text(encoding="utf-8"))
    raw.pop("subset_validation")
    p = tmp_path / "no_subset.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code = main(["run-phase3-subset", "--config", str(p)])
    assert code == 1
    err = capsys.readouterr().err
    assert "subset_validation" in err
