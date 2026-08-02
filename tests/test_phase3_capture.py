"""D6b capture harness — network-free unit tests.

Covers the pure surface of ``qt.phase3_capture`` with fake result objects and
synthetic panels: full-precision serialization (incl. the explicit exclusion
list), the 0.0-tolerance leaf comparator, the panel comparator, the cell
enumeration, and the legacy panel's verbatim ``factor.compute`` concat. No
runner, no feed, no token is touched.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd
import pytest

from qt.config import RootConfig, load_config
from qt.phase3_capture import (
    EXCLUDED_RESULT_FIELDS,
    compare_json,
    compare_panels,
    iter_cell_configs,
    legacy_factor_panel,
    to_jsonable,
)


@dataclasses.dataclass(frozen=True)
class _FakeResult:
    """Stands in for OOSResult & co.: every captured leaf type + exclusions."""

    config: object  # excluded: identity, carries the secret-file path
    elapsed_seconds: float  # excluded: timing
    cell_runtimes: dict  # excluded: timing
    report_path: Path  # excluded: location
    log_path: Path  # excluded: location
    performance: dict
    ic_stats: dict
    sign_consistency: dict
    sign_flips: dict
    fallback_reasons: dict
    boundary_dates: tuple
    weights: pd.DataFrame
    split_date: pd.Timestamp
    n_fallback: int


def _fake_result() -> _FakeResult:
    weights = pd.DataFrame(
        {"momentum_20": [0.5, 0.5], "roe": [0.25, 0.5], "fallback": [False, True]},
        index=pd.DatetimeIndex(["2023-01-31", "2023-02-28"]),
    )
    return _FakeResult(
        config=object(),
        elapsed_seconds=123.456,
        cell_runtimes={"a|b": 42.0},
        report_path=Path("/nowhere/report.md"),
        log_path=Path("/nowhere/run.log"),
        performance={"equal_weight": {"test": {"annual_return": 0.12345678901234567}}},
        ic_stats={"roe": {"test": {"ic_mean": float("nan"), "n": 3}}},
        sign_consistency={"roe": False},
        sign_flips={"roe": 2},
        fallback_reasons={"insufficient_history": 4},
        boundary_dates=(pd.Timestamp("2023-07-03"),),
        weights=weights,
        split_date=pd.Timestamp("2023-07-01"),
        n_fallback=1,
    )


def test_to_jsonable_drops_exactly_the_exclusion_list():
    tree = to_jsonable(_fake_result())
    for field in EXCLUDED_RESULT_FIELDS:
        assert field not in tree, f"excluded field leaked: {field}"
    # ...and nothing else was dropped.
    captured = {f.name for f in dataclasses.fields(_FakeResult)}
    assert set(tree) == captured - set(EXCLUDED_RESULT_FIELDS)


def test_to_jsonable_full_precision_and_leaf_types():
    tree = to_jsonable(_fake_result())
    # full float precision survives the repr round-trip (17 significant digits)
    assert tree["performance"]["equal_weight"]["test"]["annual_return"] == 0.12345678901234567
    assert math.isnan(tree["ic_stats"]["roe"]["test"]["ic_mean"])
    assert tree["boundary_dates"] == ["2023-07-03T00:00:00"]
    assert tree["split_date"] == "2023-07-01T00:00:00"
    weights = tree["weights"]
    assert weights["__dataframe__"] is True
    assert weights["columns"] == ["momentum_20", "roe", "fallback"]
    assert weights["index"] == ["2023-01-31T00:00:00", "2023-02-28T00:00:00"]
    assert weights["data"][1] == [0.5, 0.5, True]


def test_compare_json_identical_trees_have_no_diffs():
    left = to_jsonable(_fake_result())
    right = to_jsonable(_fake_result())
    assert compare_json(left, right) == {"n_diffs": 0, "diffs": []}


def test_compare_json_floats_compare_at_zero_tolerance():
    left = {"a": {"b": 1.0}}
    right = {"a": {"b": 1.0 + 1e-15}}  # 1e-16 would round to 1.0 in float64
    report = compare_json(left, right)
    assert report["n_diffs"] == 1
    assert report["diffs"][0]["path"] == "a.b"
    assert report["diffs"][0]["kind"] == "value_mismatch"


def test_compare_json_nan_at_same_path_is_equal():
    assert compare_json({"x": float("nan")}, {"x": float("nan")})["n_diffs"] == 0


def test_compare_json_reports_missing_keys_and_types():
    report = compare_json({"a": 1, "b": "s"}, {"a": 1, "c": True})
    kinds = {(d["path"], d["kind"]) for d in report["diffs"]}
    assert kinds == {("b", "missing_right"), ("c", "missing_left")}
    typed = compare_json({"a": 1}, {"a": "1"})
    assert typed["diffs"][0]["kind"] == "type_mismatch"


def test_compare_json_walks_lists():
    report = compare_json({"w": [[1.0, 2.0]]}, {"w": [[1.0, 2.5]]})
    assert report["n_diffs"] == 1
    assert report["diffs"][0]["path"] == "w[0][1]"


def _panel(values_a=(1.0, 2.0), values_b=(1.0, 2.0), col="f") -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("2023-01-03", "AAA"), ("2023-01-04", "AAA")], names=["date", "symbol"]
    )
    left = pd.DataFrame({col: list(values_a)}, index=index)
    right = pd.DataFrame({col: list(values_b)}, index=index)
    return left, right


def test_compare_panels_identical_is_clean():
    left, right = _panel()
    report = compare_panels(left, right)
    assert report["max_abs_diff"] == 0.0
    assert report["n_left_only_index"] == report["n_right_only_index"] == 0
    assert report["per_column"]["f"]["n_value_mismatch"] == 0


def test_compare_panels_value_diff_and_max_abs():
    left, right = _panel(values_b=(1.0, 2.25))
    report = compare_panels(left, right)
    assert report["per_column"]["f"]["n_value_mismatch"] == 1
    assert report["max_abs_diff"] == pytest.approx(0.25)


def test_compare_panels_nan_mask_mismatch_and_nan_equality():
    left, right = _panel(values_a=(float("nan"), 2.0), values_b=(1.0, float("nan")))
    report = compare_panels(left, right)
    col = report["per_column"]["f"]
    assert col["nan_mask_mismatch"] == 2
    assert col["n_value_mismatch"] == 0  # no comparable pair -> no value diff
    # NaN vs NaN at the same cell is NOT a mismatch
    same_nan_l, _ = _panel(values_a=(float("nan"), 2.0))
    same_nan_r, _ = _panel(values_a=(float("nan"), 2.0))
    assert compare_panels(same_nan_l, same_nan_r)["per_column"]["f"]["nan_mask_mismatch"] == 0


def test_compare_panels_grid_differences_are_reported_separately():
    left, _ = _panel()
    right = pd.DataFrame(
        {"g": [9.0]},
        index=pd.MultiIndex.from_tuples(
            [("2023-01-05", "AAA")], names=["date", "symbol"]
        ),
    )
    report = compare_panels(left, right)
    assert report["left_only_columns"] == ["f"]
    assert report["right_only_columns"] == ["g"]
    assert report["n_left_only_index"] == 2
    assert report["n_right_only_index"] == 1


def test_iter_cell_configs_single_cell_without_robustness(example_config_path):
    cfg = load_config(example_config_path)
    cells = iter_cell_configs(cfg)
    assert len(cells) == 1
    label, cell_cfg = cells[0]
    assert label == cfg.data.output_name
    assert cell_cfg is cfg


def test_iter_cell_configs_matrix_enumerates_and_honours_skips(example_config_path):
    cfg = load_config(example_config_path)
    raw = cfg.model_dump()
    raw["oos"] = {"split_date": "2024-03-01"}
    raw["robustness"] = {
        "universes": ["000016.SH", "000300.SH"],
        "windows": [{"label": "w1", "start": "2024-01-01",
                     "end": "2024-06-30", "split": "2024-03-01"}],
        "skip_cells": [{"universe": "000300.SH", "window": "w1"}],
    }
    matrix_cfg = RootConfig(**raw)
    cells = iter_cell_configs(matrix_cfg)
    assert [label for label, _ in cells] == ["000016.SH|w1"]
    _, cell_cfg = cells[0]
    assert cell_cfg.universe.index_code == "000016.SH"
    assert cell_cfg.oos.split_date == raw["oos"]["split_date"]


def test_legacy_factor_panel_is_the_verbatim_compute_concat():
    class _FakeFactor:
        def __init__(self, name, column, scale):
            self.name = name
            self._column = column
            self._scale = scale

        def compute(self, panel):
            return panel[self._column] * self._scale

    index = pd.MultiIndex.from_tuples(
        [("2023-01-03", "AAA"), ("2023-01-03", "BBB")], names=["date", "symbol"]
    )
    panel = pd.DataFrame({"close": [10.0, 20.0]}, index=index)
    factors = [_FakeFactor("f1", "close", 2.0), _FakeFactor("f2", "close", 3.0)]
    out = legacy_factor_panel(panel, factors)
    assert list(out.columns) == ["f1", "f2"]
    assert out["f1"].tolist() == [20.0, 40.0]
    assert out["f2"].tolist() == [30.0, 60.0]
