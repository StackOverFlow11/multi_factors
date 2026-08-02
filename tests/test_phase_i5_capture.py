"""D6c I5 capture harness — network-free unit tests.

Covers the pure surface of ``qt.phase_i5_capture`` with synthetic score series
and fake modules: the exclusion list (alive against the REAL result
dataclasses), the score-leg comparator with mutation evidence (value, NaN-mask,
legacy-missing, footprint), and the runner registry. No runner, no feed, no
token, no cache is touched.
"""

from __future__ import annotations

import dataclasses
import importlib

import pandas as pd

from qt import phase_i5_capture as cap
from qt.intraday_group_backtest import I5dResult
from qt.intraday_tail_framework import I5aResult
from qt.phase3_capture import to_jsonable


def _score_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-03-31"), "000001.SZ"),
         (pd.Timestamp("2026-03-31"), "000002.SZ"),
         (pd.Timestamp("2026-04-30"), "000001.SZ")],
        names=["date", "symbol"],
    )


def _legacy() -> pd.Series:
    return pd.Series([0.012345678901234567, float("nan"), -0.5],
                     index=_score_index(), name="score")


# --------------------------------------------------------------------------- #
# Exclusion list: alive against the real result dataclasses
# --------------------------------------------------------------------------- #
def test_excluded_fields_are_real_fields_of_both_results():
    for result_type in (I5aResult, I5dResult):
        fields = {f.name for f in dataclasses.fields(result_type)}
        dead = cap.EXCLUDED_RESULT_FIELDS - fields
        # `elapsed` only exists on I5dResult; `figure_paths` likewise. A dead
        # exclusion on ONE of the two is fine — a dead exclusion on BOTH means
        # the entry is a typo that excludes nothing.
        assert dead != cap.EXCLUDED_RESULT_FIELDS, (
            f"every exclusion is dead on {result_type.__name__}: {dead}"
        )
    # The value leaves must NOT be excluded.
    assert "nav_table" not in cap.EXCLUDED_RESULT_FIELDS
    assert "groups" not in cap.EXCLUDED_RESULT_FIELDS
    assert "spread_per_period" not in cap.EXCLUDED_RESULT_FIELDS
    assert "score_coverage" not in cap.EXCLUDED_RESULT_FIELDS
    assert "factor_diagnostics" not in cap.EXCLUDED_RESULT_FIELDS
    assert "liquidity_diagnostics" not in cap.EXCLUDED_RESULT_FIELDS


@dataclasses.dataclass(frozen=True)
class _Nested:
    report_path: str
    nav: float


@dataclasses.dataclass(frozen=True)
class _FakeI5Result:
    config: object
    elapsed: float
    report_path: str
    log_path: str
    figure_paths: dict
    nested: _Nested
    nav_final: float


def test_to_jsonable_with_i5_exclusions_drops_exactly_those():
    obj = _FakeI5Result(
        config=object(), elapsed=1.5, report_path="/r.md", log_path="/r.log",
        figure_paths={"nav": "/f.png"}, nested=_Nested(report_path="/n.md", nav=1.25),
        nav_final=1.019318,
    )
    tree = to_jsonable(obj, cap.EXCLUDED_RESULT_FIELDS)
    for field in cap.EXCLUDED_RESULT_FIELDS:
        assert field not in tree, f"excluded field leaked: {field}"
    assert set(tree) == {"nested", "nav_final"}
    # The exclusion recurses into nested dataclasses.
    assert tree["nested"] == {"nav": 1.25}
    assert tree["nav_final"] == 1.019318


# --------------------------------------------------------------------------- #
# Score-leg comparator (mutation evidence: each rule must have teeth)
# --------------------------------------------------------------------------- #
def test_compare_score_identical_passes():
    report = cap.compare_score_series(_legacy(), _legacy())
    assert report["verdict_pass"] is True
    assert report["max_abs_diff"] == 0.0
    assert report["n_value_mismatch"] == 0
    assert report["nan_mask_mismatch"] == 0
    assert report["n_compared"] == 2  # the NaN cell is not a compared value cell


def test_compare_score_full_precision_roundtrip():
    # 17 significant digits must survive: the comparator is 0.0-tolerance.
    served = _legacy().copy()
    served.iloc[0] = served.iloc[0] + 1e-15  # 1e-16 would round away in float64
    report = cap.compare_score_series(_legacy(), served)
    assert report["verdict_pass"] is False
    assert report["n_value_mismatch"] == 1
    assert report["max_abs_diff"] > 0.0


def test_compare_score_nan_mask_mismatch_fails():
    served = _legacy().copy()
    served.iloc[1] = 0.0  # a value where the legacy score has NaN
    report = cap.compare_score_series(_legacy(), served)
    assert report["verdict_pass"] is False
    assert report["nan_mask_mismatch"] == 1


def test_compare_score_missing_legacy_cell_fails():
    served = _legacy().iloc[1:]  # the service dropped one legacy cell
    report = cap.compare_score_series(_legacy(), served)
    assert report["verdict_pass"] is False
    assert report["n_legacy_only_index"] == 1
    assert report["legacy_only_index"]


def test_compare_score_served_only_nan_footprint_passes():
    extra = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-04-30"), "000003.SZ")], names=["date", "symbol"]
    )
    served = pd.concat([_legacy(), pd.Series([float("nan")], index=extra)])
    report = cap.compare_score_series(_legacy(), served)
    assert report["verdict_pass"] is True
    assert report["n_served_only_index"] == 1
    assert report["served_only_all_nan"] is True


def test_compare_score_served_only_finite_value_fails():
    # A finite value on a cell the legacy path never saw is a finding, not a
    # footprint row.
    extra = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-04-30"), "000003.SZ")], names=["date", "symbol"]
    )
    served = pd.concat([_legacy(), pd.Series([0.01], index=extra)])
    report = cap.compare_score_series(_legacy(), served)
    assert report["verdict_pass"] is False
    assert report["served_only_all_nan"] is False


# --------------------------------------------------------------------------- #
# Registries resolve against the real codebase
# --------------------------------------------------------------------------- #
def test_runner_registry_resolves_to_real_entry_points():
    for runner, (module_name, entry) in cap._RUNNERS.items():
        module = importlib.import_module(module_name)
        assert callable(getattr(module, entry)), f"{runner}: {entry} missing"


def test_known_score_factors_are_registered():
    from factors import registry as factor_registry

    for fid in cap._KNOWN_SCORE_FACTORS:
        factor = factor_registry.build(fid, None)
        assert factor.name == fid


def test_build_cache_recorder_passthrough_and_totals():
    class _FakeCache:
        def __init__(self, stats):
            self._stats = stats

        def stats(self):
            return dict(self._stats)

    class _FakeModule:
        def __init__(self):
            self.calls = 0

        def _build_cache(self, cfg):
            self.calls += 1
            return _FakeCache({"daily": 2, "adj_factor": 1})

    module = _FakeModule()
    with cap._BuildCacheRecorder(module) as recorder:
        cache = module._build_cache(None)
    assert cache.stats() == {"daily": 2, "adj_factor": 1}  # passthrough, unwrapped
    assert module._build_cache.__self__ is module  # restored on exit
    assert recorder.gap_fetches() == {"daily": 2, "adj_factor": 1}


def test_build_cache_recorder_handles_none_cache():
    class _FakeModule:
        def _build_cache(self, cfg):
            return None

    module = _FakeModule()
    with cap._BuildCacheRecorder(module) as recorder:
        module._build_cache(None)
    assert recorder.gap_fetches() == {}
