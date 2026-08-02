"""D6c: ``MmpEwFactor`` — the I3 ``mmp_ew`` session feature as a first-class factor.

WHAT IS PINNED HERE (network-free)

1. HAND ANCHOR — the daily equal-weight MMP value of a controlled 25-bar
   synthetic session, computed independently of the implementation (plain
   Python floats off the factor definition), bit-compared.
2. PIT — perturbing bars after the 14:50 decision cutoff moves nothing; the
   perturbation itself is asserted real and targeted first (the project's
   recorded unfailable-test shape).
3. SESSION LOWER BOUND — pre-session bars (``bar_end < session_open``) enter
   NEITHER the rolling baseline NOR the daily mean: wildly different pre-session
   content yields a bit-identical value, and exactly ``MMP_LOOKBACK`` in-session
   bars are needed before the first valid minute exists.
4. EQUIVALENCE — ``compute_intraday_mmp_ew`` IS
   ``asof_daily_features(features=["mmp_ew"])`` cell-for-cell on a multi-day,
   multi-symbol fixture (bit-exact, NaN mask included), and the Series name is
   the legacy aggregate column name (every shipped report keys on it verbatim).
5. SPLIT == WHOLE — per-symbol computation concatenated equals the whole-frame
   call (the property the D4b streaming binding relies on).
6. SURFACE — registry build of the exact name, the spec's intraday block /
   declarations, and ``compute`` surfacing the pre-aggregated panel column.

MUTATION EVIDENCE (run against this file, each mutation first asserted to
change its target; measured 2026-08-02 in this worktree):

* ``<=`` -> ``<`` in the visibility filter of ``compute_intraday_mmp_ew``
  (drops the bar whose available_time lands exactly on the cutoff):
  ``test_equivalent_to_asof_daily_features_mmp_ew_bit_for_bit`` RED, 15 of 15
  finite cells moved (rc=1); restored -> 9 passed.
* dropping the ``in_session_bars`` filter (pre-session bars feed the rolling
  baseline): ``test_pre_session_bars_are_invisible_to_the_baseline_and_the_mean``
  RED — the anchor-fixture value moved 0.0499999500000375 ->
  0.012499987500009375 (rc=1); restored -> 9 passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_aggregate import asof_daily_features
from data.clean.intraday_schema import normalize_intraday_bars
from factors.compute.minute.mmp import (
    DEFAULT_EPSILON,
    MMP_EW_FACTOR_NAME,
    MMP_LOOKBACK,
    MmpEwFactor,
    compute_intraday_mmp_ew,
)
from factors.registry import build

_DAY = "2024-01-02"
_SYM = "000001.SZ"


def _frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


def _anchor_bars(n_bars: int = 25) -> pd.DataFrame:
    """A session whose MMP daily value is hand-computable.

    Every bar has high=11 / low=9 (hl=2, mid=10) and volume=1000, so the volume
    baseline ratio is exactly 1 and the range baseline is exactly hl. Bars
    0..MMP_LOOKBACK-1 are flat (open=close=mid -> S=0, B=0); the trailing five
    bars close at the high (open=10, close=11 -> S=0.1, B=1/(hl+eps)). Only the
    last five bars have a full prior-20 baseline, so the daily value is exactly
    their common per-bar MMP.
    """
    rows = []
    for i in range(n_bars):
        t = pd.Timestamp(_DAY) + pd.Timedelta("09:31:00") + pd.Timedelta(minutes=i)
        o, c = (10.0, 10.0) if i < MMP_LOOKBACK else (10.0, 11.0)
        rows.append((t, _SYM, o, 11.0, 9.0, c, 1000.0, c * 1000.0))
    return _frame(rows)


def _random_bars() -> pd.DataFrame:
    """3 symbols x 5 days x full sessions (incl. post-cutoff bars), seeded."""
    rng = np.random.RandomState(20260802)
    days = pd.bdate_range("2024-01-02", periods=5)
    symbols = ("000001.SZ", "000002.SZ", "600000.SH")
    rows = []
    for s in symbols:
        for d in days:
            times = pd.date_range(d + pd.Timedelta("09:31:00"), periods=120, freq="1min")
            times = times.append(
                pd.date_range(d + pd.Timedelta("13:01:00"), periods=120, freq="1min")
            )
            price = 20.0 + rng.rand() * 5
            for t in times:
                price += rng.normal(0, 0.03)
                hi = price + abs(rng.normal(0, 0.02))
                lo = price - abs(rng.normal(0, 0.02))
                cl = lo + (hi - lo) * rng.rand()
                vol = float(rng.lognormal(8.0, 0.5))
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    return _frame(rows)


RANDOM_BARS = _random_bars()


# --------------------------------------------------------------------------- #
# 1. hand anchor
# --------------------------------------------------------------------------- #
def test_daily_value_matches_a_hand_computed_anchor():
    eps = DEFAULT_EPSILON
    expected = 0.1 * 1.0 * (1.0 / (2.0 + eps)) * (2.0 / (2.0 + eps))
    got = compute_intraday_mmp_ew(_anchor_bars())
    assert list(got.index) == [(pd.Timestamp(_DAY), _SYM)]
    assert got.iloc[0] == expected  # bit-exact, not approximate
    assert got.name == MMP_EW_FACTOR_NAME


# --------------------------------------------------------------------------- #
# 2. PIT: post-cutoff bars cannot move the value
# --------------------------------------------------------------------------- #
def _post_cutoff_mask(bars: pd.DataFrame) -> np.ndarray:
    trade_date = bars["bar_end"].dt.normalize()
    return (bars["available_time"] > trade_date + pd.Timedelta("14:50:00")).to_numpy()


def test_post_cutoff_bars_cannot_move_the_value():
    base = RANDOM_BARS
    mask = _post_cutoff_mask(base)
    assert mask.any(), "fixture has no post-cutoff bars — vacuous perturbation"
    out = base.copy(deep=True)
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = out.loc[mask, col] * 137.0
    out.loc[mask, "volume"] = out.loc[mask, "volume"] * 9.0e5 + 1.0
    out.loc[mask, "amount"] = out.loc[mask, "amount"] * 9.0e5 + 1.0
    # the mutation is real and targeted BEFORE it is relied on
    assert not out.loc[mask].equals(base.loc[mask]), "perturbation changed nothing"
    pd.testing.assert_frame_equal(base.loc[~mask], out.loc[~mask], check_exact=True)

    pd.testing.assert_series_equal(
        compute_intraday_mmp_ew(out), compute_intraday_mmp_ew(base), check_exact=True
    )


# --------------------------------------------------------------------------- #
# 3. session lower bound: pre-session bars feed neither baseline nor mean
# --------------------------------------------------------------------------- #
def test_pre_session_bars_are_invisible_to_the_baseline_and_the_mean():
    base = _anchor_bars(21)  # exactly MMP_LOOKBACK + 1 in-session bars -> 1 valid minute
    want = compute_intraday_mmp_ew(base)
    assert np.isfinite(want.iloc[0]), "fixture must yield one valid minute"

    def with_pre_session(volume: float, close: float) -> pd.DataFrame:
        rows = []
        for i in range(3):  # 09:00-09:02, before the 09:30 session open
            t = pd.Timestamp(_DAY) + pd.Timedelta("09:00:00") + pd.Timedelta(minutes=i)
            rows.append((t, _SYM, close, close + 1.0, close - 1.0, close, volume, close * volume))
        return pd.concat([_frame(rows), base]).sort_index(kind="mergesort")

    mild = compute_intraday_mmp_ew(with_pre_session(1000.0, 10.0))
    wild = compute_intraday_mmp_ew(with_pre_session(9.0e8, 500.0))
    pd.testing.assert_series_equal(mild, want, check_exact=True)
    pd.testing.assert_series_equal(wild, want, check_exact=True)


def test_the_first_lookback_in_session_bars_have_no_valid_minute():
    """Exactly MMP_LOOKBACK in-session bars -> NO valid minute -> NaN (the
    baseline window is strictly prior and never crosses the session open)."""
    got = compute_intraday_mmp_ew(_anchor_bars(MMP_LOOKBACK))
    assert got.isna().all()
    one_more = compute_intraday_mmp_ew(_anchor_bars(MMP_LOOKBACK + 1))
    assert np.isfinite(one_more.iloc[0])


# --------------------------------------------------------------------------- #
# 4. equivalence with the legacy aggregate feature
# --------------------------------------------------------------------------- #
def test_equivalent_to_asof_daily_features_mmp_ew_bit_for_bit():
    frame = asof_daily_features(RANDOM_BARS, features=["mmp_ew"])
    legacy = frame[frame.columns[0]]
    got = compute_intraday_mmp_ew(RANDOM_BARS)
    # the factor id IS the legacy column name (report text keyed on it verbatim)
    assert got.name == legacy.name == MMP_EW_FACTOR_NAME
    assert int(np.isfinite(got.to_numpy()).sum()) > 0, "vacuous: all NaN"
    pd.testing.assert_series_equal(got, legacy, check_exact=True)


# --------------------------------------------------------------------------- #
# 5. per-symbol split == whole
# --------------------------------------------------------------------------- #
def test_per_symbol_split_reproduces_the_whole_frame_call():
    whole = compute_intraday_mmp_ew(RANDOM_BARS)
    parts = []
    for sym in pd.Index(RANDOM_BARS.index.get_level_values("symbol")).unique():
        one = RANDOM_BARS[pd.Index(RANDOM_BARS.index.get_level_values("symbol")) == sym]
        parts.append(compute_intraday_mmp_ew(one))
    split = pd.concat(parts).sort_index()
    pd.testing.assert_series_equal(split, whole.sort_index(), check_exact=True)


# --------------------------------------------------------------------------- #
# 6. factor surface: registry + spec + compute
# --------------------------------------------------------------------------- #
def test_registry_builds_the_exact_name():
    factor = build(MMP_EW_FACTOR_NAME)
    assert isinstance(factor, MmpEwFactor)
    assert factor.name == MMP_EW_FACTOR_NAME
    assert factor.spec.factor_id == MMP_EW_FACTOR_NAME


def test_spec_declares_the_full_intraday_block_and_d1_taxonomy():
    spec = MmpEwFactor().spec
    assert spec.is_intraday is True
    assert spec.return_basis == "exec_to_exec"
    assert spec.decision_cutoff == "14:50:00"
    assert spec.data_lag == "1min"
    assert spec.session_open == "09:30:00"
    assert spec.execution_model == "next_minute_close"
    assert spec.execution_window == "[14:51:00,14:56:59]"
    assert spec.expected_ic_sign in (+1, -1)
    assert [(r.field, r.source) for r in spec.requires] == [
        (f, "stk_mins_1min") for f in ("open", "high", "low", "close", "volume")
    ]
    assert spec.adjustment.value == "returns_invariant"
    assert spec.overnight_boundary.value == "none"
    assert spec.lookback_depth == 1


def test_compute_surfaces_the_preaggregated_panel_column():
    factor = MmpEwFactor()
    series = compute_intraday_mmp_ew(RANDOM_BARS)
    panel = series.to_frame(factor.name)
    pd.testing.assert_series_equal(factor.compute(panel), series, check_exact=True)
    with pytest.raises(ValueError, match="pre-aggregated"):
        factor.compute(panel.rename(columns={factor.name: "something_else"}))
