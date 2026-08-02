"""D6c: ``IntradaySessionRetFactor`` — the I3 ``ret`` session feature as a factor.

WHAT IS PINNED HERE (network-free)

1. HAND ANCHOR — ``last_visible_close / first_visible_open - 1`` on a 3-bar
   session, with a wild post-cutoff bar that must NOT enter, bit-compared.
2. PIT — perturbing bars after the 14:50 decision cutoff moves nothing; the
   perturbation itself is asserted real and targeted first.
3. EQUIVALENCE — ``compute_intraday_session_ret`` IS
   ``asof_daily_features(features=["ret"])`` cell-for-cell on a multi-day,
   multi-symbol fixture (bit-exact, NaN mask included): the ret math keeps its
   single definition point in ``data/clean`` (R14), this factor only delegates.
   The Series name is the legacy aggregate column name.
4. SPLIT == WHOLE — per-symbol computation concatenated equals the whole-frame
   call (the property the D4b streaming binding relies on).
5. SURFACE — registry build of the exact name, the spec's intraday block /
   declarations, and ``compute`` surfacing the pre-aggregated panel column.

MUTATION EVIDENCE (run against this file, each mutation first asserted to
change its target; measured 2026-08-02 in this worktree):

* delegating to ``features=["vwap"]`` instead of ``features=["ret"]``:
  the anchor AND the equivalence test RED, 15 of 15 finite cells moved — the
  equivalence sees WHICH feature the delegation selects, not just that some
  column comes back (rc=1); restored -> 7 passed.
* passing ``decision_time="15:00:00"`` through the delegation: the anchor, the
  PIT test AND the equivalence test RED — the cutoff rides the delegation
  rather than being re-stated here (rc=1); restored -> 7 passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_aggregate import asof_daily_features
from data.clean.intraday_schema import normalize_intraday_bars
from factors.compute.minute.intraday_session_ret import (
    SESSION_RET_FACTOR_NAME,
    IntradaySessionRetFactor,
    compute_intraday_session_ret,
)
from factors.registry import build

_DAY = "2024-01-02"
_SYM = "000001.SZ"


def _frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


def _random_bars() -> pd.DataFrame:
    """3 symbols x 5 days x full sessions (incl. post-cutoff bars), seeded."""
    rng = np.random.RandomState(20260803)
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
# 1. hand anchor (+ a post-cutoff bar that must not enter)
# --------------------------------------------------------------------------- #
def test_daily_value_matches_a_hand_computed_anchor():
    rows = []
    for i, (o, c) in enumerate([(10.0, 10.1), (10.1, 10.2), (10.2, 10.3)]):
        t = pd.Timestamp(_DAY) + pd.Timedelta("09:31:00") + pd.Timedelta(minutes=i)
        rows.append((t, _SYM, o, c + 0.05, o - 0.05, c, 1000.0, c * 1000.0))
    # a wild post-cutoff bar: if it entered, the value would be 99/10 - 1
    t = pd.Timestamp(_DAY) + pd.Timedelta("14:55:00")
    rows.append((t, _SYM, 99.0, 99.5, 98.5, 99.0, 1.0e6, 99.0e6))
    got = compute_intraday_session_ret(_frame(rows))
    assert list(got.index) == [(pd.Timestamp(_DAY), _SYM)]
    assert got.iloc[0] == 10.3 / 10.0 - 1.0  # bit-exact
    assert got.name == SESSION_RET_FACTOR_NAME


# --------------------------------------------------------------------------- #
# 2. PIT: post-cutoff bars cannot move the value
# --------------------------------------------------------------------------- #
def test_post_cutoff_bars_cannot_move_the_value():
    base = RANDOM_BARS
    trade_date = base["bar_end"].dt.normalize()
    mask = (base["available_time"] > trade_date + pd.Timedelta("14:50:00")).to_numpy()
    assert mask.any(), "fixture has no post-cutoff bars — vacuous perturbation"
    out = base.copy(deep=True)
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = out.loc[mask, col] * 137.0
    assert not out.loc[mask].equals(base.loc[mask]), "perturbation changed nothing"
    pd.testing.assert_frame_equal(base.loc[~mask], out.loc[~mask], check_exact=True)

    pd.testing.assert_series_equal(
        compute_intraday_session_ret(out),
        compute_intraday_session_ret(base),
        check_exact=True,
    )


# --------------------------------------------------------------------------- #
# 3. equivalence with the legacy aggregate feature
# --------------------------------------------------------------------------- #
def test_equivalent_to_asof_daily_features_ret_bit_for_bit():
    frame = asof_daily_features(RANDOM_BARS, features=["ret"])
    legacy = frame[frame.columns[0]]
    got = compute_intraday_session_ret(RANDOM_BARS)
    # the factor id IS the legacy column name (report text keyed on it verbatim)
    assert got.name == legacy.name == SESSION_RET_FACTOR_NAME
    assert int(np.isfinite(got.to_numpy()).sum()) > 0, "vacuous: all NaN"
    pd.testing.assert_series_equal(got, legacy, check_exact=True)


# --------------------------------------------------------------------------- #
# 4. per-symbol split == whole
# --------------------------------------------------------------------------- #
def test_per_symbol_split_reproduces_the_whole_frame_call():
    whole = compute_intraday_session_ret(RANDOM_BARS)
    parts = []
    for sym in pd.Index(RANDOM_BARS.index.get_level_values("symbol")).unique():
        one = RANDOM_BARS[pd.Index(RANDOM_BARS.index.get_level_values("symbol")) == sym]
        parts.append(compute_intraday_session_ret(one))
    split = pd.concat(parts).sort_index()
    pd.testing.assert_series_equal(split, whole.sort_index(), check_exact=True)


# --------------------------------------------------------------------------- #
# 5. factor surface: registry + spec + compute
# --------------------------------------------------------------------------- #
def test_registry_builds_the_exact_name():
    factor = build(SESSION_RET_FACTOR_NAME)
    assert isinstance(factor, IntradaySessionRetFactor)
    assert factor.name == SESSION_RET_FACTOR_NAME
    assert factor.spec.factor_id == SESSION_RET_FACTOR_NAME


def test_spec_declares_the_full_intraday_block_and_d1_taxonomy():
    spec = IntradaySessionRetFactor().spec
    assert spec.is_intraday is True
    assert spec.return_basis == "exec_to_exec"
    assert spec.decision_cutoff == "14:50:00"
    assert spec.data_lag == "1min"
    assert spec.session_open == "09:30:00"
    assert spec.execution_model == "next_minute_close"
    assert spec.execution_window == "[14:51:00,14:56:59]"
    assert spec.expected_ic_sign in (+1, -1)
    assert [(r.field, r.source) for r in spec.requires] == [
        (f, "stk_mins_1min") for f in ("open", "close")
    ]
    assert spec.adjustment.value == "returns_invariant"
    assert spec.overnight_boundary.value == "none"
    assert spec.lookback_depth == 1


def test_compute_surfaces_the_preaggregated_panel_column():
    factor = IntradaySessionRetFactor()
    series = compute_intraday_session_ret(RANDOM_BARS)
    panel = series.to_frame(factor.name)
    pd.testing.assert_series_equal(factor.compute(panel), series, check_exact=True)
    with pytest.raises(ValueError, match="pre-aggregated"):
        factor.compute(panel.rename(columns={factor.name: "something_else"}))
