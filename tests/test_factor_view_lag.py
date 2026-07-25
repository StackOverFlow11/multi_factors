"""D4: availability-lag transforms (factors.view_lag) — R18 per field-class.

These pin the pure transforms the materializer composes: the daily prev-day shift
with the field-level ``open`` exception, the minute cutoff, and the ex-date
BOOLEAN. The materializer's end-to-end R18 coverage rides on top (see
tests/test_factor_materialize.py); here we prove the transforms in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from factors.view_lag import (
    daily_decision_lag,
    decision_same_day_fields,
    ex_date_mask,
    minute_decision_cutoff,
)

SYMS = ["000001.SZ", "000002.SZ"]
DATES = pd.bdate_range("2024-01-02", periods=6)


def _daily():
    rows = []
    for si, s in enumerate(SYMS):
        for di, d in enumerate(DATES):
            base = 100.0 + si * 10 + di
            rows.append((d, s, base + 0.1, base + 0.5, base - 0.5, base, 1e5 + di, base * 1e5))
    return (
        pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
        .set_index(["date", "symbol"]).sort_index()
    )


def test_decision_same_day_fields_is_open_only():
    """Policy-derived: the ONLY decision-view same-day factor field is open (R7)."""
    assert decision_same_day_fields() == frozenset({"open"})


def test_daily_decision_lag_shifts_prev_day_keeps_open():
    """close/high/low/volume/amount at d become d-1; open stays same-day."""
    panel = _daily()
    lagged = daily_decision_lag(panel)
    s = SYMS[0]
    for di in range(1, len(DATES)):
        d, prev = DATES[di], DATES[di - 1]
        # open is same-day: unchanged.
        assert lagged.loc[(d, s), "open"] == panel.loc[(d, s), "open"]
        # prev-day fields carry the d-1 value.
        for col in ("close", "high", "low", "volume", "amount"):
            assert lagged.loc[(d, s), col] == panel.loc[(prev, s), col]
    # the first date of each symbol has no d-1 -> shifted cols NaN, open present.
    d0 = DATES[0]
    assert np.isnan(lagged.loc[(d0, s), "close"])
    assert lagged.loc[(d0, s), "open"] == panel.loc[(d0, s), "open"]


def test_daily_decision_lag_does_not_mutate_input():
    panel = _daily()
    before = panel.copy()
    daily_decision_lag(panel)
    pd.testing.assert_frame_equal(panel, before)


def test_daily_decision_lag_is_per_symbol():
    """The shift never pulls one symbol's row into another's."""
    panel = _daily()
    lagged = daily_decision_lag(panel)
    # symbol B's first date is NaN independently of symbol A.
    assert np.isnan(lagged.loc[(DATES[0], SYMS[1]), "close"])


def _bars(rows):
    df = pd.DataFrame(
        rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    )
    return normalize_intraday_bars(df, freq="1min")


def test_minute_decision_cutoff_drops_post_1450_bars():
    """Bars whose available_time exceeds d 14:50 are removed; earlier kept."""
    day = "2024-01-02"
    rows = []
    # bars at 14:48, 14:49 (kept), 14:50, 14:55 (dropped: available_time = bar_end+1min)
    for hhmm in ("14:48", "14:49", "14:50", "14:55"):
        t = pd.Timestamp(f"{day} {hhmm}:00")
        rows.append((t, SYMS[0], 100.0, 100.5, 99.5, 100.0, 1e4, 1e6))
    bars = _bars(rows)
    kept = minute_decision_cutoff(bars, decision_time="14:50:00")
    kept_times = kept.index.get_level_values("time").strftime("%H:%M").tolist()
    # available_time = bar_end + 1min: 14:48->14:49(<=14:50 keep), 14:49->14:50(keep),
    # 14:50->14:51(drop), 14:55->14:56(drop).
    assert kept_times == ["14:48", "14:49"]


def test_minute_decision_cutoff_does_not_mutate_input():
    day = "2024-01-02"
    rows = [(pd.Timestamp(f"{day} 14:{m}:00"), SYMS[0], 100.0, 100.5, 99.5, 100.0, 1e4, 1e6)
            for m in (40, 55)]
    bars = _bars(rows)
    before = bars.copy()
    minute_decision_cutoff(bars)
    pd.testing.assert_frame_equal(bars, before)


def test_ex_date_mask_is_boolean_and_flags_af_changes():
    """af(d) != af(d-1) per symbol; first row False; NaN-aware."""
    idx = pd.MultiIndex.from_tuples(
        [(DATES[0], "A"), (DATES[1], "A"), (DATES[2], "A"),
         (DATES[0], "B"), (DATES[1], "B")],
        names=["date", "symbol"],
    )
    af = pd.Series([1.0, 1.0, 2.0, 3.0, 3.0], index=idx)
    mask = ex_date_mask(af)
    assert mask.dtype == bool
    assert mask.loc[(DATES[0], "A")] is np.False_ or not mask.loc[(DATES[0], "A")]  # first row
    assert not mask.loc[(DATES[1], "A")]  # 1.0 -> 1.0 no change
    assert mask.loc[(DATES[2], "A")]      # 1.0 -> 2.0 change
    assert not mask.loc[(DATES[0], "B")]  # first row of B
    assert not mask.loc[(DATES[1], "B")]  # 3.0 -> 3.0


def test_ex_date_mask_rejects_non_multiindex():
    with pytest.raises(ValueError):
        ex_date_mask(pd.Series([1.0, 2.0]))
