"""Intraday -> daily PIT aggregation tests (I3): cutoff, leakage, isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_aggregate import (
    asof_daily_features,
    resample_intraday_bars,
)
from data.clean.intraday_schema import (
    empty_intraday_bars,
    normalize_intraday_bars,
    validate_intraday_bars,
)

_RET = "intraday_ret_0930_1450"
_VOL = "intraday_realized_vol_0930_1450"
_VWAP = "intraday_vwap_0930_1450"
_L30 = "intraday_last30m_ret_1420_1450"


def _norm(rows, data_lag="1min"):
    """rows = [(time_str, symbol, close), ...] -> normalized 1min bars."""
    cl = [r[2] for r in rows]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": cl,
            "high": [c + 0.5 for c in cl],
            "low": [c - 0.5 for c in cl],
            "close": cl,
            "volume": [100.0] * len(rows),
            "amount": [10.0 * c for c in cl],
        }
    )
    return normalize_intraday_bars(df, freq="1min", data_lag=data_lag)


_DAY = "2024-01-02"


def _one_symbol_day(extra_after_cutoff=False):
    rows = [
        (f"{_DAY} 09:31:00", "000001.SZ", 10.0),
        (f"{_DAY} 09:32:00", "000001.SZ", 10.5),
        (f"{_DAY} 14:00:00", "000001.SZ", 11.0),
        (f"{_DAY} 14:30:00", "000001.SZ", 11.5),
        (f"{_DAY} 14:48:00", "000001.SZ", 12.0),
        (f"{_DAY} 14:49:00", "000001.SZ", 12.5),
    ]
    if extra_after_cutoff:
        rows += [
            (f"{_DAY} 14:51:00", "000001.SZ", 99.0),
            (f"{_DAY} 14:55:00", "000001.SZ", 99.0),
        ]
    return rows


# --------------------------------------------------------------------------- #
# basic shape + column names
# --------------------------------------------------------------------------- #
def test_columns_encode_cutoff_and_index_is_daily():
    out = asof_daily_features(_norm(_one_symbol_day()))
    assert list(out.columns) == [_RET, _VOL, _VWAP, _L30]
    assert list(out.index.names) == ["date", "symbol"]
    assert len(out) == 1
    (date, sym) = out.index[0]
    assert sym == "000001.SZ"
    assert date == pd.Timestamp("2024-01-02")  # midnight, not a minute timestamp
    assert date.hour == 0 and date.minute == 0


def test_ret_value_open_to_last_visible():
    out = asof_daily_features(_norm(_one_symbol_day()))
    # last visible close 12.5 (14:49) / first open 10.0 - 1
    assert out[_RET].iloc[0] == pytest.approx(12.5 / 10.0 - 1.0)
    # last30m: ref = last bar with bar_end <= 14:20 -> 14:00 close 11.0; last 12.5
    assert out[_L30].iloc[0] == pytest.approx(12.5 / 11.0 - 1.0)
    assert out[_VOL].iloc[0] > 0


# --------------------------------------------------------------------------- #
# PIT: cutoff leakage + availability
# --------------------------------------------------------------------------- #
def test_post_cutoff_bars_do_not_leak():
    base = asof_daily_features(_norm(_one_symbol_day(extra_after_cutoff=False)))
    # identical pre-14:50 bars, plus post-14:50 bars with garbage prices
    perturbed = asof_daily_features(_norm(_one_symbol_day(extra_after_cutoff=True)))
    pd.testing.assert_frame_equal(base, perturbed)


def test_delayed_availability_excludes_bar():
    bars = _norm(_one_symbol_day())
    base = asof_daily_features(bars)
    # push the 14:00 bar's availability past the 14:50 cutoff
    delayed = bars.copy()
    mask = delayed["bar_end"] == pd.Timestamp("2024-01-02 14:00:00")
    delayed.loc[mask, "available_time"] = pd.Timestamp("2024-01-02 15:00:00")
    got = asof_daily_features(delayed)
    # equals computing on bars with that bar physically removed
    dropped = bars[bars["bar_end"] != pd.Timestamp("2024-01-02 14:00:00")]
    expected = asof_daily_features(dropped)
    pd.testing.assert_frame_equal(got, expected)
    # and the exclusion actually changed something (vwap sums fewer bars)
    assert got[_VWAP].iloc[0] != base[_VWAP].iloc[0]


def test_all_bars_after_cutoff_returns_empty():
    rows = [
        (f"{_DAY} 14:51:00", "000001.SZ", 10.0),
        (f"{_DAY} 14:52:00", "000001.SZ", 11.0),
    ]
    out = asof_daily_features(_norm(rows))
    assert len(out) == 0
    assert list(out.index.names) == ["date", "symbol"]
    assert list(out.columns) == [_RET, _VOL, _VWAP, _L30]


# --------------------------------------------------------------------------- #
# multi-symbol / multi-day isolation
# --------------------------------------------------------------------------- #
def test_multi_symbol_multi_day_no_contamination():
    rows = [
        ("2024-01-02 09:31:00", "000001.SZ", 10.0),
        ("2024-01-02 09:32:00", "000001.SZ", 11.0),  # ret 0.10
        ("2024-01-03 09:31:00", "000001.SZ", 20.0),
        ("2024-01-03 09:32:00", "000001.SZ", 22.0),  # ret 0.10
        ("2024-01-02 09:31:00", "000002.SZ", 5.0),
        ("2024-01-02 09:32:00", "000002.SZ", 6.0),   # ret 0.20
    ]
    out = asof_daily_features(_norm(rows))
    assert len(out) == 3
    assert out.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), _RET] == pytest.approx(0.10)
    assert out.loc[(pd.Timestamp("2024-01-03"), "000001.SZ"), _RET] == pytest.approx(0.10)
    assert out.loc[(pd.Timestamp("2024-01-02"), "000002.SZ"), _RET] == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# empty input / feature selection
# --------------------------------------------------------------------------- #
def test_empty_input_returns_schema_shaped_empty():
    out = asof_daily_features(empty_intraday_bars())
    assert len(out) == 0
    assert list(out.index.names) == ["date", "symbol"]
    assert list(out.columns) == [_RET, _VOL, _VWAP, _L30]


def test_feature_subset_and_unknown():
    out = asof_daily_features(_norm(_one_symbol_day()), features=["vwap"])
    assert list(out.columns) == [_VWAP]
    with pytest.raises(ValueError, match="Unknown intraday feature"):
        asof_daily_features(_norm(_one_symbol_day()), features=["bogus"])


def test_custom_decision_time_relabels_columns():
    out = asof_daily_features(_norm(_one_symbol_day()), decision_time="14:30:00")
    assert "intraday_ret_0930_1430" in out.columns
    assert "intraday_last30m_ret_1400_1430" in out.columns
    # cutoff 14:30: the 14:30 bar's availability is 14:31 (1min lag) > 14:30, so it
    # is excluded too -> last visible bar is 14:00 (close 11.0).
    assert out["intraday_ret_0930_1430"].iloc[0] == pytest.approx(11.0 / 10.0 - 1.0)


# --------------------------------------------------------------------------- #
# derived coarse bars: available_time = max(source 1min)
# --------------------------------------------------------------------------- #
def test_resample_available_time_is_source_max():
    rows = [
        (f"{_DAY} 09:31:00", "000001.SZ", 10.0),
        (f"{_DAY} 09:32:00", "000001.SZ", 11.0),
        (f"{_DAY} 09:33:00", "000001.SZ", 12.0),
        (f"{_DAY} 09:34:00", "000001.SZ", 9.0),
        (f"{_DAY} 09:35:00", "000001.SZ", 13.0),
    ]
    bars = _norm(rows, data_lag="1min")
    coarse = resample_intraday_bars(bars, "5min")
    validate_intraday_bars(coarse)
    assert len(coarse) == 1
    row = coarse.iloc[0]
    # one 5min bar ending 09:35
    assert coarse.index.get_level_values("time")[0] == pd.Timestamp("2024-01-02 09:35:00")
    assert row["bar_end"] == pd.Timestamp("2024-01-02 09:35:00")
    assert row["bar_start"] == pd.Timestamp("2024-01-02 09:30:00")
    assert row["freq"] == "5min"
    # OHLC aggregation
    assert row["open"] == 10.0          # first
    assert row["close"] == 13.0         # last
    assert row["high"] == 13.0 + 0.5    # max high
    assert row["low"] == 9.0 - 0.5      # min low
    assert row["volume"] == 500.0       # 5 * 100
    # availability inherits the MAX source 1min available_time (09:35 + 1min lag)
    src_max = bars["available_time"].max()
    assert row["available_time"] == src_max == pd.Timestamp("2024-01-02 09:36:00")


def test_resample_keeps_symbols_separate():
    rows = [
        (f"{_DAY} 09:31:00", "000001.SZ", 10.0),
        (f"{_DAY} 09:32:00", "000001.SZ", 11.0),
        (f"{_DAY} 09:31:00", "000002.SZ", 50.0),
        (f"{_DAY} 09:32:00", "000002.SZ", 52.0),
    ]
    coarse = resample_intraday_bars(_norm(rows), "5min")
    syms = sorted(set(coarse.index.get_level_values("symbol")))
    assert syms == ["000001.SZ", "000002.SZ"]
    # each symbol's coarse close is its own last 1min close
    a = coarse.xs("000001.SZ", level="symbol").iloc[0]
    b = coarse.xs("000002.SZ", level="symbol").iloc[0]
    assert a["close"] == 11.0 and b["close"] == 52.0


def test_resample_rejects_unsupported_freq():
    with pytest.raises(ValueError, match="freq"):
        resample_intraday_bars(_norm(_one_symbol_day()), "7min")


def test_resample_realized_vol_uses_log_returns():
    # sanity: a known 2-bar vol so the feature is not silently zero
    rows = [
        (f"{_DAY} 09:31:00", "000001.SZ", 10.0),
        (f"{_DAY} 09:32:00", "000001.SZ", 11.0),
    ]
    out = asof_daily_features(_norm(rows))
    expected = abs(np.log(11.0 / 10.0))
    assert out[_VOL].iloc[0] == pytest.approx(expected)
