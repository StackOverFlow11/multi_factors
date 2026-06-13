"""Intraday schema tests — minute-precision time, PIT derivation, dedup, sort."""

from __future__ import annotations

import pandas as pd
import pytest

from data.clean.intraday_schema import (
    empty_intraday_bars,
    normalize_intraday_bars,
    validate_intraday_bars,
)


def _bars(times, symbol="000001.SZ", close=10.0):
    """Build a raw intraday input frame (columns time, symbol, OHLCV)."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(list(times)),
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "amount": 10000.0,
        }
    )


def test_time_preserves_minute_precision():
    # The daily contract normalizes to midnight; intraday must NOT.
    out = normalize_intraday_bars(
        _bars(["2024-01-02 09:31:00", "2024-01-02 09:32:00"]), freq="1min"
    )
    times = out.index.get_level_values("time")
    assert times[0] == pd.Timestamp("2024-01-02 09:31:00")
    assert times[0].hour == 9 and times[0].minute == 31  # not midnight


def test_raw_1min_bar_window_and_available_time():
    out = normalize_intraday_bars(
        _bars(["2024-01-02 09:31:00"]), freq="1min", data_lag="1min"
    )
    row = out.iloc[0]
    assert row["bar_end"] == pd.Timestamp("2024-01-02 09:31:00")
    assert row["bar_start"] == pd.Timestamp("2024-01-02 09:30:00")  # end - 1min
    assert row["available_time"] == pd.Timestamp("2024-01-02 09:32:00")  # end + lag
    assert row["freq"] == "1min"


def test_schema_can_represent_future_derived_coarser_bars():
    out = normalize_intraday_bars(
        _bars(["2024-01-02 09:35:00"]), freq="5min", data_lag="1min"
    )
    row = out.iloc[0]
    assert row["bar_start"] == pd.Timestamp("2024-01-02 09:30:00")
    assert row["freq"] == "5min"


def test_data_lag_shifts_available_time():
    out = normalize_intraday_bars(
        _bars(["2024-01-02 09:31:00"]), freq="1min", data_lag="3min"
    )
    assert out.iloc[0]["available_time"] == pd.Timestamp("2024-01-02 09:34:00")


def test_reverse_chronological_input_sorted_ascending():
    out = normalize_intraday_bars(
        _bars(
            [
                "2024-01-02 09:33:00",
                "2024-01-02 09:31:00",
                "2024-01-02 09:32:00",
            ]
        ),
        freq="1min",
    )
    times = list(out.index.get_level_values("time"))
    assert times == sorted(times)


def test_sorted_by_time_then_symbol():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-02 09:31:00"] * 2),
            "symbol": ["000002.SZ", "000001.SZ"],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "amount": 1.0,
        }
    )
    out = normalize_intraday_bars(df, freq="1min")
    assert list(out.index.get_level_values("symbol")) == ["000001.SZ", "000002.SZ"]


def test_duplicate_key_last_row_wins():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-02 09:31:00", "2024-01-02 09:31:00"]
            ),
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [10.0, 20.0],  # later row should win
            "volume": [100.0, 200.0],
            "amount": [1.0, 2.0],
        }
    )
    out = normalize_intraday_bars(df, freq="1min")
    assert len(out) == 1
    assert out.iloc[0]["close"] == 20.0
    assert out.iloc[0]["volume"] == 200.0


def test_missing_core_column_readable_error():
    df = _bars(["2024-01-02 09:31:00"]).drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        normalize_intraday_bars(df, freq="1min")


def test_missing_time_symbol_readable_error():
    df = pd.DataFrame({"open": [1.0]})
    with pytest.raises(ValueError, match="time"):
        normalize_intraday_bars(df, freq="1min")


def test_unsupported_freq_readable_error():
    with pytest.raises(ValueError, match="freq"):
        normalize_intraday_bars(_bars(["2024-01-02 09:31:00"]), freq="2min")


def test_source_trade_time_preserved_as_extra():
    df = _bars(["2024-01-02 09:31:00"])
    df["source_trade_time"] = "2024-01-02 09:31:00"
    out = normalize_intraday_bars(df, freq="1min")
    assert "source_trade_time" in out.columns


def test_normalize_does_not_mutate_input():
    df = _bars(["2024-01-02 09:31:00", "2024-01-02 09:30:00"])
    before = df.copy(deep=True)
    normalize_intraday_bars(df, freq="1min")
    pd.testing.assert_frame_equal(df, before)


def test_empty_intraday_bars_schema_and_validate():
    e = empty_intraday_bars()
    assert list(e.index.names) == ["time", "symbol"]
    for c in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "freq",
        "bar_start",
        "bar_end",
        "available_time",
    ):
        assert c in e.columns
    assert len(e) == 0
    validate_intraday_bars(e)  # must not raise


def test_validate_accepts_normalized_and_rejects_raw_frame():
    out = normalize_intraday_bars(
        _bars(["2024-01-02 09:31:00", "2024-01-02 09:32:00"]), freq="1min"
    )
    validate_intraday_bars(out)  # passes
    with pytest.raises(ValueError):
        validate_intraday_bars(_bars(["2024-01-02 09:31:00"]))  # not indexed
