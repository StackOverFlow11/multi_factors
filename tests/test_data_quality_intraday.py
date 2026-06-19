"""D3 report-only 1min intraday quality checks — synthetic, no network/cache."""

from __future__ import annotations

import pandas as pd

from data.quality.intraday import (
    check_close_outside_range,
    check_duplicate_bars,
    check_high_low_inversion,
    check_missing_minutes,
    check_negative_volume_amount,
    check_non_monotonic_time,
    check_non_positive_ohlc,
    run_intraday_checks,
)
from data.quality.report import HARD, WARNING, has_hard


def _clean() -> pd.DataFrame:
    """A small clean 1min frame (1 symbol x 3 consecutive minutes)."""
    times = pd.to_datetime(
        ["2024-01-03 09:31:00", "2024-01-03 09:32:00", "2024-01-03 09:33:00"]
    )
    return pd.DataFrame(
        {
            "bar_end": times,
            "symbol": ["000001.SZ"] * 3,
            "open": [10.0, 10.1, 10.2], "high": [10.3, 10.4, 10.5],
            "low": [9.9, 10.0, 10.1], "close": [10.1, 10.2, 10.3],
            "volume": [100.0, 110.0, 120.0], "amount": [1000.0, 1100.0, 1200.0],
        }
    )


def test_clean_intraday_has_zero_hard_findings():
    findings = run_intraday_checks(_clean())
    assert findings == []
    assert not has_hard(findings)


def test_clean_intraday_as_multiindex_also_clean():
    panel = _clean().rename(columns={"bar_end": "time"}).set_index(["time", "symbol"])
    # MultiIndex (time, symbol); reset_keys promotes -> 'time' is the bar column
    assert run_intraday_checks(panel) == []


def test_duplicate_bars_caught():
    df = _clean()
    df2 = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    f = check_duplicate_bars(df2)
    assert f is not None and f.severity == HARD and f.check == "duplicate_bars"
    assert f.count == 2


def test_non_monotonic_time_caught():
    df = _clean()
    # swap rows 1 and 2 so the timestamp steps backwards within the symbol
    df2 = df.iloc[[0, 2, 1]].reset_index(drop=True)
    f = check_non_monotonic_time(df2)
    assert f is not None and f.severity == HARD and f.count == 1


def test_non_monotonic_clean_in_order():
    assert check_non_monotonic_time(_clean()) is None


def test_non_positive_ohlc_caught():
    df = _clean()
    df.loc[0, "low"] = 0.0
    f = check_non_positive_ohlc(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_high_low_inversion_caught():
    df = _clean()
    df.loc[1, "high"] = 1.0
    df.loc[1, "low"] = 5.0
    f = check_high_low_inversion(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_close_outside_range_caught():
    df = _clean()
    df.loc[2, "close"] = 999.0
    f = check_close_outside_range(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_negative_volume_amount_caught():
    df = _clean()
    df.loc[0, "amount"] = -1.0
    f = check_negative_volume_amount(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_missing_minutes_caught_with_calendar():
    df = _clean()
    # drop the 09:32 bar
    df = df[df["bar_end"] != pd.Timestamp("2024-01-03 09:32:00")]
    cal = [
        "2024-01-03 09:31:00", "2024-01-03 09:32:00", "2024-01-03 09:33:00",
    ]
    f = check_missing_minutes(df, cal)
    assert f is not None and f.severity == WARNING and f.count == 1
    assert f.examples[0]["symbol"] == "000001.SZ"


def test_missing_minutes_none_without_calendar():
    assert check_missing_minutes(_clean(), None) is None
