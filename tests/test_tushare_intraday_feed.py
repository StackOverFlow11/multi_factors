"""TushareIntradayFeed tests — no network, fake SDK, no token leak."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data.clean.intraday_schema import validate_intraday_bars
from data.feed.tushare_intraday import TushareIntradayFeed


class _Pro:
    """Fake tushare client. stk_mins returns reverse-chronological raw rows."""

    def __init__(self):
        self.calls: list[tuple] = []

    def stk_mins(self, ts_code, freq, start_date, end_date):
        self.calls.append((ts_code, freq, start_date, end_date))
        if ts_code == "000001.SZ":
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "trade_time": [
                        "2024-01-02 09:32:00",  # reverse order on purpose
                        "2024-01-02 09:31:00",
                    ],
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [12.0, 11.0],
                    "vol": [200.0, 100.0],
                    "amount": [2.0, 1.0],
                }
            )
        return pd.DataFrame()  # empty for any other symbol


def _feed(monkeypatch, pro=None):
    pro = pro or _Pro()
    feed = TushareIntradayFeed("unused.json")
    monkeypatch.setattr(feed, "_client", lambda: pro)
    return feed, pro


def test_get_minutes_maps_and_shapes(monkeypatch):
    feed, _ = _feed(monkeypatch)
    out = feed.get_minutes(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00", freq="1min"
    )
    assert list(out.index.names) == ["time", "symbol"]
    assert "volume" in out.columns and "vol" not in out.columns
    # reverse-chronological input is sorted ascending
    times = list(out.index.get_level_values("time"))
    assert times == sorted(times)
    assert out.iloc[0]["close"] == 11.0  # 09:31 comes first


def test_vol_and_ts_code_mapping_and_bar_end(monkeypatch):
    feed, _ = _feed(monkeypatch)
    out = feed.get_minutes(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00"
    )
    assert out.iloc[0]["volume"] == 100.0  # vol -> volume
    assert out.index.get_level_values("symbol")[0] == "000001.SZ"  # ts_code -> symbol
    assert out.iloc[0]["bar_end"] == pd.Timestamp("2024-01-02 09:31:00")  # trade_time
    assert out.iloc[0]["available_time"] == pd.Timestamp("2024-01-02 09:32:00")


def test_empty_return_is_schema_shaped(monkeypatch):
    feed, _ = _feed(monkeypatch)
    out = feed.get_minutes(
        ["999999.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00"
    )
    assert len(out) == 0
    assert list(out.index.names) == ["time", "symbol"]
    validate_intraday_bars(out)  # must not raise


def test_multi_symbol_concat_and_sort(monkeypatch):
    class _P:
        def stk_mins(self, ts_code, freq, start_date, end_date):  # noqa: ARG002
            return pd.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_time": ["2024-01-02 09:31:00"],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "vol": [1.0],
                    "amount": [1.0],
                }
            )

    feed, _ = _feed(monkeypatch, _P())
    out = feed.get_minutes(
        ["000002.SZ", "000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00"
    )
    assert sorted(set(out.index.get_level_values("symbol"))) == [
        "000001.SZ",
        "000002.SZ",
    ]


def test_stk_mins_called_with_datetime_window(monkeypatch):
    feed, pro = _feed(monkeypatch)
    feed.get_minutes(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00", freq="1min"
    )
    ts_code, freq, start_date, end_date = pro.calls[0]
    assert ts_code == "000001.SZ"
    assert freq == "1min"
    assert start_date == "2024-01-02 09:30:00"
    assert end_date == "2024-01-02 09:35:00"


@pytest.mark.parametrize("freq", ["5min", "2min"])
def test_non_1min_raw_freq_rejected_before_sdk_call(monkeypatch, freq):
    feed, pro = _feed(monkeypatch)
    with pytest.raises(ValueError, match="freq"):
        feed.get_minutes(
            ["000001.SZ"],
            "2024-01-02 09:30:00",
            "2024-01-02 09:35:00",
            freq=freq,
        )
    assert pro.calls == []


def test_empty_symbols_rejected(monkeypatch):
    feed, _ = _feed(monkeypatch)
    with pytest.raises(ValueError, match="symbol"):
        feed.get_minutes([], "2024-01-02 09:30:00", "2024-01-02 09:35:00")


def test_no_token_leak(tmp_path, monkeypatch):
    secret = tmp_path / "config.json"
    fake_token = "FAKE_TOKEN_DO_NOT_LEAK_abcdef0123456789"
    secret.write_text(json.dumps({"tushare": {"token": fake_token}}))
    feed = TushareIntradayFeed(str(secret))
    monkeypatch.setattr(feed, "_client", lambda: _Pro())  # never reads the token
    out = feed.get_minutes(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00"
    )
    assert fake_token not in repr(feed)
    assert fake_token not in out.to_csv()
