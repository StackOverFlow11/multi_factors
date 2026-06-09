"""TushareCovariatesFeed mapping tests — no network, fake SDK."""

from __future__ import annotations

import pandas as pd

from data.feed.tushare_covariates import TushareCovariatesFeed


class _Pro:
    def stock_basic(self, fields):  # noqa: ARG002
        return pd.DataFrame(
            {"ts_code": ["000001.SZ", "999999.SZ"], "industry": ["银行", "其他"]}
        )

    def daily_basic(self, ts_code, start_date, end_date, fields):  # noqa: ARG002
        return pd.DataFrame(
            {"ts_code": [ts_code], "trade_date": ["20240301"], "total_mv": [123.0]}
        )

    def index_member_all(self, ts_code):
        # one active SW membership (out_date None) carrying all three levels.
        return pd.DataFrame(
            {
                "l1_name": ["食品饮料"],
                "l2_name": ["白酒Ⅱ"],
                "l3_name": ["白酒Ⅲ"],
                "in_date": ["20010731"],
                "out_date": [None],
                "ts_code": [ts_code],
            }
        )


def _feed(monkeypatch):
    feed = TushareCovariatesFeed("x.json")
    monkeypatch.setattr(feed, "_client", lambda: _Pro())
    return feed


def test_industry_filters_to_requested(monkeypatch):
    assert _feed(monkeypatch).industry(["000001.SZ"]) == {"000001.SZ": "银行"}


def test_market_cap_maps_columns(monkeypatch):
    out = _feed(monkeypatch).market_cap(["000001.SZ"], "2024-03-01", "2024-03-31")
    assert list(out.columns) == ["date", "symbol", "market_cap"]
    assert out.iloc[0]["market_cap"] == 123.0
    assert str(out["date"].dtype).startswith("datetime64")


def test_pit_sw_intervals_default_level_is_l1(monkeypatch):
    out = _feed(monkeypatch).pit_sw_intervals(["600519.SH"])  # default L1
    name, in_d, out_d = out["600519.SH"][0]
    assert name == "食品饮料"  # L1 broad sector (default)
    assert in_d == pd.Timestamp("2001-07-31")
    assert out_d is None  # active membership


def test_pit_sw_intervals_selects_level(monkeypatch):
    feed = _feed(monkeypatch)
    assert feed.pit_sw_intervals(["600519.SH"], level="L1")["600519.SH"][0][0] == "食品饮料"
    assert feed.pit_sw_intervals(["600519.SH"], level="L2")["600519.SH"][0][0] == "白酒Ⅱ"
    assert feed.pit_sw_intervals(["600519.SH"], level="L3")["600519.SH"][0][0] == "白酒Ⅲ"


def test_pit_sw_intervals_bad_level_raises(monkeypatch):
    import pytest

    with pytest.raises(ValueError, match="industry level"):
        _feed(monkeypatch).pit_sw_intervals(["600519.SH"], level="L9")
