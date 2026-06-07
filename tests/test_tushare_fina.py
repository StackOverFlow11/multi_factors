"""TushareFinancialFeed mapping tests — no network, fake SDK."""

from __future__ import annotations

import pandas as pd

from data.feed.tushare_fina import TushareFinancialFeed


class _Pro:
    def fina_indicator(self, ts_code, start_date, end_date, fields):  # noqa: ARG002
        return pd.DataFrame(
            {
                "ts_code": [ts_code],
                "ann_date": ["20240420"],
                "end_date": ["20240331"],
                "roe": [3.1],
            }
        )


def test_get_fina_indicator_maps_columns(monkeypatch):
    feed = TushareFinancialFeed("x.json")
    monkeypatch.setattr(feed, "_client", lambda: _Pro())
    out = feed.get_fina_indicator(["000001.SZ"], "2024-01-01", "2024-12-31", fields=["roe"])
    assert list(out.columns) == ["symbol", "ann_date", "end_date", "roe"]
    assert out.iloc[0]["symbol"] == "000001.SZ"
    assert out.iloc[0]["roe"] == 3.1


def test_get_fina_indicator_empty_is_schema_shaped(monkeypatch):
    feed = TushareFinancialFeed("x.json")

    class _Empty:
        def fina_indicator(self, **_kw):
            return pd.DataFrame()

    monkeypatch.setattr(feed, "_client", lambda: _Empty())
    out = feed.get_fina_indicator(["000001.SZ"], "2024-01-01", "2024-12-31", fields=["roe"])
    assert list(out.columns) == ["symbol", "ann_date", "end_date", "roe"]
    assert len(out) == 0
