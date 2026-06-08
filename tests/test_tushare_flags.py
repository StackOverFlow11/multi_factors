"""TushareFlagsFeed mapping tests — no network, fake SDK."""

from __future__ import annotations

import pandas as pd

from data.feed.tushare_flags import TushareFlagsFeed


class _Pro:
    def suspend_d(self, ts_code, start_date, end_date, suspend_type):  # noqa: ARG002
        if ts_code == "000001.SZ":
            return pd.DataFrame(
                {"ts_code": ["000001.SZ"], "trade_date": ["20240304"], "suspend_type": ["S"]}
            )
        return pd.DataFrame()

    def namechange(self, ts_code):  # noqa: ARG002
        # duplicated rows + an ST and a non-ST interval
        return pd.DataFrame(
            {
                "ts_code": [ts_code, ts_code, ts_code],
                "name": ["*ST X", "*ST X", "X"],
                "start_date": ["20240301", "20240301", "20230101"],
                "end_date": [None, None, "20240229"],
                "ann_date": ["20240229", "20240229", "20221231"],
                "change_reason": ["ST", "ST", "撤销"],
            }
        )

    def stk_limit(self, ts_code, start_date, end_date):  # noqa: ARG002
        return pd.DataFrame(
            {"trade_date": ["20240304"], "ts_code": [ts_code], "up_limit": [11.0], "down_limit": [9.0]}
        )


def _feed(monkeypatch):
    feed = TushareFlagsFeed("x.json")
    monkeypatch.setattr(feed, "_client", lambda: _Pro())
    return feed


def test_suspended_returns_date_symbol_pairs(monkeypatch):
    feed = _feed(monkeypatch)
    out = feed.suspended(["000001.SZ", "000002.SZ"], "2024-03-01", "2024-03-31")
    assert (pd.Timestamp("2024-03-04"), "000001.SZ") in out
    assert all(sym != "000002.SZ" for _, sym in out)  # 000002 not suspended


def test_st_intervals_dedupe_and_flag(monkeypatch):
    feed = _feed(monkeypatch)
    intervals = feed.st_intervals(["000001.SZ"])["000001.SZ"]
    # duplicate (*ST X) rows collapsed to one; an ST interval is present
    assert len(intervals) == 2
    assert any(is_st for _, _, is_st in intervals)
    assert any(not is_st for _, _, is_st in intervals)


def test_limits_maps_columns(monkeypatch):
    feed = _feed(monkeypatch)
    lim = feed.limits(["000001.SZ"], "2024-03-01", "2024-03-31")
    assert list(lim.columns) == ["date", "symbol", "up_limit", "down_limit"]
    assert lim.iloc[0]["up_limit"] == 11.0
    assert lim.iloc[0]["down_limit"] == 9.0
    assert str(lim["date"].dtype).startswith("datetime64")
