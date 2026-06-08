"""IndexConstituentsFeed mapping tests — no network, fake SDK."""

from __future__ import annotations

import pandas as pd

from data.feed.index_feed import IndexConstituentsFeed


class _FakePro:
    """Stand-in for the tushare pro client returning an index_weight frame."""

    def index_weight(self, index_code, start_date, end_date):  # noqa: ARG002
        return pd.DataFrame(
            {
                "index_code": ["000300.SH"] * 3,
                "con_code": ["000002.SZ", "000001.SZ", "000001.SZ"],
                "trade_date": ["20240131", "20240131", "20240229"],
                "weight": [1.5, 2.5, 2.0],
            }
        )


def _feed(monkeypatch):
    feed = IndexConstituentsFeed("fake.json")
    monkeypatch.setattr(feed, "_client", lambda: _FakePro())
    return feed


def test_get_constituents_maps_to_canonical_columns(monkeypatch):
    feed = _feed(monkeypatch)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    assert list(out.columns) == ["date", "symbol", "weight"]
    assert out["symbol"].dtype == object
    assert str(out["date"].dtype).startswith("datetime64")


def test_get_constituents_sorted_by_date_symbol(monkeypatch):
    feed = _feed(monkeypatch)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    pairs = list(zip(out["date"], out["symbol"]))
    assert pairs == sorted(pairs)
    # 20240131 cross-section: both symbols present, ascending
    jan = out[out["date"] == pd.Timestamp("2024-01-31")]["symbol"].tolist()
    assert jan == ["000001.SZ", "000002.SZ"]


def test_get_constituents_pages_long_window(monkeypatch):
    # A >90-day window must be split into multiple index_weight calls and the
    # snapshots concatenated, so the tushare ~6000-row cap can't drop early dates.
    class _PagingPro:
        def __init__(self):
            self.calls = []

        def index_weight(self, index_code, start_date, end_date):  # noqa: ARG002
            self.calls.append((start_date, end_date))
            return pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": [start_date],  # one snapshot per window
                    "weight": [1.0],
                }
            )

    feed = IndexConstituentsFeed("fake.json")
    pro = _PagingPro()
    monkeypatch.setattr(feed, "_client", lambda: pro)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-06-30")
    assert len(pro.calls) >= 2  # window was paged
    assert out["date"].nunique() >= 2  # snapshots from multiple windows kept


def test_get_constituents_empty_result_is_schema_shaped(monkeypatch):
    feed = IndexConstituentsFeed("fake.json")

    class _Empty:
        def index_weight(self, **_kw):
            return pd.DataFrame()

    monkeypatch.setattr(feed, "_client", lambda: _Empty())
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    assert list(out.columns) == ["date", "symbol", "weight"]
    assert len(out) == 0
