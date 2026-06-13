"""P4-1: market-bars read-through cache (network-free, fake tushare client).

Pins the P4-1 acceptance:
  * full cache miss populates daily + adj_factor cache;
  * full cache hit makes ZERO endpoint calls;
  * a partial symbol/date gap fetches ONLY the missing range;
  * an empty endpoint return still records coverage (no needless refetch);
  * a duplicate upsert keeps one row per (symbol, date);
  * cached raw + adj_factor output equals the direct full-fetch output after
    front_adjust();
  * the cache + ledger contain no token / secret-file content.

No test hits the network or reads the real token: a fake ``pro`` client with
call counters drives both the cache path and the direct path.
"""

from __future__ import annotations

import json

import pandas as pd

from data.cache import CacheParquetStore, CoverageLedger, TushareCache
from data.cache.intervals import subtract_intervals
from data.clean.adjust import front_adjust
from data.clean.schema import validate_panel
from data.feed.tushare_feed import TushareFeed

FAKE_TOKEN = "FAKE_TUSHARE_TOKEN_do_not_leak_0123456789abcdef"
SECRET_PATH_MARKER = "/abs/path/to/.config.json"


# --------------------------------------------------------------------------- #
# Fake tushare client: deterministic daily/adj rows + per-endpoint call counts.
# --------------------------------------------------------------------------- #
class FakePro:
    """A fake ``pro`` whose ``daily``/``adj_factor`` count calls and return
    deterministic rows for the requested compact [start, end] interval."""

    def __init__(self):
        self.daily_calls = 0
        self.adj_calls = 0
        self.daily_ranges: list[tuple[str, str, str]] = []

    def _trading_days(self, start_compact, end_compact):
        s = pd.Timestamp(start_compact)
        e = pd.Timestamp(end_compact)
        return pd.bdate_range(s, e)

    def daily(self, ts_code, start_date, end_date, **_):
        self.daily_calls += 1
        self.daily_ranges.append((ts_code, start_date, end_date))
        days = self._trading_days(start_date, end_date)
        if len(days) == 0:
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "open", "high", "low",
                         "close", "vol", "amount"]
            )
        base = 10.0 + 0.1 * (hash(ts_code) % 7)
        rows = []
        for i, d in enumerate(days):
            px = base + i  # strictly rising raw close
            rows.append({
                "ts_code": ts_code,
                "trade_date": d.strftime("%Y%m%d"),
                "open": px - 0.2, "high": px + 0.3, "low": px - 0.4,
                "close": px, "vol": 1000.0 + i, "amount": 1.0e6 + i,
            })
        return pd.DataFrame(rows)

    def adj_factor(self, ts_code, start_date, end_date, **_):
        self.adj_calls += 1
        days = self._trading_days(start_date, end_date)
        if len(days) == 0:
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        # a mid-window split so front_adjust is non-trivial.
        return pd.DataFrame({
            "ts_code": [ts_code] * len(days),
            "trade_date": [d.strftime("%Y%m%d") for d in days],
            "adj_factor": [1.0 if i < len(days) // 2 else 2.0
                           for i in range(len(days))],
        })


class FakeProEmpty:
    """A fake ``pro`` that always returns empty frames (records coverage)."""

    def __init__(self):
        self.daily_calls = 0
        self.adj_calls = 0

    def daily(self, ts_code, start_date, end_date, **_):
        self.daily_calls += 1
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "open", "high", "low",
                     "close", "vol", "amount"]
        )

    def adj_factor(self, ts_code, start_date, end_date, **_):
        self.adj_calls += 1
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])


def _write_fake_config(tmp_path):
    cfg = {"tushare": {"token": FAKE_TOKEN}, "secret_path": SECRET_PATH_MARKER}
    path = tmp_path / "fake_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _feed_with_cache(tmp_path, monkeypatch, pro, *, force_refresh=(), today=None):
    """A TushareFeed wired to a cache rooted in tmp, with ``pro`` injected.

    ``today`` is fixed far in the future-relative-past so refresh_recent_days
    never triggers (the windows here are historical)."""
    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", lambda token=None: pro)
    root = str(tmp_path / "cache")
    cache = TushareCache(
        CacheParquetStore(root),
        CoverageLedger(root),
        refresh_recent_days=14,
        force_refresh=force_refresh,
        today=pd.Timestamp(today) if today else pd.Timestamp("2024-12-31"),
    )
    feed = TushareFeed(secret_file=str(_write_fake_config(tmp_path)), cache=cache)
    return feed, root


# --------------------------------------------------------------------------- #
# interval algebra (the gap planner's core)
# --------------------------------------------------------------------------- #
def test_subtract_intervals_full_partial_none():
    s, e = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")
    # nothing covered -> one gap == the whole request
    assert subtract_intervals(s, e, []) == [(s, e)]
    # fully covered -> no gap
    assert subtract_intervals(s, e, [(s, e)]) == []
    # left covered -> right gap only
    gaps = subtract_intervals(s, e, [(pd.Timestamp("2024-01-01"),
                                      pd.Timestamp("2024-01-15"))])
    assert gaps == [(pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-31"))]


# --------------------------------------------------------------------------- #
# full miss / full hit
# --------------------------------------------------------------------------- #
def test_full_miss_then_full_hit(tmp_path, monkeypatch):
    pro = FakePro()
    feed, root = _feed_with_cache(tmp_path, monkeypatch, pro)
    symbols = ["000001.SZ", "000002.SZ"]

    panel1 = feed.get_bars(symbols, "2024-01-01", "2024-01-31")
    validate_panel(panel1)
    assert pro.daily_calls == 2 and pro.adj_calls == 2  # one per symbol, full miss
    # cache files exist
    assert (CacheParquetStore(root).symbol_path("market_daily", "000001.SZ")).exists()
    assert (CacheParquetStore(root).symbol_path("adj_factor", "000001.SZ")).exists()

    # second identical run: ZERO endpoint calls, identical panel.
    panel2 = feed.get_bars(symbols, "2024-01-01", "2024-01-31")
    assert pro.daily_calls == 2 and pro.adj_calls == 2  # unchanged
    pd.testing.assert_frame_equal(panel1, panel2)


# --------------------------------------------------------------------------- #
# partial gap
# --------------------------------------------------------------------------- #
def test_partial_gap_fetches_only_missing_range(tmp_path, monkeypatch):
    pro = FakePro()
    feed, _ = _feed_with_cache(tmp_path, monkeypatch, pro)
    feed.get_bars(["000001.SZ"], "2024-01-01", "2024-01-15")
    assert pro.daily_calls == 1
    pro.daily_ranges.clear()

    # extend the window forward: only the new tail [01-16, 01-31] is fetched.
    feed.get_bars(["000001.SZ"], "2024-01-01", "2024-01-31")
    assert pro.daily_calls == 2  # exactly one more fetch
    sym, s, e = pro.daily_ranges[-1]
    assert s == "20240116" and e == "20240131"


# --------------------------------------------------------------------------- #
# empty endpoint return still records coverage
# --------------------------------------------------------------------------- #
def test_empty_return_records_coverage_no_refetch(tmp_path, monkeypatch):
    pro = FakeProEmpty()
    feed, root = _feed_with_cache(tmp_path, monkeypatch, pro)
    panel = feed.get_bars(["000001.SZ"], "2024-01-01", "2024-01-31")
    assert panel.empty  # no rows, but no error
    assert pro.daily_calls == 1

    ledger = CoverageLedger(root).read()
    daily_rows = ledger[ledger["endpoint"] == "market_daily"]
    assert len(daily_rows) == 1
    assert daily_rows.iloc[0]["status"] == "empty"
    assert daily_rows.iloc[0]["row_count"] == 0

    # a second run sees the empty range as covered -> no extra daily call.
    feed.get_bars(["000001.SZ"], "2024-01-01", "2024-01-31")
    assert pro.daily_calls == 1


# --------------------------------------------------------------------------- #
# duplicate upsert keeps one row per (symbol, date)
# --------------------------------------------------------------------------- #
def test_duplicate_upsert_keeps_one_row_per_key(tmp_path):
    store = CacheParquetStore(str(tmp_path / "cache"))
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 11.0],
    })
    store.upsert_symbol("market_daily", "000001.SZ", rows, ["date", "symbol"])
    # re-upsert the SAME keys with new values -> latest wins, still 2 rows.
    rows2 = rows.copy()
    rows2["close"] = [99.0, 98.0]
    n = store.upsert_symbol("market_daily", "000001.SZ", rows2, ["date", "symbol"])
    assert n == 2
    cached = store.read_symbol("market_daily", "000001.SZ")
    assert len(cached) == 2
    jan1 = cached.loc[cached["date"] == pd.Timestamp("2024-01-01"), "close"].iloc[0]
    assert jan1 == 99.0  # the re-fetched value replaced the old one


# --------------------------------------------------------------------------- #
# qfq equivalence: cache path == direct path, after front_adjust
# --------------------------------------------------------------------------- #
def test_cached_equals_direct_after_front_adjust(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ"]
    start, end = "2024-01-01", "2024-01-31"

    # direct path (no cache)
    pro_direct = FakePro()
    import tushare as ts
    monkeypatch.setattr(ts, "pro_api", lambda token=None: pro_direct)
    direct_feed = TushareFeed(secret_file=str(_write_fake_config(tmp_path)))
    direct_panel = direct_feed.get_bars(symbols, start, end)
    direct_qfq = front_adjust(direct_panel)

    # cache path (fresh fake client, fresh cache)
    pro_cached = FakePro()
    cache_feed, _ = _feed_with_cache(tmp_path, monkeypatch, pro_cached)
    cached_panel = cache_feed.get_bars(symbols, start, end)
    cached_qfq = front_adjust(cached_panel)

    pd.testing.assert_frame_equal(direct_panel, cached_panel)
    pd.testing.assert_frame_equal(direct_qfq, cached_qfq)


# --------------------------------------------------------------------------- #
# no secret leaks into cache files
# --------------------------------------------------------------------------- #
def test_cache_files_contain_no_secret(tmp_path, monkeypatch):
    pro = FakePro()
    feed, root = _feed_with_cache(tmp_path, monkeypatch, pro)
    feed.get_bars(["000001.SZ"], "2024-01-01", "2024-01-31")

    from pathlib import Path

    blobs = b""
    for p in Path(root).rglob("*.parquet"):
        blobs += p.read_bytes()
    assert FAKE_TOKEN.encode() not in blobs
    assert SECRET_PATH_MARKER.encode() not in blobs
    # the ledger columns carry no token/secret-path fields
    ledger = CoverageLedger(root).read()
    assert "token" not in [c.lower() for c in ledger.columns]
    for col in ledger.columns:
        joined = "".join(str(v) for v in ledger[col].tolist())
        assert FAKE_TOKEN not in joined and SECRET_PATH_MARKER not in joined
