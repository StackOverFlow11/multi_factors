"""TushareIntradayCache (I2) tests — fake SDK, network-free, no token.

Covers cold/warm/partial/empty/failed read-through, idempotent upsert, the
1min-only raw guard, ledger/store schema (no secret), and cached==direct
equality after normalization.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.cache.intraday_cache import ENDPOINT, TushareIntradayCache
from data.cache.intraday_coverage import (
    INTRADAY_LEDGER_COLUMNS,
    IntradayCoverageLedger,
)
from data.cache.intraday_parquet_store import STORED_COLUMNS, IntradayParquetStore
from data.clean.intraday_schema import validate_intraday_bars
from data.feed.tushare_intraday import TushareIntradayFeed

_FIXED_CLOCK = lambda: pd.Timestamp("2026-06-13 10:00:00")  # noqa: E731


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FetchRecorder:
    """A window-aware fake fetch callable; records the (symbol, start, end) calls."""

    def __init__(self, catalog: dict[str, list[tuple[str, float]]]):
        self.catalog = catalog
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, symbol, start_dt, end_dt):
        self.calls.append((symbol, start_dt, end_dt))
        s, e = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
        rows = [
            (tt, close)
            for tt, close in self.catalog.get(symbol, [])
            if s <= pd.Timestamp(tt) <= e
        ]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "ts_code": [symbol] * len(rows),
                "trade_time": [r[0] for r in rows],
                "open": [r[1] for r in rows],
                "high": [r[1] for r in rows],
                "low": [r[1] for r in rows],
                "close": [r[1] for r in rows],
                "vol": [100.0] * len(rows),
                "amount": [1000.0] * len(rows),
            }
        )


class _FakePro:
    """Fake tushare client exposing window-aware ``stk_mins``."""

    def __init__(self, catalog):
        self._rec = _FetchRecorder(catalog)

    def stk_mins(self, ts_code, freq, start_date, end_date):  # noqa: ARG002
        return self._rec(ts_code, start_date, end_date)


def _cache(tmp_path, **kw):
    root = str(tmp_path / "cache")
    store = IntradayParquetStore(root)
    ledger = IntradayCoverageLedger(root)
    return TushareIntradayCache(store, ledger, clock=_FIXED_CLOCK, **kw), store, ledger


_CATALOG = {
    "000001.SZ": [
        ("2024-01-02 09:31:00", 11.0),
        ("2024-01-02 09:32:00", 12.0),
        ("2024-01-03 09:31:00", 13.0),
        ("2024-01-04 09:31:00", 14.0),
    ]
}


# --------------------------------------------------------------------------- #
# cold / warm / partial
# --------------------------------------------------------------------------- #
def test_cold_miss_writes_raw_1min(tmp_path):
    cache, store, ledger = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)
    out = cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    assert not out.empty
    assert list(out.columns) == [
        "time", "symbol", "open", "high", "low", "close",
        "volume", "amount", "source_trade_time",
    ]
    # raw bars persisted
    stored = store.read_range(ENDPOINT, "000001.SZ", "1min",
                              "2024-01-02 00:00:00", "2024-01-02 23:59:59")
    assert list(stored.columns) == STORED_COLUMNS
    assert len(stored) == 2  # 09:31 + 09:32
    # coverage recorded ok
    led = ledger.read()
    assert (led["status"] == "ok").any()
    assert fetch.calls  # the API was hit on a cold miss


def test_warm_identical_zero_sdk_calls(tmp_path):
    cache, _, _ = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)
    cold = cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    fetch.calls.clear()
    warm = cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    assert fetch.calls == []  # zero SDK calls on a fully-covered warm request
    pd.testing.assert_frame_equal(cold, warm)


def test_partial_gap_fetches_only_uncovered_days(tmp_path):
    cache, _, _ = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)
    cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:00:00", "2024-01-03 15:00:00", fetch
    )
    fetch.calls.clear()
    cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:00:00", "2024-01-04 15:00:00", fetch
    )
    # only 2024-01-04 was uncovered -> a single window starting that day
    assert len(fetch.calls) == 1
    _, start_dt, end_dt = fetch.calls[0]
    assert start_dt == "2024-01-04 00:00:00"
    assert end_dt == "2024-01-04 23:59:59"


def test_empty_records_coverage_and_avoids_refetch(tmp_path):
    cache, _, ledger = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)  # 999999.SZ absent -> empty
    out1 = cache.stk_mins_1min(
        ["999999.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    assert out1.empty
    led = ledger.read()
    assert (led["status"] == "empty").any()
    fetch.calls.clear()
    out2 = cache.stk_mins_1min(
        ["999999.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    assert fetch.calls == []  # empty coverage prevents a needless refetch
    assert out2.empty


def test_failed_fetch_records_no_coverage_and_retries(tmp_path):
    cache, _, ledger = _cache(tmp_path)

    def raising_fetch(symbol, start_dt, end_dt):
        raise ConnectionError("transient")

    with pytest.raises(ConnectionError):
        cache.stk_mins_1min(
            ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00",
            raising_fetch,
        )
    assert ledger.read().empty  # a failed fetch is NOT coverage
    # a later run with a working fetch still sees the gap and retries
    good = _FetchRecorder(_CATALOG)
    out = cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", good
    )
    assert not out.empty
    assert good.calls  # the gap was retried


# --------------------------------------------------------------------------- #
# upsert idempotency / guard / schema
# --------------------------------------------------------------------------- #
def test_duplicate_upsert_keeps_one_row_per_key(tmp_path):
    # direct store idempotency
    store = IntradayParquetStore(str(tmp_path / "c"))
    rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "bar_end": [pd.Timestamp("2024-01-02 09:31:00")],
            "source_trade_time": ["2024-01-02 09:31:00"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [11.0],
            "volume": [100.0], "amount": [1000.0], "freq": ["1min"],
        }
    )
    store.upsert(ENDPOINT, "000001.SZ", "1min", rows, ["symbol", "freq", "bar_end"])
    store.upsert(ENDPOINT, "000001.SZ", "1min", rows, ["symbol", "freq", "bar_end"])
    got = store.read_range(ENDPOINT, "000001.SZ", "1min",
                           "2024-01-02 00:00:00", "2024-01-02 23:59:59")
    assert len(got) == 1  # one row per (symbol, freq, bar_end)

    # also via cache force_refresh (re-fetches but does not double rows)
    cache, _, _ = _cache(tmp_path, force_refresh=True)
    fetch = _FetchRecorder(_CATALOG)
    cache.stk_mins_1min(["000001.SZ"], "2024-01-02 09:30:00",
                        "2024-01-02 14:55:00", fetch)
    cache.stk_mins_1min(["000001.SZ"], "2024-01-02 09:30:00",
                        "2024-01-02 14:55:00", fetch)
    cstore = cache._store
    got2 = cstore.read_range(ENDPOINT, "000001.SZ", "1min",
                             "2024-01-02 00:00:00", "2024-01-02 23:59:59")
    assert len(got2) == 2  # 09:31 + 09:32, no duplicates despite two fetches


def test_non_1min_raw_request_cannot_hit_fetch(tmp_path):
    cache, _, _ = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)
    with pytest.raises(ValueError, match="freq"):
        cache.stk_mins_1min(
            ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00",
            fetch, freq="5min",
        )
    assert fetch.calls == []  # rejected before any fetch


def test_ledger_and_store_carry_no_secret(tmp_path):
    cache, store, ledger = _cache(tmp_path)
    fetch = _FetchRecorder(_CATALOG)
    cache.stk_mins_1min(
        ["000001.SZ"], "2024-01-02 09:30:00", "2024-01-02 14:55:00", fetch
    )
    # ledger holds only endpoint metadata; no token-bearing column
    assert list(ledger.read().columns) == INTRADAY_LEDGER_COLUMNS
    assert not any("token" in c for c in INTRADAY_LEDGER_COLUMNS)
    # stored bars hold only raw market columns
    stored = store.read_range(ENDPOINT, "000001.SZ", "1min",
                              "2024-01-02 00:00:00", "2024-01-02 23:59:59")
    assert list(stored.columns) == STORED_COLUMNS
    assert not any("token" in c for c in STORED_COLUMNS)


# --------------------------------------------------------------------------- #
# cached path == direct path
# --------------------------------------------------------------------------- #
def test_cached_output_equals_direct_feed_output(tmp_path, monkeypatch):
    pro = _FakePro(_CATALOG)
    start, end = "2024-01-02 09:30:00", "2024-01-02 09:35:00"

    direct = TushareIntradayFeed("unused.json")
    monkeypatch.setattr(direct, "_client", lambda: pro)
    out_direct = direct.get_minutes(["000001.SZ"], start, end)

    cache, _, _ = _cache(tmp_path)
    cached = TushareIntradayFeed("unused.json", cache=cache)
    monkeypatch.setattr(cached, "_client", lambda: pro)
    out_cached = cached.get_minutes(["000001.SZ"], start, end)

    validate_intraday_bars(out_cached)
    pd.testing.assert_frame_equal(out_direct, out_cached)


def test_cached_feed_empty_window_is_schema_shaped(tmp_path, monkeypatch):
    pro = _FakePro(_CATALOG)
    cache, _, _ = _cache(tmp_path)
    feed = TushareIntradayFeed("unused.json", cache=cache)
    monkeypatch.setattr(feed, "_client", lambda: pro)
    out = feed.get_minutes(["999999.SZ"], "2024-01-02 09:30:00", "2024-01-02 09:35:00")
    assert len(out) == 0
    assert list(out.index.names) == ["time", "symbol"]
    validate_intraday_bars(out)
