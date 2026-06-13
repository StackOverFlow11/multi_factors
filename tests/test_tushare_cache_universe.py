"""P4-2: universe + tradability read-through cache (network-free, fake clients).

Pins the P4-2 acceptance for the five new endpoints (index_weight, suspend_d,
namechange, stk_limit, stock_basic):
  * cold miss populates the cache; an identical warm rerun makes ZERO calls;
  * cached feed output EQUALS the direct (uncached) feed output, so PIT as-of
    membership / suspension / ST intervals / raw price limits / list_date are
    all unchanged;
  * an empty endpoint return still records coverage (no needless refetch);
  * a FAILED fetch records NO coverage (a later run retries);
  * a duplicate upsert keeps one row per the endpoint's natural key;
  * the cache-disabled path is byte-for-byte the direct path;
  * the run-log stats line names the new endpoints;
  * cache files + ledger contain no token / secret-file content.

No test hits the network or reads a real token: fake ``pro`` clients with call
counters drive both the cache path and the direct path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.cache import CacheParquetStore, CoverageLedger, TushareCache
from data.cache.tushare_cache import (
    INDEX_WEIGHT,
    NAMECHANGE,
    STK_LIMIT,
    STOCK_BASIC,
    SUSPEND_D,
)
from data.feed.index_feed import IndexConstituentsFeed
from data.feed.tushare_covariates import TushareCovariatesFeed
from data.feed.tushare_flags import TushareFlagsFeed

FAKE_TOKEN = "FAKE_TUSHARE_TOKEN_do_not_leak_0123456789abcdef"
SECRET_PATH_MARKER = "/abs/path/to/.config.json"


# --------------------------------------------------------------------------- #
# Fake tushare clients (deterministic rows + per-endpoint call counters).
# --------------------------------------------------------------------------- #
class _FakeIndexPro:
    """``index_weight`` with API-like window filtering + a call counter."""

    _ROWS = [
        ("20240131", "000002.SZ", 1.5),
        ("20240131", "000001.SZ", 2.5),
        ("20240229", "000001.SZ", 2.0),
    ]

    def __init__(self):
        self.calls = 0

    def index_weight(self, index_code, start_date, end_date, **_):
        self.calls += 1
        s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
        keep = [r for r in self._ROWS if s <= pd.Timestamp(r[0]) <= e]
        if not keep:
            return pd.DataFrame(
                columns=["index_code", "con_code", "trade_date", "weight"]
            )
        return pd.DataFrame(
            {
                "index_code": [index_code] * len(keep),
                "con_code": [r[1] for r in keep],
                "trade_date": [r[0] for r in keep],
                "weight": [r[2] for r in keep],
            }
        )


class _FakeFlagsPro:
    """``suspend_d`` / ``namechange`` / ``stk_limit`` with call counters."""

    def __init__(self):
        self.suspend_calls = 0
        self.namechange_calls = 0
        self.stk_limit_calls = 0

    def suspend_d(self, ts_code, start_date, end_date, suspend_type, **_):
        self.suspend_calls += 1
        if ts_code == "000001.SZ":
            return pd.DataFrame(
                {"ts_code": ["000001.SZ"], "trade_date": ["20240110"],
                 "suspend_type": ["S"]}
            )
        return pd.DataFrame()

    def namechange(self, ts_code, **_):
        self.namechange_calls += 1
        return pd.DataFrame(
            {
                "ts_code": [ts_code, ts_code, ts_code],
                "name": ["*ST X", "*ST X", "X"],
                "start_date": ["20240301", "20240301", "20230101"],
                "end_date": [None, None, "20240229"],
            }
        )

    def stk_limit(self, ts_code, start_date, end_date, **_):
        self.stk_limit_calls += 1
        return pd.DataFrame(
            {
                "trade_date": ["20240110", "20240111"],
                "ts_code": [ts_code, ts_code],
                "up_limit": [11.0, 11.5],
                "down_limit": [9.0, 9.2],
            }
        )


class _FakeBasicPro:
    """``stock_basic`` (global snapshot) with a call counter."""

    def __init__(self):
        self.calls = 0

    def stock_basic(self, **_):
        self.calls += 1
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000300.SH"],
                "list_date": ["19910403", "19910101", "20050408"],
            }
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_cache(root, *, today="2024-12-31", force_refresh=()):
    """A cache with a frozen clock so dimension freshness is deterministic."""
    ts = pd.Timestamp(today)
    return TushareCache(
        CacheParquetStore(root),
        CoverageLedger(root),
        refresh_recent_days=14,
        refresh_dimension_days=30,
        force_refresh=force_refresh,
        today=ts,
        clock=lambda: ts,
    )


def _index_feed(monkeypatch, pro, cache=None):
    feed = IndexConstituentsFeed("x.json", cache=cache)
    monkeypatch.setattr(feed, "_client", lambda: pro)
    return feed


def _flags_feed(monkeypatch, pro, cache=None):
    feed = TushareFlagsFeed("x.json", cache=cache)
    monkeypatch.setattr(feed, "_client", lambda: pro)
    return feed


def _cov_feed(monkeypatch, pro, cache=None):
    feed = TushareCovariatesFeed("x.json", cache=cache)
    monkeypatch.setattr(feed, "_client", lambda: pro)
    return feed


# --------------------------------------------------------------------------- #
# index_weight: cold/warm + equals direct + PIT preserved
# --------------------------------------------------------------------------- #
def test_index_weight_cold_warm_zero_calls(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    pro = _FakeIndexPro()
    feed = _index_feed(monkeypatch, pro, _make_cache(root))
    out1 = feed.get_constituents("000300.SH", "2024-01-01", "2024-03-31")
    assert pro.calls > 0  # cold: paged window(s) fetched
    assert out1["date"].nunique() == 2

    # warm rerun on a FRESH cache over the same root -> zero index_weight calls.
    pro2 = _FakeIndexPro()
    warm_cache = _make_cache(root)
    warm = _index_feed(monkeypatch, pro2, warm_cache)
    out2 = warm.get_constituents("000300.SH", "2024-01-01", "2024-03-31")
    assert pro2.calls == 0
    pd.testing.assert_frame_equal(out1, out2)
    assert warm_cache.stats()[INDEX_WEIGHT] == 0


def test_index_weight_cached_equals_direct(tmp_path, monkeypatch):
    direct = _index_feed(monkeypatch, _FakeIndexPro())
    direct_out = direct.get_constituents("000300.SH", "2024-01-01", "2024-03-31")

    root = str(tmp_path / "cache")
    cached = _index_feed(monkeypatch, _FakeIndexPro(), _make_cache(root))
    cached_out = cached.get_constituents("000300.SH", "2024-01-01", "2024-03-31")

    pd.testing.assert_frame_equal(direct_out, cached_out)
    # PIT preserved: every stored snapshot date kept (latest-as-of stays the
    # universe's job); the 2024-01-31 cross-section is both names, ascending.
    jan = cached_out[cached_out["date"] == pd.Timestamp("2024-01-31")]["symbol"].tolist()
    assert jan == ["000001.SZ", "000002.SZ"]


# --------------------------------------------------------------------------- #
# suspend_d: cold/warm + equals direct
# --------------------------------------------------------------------------- #
def test_suspend_d_cold_warm_zero_calls_equals_direct(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ"]
    direct = _flags_feed(monkeypatch, _FakeFlagsPro())
    direct_out = direct.suspended(symbols, "2024-01-01", "2024-01-31")

    root = str(tmp_path / "cache")
    pro = _FakeFlagsPro()
    cached = _flags_feed(monkeypatch, pro, _make_cache(root))
    cold_out = cached.suspended(symbols, "2024-01-01", "2024-01-31")
    assert pro.suspend_calls == 2  # one per symbol (full miss)
    assert cold_out == direct_out  # set equality

    pro2 = _FakeFlagsPro()
    warm_cache = _make_cache(root)
    warm = _flags_feed(monkeypatch, pro2, warm_cache)
    warm_out = warm.suspended(symbols, "2024-01-01", "2024-01-31")
    assert pro2.suspend_calls == 0  # fully covered
    assert warm_out == direct_out
    assert warm_cache.stats()[SUSPEND_D] == 0


# --------------------------------------------------------------------------- #
# namechange / ST intervals: cold/warm + equals direct
# --------------------------------------------------------------------------- #
def test_namechange_cold_warm_zero_calls_equals_direct(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ"]
    direct = _flags_feed(monkeypatch, _FakeFlagsPro())
    direct_out = direct.st_intervals(symbols)

    root = str(tmp_path / "cache")
    pro = _FakeFlagsPro()
    cached = _flags_feed(monkeypatch, pro, _make_cache(root))
    cold_out = cached.st_intervals(symbols)
    assert pro.namechange_calls == 2

    pro2 = _FakeFlagsPro()
    warm_cache = _make_cache(root)
    warm = _flags_feed(monkeypatch, pro2, warm_cache)
    warm_out = warm.st_intervals(symbols)
    assert pro2.namechange_calls == 0  # dimension snapshot fresh -> no refetch
    assert warm_cache.stats()[NAMECHANGE] == 0

    # interval SETS equal across direct / cold / warm (dedupe + ST flag preserved)
    assert set(direct_out) == set(cold_out) == set(warm_out)
    for sym in symbols:
        assert set(direct_out[sym]) == set(cold_out[sym]) == set(warm_out[sym])
        assert any(is_st for _, _, is_st in cold_out[sym])
        assert any(not is_st for _, _, is_st in cold_out[sym])
        # the open (active) interval keeps a None end exactly like the direct path
        assert any(end is None for _, end, _ in cold_out[sym])


# --------------------------------------------------------------------------- #
# stk_limit: cold/warm + equals direct (raw price terms)
# --------------------------------------------------------------------------- #
def test_stk_limit_cold_warm_zero_calls_equals_direct(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ"]
    direct = _flags_feed(monkeypatch, _FakeFlagsPro())
    direct_out = direct.limits(symbols, "2024-01-01", "2024-01-31")

    root = str(tmp_path / "cache")
    pro = _FakeFlagsPro()
    cached = _flags_feed(monkeypatch, pro, _make_cache(root))
    cold_out = cached.limits(symbols, "2024-01-01", "2024-01-31")
    assert pro.stk_limit_calls == 2
    pd.testing.assert_frame_equal(direct_out, cold_out)
    # raw price terms preserved (no qfq scaling in the cache)
    assert cold_out.iloc[0]["up_limit"] == 11.0

    pro2 = _FakeFlagsPro()
    warm_cache = _make_cache(root)
    warm = _flags_feed(monkeypatch, pro2, warm_cache)
    warm_out = warm.limits(symbols, "2024-01-01", "2024-01-31")
    assert pro2.stk_limit_calls == 0
    pd.testing.assert_frame_equal(direct_out, warm_out)
    assert warm_cache.stats()[STK_LIMIT] == 0


# --------------------------------------------------------------------------- #
# stock_basic / listing_dates: cold/warm + equals direct
# --------------------------------------------------------------------------- #
def test_stock_basic_cold_warm_zero_calls_equals_direct(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ"]
    direct = _cov_feed(monkeypatch, _FakeBasicPro())
    direct_out = direct.listing_dates(symbols)

    root = str(tmp_path / "cache")
    pro = _FakeBasicPro()
    cached = _cov_feed(monkeypatch, pro, _make_cache(root))
    cold_out = cached.listing_dates(symbols)
    assert pro.calls == 1  # ONE global snapshot, not per-symbol
    assert cold_out == direct_out
    assert cold_out["000001.SZ"] == pd.Timestamp("1991-04-03")

    pro2 = _FakeBasicPro()
    warm_cache = _make_cache(root)
    warm = _cov_feed(monkeypatch, pro2, warm_cache)
    warm_out = warm.listing_dates(symbols)
    assert pro2.calls == 0  # dimension snapshot fresh -> no refetch
    assert warm_out == direct_out
    assert warm_cache.stats()[STOCK_BASIC] == 0


# --------------------------------------------------------------------------- #
# empty endpoint return still records coverage (no needless refetch)
# --------------------------------------------------------------------------- #
def test_empty_return_records_coverage_no_refetch(tmp_path):
    root = str(tmp_path / "cache")
    cache = _make_cache(root)
    calls = {"n": 0}

    def fetch(symbol, s, e):
        calls["n"] += 1
        return pd.DataFrame()  # always empty

    out = cache.suspend_d(["000999.SZ"], "2024-01-01", "2024-01-31", fetch)
    assert out.empty and calls["n"] == 1

    ledger = CoverageLedger(root).read()
    rows = ledger[ledger["endpoint"] == SUSPEND_D]
    assert len(rows) == 1 and rows.iloc[0]["status"] == "empty"

    # a second run sees the empty range as covered -> no extra call.
    cache2 = _make_cache(root)
    cache2.suspend_d(["000999.SZ"], "2024-01-01", "2024-01-31", fetch)
    assert calls["n"] == 1


def test_stock_basic_empty_snapshot_records_coverage(tmp_path):
    root = str(tmp_path / "cache")
    cache = _make_cache(root)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return pd.DataFrame()

    assert cache.stock_basic(fetch).empty and calls["n"] == 1
    # fresh instance, same root: snapshot recorded (empty) -> no refetch.
    _make_cache(root).stock_basic(fetch)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# failed fetch does NOT record coverage (a later run retries)
# --------------------------------------------------------------------------- #
def test_failed_fetch_records_no_coverage(tmp_path):
    root = str(tmp_path / "cache")
    cache = _make_cache(root)

    def boom(symbol, s, e):
        raise RuntimeError("transient network error")

    with pytest.raises(RuntimeError):
        cache.suspend_d(["000001.SZ"], "2024-01-01", "2024-01-31", boom)

    # nothing covered -> a later good fetch retries and populates.
    assert CoverageLedger(root).read().empty
    cache2 = _make_cache(root)
    good_calls = {"n": 0}

    def good(symbol, s, e):
        good_calls["n"] += 1
        return pd.DataFrame(
            {"ts_code": [symbol], "trade_date": ["20240110"], "suspend_type": ["S"]}
        )

    out = cache2.suspend_d(["000001.SZ"], "2024-01-01", "2024-01-31", good)
    assert good_calls["n"] == 1 and len(out) == 1


def test_failed_snapshot_fetch_records_no_coverage(tmp_path):
    root = str(tmp_path / "cache")
    cache = _make_cache(root)

    def boom():
        raise RuntimeError("transient network error")

    with pytest.raises(RuntimeError):
        cache.stock_basic(boom)
    assert CoverageLedger(root).read().empty


# --------------------------------------------------------------------------- #
# duplicate upsert keeps one row per natural key (index_weight / suspend_d)
# --------------------------------------------------------------------------- #
def test_duplicate_upsert_one_row_per_natural_key(tmp_path):
    store = CacheParquetStore(str(tmp_path / "cache"))
    rows = pd.DataFrame(
        {
            "index_code": ["000300.SH", "000300.SH"],
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "symbol": ["000001.SZ", "000002.SZ"],
            "weight": [2.5, 1.5],
        }
    )
    store.upsert_symbol(INDEX_WEIGHT, "000300.SH", rows, ["date", "symbol"])
    rows2 = rows.copy()
    rows2["weight"] = [9.9, 8.8]  # re-fetch same keys, new weights
    n = store.upsert_symbol(INDEX_WEIGHT, "000300.SH", rows2, ["date", "symbol"])
    assert n == 2
    cached = store.read_symbol(INDEX_WEIGHT, "000300.SH")
    w = cached.loc[cached["symbol"] == "000001.SZ", "weight"].iloc[0]
    assert w == 9.9  # latest wins


def test_suspend_dedup_keeps_one_row_per_key(tmp_path):
    store = CacheParquetStore(str(tmp_path / "cache"))
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "symbol": ["000001.SZ", "000001.SZ"],
            "suspend_type": ["S", "S"],
        }
    )
    store.upsert_symbol(SUSPEND_D, "000001.SZ", rows, ["date", "symbol", "suspend_type"])
    n = store.upsert_symbol(
        SUSPEND_D, "000001.SZ", rows, ["date", "symbol", "suspend_type"]
    )
    assert n == 1  # both upserts collapse to one natural-key row


# --------------------------------------------------------------------------- #
# cache-disabled path is the direct path (no cache files created)
# --------------------------------------------------------------------------- #
def test_cache_disabled_path_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    direct = _flags_feed(monkeypatch, _FakeFlagsPro(), cache=None)
    out = direct.limits(["000001.SZ"], "2024-01-01", "2024-01-31")
    assert list(out.columns) == ["date", "symbol", "up_limit", "down_limit"]
    assert not root.exists()  # cache=None never touches the cache tree


# --------------------------------------------------------------------------- #
# force_refresh re-pulls a dimension snapshot even when fresh
# --------------------------------------------------------------------------- #
def test_force_refresh_repulls_dimension(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    pro = _FakeBasicPro()
    cached = _cov_feed(monkeypatch, pro, _make_cache(root))
    cached.listing_dates(["000001.SZ"])
    assert pro.calls == 1

    pro2 = _FakeBasicPro()
    forced = _cov_feed(
        monkeypatch, pro2, _make_cache(root, force_refresh=(STOCK_BASIC,))
    )
    forced.listing_dates(["000001.SZ"])
    assert pro2.calls == 1  # force_refresh ignores freshness


# --------------------------------------------------------------------------- #
# run-log stats line names the new endpoints
# --------------------------------------------------------------------------- #
def test_cache_stats_line_includes_new_endpoints():
    from qt.pipeline import _format_cache_stats

    line = _format_cache_stats(
        {"market_daily": 1, "index_weight": 2, "suspend_d": 3,
         "namechange": 4, "stk_limit": 5, "stock_basic": 6}
    )
    assert line.startswith("data cache: market_daily_gap_fetches=1")
    for token in (
        "index_weight_gap_fetches=2",
        "suspend_d_gap_fetches=3",
        "namechange_gap_fetches=4",
        "stk_limit_gap_fetches=5",
        "stock_basic_gap_fetches=6",
    ):
        assert token in line


def test_log_run_cache_stats_emits_full_line(caplog):
    import logging

    from qt.pipeline import _log_run_cache_stats

    class _Cache:
        def stats(self):
            return {"market_daily": 0, "index_weight": 0, "suspend_d": 0,
                    "namechange": 0, "stk_limit": 0, "stock_basic": 0}

    logger = logging.getLogger("qt.test_p42_cache_line")
    with caplog.at_level(logging.INFO, logger="qt.test_p42_cache_line"):
        _log_run_cache_stats(_Cache(), logger)
        _log_run_cache_stats(None, logger)  # disabled -> no line
    assert "stock_basic_gap_fetches=0" in caplog.text
    assert caplog.text.count("data cache:") == 1


# --------------------------------------------------------------------------- #
# no token / secret-file content in cache files or ledger
# --------------------------------------------------------------------------- #
def test_cache_files_and_ledger_contain_no_secret(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    cache = _make_cache(root)

    # exercise every new endpoint so all parquet partitions exist.
    idx = _index_feed(monkeypatch, _FakeIndexPro(), cache)
    idx.get_constituents("000300.SH", "2024-01-01", "2024-03-31")
    flags = _flags_feed(monkeypatch, _FakeFlagsPro(), cache)
    flags.suspended(["000001.SZ"], "2024-01-01", "2024-01-31")
    flags.st_intervals(["000001.SZ"])
    flags.limits(["000001.SZ"], "2024-01-01", "2024-01-31")
    cov = _cov_feed(monkeypatch, _FakeBasicPro(), cache)
    cov.listing_dates(["000001.SZ"])

    from pathlib import Path

    blobs = b""
    for p in Path(root).rglob("*.parquet"):
        blobs += p.read_bytes()
    assert FAKE_TOKEN.encode() not in blobs
    assert SECRET_PATH_MARKER.encode() not in blobs

    ledger = CoverageLedger(root).read()
    assert "token" not in [c.lower() for c in ledger.columns]
    for col in ledger.columns:
        joined = "".join(str(v) for v in ledger[col].tolist())
        assert FAKE_TOKEN not in joined and SECRET_PATH_MARKER not in joined
