"""D4 coverage-ledger scaling: record_many batch API + in-process lookup cache.

Network-free, synthetic ledger rows. Asserts the batch path matches repeated
single records (schema/order/values), coverage semantics are unchanged
(ok/empty count; failed/not_ready do not), the in-process cache does not re-read
the full parquet for repeated identical lookups, an out-of-instance write is not
served stale, and the not-ready split writes its 1-2 rows in one batch.
"""

from __future__ import annotations

import pandas as pd

from data.cache.coverage import LEDGER_COLUMNS, CoverageLedger
from data.cache.intraday_coverage import INTRADAY_LEDGER_COLUMNS, IntradayCoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache

_TS = pd.Timestamp("2026-06-13 21:00:00")


# --------------------------------------------------------------------------- #
# daily CoverageLedger
# --------------------------------------------------------------------------- #
def _daily_row(key, start, end, status, fetched_at=_TS, **over):
    row = {
        "endpoint": "market_daily", "key_type": "symbol", "key": key,
        "start_date": start, "end_date": end, "fields_hash": "h",
        "row_count": 1 if status in ("ok",) else 0, "status": status,
        "fetched_at": fetched_at, "source_version": None,
    }
    row.update(over)
    return row


def test_record_many_equals_repeated_record(tmp_path):
    rows = [
        _daily_row("000001.SZ", "2024-01-02", "2024-01-03", "ok"),
        _daily_row("000002.SZ", "2024-01-02", "2024-01-03", "empty"),
        _daily_row("000003.SZ", None, None, "ok", fetched_at=pd.Timestamp("2026-06-14")),
    ]
    batch = CoverageLedger(str(tmp_path / "b"))
    batch.record_many(rows)

    single = CoverageLedger(str(tmp_path / "s"))
    for r in rows:
        single.record(**{k: v for k, v in r.items()})

    a = batch.read().reset_index(drop=True)
    b = single.read().reset_index(drop=True)
    assert list(a.columns) == LEDGER_COLUMNS
    assert list(b.columns) == LEDGER_COLUMNS
    pd.testing.assert_frame_equal(a, b)


def test_record_many_empty_is_noop(tmp_path):
    led = CoverageLedger(str(tmp_path / "c"))
    led.record_many([])
    assert not led.path.exists()  # no file created for an empty batch
    assert led.read().empty


def test_covered_intervals_counts_only_ok_empty(tmp_path):
    led = CoverageLedger(str(tmp_path / "c"))
    led.record_many([
        _daily_row("000001.SZ", "2024-01-02", "2024-01-03", "ok"),
        _daily_row("000001.SZ", "2024-01-04", "2024-01-05", "empty"),
        _daily_row("000001.SZ", "2024-01-06", "2024-01-07", "failed"),
        _daily_row("000001.SZ", "2024-01-08", "2024-01-09", "not_ready"),
    ])
    intervals = led.covered_intervals("market_daily", "000001.SZ")
    got = {(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in intervals}
    assert got == {("2024-01-02", "2024-01-03"), ("2024-01-04", "2024-01-05")}


def test_snapshot_fetched_at_returns_latest_successful(tmp_path):
    led = CoverageLedger(str(tmp_path / "c"))
    led.record_many([
        _daily_row("g", None, None, "ok", fetched_at=pd.Timestamp("2026-06-10")),
        _daily_row("g", None, None, "empty", fetched_at=pd.Timestamp("2026-06-12")),
        _daily_row("g", None, None, "failed", fetched_at=pd.Timestamp("2026-06-20")),
        _daily_row("g", None, None, "not_ready", fetched_at=pd.Timestamp("2026-06-21")),
    ], )
    # latest among ok/empty only (failed/not_ready ignored)
    assert led.snapshot_fetched_at("market_daily", "g") == pd.Timestamp("2026-06-12")
    assert led.snapshot_fetched_at("market_daily", "absent") is None


# --------------------------------------------------------------------------- #
# in-process cache: repeated lookups do not re-read the parquet
# --------------------------------------------------------------------------- #
def test_repeated_lookups_do_not_reread_parquet(tmp_path):
    writer = CoverageLedger(str(tmp_path / "c"))
    writer.record_many([
        _daily_row("000001.SZ", "2024-01-02", "2024-01-03", "ok"),
        _daily_row("g", None, None, "ok"),
    ])
    # a FRESH (cold-cache) instance over the same file
    reader = CoverageLedger(str(tmp_path / "c"))
    calls = {"n": 0}
    orig = reader._read_frame

    def counting():
        calls["n"] += 1
        return orig()

    reader._read_frame = counting
    for _ in range(5):
        reader.covered_intervals("market_daily", "000001.SZ")
        reader.snapshot_fetched_at("market_daily", "g")
    reader.read()
    reader.read()
    assert calls["n"] == 1  # one cold read; cache + memo serve the rest


def test_external_write_is_not_served_stale(tmp_path):
    a = CoverageLedger(str(tmp_path / "c"))
    # cold lookup over an absent file -> empty, caches "absent"
    assert a.covered_intervals("market_daily", "000001.SZ") == []
    # a different instance writes a covering row
    b = CoverageLedger(str(tmp_path / "c"))
    b.record_many([_daily_row("000001.SZ", "2024-01-02", "2024-01-03", "ok")])
    # the first instance must see the new coverage (mtime invalidation), not the
    # stale "absent" it cached on the cold lookup.
    intervals = a.covered_intervals("market_daily", "000001.SZ")
    assert len(intervals) == 1


def test_read_returns_copy_not_internal_frame(tmp_path):
    led = CoverageLedger(str(tmp_path / "c"))
    led.record_many([_daily_row("000001.SZ", "2024-01-02", "2024-01-03", "ok")])
    frame = led.read()
    frame.loc[0, "status"] = "MUTATED"
    # mutating the returned copy must not corrupt the ledger's cached lookups
    intervals = led.covered_intervals("market_daily", "000001.SZ")
    assert len(intervals) == 1


# --------------------------------------------------------------------------- #
# intraday IntradayCoverageLedger mirrors the pattern
# --------------------------------------------------------------------------- #
def _intra_row(key, start, end, status, fetched_at=_TS):
    return {
        "endpoint": "stk_mins_1min", "key_type": "symbol", "key": key,
        "raw_freq": "1min", "start_time": start, "end_time": end,
        "fields_hash": "h", "row_count": 1 if status == "ok" else 0,
        "status": status, "fetched_at": fetched_at,
    }


def test_intraday_record_many_equals_repeated_record(tmp_path):
    rows = [
        _intra_row("000001.SZ", "2024-01-02 09:31:00", "2024-01-02 15:00:00", "ok"),
        _intra_row("000002.SZ", "2024-01-03 09:31:00", "2024-01-03 15:00:00", "empty"),
    ]
    batch = IntradayCoverageLedger(str(tmp_path / "b"))
    batch.record_many(rows)
    single = IntradayCoverageLedger(str(tmp_path / "s"))
    for r in rows:
        single.record(**r)
    a = batch.read().reset_index(drop=True)
    b = single.read().reset_index(drop=True)
    assert list(a.columns) == INTRADAY_LEDGER_COLUMNS
    pd.testing.assert_frame_equal(a, b)


def test_intraday_covered_day_intervals_semantics_and_cache(tmp_path):
    writer = IntradayCoverageLedger(str(tmp_path / "c"))
    writer.record_many([
        _intra_row("000001.SZ", "2024-01-02 09:31:00", "2024-01-02 15:00:00", "ok"),
        _intra_row("000001.SZ", "2024-01-03 09:31:00", "2024-01-03 15:00:00", "failed"),
    ])
    reader = IntradayCoverageLedger(str(tmp_path / "c"))
    calls = {"n": 0}
    orig = reader._read_frame

    def counting():
        calls["n"] += 1
        return orig()

    reader._read_frame = counting
    days = None
    for _ in range(4):
        days = reader.covered_day_intervals("stk_mins_1min", "000001.SZ", "1min")
    assert calls["n"] == 1  # one cold read served all four lookups
    # only the ok day counts (failed ignored), day-normalized
    assert days == [(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"))]


def test_intraday_record_many_empty_is_noop(tmp_path):
    led = IntradayCoverageLedger(str(tmp_path / "c"))
    led.record_many([])
    assert not led.path.exists()
    assert led.read().empty


# --------------------------------------------------------------------------- #
# cache wiring: not-ready split writes its rows in ONE batch
# --------------------------------------------------------------------------- #
class _DBFetch:
    def __init__(self, catalog):
        self.catalog = catalog
        self.calls: list[tuple] = []

    def __call__(self, symbol, s, e):
        self.calls.append((symbol, s, e))
        S, E = pd.Timestamp(s), pd.Timestamp(e)
        rows = [r for r in self.catalog.get(symbol, []) if S <= pd.Timestamp(r[0]) <= E]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": [symbol] * len(rows),
            "trade_date": [r[0].replace("-", "") for r in rows],
            "pe": [r[1] for r in rows], "pb": [r[2] for r in rows],
            "total_mv": [r[3] for r in rows],
        })


_DB_CAT = {"000001.SZ": [
    ("2024-01-02", 10.0, 1.0, 1000.0),
    ("2024-01-03", 11.0, 1.1, 1100.0),
    ("2024-01-04", 12.0, 1.2, 1200.0),
]}


def _cache(tmp_path, **kw):
    root = str(tmp_path / "cache")
    return TushareCache(
        CacheParquetStore(root), CoverageLedger(root),
        clock=lambda: pd.Timestamp("2026-06-13 21:00:00"), **kw,
    )


def test_not_ready_split_records_in_one_batch(tmp_path):
    today = pd.Timestamp("2024-01-05")
    cache = _cache(tmp_path, refresh_recent_days=0, not_ready_days=1, today=today)
    batches: list[int] = []
    orig = cache._ledger.record_many

    def spy(rows):
        rows = list(rows)
        batches.append(len(rows))
        return orig(rows)

    cache._ledger.record_many = spy
    f = _DBFetch(_DB_CAT)
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-05", f)

    # the single fetched gap carves hist (ok, through 01-04) + pending 01-05
    # (not_ready) and writes BOTH in ONE record_many call.
    assert batches == [2]
    led = cache._ledger.read()
    assert (led["status"] == "not_ready").any()
    assert (led["status"] == "ok").any()


def test_plain_gap_records_single_row_batch(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0)  # not_ready_days=0 default
    batches: list[int] = []
    orig = cache._ledger.record_many

    def spy(rows):
        rows = list(rows)
        batches.append(len(rows))
        return orig(rows)

    cache._ledger.record_many = spy
    f = _DBFetch(_DB_CAT)
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-04", f)
    assert batches == [1]  # plain gap -> one coverage row, one batch


def test_ledger_columns_have_no_secret_fields():
    # structural: the ledger schema is endpoint metadata only (no token field).
    for col in LEDGER_COLUMNS + INTRADAY_LEDGER_COLUMNS:
        assert "token" not in col.lower()
        assert "secret" not in col.lower()
