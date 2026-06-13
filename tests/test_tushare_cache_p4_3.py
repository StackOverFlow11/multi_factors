"""P4-3 cache endpoint tests: daily_basic / fina_indicator / index_member_all.

Covers cold/warm/partial/failed read-through, the not-ready pending window, the
fina late-disclosure tail, and the per-endpoint update summary. Fake fetch
callables; network-free; no token.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import (
    DAILY_BASIC,
    FINA_INDICATOR,
    INDEX_MEMBER_ALL,
    TushareCache,
)

_CLK = lambda: pd.Timestamp("2026-06-13 21:00:00")  # noqa: E731


def _cache(tmp_path, **kw):
    root = str(tmp_path / "cache")
    return TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK, **kw)


# --------------------------------------------------------------------------- #
# daily_basic — dense per-symbol date-range
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
            "pe": [r[1] for r in rows],
            "pb": [r[2] for r in rows],
            "total_mv": [r[3] for r in rows],
        })


_DB_CAT = {"000001.SZ": [
    ("2024-01-02", 10.0, 1.0, 1000.0),
    ("2024-01-03", 11.0, 1.1, 1100.0),
    ("2024-01-04", 12.0, 1.2, 1200.0),
]}


def test_daily_basic_cold_warm_partial(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0)
    f = _DBFetch(_DB_CAT)
    cold = cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", f)
    assert list(cold.columns) == ["date", "symbol", "pe", "pb", "total_mv"]
    assert len(cold) == 2
    assert f.calls  # cold hit the API
    # warm identical -> zero calls
    f.calls.clear()
    warm = cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", f)
    assert f.calls == []
    pd.testing.assert_frame_equal(
        cold.sort_values(["date", "symbol"]).reset_index(drop=True),
        warm.sort_values(["date", "symbol"]).reset_index(drop=True),
    )
    # partial: extend by one day -> only the new day fetched
    f.calls.clear()
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-04", f)
    assert len(f.calls) == 1
    assert f.calls[0][1] == "20240104" and f.calls[0][2] == "20240104"


def test_daily_basic_failed_fetch_records_no_coverage(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0)

    def boom(symbol, s, e):
        raise ConnectionError("transient")

    with pytest.raises(ConnectionError):
        cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", boom)
    assert cache._ledger.read().empty  # failed fetch is not coverage
    # retried successfully later
    good = _DBFetch(_DB_CAT)
    out = cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", good)
    assert len(out) == 2 and good.calls


def test_daily_basic_not_ready_today_is_retried(tmp_path):
    # today = 2024-01-05; data published only through 01-04 at 21:00.
    cat = {"000001.SZ": _DB_CAT["000001.SZ"]}  # no 01-05 row yet
    today = pd.Timestamp("2024-01-05")
    cache = _cache(tmp_path, refresh_recent_days=0, not_ready_days=1, today=today)
    f = _DBFetch(cat)
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-05", f)
    led = cache._ledger.read()
    # the pending day 01-05 returned nothing -> recorded not_ready (NOT coverage)
    assert (led["status"] == "not_ready").any()
    # a later run still sees 01-05 as a gap and refetches it
    f.calls.clear()
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-05", f)
    assert any(c[1] == "20240105" for c in f.calls)


def test_not_ready_disabled_covers_today(tmp_path):
    # not_ready_days=0 (default): today's empty would be covered -> NOT retried.
    today = pd.Timestamp("2024-01-05")
    cache = _cache(tmp_path, refresh_recent_days=0, not_ready_days=0, today=today)
    f = _DBFetch(_DB_CAT)
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-05", f)
    f.calls.clear()
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-05", f)
    assert f.calls == []  # whole range (incl 01-05) recorded as covered


# --------------------------------------------------------------------------- #
# fina_indicator — report-period range, keeps ann_date, late-disclosure tail
# --------------------------------------------------------------------------- #
class _FinaFetch:
    """Fake fina fetch: returns the full superset (roe / netprofit_yoy / gpm)."""

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
            "end_date": [r[0].replace("-", "") for r in rows],
            "ann_date": [r[1].replace("-", "") for r in rows],
            "roe": [r[2] for r in rows],
            "netprofit_yoy": [r[3] for r in rows],
            "grossprofit_margin": [r[4] for r in rows],
        })


# (end_date, ann_date, roe, np_yoy, grossprofit_margin)
_FINA_CAT = {"000001.SZ": [
    ("2023-03-31", "2023-04-20", 5.0, 10.0, 30.0),
    ("2023-06-30", "2023-08-15", 6.0, 12.0, 31.0),
]}


def test_fina_stores_superset_keeps_ann_date(tmp_path):
    cache = _cache(tmp_path)
    f = _FinaFetch(_FINA_CAT)
    cold = cache.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)
    # the cache ALWAYS stores the canonical superset (field-set independent)
    assert list(cold.columns) == [
        "symbol", "ann_date", "end_date", "roe", "netprofit_yoy", "grossprofit_margin",
    ]
    assert set(cold["ann_date"]) == {"20230420", "20230815"}  # disclosure dates kept
    assert len(cold) == 2
    # warm identical request (fina tail off) -> zero calls
    f.calls.clear()
    cache2 = TushareCache(
        cache._store, cache._ledger, clock=_CLK,
        recent_tail_overrides={FINA_INDICATOR: 0},
    )
    warm = cache2.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)
    assert f.calls == []
    assert len(warm) == 2


def test_fina_subset_warm_then_superset_no_keyerror(tmp_path):
    # REGRESSION (codex acceptance blocker): a warm that only NEEDED roe must not
    # block a later request that needs grossprofit_margin. Because the cache stores
    # the superset, the second read finds the column (no KeyError) with real data.
    cache = TushareCache(
        cache_store := CacheParquetStore(str(tmp_path / "c")),
        cache_led := CoverageLedger(str(tmp_path / "c")),
        clock=_CLK, recent_tail_overrides={FINA_INDICATOR: 0},
    )
    f = _FinaFetch(_FINA_CAT)
    cache.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)  # cold warm
    f.calls.clear()
    # a DIFFERENT instance over the SAME store/ledger (a later config's warm cache)
    cache2 = TushareCache(
        cache_store, cache_led, clock=_CLK, recent_tail_overrides={FINA_INDICATOR: 0}
    )
    out = cache2.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)
    assert f.calls == []  # warm hit (no refetch)
    assert "grossprofit_margin" in out.columns
    assert set(out["grossprofit_margin"]) == {30.0, 31.0}  # real data, not NaN


def test_fina_tail_refetches_recent_periods(tmp_path):
    # a long fina tail re-fetches the trailing window of the requested range every
    # run, so a LATE disclosure of a recent period is caught.
    cache = _cache(tmp_path, recent_tail_overrides={FINA_INDICATOR: 400})
    f = _FinaFetch(_FINA_CAT)
    cache.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)
    f.calls.clear()
    # second identical run still refetches the trailing window (late disclosures)
    cache.fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", f)
    assert f.calls  # NOT zero — the tail is always refetched


def test_fina_fields_cover_all_financial_factors(tmp_path):  # noqa: ARG001
    # drift guard: the cache superset must cover every project financial factor.
    from data.cache.tushare_cache import FINA_FIELDS
    from factors.compute.financial import SUPPORTED_FIELDS

    assert set(FINA_FIELDS) >= set(SUPPORTED_FIELDS)


# --------------------------------------------------------------------------- #
# index_member_all — per-symbol dimension (SW in/out intervals)
# --------------------------------------------------------------------------- #
class _MemberFetch:
    def __init__(self, catalog):
        self.catalog = catalog
        self.calls: list[str] = []

    def __call__(self, symbol):
        self.calls.append(symbol)
        rows = self.catalog.get(symbol, [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": [symbol] * len(rows),
            "l1_name": [r[0] for r in rows],
            "l2_name": [r[1] for r in rows],
            "l3_name": [r[2] for r in rows],
            "in_date": [r[3] for r in rows],
            "out_date": [r[4] for r in rows],
        })


_MEM_CAT = {"000001.SZ": [
    ("Banks", "Banks2", "Banks3", "20200101", None),
]}


def test_index_member_all_cold_warm(tmp_path):
    cache = _cache(tmp_path, refresh_dimension_days=30)
    f = _MemberFetch(_MEM_CAT)
    cold = cache.index_member_all(["000001.SZ"], f)
    assert list(cold.columns) == [
        "symbol", "l1_name", "l2_name", "l3_name", "in_date", "out_date",
    ]
    assert cold.iloc[0]["l1_name"] == "Banks"
    assert pd.isna(cold.iloc[0]["out_date"])  # active membership
    assert f.calls == ["000001.SZ"]
    # warm: snapshot fresh -> zero calls
    f.calls.clear()
    warm = cache.index_member_all(["000001.SZ"], f)
    assert f.calls == []
    assert len(warm) == 1


def test_update_summary_counts(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0)
    cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", _DBFetch(_DB_CAT))
    summ = cache.update_summary()
    assert summ[DAILY_BASIC]["requests"] >= 1
    assert summ[DAILY_BASIC]["rows_written"] == 2
    assert summ[INDEX_MEMBER_ALL]["requests"] == 0  # untouched endpoint seeds 0
