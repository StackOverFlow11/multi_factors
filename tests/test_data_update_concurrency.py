"""D5 bounded-concurrency cache warm: config + serial==concurrent + failure model.

Network-free, synthetic fetch callables. Proves the opt-in concurrency config,
that ``max_workers>1`` produces the SAME frames / ledger rows / request counts as
serial, and that the failure / empty / not_ready semantics are unchanged
(successes durable, failures uncovered+retryable).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache
from qt.config import DataUpdateCfg, DataUpdateConcurrencyCfg, load_config

_CLK = lambda: pd.Timestamp("2026-06-13 21:00:00")  # noqa: E731


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_default_config_concurrency_is_serial():
    cfg = load_config("config/data_update.yaml")
    assert cfg.data_update is not None
    assert cfg.data_update.concurrency.max_workers == 1


def test_data_update_cfg_without_concurrency_block_defaults_serial():
    assert DataUpdateCfg().concurrency.max_workers == 1


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_max_workers_below_one_rejected(bad):
    with pytest.raises(ValidationError):
        DataUpdateConcurrencyCfg(max_workers=bad)


def test_max_workers_accepts_positive():
    assert DataUpdateConcurrencyCfg(max_workers=8).max_workers == 8


# --------------------------------------------------------------------------- #
# synthetic dense fetch (daily_basic-shaped)
# --------------------------------------------------------------------------- #
_CAT = {
    "000001.SZ": [("2024-01-02", 10.0, 1.0, 1000.0), ("2024-01-03", 11.0, 1.1, 1100.0)],
    "000002.SZ": [("2024-01-02", 20.0, 2.0, 2000.0), ("2024-01-03", 21.0, 2.1, 2100.0)],
    "000003.SZ": [("2024-01-02", 30.0, 3.0, 3000.0), ("2024-01-03", 31.0, 3.1, 3100.0)],
    "000004.SZ": [("2024-01-02", 40.0, 4.0, 4000.0), ("2024-01-03", 41.0, 4.1, 4100.0)],
}
_SYMS = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]


class _DBFetch:
    def __init__(self, catalog=_CAT, fail_on=None):
        self.catalog = catalog
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def __call__(self, symbol, s, e):
        self.calls.append(symbol)
        if symbol in self.fail_on:
            raise ConnectionError("transient boom")
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


def _cache(tmp_path, **kw):
    root = str(tmp_path / "cache")
    return TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK, **kw)


def _sorted(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def _ledger_sorted(led):
    cols = [c for c in led.columns if c != "fetched_at"]
    return led.sort_values(["key", "start_date"]).reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- #
# serial == concurrent equivalence
# --------------------------------------------------------------------------- #
def test_concurrent_equals_serial_frames_and_ledger(tmp_path):
    serial = _cache(tmp_path / "s", refresh_recent_days=0, max_workers=1)
    concurrent = _cache(tmp_path / "c", refresh_recent_days=0, max_workers=4)
    a = serial.daily_basic(_SYMS, "2024-01-02", "2024-01-03", _DBFetch())
    b = concurrent.daily_basic(_SYMS, "2024-01-02", "2024-01-03", _DBFetch())
    # identical output frames
    pd.testing.assert_frame_equal(_sorted(a), _sorted(b))
    # identical ledger contents (fetched_at fixed by _CLK; order-independent)
    pd.testing.assert_frame_equal(
        _ledger_sorted(serial._ledger.read()),
        _ledger_sorted(concurrent._ledger.read()),
    )
    # identical request counts
    assert serial.stats() == concurrent.stats()


def test_concurrent_warm_then_cold_warm_is_zero_calls(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0, max_workers=4)
    f1 = _DBFetch()
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f1)
    assert sorted(f1.calls) == _SYMS  # cold: each symbol fetched once
    f2 = _DBFetch()
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f2)
    assert f2.calls == []  # warm: fully covered -> zero fetches


# --------------------------------------------------------------------------- #
# failure semantics: successes durable, failure uncovered + retryable
# --------------------------------------------------------------------------- #
def test_concurrent_failure_keeps_successes_durable_and_retries(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0, max_workers=4)
    f = _DBFetch(fail_on={"000002.SZ"})
    with pytest.raises(ConnectionError):
        cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f)
    led = cache._ledger.read()
    covered = set(led.loc[led["status"].isin(["ok", "empty"]), "key"].astype(str))
    # the three good symbols are durable; the failed one is NOT recorded
    assert {"000001.SZ", "000003.SZ", "000004.SZ"} <= covered
    assert "000002.SZ" not in covered
    # a later run (now healthy) refetches ONLY the previously-failed symbol
    f2 = _DBFetch()
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f2)
    assert f2.calls == ["000002.SZ"]


def test_concurrent_empty_return_counts_as_coverage(tmp_path):
    cache = _cache(tmp_path, refresh_recent_days=0, max_workers=4)
    # 000004 has no rows in this catalog -> empty return, still coverage
    cat = {k: v for k, v in _CAT.items() if k != "000004.SZ"}
    f = _DBFetch(catalog=cat)
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f)
    f2 = _DBFetch(catalog=cat)
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-03", f2)
    assert f2.calls == []  # empty return recorded -> no needless refetch


def test_concurrent_not_ready_still_pending(tmp_path):
    # today=2024-01-04, not_ready_days=1: 01-04 unpublished -> not_ready (not covered)
    today = pd.Timestamp("2024-01-04")
    cache = _cache(
        tmp_path, refresh_recent_days=0, not_ready_days=1, today=today, max_workers=4
    )
    f = _DBFetch()  # catalog has no 01-04 rows
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-04", f)
    led = cache._ledger.read()
    assert (led["status"] == "not_ready").any()
    f2 = _DBFetch()
    cache.daily_basic(_SYMS, "2024-01-02", "2024-01-04", f2)
    assert f2.calls  # the pending (not_ready) day is still a gap -> retried


def test_single_symbol_concurrent_uses_serial_path(tmp_path):
    # one symbol -> len(symbols) == 1 -> serial loop even with max_workers>1
    cache = _cache(tmp_path, refresh_recent_days=0, max_workers=4)
    f = _DBFetch()
    out = cache.daily_basic(["000001.SZ"], "2024-01-02", "2024-01-03", f)
    assert len(out) == 2 and f.calls == ["000001.SZ"]
