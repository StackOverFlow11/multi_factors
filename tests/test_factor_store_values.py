"""Sparse factor-value storage (refactor D3, commit 3): write/read/upsert + locking.

Network-free. Covers the R21 write semantics (atomic + per-artifact lock), the
"corrupt file is a miss" rule, fingerprint read-validation (schema void / price_level
per-symbol void), and the concurrent-gap-fill zero-lost-rows guard.
"""

from __future__ import annotations

import multiprocessing as mp

import pandas as pd

from factors.store import FactorValueStore, StoreKey
from factors.store.fingerprint import adj_events_hash

_KEY = StoreKey(factor_id="demo_factor", params_hash="p0", code_hash="c0", view="decision")
_FP = {"fingerprint_version": "v1", "schema_version": "SCHEMA0", "adjustment": "none", "adj_events": None}


def _series(dates, symbol="000001.SZ", start=0.0):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), symbol) for d in dates], names=["date", "symbol"]
    )
    return pd.Series(
        [start + i for i in range(len(dates))], index=idx, name=_KEY.factor_id, dtype=float
    )


def test_write_then_read_roundtrips(tmp_path):
    store = FactorValueStore(tmp_path)
    s = _series(["2024-01-02", "2024-01-03"])
    assert store.write(_KEY, s, fingerprint=_FP) == 2
    back = store.read(_KEY)
    assert back.equals(s.sort_index())


def test_read_missing_is_none(tmp_path):
    assert FactorValueStore(tmp_path).read(_KEY) is None


def test_corrupt_file_is_a_miss_not_a_crash(tmp_path):
    store = FactorValueStore(tmp_path)
    store.write(_KEY, _series(["2024-01-02"]), fingerprint=_FP)
    store.path(_KEY).write_bytes(b"not a parquet file")
    assert store.read(_KEY) is None  # a miss, never an exception


def test_write_leaves_no_tmp_residue(tmp_path):
    store = FactorValueStore(tmp_path)
    store.write(_KEY, _series(["2024-01-02"]), fingerprint=_FP)
    residue = list(store.path(_KEY).parent.glob("*.tmp"))
    assert residue == []


def test_upsert_merges_and_dedups_keep_last(tmp_path):
    store = FactorValueStore(tmp_path)
    store.write(_KEY, _series(["2024-01-02", "2024-01-03"], start=0.0), fingerprint=_FP)
    # overlapping (jan-03) + a new date (jan-04); new rows win the collision
    over = _series(["2024-01-03", "2024-01-04"], start=99.0)
    store.upsert(_KEY, over, fingerprint=_FP)
    merged = store.read(_KEY)
    assert len(merged) == 3
    assert merged.loc[(pd.Timestamp("2024-01-03"), "000001.SZ")] == 99.0


def test_stored_fingerprint_is_readable(tmp_path):
    store = FactorValueStore(tmp_path)
    store.write(_KEY, _series(["2024-01-02"]), fingerprint=_FP)
    assert store.stored_fingerprint(_KEY) == _FP


def test_read_valid_matching_fingerprint_returns_all(tmp_path):
    store = FactorValueStore(tmp_path)
    s = _series(["2024-01-02", "2024-01-03"])
    store.write(_KEY, s, fingerprint=_FP)
    got = store.read_valid(_KEY, expected_fingerprint=_FP)
    assert got is not None and len(got) == 2


def test_read_valid_schema_mismatch_voids_whole_artifact(tmp_path):
    store = FactorValueStore(tmp_path)
    store.write(_KEY, _series(["2024-01-02"]), fingerprint=_FP)
    expected = dict(_FP, schema_version="SCHEMA1")
    assert store.read_valid(_KEY, expected_fingerprint=expected) is None


def test_read_valid_price_level_voids_only_the_restated_symbol(tmp_path):
    store = FactorValueStore(tmp_path)
    # two symbols with per-symbol adj hashes (a price_level artifact)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "AAA"), (pd.Timestamp("2024-01-02"), "BBB")],
        names=["date", "symbol"],
    )
    s = pd.Series([1.0, 2.0], index=idx, name=_KEY.factor_id)
    aaa = adj_events_hash(pd.Series([1.0], index=pd.to_datetime(["2024-01-02"])))
    bbb = adj_events_hash(pd.Series([2.0], index=pd.to_datetime(["2024-01-02"])))
    stored_fp = {"fingerprint_version": "v1", "schema_version": "S", "adjustment": "price_level",
                 "adj_events": {"AAA": aaa, "BBB": bbb}}
    store.write(_KEY, s, fingerprint=stored_fp)
    # BBB's anchor was restated (ex-date): its stored hash no longer matches expected
    bbb_new = adj_events_hash(pd.Series([9.9], index=pd.to_datetime(["2024-01-02"])))
    expected_fp = dict(stored_fp, adj_events={"AAA": aaa, "BBB": bbb_new})
    got = store.read_valid(_KEY, expected_fingerprint=expected_fp)
    assert list(got.index.get_level_values("symbol")) == ["AAA"]


# --------------------------------------------------------------------------- #
# concurrent gap-fill: two processes upsert the SAME factor -> zero lost rows
# --------------------------------------------------------------------------- #
def _concurrent_worker(root: str, dates: list[str]):
    store = FactorValueStore(root)
    fp = _FP
    # one upsert per date so the read-modify-write windows of the two processes
    # interleave; the per-artifact lock must serialize them (no lost update).
    for d in dates:
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp(d), "000001.SZ")], names=["date", "symbol"])
        store.upsert(_KEY, pd.Series([1.0], index=idx, name=_KEY.factor_id), fingerprint=fp)


def test_concurrent_gap_fill_loses_no_rows(tmp_path):
    dates_a = [f"2024-01-{d:02d}" for d in range(1, 26)]  # 25 distinct days
    dates_b = [f"2024-02-{d:02d}" for d in range(1, 26)]  # 25 distinct days (disjoint)
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_concurrent_worker, args=(str(tmp_path), dates_a)),
        ctx.Process(target=_concurrent_worker, args=(str(tmp_path), dates_b)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    merged = FactorValueStore(tmp_path).read(_KEY)
    # every one of the 50 disjoint days survived (no read-modify-write lost update)
    assert len(merged) == 50
    assert set(merged.index.get_level_values("date")) == {
        pd.Timestamp(d) for d in dates_a + dates_b
    }
