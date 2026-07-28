"""Tail-recompute incremental engine (refactor D3, commit 4, §3.3 / R4 / R5).

Network-free. A faithful NESTED-LOOKBACK fixture factor (a trailing-L sum of a
trailing-B mean, so its transitive depth is L+B-2, NOT the headline L) exercises
the batch==incremental invariant AND the §六.18 teeth (sizing the overlap by the
headline instead of the transitive depth spuriously mismatches). Plus: the
overlap window follows the cache config (R5), a missing lookback_depth is a loud
error, the cold path, and the mismatch-action classification by adjustment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import MARKET_DAILY
from factors.base import Factor
from factors.spec import FactorSpec, PanelField
from factors.store import (
    CacheHorizonConfig,
    FactorValueStore,
    StoreKey,
    data_fingerprint,
    factor_lookback_depth,
    overlap_window,
    schema_version_hash,
    tail_recompute,
)


def _fingerprint_for(adjustment):
    """A stored fingerprint for the given adjustment (price_level needs adj_events)."""
    if adjustment == "price_level":
        return {
            "fingerprint_version": "v1",
            "schema_version": schema_version_hash(),
            "adjustment": "price_level",
            "adj_events": {"AAA": "h_aaa", "BBB": "h_bbb"},
        }
    return data_fingerprint(adjustment=adjustment)

_L = 3  # trailing sum window (the "headline")
_B = 3  # nested trailing-mean window; transitive depth = _L + _B - 2 = 4
_TRANSITIVE_DEPTH = _L + _B - 2
_NAME = "nl_fixture"


# --------------------------------------------------------------------------- #
# fixture factor + a faithful nested-lookback recompute callable
# --------------------------------------------------------------------------- #
def _make_factor(*, lookback_depth, adjustment="returns_invariant"):
    class _NestedLookbackFixture(Factor):
        name = _NAME
        spec = FactorSpec(
            factor_id=_NAME,
            version="1.0",
            description="fixture: trailing-L sum of a trailing-B mean (nested lookback)",
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=("close",),
            requires=(PanelField("close", source=MARKET_DAILY),),
            adjustment=adjustment,
            overnight_boundary="none",
            lookback_depth=lookback_depth,
        )

        def compute(self, panel):  # pragma: no cover - engine uses the injected recompute
            return panel["close"].rename(self.name)

    return _NestedLookbackFixture()


def _business_dates(n):
    return pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=n))


def _raw_panel(dates, symbols=("AAA", "BBB")):
    rng = np.random.default_rng(7)
    rows = {}
    for sym in symbols:
        vals = rng.normal(10.0, 1.0, size=len(dates))
        for d, v in zip(dates, vals):
            rows[(d, sym)] = float(v)
    idx = pd.MultiIndex.from_tuples(list(rows), names=["date", "symbol"])
    return pd.Series(list(rows.values()), index=idx, name="raw").sort_index()


def _make_recompute(raw, *, delta_symbol=None, delta=0.0):
    """A materializer emitting [emit_start, end] after loading ``warmup`` history.

    Computes factor[d] = sum_{L} ( mean_{B} raw ) per symbol. Each date's value is
    computed from its OWN explicit trailing window (a fixed-order sum), so it is
    BITWISE-STABLE across different load windows — batch and incremental agree
    exactly (unlike a running rolling sum, whose accumulation path depends on the
    array start). A value is honest NaN if any raw its window needs was not loaded
    (the warm-up under-sampling the §六.18 teeth exploit). ``delta_symbol`` shifts
    one symbol's raw (models an upstream data revision).
    """
    axis = pd.DatetimeIndex(sorted(pd.unique(raw.index.get_level_values("date"))))
    n = len(axis)
    symbols = sorted(set(map(str, raw.index.get_level_values("symbol"))))
    # per-symbol raw indexed by axis position
    by_sym: dict[str, np.ndarray] = {}
    for sym in symbols:
        col = raw[raw.index.get_level_values("symbol") == sym]
        arr = np.full(n, np.nan)
        for (d, _s), v in col.items():
            arr[int(axis.get_loc(d))] = float(v)
        if delta_symbol == sym and delta:
            arr = arr + delta
        by_sym[sym] = arr

    def recompute(emit_start, end, warmup):
        end_pos = int(axis.get_loc(pd.Timestamp(end)))
        if emit_start is None:
            emit_pos = 0
            load_lo = 0
        else:
            emit_pos = int(axis.get_loc(pd.Timestamp(emit_start)))
            load_lo = max(0, emit_pos - warmup)  # loaded input = [emit-warmup, end]
        rows: list[tuple[pd.Timestamp, str]] = []
        vals: list[float] = []
        for sym in symbols:
            arr = by_sym[sym]
            for i in range(emit_pos, end_pos + 1):
                lo = i - (_L - 1) - (_B - 1)  # earliest raw position the window needs
                if lo < load_lo or lo < 0:
                    value = np.nan  # under-sampled (raw not loaded / no history)
                else:
                    # base[j] = mean(raw[j-B+1 .. j]); factor = sum_{j=i-L+1}^{i} base[j]
                    base = [float(arr[j - _B + 1 : j + 1].mean()) for j in range(i - _L + 1, i + 1)]
                    value = float(np.sum(base))
                rows.append((axis[i], sym))
                vals.append(value)
        idx = pd.MultiIndex.from_tuples(rows, names=["date", "symbol"])
        return pd.Series(vals, index=idx, name=_NAME).sort_index()

    return recompute


def _key():
    return StoreKey(factor_id=_NAME, params_hash="p", code_hash="c", view="decision")


def _fp():
    return data_fingerprint(adjustment="returns_invariant")


def _nan_aware_equal(a: pd.Series, b: pd.Series) -> bool:
    a = a.sort_index(kind="mergesort")
    b = b.reindex(a.index)
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return bool(np.array_equal(av, bv, equal_nan=True)) and a.index.equals(b.index)


# --------------------------------------------------------------------------- #
# overlap window sizing (R4 / R5)
# --------------------------------------------------------------------------- #
def test_overlap_window_is_max_of_depth_and_horizon():
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH)
    # small horizon -> depth dominates
    assert overlap_window(factor, CacheHorizonConfig(refresh_recent_days=1)) == _TRANSITIVE_DEPTH
    # large horizon -> horizon dominates (R5: follows the live cache config)
    assert overlap_window(factor, CacheHorizonConfig(refresh_recent_days=20)) == 20


def test_overlap_window_widens_when_refresh_recent_days_grows():
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH)
    small = overlap_window(factor, CacheHorizonConfig(refresh_recent_days=5))
    big = overlap_window(factor, CacheHorizonConfig(refresh_recent_days=30))
    assert big > small and big == 30


def test_missing_lookback_depth_is_a_readable_error():
    factor = _make_factor(lookback_depth=None)
    with pytest.raises(ValueError, match="no spec.lookback_depth"):
        factor_lookback_depth(factor)
    with pytest.raises(ValueError, match="no spec.lookback_depth"):
        overlap_window(factor, CacheHorizonConfig(refresh_recent_days=1))


# --------------------------------------------------------------------------- #
# batch == incremental on a NESTED-LOOKBACK factor (§3.3 / R4 / §六.18)
# --------------------------------------------------------------------------- #
def test_batch_equals_incremental_with_correct_transitive_depth(tmp_path):
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    recompute = _make_recompute(raw)
    today = dates[-1]
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH)
    cfg = CacheHorizonConfig(refresh_recent_days=1)  # depth dominates the overlap

    # batch: one full compute
    batch_store = FactorValueStore(tmp_path / "batch")
    full = recompute(None, today, _TRANSITIVE_DEPTH)
    batch_store.write(_key(), full, fingerprint=_fp())

    # incremental: seed the store truncated, then tail-recompute up to today
    inc_store = FactorValueStore(tmp_path / "inc")
    cutoff = dates[-6]
    seeded = full[full.index.get_level_values("date") <= cutoff]
    inc_store.write(_key(), seeded, fingerprint=_fp())
    result = tail_recompute(
        inc_store, _key(), factor, recompute=recompute, today=today,
        horizon_cfg=cfg, fingerprint=_fp(),
    )

    assert result.mismatched_symbols == ()  # no FALSE revision from under-sampling
    assert result.revision_detected is False
    assert result.rows_appended > 0
    assert _nan_aware_equal(inc_store.read(_key()), batch_store.read(_key()))


def test_headline_depth_spuriously_mismatches_the_overlap(tmp_path):
    # §六.18 teeth: sizing by the HEADLINE (_L=3 < transitive depth 4) under-samples
    # the overlap's earliest day -> a spurious mismatch (a false nightly recompute).
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    recompute = _make_recompute(raw)
    today = dates[-1]
    full = recompute(None, today, _TRANSITIVE_DEPTH)

    inc_store = FactorValueStore(tmp_path / "inc")
    cutoff = dates[-6]
    seeded = full[full.index.get_level_values("date") <= cutoff]
    inc_store.write(_key(), seeded, fingerprint=_fp())

    headline_factor = _make_factor(lookback_depth=_L)  # WRONG: headline, not transitive
    result = tail_recompute(
        inc_store, _key(), headline_factor, recompute=recompute, today=today,
        horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=_fp(),
    )
    assert result.mismatched_symbols != ()  # the headline window deceived it


# --------------------------------------------------------------------------- #
# cold path
# --------------------------------------------------------------------------- #
def test_cold_store_does_a_full_compute(tmp_path):
    dates = _business_dates(20)
    raw = _raw_panel(dates)
    recompute = _make_recompute(raw)
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH)
    store = FactorValueStore(tmp_path)
    result = tail_recompute(
        store, _key(), factor, recompute=recompute, today=dates[-1],
        horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=_fp(),
    )
    assert result.action == "cold_full"
    assert result.rows_appended == len(store.read(_key()))


# --------------------------------------------------------------------------- #
# mismatch action classification by adjustment (§3.3 修订 2)
# --------------------------------------------------------------------------- #
def _seed_and_revise(tmp_path, factor, adjustment, delta_symbol="AAA", caplog=None):
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    today = dates[-1]
    clean = _make_recompute(raw)
    full = clean(None, today, _TRANSITIVE_DEPTH)
    store = FactorValueStore(tmp_path)
    cutoff = dates[-6]
    fp = _fingerprint_for(adjustment)
    store.write(_key(), full[full.index.get_level_values("date") <= cutoff], fingerprint=fp)
    # a recompute whose overlap values are REVISED for one symbol
    revised = _make_recompute(raw, delta_symbol=delta_symbol, delta=5.0)
    logger = logging.getLogger("test.tailrecompute")
    return tail_recompute(
        store, _key(), factor, recompute=revised, today=today,
        horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=fp,
        logger=logger,
    )


def test_returns_invariant_mismatch_is_a_loud_revision(tmp_path, caplog):
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH, adjustment="returns_invariant")
    with caplog.at_level(logging.WARNING):
        result = _seed_and_revise(tmp_path, factor, "returns_invariant")
    assert "AAA" in result.mismatched_symbols
    assert result.revision_detected is True
    assert "AAA" in result.recolumned_symbols
    assert any("REVISION detected" in r.message for r in caplog.records)


def test_price_level_mismatch_is_expected_not_a_loud_revision(tmp_path, caplog):
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH, adjustment="price_level")
    with caplog.at_level(logging.WARNING):
        result = _seed_and_revise(tmp_path, factor, "price_level")
    assert "AAA" in result.mismatched_symbols
    # price_level: an ex-date re-based the level -> EXPECTED, not a loud revision
    assert result.revision_detected is False
    assert any("price_level overlap mismatch" in n for n in result.notes)
    assert not any("REVISION detected" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# LOW-1: a schema-version change is a MISS (full recompute), NOT a data revision
# --------------------------------------------------------------------------- #
def test_schema_version_change_triggers_full_recompute_not_a_revision(tmp_path, caplog):
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    recompute = _make_recompute(raw)
    today = dates[-1]
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH, adjustment="returns_invariant")
    full = recompute(None, today, _TRANSITIVE_DEPTH)

    store = FactorValueStore(tmp_path)
    cutoff = dates[-6]
    # seed under an OLD schema version, then tail-recompute under a NEW one
    fp_old = dict(_fp(), schema_version="SCHEMA_OLD")
    store.write(
        _key(), full[full.index.get_level_values("date") <= cutoff], fingerprint=fp_old
    )
    fp_new = dict(_fp(), schema_version="SCHEMA_NEW")
    with caplog.at_level(logging.WARNING):
        result = tail_recompute(
            store, _key(), factor, recompute=recompute, today=today,
            horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=fp_new,
            logger=logging.getLogger("test.tailrecompute"),
        )
    # a schema change is a MISS -> full recompute, NOT a loud returns_invariant revision
    assert result.action == "schema_void_full"
    assert result.revision_detected is False
    assert result.mismatched_symbols == ()
    assert not any("REVISION detected" in r.message for r in caplog.records)
    # the store now carries the new schema and the full column
    assert store.stored_fingerprint(_key())["schema_version"] == "SCHEMA_NEW"
    assert len(store.read(_key())) == len(full)


# --------------------------------------------------------------------------- #
# D4c footprint fills: stored NaN -> finite is NOT an upstream revision (D5 C4)
# --------------------------------------------------------------------------- #
def _seed_with_footprint_nans(tmp_path, nan_symbol="AAA", nan_tail_days=3):
    """Seed a store whose overlap carries explicit NaN FOOTPRINT rows.

    Mirrors ``factors.service._record_fill_footprint``: a fill covered these
    cells and the factor produced nothing, so the store holds NaN. After the
    upstream cache warms, the same recompute returns FINITE values there.
    """
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    clean = _make_recompute(raw)
    today = dates[-1]
    full = clean(None, today, _TRANSITIVE_DEPTH)
    cutoff = dates[-6]
    seeded = full[full.index.get_level_values("date") <= cutoff].copy()
    d = seeded.index.get_level_values("date")
    s = seeded.index.get_level_values("symbol")
    footprint = (s == nan_symbol) & (d > cutoff - pd.Timedelta(days=nan_tail_days))
    assert footprint.any()
    seeded[footprint] = np.nan
    store = FactorValueStore(tmp_path)
    store.write(_key(), seeded, fingerprint=_fp())
    return store, clean, full, today


def test_stored_nan_refilled_is_a_footprint_fill_not_a_revision(tmp_path, caplog):
    store, clean, full, today = _seed_with_footprint_nans(tmp_path)
    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH, adjustment="returns_invariant")
    logger = logging.getLogger("test.tailrecompute")
    with caplog.at_level(logging.WARNING):
        result = tail_recompute(
            store, _key(), factor, recompute=clean, today=today,
            horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=_fp(),
            logger=logger,
        )
    # a stored NaN asserts no value, so refilling it is NOT a revision ...
    assert result.revision_detected is False
    assert result.mismatched_symbols == ()
    assert result.recolumned_symbols == ()
    # ... it is counted as a footprint fill ...
    assert any("filled_after_footprint: 1 symbol(s)" in n for n in result.notes)
    assert not any("REVISION detected" in r.message for r in caplog.records)
    # ... and the recomputed values are absorbed (the store now matches batch).
    assert _nan_aware_equal(store.read(_key()), full)


def test_finite_to_nan_is_still_a_loud_revision(tmp_path, caplog):
    dates = _business_dates(30)
    raw = _raw_panel(dates)
    clean = _make_recompute(raw)
    today = dates[-1]
    full = clean(None, today, _TRANSITIVE_DEPTH)
    store = FactorValueStore(tmp_path)
    cutoff = dates[-6]
    store.write(_key(), full[full.index.get_level_values("date") <= cutoff], fingerprint=_fp())

    def nan_out(emit_start, end, warmup):
        s = clean(emit_start, end, warmup).copy()
        s[s.index.get_level_values("symbol") == "AAA"] = np.nan
        return s

    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH, adjustment="returns_invariant")
    with caplog.at_level(logging.WARNING):
        result = tail_recompute(
            store, _key(), factor, recompute=nan_out, today=today,
            horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=_fp(),
            logger=logging.getLogger("test.tailrecompute"),
        )
    # a stored FINITE value contradicted (here: erased to NaN) IS a revision.
    assert "AAA" in result.mismatched_symbols
    assert result.revision_detected is True
    assert any("REVISION detected" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# cross-sectional intermediate payload: explicitly NOT WIRED (D5 C4 deferral)
# --------------------------------------------------------------------------- #
def test_intermediate_payload_raises_the_named_not_wired_error(tmp_path):
    from factors.store.incremental import IntermediatePayloadNotWiredError

    dates = _business_dates(10)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"])
    frame = pd.DataFrame(
        {"stat_a": np.arange(len(idx), dtype=float), "stat_b": 1.0}, index=idx
    )
    store = FactorValueStore(tmp_path)
    store.write_frame(_key(), frame, fingerprint=_fp())

    factor = _make_factor(lookback_depth=_TRANSITIVE_DEPTH)
    recompute = _make_recompute(_raw_panel(dates))
    with pytest.raises(IntermediatePayloadNotWiredError, match="NOT") as excinfo:
        tail_recompute(
            store, _key(), factor, recompute=recompute, today=dates[-1],
            horizon_cfg=CacheHorizonConfig(refresh_recent_days=1), fingerprint=_fp(),
        )
    msg = str(excinfo.value)
    assert "per-symbol INTERMEDIATE" in msg
    assert "deliberate deferral" in msg
    # a ValueError subclass, so any legacy ``except ValueError`` still catches it.
    assert isinstance(excinfo.value, ValueError)
