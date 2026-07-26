"""D4c: the value store must not depend on the universe it was filled with.

Network-free. Two INDEPENDENT defects lived here and are fixed differently, so
they are tested separately (design revision A2):

* MODE 1 — silent cross-section shrink (EVERY factor, nothing to do with
  cross-sections). The read-through gap criterion asked only which requested
  DATES the store lacked, so a second batch of symbols over already-covered dates
  found "no gap" and was served without them: measured on this fixture before the
  fix, ``12 names filled -> 24 asked`` returned 24 rows instead of 48, dropped 12
  names, and made ZERO provider calls. Fixed by asking per (date, SYMBOL).
* MODE 2 — a cross-sectional value polluted by the filling universe.
  ``intraday_amp_cut``'s step 4 z-scores each date ACROSS the universe, so its
  value is a function of who else was loaded, while the store key
  (factor, params, code, view) says nothing about that. Measured before the fix:
  the 24 surviving cells ALL differed from the truth, max|delta| 0.194447. Fixed
  by storing the UNIVERSE-INDEPENDENT per-symbol intermediate and running the
  combine at read-assembly, over the universe the reader asked for. A universe
  KEY DIMENSION was rejected (A2): it would over-invalidate the ten factors whose
  values are universe-free, and a PIT universe is a time-varying set that such a
  dimension could not even hold.

Every invariance claim below has recorded mutation evidence, and each mutation
was first asserted to actually change its target.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.availability_policy import OvernightBoundary, View
from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors import service as service_mod
from factors.compute.minute.binding import (
    _MINUTE_STREAM_BINDINGS,
    minute_intermediate_columns,
    minute_stats_from_bars,
)
from factors.compute.minute.intraday_amp_cut import (
    AMP_CUT_MIN_CROSS_SECTION,
    V_MEAN_COL,
    V_STD_COL,
    IntradayAmpCutFactor,
)
from factors.materialize import (
    MaterializeSources,
    materialize_intermediate_range,
    materialize_range,
    payload_columns,
    stores_intermediate,
)
from factors.service import DecisionPoint, panel
from factors.store.fingerprint import data_fingerprint
from factors.store.keys import store_key
from factors.store.values import FactorValueStore

#: 24 names = twice ``AMP_CUT_MIN_CROSS_SECTION``, so both halves of the split
#: below can pass the date-wise gate on their own and the contrast is about the
#: universe, not about the gate.
N_SYMBOLS = 24
SYMBOLS = [f"6000{i:02d}.SH" for i in range(N_SYMBOLS)]
HALF = SYMBOLS[:12]
DATES = pd.bdate_range("2021-01-04", periods=32)
EMIT = list(DATES[24:26])  # 2 dates x 24 names = 48 requested cells
BARS_PER_DAY = 238

#: The cross-sectional factor (the only one) and two controls: a per-symbol-pure
#: minute factor and a daily factor. Mode 1 is not about cross-sections, so it is
#: asserted on all three.
CROSS_SECTIONAL_ID = "intraday_amp_cut_10"
PURE_MINUTE_ID = "volume_peak_count_20"
DAILY_ID = "momentum_20"
#: A factor whose per-day gates reject many days, so it emits NO row for many of
#: the cells it is asked about (measured on real bars: 44-45% of the requested
#: cells over 24 CSI500 names x March 2023). That is the case where "no stored row"
#: has to be distinguishable from "never computed".
SPARSE_MINUTE_ID = "ridge_minute_return_20"


def _minute() -> pd.DataFrame:
    rng = np.random.RandomState(5)
    rows: list[tuple] = []
    for si, s in enumerate(SYMBOLS):
        for d in DATES:
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 3 + rng.normal(0, 2)
            for i in range(BARS_PER_DAY):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


def _daily() -> pd.DataFrame:
    rng = np.random.RandomState(3)
    rows = []
    for si, s in enumerate(SYMBOLS):
        px = 100.0 + si * 2 + np.cumsum(rng.normal(0, 1.0, len(DATES)))
        for d, p in zip(DATES, px):
            rows.append((d, s, p - 0.3, p + 0.5, p - 0.5, p, 1e5, p * 1e5))
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return pd.DataFrame(rows, columns=cols).set_index(["date", "symbol"]).sort_index()


MINUTE = _minute()
DAILY = _daily()


class MinuteProv:
    """Honours ``symbols`` (like the cache reader) and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def minute_bars(self, symbols, start, end):
        self.calls += 1
        if not symbols:
            return MINUTE.iloc[0:0]
        t = MINUTE.index.get_level_values("time")
        sym = MINUTE.index.get_level_values("symbol")
        keep = (
            (t >= pd.Timestamp(start))
            & (t <= pd.Timestamp(end))
            & sym.isin(list(symbols))
        )
        return MINUTE[keep]

    def earliest_available(self, symbols):
        return DATES[0]


class DailyProv:
    def __init__(self) -> None:
        self.calls = 0

    def daily_panel(self, symbols, start, end):
        self.calls += 1
        d = DAILY.index.get_level_values("date")
        sym = DAILY.index.get_level_values("symbol")
        return DAILY[
            (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)) & sym.isin(list(symbols))
        ]


def _sources(fid):
    if fid == DAILY_ID:
        return MaterializeSources(daily=DailyProv())
    return MaterializeSources(minute=MinuteProv())


def _provider(src):
    return src.daily if src.minute is None else src.minute


def _ask(store, fid, universe, src=None, dates=None):
    src = src or _sources(fid)
    got = panel(
        [fid], universe, [DecisionPoint(date=d) for d in (dates or EMIT)],
        store=store, sources=src,
    )
    return got[fid], src


def _fresh_reference(fid, universe, dates=None):
    """What a COLD store answers for the same request — the truth to match."""
    with tempfile.TemporaryDirectory() as td:
        series, _ = _ask(FactorValueStore(td), fid, universe, dates=dates)
        return series.sort_index()


def _names(series) -> set[str]:
    return set(map(str, series.index.get_level_values("symbol")))


def _assert_matches_reference(got, ref, fid):
    got, ref = got.sort_index(), ref.sort_index()
    assert got.index.equals(ref.index), f"{fid}: index differs from a cold store's"
    a, b = got.to_numpy(), ref.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{fid}: NaN mask differs"
    finite = ~np.isnan(a)
    assert finite.sum() > 0, f"{fid}: vacuous (no finite values to compare)"
    assert np.array_equal(a[finite], b[finite]), f"{fid}: values differ from a cold store's"


# --------------------------------------------------------------------------- #
# MODE 1 — the (date, symbol) gap criterion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID, DAILY_ID])
def test_second_symbol_batch_is_filled_not_silently_dropped(fid):
    """12 names filled -> 24 asked on the SAME store must give 24 names / 48 rows.

    Measured BEFORE the fix (all three factors): 24 rows, 12 names, ZERO provider
    calls — the second batch was dropped without a recompute or a word.
    """
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        src = _sources(fid)
        _ask(store, fid, HALF, src=src)
        calls_after_fill = _provider(src).calls
        got, _ = _ask(store, fid, SYMBOLS, src=src)

    assert _names(got) == set(SYMBOLS), f"{fid}: names missing from the served panel"
    assert len(got) == len(EMIT) * N_SYMBOLS
    assert _provider(src).calls > calls_after_fill, (
        f"{fid}: the second batch must trigger a fill, not a silent hit"
    )
    _assert_matches_reference(got, _fresh_reference(fid, SYMBOLS), fid)


def test_only_the_missing_symbols_are_refilled():
    """The fill covers the names that need one — not the whole universe again.

    The provider is asked for one symbol at a time (D4b streaming), so the call
    count IS the number of refilled names.
    """
    fid = PURE_MINUTE_ID
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        src = _sources(fid)
        _ask(store, fid, HALF, src=src)
        after_fill = _provider(src).calls
        _ask(store, fid, SYMBOLS, src=src)
        refills = _provider(src).calls - after_fill
    assert refills == N_SYMBOLS - len(HALF) == 12


@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID, DAILY_ID])
def test_date_only_gap_criterion_reintroduces_the_silent_drop(fid, monkeypatch):
    """MUTATION for mode 1: reverting the criterion to DATES ONLY drops the names.

    The mutation is the pre-fix line (``dates[~dates.isin(have_dates)]``) restored
    with the symbol half removed. Asserting the drop HAPPENS is what proves the
    (date, symbol) criterion is load-bearing — with it, the test above sees 24
    names; with the mutation this test sees 12 and no fill.

    Non-vacuity is asserted first: the store really was filled with the 12, so
    "no names" cannot make this pass for the wrong reason.
    """
    def date_only(stored, dates, symbols, *, force_all):
        if force_all or stored is None or stored.empty:
            return dates, list(symbols)
        have = pd.DatetimeIndex(pd.unique(stored.index.get_level_values("date")))
        missing = dates[~dates.isin(have)]
        return missing, (list(symbols) if len(missing) else [])

    monkeypatch.setattr(service_mod, "_missing_request", date_only)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        src = _sources(fid)
        _ask(store, fid, HALF, src=src)
        after_fill = _provider(src).calls
        assert after_fill > 0, "vacuous: the first fill did nothing"
        got, _ = _ask(store, fid, SYMBOLS, src=src)
    assert _names(got) == set(HALF), (
        f"{fid}: the date-only criterion should serve only the filled names"
    )
    assert _provider(src).calls == after_fill, "the date-only criterion made no fill"


@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID, DAILY_ID, SPARSE_MINUTE_ID])
def test_warm_identical_request_still_hits_the_store(fid):
    """The strict criterion must not defeat the store: a repeat of the SAME request
    makes no provider call at all — INCLUDING for a factor that emits nothing for
    many of the cells it is asked about.

    That last case is the one the criterion could have broken, and it is why a fill
    records the cells it covered (:func:`service._record_fill_footprint`): without
    it, "no row" reads as "never computed" and every request re-materializes.
    """
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        src = _sources(fid)
        _ask(store, fid, SYMBOLS, src=src)
        after_fill = _provider(src).calls
        assert after_fill > 0, f"{fid}: vacuous (the cold fill read nothing)"
        _ask(store, fid, SYMBOLS, src=src)
        assert _provider(src).calls == after_fill, f"{fid}: warm request re-filled"


def test_a_fill_records_the_cells_it_covered_even_when_empty():
    """Absent from the store must mean NEVER COMPUTED, so a fill writes a row for
    every cell it covered — NaN where the factor emits nothing.

    Non-vacuity first: the sparse factor really does leave many of these cells
    without a value on this fixture (asserted, not assumed), which is what makes
    the property observable at all.

    MUTATION (run): making ``_record_fill_footprint`` return the materialized frame
    unchanged -> the warm-hit test above FAILS for the sparse factor (rc=1) and this
    test's coverage assertion fails; restored -> both pass (rc=0).
    """
    fid = SPARSE_MINUTE_ID
    factor = factor_registry.build(fid)
    key = store_key(factor, view=View.DECISION.value)
    grid = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(EMIT), SYMBOLS], names=["date", "symbol"]
    )
    # NON-VACUITY, measured on the engine itself rather than assumed: the factor
    # really does emit NO row for most of these cells (12 of 48 on this fixture),
    # which is the only thing that makes the footprint observable.
    emitted = materialize_range(
        factor, view=View.DECISION, symbols=SYMBOLS, emit_start=EMIT[0],
        emit_end=EMIT[-1], sources=MaterializeSources(minute=MinuteProv()),
    )
    assert 0 < len(emitted) < len(grid), (
        f"vacuous: the factor emits {len(emitted)} rows for {len(grid)} requested "
        f"cells — this test needs SOME but not all"
    )
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _ask(store, fid, SYMBOLS)
        stored = store.read_frame(key)
    assert grid.difference(stored.index).empty, (
        "the fill left requested cells with no row — they would be re-materialized "
        "on every later request"
    )


# --------------------------------------------------------------------------- #
# MODE 2 — the cross-sectional value and its universe
# --------------------------------------------------------------------------- #
def test_cross_universe_reuse_matches_a_cold_store_both_directions():
    """The polluting direction (12 -> 24) AND the polluted direction (24 -> 12).

    The reverse direction is the one a date-only reader would call a clean hit:
    every requested cell IS in the store, so nothing is refilled — and before the
    fix all 24 served cells were still wrong (max|delta| 0.194447), because they
    had been z-scored across 24 names and were being served to a 12-name request.
    """
    fid = CROSS_SECTIONAL_ID
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _ask(store, fid, HALF)
        widened, _ = _ask(store, fid, SYMBOLS)
    _assert_matches_reference(widened, _fresh_reference(fid, SYMBOLS), f"{fid} 12->24")

    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        src = _sources(fid)
        _ask(store, fid, SYMBOLS, src=src)
        after_fill = _provider(src).calls
        narrowed, _ = _ask(store, fid, HALF, src=src)
        assert _provider(src).calls == after_fill, "the narrowing request refilled"
    _assert_matches_reference(narrowed, _fresh_reference(fid, HALF), f"{fid} 24->12")


def test_stored_intermediate_is_universe_independent():
    """THE PREMISE of the mode-2 fix, asserted rather than assumed.

    The same symbol's stored intermediate rows are BIT-IDENTICAL whether the store
    was filled with 12 names or 24. If this ever fails, storing the intermediate is
    no better than storing the value and the whole approach is void.
    """
    fid = CROSS_SECTIONAL_ID
    factor = factor_registry.build(fid)
    key = store_key(factor, view=View.DECISION.value)
    frames = {}
    for label, universe in (("half", HALF), ("full", SYMBOLS)):
        with tempfile.TemporaryDirectory() as td:
            store = FactorValueStore(td)
            _ask(store, fid, universe)
            frames[label] = store.read_frame(key)

    half, full = frames["half"], frames["full"]
    assert list(half.columns) == [V_MEAN_COL, V_STD_COL]
    shared = half.index.intersection(full.index)
    assert len(shared) > 0, "vacuous: no shared (date, symbol) rows"
    assert set(map(str, shared.get_level_values("symbol"))) == set(HALF)
    a = half.loc[shared].to_numpy()
    b = full.loc[shared].to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), (
        "the per-symbol intermediate moved with the universe — it is not universe-free"
    )


def test_min_cross_section_gate_still_bites_on_the_read_path():
    """``AMP_CUT_MIN_CROSS_SECTION`` (=10) is a DEFINITION constant and now gates at
    read-assembly: 9 names -> every value NaN, 10 names -> finite values.

    Asserted from a store warmed with all 24 names, i.e. exactly where a relaxed
    gate would be tempting (the intermediate for 24 names IS in the store, and
    serving 9 of them from a 24-name cross-section is the defect this replaced).
    """
    assert AMP_CUT_MIN_CROSS_SECTION == 10  # never relaxed to accommodate the store
    fid = CROSS_SECTIONAL_ID
    for n, want_finite in ((AMP_CUT_MIN_CROSS_SECTION - 1, False),
                           (AMP_CUT_MIN_CROSS_SECTION, True)):
        with tempfile.TemporaryDirectory() as td:
            store = FactorValueStore(td)
            _ask(store, fid, SYMBOLS)          # warm with the FULL universe
            got, _ = _ask(store, fid, SYMBOLS[:n])
        finite = int(np.isfinite(got.to_numpy()).sum())
        assert len(got) == len(EMIT) * n, f"n={n}: wrong row count"
        if want_finite:
            assert finite == len(got), f"n={n}: the gate should pass, got {finite} finite"
        else:
            assert finite == 0, f"n={n}: below the gate, got {finite} finite values"


def test_the_cross_sectional_combine_runs_once_per_request():
    """The combine is a per-request read-assembly step, not a per-symbol one.

    (It is also what makes the served value a function of the REQUESTED universe;
    running it per symbol would hand it a one-name cross-section, which the gate
    turns entirely into NaN.)
    """
    calls: list[int] = []
    real = service_mod.combine_minute_stats

    def counted(factor, stats):
        calls.append(len(stats))
        return real(factor, stats)

    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(service_mod, "combine_minute_stats", counted)
            _ask(store, CROSS_SECTIONAL_ID, SYMBOLS)
    assert len(calls) == 1, f"expected ONE combine per request, got {len(calls)}"
    assert calls[0] == len(EMIT) * N_SYMBOLS, "the combine did not see the whole request"


# --------------------------------------------------------------------------- #
# Payload shape: what each kind of factor stores
# --------------------------------------------------------------------------- #
def test_only_the_cross_sectional_factor_stores_an_intermediate():
    """The other ten minute factors and the daily factors are NOT dragged into the
    two-stage form — their values are universe-free, so they store their value."""
    for cls in _MINUTE_STREAM_BINDINGS:
        factor = factor_registry.build(cls().name)
        expect = cls is IntradayAmpCutFactor
        assert stores_intermediate(factor) is expect, factor.name
        assert payload_columns(factor) == (
            (V_MEAN_COL, V_STD_COL) if expect else (factor.name,)
        ), factor.name
    for fid in (DAILY_ID, "volatility_20", "value_ep"):
        factor = factor_registry.build(fid)
        assert stores_intermediate(factor) is False, fid
        assert payload_columns(factor) == (factor.name,), fid


@pytest.mark.parametrize("fid", [PURE_MINUTE_ID, DAILY_ID])
def test_value_factors_keep_the_single_value_column_artifact(fid):
    """On-disk shape of a value factor is unchanged: date, symbol, {factor_id}."""
    factor = factor_registry.build(fid)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _ask(store, fid, SYMBOLS)
        names = pq.read_schema(store.path(key)).names
        series = store.read(key)  # the Series API still serves them
    assert names == ["date", "symbol", fid]
    assert series is not None and series.name == fid


def test_cross_sectional_artifact_carries_the_intermediate_and_refuses_the_series_api():
    """The amp_cut artifact holds (v_mean, v_std); reading it as a value Series is a
    readable error, not a silent pick of one column."""
    factor = factor_registry.build(CROSS_SECTIONAL_ID)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _ask(store, CROSS_SECTIONAL_ID, SYMBOLS)
        assert pq.read_schema(store.path(key)).names == ["date", "symbol", V_MEAN_COL, V_STD_COL]
        with pytest.raises(ValueError, match="single value column"):
            store.read(key)


def _stale_value_artifact(store, key, factor):
    """Write pre-D4c VALUE rows under the cross-sectional factor's key.

    The fingerprint is the CURRENT one on purpose: a stale artifact that also has
    a stale fingerprint is voided by the fingerprint check and proves nothing about
    the payload-shape check. (Measured: with a bogus ``schema_version`` this
    fixture kept passing with the shape check removed — the test was passing for
    the wrong reason, so the fingerprint was made valid to leave the shape check as
    the ONLY thing that can void it.)
    """
    fingerprint = data_fingerprint(adjustment=factor.spec.adjustment)
    stale = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [(EMIT[0], SYMBOLS[0]), (EMIT[0], SYMBOLS[1])], names=["date", "symbol"]
        ),
        name=factor.name,
    )
    store.write(key, stale, fingerprint=fingerprint)
    return fingerprint


def test_a_stale_payload_shape_is_a_read_miss():
    """An artifact holding the OTHER payload shape is a MISS, never a partial read.

    The store key does not change with the payload shape — ``code_hash`` covers the
    factor module and its shared set, not the binding that decides value-vs-
    intermediate — so this check is what stops pre-D4c value rows from being read
    as an intermediate (and, with the same one line, an intermediate from being read
    as a value).

    MUTATION (run): removing the column comparison in ``read_valid_frame`` -> this
    test FAILS (rc=1); restored -> passes (rc=0).
    """
    factor = factor_registry.build(CROSS_SECTIONAL_ID)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        fingerprint = _stale_value_artifact(store, key, factor)
        assert store.read_frame(key) is not None  # the stale artifact IS on disk
        assert store.read_valid_frame(
            key, expected_fingerprint=fingerprint, columns=payload_columns(factor)
        ) is None, "value rows were served as an intermediate"
        # ... and the mirror: an intermediate must not be served as a value.
        assert store.read_valid_frame(
            key, expected_fingerprint=fingerprint, columns=(V_MEAN_COL, V_STD_COL, "z")
        ) is None


def test_a_stale_payload_shape_is_overwritten_not_merged():
    """The refill must REPLACE a differently-shaped artifact, not concatenate onto it.

    Merging would build a three-value-column frame, each half NaN where the other
    half's columns are — an artifact that is neither shape and that the reader would
    then reject wholesale on every future request.

    MUTATION (run): removing the column comparison in ``upsert_frame`` -> this test
    FAILS (rc=1) with the artifact carrying
    ``[date, symbol, intraday_amp_cut_10, v_mean, v_std]``; restored -> passes.
    """
    factor = factor_registry.build(CROSS_SECTIONAL_ID)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _stale_value_artifact(store, key, factor)
        got, _ = _ask(store, CROSS_SECTIONAL_ID, SYMBOLS)
        assert pq.read_schema(store.path(key)).names == [
            "date", "symbol", V_MEAN_COL, V_STD_COL
        ], "the stale value rows were merged into the intermediate artifact"
    _assert_matches_reference(got, _fresh_reference(CROSS_SECTIONAL_ID, SYMBOLS), "stale")


@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID, SPARSE_MINUTE_ID])
def test_served_panel_matches_the_direct_engine_and_adds_only_nan_rows(fid):
    """Going through the store must not change a value, and may only ADD empty rows.

    The direct engine (``materialize_range``, no store at all) is the independent
    reference. Every cell it emits must come back identical; the rows the served
    panel has on top of that are the footprint cells, which must ALL be NaN — a
    finite value appearing where the engine emits nothing would mean the store had
    invented one. This is the disclosed shape change of D4c: a request for a cell a
    factor never emits now comes back as an explicit NaN row instead of no row.
    """
    factor = factor_registry.build(fid)
    with tempfile.TemporaryDirectory() as td:
        served, _ = _ask(FactorValueStore(td), fid, SYMBOLS)
    direct = materialize_range(
        factor, view=View.DECISION, symbols=SYMBOLS, emit_start=EMIT[0],
        emit_end=EMIT[-1], sources=MaterializeSources(minute=MinuteProv()),
    ).sort_index()
    assert direct.index.difference(served.index).empty, "the store lost engine rows"
    aligned = served.reindex(direct.index).to_numpy()
    d = direct.to_numpy()
    assert np.array_equal(np.isnan(aligned), np.isnan(d)), f"{fid}: NaN mask moved"
    finite = ~np.isnan(d)
    assert finite.sum() > 0, f"{fid}: vacuous (the engine emitted no finite value)"
    assert np.array_equal(aligned[finite], d[finite]), f"{fid}: values moved"
    extra = served.index.difference(direct.index)
    assert np.isnan(served.reindex(extra).to_numpy()).all(), (
        f"{fid}: the store served a finite value where the engine emits no row"
    )


@pytest.mark.parametrize("fid", [cls().name for cls in _MINUTE_STREAM_BINDINGS])
def test_declared_intermediate_columns_match_what_per_symbol_produces(fid):
    """The declaration the store validates against is checked against the frame the
    per-symbol stage actually returns — for EVERY bound minute factor, not just the
    one that uses it today.

    MUTATION (run): declaring ``intermediate_columns=("value","extra")`` on the
    amp_cut binding -> this test FAILS for intraday_amp_cut_10 (rc=1); restored ->
    passes (rc=0).
    """
    factor = factor_registry.build(fid)
    bars = MINUTE[MINUTE.index.get_level_values("symbol") == SYMBOLS[0]]
    produced = tuple(str(c) for c in minute_stats_from_bars(factor, bars).columns)
    assert minute_intermediate_columns(factor) == produced


def test_intermediate_entry_point_refuses_a_value_factor():
    """Asking for the intermediate of a factor whose value IS the intermediate is a
    readable error: one thing must not have two names."""
    factor = factor_registry.build(PURE_MINUTE_ID)
    with pytest.raises(ValueError, match="stores its VALUE"):
        materialize_intermediate_range(
            factor, view=View.DECISION, symbols=SYMBOLS, emit_start=EMIT[0],
            emit_end=EMIT[-1], sources=MaterializeSources(minute=MinuteProv()),
        )


def test_intermediate_entry_point_refuses_a_masked_factor(monkeypatch):
    """A masked cross-sectional factor is refused, not silently served unmasked.

    The ex-date mask applies to the VALUE, which the intermediate does not carry.
    No closing factor is in this combination, which is exactly why the branch needs
    a test: an unreachable raise is the next AttributeError's host.
    """
    factor = factor_registry.build(CROSS_SECTIONAL_ID)
    real_spec = type(factor).spec

    class _Masked:
        def __get__(self, obj, owner=None):
            spec = real_spec.__get__(obj, owner)
            object.__setattr__(spec, "overnight_boundary", OvernightBoundary.MASKED)
            return spec

    monkeypatch.setattr(type(factor), "spec", _Masked())
    assert factor.spec.overnight_boundary is OvernightBoundary.MASKED  # mutation landed
    with pytest.raises(NotImplementedError, match="masked"):
        materialize_intermediate_range(
            factor, view=View.DECISION, symbols=SYMBOLS, emit_start=EMIT[0],
            emit_end=EMIT[-1], sources=MaterializeSources(minute=MinuteProv()),
        )


# --------------------------------------------------------------------------- #
# P8 for the newly stored payload
# --------------------------------------------------------------------------- #
def test_single_fill_equals_batch_fill_for_the_stored_intermediate():
    """§3.5 P8 on the cross-sectional factor's NEW payload: a store filled one date
    at a time and a store filled in one batch are bit-identical.

    P8 could not be asserted for this factor's stored artifact before — what was
    stored was a universe-dependent value, so "the same store" was not a
    well-defined thing to compare.
    """
    fid = CROSS_SECTIONAL_ID
    factor = factor_registry.build(fid)
    key = store_key(factor, view=View.DECISION.value)
    dates = list(DATES[24:30])
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a, b = FactorValueStore(ta), FactorValueStore(tb)
        for d in dates:
            _ask(a, fid, SYMBOLS, dates=[d])
        _ask(b, fid, SYMBOLS, dates=dates)
        fa, fb = a.read_frame(key), b.read_frame(key)
    assert fa.index.equals(fb.index)
    assert list(fa.columns) == list(fb.columns) == [V_MEAN_COL, V_STD_COL]
    x, y = fa.to_numpy(), fb.to_numpy()
    assert np.array_equal(np.isnan(x), np.isnan(y))
    finite = ~np.isnan(x)
    assert finite.sum() > 0, "vacuous: no finite intermediate rows"
    assert np.array_equal(x[finite], y[finite])
