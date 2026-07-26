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

import inspect
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.availability_policy import OvernightBoundary, View
from data.clean.intraday_schema import normalize_intraday_bars
from factors import materialize as materialize_mod
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
    make_recompute_fn,
    materialize_intermediate_range,
    materialize_range,
    payload_columns,
    requested_universe,
    stores_intermediate,
)
from factors.service import DecisionPoint, cross_section, panel
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
# The caller's symbol list is normalized ONCE, at every entry point
# --------------------------------------------------------------------------- #
#: 14 unique names with two of them repeated: what a caller passes when a
#: universe is assembled from overlapping sources. The clean grid is
#: len(EMIT) x 14; anything larger means a repeat was materialized twice.
_DUP_CLEAN = SYMBOLS[:14]
_DUP_REQUEST = _DUP_CLEAN + [_DUP_CLEAN[0], _DUP_CLEAN[5]]


def _symbol_entry_points() -> dict[str, list[str]]:
    """Public callables of the two modules that take a CALLER-SUPPLIED symbol list.

    DERIVED by signature inspection, never hand-listed: a guard that only checks
    the entry points someone remembered to write down can only confirm what they
    already knew (#82). The provider Protocols also mention ``symbols``, but they
    are the OTHER side of the boundary — the engine hands them an already
    normalized list — and they fall out on their own because a Protocol's call
    signature is ``(*args, **kwargs)``.
    """
    out: dict[str, list[str]] = {}
    for mod in (service_mod, materialize_mod):
        for name, obj in vars(mod).items():
            if name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            try:
                params = list(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                continue
            hits = [p for p in params if p in ("symbols", "universe")]
            if hits:
                out[f"{mod.__name__}.{name}"] = hits
    return out


def _probe_panel(universe):
    with tempfile.TemporaryDirectory() as td:
        return panel(
            [PURE_MINUTE_ID], universe, [DecisionPoint(date=d) for d in EMIT],
            store=FactorValueStore(td), sources=_sources(PURE_MINUTE_ID),
        )


def _probe_cross_section(universe):
    with tempfile.TemporaryDirectory() as td:
        return cross_section(
            [PURE_MINUTE_ID], universe, DecisionPoint(date=EMIT[0]),
            store=FactorValueStore(td), sources=_sources(PURE_MINUTE_ID),
        )


def _probe_materialize_range(universe):
    return materialize_range(
        factor_registry.build(PURE_MINUTE_ID), view=View.DECISION, symbols=universe,
        emit_start=EMIT[0], emit_end=EMIT[-1],
        sources=MaterializeSources(minute=MinuteProv()),
    )


def _probe_materialize_intermediate_range(universe):
    return materialize_intermediate_range(
        factor_registry.build(CROSS_SECTIONAL_ID), view=View.DECISION, symbols=universe,
        emit_start=EMIT[0], emit_end=EMIT[-1],
        sources=MaterializeSources(minute=MinuteProv()),
    )


def _probe_make_recompute_fn(universe):
    fn = make_recompute_fn(
        factor_registry.build(PURE_MINUTE_ID), view=View.DECISION, symbols=universe,
        sources=MaterializeSources(minute=MinuteProv()), data_start=DATES[0],
    )
    return fn(EMIT[0], EMIT[-1], 20)


def _probe_requested_universe(universe):
    return pd.Series(0.0, index=pd.Index(requested_universe(universe), name="symbol"))


#: entry point -> how to call it with a caller list. A new entry point without a
#: recipe fails the surface test below; the recipes cannot be derived (each
#: signature differs), but WHICH ones must exist is.
_ENTRY_PROBES = {
    "factors.service.panel": _probe_panel,
    "factors.service.cross_section": _probe_cross_section,
    "factors.materialize.materialize_range": _probe_materialize_range,
    "factors.materialize.materialize_intermediate_range": _probe_materialize_intermediate_range,
    "factors.materialize.make_recompute_fn": _probe_make_recompute_fn,
    "factors.materialize.requested_universe": _probe_requested_universe,
}


def test_the_symbol_entry_point_surface_is_fully_probed():
    """Every entry point taking a caller symbol list has a duplicate probe.

    This is the half that cannot be forgotten: the surface is discovered by
    inspection, so adding a function that accepts a universe and forgetting to
    normalize it fails HERE, before anyone has to think of testing it.
    """
    found = _symbol_entry_points()
    assert set(found) == set(_ENTRY_PROBES), (
        f"unprobed symbol entry points: {sorted(set(found) - set(_ENTRY_PROBES))}; "
        f"stale probes: {sorted(set(_ENTRY_PROBES) - set(found))}"
    )


@pytest.mark.parametrize("entry", sorted(_ENTRY_PROBES))
def test_every_symbol_entry_point_normalizes_the_caller_list(entry):
    """A repeated name must not produce a repeated row, ANYWHERE.

    Measured before the fix, with the service building its fill footprint from
    the raw caller list: the store held 48 rows instead of 42 with 6 duplicated
    index entries, and ``intraday_amp_cut_10`` moved all 42 cells (max|delta|
    1.526e-01) because the repeat entered its date's cross-section twice.

    MUTATION (run): reverting ``service.panel`` to ``[str(s) for s in universe]``
    -> this test FAILS for both service entries (rc=1); restored -> passes.
    """
    probe = _ENTRY_PROBES[entry]
    duped = probe(_DUP_REQUEST)
    clean = probe(_DUP_CLEAN)
    assert not duped.index.duplicated().any(), f"{entry}: duplicated index rows"
    assert duped.index.equals(clean.index), f"{entry}: index differs from the clean call"
    a, b = duped.to_numpy(), clean.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{entry}: NaN mask differs"
    finite = ~np.isnan(a)
    assert np.array_equal(a[finite], b[finite]), f"{entry}: values differ"


@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID])
def test_a_duplicate_in_one_request_does_not_pollute_the_store(fid):
    """And the pollution must not be PERSISTENT: the artifact a duplicated request
    leaves behind is what every later request reads.

    Before the fix the later CLEAN request read 48 rows with 6 duplicated index
    entries — enough to make a consumer's ``reindex`` raise outright — and for the
    cross-sectional factor every one of its 42 cells was wrong.
    """
    factor = factor_registry.build(fid)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        reference, _ = _ask(FactorValueStore(td), fid, _DUP_CLEAN)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        _ask(store, fid, _DUP_REQUEST)
        on_disk = store.read_frame(key)
        later, _ = _ask(store, fid, _DUP_CLEAN)  # a completely clean later request
    assert not on_disk.index.duplicated().any(), "duplicated rows persisted to disk"
    assert len(on_disk) == len(EMIT) * len(_DUP_CLEAN)
    _assert_matches_reference(later, reference, f"{fid} after a duplicated request")


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

    THE ``finite.sum() > 0`` GUARD IS LOAD-BEARING and was missing at first: two
    all-NaN frames have equal NaN masks and equal (empty) finite values, so the
    comparison is satisfied by an intermediate carrying no information at all.
    MUTATION (run before the guard existed): multiplying ``_amp_cut_per_symbol``'s
    frame by NaN — same shape, same columns, same index — left this test PASSING
    (rc=0); with the guard it FAILS. That is the eighth "impossible to fail" test
    caught in this repo, and it was sitting under the words "THE PREMISE".
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
    finite = ~np.isnan(a)
    assert finite.sum() > 0, (
        "vacuous: the intermediate is all-NaN, so equality here would be satisfied "
        "by a stage that computed nothing"
    )
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[finite], b[~np.isnan(b)]), (
        "the per-symbol intermediate moved with the universe — it is not universe-free"
    )


class _StaggeredProv:
    """Same bars, but each symbol's history starts on its own date, and
    ``earliest_available`` answers a symbol LIST in one of three ways."""

    #: first 12 names have full history; the rest start 16 trading days in, so a
    #: min-over-symbols floor and a max-over-symbols floor are far apart.
    FIRST = {s: (DATES[0] if i < 12 else DATES[16]) for i, s in enumerate(SYMBOLS)}

    def __init__(self, mode: str) -> None:
        self.mode = mode
        d = pd.DatetimeIndex(MINUTE.index.get_level_values("time")).normalize()
        s = pd.Index(MINUTE.index.get_level_values("symbol"))
        keep = np.array([day >= self.FIRST[str(sym)] for day, sym in zip(d, s)])
        self._bars = MINUTE[keep]

    def minute_bars(self, symbols, start, end):
        if not symbols:
            return self._bars.iloc[0:0]
        t = self._bars.index.get_level_values("time")
        sym = self._bars.index.get_level_values("symbol")
        return self._bars[
            (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & sym.isin(list(symbols))
        ]

    def earliest_available(self, symbols):
        firsts = [self.FIRST[str(s)] for s in symbols if str(s) in self.FIRST]
        if not firsts:
            return DATES[0]
        if self.mode == "min":
            return min(firsts)  # a LOWER bound over the symbols (the contract)
        if self.mode == "constant":
            return DATES[0]  # a documented global constant (also a lower bound)
        if self.mode == "max":
            return max(firsts)  # "the floor covering ALL of them" — NOT a lower bound
        raise ValueError(self.mode)


def _stored_intermediate(universe, mode):
    factor = factor_registry.build(CROSS_SECTIONAL_ID)
    key = store_key(factor, view=View.DECISION.value)
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        panel(
            [CROSS_SECTIONAL_ID], universe, [DecisionPoint(date=d) for d in EMIT],
            store=store, sources=MaterializeSources(minute=_StaggeredProv(mode)),
        )
        return store.read_frame(key)


def _intermediate_universe_gap(mode) -> float:
    half = _stored_intermediate(HALF, mode)
    full = _stored_intermediate(SYMBOLS, mode)
    shared = half.index.intersection(full.index)
    assert len(shared) > 0, "vacuous: no shared rows to compare"
    a, b = half.loc[shared].to_numpy(), full.loc[shared].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0, "vacuous: no finite pairs to compare"
    return float(np.max(np.abs(a[both] - b[both])))


@pytest.mark.parametrize("mode", ["min", "constant"])
def test_a_lower_bound_floor_keeps_the_intermediate_universe_free(mode):
    """The premise holds for ANY floor that is a lower bound over the symbols.

    ``earliest_available`` is the only universe-dependent input the per-symbol
    stage has, so this is where universe-independence is actually decided (see the
    contract on ``MinuteBarProvider.earliest_available``).
    """
    assert _intermediate_universe_gap(mode) == 0.0


def test_a_max_form_floor_breaks_the_premise():
    """MUTATION-SHAPED, and committed because the fix is someone ELSE's future work:
    a floor answered as "the date from which ALL these symbols have data" is not a
    lower bound, and the stored intermediate stops being universe-free.

    Asserting the BREAK (measured 9.322e-03 against 0.000e+00 for both lower-bound
    forms) is what makes the contract a checked fact rather than a comment. The
    open D5/D6 item "derive the declared floor per symbol from the coverage ledger"
    is exactly where a max-form implementation would be natural to write.
    """
    assert _intermediate_universe_gap("max") > 0.0


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


#: Names the fixture has NO bars for. A universe member with nothing to compute is
#: what makes the two payloads' served shapes differ observably; without one, the
#: "extra rows are NaN" assertion below is structurally empty for the factors whose
#: emit grid is already dense (measured: 2 of the 3 parameters had zero extra rows).
GHOST_SYMBOLS = ["999001.SZ", "999002.SZ", "999003.SZ"]


@pytest.mark.parametrize("fid", [CROSS_SECTIONAL_ID, PURE_MINUTE_ID, SPARSE_MINUTE_ID])
def test_served_panel_matches_the_direct_engine_and_adds_only_nan_rows(fid):
    """Going through the store must not change a value, and may only ADD empty rows.

    The direct engine (``materialize_range``, no store at all) is the independent
    reference. Every cell it emits must come back identical; anything the served
    panel has on top of that is a footprint cell and must be NaN — a finite value
    where the engine emits no row would mean the store invented one.

    HOW MUCH is added differs by payload, and the test asserts the difference
    rather than a single claim that is only true for one of them: a VALUE payload
    gains explicit NaN rows (asserted to be non-empty here, so the NaN assertion is
    not vacuous), while the CROSS-SECTIONAL payload gains none, because its combine
    emits only finite-pair cells and drops the footprint rows. Measured with the
    three no-bar names: value payloads serve 54 of 54 requested cells, the
    cross-sectional one serves 48 and omits the ghosts entirely.
    """
    universe = SYMBOLS + GHOST_SYMBOLS
    factor = factor_registry.build(fid)
    with tempfile.TemporaryDirectory() as td:
        served, _ = _ask(FactorValueStore(td), fid, universe)
    direct = materialize_range(
        factor, view=View.DECISION, symbols=universe, emit_start=EMIT[0],
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
    ghosts_served = {
        s for s in map(str, served.index.get_level_values("symbol"))
    } & set(GHOST_SYMBOLS)
    if stores_intermediate(factor):
        assert len(extra) == 0, (
            f"{fid}: the combine decides the served shape; footprint rows must not "
            f"reach the panel"
        )
        assert not ghosts_served, f"{fid}: a name with nothing to standardize is absent"
    else:
        assert len(extra) >= len(EMIT) * len(GHOST_SYMBOLS), (
            f"{fid}: expected explicit NaN rows for the no-bar names, got {len(extra)}"
        )
        assert ghosts_served == set(GHOST_SYMBOLS), (
            f"{fid}: a value payload must carry the requested no-bar names as NaN"
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
