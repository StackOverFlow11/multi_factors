"""D4b: per-symbol STREAMING minute materialization (design §九 D4b / A1).

Network-free. D4's materializer loaded every symbol's minute bars into ONE frame,
which the evaluation plane cannot hold (measured: 52.7 GB nominal / ~115 GB peak
RSS vs 56 GB available — ``docs/factors/d5_saturation_feasibility.md``). D4b reads
one symbol at a time, reduces it to that symbol's daily intermediate, discards the
bars, and runs the factor's cross-sectional combine ONCE on the assembled panel.

This file pins the properties that make that a value-preserving change:

* the two granularities agree — ``combine(per_symbol(bars)) == minute_raw_from_bars``
  for every bound minute factor, so the split cannot drift from the whole-factor call;
* streaming reproduces the load-geometry-free truth (compute on the whole history in
  ONE frame) cell-for-cell, INCLUDING ``intraday_amp_cut``, the one factor whose
  combine is a real cross-section;
* the cut is BEFORE that cross-section: splitting around the WHOLE factor instead is
  kept as a committed NEGATIVE CONTROL (it turns every value into NaN), and
  ``AMP_CUT_MIN_CROSS_SECTION`` is asserted un-relaxed;
* memory-boundedness is structural (the engine never asks a provider for more than
  one symbol) rather than a claim;
* per-symbol saturation reaches the same terminal per symbol, and a thin symbol no
  longer drags a dense one's load depth;
* the engine re-filters what a provider returns instead of trusting it.

Every invariance claim here has recorded mutation evidence in its docstring, and each
mutation was first asserted to actually change its target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import View
from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors.compute.minute.binding import (
    _MINUTE_BINDINGS,
    _MINUTE_STREAM_BINDINGS,
    CROSS_SECTIONAL_MINUTE_FACTORS,
    combine_minute_stats,
    is_cross_sectional_minute,
    minute_raw_from_bars,
    minute_stats_from_bars,
)
from factors.compute.minute.intraday_amp_cut import (
    AMP_CUT_MIN_CROSS_SECTION,
    IntradayAmpCutFactor,
)
from factors.materialize import MaterializeSources, materialize_range
from factors.view_lag import minute_decision_cutoff

#: The ten minute factors with a bars-only binding (valley_price_quantile is the
#: deliberately deferred eleventh — it also needs the daily panel).
STREAMED_FACTOR_IDS = (
    "jump_amount_corr_20",
    "minute_ideal_amp_10",
    "amp_marginal_anomaly_vol_20",
    "volume_peak_count_20",
    "intraday_amp_cut_10",
    "peak_interval_kurtosis_20",
    "valley_relative_vwap_20",
    "valley_ridge_vwap_ratio_20",
    "ridge_minute_return_20",
    "peak_ridge_amount_ratio_20",
)

#: >= AMP_CUT_MIN_CROSS_SECTION symbols, so intraday_amp_cut's date-wise gate can
#: pass and the reconciliation below is non-vacuous for it.
N_SYMBOLS = 12
SYMBOLS = [f"6000{i:02d}.SH" for i in range(N_SYMBOLS)]
DATES = pd.bdate_range("2021-01-04", periods=48)
EMIT_START, EMIT_END = DATES[44], DATES[47]
BARS_PER_DAY = 238


def _session(rows, rng, symbol, day, n_bars):
    base = pd.Timestamp(day) + pd.Timedelta("09:31:00")
    price = 100.0 + rng.normal(0, 2)
    for i in range(n_bars):
        t = base + pd.Timedelta(minutes=i)
        price += rng.normal(0, 0.05)
        slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
        erupt = 6.0 if (rng.rand() < 0.06) else 1.0
        vol = slot * erupt * (1.0 + 0.1 * rng.rand())
        w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
        hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
        cl = lo + (hi - lo) * rng.rand()
        rows.append((t, symbol, price, hi, lo, cl, vol, cl * vol))


def _frame(rows) -> pd.DataFrame:
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


def _dense_minute() -> pd.DataFrame:
    rng = np.random.RandomState(31)
    rows: list[tuple] = []
    for s in SYMBOLS:
        for d in DATES:
            _session(rows, rng, s, d, BARS_PER_DAY)
    return _frame(rows)


DENSE = _dense_minute()


class StreamProv:
    """Well-behaved provider: honours ``symbols``, records what it was asked for."""

    def __init__(self, bars: pd.DataFrame = DENSE):
        self._b = bars
        self.symbol_counts: list[int] = []
        self.max_rows_served = 0

    def minute_bars(self, symbols, start, end):
        self.symbol_counts.append(len(list(symbols)))
        if not symbols:
            return self._b.iloc[0:0]
        t = self._b.index.get_level_values("time")
        sym = pd.Index(self._b.index.get_level_values("symbol"))
        out = self._b[
            (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & sym.isin([str(s) for s in symbols])
        ]
        self.max_rows_served = max(self.max_rows_served, len(out))
        return out

    def earliest_available(self, symbols):
        return DATES[0]


class SloppyProv(StreamProv):
    """Contract-violating provider: IGNORES ``symbols`` and returns everything."""

    def minute_bars(self, symbols, start, end):
        self.symbol_counts.append(len(list(symbols)))
        if not symbols:
            return self._b.iloc[0:0]
        t = self._b.index.get_level_values("time")
        return self._b[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]


def _truth(factor, bars: pd.DataFrame = DENSE, view=View.CLOSE) -> pd.Series:
    """Load-geometry-free reference: the WHOLE history in ONE frame, one call.

    Independent of the streaming engine (it calls the factor's own whole-factor
    binding), so agreeing with it is evidence, not a tautology. This is the same
    ground truth ``test_locking_offset_is_load_bearing`` uses.
    """
    work = bars if view is View.CLOSE else minute_decision_cutoff(bars, decision_time="14:50:00")
    full = minute_raw_from_bars(factor, work)
    d = full.index.get_level_values("date")
    return full[(d >= EMIT_START) & (d <= EMIT_END)].sort_index(kind="mergesort")


def _streamed(factor, provider=None, view=View.CLOSE, symbols=None) -> pd.Series:
    return materialize_range(
        factor,
        view=view,
        symbols=list(symbols or SYMBOLS),
        emit_start=EMIT_START,
        emit_end=EMIT_END,
        sources=MaterializeSources(minute=provider or StreamProv()),
        decision_cutoff="14:50:00",
    )


#: Per-factor budget for the ONE attributable difference between the streamed
#: load and the whole-history load: pandas' rolling accumulation runs over a
#: differently-sized prefix, so a factor that accumulates can differ in the last
#: bits. MEASURED on this fixture: nine of the ten factors are EXACTLY equal
#: (max|abs| = 0.000e+00) and only ``jump_amount_corr_20`` moves, by max|abs| =
#: 5.551e-16 / max|rel| = 4.893e-14 — four orders of magnitude under the budget
#: below, so a real missing-row or wrong-window effect could not hide inside it
#: (that effect is O(1), cf. the 416.0-vs-438.0 locking-offset witness). The
#: exactness of the other nine is PINNED here on purpose: if a change turned one
#: of them from exact into merely-close, this table would have to be edited, and
#: that edit is the review signal.
_RECONCILE_ATOL: dict[str, float] = {"jump_amount_corr_20": 1e-12}


def _assert_bit_identical(got: pd.Series, want: pd.Series, label: str, atol: float = 0.0) -> int:
    """Index and NaN mask must match EXACTLY; values within ``atol`` (default 0).

    The NaN mask is never given a tolerance: a finite<->NaN flip is the coverage
    change this whole step must not make, and it is not a float artefact.
    """
    assert got.index.equals(want.index), f"{label}: index differs"
    g, w = got.to_numpy(), want.to_numpy()
    assert np.array_equal(np.isnan(g), np.isnan(w)), f"{label}: NaN mask differs"
    finite = ~np.isnan(g)
    if atol == 0.0:
        assert np.array_equal(g[finite], w[finite]), f"{label}: finite values differ"
    else:
        assert np.allclose(g[finite], w[finite], rtol=0.0, atol=atol), (
            f"{label}: finite values differ by more than the float-reorder budget"
        )
    return int(finite.sum())


# --------------------------------------------------------------------------- #
# 1. The two granularities cannot drift apart
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor_id", STREAMED_FACTOR_IDS)
def test_split_stages_reproduce_the_whole_factor_call(factor_id):
    """``combine(per_symbol(bars)) == minute_raw_from_bars(bars)`` on the SAME bars.

    This is the structural link between the streaming engine and the whole-factor
    call the eval runners / panel freeze use: if a future edit changed one stage
    without the other, the two granularities would disagree here.
    """
    factor = factor_registry.build(factor_id)
    want = minute_raw_from_bars(factor, DENSE)
    got = combine_minute_stats(factor, minute_stats_from_bars(factor, DENSE))
    n = _assert_bit_identical(got.sort_index(), want.sort_index(), factor_id)
    assert n > 0, f"{factor_id}: vacuous (no finite values on the fixture)"


def test_stream_bindings_cover_exactly_the_bound_minute_factors():
    """A factor bound at one granularity but not the other is a readable error."""
    assert set(_MINUTE_STREAM_BINDINGS) == set(_MINUTE_BINDINGS)
    assert CROSS_SECTIONAL_MINUTE_FACTORS <= set(_MINUTE_STREAM_BINDINGS)
    assert is_cross_sectional_minute(IntradayAmpCutFactor())
    assert not is_cross_sectional_minute(factor_registry.build("volume_peak_count_20"))


def test_deferred_factor_still_raises_readably_through_the_split_stages():
    """valley_price_quantile is deferred: both stages must say so, not mis-compute."""
    factor = factor_registry.build("valley_price_quantile_20")
    with pytest.raises(NotImplementedError, match="valley_price_quantile"):
        minute_stats_from_bars(factor, DENSE)
    with pytest.raises(NotImplementedError, match="valley_price_quantile"):
        combine_minute_stats(factor, pd.DataFrame())


# --------------------------------------------------------------------------- #
# 2. Cell-by-cell reconciliation: streaming == whole-universe single frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor_id", STREAMED_FACTOR_IDS)
def test_streaming_reproduces_the_whole_universe_value(factor_id):
    """Per-symbol streaming == the one-frame, whole-history truth, cell for cell.

    Index and NaN mask exact; values exact for nine of the ten factors and within
    the measured float-reorder budget for the tenth (see ``_RECONCILE_ATOL``).

    MUTATION (run, rc=1): dropping the per-symbol re-filter in
    ``factors.materialize._symbol_bars`` while a sloppy provider is in play makes
    every symbol see the whole universe -> duplicated index entries; asserted
    separately in ``test_sloppy_provider_is_re_filtered``.
    """
    factor = factor_registry.build(factor_id)
    n = _assert_bit_identical(
        _streamed(factor), _truth(factor), factor_id, atol=_RECONCILE_ATOL.get(factor_id, 0.0)
    )
    assert n > 0, f"{factor_id}: vacuous (no finite values in the emit window)"


@pytest.mark.parametrize("factor_id", ["volume_peak_count_20", "intraday_amp_cut_10"])
def test_streaming_reproduces_the_whole_universe_value_decision_view(factor_id):
    """Same, through the DECISION view — the 14:50 cutoff is applied per symbol."""
    factor = factor_registry.build(factor_id)
    n = _assert_bit_identical(
        _streamed(factor, view=View.DECISION),
        _truth(factor, view=View.DECISION),
        factor_id,
    )
    assert n > 0, f"{factor_id}: vacuous"


# --------------------------------------------------------------------------- #
# 3. The cut must sit BEFORE the cross-section (the amp_cut coupling)
# --------------------------------------------------------------------------- #
def test_splitting_around_the_whole_factor_destroys_the_cross_sectional_factor():
    """NEGATIVE CONTROL, committed: the WRONG cut turns intraday_amp_cut all-NaN.

    Splitting per symbol around the whole factor hands its step-4 z-score a
    one-symbol cross-section, which is below ``AMP_CUT_MIN_CROSS_SECTION`` (=10)
    by definition -> every value NaN while the index still looks right. The probe
    measured exactly this on real bars (1692/1692 rows NaN, 0 comparable cells).
    Keeping it as a test is what stops a future "simplification" from moving the
    cut around the factor and shipping a silently NaN column.
    """
    factor = factor_registry.build("intraday_amp_cut_10")
    per_factor_parts = []
    for s in SYMBOLS:
        one = DENSE[pd.Index(DENSE.index.get_level_values("symbol")) == s]
        per_factor_parts.append(minute_raw_from_bars(factor, one))  # the WRONG cut
    wrong = pd.concat(per_factor_parts).sort_index()
    right = _truth(factor)

    d = wrong.index.get_level_values("date")
    wrong_win = wrong[(d >= EMIT_START) & (d <= EMIT_END)].sort_index()
    assert wrong_win.index.equals(right.index), "the wrong cut keeps the index — that is the trap"
    assert int(wrong_win.notna().sum()) == 0, "the wrong cut must be entirely NaN"
    assert int(right.notna().sum()) > 0, "vacuous: the right cut produced nothing to lose"
    # ...and the streaming engine takes the RIGHT cut.
    _assert_bit_identical(_streamed(factor), right, "intraday_amp_cut streamed")


def test_amp_cut_cross_section_gate_is_not_relaxed_for_streaming():
    """The definition constant stays at 10 and still bites BELOW it (honest NaN).

    Streaming would be trivially "fixed" by widening or dropping this gate; that
    would be a definition change. With 9 symbols requested the date's finite pair
    count is below the gate and every value is NaN — through the streaming path.
    """
    assert AMP_CUT_MIN_CROSS_SECTION == 10
    factor = factor_registry.build("intraday_amp_cut_10")
    thin = _streamed(factor, symbols=SYMBOLS[:9])
    assert len(thin) > 0, "vacuous: no rows emitted at all"
    assert int(thin.notna().sum()) == 0, "a sub-threshold cross-section must be all NaN"
    # the same universe one symbol larger clears the gate (so the gate, not the
    # plumbing, is what produced the NaNs above)
    wide = _streamed(factor, symbols=SYMBOLS[:10])
    assert int(wide.notna().sum()) > 0


# --------------------------------------------------------------------------- #
# 4. Memory-boundedness is structural, not a claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor_id", STREAMED_FACTOR_IDS)
def test_provider_is_never_asked_for_more_than_one_symbol(factor_id):
    """The engine asks for ONE symbol per call — the whole-universe frame is never
    requested, so it can never be built.

    MUTATION (run, rc=1): restoring the whole-universe request
    (``minute_bars(list(symbols), ...)``) makes the recorded call carry 12 symbols
    and this fails; restored -> rc=0.
    """
    prov = StreamProv()
    _streamed(factor_registry.build(factor_id), provider=prov)
    assert prov.symbol_counts, "vacuous: the provider was never called"
    assert max(prov.symbol_counts) == 1, (
        f"{factor_id}: engine requested {max(prov.symbol_counts)} symbols in one call"
    )
    # and the biggest frame it ever handled is one symbol's worth of bars
    assert prov.max_rows_served <= len(DATES) * BARS_PER_DAY


def test_sloppy_provider_is_re_filtered():
    """A provider that ignores ``symbols`` must not corrupt the streamed result.

    Streaming makes the engine depend on the provider honouring its argument; a
    provider returning the whole store would emit every other symbol's rows once
    per streamed symbol. The engine re-filters, so the sloppy provider produces
    the SAME values as the well-behaved one.

    MUTATION (run, rc=1): removing the re-filter in ``_symbol_bars`` makes this
    raise on duplicate index entries / mismatched length; restored -> rc=0.
    """
    factor = factor_registry.build("volume_peak_count_20")
    clean = _streamed(factor, provider=StreamProv())
    sloppy = _streamed(factor, provider=SloppyProv())
    n = _assert_bit_identical(sloppy, clean, "sloppy provider")
    assert n > 0
    assert not sloppy.index.duplicated().any(), "duplicate (date, symbol) rows"


# --------------------------------------------------------------------------- #
# 5. Per-symbol saturation: same terminal, no cross-symbol drag
# --------------------------------------------------------------------------- #
#: A thin symbol that can NEVER accumulate ``lookback_days`` valid days before the
#: emit window (it lists inside it) — the shape that, under the universe-wide
#: criterion, dragged every other symbol's load back to the declared floor (84 of
#: the evaluation universe's 995 symbols are of this kind).
_LATE_SYMBOL = "600099.SH"


def _dense_plus_late() -> pd.DataFrame:
    rng = np.random.RandomState(77)
    rows: list[tuple] = []
    for d in DATES[-6:]:  # lists only just before the emit window
        _session(rows, rng, _LATE_SYMBOL, d, BARS_PER_DAY)
    return pd.concat([DENSE, _frame(rows)]).sort_index(kind="mergesort")


DENSE_PLUS_LATE = _dense_plus_late()


def test_a_late_listing_symbol_does_not_change_the_others_values():
    """Per-symbol saturation: each symbol expands to ITS OWN terminal.

    Adding a symbol that can never satisfy the criterion leaves every other
    symbol's value bit-identical (they are decided on their own history), which is
    the whole point of moving the decision per symbol. Under the universe-wide
    criterion this symbol forced the entire load to the declared floor.

    Note the values compared are the per-symbol-pure ones; ``intraday_amp_cut``
    has its own test below because a new symbol legitimately joins its
    cross-section.
    """
    factor = factor_registry.build("volume_peak_count_20")
    without = _streamed(factor, provider=StreamProv())
    with_late = _streamed(
        factor, provider=StreamProv(DENSE_PLUS_LATE), symbols=[*SYMBOLS, _LATE_SYMBOL]
    )
    shared = with_late[
        pd.Index(with_late.index.get_level_values("symbol")).isin(SYMBOLS)
    ]
    n = _assert_bit_identical(shared, without, "late-listing neighbour")
    assert n > 0


#: A LONG, NARROW fixture: long enough that a dense symbol's first expansion
#: chunk (``lookback_depth*2 + 40`` = 120 calendar days at depth 40) stops well
#: short of the declared floor, so "saturated early" and "loaded to the floor" are
#: DISTINGUISHABLE. On the short fixture above both coincide, which would let this
#: test pass with the old universe-wide drag still in place.
LONG_DATES = pd.bdate_range("2020-01-01", periods=200)
LONG_DENSE = ["600200.SH", "600201.SH"]
LONG_EMIT_START, LONG_EMIT_END = LONG_DATES[196], LONG_DATES[199]


def _long_minute() -> pd.DataFrame:
    rng = np.random.RandomState(4242)
    rows: list[tuple] = []
    for s in LONG_DENSE:
        for d in LONG_DATES:
            _session(rows, rng, s, d, BARS_PER_DAY)
    for d in LONG_DATES[-6:]:  # the late listing: inside the emit neighbourhood
        _session(rows, rng, _LATE_SYMBOL, d, BARS_PER_DAY)
    return _frame(rows)


LONG_MINUTE = _long_minute()


class LongProv(StreamProv):
    def __init__(self):
        super().__init__(LONG_MINUTE)

    def earliest_available(self, symbols):
        return LONG_DATES[0]


def test_a_late_listing_symbol_still_reaches_its_own_floor():
    """...and the thin symbol is NOT dropped: it is loaded to its declared floor,
    while its dense neighbours stop at their own (much shallower) saturation.

    The zero-output veto lives on per symbol — what changed is only that it no
    longer vetoes on behalf of the other 995. Under the universe-wide criterion
    every symbol here would share the thin symbol's floor-deep load start.

    MUTATION (run, rc=1): forcing the criterion to False (never saturate) drags
    the dense symbols to the floor too and ``late_deepest < dense_deepest`` fails
    — i.e. this assertion is what distinguishes per-symbol saturation from the
    universe-wide drag; see ``test_universe_wide_drag_mutation``.
    """
    factor = factor_registry.build("volume_peak_count_20")
    starts: dict[str, list[pd.Timestamp]] = {}

    class RecordingProv(LongProv):
        def minute_bars(self, symbols, start, end):
            if symbols:
                starts.setdefault(str(list(symbols)[0]), []).append(pd.Timestamp(start))
            return super().minute_bars(symbols, start, end)

    materialize_range(
        factor, view=View.CLOSE, symbols=[*LONG_DENSE, _LATE_SYMBOL],
        emit_start=LONG_EMIT_START, emit_end=LONG_EMIT_END,
        sources=MaterializeSources(minute=RecordingProv()),
    )
    assert _LATE_SYMBOL in starts, "the thin symbol was never loaded (silently dropped)"
    late_deepest = min(starts[_LATE_SYMBOL])
    dense_deepest = min(min(v) for k, v in starts.items() if k != _LATE_SYMBOL)
    assert late_deepest <= LONG_DATES[0], "the thin symbol must expand to its declared floor"
    assert late_deepest < dense_deepest, (
        "the thin symbol must go DEEPER than its dense neighbours — otherwise this "
        "test would pass with the old universe-wide drag still in place"
    )


def test_universe_wide_drag_mutation(monkeypatch):
    """MUTATION for the test above: with saturation disabled every symbol is
    dragged to the declared floor, so the depth contrast vanishes.

    The mutation is asserted to actually bite first (the dense symbols' deepest
    load start MOVES), which is what makes the contrast above evidence rather
    than a coincidence of the fixture.
    """
    import factors.materialize as mat

    factor = factor_registry.build("volume_peak_count_20")

    def record() -> dict[str, list[pd.Timestamp]]:
        starts: dict[str, list[pd.Timestamp]] = {}

        class RecordingProv(LongProv):
            def minute_bars(self, symbols, start, end):
                if symbols:
                    starts.setdefault(str(list(symbols)[0]), []).append(pd.Timestamp(start))
                return super().minute_bars(symbols, start, end)

        materialize_range(
            factor, view=View.CLOSE, symbols=[*LONG_DENSE, _LATE_SYMBOL],
            emit_start=LONG_EMIT_START, emit_end=LONG_EMIT_END,
            sources=MaterializeSources(minute=RecordingProv()),
        )
        return starts

    clean = record()
    monkeypatch.setattr(mat, "_pooled_pool_saturated", lambda *a, **k: False)
    mutated = record()

    clean_dense = min(min(clean[s]) for s in LONG_DENSE)
    mutated_dense = min(min(mutated[s]) for s in LONG_DENSE)
    assert mutated_dense < clean_dense, "the mutation did not change its target"
    assert mutated_dense <= LONG_DATES[0], "the mutation should drag the dense names to the floor"


def test_an_early_saturating_symbol_matches_the_whole_history_value():
    """The soundness claim of per-symbol saturation, on a fixture where it BITES.

    A dense symbol here stops at its first expansion chunk — far short of the
    declared floor — so this compares a genuinely EARLY-terminated load against
    computing on the symbol's whole history. That is the whole content of the
    criterion: "further history cannot change any emit value".

    The early termination is asserted HERE, not borrowed from the test above: if
    the fixture ever grew short enough that the first chunk already reached the
    floor, both sides would load the same bars and this would agree for the wrong
    reason — a test that cannot fail.
    """
    factor = factor_registry.build("volume_peak_count_20")
    starts: list[pd.Timestamp] = []

    class RecordingProv(LongProv):
        def minute_bars(self, symbols, start, end):
            starts.append(pd.Timestamp(start))
            return super().minute_bars(symbols, start, end)

    got = materialize_range(
        factor, view=View.CLOSE, symbols=list(LONG_DENSE),
        emit_start=LONG_EMIT_START, emit_end=LONG_EMIT_END,
        sources=MaterializeSources(minute=RecordingProv()),
    )
    assert min(starts) > LONG_DATES[0], (
        "the dense symbols reached the declared floor, so 'early saturation' is not "
        "being exercised and this comparison would be vacuous"
    )
    full = minute_raw_from_bars(factor, LONG_MINUTE)
    d = full.index.get_level_values("date")
    want = full[
        (d >= LONG_EMIT_START) & (d <= LONG_EMIT_END)
        & pd.Index(full.index.get_level_values("symbol")).isin(LONG_DENSE)
    ].sort_index(kind="mergesort")
    n = _assert_bit_identical(got, want, "early-saturating symbol")
    assert n > 0, "vacuous: no finite values"


def test_cross_sectional_factor_sees_the_whole_universe_not_the_streamed_symbol():
    """intraday_amp_cut's combine runs on the ASSEMBLED panel: a new symbol joins
    the cross-section and moves the z-scores, exactly as in the one-frame path.

    The move is asserted, not assumed. If the extra member left every existing
    value untouched, agreeing with the one-frame reference would prove nothing
    about WHICH cross-section the combine saw — the streamed values would be
    consistent with a per-symbol cross-section too.
    """
    factor = factor_registry.build("intraday_amp_cut_10")
    got = _streamed(
        factor, provider=StreamProv(DENSE_PLUS_LATE), symbols=[*SYMBOLS, _LATE_SYMBOL]
    )
    want = _truth(factor, bars=DENSE_PLUS_LATE)
    n = _assert_bit_identical(got, want, "amp_cut with a late-listing member")
    assert n > 0

    without = _streamed(factor)  # the same 12 symbols, no new member
    shared = got[pd.Index(got.index.get_level_values("symbol")).isin(SYMBOLS)]
    common = without.index.intersection(shared.index)
    assert len(common) > 0, "vacuous: the two runs share no cell"
    moved = int((shared.loc[common].to_numpy() != without.loc[common].to_numpy()).sum())
    assert moved > 0, (
        "the extra cross-section member changed nothing, so this test cannot tell "
        "a whole-universe combine from a per-symbol one"
    )


# --------------------------------------------------------------------------- #
# 6. The per-symbol trailing trim (disclosed semantic refinement)
# --------------------------------------------------------------------------- #
_SPARSE_SYMBOL = "600098.SH"


def _dense_plus_sparse() -> pd.DataFrame:
    """One symbol trading only every third day: its own trailing N trading days
    reach further back than the loaded universe's union of N days."""
    rng = np.random.RandomState(99)
    rows: list[tuple] = []
    for i, d in enumerate(DATES):
        if i % 3 == 0:
            _session(rows, rng, _SPARSE_SYMBOL, d, BARS_PER_DAY)
    return pd.concat([DENSE, _frame(rows)]).sort_index(kind="mergesort")


DENSE_PLUS_SPARSE = _dense_plus_sparse()


def test_per_symbol_trim_uses_the_symbols_own_trading_days():
    """DISCLOSED REFINEMENT: the trailing trim counts THIS symbol's trading days.

    Under the union rule a densely-traded neighbour's calendar decided how much
    history a sparse symbol was warmed with — itself a load-geometry dependence
    (red line #6). This pins BOTH halves so the refinement is a fact, not a
    side effect:

    1. the two rules really disagree on this fixture (the union's ``warmup``-th
       day back is LATER than the sparse symbol's own, so the union rule keeps
       LESS of its history) — if they agreed, half of this test would be vacuous;
    2. the streamed value equals the load-geometry-free truth (the symbol's whole
       history in one frame), which is the property the trim exists to deliver.
    """
    from factors.materialize import _warmup_start

    factor = factor_registry.build("jump_amount_corr_20")  # a BOUNDED minute factor
    warmup = int(factor.spec.lookback_depth)

    # (1) the rules disagree — computed from the fixture, not asserted by fiat.
    all_days = pd.DatetimeIndex(
        pd.unique(pd.DatetimeIndex(DENSE_PLUS_SPARSE.index.get_level_values("time")).normalize())
    )
    sym_level = pd.Index(DENSE_PLUS_SPARSE.index.get_level_values("symbol"))
    sparse_bars = DENSE_PLUS_SPARSE[sym_level == _SPARSE_SYMBOL]
    own_days = pd.DatetimeIndex(
        pd.unique(pd.DatetimeIndex(sparse_bars.index.get_level_values("time")).normalize())
    )
    union_keep = _warmup_start(all_days, EMIT_START, warmup)
    own_keep = _warmup_start(own_days, EMIT_START, warmup)
    assert own_keep < union_keep, (
        "vacuous fixture: the union rule and the per-symbol rule pick the same "
        "cutoff here, so this test would not exercise the refinement"
    )

    # (2) the streamed value is the load-geometry-free one.
    got = _streamed(
        factor, provider=StreamProv(DENSE_PLUS_SPARSE), symbols=[*SYMBOLS, _SPARSE_SYMBOL]
    )
    want = _truth(factor, bars=DENSE_PLUS_SPARSE)
    mine = pd.Index(got.index.get_level_values("symbol")) == _SPARSE_SYMBOL
    assert int(mine.sum()) > 0, "vacuous: the sparse symbol emitted nothing"
    _assert_bit_identical(
        got, want, "per-symbol trim", atol=_RECONCILE_ATOL["jump_amount_corr_20"]
    )
