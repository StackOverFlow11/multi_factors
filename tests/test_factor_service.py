"""D4: the factor service (factors.service) — the §3.5 P8 acceptance.

Network-free. Proves the hardest guarantee: two COLD stores, one filled by
repeated per-date ``cross_section`` and one by a single batch ``panel``, hold
BIT-IDENTICAL raw values (design §3.5 P8). Also the read-layer smoke
(``panel[d] ≡ cross_section(d)``), the view x basis pairing gate, and the
read-through no-recompute-on-hit property.

The P8 test uses THREE factors that stress the tail-splice / leading-NaN
boundary differently:
* ``momentum_20`` — window-local (a two-point ratio, NO accumulation), so single
  and batch are EXACTLY bit-identical; its transitive depth (21) exceeds its
  headline window (20), so the warmup trim uses the transitive depth, not the
  headline (the §六.18 deception guard);
* ``volatility_20`` — rolling std (pandas accumulation), single==batch up to an
  attributable float-reorder (<= 1e-12);
* ``volume_peak_count_20`` — a genuinely NESTED minute factor (depth 40 = 20 +
  20 baseline), single==batch up to the same float-reorder.

Mutation evidence (recorded): forcing the materializer's warmup BELOW the
factor's window makes the per-date fill under-warm every date while the batch
fill only under-warms its first — so the two stores DIVERGE. The test that
applies this mutation asserts the divergence, proving the warmup trim (and the
transitive ``lookback_depth``) is load-bearing.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import ReturnBasis, View
from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors import service as service_mod
from factors.materialize import MaterializeSources, materialize_range
from factors.service import DecisionPoint, cross_section, panel
from factors.store.keys import store_key
from factors.store.values import FactorValueStore

SYMS = ["000001.SZ", "000002.SZ"]
DATES = pd.bdate_range("2021-01-04", periods=75)


# --------------------------------------------------------------------------- #
# Synthetic data + providers
# --------------------------------------------------------------------------- #
def _daily():
    rng = np.random.RandomState(0)
    rows = []
    for si, s in enumerate(SYMS):
        px = 100.0 + si * 20 + np.cumsum(rng.normal(0, 1.0, len(DATES)))
        for d, p in zip(DATES, px):
            rows.append((d, s, p - 0.3, p + 0.5, p - 0.5, p, 1e5, p * 1e5))
    return (
        pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
        .set_index(["date", "symbol"]).sort_index()
    )


def _rich_minute():
    """55 days x 238-bar sessions with same-slot baseline + eruptions -> the
    peak/valley taxonomy produces finite nested-factor values."""
    rng = np.random.RandomState(7)
    rows = []
    for si, s in enumerate(SYMS):
        for d in DATES[:55]:
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 5 + rng.normal(0, 2)
            for i in range(238):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot_base = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot_base * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    frame = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return normalize_intraday_bars(frame, freq="1min")


DAILY = _daily()
MINUTE = _rich_minute()


class DailyProv:
    def daily_panel(self, symbols, start, end):
        m = DAILY.index.get_level_values("date")
        return DAILY[(m >= pd.Timestamp(start)) & (m <= pd.Timestamp(end))]


class MinuteProv:
    def __init__(self):
        self.calls = 0

    def minute_bars(self, symbols, start, end):
        self.calls += 1
        if not symbols:
            return MINUTE.iloc[0:0]
        t = MINUTE.index.get_level_values("time")
        return MINUTE[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]

    def earliest_available(self, symbols):
        """DECLARED data start of this fixture (never inferred from row counts)."""
        return DATES[0]


def _sparse_minute():
    """Sparse-valid minute data: every 3rd day is thin (fewer bars) so the
    valid-day density is < 1 — the window where a fixed lookback_depth trim makes
    valid-day-POOLED factors (ridge / valley_ridge / peak_ridge) diverge
    finite<->NaN between a single-date fill and a batch fill (the review HIGH)."""
    rng = np.random.RandomState(11)
    rows = []
    for si, s in enumerate(SYMS):
        for di, d in enumerate(DATES[:80]):
            n = 120 if (di % 3 == 0) else 238
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 5 + rng.normal(0, 2)
            for i in range(n):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    frame = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return normalize_intraday_bars(frame, freq="1min")


SPARSE_MINUTE = _sparse_minute()


class SparseMinuteProv:
    def minute_bars(self, symbols, start, end):
        if not symbols:
            return SPARSE_MINUTE.iloc[0:0]
        t = SPARSE_MINUTE.index.get_level_values("time")
        return SPARSE_MINUTE[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]

    def earliest_available(self, symbols):
        return DATES[0]


#: Gap geometry (the review counterexample, sized to actually exercise it).
#: One expansion chunk is ``lookback_depth*2 + 40`` CALENDAR days ~= 85 trading
#: days at depth 40. The gap must be wide enough that a WHOLE expansion step
#: lands INSIDE it (adding no bars at all -> emit values unchanged -> a
#: value-stability fixed point terminates falsely) while REAL earlier history
#: still exists beyond it: 200 trading days spans more than two chunks.
_GAP_TRADING_DAYS = 200
_PRE_GAP_DAYS = 60
_POST_GAP_DAYS = 30
_GAP_BARS_PER_DAY = 238  # a full session: the pooled factors need enough
                         # classifiable + ridge/valley bars to yield valid days
GAP_DATES = pd.bdate_range(
    "2021-01-04", periods=_PRE_GAP_DAYS + _GAP_TRADING_DAYS + _POST_GAP_DAYS
)


def _gapped_minute():
    """The REVIEW COUNTEREXAMPLE shape: rich bars, then a LONG no-bar gap
    (a suspension) wider than one expansion chunk, then rich bars again.

    The emit window sits AFTER the gap and its trailing valid-day pool must reach
    back ACROSS the gap. A value-stability fixed point terminates falsely here
    (one chunk lands entirely inside the gap -> emit values unchanged), and an
    unchanged row count inside the gap looks exactly like the data start — the
    two failure modes the structural criterion replaces.
    """
    rng = np.random.RandomState(23)
    rows = []
    pre = GAP_DATES[:_PRE_GAP_DAYS]
    post = GAP_DATES[_PRE_GAP_DAYS + _GAP_TRADING_DAYS:]
    for si, s in enumerate(SYMS):
        for d in list(pre) + list(post):
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 5 + rng.normal(0, 2)
            for i in range(_GAP_BARS_PER_DAY):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    frame = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return normalize_intraday_bars(frame, freq="1min"), list(post)


GAPPED_MINUTE, GAPPED_POST_DATES = _gapped_minute()


class GappedMinuteProv:
    def minute_bars(self, symbols, start, end):
        if not symbols:
            return GAPPED_MINUTE.iloc[0:0]
        t = GAPPED_MINUTE.index.get_level_values("time")
        return GAPPED_MINUTE[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]

    def earliest_available(self, symbols):
        """DECLARED start — the mid-history gap must never be mistaken for it."""
        return GAP_DATES[0]


def _sources():
    return MaterializeSources(daily=DailyProv(), minute=MinuteProv())


def _sparse_sources():
    return MaterializeSources(minute=SparseMinuteProv())


def _gapped_sources():
    return MaterializeSources(minute=GappedMinuteProv())


def _store_series(store, factor_id, view=View.DECISION):
    factor = factor_registry.build(factor_id)
    key = store_key(factor, view=view.value)
    return store.read(key)


def _fill_single(store, factor_ids, dates, src, universe=None):
    for d in dates:
        cross_section(factor_ids, universe or SYMS, DecisionPoint(date=d),
                      store=store, sources=src)


def _fill_batch(store, factor_ids, dates, src, universe=None):
    panel(factor_ids, universe or SYMS, [DecisionPoint(date=d) for d in dates],
          store=store, sources=src)


# --------------------------------------------------------------------------- #
# P8: single-fill == batch-fill
# --------------------------------------------------------------------------- #
def test_single_fill_equals_batch_fill_momentum_bit_identical():
    """Window-local momentum: the two cold stores are EXACTLY bit-identical."""
    dates = list(DATES[40:56])
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a, b = FactorValueStore(ta), FactorValueStore(tb)
        _fill_single(a, ["momentum_20"], dates, _sources())
        _fill_batch(b, ["momentum_20"], dates, _sources())
        sa = _store_series(a, "momentum_20").sort_index()
        sb = _store_series(b, "momentum_20").sort_index()
        assert sa.index.equals(sb.index)
        av, bv = sa.to_numpy(), sb.to_numpy()
        assert np.array_equal(np.isnan(av), np.isnan(bv))
        assert np.array_equal(av[~np.isnan(av)], bv[~np.isnan(bv)])  # BIT identical


def test_single_fill_equals_batch_fill_rolling_and_nested_minute():
    """volatility (rolling) + volume_peak_count (NESTED minute, depth 40): the two
    stores match up to the attributable pandas-rolling float-reorder (<= 1e-12)."""
    dates = list(DATES[42:54])
    for fid in ("volatility_20", "volume_peak_count_20"):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            a, b = FactorValueStore(ta), FactorValueStore(tb)
            _fill_single(a, [fid], dates, _sources())
            _fill_batch(b, [fid], dates, _sources())
            sa = _store_series(a, fid).sort_index()
            sb = _store_series(b, fid).sort_index()
            assert sa.index.equals(sb.index), fid
            av, bv = sa.to_numpy(), sb.to_numpy()
            assert np.array_equal(np.isnan(av), np.isnan(bv)), fid
            finite = ~np.isnan(av)
            # non-vacuous: the nested factor really produced values in this window.
            assert finite.sum() > 0, fid
            assert np.allclose(av[finite], bv[finite], rtol=1e-12, atol=1e-12), fid


def test_reduced_warmup_breaks_single_equals_batch(monkeypatch):
    """MUTATION: forcing the warmup below the factor window makes per-date fill
    under-warm every date while batch under-warms only its first -> the stores
    DIVERGE. Proves the warmup trim (transitive lookback_depth) is load-bearing."""
    dates = list(DATES[40:56])

    def _short_warmup(factor, **kw):
        # 10 << momentum's window of 20: every per-date fill under-warms (all NaN),
        # but the batch fill (loading 10 before d1) still computes its tail dates
        # d >= d1 + (window - 10) -> the two NaN patterns diverge.
        kw["warmup"] = 10
        return materialize_range(factor, **kw)

    monkeypatch.setattr(service_mod, "materialize_range", _short_warmup)
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a, b = FactorValueStore(ta), FactorValueStore(tb)
        _fill_single(a, ["momentum_20"], dates, _sources())
        _fill_batch(b, ["momentum_20"], dates, _sources())
        sa = _store_series(a, "momentum_20").sort_index()
        sb = _store_series(b, "momentum_20").sort_index()
        av, bv = sa.to_numpy(), sb.to_numpy()
        # the NaN pattern (per-date all-NaN vs batch finite near the tail) differs.
        assert not np.array_equal(np.isnan(av), np.isnan(bv))


# --------------------------------------------------------------------------- #
# P8 EXPANSION: valid-day POOLED factors (review HIGH) on a sparse-valid window
# --------------------------------------------------------------------------- #
# WHY these factors and this window (the review's "false confidence"): the P8
# single==batch test above uses volume_peak_count, which is ALSO valid-day pooled
# but happened to have dense-enough valid days to stay clean — it is the
# "happens-to-be-clean nested representative". The review showed that on a sparse-
# valid window ridge_minute_return / valley_ridge_vwap_ratio / peak_ridge_amount_
# ratio produce a single-fill=finite / batch-fill=NaN divergence (e.g. the clean
# cell ridge_minute_return_20 (2021-03-18, 000002.SZ): single=1.6976..., batch=NaN),
# because the trailing pool counts VALID days and a fixed lookback_depth trim
# truncates the pool differently for a per-date fill vs a batch fill. The
# materializer now loads these factors to SATURATION (real data start, no trim),
# so the value is load-geometry-free and the two stores agree.
_POOLED_DIVERGENT = ["ridge_minute_return_20", "valley_ridge_vwap_ratio_20", "peak_ridge_amount_ratio_20"]


def test_pooled_factor_single_equals_batch_on_sparse_valid_window():
    """The valid-day-pooled factors that diverge under a fixed trim now agree
    (saturation load) — single-fill store == batch-fill store, and the pool is
    fuller than a truncated one so the result is non-vacuous (real finite values)."""
    dates = list(DATES[55:75])
    for fid in _POOLED_DIVERGENT:
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            a, b = FactorValueStore(ta), FactorValueStore(tb)
            _fill_single(a, [fid], dates, _sparse_sources())
            _fill_batch(b, [fid], dates, _sparse_sources())
            sa = _store_series(a, fid).sort_index()
            sb = _store_series(b, fid).sort_index()
            assert sa.index.equals(sb.index), fid
            av, bv = sa.to_numpy(), sb.to_numpy()
            assert np.array_equal(np.isnan(av), np.isnan(bv)), f"{fid}: NaN mask diverges"
            finite = ~np.isnan(av)
            assert finite.sum() > 0, f"{fid}: vacuous (no finite values)"
            # saturation gives shared-prefix accumulation -> BIT identical.
            assert np.array_equal(av[finite], bv[finite]), f"{fid}: finite values diverge"


def test_pooled_single_equals_batch_across_a_long_no_bar_gap():
    """THE REVIEW COUNTEREXAMPLE, committed: a no-bar gap WIDER than one expansion
    chunk (``_GAP_TRADING_DAYS`` = a suspension), emit AFTER the gap, and the trailing
    valid-day pool must reach back ACROSS it.

    This is what defeated the value-stability fixed point (a chunk landing wholly
    inside the gap leaves the emit values unchanged -> false 'stable') and the
    row-count data-start inference (an unchanged count inside the gap looks like
    exhaustion). The structural criterion counts FINAL valid days in the locked
    sub-window, so a gap contributes none, the count stalls, and expansion
    continues to the declared floor.

    MUTATION EVIDENCE (run for this commit):
    * reverting the criterion to the value-stability fixed point -> this test
      FAILS (rc=1): both fills terminate inside the gap and emit all-NaN, caught
      by the non-vacuity assertion below; restored -> passes (rc=0).
    * dropping the baseline LOCKING OFFSET does NOT fail on THIS fixture (the
      valid-day count alone already forces enough history here); it IS isolated
      by ``test_dropping_the_locking_offset_yields_a_wrong_value``, whose sparse
      recent block puts an unlocked band inside the criterion's margin.
    """
    dates = GAPPED_POST_DATES[10:22]  # after the gap, pool still spans it
    for fid in _POOLED_DIVERGENT:
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            a, b = FactorValueStore(ta), FactorValueStore(tb)
            _fill_single(a, [fid], dates, _gapped_sources())
            _fill_batch(b, [fid], dates, _gapped_sources())
            sa = _store_series(a, fid).sort_index()
            sb = _store_series(b, fid).sort_index()
            assert sa.index.equals(sb.index), fid
            av, bv = sa.to_numpy(), sb.to_numpy()
            assert np.array_equal(np.isnan(av), np.isnan(bv)), f"{fid}: NaN mask diverges"
            finite = ~np.isnan(av)
            assert finite.sum() > 0, f"{fid}: vacuous (no finite values across the gap)"
            assert np.allclose(av[finite], bv[finite], rtol=0.0, atol=1e-12), fid


#: Zero-output fixture (review HIGH): symbol A has bars throughout; symbol B has
#: an ancient block and a recent block with a LONG suspension between, so in a
#: shallow load window B produces ZERO factor rows.
_ZO_DATES = pd.bdate_range("2021-01-04", periods=130)
_ZO_A, _ZO_B = "600000.SH", "600519.SH"
_ZO_SYMS = [_ZO_A, _ZO_B]


def _zero_output_minute():
    rng = np.random.RandomState(99)
    rows = []
    plan = ((_ZO_A, list(range(130))), (_ZO_B, list(range(10, 25)) + list(range(120, 130))))
    for si, (s, days) in enumerate(plan):
        for di in days:
            base = pd.Timestamp(_ZO_DATES[di]) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 5 + rng.normal(0, 2)
            for i in range(238):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    frame = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return normalize_intraday_bars(frame, freq="1min")


ZERO_OUTPUT_MINUTE = _zero_output_minute()


class ZeroOutputMinuteProv:
    def minute_bars(self, symbols, start, end):
        if not symbols:
            return ZERO_OUTPUT_MINUTE.iloc[0:0]
        t = ZERO_OUTPUT_MINUTE.index.get_level_values("time")
        sym = ZERO_OUTPUT_MINUTE.index.get_level_values("symbol")
        keep = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & sym.isin(list(symbols))
        return ZERO_OUTPUT_MINUTE[keep]

    def earliest_available(self, symbols):
        return _ZO_DATES[0]


def test_zero_output_symbol_is_not_silently_dropped():
    """REVIEW HIGH: a symbol producing ZERO rows in the current load window must
    still VETO saturation — otherwise a single-date fill drops it from the
    cross-section entirely while a batch fill keeps it (silent name-dropping =
    sample-coverage bias, forbidden by the I5a red line).

    Measured before the fix: volume_peak_count_20 single-fill ``600519.SH rows=0``
    vs batch-fill ``rows=10 finite=6`` — the criterion's per-symbol counts were
    ``{'600000.SH': 67}`` with 600519 absent entirely. After the fix both fills
    carry the same symbols and the same rows.
    """
    dates = list(_ZO_DATES[90:130])  # the review's emit window (indices 90..129)
    fid = "volume_peak_count_20"
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a, b = FactorValueStore(ta), FactorValueStore(tb)
        _fill_single(a, [fid], dates, MaterializeSources(minute=ZeroOutputMinuteProv()),
                     universe=_ZO_SYMS)
        _fill_batch(b, [fid], dates, MaterializeSources(minute=ZeroOutputMinuteProv()),
                    universe=_ZO_SYMS)
        sa = _store_series(a, fid).sort_index()
        sb = _store_series(b, fid).sort_index()
        # the thin symbol must be present in BOTH stores (never silently dropped)
        syms_a = set(map(str, sa.index.get_level_values("symbol")))
        syms_b = set(map(str, sb.index.get_level_values("symbol")))
        assert _ZO_B in syms_a, f"{_ZO_B} silently dropped by the single-date fill"
        assert syms_a == syms_b, f"symbol sets diverge: single={syms_a} batch={syms_b}"
        assert sa.index.equals(sb.index)
        av, bv = sa.to_numpy(), sb.to_numpy()
        assert np.array_equal(np.isnan(av), np.isnan(bv))
        finite = ~np.isnan(av)
        assert finite.sum() > 0
        assert np.allclose(av[finite], bv[finite], rtol=0.0, atol=1e-12)


def test_zero_output_symbol_veto_mutation(monkeypatch):
    """MUTATION for the HIGH: letting a ZERO-OUTPUT symbol count as saturated (the
    pre-fix behaviour) reintroduces the silent drop -> this test's assertion of
    the drop proves the zero-output veto is load-bearing.

    THE MUTATION'S EXPRESSION CHANGED WITH D4b; THE DEFECT IT ENCODES DID NOT.
    It used to be written as "restrict the criterion's symbol list to the symbols
    present in the output", which was the faithful shape while ONE criterion call
    decided the load depth for the WHOLE universe: the thin name was absent from
    that shared output, so filtering the list removed its veto. D4b decides depth
    PER SYMBOL, so the list holds exactly one name and the filtered-list form
    became INERT — measured on this fixture: 72 criterion calls (70 from the
    per-date fill, 2 from the batch fill), the real and filtered forms answer
    identically in 72 of 72, because all 20 calls with zero output also have
    ``loaded_days <= baseline_days`` and are already vetoed by the guard above the
    symbol loop.

    STATED PRECISELY, because the distinction matters: an inert mutation makes the
    assertion below — which asserts that the drop HAPPENS — impossible to satisfy.
    So the old expression did not become a test that cannot fail; it became a test
    that CANNOT PASS, and it went red on the D4b engine. It was not deleted or
    weakened to get green (§六.16): the property it protects still holds and is
    still asserted, with the same fixture, the same factor and the same
    assertions; only the mutation's wording moved to a form that still bites under
    per-symbol saturation ("no output -> no veto"). It reproduces the original
    divergence exactly: single-fill carries only 600000.SH (40 rows) while the
    batch fill carries both (50 rows).
    """
    import factors.materialize as mat

    real = mat._pooled_pool_saturated

    def zero_output_is_saturated(full, bars, emit_start, *, baseline_days, lookback_days, symbols):
        if full.empty:  # the defect: a symbol that produced nothing gets no veto
            return True
        return real(full, bars, emit_start, baseline_days=baseline_days,
                    lookback_days=lookback_days, symbols=symbols)

    monkeypatch.setattr(mat, "_pooled_pool_saturated", zero_output_is_saturated)
    dates = list(_ZO_DATES[90:130])  # the review's emit window (indices 90..129)
    fid = "volume_peak_count_20"
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a, b = FactorValueStore(ta), FactorValueStore(tb)
        _fill_single(a, [fid], dates, MaterializeSources(minute=ZeroOutputMinuteProv()),
                     universe=_ZO_SYMS)
        _fill_batch(b, [fid], dates, MaterializeSources(minute=ZeroOutputMinuteProv()),
                    universe=_ZO_SYMS)
        sa, sb = _store_series(a, fid), _store_series(b, fid)
        syms_a = set(map(str, sa.index.get_level_values("symbol"))) if sa is not None else set()
        syms_b = set(map(str, sb.index.get_level_values("symbol"))) if sb is not None else set()
        # NON-VACUITY GUARD: the disjunction below is trivially true when BOTH
        # stores are empty (nothing filled at all), which would make this mutation
        # test pass without exercising the mutation. Require real content first.
        assert syms_a, "single-fill store is empty — the mutation test would be vacuous"
        assert syms_b, "batch-fill store is empty — the mutation test would be vacuous"
        assert _ZO_B in syms_b, (
            "the batch fill must still carry the thin symbol; otherwise the "
            "single-vs-batch contrast below proves nothing"
        )
        assert syms_a != syms_b or _ZO_B not in syms_a, (
            "the zero-output-is-saturated criterion should drop the thin symbol from "
            "the single-date fill"
        )


class DistantFloorMinuteProv(ZeroOutputMinuteProv):
    """Same bars, but a DECLARED floor far in the past — so the saturation loop
    cannot terminate by reaching the floor within a bounded number of chunks."""

    def earliest_available(self, symbols):
        return pd.Timestamp("1990-01-01")


def test_saturation_exhaustion_raises_loudly(monkeypatch):
    """Exhausting the chunk budget must RAISE, never return an unsaturated value.

    Red line #9 (no silent degradation): the branch is unreachable in practice
    (500 chunks is >160 years), which is exactly why it needs a test — a raise
    that never executes is the next AttributeError's host. Recipe: cap the budget
    at 2 chunks, use a symbol that never accumulates ``lookback_days`` valid days,
    and declare a floor too distant to reach — so neither termination condition
    fires and the loop must fall through to the raise.
    """
    import factors.materialize as mat

    monkeypatch.setattr(mat, "_MAX_SATURATION_CHUNKS", 2)
    factor = factor_registry.build("volume_peak_count_20")
    emit = _ZO_DATES[128]
    with pytest.raises(RuntimeError, match="pooled saturation did not converge"):
        mat.materialize_range(
            factor, view=View.DECISION, symbols=_ZO_SYMS,
            emit_start=emit, emit_end=emit,
            sources=MaterializeSources(minute=DistantFloorMinuteProv()),
        )


def test_saturation_exhaustion_message_is_actionable(monkeypatch):
    """The raise names the factor and the declared floor (an operator must be able
    to tell WHICH factor stalled and how deep the search went)."""
    import factors.materialize as mat

    monkeypatch.setattr(mat, "_MAX_SATURATION_CHUNKS", 2)
    factor = factor_registry.build("volume_peak_count_20")
    emit = _ZO_DATES[128]
    with pytest.raises(RuntimeError) as excinfo:
        mat.materialize_range(
            factor, view=View.DECISION, symbols=_ZO_SYMS,
            emit_start=emit, emit_end=emit,
            sources=MaterializeSources(minute=DistantFloorMinuteProv()),
        )
    message = str(excinfo.value)
    assert factor.name in message           # which factor stalled
    assert "1990-01-01" in message          # the declared floor
    assert "expansion chunks" in message    # how far the search went


#: Locking-offset isolation fixture (review task 3(b), adopted verbatim in shape).
_LO_DATES = pd.bdate_range("2020-01-01", periods=210)
_LO_SYM = "600000.SH"
_LO_E_IDX = 200
#: deep history (reachable only by a second expansion) + a SPARSE recent block of
#: 33 every-other-day bar-days ending exactly at the emit date. In the shallow
#: window the sparse block is all there is, and its local days 10..19 are
#: classifiable but NOT yet final (baseline below full depth) — the UNLOCKED band
#: the offset exists to exclude.
_LO_BAR_DAYS = list(range(40, 100)) + [_LO_E_IDX - 2 * j for j in range(32, -1, -1)]


def _locking_offset_minute():
    rng = np.random.RandomState(2026)
    rows = []
    for di in _LO_BAR_DAYS:
        base = pd.Timestamp(_LO_DATES[di]) + pd.Timedelta("09:31:00")
        price = 100.0 + rng.normal(0, 2)
        for i in range(238):
            t = base + pd.Timedelta(minutes=i)
            price += rng.normal(0, 0.05)
            slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
            erupt = 6.0 if (rng.rand() < 0.06) else 1.0
            vol = slot * erupt * (1.0 + 0.1 * rng.rand())
            w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
            hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
            cl = lo + (hi - lo) * rng.rand()
            rows.append((t, _LO_SYM, price, hi, lo, cl, vol, cl * vol))
    frame = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return normalize_intraday_bars(frame, freq="1min")


LOCKING_OFFSET_MINUTE = _locking_offset_minute()


class LockingOffsetProv:
    def minute_bars(self, symbols, start, end):
        if not symbols:
            return LOCKING_OFFSET_MINUTE.iloc[0:0]
        t = LOCKING_OFFSET_MINUTE.index.get_level_values("time")
        return LOCKING_OFFSET_MINUTE[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]

    def earliest_available(self, symbols):
        return _LO_DATES[0]


def test_locking_offset_is_load_bearing():
    """The BASELINE LOCKING OFFSET produces the load-geometry-free value.

    Ground truth = computing on the WHOLE history at once. With the offset the
    saturation loop expands past the unlocked band and reproduces it exactly;
    dropping the offset (``loaded_days[baseline_days]`` -> ``loaded_days[0]``)
    lets the criterion stop one chunk early with the pool still drawing on days
    whose classification is not final, yielding a silently WRONG value.

    MUTATION rc: with ``baseline_days=0`` forced this test FAILS (rc=1, the
    mutated value != truth); unmutated it passes (rc=0). This replaces the
    earlier honest note that the offset was not isolated by the gap fixtures.
    """
    import factors.materialize as mat
    from factors.compute.minute.binding import minute_raw_from_bars
    from factors.view_lag import minute_decision_cutoff

    factor = factor_registry.build("volume_peak_count_20")
    emit = _LO_DATES[_LO_E_IDX]
    truth_full = minute_raw_from_bars(
        factor, minute_decision_cutoff(LOCKING_OFFSET_MINUTE, decision_time="14:50:00")
    )
    truth = float(truth_full.loc[(emit, _LO_SYM)])

    got = mat.materialize_range(
        factor, view=View.DECISION, symbols=[_LO_SYM], emit_start=emit, emit_end=emit,
        sources=MaterializeSources(minute=LockingOffsetProv()),
    )
    assert float(got.iloc[0]) == truth, (
        "the saturated value must equal the whole-history (load-geometry-free) value"
    )


def test_dropping_the_locking_offset_yields_a_wrong_value(monkeypatch):
    """MUTATION evidence for the offset: forcing ``baseline_days=0`` makes the
    criterion accept the unlocked band and return a value that differs from the
    whole-history truth (measured on this fixture: 438.0 vs the true 416.0)."""
    import factors.materialize as mat
    from factors.compute.minute.binding import minute_raw_from_bars
    from factors.view_lag import minute_decision_cutoff

    factor = factor_registry.build("volume_peak_count_20")
    emit = _LO_DATES[_LO_E_IDX]
    truth = float(
        minute_raw_from_bars(
            factor, minute_decision_cutoff(LOCKING_OFFSET_MINUTE, decision_time="14:50:00")
        ).loc[(emit, _LO_SYM)]
    )
    real = mat._pooled_pool_saturated

    def no_offset(full, bars, emit_start, *, baseline_days, lookback_days, symbols):
        return real(full, bars, emit_start, baseline_days=0,
                    lookback_days=lookback_days, symbols=symbols)

    monkeypatch.setattr(mat, "_pooled_pool_saturated", no_offset)
    mutated = mat.materialize_range(
        factor, view=View.DECISION, symbols=[_LO_SYM], emit_start=emit, emit_end=emit,
        sources=MaterializeSources(minute=LockingOffsetProv()),
    )
    assert float(mutated.iloc[0]) != truth, (
        "dropping the locking offset must produce a load-geometry-dependent value"
    )


def test_disabling_saturation_reintroduces_pooled_divergence(monkeypatch):
    """MUTATION: treating a pooled factor as fixed-depth (saturation OFF) brings
    back the single-fill/batch-fill finite<->NaN divergence -> the saturation
    load is load-bearing. rc=1 (this asserts the divergence) with the mutation;
    rc=0 (the test above) without it."""
    import factors.materialize as mat

    monkeypatch.setattr(mat, "is_valid_day_pooled", lambda factor: False)
    dates = list(DATES[55:75])
    any_diverged = False
    for fid in _POOLED_DIVERGENT:
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            a, b = FactorValueStore(ta), FactorValueStore(tb)
            _fill_single(a, [fid], dates, _sparse_sources())
            _fill_batch(b, [fid], dates, _sparse_sources())
            sa = _store_series(a, fid).sort_index()
            sb = _store_series(b, fid).sort_index()
            av = sa.reindex(sb.index).to_numpy()
            bv = sb.to_numpy()
            if not np.array_equal(np.isnan(av), np.isnan(bv)):
                any_diverged = True
    assert any_diverged, "saturation-off must reintroduce a pooled divergence"


# --------------------------------------------------------------------------- #
# read-layer smoke + read-through
# --------------------------------------------------------------------------- #
def test_panel_equals_cross_section_on_same_store():
    dates = list(DATES[40:52])
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        pan = panel(["momentum_20", "volatility_20"], SYMS,
                    [DecisionPoint(date=d) for d in dates], store=store, sources=src)
        d = dates[6]
        xs = cross_section(["momentum_20", "volatility_20"], SYMS, DecisionPoint(date=d),
                           store=store, sources=src)
        pan_d = pan[pan.index.get_level_values("date") == d].sort_index()
        xs = xs.sort_index()
        assert xs.index.equals(pan_d.index)
        assert np.allclose(xs.to_numpy(), pan_d.to_numpy(), equal_nan=True, rtol=0, atol=0)


def test_read_through_hit_does_not_recompute():
    """A second call over already-covered dates does not re-hit the provider."""
    dates = list(DATES[40:50])
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        panel(["jump_amount_corr_20"], SYMS, [DecisionPoint(date=d) for d in dates],
              store=store, sources=src)
        calls_after_fill = src.minute.calls
        assert calls_after_fill > 0
        # second identical call: store hit -> no new minute read
        panel(["jump_amount_corr_20"], SYMS, [DecisionPoint(date=d) for d in dates],
              store=store, sources=src)
        assert src.minute.calls == calls_after_fill


def test_service_returns_only_universe_symbols():
    dates = list(DATES[40:44])
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        pan = panel(["momentum_20"], [SYMS[0]], [DecisionPoint(date=d) for d in dates],
                    store=store, sources=src)
        got_syms = set(pan.index.get_level_values("symbol"))
        assert got_syms <= {SYMS[0]}


# --------------------------------------------------------------------------- #
# pairing gate
# --------------------------------------------------------------------------- #
def test_illegal_view_basis_pairing_raises():
    """close view scored on exec_to_exec (or decision on close_to_close) is a
    readable construction-time error, not a doc convention (§1.4)."""
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        with pytest.raises(ValueError, match="illegal view/basis pairing"):
            cross_section(["momentum_20"], SYMS, DecisionPoint(date=DATES[40]),
                          store=store, sources=src, view=View.CLOSE, basis=ReturnBasis.EXEC_TO_EXEC)
        with pytest.raises(ValueError, match="illegal view/basis pairing"):
            cross_section(["momentum_20"], SYMS, DecisionPoint(date=DATES[40]),
                          store=store, sources=src, view=View.DECISION, basis=ReturnBasis.CLOSE_TO_CLOSE)


def test_legal_pairings_are_accepted():
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        # decision <-> exec_to_exec (default) and close <-> close_to_close
        cross_section(["momentum_20"], SYMS, DecisionPoint(date=DATES[40]),
                      store=store, sources=src)
        cross_section(["momentum_20"], SYMS, DecisionPoint(date=DATES[40]),
                      store=store, sources=src, view=View.CLOSE, basis=ReturnBasis.CLOSE_TO_CLOSE)


def test_panel_requires_uniform_cutoff():
    src = _sources()
    with tempfile.TemporaryDirectory() as td:
        store = FactorValueStore(td)
        with pytest.raises(ValueError, match="share one cutoff"):
            panel(["momentum_20"], SYMS,
                  [DecisionPoint(date=DATES[40], cutoff="14:50:00"),
                   DecisionPoint(date=DATES[41], cutoff="14:45:00")],
                  store=store, sources=src)
