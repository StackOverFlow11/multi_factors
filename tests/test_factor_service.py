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


def _sources():
    return MaterializeSources(daily=DailyProv(), minute=MinuteProv())


def _sparse_sources():
    return MaterializeSources(minute=SparseMinuteProv())


def _store_series(store, factor_id, view=View.DECISION):
    factor = factor_registry.build(factor_id)
    key = store_key(factor, view=view.value)
    return store.read(key)


def _fill_single(store, factor_ids, dates, src):
    for d in dates:
        cross_section(factor_ids, SYMS, DecisionPoint(date=d), store=store, sources=src)


def _fill_batch(store, factor_ids, dates, src):
    panel(factor_ids, SYMS, [DecisionPoint(date=d) for d in dates], store=store, sources=src)


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
