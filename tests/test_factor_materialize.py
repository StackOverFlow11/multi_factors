"""D4: the materializer engine (factors.materialize) — R18 lag + leakage + binding.

Network-free. Covers, against the §九 D4 acceptance:
* R18 full lag: daily prev-day fields at ``<= d-1``, field-level ``open`` same-day,
  daily_basic/enriched at ``<= d-1``, minute ``available_time <= d 14:50``;
* decision-after perturbation leaves the decision-view value bit-unchanged, and
  perturbing ``open[d]`` DOES change it (the reverse teeth);
* the minute binding is correct (close-view materialize == direct compute_X);
* ex-date masking (a fixture ``masked`` factor; the 14 closing factors are all
  non-masked so this path is a no-op for them);
* the live-config wiring of ``build_horizon_config`` (no default fallback, R5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import View
from data.clean.intraday_schema import normalize_intraday_bars
from factors.base import Factor
from factors.compute.candidates import OvernightMomentumFactor, ValueFactor, VolatilityFactor
from factors.compute.minute.jump_amount_corr import (
    JumpAmountCorrFactor,
    compute_jump_amount_corr,
)
from factors.materialize import (
    MaterializeSources,
    build_horizon_config,
    is_minute_factor,
    make_recompute_fn,
    materialize_range,
)
from factors.spec import FactorSpec, PanelField

SYMS = ["000001.SZ", "000002.SZ"]
DATES = pd.bdate_range("2021-01-04", periods=70)


# --------------------------------------------------------------------------- #
# Synthetic data + fake providers
# --------------------------------------------------------------------------- #
def _daily(perturb=None):
    """Enriched close-view daily panel (+ a value_ep column) for the fakes.

    ``perturb`` = (date, symbol, column, new_value) mutates one cell.
    """
    rng = np.random.RandomState(0)
    rows = []
    for si, s in enumerate(SYMS):
        px = 100.0 + si * 20 + np.cumsum(rng.normal(0, 1.0, len(DATES)))
        for d, p in zip(DATES, px):
            rows.append((d, s, p - 0.3, p + 0.5, p - 0.5, p, 1e5, p * 1e5, 1.0 / (10.0 + p * 0.01)))
    frame = pd.DataFrame(
        rows, columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount", "value_ep"]
    ).set_index(["date", "symbol"]).sort_index()
    if perturb is not None:
        d, s, col, val = perturb
        frame.loc[(d, s), col] = val
    return frame


def _minute(perturb=None):
    """Two symbols, 40 days, a session spanning across 14:50 so the cutoff bites."""
    rng = np.random.RandomState(1)
    rows = []
    for s in SYMS:
        for d in DATES[:40]:
            # bars from 14:40 to 14:59 (across the 14:50 cutoff)
            base = pd.Timestamp(d) + pd.Timedelta("14:40:00")
            for i in range(20):
                t = base + pd.Timedelta(minutes=i)
                o = 100.0 + rng.normal(0, 1)
                w = 4.0 if i in (2, 6, 12) else 0.5
                rows.append((t, s, o, o + w, o - w, o + rng.normal(0, 0.2), 1e4, 1e6 + rng.normal(0, 1e5)))
    frame = pd.DataFrame(
        rows, columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    )
    bars = normalize_intraday_bars(frame, freq="1min")
    if perturb is not None:
        t, s, col, val = perturb
        bars.loc[(pd.Timestamp(t), s), col] = val
    return bars


class DailyProv:
    def __init__(self, panel):
        self._p = panel

    def daily_panel(self, symbols, start, end):
        m = self._p.index.get_level_values("date")
        return self._p[(m >= pd.Timestamp(start)) & (m <= pd.Timestamp(end))]


class MinuteProv:
    def __init__(self, bars):
        self._b = bars

    def minute_bars(self, symbols, start, end):
        if not symbols:
            return self._b.iloc[0:0]
        t = self._b.index.get_level_values("time")
        return self._b[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]


def _sources(daily=None, minute=None, adj=None):
    return MaterializeSources(
        daily=DailyProv(daily) if daily is not None else None,
        minute=MinuteProv(minute) if minute is not None else None,
        adj_factor=adj,
    )


# --------------------------------------------------------------------------- #
# Factor kind classification
# --------------------------------------------------------------------------- #
def test_is_minute_factor():
    assert is_minute_factor(JumpAmountCorrFactor())
    assert not is_minute_factor(VolatilityFactor(window=20))
    assert not is_minute_factor(ValueFactor("value_ep"))


# --------------------------------------------------------------------------- #
# Daily decision-view materialization + R18 daily lag
# --------------------------------------------------------------------------- #
def test_daily_decision_materialize_is_finite():
    src = _sources(daily=_daily())
    v = materialize_range(VolatilityFactor(window=20), view=View.DECISION, symbols=SYMS,
                          emit_start=DATES[40], emit_end=DATES[55], sources=src)
    assert np.isfinite(v.to_numpy()).all() and len(v) == 16 * 2


def test_r18_close_is_prev_day_open_is_same_day():
    """A close-reading factor's decision value at d is UNCHANGED when close[d]
    is perturbed (decision uses close[d-1]); an open-reading factor's value at d
    DOES change when open[d] is perturbed (open is field-level same-day, R7)."""
    d = DATES[55]
    base_daily = _daily()
    src0 = _sources(daily=base_daily)

    # (a) close[d] perturbation -> decision-view volatility at d bit-unchanged.
    vol = VolatilityFactor(window=20)
    v0 = materialize_range(vol, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src0)
    src_close = _sources(daily=_daily(perturb=(d, SYMS[0], "close", 999.0)))
    v1 = materialize_range(vol, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src_close)
    assert v0.loc[(d, SYMS[0])] == v1.loc[(d, SYMS[0])]  # close[d] not visible at 14:50

    # (b) open[d] perturbation -> overnight (reads open same-day) at d CHANGES.
    ov = OvernightMomentumFactor(window=20)
    o0 = materialize_range(ov, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src0)
    src_open = _sources(daily=_daily(perturb=(d, SYMS[0], "open", 999.0)))
    o1 = materialize_range(ov, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src_open)
    assert o0.loc[(d, SYMS[0])] != o1.loc[(d, SYMS[0])]  # open[d] IS visible (reverse teeth)

    # (c) close[d] perturbation -> overnight at d unchanged (it never reads close[d]).
    o2 = materialize_range(ov, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src_close)
    assert o0.loc[(d, SYMS[0])] == o2.loc[(d, SYMS[0])]


def test_r18_daily_basic_enriched_column_is_prev_day():
    """value_ep (daily_basic-derived) at d uses the d-1 published ratio (<= d-1)."""
    d = DATES[55]
    src0 = _sources(daily=_daily())
    vf = ValueFactor("value_ep")
    v0 = materialize_range(vf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src0)
    # perturbing value_ep[d] must NOT change the decision value at d (uses d-1)...
    src_d = _sources(daily=_daily(perturb=(d, SYMS[0], "value_ep", 999.0)))
    v_d = materialize_range(vf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src_d)
    assert v0.loc[(d, SYMS[0])] == v_d.loc[(d, SYMS[0])]
    # ...but perturbing value_ep[d-1] MUST change it.
    src_p = _sources(daily=_daily(perturb=(DATES[54], SYMS[0], "value_ep", 999.0)))
    v_p = materialize_range(vf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d, sources=src_p)
    assert v0.loc[(d, SYMS[0])] != v_p.loc[(d, SYMS[0])]


# --------------------------------------------------------------------------- #
# Minute cutoff + leakage + binding correctness
# --------------------------------------------------------------------------- #
def test_minute_cutoff_post_1450_bar_does_not_leak():
    """Perturbing a post-14:50 bar leaves the decision-view value bit-unchanged;
    perturbing a pre-14:50 bar changes it (the cutoff is load-bearing)."""
    d = DATES[35]
    base = _minute()
    src0 = _sources(minute=base)
    jf = JumpAmountCorrFactor()
    j0 = materialize_range(jf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d,
                           sources=src0, decision_cutoff="14:50:00")
    # a 14:55 bar (available_time 14:56 > 14:50) is filtered out -> no effect.
    post_t = pd.Timestamp(d) + pd.Timedelta("14:55:00")
    src_post = _sources(minute=_minute(perturb=(post_t, SYMS[0], "amount", 9e9)))
    j_post = materialize_range(jf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d,
                               sources=src_post, decision_cutoff="14:50:00")
    assert j0.equals(j_post) or np.array_equal(j0.to_numpy(), j_post.to_numpy(), equal_nan=True)
    # a 14:42 bar (available_time 14:43 <= 14:50) is inside the cutoff -> changes.
    pre_t = pd.Timestamp(d) + pd.Timedelta("14:42:00")
    src_pre = _sources(minute=_minute(perturb=(pre_t, SYMS[0], "amount", 9e9)))
    j_pre = materialize_range(jf, view=View.DECISION, symbols=SYMS, emit_start=d, emit_end=d,
                              sources=src_pre, decision_cutoff="14:50:00")
    assert not np.array_equal(j0.to_numpy(), j_pre.to_numpy(), equal_nan=True)


def test_minute_binding_close_view_matches_direct_compute():
    """close-view minute materialize == the free compute_* on the same bar window
    (proves the binding calls the right function with the right constants)."""
    src = _sources(minute=_minute())
    jf = JumpAmountCorrFactor()
    got = materialize_range(jf, view=View.CLOSE, symbols=SYMS, emit_start=DATES[30], emit_end=DATES[38], sources=src)
    win = MinuteProv(_minute()).minute_bars(
        SYMS, DATES[30] - pd.Timedelta(days=60), DATES[38] + pd.Timedelta(days=1)
    )
    direct = compute_jump_amount_corr(win, lookback_days=jf.lookback_days, name=jf.name)
    dd = direct.index.get_level_values("date")
    direct_win = direct[(dd >= DATES[30]) & (dd <= DATES[38])].sort_index()
    got = got.reindex(direct_win.index)
    # A wrong binding (wrong function or wrong constant) would differ by O(1); the
    # only residual is the pandas rolling-sum accumulation over a differently-sized
    # loaded window (~1e-15, an attributable float-reorder, §五), so a tight
    # tolerance proves the binding while allowing the reorder.
    assert np.allclose(got.to_numpy(), direct_win.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# ex-date masking (fixture masked factor; the 14 are non-masked, no-op)
# --------------------------------------------------------------------------- #
class _MaskedDailyFactor(Factor):
    """Fixture-only masked factor: surfaces close; NaN'd on ex-dates by the engine."""

    name = "masked_fixture"

    @property
    def spec(self) -> FactorSpec:
        return FactorSpec(
            factor_id="masked_fixture", version="1.0", description="fixture masked factor",
            expected_ic_sign=1, is_intraday=False, forward_return_horizon=1,
            return_basis="close_to_close", input_fields=("close",),
            requires=(PanelField("close", source="market_daily"),),
            adjustment="returns_invariant", overnight_boundary="masked",
            family="fixture", min_history_bars=0, lookback_depth=1,
        )

    def compute(self, panel):
        return panel["close"].rename(self.name)


class AdjProv:
    def __init__(self, series):
        self._s = series

    def adj_factor(self, symbols, start, end):
        m = self._s.index.get_level_values("date")
        return self._s[(m >= pd.Timestamp(start)) & (m <= pd.Timestamp(end))]


def test_ex_date_masking_nans_the_ex_date_row():
    """A masked factor's ex-date (d, symbol) is NaN; non-masked factors skip this."""
    d_ex = DATES[50]
    # adj_factor jumps on d_ex for SYMS[0] only.
    af_rows = []
    for s in SYMS:
        for di, d in enumerate(DATES):
            af = 1.0 if (s != SYMS[0] or d < d_ex) else 2.0
            af_rows.append((d, s, af))
    af = pd.DataFrame(af_rows, columns=["date", "symbol", "af"]).set_index(["date", "symbol"])["af"]
    src = _sources(daily=_daily(), adj=AdjProv(af))
    got = materialize_range(_MaskedDailyFactor(), view=View.DECISION, symbols=SYMS,
                            emit_start=DATES[48], emit_end=DATES[52], sources=src)
    assert np.isnan(got.loc[(d_ex, SYMS[0])])            # ex-date masked
    assert np.isfinite(got.loc[(d_ex, SYMS[1])])         # other symbol untouched
    assert np.isfinite(got.loc[(DATES[49], SYMS[0])])    # non-ex-date untouched


def test_non_masked_factor_never_consults_adj_provider():
    """The 14 closing factors are non-masked: the ex-date path is a no-op (no
    AdjFactorProvider needed)."""
    src = _sources(daily=_daily())  # NO adj provider
    v = materialize_range(VolatilityFactor(window=20), view=View.DECISION, symbols=SYMS,
                          emit_start=DATES[45], emit_end=DATES[46], sources=src)
    assert len(v) == 4  # ran fine without an adj provider


# --------------------------------------------------------------------------- #
# recompute-fn factory (D3 tail-recompute compatibility)
# --------------------------------------------------------------------------- #
def test_make_recompute_fn_matches_materialize_range():
    src = _sources(daily=_daily())
    vol = VolatilityFactor(window=20)
    fn = make_recompute_fn(vol, view=View.DECISION, symbols=SYMS, sources=src, data_start=DATES[0])
    a = fn(DATES[40], DATES[50], vol.spec.lookback_depth)
    b = materialize_range(vol, view=View.DECISION, symbols=SYMS,
                          emit_start=DATES[40], emit_end=DATES[50], sources=src)
    assert np.array_equal(a.sort_index().to_numpy(), b.sort_index().to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# live-config wiring (R5): no default fallback
# --------------------------------------------------------------------------- #
class _LiveCacheCfg:
    refresh_recent_days = 14
    fina_tail_days = 400


class _PartialCacheCfg:
    refresh_recent_days = 14  # fina_tail_days missing


def test_build_horizon_config_from_live_config():
    hc = build_horizon_config(_LiveCacheCfg())
    assert hc.refresh_recent_days == 14
    assert hc.fina_tail_days == 400
    assert hc.overrides() == {"fina_indicator": 400}


def test_build_horizon_config_refuses_missing_attr():
    with pytest.raises(ValueError, match="fina_tail_days"):
        build_horizon_config(_PartialCacheCfg())
