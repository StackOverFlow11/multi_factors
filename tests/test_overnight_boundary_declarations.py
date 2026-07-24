"""NET-NEW P10: ``overnight_boundary`` declarations are TESTABLE (D2, D0 §1.2).

The D0 catalogue stamped this property NET-NEW and pinned it to D2: every
factor's ``overnight_boundary`` declaration must be checkable by the UNIFORM
BASIS RESCALE of D0 §1.2's reading (JC2) — multiply the symbol's entire
``< d`` history by a constant λ, simulating ``d`` being an ex-date:

* minute factors: scale the raw PRICE channels (open/high/low/close) of every
  bar strictly before day ``d`` — and ONLY those. ``amount`` is traded VALUE
  in RMB, which no split/dividend rescales (PR-M's documented fact), and
  ``volume`` is deliberately untouched — the volume-channel split pollution is
  OUTSIDE both taxonomy axes (D0 note 3) and rescaling it would disturb the
  peak classification this axis does not judge. (A first draft of this file
  scaled ``amount`` too and the jump factor's cross-day pooled correlation
  rightly moved — that draft was simulating a rescale no real ex-date
  performs; the D0 §1.2 wording, prices only, is the faithful probe.)
* daily factors: scale the RAW prices of every row before ``d`` by λ and the
  ``adj_factor`` by 1/λ (that is exactly what an ex-date does to the raw
  series), then run the REAL ``front_adjust`` -> compute chain.

Declared ``none`` -> the day-``d`` value must NOT move (no raw-price
comparison crosses the boundary). Declared ``crossed_disclosed`` -> the value
MUST move detectably (the positive control that keeps the invariance tests
honest — §3.5's pair-migration rule). λ = 0.5 is a power of two, so every
within-day ratio rescales EXACTLY in IEEE arithmetic and the ``none`` checks
can assert bit-equality, not approx.

Why NOT a single-point perturbation: any multi-day-lookback factor depends on
``d-1`` data, so perturbing one number fails everything and the axis loses all
discriminating power (D0 §1.2's explicit caution). The uniform rescale is the
one probe that isolates "does the BASIS BREAK enter the value".

Mutation evidence (run for this commit, recorded in the acceptance report):
making ridge_minute_return's lag CROSS the day boundary (dropping the
``trade_date`` grouping in its ``prev_close``) fails
``test_ridge_minute_return_is_invariant_to_a_pre_d_basis_rescale`` (rc=1) —
the rescale probe detects exactly the boundary-straddling return the ``none``
declaration forbids; restoring the within-day lag passes (rc=0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.adjust import front_adjust
from data.clean.intraday_schema import normalize_intraday_bars
from factors.compute.minute.intraday_amp_cut import (
    IntradayAmpCutFactor,
    compute_amp_cut_stats,
)
from factors.compute.minute.jump_amount_corr import (
    JumpAmountCorrFactor,
    compute_jump_amount_corr,
)
from factors.compute.minute.minute_ideal_amplitude import (
    MinuteIdealAmplitudeFactor,
    compute_minute_ideal_amplitude,
)
from factors.compute.minute.ridge_minute_return import (
    RidgeMinuteReturnFactor,
    compute_ridge_minute_return,
)
from factors.compute.minute.valley_price_quantile import (
    ValleyPriceQuantileFactor,
    compute_valley_price_quantile_stats,
)
from factors.compute.momentum import MomentumFactor

_SYM = "000001.SZ"
_LAMBDA = 0.5  # power of two -> within-day ratios rescale EXACTLY


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minute_bars(rows):
    """rows = [(time, symbol, open, high, low, close, volume, amount)] -> bars."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": [float(r[2]) for r in rows],
            "high": [float(r[3]) for r in rows],
            "low": [float(r[4]) for r in rows],
            "close": [float(r[5]) for r in rows],
            "volume": [float(r[6]) for r in rows],
            "amount": [float(r[7]) for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _rescale_before(bars: pd.DataFrame, day: str, lam: float = _LAMBDA) -> pd.DataFrame:
    """The uniform basis rescale: scale the PRICE channels of every bar < ``day``.

    ``amount`` and ``volume`` untouched on purpose (module docstring; D0 §1.2
    wording + note 3 — a real ex-date rescales prices, not traded RMB value).
    """
    out = bars.copy()
    before = out["bar_end"].dt.normalize() < pd.Timestamp(day)
    for col in ("open", "high", "low", "close"):
        out.loc[before, col] = out.loc[before, col] * lam
    return out


def _session(day, specs, sym=_SYM, start="09:31:00"):
    """specs = [(open, high, low, close, volume, amount), ...] -> row tuples."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, *spec) for i, spec in enumerate(specs)
    ]


def _flat_day(day, n, price=100.0, volume=100.0, sym=_SYM):
    """A flat background day: constant price, constant volume (baseline builder)."""
    return _session(
        day, [(price, price, price, price, volume, price * volume)] * n, sym=sym
    )


# --------------------------------------------------------------------------- #
# NONE declarations: the day-d value must NOT move (bit-equal under λ=0.5)
# --------------------------------------------------------------------------- #
def test_ridge_minute_return_is_invariant_to_a_pre_d_basis_rescale():
    """PR-K declares none: within-day lags never straddle the simulated ex-date."""
    assert RidgeMinuteReturnFactor().spec.overnight_boundary == "none"
    n = 12
    kw = dict(min_valid_days=1, min_classifiable=1, min_ridge_bars=1)
    rows = []
    for i in range(10):
        day = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _flat_day(day, n)
    # Two engineered days. Ridge runs at slots (0,1) AND (3,4): the run at the
    # DAY'S FIRST BAR is what gives this probe teeth — under the pinned
    # within-day lag that bar carries no return, but a defective CROSS-DAY lag
    # would price it against the previous (rescaled) day's close and the
    # invariance below would break. Moving closes keep both days valid, so day
    # d's trailing window really contains rescaled (< d) data — the invariance
    # is not vacuously true.
    for day in ("2021-07-11", "2021-07-12"):
        specs = []
        closes = [100.0, 100.0, 100.0, 150.0, 75.0] + [75.0] * (n - 5)
        for i in range(n):
            vol = 200.0 if i in (0, 1, 3, 4) else 100.0
            c = closes[i]
            specs.append((c, c, c, c, vol, c * vol))
        rows += _session(day, specs)

    bars = _minute_bars(rows)
    d = pd.Timestamp("2021-07-12")
    base = compute_ridge_minute_return(bars, **kw)
    rescaled = compute_ridge_minute_return(_rescale_before(bars, "2021-07-12"), **kw)
    assert base.loc[(d, _SYM)] == rescaled.loc[(d, _SYM)]
    # sanity: the window really contains a < d valid day (sum over TWO days:
    # per day r1 = 0, r3 = +0.5, r4 = -0.5 -> 0; slot 0 carries no return)
    assert base.loc[(d, _SYM)] == pytest.approx(0.0)


def test_intraday_amp_cut_stats_are_invariant_to_a_pre_d_basis_rescale():
    """PR-G declares none: amp and the 1-min return are within-day ratios."""
    assert IntradayAmpCutFactor().spec.overnight_boundary == "none"
    n = 8
    kw = dict(min_day_minutes=2, min_valid_days=2, lam=0.25)
    rows = []
    rng = np.random.RandomState(3)
    for day_i, day in enumerate(("2021-07-01", "2021-07-02", "2021-07-05")):
        specs = []
        for i in range(n):
            low = 100.0 + i + day_i
            high = low * (1.0 + 0.01 * (1 + ((i + day_i) % 4)))
            close = low * (1.0 + 0.005 * ((i + 2 * day_i) % 5))
            specs.append((low, high, low, close, 100.0, close * 100.0))
        rows += _session(day, specs)
    bars = _minute_bars(rows)
    del rng
    d = pd.Timestamp("2021-07-05")
    base = compute_amp_cut_stats(bars, **kw)
    rescaled = compute_amp_cut_stats(_rescale_before(bars, "2021-07-05"), **kw)
    assert base.loc[(d, _SYM), "v_mean"] == rescaled.loc[(d, _SYM), "v_mean"]
    assert base.loc[(d, _SYM), "v_std"] == rescaled.loc[(d, _SYM), "v_std"]
    # sanity: the trailing stats really pooled < d days (min_valid_days=2)
    assert np.isfinite(base.loc[(d, _SYM), "v_std"])


def test_jump_amount_corr_is_invariant_to_a_pre_d_basis_rescale():
    """PR-C declares none: the amplitude z-score is a same-day ratio (exactly
    invariant under the rescale) and the correlated quantity is ``amount``,
    which a real ex-date does not rescale — so the jump set and every pooled
    pair are bit-identical."""
    assert JumpAmountCorrFactor().spec.overnight_boundary == "none"
    n = 10
    kw = dict(min_pairs=2)
    rows = []
    for day_i, day in enumerate(("2021-07-01", "2021-07-02", "2021-07-05")):
        specs = []
        for i in range(n):
            open_ = 100.0
            # two clear amplitude jumps per day at slots 3 and 6
            width = 8.0 if i in (3, 6) else 1.0
            high, low = open_ + width, open_ - width
            amount = 1000.0 + 137.0 * ((i * (day_i + 2)) % 7)
            specs.append((open_, high, low, open_, 100.0, amount))
        rows += _session(day, specs)
    bars = _minute_bars(rows)
    d = pd.Timestamp("2021-07-05")
    base = compute_jump_amount_corr(bars, **kw)
    rescaled = compute_jump_amount_corr(_rescale_before(bars, "2021-07-05"), **kw)
    assert base.loc[(d, _SYM)] == rescaled.loc[(d, _SYM)]
    assert np.isfinite(base.loc[(d, _SYM)])


def test_momentum_is_invariant_to_a_simulated_ex_date_through_front_adjust():
    """momentum_20 declares none: both legs sit on one continuous qfq basis.

    The daily-side probe runs the REAL composed chain: raw prices + adj_factor
    -> front_adjust -> compute. Simulating an ex-date at ``d`` multiplies the
    raw prices BEFORE d by λ and their adj_factor by 1/λ — front_adjust must
    absorb the break so the day-d momentum is bit-identical.
    """
    factor = MomentumFactor(window=5)
    assert factor.spec.overnight_boundary == "none"
    dates = pd.bdate_range("2024-01-02", periods=10)
    closes = 100.0 + np.arange(10, dtype=float) * 2.0
    index = pd.MultiIndex.from_product([dates, [_SYM]], names=["date", "symbol"])
    panel = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": 100.0,
            "amount": closes * 100.0,
            "adj_factor": 1.0,
        },
        index=index,
    )
    d = dates[-1]
    base = factor.compute(front_adjust(panel))

    exdate = panel.copy()
    before = exdate.index.get_level_values("date") < d
    for col in ("open", "high", "low", "close"):
        exdate.loc[before, col] = exdate.loc[before, col] * _LAMBDA
    exdate.loc[before, "adj_factor"] = exdate.loc[before, "adj_factor"] / _LAMBDA
    rescaled = factor.compute(front_adjust(exdate))
    assert base.loc[(d, _SYM)] == rescaled.loc[(d, _SYM)]
    assert np.isfinite(base.loc[(d, _SYM)])


# --------------------------------------------------------------------------- #
# CROSSED_DISCLOSED declarations: the value MUST move (positive controls)
# --------------------------------------------------------------------------- #
def test_valley_price_quantile_moves_under_a_pre_d_basis_rescale():
    """PR-L declares crossed_disclosed: prev_close enters day d's range on the
    OLD basis, so the simulated ex-date must move the day-d value detectably."""
    assert ValleyPriceQuantileFactor().spec.overnight_boundary == "crossed_disclosed"
    n = 8
    kw = dict(min_valid_days=1, min_classifiable=1, min_valley_bars=1)
    rows = []
    for i in range(10):
        day = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _flat_day(day, n, price=105.0)
    # test day: prices 100..110, prev_close (105) sits INSIDE the day's range,
    # so before the rescale it does not widen the range; after λ=0.5 it drops
    # to 52.5 and stretches the low end -> q_day must change.
    specs = []
    for i in range(n):
        low = 100.0 + i
        high = low + 2.0
        specs.append((low, high, low, low + 1.0, 100.0, (low + 1.0) * 100.0))
    rows += _session("2021-07-12", specs)
    bars = _minute_bars(rows)
    d = pd.Timestamp("2021-07-12")
    base = compute_valley_price_quantile_stats(bars, **kw)
    rescaled = compute_valley_price_quantile_stats(
        _rescale_before(bars, "2021-07-12"), **kw
    )
    assert np.isfinite(base.loc[(d, _SYM)]) and np.isfinite(rescaled.loc[(d, _SYM)])
    assert base.loc[(d, _SYM)] != rescaled.loc[(d, _SYM)]


def test_minute_ideal_amplitude_moves_under_a_pre_d_basis_rescale():
    """PR-D declares crossed_disclosed: the pooled ranking key is the RAW close
    across the multi-day window, so rescaling the < d days re-interleaves the
    pool and the top/bottom cut must move (the D0 note-1 candidate confirmed)."""
    assert MinuteIdealAmplitudeFactor().spec.overnight_boundary == "crossed_disclosed"
    n = 6
    kw = dict(min_minutes=4, lam=0.25)
    rows = []
    # day A: HIGH closes (~200) with SMALL amps; day B (= d): closes ~100 with
    # LARGE amps. Before the rescale the pooled top-k by close comes from day A
    # (small amps); after halving day A's prices its closes sit at ~100 and the
    # top-k re-interleaves -> V_high - V_low must change.
    specs_a = []
    for i in range(n):
        low = 200.0 + 2.0 * i
        high = low * 1.001
        specs_a.append((low, high, low, low + 1.0, 100.0, (low + 1.0) * 100.0))
    rows += _session("2021-07-09", specs_a)
    specs_b = []
    for i in range(n):
        low = 100.0 + 2.0 * i
        high = low * (1.0 + 0.05 * (i + 1))
        specs_b.append((low, high, low, low + 1.5, 100.0, (low + 1.5) * 100.0))
    rows += _session("2021-07-12", specs_b)
    bars = _minute_bars(rows)
    d = pd.Timestamp("2021-07-12")
    base = compute_minute_ideal_amplitude(bars, **kw)
    rescaled = compute_minute_ideal_amplitude(
        _rescale_before(bars, "2021-07-12"), **kw
    )
    assert np.isfinite(base.loc[(d, _SYM)]) and np.isfinite(rescaled.loc[(d, _SYM)])
    assert base.loc[(d, _SYM)] != rescaled.loc[(d, _SYM)]
