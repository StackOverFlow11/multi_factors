"""#adjustment declaration property tests — NET-NEW pinned to D3 (D0 §3.4 P9).

The store fingerprint is DERIVED FROM the ``adjustment`` declaration (Commit 2),
so the declaration must be TESTABLE:

* ``returns_invariant`` — perturbing the price-adjustment anchor leaves the factor
  value bit-for-bit unchanged. Concretely tested end-to-end on a DAILY factor
  (volatility_20, qfq-anchor rescale) AND on a MINUTE factor (valley_relative_vwap,
  the within-day VWAP ratio under a per-day price rescale), and asserted at the
  FINGERPRINT level for ALL 9 returns_invariant closing factors (a returns_invariant
  fingerprint must OMIT the adj-factor event table — the whole point of the
  declaration).
* ``price_level`` — the mechanism is fixture-tested (zero closing-set members is NOT
  waived, A5-F02): the fingerprint INCLUDES the per-symbol adj-factor event hash and
  changes when the anchor is restated; mislabelling it ``returns_invariant`` makes
  the fingerprint blind to the anchor (the §六.17 stale-reuse trap).
* ``none`` — the input-list static check (adjustment='none' cannot require a price
  channel) is enforced at construction; here we assert it holds for the 5 none
  closing factors and their fingerprint omits the anchor too.

Each invariance test carries mutation evidence (a positive control that the anchor
rescale is non-trivial / that a level-dependent computation WOULD be caught).

Network-free (synthetic panels + synthetic 1min bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import Adjustment
from data.clean.intraday_schema import normalize_intraday_bars
from factors.base import Factor
from factors.compute.candidates import ValueFactor, VolatilityFactor
from factors.compute.minute.amp_marginal_anomaly_vol import AmpMarginalAnomalyVolFactor
from factors.compute.minute.intraday_amp_cut import IntradayAmpCutFactor
from factors.compute.minute.jump_amount_corr import JumpAmountCorrFactor
from factors.compute.minute.minute_ideal_amplitude import MinuteIdealAmplitudeFactor
from factors.compute.minute.peak_interval_kurtosis import PeakIntervalKurtosisFactor
from factors.compute.minute.peak_ridge_amount_ratio import PeakRidgeAmountRatioFactor
from factors.compute.minute.ridge_minute_return import RidgeMinuteReturnFactor
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.compute.minute.valley_relative_vwap import (
    VALLEY_VWAP_LOOKBACK_DAYS,
    ValleyRelativeVwapFactor,
    compute_valley_relative_vwap,
)
from factors.compute.minute.valley_ridge_vwap_ratio import ValleyRidgeVwapRatioFactor
from factors.compute.minute.volume_peak_count import VolumePeakCountFactor
from factors.compute.minute.primitives import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_SIGMA_K,
)
from factors.ops import ts_std
from factors.spec import FactorSpec, PanelField
from factors.store import adj_events_hash, data_fingerprint


# --------------------------------------------------------------------------- #
# the 14 closing factors, by class default
# --------------------------------------------------------------------------- #
def _closing_factors():
    return [
        JumpAmountCorrFactor(),
        MinuteIdealAmplitudeFactor(),
        AmpMarginalAnomalyVolFactor(),
        VolumePeakCountFactor(),
        IntradayAmpCutFactor(),
        PeakIntervalKurtosisFactor(),
        ValleyRelativeVwapFactor(),
        ValleyRidgeVwapRatioFactor(),
        RidgeMinuteReturnFactor(),
        ValleyPriceQuantileFactor(),
        PeakRidgeAmountRatioFactor(),
        ValueFactor("value_ep"),
        ValueFactor("value_bp"),
        VolatilityFactor(20),
    ]


def test_closing_set_adjustment_census_matches_d0():
    factors = _closing_factors()
    ri = [f for f in factors if f.spec.adjustment is Adjustment.RETURNS_INVARIANT]
    none = [f for f in factors if f.spec.adjustment is Adjustment.NONE]
    price = [f for f in factors if f.spec.adjustment is Adjustment.PRICE_LEVEL]
    assert len(factors) == 14
    assert len(ri) == 9  # D0 §二: 9 returns_invariant
    assert len(none) == 5  # D0 §二: 5 none
    assert len(price) == 0  # D0 §二: 0 price_level (mechanism stays fixture-tested)


def test_returns_invariant_fingerprints_omit_the_anchor_batch():
    # The D3 mechanism applied to ALL 9 returns_invariant closing factors: their
    # data fingerprint must NOT include the adj-factor event table (safe to reuse
    # a stored value across an ex-date, because the anchor cancels in the ratio).
    for f in _closing_factors():
        if f.spec.adjustment is Adjustment.RETURNS_INVARIANT:
            fp = data_fingerprint(adjustment=f.spec.adjustment)
            assert fp["adj_events"] is None, f.name


def test_none_factors_omit_the_anchor_and_pass_the_input_list_check_batch():
    for f in _closing_factors():
        if f.spec.adjustment is Adjustment.NONE:
            # constructed successfully => the D0 §1.1 input-list static check held
            # (adjustment='none' cannot require a price-channel field), enforced in
            # FactorSpec._check_declarations at construction time.
            fp = data_fingerprint(adjustment=f.spec.adjustment)
            assert fp["adj_events"] is None, f.name


# --------------------------------------------------------------------------- #
# returns_invariant runtime — DAILY (volatility_20, real factor)
# --------------------------------------------------------------------------- #
def _daily_close_panel(n=40, symbols=("AAA", "BBB")):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(11)
    rows = {}
    for sym in symbols:
        prices = 10.0 + np.cumsum(rng.normal(0, 0.2, size=n))
        prices = np.abs(prices) + 1.0  # strictly positive
        for d, p in zip(dates, prices):
            rows[(d, sym)] = float(p)
    idx = pd.MultiIndex.from_tuples(list(rows), names=["date", "symbol"])
    return pd.DataFrame({"close": list(rows.values())}, index=idx)


def _rescale_close_per_symbol(panel, factors):
    out = panel.copy()
    close = out["close"].copy()
    for sym, lam in factors.items():
        mask = close.index.get_level_values("symbol") == sym
        close[mask] = close[mask] * lam
    out["close"] = close
    return out


# Power-of-two anchor factors: a qfq anchor rescale is multiplicative, and for a
# genuinely returns_invariant factor the anchor cancels MATHEMATICALLY; a power-of-two
# multiplier makes the cancellation EXACT in float (only the exponent shifts, the
# mantissa is untouched), so the invariance is testable bit-for-bit rather than to a
# tolerance. A level-dependent factor is still moved by it (the teeth test below).
_ANCHOR = {"AAA": 0.5, "BBB": 4.0}


def test_volatility_is_invariant_to_a_qfq_anchor_rescale():
    panel = _daily_close_panel()
    factor = VolatilityFactor(20)
    base = factor.compute(panel)
    scaled = factor.compute(_rescale_close_per_symbol(panel, _ANCHOR))
    assert np.array_equal(base.to_numpy(), scaled.to_numpy(), equal_nan=True)


def test_the_anchor_rescale_is_nontrivial_price_level_would_be_caught():
    # MUTATION / teeth: a price-LEVEL statistic (rolling std of the raw level, not of
    # returns) IS moved by the same multiplicative rescale, so the test above is not
    # vacuously true — it would catch a level-dependent factor.
    panel = _daily_close_panel()
    level_std = ts_std(panel["close"], 20)
    level_std_scaled = ts_std(_rescale_close_per_symbol(panel, _ANCHOR)["close"], 20)
    assert not np.array_equal(
        level_std.to_numpy(), level_std_scaled.to_numpy(), equal_nan=True
    )


# --------------------------------------------------------------------------- #
# returns_invariant runtime — MINUTE (valley_relative_vwap within-day ratio)
# --------------------------------------------------------------------------- #
_SYM = "000001.SZ"
_TEST_DAY = pd.Timestamp("2021-07-11")
_BG_DAYS = 10
_CASE_N = 12
_CASE_ERUPT = (3, 7)
_VWAP_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=VALLEY_VWAP_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_valley_bars=1,  # the single engineered day is the only valid one
)


def _bars(rows):
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": [float(r[2]) for r in rows],
            "amount": [float(r[3]) for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, vols, amts, start="09:31:00"):
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), _SYM, v, a)
        for i, (v, a) in enumerate(zip(vols, amts))
    ]


def _background(n_days, n_slots):
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, [1000.0] * n_slots)
    return rows


def _case_day():
    vols = [100.0] * _CASE_N
    amts = [0.0] * _CASE_N
    valley_slots = [s for s in range(_CASE_N) if s not in _CASE_ERUPT]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 5 else 1200.0
    for s in _CASE_ERUPT:
        vols[s] = 200.0
        amts[s] = 4000.0
    return vols, amts


def test_valley_vwap_ratio_is_invariant_to_a_per_day_price_rescale():
    # price_per_bar = amount / volume; scaling a whole DAY's amounts by a constant
    # (== scaling the day's price level by that constant, an ex-date adjustment)
    # leaves the within-day VWAP RATIO unchanged (both legs scale together).
    vols, amts = _case_day()
    rows = _background(_BG_DAYS, _CASE_N) + _session("2021-07-11", vols, amts)
    base = compute_valley_relative_vwap(_bars(rows), **_VWAP_KW).loc[(_TEST_DAY, _SYM)]

    scaled_amts = [a * 4.0 for a in amts]  # uniform within-day price rescale (power of 2)
    rows_scaled = _background(_BG_DAYS, _CASE_N) + _session("2021-07-11", vols, scaled_amts)
    scaled = compute_valley_relative_vwap(_bars(rows_scaled), **_VWAP_KW).loc[(_TEST_DAY, _SYM)]
    assert scaled == base  # bit-for-bit


def test_within_day_nonuniform_rescale_does_change_the_ratio_teeth():
    # MUTATION / teeth: a NON-uniform within-day price change (only the valley bars)
    # is NOT an anchor adjustment and DOES move the ratio, so the invariance above is
    # not vacuous.
    vols, amts = _case_day()
    rows = _background(_BG_DAYS, _CASE_N) + _session("2021-07-11", vols, amts)
    base = compute_valley_relative_vwap(_bars(rows), **_VWAP_KW).loc[(_TEST_DAY, _SYM)]

    tweaked = list(amts)
    for s in range(_CASE_N):
        if s not in _CASE_ERUPT:
            tweaked[s] = tweaked[s] * 2.0  # only valleys -> ratio must move
    rows_tweaked = _background(_BG_DAYS, _CASE_N) + _session("2021-07-11", vols, tweaked)
    tweaked_val = compute_valley_relative_vwap(_bars(rows_tweaked), **_VWAP_KW).loc[
        (_TEST_DAY, _SYM)
    ]
    assert tweaked_val != base


# --------------------------------------------------------------------------- #
# price_level FIXTURE mechanism (A5-F02: zero members NOT waived)
# --------------------------------------------------------------------------- #
class _PriceLevelFixtureFactor(Factor):
    name = "fixture_price_level"
    spec = FactorSpec(
        factor_id="fixture_price_level",
        version="1.0",
        description="fixture: depends on the adjusted price LEVEL",
        expected_ic_sign=1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="market_daily"),),
        adjustment="price_level",
        overnight_boundary="none",
    )

    def compute(self, panel):  # pragma: no cover - fixture
        return panel["close"].rename(self.name)


def _adj_series(values, dates=("2024-01-02", "2024-01-03")):
    return pd.Series(list(values), index=pd.to_datetime(list(dates)))


def test_price_level_fingerprint_includes_and_tracks_the_anchor():
    factor = _PriceLevelFixtureFactor()
    assert factor.spec.adjustment is Adjustment.PRICE_LEVEL
    events = {"AAA": _adj_series([1.0, 1.0])}
    fp = data_fingerprint(adjustment=factor.spec.adjustment, adj_events=events)
    assert fp["adj_events"]["AAA"] == adj_events_hash(events["AAA"])

    # perturb the anchor (an ex-date restates the cumulative factor) -> the
    # fingerprint changes, so the stored value is correctly invalidated on read.
    restated = {"AAA": _adj_series([1.0, 2.0])}
    fp2 = data_fingerprint(adjustment=factor.spec.adjustment, adj_events=restated)
    assert fp2["adj_events"]["AAA"] != fp["adj_events"]["AAA"]


def test_mislabelling_price_level_as_returns_invariant_hides_the_anchor():
    # MUTATION / §六.17 trap: if the SAME factor were (wrongly) declared
    # returns_invariant, the fingerprint would ignore the anchor entirely, so an
    # ex-date restatement produces the SAME fingerprint -> a stale value would be
    # reused. This is exactly why the declaration must be testable + correct.
    events = {"AAA": _adj_series([1.0, 1.0])}
    # returns_invariant refuses the anchor outright (a contradiction, Commit 2), so a
    # mislabel cannot even carry the anchor into the fingerprint:
    with pytest.raises(ValueError):
        data_fingerprint(adjustment="returns_invariant", adj_events=events)
    # and WITHOUT the anchor the fingerprint is identical no matter how the anchor is
    # later restated (an ex-date) -> a stale value would be reused (the trap):
    fp_before_exdate = data_fingerprint(adjustment="returns_invariant")
    fp_after_exdate = data_fingerprint(adjustment="returns_invariant")
    assert fp_before_exdate == fp_after_exdate
