"""PR-E: amplitude marginal-anomaly relative-volatility factor.

The factor DERIVES 5min bars from 1min (``resample_intraday_bars``), PIT-truncates
them at 14:50, forms WITHIN-DAY-lagged ``(|Δamp|, r)`` pairs, pools the trailing
``N`` trading days, selects the bars whose ``|Δamp| > μ + σ`` of the pooled
``|Δamp|``, and returns the ddof=1 std of the RETURNS on the selected bars. Sign is
pre-registered +1 (high anomaly-bar relative vol -> higher forward return).

Hand cases build 1min bars ON the 5min grid (one 1min bar per 5min bucket) so the
derived 5min bar equals the 1min bar and ``amp = high/low - 1`` is exact; a separate
test exercises a multi-bar 5min bucket (resample boundary) and the
``available_time = max(source)`` PIT rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_aggregate import resample_intraday_bars
from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AMP_ANOMALY_FREQ,
    AMP_ANOMALY_LOOKBACK_DAYS,
    AMP_ANOMALY_MIN_POOL,
    AMP_ANOMALY_MIN_SELECTED,
    AMP_ANOMALY_SIGMA_K,
    AmpMarginalAnomalyVolFactor,
    compute_amp_marginal_anomaly_vol,
)
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, high, low, close), ...] -> normalized 1min bars.

    ``open`` is set to ``close`` harmlessly; the factor reads ``high``/``low``
    (per-bar amplitude ``high/low - 1``) and ``close`` (the within-day return),
    decoupled from each other on purpose.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": [r[4] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0] * len(rows),
            "amount": [1.0] * len(rows),
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _grid_session(day, closes, amps, sym=_SYM, start="09:35:00"):
    """One session, one 1min bar PER 5min grid point (start, start+5m, ...).

    Bar i: ``low=100``, ``high=100*(1+amp_i)`` (amp exact), ``close=close_i``. Because
    each bar sits alone on the 5min grid, the derived 5min bar equals it. ``start`` is
    an on-grid time (mm % 5 == 0) so ``ceil`` keeps each bar in its own bucket.
    """
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=5 * i), sym, 100.0 * (1.0 + a), 100.0, c)
        for i, (c, a) in enumerate(zip(closes, amps))
    ]


def _pairs(closes, amps):
    """Within-day (|Δamp|, r) pairs for one day's bar sequence (first bar dropped)."""
    dabs = np.abs(np.diff(np.asarray(amps, dtype=float)))
    c = np.asarray(closes, dtype=float)
    ret = c[1:] / c[:-1] - 1.0
    return dabs, ret


def _anomaly_vol_ref(dabs, ret, min_pool, min_sel, k=1.0):
    """Hand reference: gate on pool size, select |Δamp| > μ + kσ, std(ret|sel, ddof=1)."""
    d = np.asarray(dabs, dtype=float)
    r = np.asarray(ret, dtype=float)
    if d.size < min_pool:
        return float("nan")
    thr = d.mean() + k * d.std(ddof=1)
    mask = d > thr
    if int(mask.sum()) < min_sel:
        return float("nan")
    return float(r[mask].std(ddof=1))


# --------------------------------------------------------------------------- #
# Correctness vs a hand-computed reference
# --------------------------------------------------------------------------- #
def test_single_day_hand_value_two_anomalies():
    # 6 bars -> 5 within-day pairs. |Δamp| pool = [0.01,0.01,0.20,0.20,0.01];
    # mean 0.086, std(ddof=1) ~0.10407, threshold ~0.19007. Only the two 0.20 bars
    # (pairs t3,t4) exceed it -> selected returns [110/100-1, 100/110-1].
    closes = [100.0, 100.0, 100.0, 110.0, 100.0, 105.0]
    amps = [0.02, 0.03, 0.04, 0.24, 0.04, 0.05]
    out = compute_amp_marginal_anomaly_vol(
        _bars(_grid_session("2021-07-01", closes, amps)),
        lookback_days=20, min_pool=5, min_selected=2,
    )
    d1 = pd.Timestamp("2021-07-01")
    expected = float(np.array([110.0 / 100.0 - 1.0, 100.0 / 110.0 - 1.0]).std(ddof=1))
    assert out.loc[(d1, _SYM)] == pytest.approx(expected)
    # cross-check against the reference on the raw pool
    dabs, ret = _pairs(closes, amps)
    assert out.loc[(d1, _SYM)] == pytest.approx(_anomaly_vol_ref(dabs, ret, 5, 2))
    # bare default column name (window suffix lives on the Factor, like PR-D).
    assert out.name == "amp_marginal_anomaly_vol"


def test_single_day_hand_value_larger_pool():
    # 8 bars -> 7 pairs. |Δamp| pool has two clear anomalies (the 0.60 jumps);
    # their returns are +0.20 and -1/6, so the std is a meaningful spread.
    closes = [100.0, 100.0, 120.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    amps = [0.02, 0.03, 0.60, 0.05, 0.06, 0.07, 0.08, 0.09]
    out = compute_amp_marginal_anomaly_vol(
        _bars(_grid_session("2021-07-01", closes, amps)),
        lookback_days=20, min_pool=7, min_selected=2,
    )
    d1 = pd.Timestamp("2021-07-01")
    # anomalies are pair1 (100->120 amp jump bar) and pair2 (amp drop bar):
    # returns 120/100-1 and 100/120-1.
    expected = float(np.array([120.0 / 100.0 - 1.0, 100.0 / 120.0 - 1.0]).std(ddof=1))
    assert out.loc[(d1, _SYM)] == pytest.approx(expected)
    dabs, ret = _pairs(closes, amps)
    assert out.loc[(d1, _SYM)] == pytest.approx(_anomaly_vol_ref(dabs, ret, 7, 2))


def test_trailing_window_pools_two_days_no_cross_day_pair():
    # lookback_days=2, min_pool=10: day2 pools BOTH days' 5 pairs (n=10). The pool is
    # the CONCATENATION of per-day pairs (no pair crosses the overnight gap); day1
    # alone (5 pairs < 10) -> NaN.
    closes1 = [100.0, 100.0, 100.0, 100.0, 120.0, 100.0]
    amps1 = [0.02, 0.03, 0.04, 0.05, 0.65, 0.05]
    closes2 = [100.0, 100.0, 120.0, 100.0, 100.0, 100.0]
    amps2 = [0.02, 0.03, 0.60, 0.05, 0.06, 0.07]
    rows = _grid_session("2021-07-01", closes1, amps1) + _grid_session(
        "2021-07-02", closes2, amps2
    )
    out = compute_amp_marginal_anomaly_vol(
        _bars(rows), lookback_days=2, min_pool=10, min_selected=2
    )
    d1, d2 = pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-02")
    assert np.isnan(out.loc[(d1, _SYM)])  # only 5 pairs in day1's window
    dabs1, ret1 = _pairs(closes1, amps1)
    dabs2, ret2 = _pairs(closes2, amps2)
    pooled_dabs = np.concatenate([dabs1, dabs2])
    pooled_ret = np.concatenate([ret1, ret2])
    assert out.loc[(d2, _SYM)] == pytest.approx(
        _anomaly_vol_ref(pooled_dabs, pooled_ret, 10, 2)
    )


def test_within_day_lag_does_not_cross_days():
    # lookback_days=1: day2's pool is day2 ALONE. Perturbing day1's LAST bar must not
    # change day2's factor -- day2's first bar has no Δamp/r referencing day1 (no
    # cross-day lag), and day1 is not in day2's trailing-1 window either.
    closes2 = [100.0, 100.0, 120.0, 100.0, 100.0, 100.0]
    amps2 = [0.02, 0.03, 0.65, 0.05, 0.06, 0.07]
    day1 = _grid_session(
        "2021-07-01", [100.0, 100.0, 100.0, 120.0, 100.0, 100.0],
        [0.02, 0.03, 0.60, 0.05, 0.06, 0.07],
    )
    day2 = _grid_session("2021-07-02", closes2, amps2)
    a = compute_amp_marginal_anomaly_vol(
        _bars(day1 + day2), lookback_days=1, min_pool=4, min_selected=2
    )
    # wildly perturb day1's LAST bar (amplitude AND close)
    day1_perturbed = day1[:-1] + [
        (day1[-1][0], _SYM, 100.0 * (1.0 + 0.99), 100.0, 199.0)
    ]
    b = compute_amp_marginal_anomaly_vol(
        _bars(day1_perturbed + day2), lookback_days=1, min_pool=4, min_selected=2
    )
    d2 = pd.Timestamp("2021-07-02")
    assert np.isfinite(a.loc[(d2, _SYM)])
    assert a.loc[(d2, _SYM)] == pytest.approx(b.loc[(d2, _SYM)])


# --------------------------------------------------------------------------- #
# PIT truncation at 14:50 (leakage) + 5min available_time provenance
# --------------------------------------------------------------------------- #
def test_resample_boundary_hand_case_and_available_time_is_source_max():
    # 5 one-minute bars in ONE 5min bucket (14:46..14:50): the derived 5min bar takes
    # open=first, high=max, low=min, close=last, bar_start=min(source), bar_end=max,
    # and available_time = MAX(source available_time) = 14:51 (PIT-faithful).
    rows = [
        (pd.Timestamp("2021-07-01 14:46"), _SYM, 105.0, 99.0, 100.0),
        (pd.Timestamp("2021-07-01 14:47"), _SYM, 110.0, 98.0, 101.0),
        (pd.Timestamp("2021-07-01 14:48"), _SYM, 103.0, 97.0, 102.0),
        (pd.Timestamp("2021-07-01 14:49"), _SYM, 108.0, 101.0, 103.0),
        (pd.Timestamp("2021-07-01 14:50"), _SYM, 106.0, 100.0, 104.0),
    ]
    coarse = resample_intraday_bars(_bars(rows), AMP_ANOMALY_FREQ)
    assert len(coarse) == 1
    bar = coarse.iloc[0]
    assert bar["open"] == pytest.approx(100.0)  # first bar's open (== its close here)
    assert bar["high"] == pytest.approx(110.0)  # max of source highs
    assert bar["low"] == pytest.approx(97.0)    # min of source lows
    assert bar["close"] == pytest.approx(104.0)  # last bar's close
    assert bar["bar_end"] == pd.Timestamp("2021-07-01 14:50")
    assert bar["bar_start"] == pd.Timestamp("2021-07-01 14:45")  # min source bar_start
    # available_time is the MAX source available_time (= 14:50 + 1min), NOT bar_end+lag.
    assert bar["available_time"] == pd.Timestamp("2021-07-01 14:51")


def test_pit_truncation_excludes_post_1450_bars():
    # Morning grid pool (known factor) + a 14:50 bucket (available 14:51 > cutoff) that
    # carries an extreme amplitude; the extreme bucket must never enter the factor.
    closes = [100.0, 100.0, 100.0, 110.0, 100.0, 105.0]
    amps = [0.02, 0.03, 0.04, 0.24, 0.04, 0.05]
    morning = _grid_session("2021-07-01", closes, amps)  # 09:35..10:00, all visible
    late = [
        (pd.Timestamp("2021-07-01 14:46"), _SYM, 1000.0, 1.0, 500.0),
        (pd.Timestamp("2021-07-01 14:47"), _SYM, 1000.0, 1.0, 500.0),
        (pd.Timestamp("2021-07-01 14:50"), _SYM, 1000.0, 1.0, 500.0),
    ]
    out = compute_amp_marginal_anomaly_vol(
        _bars(morning + late), lookback_days=20, min_pool=5, min_selected=2
    )
    d1 = pd.Timestamp("2021-07-01")
    expected = float(np.array([110.0 / 100.0 - 1.0, 100.0 / 110.0 - 1.0]).std(ddof=1))
    assert out.loc[(d1, _SYM)] == pytest.approx(expected)


def test_perturbing_post_1450_bars_does_not_change_factor():
    closes = [100.0, 100.0, 100.0, 110.0, 100.0, 105.0]
    amps = [0.02, 0.03, 0.04, 0.24, 0.04, 0.05]
    morning = _grid_session("2021-07-01", closes, amps)
    a = compute_amp_marginal_anomaly_vol(
        _bars(morning + [(pd.Timestamp("2021-07-01 14:50"), _SYM, 900.0, 100.0, 800.0)]),
        lookback_days=20, min_pool=5, min_selected=2,
    )
    b = compute_amp_marginal_anomaly_vol(
        _bars(morning + [
            (pd.Timestamp("2021-07-01 14:50"), _SYM, 9000.0, 1.0, 5000.0),
            (pd.Timestamp("2021-07-01 15:00"), _SYM, 9500.0, 1.0, 9999.0),
        ]),
        lookback_days=20, min_pool=5, min_selected=2,
    )
    d1 = pd.Timestamp("2021-07-01")
    assert a.loc[(d1, _SYM)] == pytest.approx(b.loc[(d1, _SYM)])


# --------------------------------------------------------------------------- #
# Gates, isolation, guards
# --------------------------------------------------------------------------- #
def test_min_pool_gate_default_460_returns_nan_on_small_pool():
    # A handful of bars with the REAL default gate (460) -> NaN (honest missing).
    closes = [100.0 + i for i in range(12)]
    amps = [0.01 * (i + 1) for i in range(12)]
    out = compute_amp_marginal_anomaly_vol(_bars(_grid_session("2021-07-01", closes, amps)))
    assert out.dropna().empty


def test_min_selected_gate_returns_nan_when_too_few_anomalies():
    # Constant |Δamp| -> std 0 -> threshold == μ -> strict '>' selects NOTHING
    # -> n_sel 0 < min_selected -> NaN even though the pool is large enough.
    closes = [100.0 + i for i in range(6)]
    amps = [0.02 + 0.02 * i for i in range(6)]  # constant Δamp of 0.02
    out = compute_amp_marginal_anomaly_vol(
        _bars(_grid_session("2021-07-01", closes, amps)),
        lookback_days=20, min_pool=5, min_selected=2,
    )
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-01"), _SYM)])


def test_per_symbol_isolation():
    closes_a = [100.0, 100.0, 100.0, 110.0, 100.0, 105.0]
    amps_a = [0.02, 0.03, 0.04, 0.24, 0.04, 0.05]
    closes_b = [100.0, 100.0, 120.0, 100.0, 100.0, 100.0]
    amps_b = [0.02, 0.03, 0.60, 0.05, 0.06, 0.07]
    rows = _grid_session("2021-07-01", closes_a, amps_a, sym="AAA.SZ") + _grid_session(
        "2021-07-01", closes_b, amps_b, sym="BBB.SZ"
    )
    out = compute_amp_marginal_anomaly_vol(
        _bars(rows), lookback_days=20, min_pool=5, min_selected=2
    )
    d1 = pd.Timestamp("2021-07-01")
    da, ra = _pairs(closes_a, amps_a)
    db, rb = _pairs(closes_b, amps_b)
    assert out.loc[(d1, "AAA.SZ")] == pytest.approx(_anomaly_vol_ref(da, ra, 5, 2))
    assert out.loc[(d1, "BBB.SZ")] == pytest.approx(_anomaly_vol_ref(db, rb, 5, 2))
    # the two symbols get DIFFERENT values (no cross-symbol pooling)
    assert out.loc[(d1, "AAA.SZ")] != pytest.approx(out.loc[(d1, "BBB.SZ")])


def test_amplitude_guards_drop_bad_bars_before_lag():
    # A low<=0 bar and a high<low bar are dropped BEFORE the within-day lag, so the
    # surviving 6 good bars form the pool and their consecutive diffs skip the dropped
    # bars (matching the reference on the good sequence only).
    rows = _grid_session("2021-07-01",
        [100.0, 90.0, 100.0, 120.0, 95.0, 100.0, 105.0, 106.0],
        [0.02, 0.02, 0.04, 0.30, 0.02, 0.04, 0.05, 0.06],
    )
    rows[1] = (rows[1][0], _SYM, 101.0, 0.0, 90.0)     # low<=0 -> dropped
    rows[4] = (rows[4][0], _SYM, 99.0, 100.0, 95.0)    # high<low -> dropped
    out = compute_amp_marginal_anomaly_vol(
        _bars(rows), lookback_days=20, min_pool=4, min_selected=2
    )
    d1 = pd.Timestamp("2021-07-01")
    good_closes = [100.0, 100.0, 120.0, 100.0, 105.0, 106.0]
    good_amps = [0.02, 0.04, 0.30, 0.04, 0.05, 0.06]
    dabs, ret = _pairs(good_closes, good_amps)
    assert out.loc[(d1, _SYM)] == pytest.approx(_anomaly_vol_ref(dabs, ret, 4, 2))


def test_empty_bars_yield_empty_schema_series():
    out = compute_amp_marginal_anomaly_vol(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "amp_marginal_anomaly_vol"


def test_input_bars_not_mutated():
    bars = _bars(_grid_session("2021-07-01", [100.0, 101.0, 102.0, 103.0],
                               [0.01, 0.02, 0.03, 0.04]))
    before = bars.copy(deep=True)
    compute_amp_marginal_anomaly_vol(bars, lookback_days=20, min_pool=3, min_selected=2)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_grid_session("2021-07-01", [100.0, 101.0, 102.0], [0.01, 0.02, 0.03]))
    with pytest.raises(ValueError):
        compute_amp_marginal_anomaly_vol(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_amp_marginal_anomaly_vol(bars, min_pool=1)
    with pytest.raises(ValueError):
        compute_amp_marginal_anomaly_vol(bars, min_selected=1)
    with pytest.raises(ValueError):
        compute_amp_marginal_anomaly_vol(bars, sigma_k=-0.5)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = AmpMarginalAnomalyVolFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "amp_marginal_anomaly_vol_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == 1  # POSITIVE (report IC positive)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    # is_intraday=False => the whole minute block MUST be None (validated by FactorSpec).
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_defaults_match_module_constants():
    assert AMP_ANOMALY_LOOKBACK_DAYS == 20
    assert AMP_ANOMALY_FREQ == "5min"
    assert AMP_ANOMALY_SIGMA_K == 1.0
    assert AMP_ANOMALY_MIN_POOL == 460
    assert AMP_ANOMALY_MIN_SELECTED == 20
    f = AmpMarginalAnomalyVolFactor()
    assert f.lookback_days == AMP_ANOMALY_LOOKBACK_DAYS
    assert f.name == "amp_marginal_anomaly_vol_20"


def test_factor_subclass_window_tracks_name():
    f = AmpMarginalAnomalyVolFactor(lookback_days=10)
    assert f.name == "amp_marginal_anomaly_vol_10"
    assert f.spec.factor_id == "amp_marginal_anomaly_vol_10"


def test_factor_compute_selects_preaggregated_column():
    f = AmpMarginalAnomalyVolFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.42]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.42
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = AmpMarginalAnomalyVolFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        AmpMarginalAnomalyVolFactor(lookback_days=0)
