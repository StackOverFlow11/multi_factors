"""PR-D: minute ideal-amplitude factor (aggregation + spec + PIT truncation).

The factor pools the 1min bars of a symbol's trailing ``N`` trading days (PIT-
truncated at 14:50 per bar), ranks the pooled minutes by their raw close price,
and returns ``V_high - V_low`` where ``V_high``/``V_low`` are the mean per-minute
amplitude (``high/low - 1``) of the top / bottom ``floor(lambda*n)`` minutes by
close. Sign is pre-registered -1 (high ideal amplitude -> lower forward return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_amplitude import (
    IDEAL_AMP_LAMBDA,
    IDEAL_AMP_LOOKBACK_DAYS,
    IDEAL_AMP_MIN_MINUTES,
    compute_minute_ideal_amplitude,
)
from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from factors.compute.intraday_derived import MinuteIdealAmplitudeFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, high, low, close), ...] -> normalized 1min bars.

    ``open``/``volume``/``amount`` are filled harmlessly; the factor reads
    ``high``/``low`` (per-minute amplitude ``high/low - 1``) and ``close`` (the
    price-rank cut). ``close`` is set INDEPENDENTLY of ``high``/``low`` because the
    factor decouples them: rank by close, measure amplitude by high/low.
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


def _session(day, closes, amps, sym=_SYM, start="09:31:00"):
    """One session: minute i at ``low=100``, ``high=100*(1+amp_i)`` (amp exact),
    ``close=close_i`` (controls the price rank). Starts at ``start`` so every bar
    stays inside the 14:50 PIT window when ``start`` is early enough."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, 100.0 * (1.0 + a), 100.0, c)
        for i, (c, a) in enumerate(zip(closes, amps))
    ]


def _ideal_amp_ref(closes, amps, bar_ends, lam, min_minutes):
    """Hand reference: rank pooled minutes by (close, bar_end); V_high - V_low."""
    n = len(closes)
    if n < min_minutes:
        return float("nan")
    k = int(np.floor(lam * n))
    if k < 1:
        return float("nan")
    be = np.asarray([pd.Timestamp(b).value for b in bar_ends], dtype="int64")
    order = np.lexsort((be, np.asarray(closes, dtype=float)))  # close primary
    a = np.asarray(amps, dtype=float)[order]
    return float(a[-k:].mean() - a[:k].mean())


# --------------------------------------------------------------------------- #
# Correctness vs a hand-computed reference
# --------------------------------------------------------------------------- #
def test_single_day_hand_value_n4_k1():
    # n=4, k=floor(0.25*4)=1: V_high = amp of the single highest-close minute,
    # V_low = amp of the single lowest-close minute.
    closes = [100.0, 101.0, 102.0, 103.0]
    amps = [0.010, 0.040, 0.020, 0.030]
    bars = _bars(_session("2021-07-01", closes, amps))
    out = compute_minute_ideal_amplitude(
        bars, lookback_days=20, lam=0.25, min_minutes=4
    )
    d1 = pd.Timestamp("2021-07-01")
    # highest close 103 -> amp 0.030; lowest close 100 -> amp 0.010; diff 0.020.
    assert out.loc[(d1, _SYM)] == pytest.approx(0.030 - 0.010)
    # bare default column name (the window suffix lives on the Factor, like jump).
    assert out.name == "minute_ideal_amp"


def test_single_day_hand_value_n8_k2():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    amps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    bars = _bars(_session("2021-07-01", closes, amps))
    out = compute_minute_ideal_amplitude(
        bars, lookback_days=20, lam=0.25, min_minutes=8
    )
    d1 = pd.Timestamp("2021-07-01")
    # k=2: V_high = mean(amp of closes 106,107) = mean(0.07,0.08) = 0.075;
    #      V_low  = mean(amp of closes 100,101) = mean(0.01,0.02) = 0.015.
    assert out.loc[(d1, _SYM)] == pytest.approx(0.075 - 0.015)


def test_trailing_window_pools_two_days():
    # lookback_days=2, min_minutes=8: day2 pools BOTH days' 4 bars (n=8, k=2);
    # day1 alone has only 4 bars (< 8) -> NaN (warm-up handled by the min gate).
    closes1 = [100.0, 101.0, 102.0, 103.0]
    amps1 = [0.01, 0.05, 0.02, 0.06]
    closes2 = [104.0, 105.0, 106.0, 107.0]
    amps2 = [0.03, 0.07, 0.04, 0.08]
    rows = _session("2021-07-01", closes1, amps1) + _session(
        "2021-07-02", closes2, amps2
    )
    bars = _bars(rows)
    out = compute_minute_ideal_amplitude(
        bars, lookback_days=2, lam=0.25, min_minutes=8
    )
    d1, d2 = pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-02")
    assert np.isnan(out.loc[(d1, _SYM)])  # only 4 bars in day1's window
    # day2 pool: closes 100..107, amps as pooled. 2 lowest closes 100(0.01),101(0.05)
    # -> V_low 0.03; 2 highest 106(0.04),107(0.08) -> V_high 0.06; diff 0.03.
    pooled_closes = closes1 + closes2
    pooled_amps = amps1 + amps2
    assert out.loc[(d2, _SYM)] == pytest.approx(
        (0.04 + 0.08) / 2 - (0.01 + 0.05) / 2
    )
    # cross-check against the reference helper on the raw pool ordering.
    be = [
        pd.Timestamp("2021-07-01 09:31") + pd.Timedelta(minutes=i) for i in range(4)
    ] + [pd.Timestamp("2021-07-02 09:31") + pd.Timedelta(minutes=i) for i in range(4)]
    assert out.loc[(d2, _SYM)] == pytest.approx(
        _ideal_amp_ref(pooled_closes, pooled_amps, be, 0.25, 8)
    )


def test_trailing_window_drops_days_older_than_lookback():
    # 3 days, lookback_days=2, min_minutes=4: day3's 2-day window is {day2, day3};
    # day1's bars must NOT enter day3's pool. Give day1 an extreme amp so leakage
    # would be visible, then confirm day3 == pool of day2+day3 only.
    closes = [100.0, 101.0, 102.0, 103.0]
    amps_hi = [0.90, 0.90, 0.90, 0.90]  # day1 extreme (should be dropped at day3)
    amps2 = [0.01, 0.05, 0.02, 0.06]
    amps3 = [0.03, 0.07, 0.04, 0.08]
    rows = (
        _session("2021-07-01", closes, amps_hi)
        + _session("2021-07-02", [104.0, 105.0, 106.0, 107.0], amps2)
        + _session("2021-07-05", [108.0, 109.0, 110.0, 111.0], amps3)
    )
    out = compute_minute_ideal_amplitude(
        _bars(rows), lookback_days=2, lam=0.25, min_minutes=8
    )
    d3 = pd.Timestamp("2021-07-05")
    pooled_closes = [104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]
    pooled_amps = amps2 + amps3
    be = [
        pd.Timestamp("2021-07-02 09:31") + pd.Timedelta(minutes=i) for i in range(4)
    ] + [pd.Timestamp("2021-07-05 09:31") + pd.Timedelta(minutes=i) for i in range(4)]
    assert out.loc[(d3, _SYM)] == pytest.approx(
        _ideal_amp_ref(pooled_closes, pooled_amps, be, 0.25, 8)
    )


# --------------------------------------------------------------------------- #
# PIT truncation at 14:50 (leakage)
# --------------------------------------------------------------------------- #
def test_pit_truncation_excludes_post_1450_bars():
    # Bars spanning the 14:50 cutoff: only bar_end with available_time (=bar_end+1min)
    # <= 14:50 survive. Build a small session where the post-14:50 minutes carry an
    # extreme amplitude that would dominate the cut if it leaked in.
    day = "2021-07-01"
    pre = [
        (pd.Timestamp(f"{day} 14:46"), _SYM, 101.0, 100.0, 100.0),  # amp 0.01, close 100
        (pd.Timestamp(f"{day} 14:47"), _SYM, 104.0, 100.0, 101.0),  # amp 0.04, close 101
        (pd.Timestamp(f"{day} 14:48"), _SYM, 102.0, 100.0, 102.0),  # amp 0.02, close 102
        (pd.Timestamp(f"{day} 14:49"), _SYM, 103.0, 100.0, 103.0),  # amp 0.03, close 103 (last incl.)
    ]
    post = [
        (pd.Timestamp(f"{day} 14:50"), _SYM, 190.0, 100.0, 104.0),  # available 14:51 -> excluded
        (pd.Timestamp(f"{day} 14:51"), _SYM, 195.0, 100.0, 105.0),  # excluded
        (pd.Timestamp(f"{day} 15:00"), _SYM, 199.0, 100.0, 106.0),  # excluded
    ]
    out = compute_minute_ideal_amplitude(
        _bars(pre + post), lookback_days=20, lam=0.25, min_minutes=4
    )
    d1 = pd.Timestamp("2021-07-01")
    # Only the 4 pre-14:50 bars enter: k=1 -> V_high amp(close 103)=0.03,
    # V_low amp(close 100)=0.01 -> 0.02. The extreme post bars never appear.
    assert out.loc[(d1, _SYM)] == pytest.approx(0.02)


def test_perturbing_post_1450_bars_does_not_change_factor():
    day = "2021-07-01"
    pre = [
        (pd.Timestamp(f"{day} 14:46"), _SYM, 101.0, 100.0, 100.0),
        (pd.Timestamp(f"{day} 14:47"), _SYM, 104.0, 100.0, 101.0),
        (pd.Timestamp(f"{day} 14:48"), _SYM, 102.0, 100.0, 102.0),
        (pd.Timestamp(f"{day} 14:49"), _SYM, 103.0, 100.0, 103.0),
    ]
    a = compute_minute_ideal_amplitude(
        _bars(pre + [(pd.Timestamp(f"{day} 14:50"), _SYM, 190.0, 100.0, 104.0)]),
        lookback_days=20, lam=0.25, min_minutes=4,
    )
    # wildly perturb every post-14:50 bar (amplitude AND close)
    b = compute_minute_ideal_amplitude(
        _bars(pre + [
            (pd.Timestamp(f"{day} 14:50"), _SYM, 900.0, 100.0, 500.0),
            (pd.Timestamp(f"{day} 15:00"), _SYM, 950.0, 100.0, 999.0),
        ]),
        lookback_days=20, lam=0.25, min_minutes=4,
    )
    d1 = pd.Timestamp("2021-07-01")
    assert a.loc[(d1, _SYM)] == pytest.approx(b.loc[(d1, _SYM)])


# --------------------------------------------------------------------------- #
# Gates, ties, isolation, guards
# --------------------------------------------------------------------------- #
def test_min_minutes_gate_default_1150_returns_nan_on_small_pool():
    # A handful of bars with the REAL default gate (1150) -> NaN (honest missing).
    closes = [100.0 + i for i in range(10)]
    amps = [0.01 * (i + 1) for i in range(10)]
    bars = _bars(_session("2021-07-01", closes, amps))
    out = compute_minute_ideal_amplitude(bars)  # defaults: N=10, lam=0.25, min=1150
    assert out.dropna().empty


def test_tie_break_is_deterministic_by_bar_end():
    # Two minutes share close=100; the earlier bar_end must be the "lowest close"
    # pick at k=1. Swapping their amps would change the result, so asserting the
    # earlier-bar amp locks the (close, bar_end) tie order.
    rows = [
        (pd.Timestamp("2021-07-01 09:31"), _SYM, 101.0, 100.0, 100.0),  # amp 0.01, close 100 (t0)
        (pd.Timestamp("2021-07-01 09:32"), _SYM, 110.0, 100.0, 100.0),  # amp 0.10, close 100 (t1)
        (pd.Timestamp("2021-07-01 09:33"), _SYM, 102.0, 100.0, 101.0),  # amp 0.02, close 101
        (pd.Timestamp("2021-07-01 09:34"), _SYM, 103.0, 100.0, 102.0),  # amp 0.03, close 102 (high)
    ]
    out = compute_minute_ideal_amplitude(
        _bars(rows), lookback_days=20, lam=0.25, min_minutes=4
    )
    d1 = pd.Timestamp("2021-07-01")
    # k=1: lowest close is the 09:31 bar (earlier bar_end wins the tie) -> V_low 0.01;
    # highest close 102 -> V_high 0.03; factor 0.02. (If the 09:32 bar won the tie,
    # V_low would be 0.10 and factor -0.07.)
    assert out.loc[(d1, _SYM)] == pytest.approx(0.02)


def test_per_symbol_isolation():
    closes = [100.0, 101.0, 102.0, 103.0]
    amps_a = [0.01, 0.02, 0.03, 0.04]
    amps_b = [0.08, 0.07, 0.06, 0.05]
    rows = _session("2021-07-01", closes, amps_a, sym="AAA.SZ") + _session(
        "2021-07-01", closes, amps_b, sym="BBB.SZ"
    )
    out = compute_minute_ideal_amplitude(
        _bars(rows), lookback_days=20, lam=0.25, min_minutes=4
    )
    d1 = pd.Timestamp("2021-07-01")
    # k=1 each: A -> amp(close103)=0.04 - amp(close100)=0.01 = 0.03;
    #           B -> amp(close103)=0.05 - amp(close100)=0.08 = -0.03.
    assert out.loc[(d1, "AAA.SZ")] == pytest.approx(0.04 - 0.01)
    assert out.loc[(d1, "BBB.SZ")] == pytest.approx(0.05 - 0.08)


def test_amplitude_guards_drop_bad_bars():
    # low<=0 and high<low bars are dropped before ranking; remaining 4 good bars
    # (closes 100..103) give the plain k=1 value.
    rows = [
        (pd.Timestamp("2021-07-01 09:31"), _SYM, 101.0, 0.0, 90.0),    # low<=0 dropped
        (pd.Timestamp("2021-07-01 09:32"), _SYM, 99.0, 100.0, 95.0),   # high<low dropped
        (pd.Timestamp("2021-07-01 09:33"), _SYM, 101.0, 100.0, 100.0),  # amp 0.01, close 100
        (pd.Timestamp("2021-07-01 09:34"), _SYM, 104.0, 100.0, 101.0),  # amp 0.04, close 101
        (pd.Timestamp("2021-07-01 09:35"), _SYM, 102.0, 100.0, 102.0),  # amp 0.02, close 102
        (pd.Timestamp("2021-07-01 09:36"), _SYM, 103.0, 100.0, 103.0),  # amp 0.03, close 103
    ]
    out = compute_minute_ideal_amplitude(
        _bars(rows), lookback_days=20, lam=0.25, min_minutes=4
    )
    d1 = pd.Timestamp("2021-07-01")
    assert out.loc[(d1, _SYM)] == pytest.approx(0.03 - 0.01)


def test_empty_bars_yield_empty_schema_series():
    out = compute_minute_ideal_amplitude(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "minute_ideal_amp"


def test_input_bars_not_mutated():
    bars = _bars(_session("2021-07-01", [100.0, 101.0, 102.0, 103.0],
                          [0.01, 0.02, 0.03, 0.04]))
    before = bars.copy(deep=True)
    compute_minute_ideal_amplitude(bars, lookback_days=20, lam=0.25, min_minutes=4)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0, 101.0, 102.0], [0.01, 0.02, 0.03]))
    with pytest.raises(ValueError):
        compute_minute_ideal_amplitude(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_minute_ideal_amplitude(bars, lam=0.0)
    with pytest.raises(ValueError):
        compute_minute_ideal_amplitude(bars, lam=0.6)
    with pytest.raises(ValueError):
        compute_minute_ideal_amplitude(bars, min_minutes=1)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = MinuteIdealAmplitudeFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "minute_ideal_amp_10"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == -1
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    # is_intraday=False => the whole minute block MUST be None (validated by FactorSpec).
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_defaults_match_module_constants():
    assert IDEAL_AMP_LOOKBACK_DAYS == 10
    assert IDEAL_AMP_LAMBDA == 0.25
    assert IDEAL_AMP_MIN_MINUTES == 1150
    f = MinuteIdealAmplitudeFactor()
    assert f.lookback_days == IDEAL_AMP_LOOKBACK_DAYS
    assert f.name == "minute_ideal_amp_10"


def test_factor_subclass_window_tracks_name():
    f = MinuteIdealAmplitudeFactor(lookback_days=5)
    assert f.name == "minute_ideal_amp_5"
    assert f.spec.factor_id == "minute_ideal_amp_5"


def test_factor_compute_selects_preaggregated_column():
    f = MinuteIdealAmplitudeFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.25]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.25
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = MinuteIdealAmplitudeFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        MinuteIdealAmplitudeFactor(lookback_days=0)
