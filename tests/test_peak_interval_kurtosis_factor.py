"""PR-H: volume-peak INTERVAL-KURTOSIS factor.

Same volume-peak identification as PR-F (REUSED, not re-implemented), different
statistic: the gaps between consecutive same-day peaks, measured in TRADING MINUTES,
pooled over the trailing N valid days, reduced to their Fisher excess (bias-corrected)
kurtosis. Sign is pre-registered +1 (peakier / fatter-tailed interval distribution =
bursty informed trading -> higher forward return).

Hand cases build a constant BACKGROUND of prior days so the same-slot baseline μ/σ is
exact, then a test day whose eruptive / mild pattern places peaks at chosen positions,
so the pooled interval multiset -- and therefore the kurtosis -- is known exactly.
NOTE two peaks can never be 1 trading minute apart: adjacent eruptive minutes are
RIDGES under the reused taxonomy, so the smallest possible interval is 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_peak_interval import (
    PEAK_INTERVAL_LOOKBACK_DAYS,
    PEAK_INTERVAL_MIN_INTERVALS,
    compute_peak_interval_kurtosis,
    excess_kurtosis,
    peak_intervals_by_day,
)
from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_SIGMA_K,
    compute_volume_peak_count,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import PeakIntervalKurtosisFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, volume), ...] -> normalized 1min bars.

    OHLC are dummy constants (the factor reads only ``volume`` for the slot baseline);
    ``amount`` mirrors volume harmlessly. ``normalize_intraday_bars`` sets
    ``available_time = bar_end + 1min`` so the 14:50 PIT cutoff excludes any bar with
    ``bar_end >= 14:50``.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": [float(r[2]) for r in rows],
            "amount": [float(r[2]) for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, vols, sym=_SYM, start="09:31:00"):
    """One session of CONSECUTIVE 1-minute bars (start, start+1m, ...) with ``vols``."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [(base + pd.Timedelta(minutes=i), sym, v) for i, v in enumerate(vols)]


def _background(n_days, n_slots, sym=_SYM, start_day="2021-07-01", start="09:31:00"):
    """``n_days`` prior days of flat volume-100 sessions -> baseline μ=100, σ=0."""
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, sym=sym, start=start)
    return rows


def _peaks_at(n_slots, positions):
    """Volume list of ``n_slots`` mild (100) bars with 200 at each of ``positions``.

    Every position must be an interior slot >= 2 apart, so each is an isolated eruption
    with mild 1-minute neighbours -> a PEAK under the reused taxonomy.
    """
    vols = [100.0] * n_slots
    for p in positions:
        vols[p] = 200.0
    return vols


_BG_DAYS = 10
_TEST_DAY = pd.Timestamp("2021-07-11")

# Gates small enough that the single engineered test day is the only VALID day.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=PEAK_INTERVAL_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_intervals=4,
)


# --------------------------------------------------------------------------- #
# Hand-computed kurtosis values (>= 2 non-trivial cases) + pandas.kurt() alignment
# --------------------------------------------------------------------------- #
def test_hand_value_case_a_intervals_2_2_2_8():
    # 17 slots, peaks at positions 1, 3, 5, 7, 15 -> intervals [2, 2, 2, 8].
    # By hand (Fisher excess, bias-corrected, n=4): mean=3.5, M2=27, M4=425.25,
    #   kurt = n(n+1)(n-1)M4 / ((n-2)(n-3)M2^2) - 3(n-1)^2/((n-2)(n-3))
    #        = 60*425.25/(2*729) - 13.5 = 17.5 - 13.5 = 4.0
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(4.0)
    # independent verification that our estimator IS the pandas .kurt() convention
    assert pd.Series([2.0, 2.0, 2.0, 8.0]).kurt() == pytest.approx(4.0)


def test_hand_value_case_b_six_intervals():
    # 25 slots, peaks at 1, 3, 6, 10, 15, 21, 23 -> intervals [2, 3, 4, 5, 6, 2].
    # By hand (n=6): mean=11/3, M2=120/9, M4=3924/81,
    #   kurt = 210*(3924/81) / (12*(14400/81)) - 3*25/12 = 4.76875 - 6.25 = -1.48125
    rows = _background(_BG_DAYS, 25)
    rows += _session("2021-07-11", _peaks_at(25, [1, 3, 6, 10, 15, 21, 23]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(-1.48125)
    assert pd.Series([2.0, 3.0, 4.0, 5.0, 6.0, 2.0]).kurt() == pytest.approx(-1.48125)


def test_excess_kurtosis_matches_pandas_kurt_on_random_samples():
    # The estimator convention is PINNED to pandas .kurt() (== scipy kurtosis with
    # fisher=True, bias=False). Verify on random samples, not just the hand cases.
    rng = np.random.default_rng(11)
    for n in (4, 5, 20, 137):
        x = rng.integers(1, 60, size=n).astype(float)
        if np.ptp(x) == 0:
            continue
        assert excess_kurtosis(x) == pytest.approx(pd.Series(x).kurt())
    # a normal-ish large sample sits near 0 (Fisher convention, not Pearson's 3)
    big = rng.standard_normal(200_000)
    assert abs(excess_kurtosis(big)) < 0.1


def test_excess_kurtosis_needs_four_points_and_variance():
    assert np.isnan(excess_kurtosis(np.array([2.0, 4.0, 6.0])))  # n < 4
    assert np.isnan(excess_kurtosis(np.array([3.0, 3.0, 3.0, 3.0])))  # zero variance


# --------------------------------------------------------------------------- #
# Intervals are TRADING MINUTES (the lunch break is NOT 90 minutes)
# --------------------------------------------------------------------------- #
def _lunch_rows(peak_slots, sym=_SYM):
    """Days spanning the lunch break: 11:26..11:30 then 13:01..13:05."""
    slots = [f"11:{m}:00" for m in range(26, 31)] + [
        f"13:0{m}:00" for m in range(1, 6)
    ]

    def day(d, vols):
        base = pd.Timestamp(d)
        return [(base + pd.Timedelta(s), sym, v) for s, v in zip(slots, vols)]

    rows = []
    for i in range(_BG_DAYS):
        d = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += day(d, [100.0] * len(slots))
    vols = [100.0] * len(slots)
    for p in peak_slots:
        vols[p] = 200.0
    rows += day("2021-07-11", vols)
    return rows


def test_lunch_break_interval_is_trading_minutes_not_wall_clock():
    # Peaks at 11:29 (index 3) and 13:02 (index 6). The visible trading-minute sequence
    # is 11:26,11:27,11:28,11:29,11:30,13:01,13:02,... so 11:30 -> 13:01 is ONE slot
    # apart and the peak-to-peak interval is 3 TRADING minutes -- NOT the 93 minutes of
    # wall clock. (PINNED interpretation; the report is silent.)
    rows = _lunch_rows([3, 6])
    visible = prepare_visible_minute_bars(_bars(rows))
    g = visible[visible["symbol"] == _SYM].reset_index(drop=True)
    work = peak_mask_for_symbol(g)
    intervals = peak_intervals_by_day(work)
    np.testing.assert_array_equal(intervals.loc[_TEST_DAY], np.array([3]))
    # the wall-clock gap really is 93 minutes -- the point of the pinned choice
    peaks = work.loc[work["peak"], "bar_end"].tolist()
    assert (peaks[1] - peaks[0]) == pd.Timedelta(minutes=93)


def test_missing_minute_shrinks_interval_by_one_slot():
    # A minute absent from the cache is simply not a tradable slot in our sequence, so
    # an interval spanning it is one SHORTER. Disclosed consequence of measuring in
    # observed trading slots (rare in practice; the same stance PR-F takes on gaps).
    full = _background(_BG_DAYS, 12) + _session("2021-07-11", _peaks_at(12, [1, 5]))
    a = compute_peak_interval_kurtosis(_bars(full), **_KW)
    dropped = [r for r in full if r[0] != pd.Timestamp("2021-07-11 09:34")]
    visible = prepare_visible_minute_bars(_bars(dropped))
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    # peaks at 09:32 and 09:36 with 09:34 missing -> 3 slots apart, not 4
    np.testing.assert_array_equal(peak_intervals_by_day(work).loc[_TEST_DAY], [3])
    assert np.isnan(a.loc[(_TEST_DAY, _SYM)])  # single interval -> no kurtosis


# --------------------------------------------------------------------------- #
# A day with < 2 peaks contributes ZERO intervals but is still a VALID day
# --------------------------------------------------------------------------- #
def test_single_peak_day_contributes_no_intervals_but_stays_valid():
    # Day 1: exactly one peak -> 0 intervals (NOT NaN-poisoning, NOT skipped).
    # Day 2: five peaks -> intervals [2, 2, 2, 8] -> pooled kurtosis 4.0.
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", _peaks_at(17, [3]))
    rows += _session("2021-07-12", _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    d1, d2 = pd.Timestamp("2021-07-11"), pd.Timestamp("2021-07-12")
    # day 1 IS emitted (a valid day) but has too few pooled intervals -> honest NaN
    assert (d1, _SYM) in out.index
    assert np.isnan(out.loc[(d1, _SYM)])
    # day 2 pools day 1 (0 intervals) + day 2 (4 intervals) -> exactly the 4 intervals
    assert out.loc[(d2, _SYM)] == pytest.approx(4.0)


def test_zero_peak_day_is_valid_and_pools_nothing():
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", [100.0] * 17)  # no eruption at all
    rows += _session("2021-07-12", _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-11"), _SYM)])
    assert out.loc[(pd.Timestamp("2021-07-12"), _SYM)] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# The two NaN gates: too few pooled intervals / zero variance
# --------------------------------------------------------------------------- #
def test_too_few_pooled_intervals_is_nan():
    # 4 peaks -> 3 intervals, below min_intervals=4 -> NaN (kurtosis is wild on tiny
    # samples; honest missing rather than a number).
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", _peaks_at(17, [1, 3, 5, 7]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert np.isnan(out.loc[(_TEST_DAY, _SYM)])


def test_zero_variance_pool_is_nan():
    # 5 evenly spaced peaks -> intervals [2, 2, 2, 2]: enough of them, but kurtosis is
    # undefined with zero variance -> NaN, never a divide-by-zero inf/garbage.
    rows = _background(_BG_DAYS, 12)
    rows += _session("2021-07-11", _peaks_at(12, [1, 3, 5, 7, 9]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert np.isnan(out.loc[(_TEST_DAY, _SYM)])


def test_default_min_intervals_gate_is_twenty():
    assert PEAK_INTERVAL_MIN_INTERVALS == 20
    assert PEAK_INTERVAL_LOOKBACK_DAYS == 20


# --------------------------------------------------------------------------- #
# Reuse non-drift: the peaks PR-H intervals ARE PR-F's peaks
# --------------------------------------------------------------------------- #
def test_peaks_used_are_identical_to_pr_f_peak_count():
    # Same bars through both factors: the number of intervals on a day must be exactly
    # (PR-F's peak count for that day) - 1. This is the anti-drift lock on the §0 reuse:
    # if PR-H ever re-implemented the taxonomy, this would break.
    rows = _background(_BG_DAYS, 25)
    rows += _session("2021-07-11", _peaks_at(25, [1, 3, 6, 10, 15, 21, 23]))
    bars = _bars(rows)

    pr_f = compute_volume_peak_count(
        bars,
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        lookback_days=1,
        min_valid_days=1,
        min_classifiable=1,
    )
    n_peaks = pr_f.loc[(_TEST_DAY, _SYM)]
    assert n_peaks == pytest.approx(7.0)

    visible = prepare_visible_minute_bars(bars)
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    intervals = peak_intervals_by_day(work).loc[_TEST_DAY]
    assert len(intervals) == int(n_peaks) - 1
    np.testing.assert_array_equal(intervals, np.array([2, 3, 4, 5, 6, 2]))


def test_ridge_minutes_never_produce_a_one_minute_interval():
    # Adjacent eruptive minutes are RIDGES (not peaks) under the reused taxonomy, so no
    # interval can ever be 1. Eruptions at 1,2 (ridge pair) and clean peaks at 5,9,13,17.
    vols = [100.0] * 20
    for p in (1, 2, 5, 9, 13, 17):
        vols[p] = 200.0
    rows = _background(_BG_DAYS, 20) + _session("2021-07-11", vols)
    visible = prepare_visible_minute_bars(_bars(rows))
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    intervals = peak_intervals_by_day(work).loc[_TEST_DAY]
    np.testing.assert_array_equal(intervals, np.array([4, 4, 4]))
    assert intervals.min() >= 2


# --------------------------------------------------------------------------- #
# PIT: no lookahead, no post-cutoff influence
# --------------------------------------------------------------------------- #
def test_perturbing_post_1450_bars_does_not_change_factor():
    rows = _background(_BG_DAYS, 17) + _session(
        "2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15])
    )
    a = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    late = rows + [
        (pd.Timestamp("2021-07-11 14:50"), _SYM, 9_999.0),
        (pd.Timestamp("2021-07-11 14:55"), _SYM, 9_999.0),
        (pd.Timestamp("2021-07-11 14:56"), _SYM, 1.0),
    ]
    b = compute_peak_interval_kurtosis(_bars(late), **_KW)
    key = (_TEST_DAY, _SYM)
    assert a.loc[key] == pytest.approx(4.0)
    assert a.loc[key] == pytest.approx(b.loc[key])


def test_future_day_does_not_change_earlier_factor():
    base = _background(_BG_DAYS, 17) + _session(
        "2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15])
    )
    a = compute_peak_interval_kurtosis(_bars(base), **_KW)
    future = base + _session(
        "2021-07-12", [9_999.0, 1.0, 9_999.0, 1.0] + [500.0] * 13
    )
    b = compute_peak_interval_kurtosis(_bars(future), **_KW)
    key = (_TEST_DAY, _SYM)
    assert np.isfinite(a.loc[key])
    assert a.loc[key] == pytest.approx(b.loc[key])


# --------------------------------------------------------------------------- #
# Per-symbol isolation
# --------------------------------------------------------------------------- #
def test_per_symbol_isolation():
    rows = _background(_BG_DAYS, 25, sym="AAA.SZ")
    rows += _session("2021-07-11", _peaks_at(25, [1, 3, 5, 7, 15]), sym="AAA.SZ")
    rows += _background(_BG_DAYS, 25, sym="BBB.SZ")
    rows += _session(
        "2021-07-11", _peaks_at(25, [1, 3, 6, 10, 15, 21, 23]), sym="BBB.SZ"
    )
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(4.0)
    assert out.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(-1.48125)


# --------------------------------------------------------------------------- #
# Window mechanics
# --------------------------------------------------------------------------- #
def test_pool_is_the_trailing_lookback_valid_days():
    # 3 valid days: day1 [2,2,2,8], day2 [2,2,2,8], day3 [4,4,4] (evenly spaced).
    # With lookback_days=2 the day-3 pool is days 2+3 = [2,2,2,8,4,4,4]; a lookback of 3
    # would also include day 1 and give a different value. Assert day 3 == the 2-day pool.
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15]))
    rows += _session("2021-07-12", _peaks_at(17, [1, 3, 5, 7, 15]))
    rows += _session("2021-07-13", _peaks_at(17, [1, 5, 9, 13]))
    out = compute_peak_interval_kurtosis(
        _bars(rows), **{**_KW, "lookback_days": 2}
    )
    expected = pd.Series([2.0, 2.0, 2.0, 8.0, 4.0, 4.0, 4.0]).kurt()
    assert out.loc[(pd.Timestamp("2021-07-13"), _SYM)] == pytest.approx(expected)


def test_min_valid_days_floor_returns_nan_until_enough_valid_days():
    rows = _background(_BG_DAYS, 17)
    for k in range(3):
        d = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        rows += _session(d, _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(
        _bars(rows), **{**_KW, "min_valid_days": 3}
    )
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-11"), _SYM)])
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-12"), _SYM)])
    assert np.isfinite(out.loc[(pd.Timestamp("2021-07-13"), _SYM)])


def test_min_classifiable_gate_invalidates_thin_days():
    rows = _background(_BG_DAYS, 17)
    rows += _session("2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(
        _bars(rows), **{**_KW, "min_classifiable": 100}
    )
    assert out.dropna().empty


def test_baseline_insufficient_yields_no_value():
    # Only 9 prior days -> fewer than baseline_min_obs (=10) same-slot obs -> nothing
    # classifiable -> no valid day -> no value at all.
    rows = _background(9, 17)
    rows += _session("2021-07-10", _peaks_at(17, [1, 3, 5, 7, 15]))
    out = compute_peak_interval_kurtosis(_bars(rows), **_KW)
    assert out.dropna().empty


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_peak_interval_kurtosis(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "peak_interval_kurtosis"


def test_input_bars_not_mutated():
    bars = _bars(
        _background(_BG_DAYS, 17) + _session("2021-07-11", _peaks_at(17, [1, 3, 5, 7, 15]))
    )
    before = bars.copy(deep=True)
    compute_peak_interval_kurtosis(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0, 100.0, 100.0]))
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, baseline_days=1)
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, baseline_min_obs=1)
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, sigma_k=-0.5)
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, min_valid_days=0)
    with pytest.raises(ValueError):
        compute_peak_interval_kurtosis(bars, min_classifiable=0)
    with pytest.raises(ValueError):
        # kurtosis is undefined below 4 observations
        compute_peak_interval_kurtosis(bars, min_intervals=3)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = PeakIntervalKurtosisFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "peak_interval_kurtosis_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == 1  # POSITIVE (report RankIC +7.19%)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert spec.input_fields == ("volume",)
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_spec_discloses_pinned_interpretations():
    # The two choices the report is silent on MUST be on the spec a reader sees.
    desc = PeakIntervalKurtosisFactor().spec.description
    assert "TRADING MINUTE" in desc.upper()
    assert "FISHER" in desc.upper() and "bias-corrected" in desc


def test_factor_subclass_window_tracks_name():
    f = PeakIntervalKurtosisFactor(lookback_days=10)
    assert f.name == "peak_interval_kurtosis_10"
    assert f.spec.factor_id == "peak_interval_kurtosis_10"


def test_factor_compute_selects_preaggregated_column():
    f = PeakIntervalKurtosisFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [1.25]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 1.25
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = PeakIntervalKurtosisFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        PeakIntervalKurtosisFactor(lookback_days=0)
