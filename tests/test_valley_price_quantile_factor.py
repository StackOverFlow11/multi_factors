"""PR-L: VALLEY WEIGHTED-PRICE-QUANTILE factor.

Same volume classification as PR-F/PR-H/PR-I/PR-J/PR-K (REUSED, not re-implemented),
a new STATISTIC family: a price POSITION (where in the day's range the valley VWAP
sits) rather than a count / timing moment / price RATIO / return. Sign is pre-registered
+1.

TWO cross-sectional/lookahead subtleties get dedicated tests here, because they are what
makes this factor harder than its five siblings:

  1. PREV_CLOSE extends the price range. The range is
     ``[min(intraday low, prev_close), max(intraday high, prev_close)]`` where prev_close
     is the last VISIBLE (<=14:50) raw close of the PREVIOUS trading day -- ONE visibility
     definition for the whole factor. Hand cases cover prev_close inside the intraday
     range, above it, and below it.
  2. REVERSAL NEUTRALIZATION at T-1. The report residualizes against a 20-day reversal
     factor. The naive ``-(close_d/close_{d-20} - 1)`` reads day d's CLOSE, which is 15:00
     information and a lookahead at our 14:50 decision. We use
     ``rev20 = -(close_{d-1}/close_{d-21} - 1)`` on FRONT-ADJUSTED daily closes.
     ``test_reversal_uses_t_minus_1_close_not_day_d`` perturbs ONE SYMBOL's day-d close
     and asserts the factor is UNCHANGED -- the single most important test in this PR.
     The single-symbol shape is load-bearing: a UNIFORM bump of every cross-section
     member's ``close_d`` sends the naive regressor to an AFFINE reparametrization of
     itself, and intercept-OLS residuals are invariant under that, so a uniform bump
     passes under BOTH bases and proves nothing. Two companions keep it honest:
     ``test_perturbing_the_t_minus_1_close_does_move_the_factor`` shows the closes are
     genuinely an input, and
     ``test_uniform_close_d_bump_cannot_distinguish_the_two_reversal_bases`` records why
     the uniform shape is inadmissible. Verified by monkeypatching ``reversal_20`` to the
     naive ``shift(0)/shift(days)`` form: the critical test FAILS there and passes on the
     real implementation.

Hand cases build a constant BACKGROUND of 10 prior days so the same-slot baseline is
exact (mu=100, sigma=0 -> the eruptive threshold is exactly 100) and, because
``VOLUME_PRV_BASELINE_MIN_OBS`` is 10, so that the engineered test day is the ONLY valid
day -- the trailing mean therefore equals that day's quantile in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_valley_quantile import (
    VALLEY_QUANTILE_LOOKBACK_DAYS,
    VALLEY_QUANTILE_MIN_CROSS_SECTION,
    VALLEY_QUANTILE_MIN_VALLEY_BARS,
    VALLEY_QUANTILE_REVERSAL_DAYS,
    compute_valley_price_quantile,
    compute_valley_price_quantile_stats,
    residualize_on_reversal,
    reversal_20,
    valley_price_quantile_by_day,
)
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import ValleyPriceQuantileFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"
_TEST_DAY = pd.Timestamp("2021-07-11")
_BG_DAYS = 10  # == VOLUME_PRV_BASELINE_MIN_OBS -> only the engineered day is classifiable


# --------------------------------------------------------------------------- #
# Bar builders
# --------------------------------------------------------------------------- #
def _bars(rows):
    """rows = [(time, symbol, volume, amount, high, low, close), ...] -> 1min bars.

    ``amount`` is independent of ``volume`` (the per-bar price is ``amount/volume``) and
    high / low / close are set per bar, because this factor reads the day's price RANGE
    (max high, min low) and the previous day's last visible CLOSE alongside the valley
    VWAP. ``normalize_intraday_bars`` sets ``available_time = bar_end + 1min``, so the
    14:50 PIT cutoff excludes any bar with ``bar_end >= 14:50``.
    """
    return normalize_intraday_bars(
        pd.DataFrame(
            {
                "time": pd.to_datetime([r[0] for r in rows]),
                "symbol": [r[1] for r in rows],
                "open": [float(r[6]) for r in rows],
                "high": [float(r[4]) for r in rows],
                "low": [float(r[5]) for r in rows],
                "close": [float(r[6]) for r in rows],
                "volume": [float(r[2]) for r in rows],
                "amount": [float(r[3]) for r in rows],
            }
        ),
        freq="1min",
    )


def _session(day, vols, amts, highs, lows, closes, sym=_SYM, start="09:31:00"):
    """One session of CONSECUTIVE 1-minute bars carrying the given per-bar fields."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, v, a, h, lo, c)
        for i, (v, a, h, lo, c) in enumerate(zip(vols, amts, highs, lows, closes))
    ]


def _background(n_days, n_slots, sym=_SYM, start_day="2021-07-01", last_close=10.0):
    """``n_days`` prior days of flat volume-100 / amount-1000 sessions.

    Same-slot baseline becomes mu=100, sigma=0 -> the eruptive threshold is exactly 100,
    so on the test day a volume of 100 is a VALLEY and anything above erupts. The LAST
    bar of the LAST background day carries ``last_close``: that bar is exactly the
    ``prev_close`` the test day's price range extends with.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        closes = [10.0] * n_slots
        if i == n_days - 1:
            closes[-1] = last_close
        rows += _session(
            day,
            [100.0] * n_slots,
            [1000.0] * n_slots,
            [10.0] * n_slots,
            [10.0] * n_slots,
            closes,
            sym=sym,
        )
    return rows


# Gates small enough that the single engineered test day is the only VALID day; the
# valley-bar and valid-day floors get their own dedicated tests below.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=VALLEY_QUANTILE_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_valley_bars=1,
)


# --------------------------------------------------------------------------- #
# The engineered test day: valley VWAP = 11.0, intraday high 12.0, intraday low 8.0
# --------------------------------------------------------------------------- #
# 12 slots, eruptions at slots 3 and 7 (volume 200, amount 4000 -> price 20).
#   valley bars: 10 x volume 100 (total 1000); five carry amount 1000 (price 10), five
#     amount 1200 (price 12) -> valley VWAP = 11000 / 1000 = 11.0 exactly.
#   price range: every bar has high 11.5 / low 9.5 EXCEPT slot 5 (high 12.0) and slot 9
#     (low 8.0) -- so max(high) = 12.0 and min(low) = 8.0 come from DIFFERENT bars and a
#     first-bar / last-bar bug cannot pass.
_N = 12
_ERUPT = (3, 7)
_HIGH_SLOT, _LOW_SLOT = 5, 9
_VALLEY_VWAP = 11.0
_INTRADAY_HIGH, _INTRADAY_LOW = 12.0, 8.0


def _test_day_arrays():
    vols = [100.0] * _N
    amts = [0.0] * _N
    valley_slots = [s for s in range(_N) if s not in _ERUPT]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 5 else 1200.0
    for s in _ERUPT:
        vols[s] = 200.0
        amts[s] = 4000.0
    highs = [11.5] * _N
    lows = [9.5] * _N
    highs[_HIGH_SLOT] = _INTRADAY_HIGH
    lows[_LOW_SLOT] = _INTRADAY_LOW
    closes = [10.0] * _N
    return vols, amts, highs, lows, closes


def _rows_with_prev_close(prev_close, n_slots=_N):
    """Background (ending on ``prev_close``) + the engineered test day."""
    vols, amts, highs, lows, closes = _test_day_arrays()
    return _background(_BG_DAYS, n_slots, last_close=prev_close) + _session(
        "2021-07-11", vols, amts, highs, lows, closes
    )


# --------------------------------------------------------------------------- #
# Hand-computed quantiles (3 cases: prev_close inside / above / below the range)
# --------------------------------------------------------------------------- #
def test_hand_value_prev_close_inside_intraday_range():
    """prev_close 10.0 is inside [8, 12] -> range unchanged; q = (11-8)/(12-8) = 0.75."""
    out = compute_valley_price_quantile_stats(_bars(_rows_with_prev_close(10.0)), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.75)


def test_hand_value_prev_close_above_intraday_high_extends_the_top():
    """prev_close 16.0 > intraday high 12.0 -> hi becomes 16; q = (11-8)/(16-8) = 0.375.

    A higher previous close pushes the SAME valley VWAP to a LOWER position in the range
    -- the whole point of including prev_close in the range.
    """
    out = compute_valley_price_quantile_stats(_bars(_rows_with_prev_close(16.0)), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.375)


def test_hand_value_prev_close_below_intraday_low_extends_the_bottom():
    """prev_close 4.0 < intraday low 8.0 -> lo becomes 4; q = (11-4)/(12-4) = 0.875.

    Opposite direction from the case above, so a sign / min-max inversion bug in the
    range construction cannot pass both.
    """
    out = compute_valley_price_quantile_stats(_bars(_rows_with_prev_close(4.0)), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.875)


def test_quantile_is_in_the_unit_interval_in_normal_conditions():
    """A valley VWAP inside the range gives q in [0, 1] (the no-clip sanity check)."""
    for prev_close in (4.0, 10.0, 16.0):
        out = compute_valley_price_quantile_stats(
            _bars(_rows_with_prev_close(prev_close)), **_KW
        )
        q = float(out.loc[(_TEST_DAY, _SYM)])
        assert 0.0 <= q <= 1.0


def test_quantile_is_not_clipped_when_the_vwap_escapes_the_range():
    """A VWAP outside [lo, hi] is EXPOSED as q outside [0,1], never clipped.

    The pinned choice: an out-of-range quantile means the range construction is wrong,
    and silently clipping it to [0,1] would hide exactly the defect worth seeing. Here
    the bars' high/low are deliberately set BELOW the traded price so the VWAP escapes.
    """
    vols, amts, highs, lows, closes = _test_day_arrays()
    highs = [10.2] * _N  # max high 10.2 < valley VWAP 11.0
    lows = [9.5] * _N
    rows = _background(_BG_DAYS, _N, last_close=10.0) + _session(
        "2021-07-11", vols, amts, highs, lows, closes
    )
    out = compute_valley_price_quantile_stats(_bars(rows), **_KW)
    q = float(out.loc[(_TEST_DAY, _SYM)])
    # hi = max(10.2, 10.0) = 10.2, lo = min(9.5, 10.0) = 9.5 -> (11-9.5)/0.7 > 1
    assert q == pytest.approx((11.0 - 9.5) / (10.2 - 9.5))
    assert q > 1.0


def test_valley_vwap_uses_the_amount_over_volume_identity():
    """The valley leg is Σamount/Σvolume, verified the LONG way from per-bar prices."""
    bars = _bars(_rows_with_prev_close(10.0))
    visible = prepare_visible_minute_bars(
        bars, extra_columns=("amount", "high", "low", "close")
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    day = work[work["trade_date"] == _TEST_DAY]
    v = day["volume"].to_numpy(dtype=float)
    a = day["amount"].to_numpy(dtype=float)
    is_valley = day["valley"].to_numpy(dtype=bool)
    long_way = float((a[is_valley] / v[is_valley] * v[is_valley]).sum() / v[is_valley].sum())
    assert long_way == pytest.approx(_VALLEY_VWAP)
    assert float(a[is_valley].sum() / v[is_valley].sum()) == pytest.approx(_VALLEY_VWAP)


# --------------------------------------------------------------------------- #
# Day validity gates
# --------------------------------------------------------------------------- #
def _work_frame(days, *, high, low, close, n_bars=4):
    """Hand-built ``peak_mask_for_symbol``-shaped frame with EVERY bar a classifiable valley.

    Bypasses the same-slot baseline warm-up on purpose: the only thing under test is the
    prev_close rule, so every day here is classifiable by construction and a day can only
    be invalid because of the range / previous-close gates.
    """
    rows = []
    for day in days:
        base = pd.Timestamp(day) + pd.Timedelta("09:31:00")
        for i in range(n_bars):
            rows.append(
                {
                    "bar_end": base + pd.Timedelta(minutes=i),
                    "trade_date": pd.Timestamp(day),
                    "volume": 100.0,
                    "amount": 1000.0,  # per-bar price 10.0 -> valley VWAP 10.0
                    "high": high,
                    "low": low,
                    "close": close,
                    "valley": True,
                    "classifiable": True,
                }
            )
    return pd.DataFrame(rows)


def test_first_day_of_the_window_has_no_prev_close_and_is_invalid():
    """A symbol's FIRST visible day cannot have a prev_close -> no value that day.

    Every day in this frame is classifiable and has a real high/low range, so the ONLY
    reason the first day is absent is the missing previous close -- and the later days,
    which do have one, are present.
    """
    days = ["2021-07-01", "2021-07-02", "2021-07-05"]
    work = _work_frame(days, high=12.0, low=8.0, close=10.0)
    q = valley_price_quantile_by_day(work, min_valley_bars=1, min_classifiable=1)
    assert pd.Timestamp(days[0]) not in q.index
    assert pd.Timestamp(days[1]) in q.index
    assert pd.Timestamp(days[2]) in q.index
    # prev_close 10.0 sits inside [8, 12], so the range is unchanged: (10-8)/(12-8).
    assert float(q.loc[pd.Timestamp(days[1])]) == pytest.approx(0.5)


def test_flat_day_with_hi_equal_lo_is_invalid():
    """hi <= lo (no visible price movement at all) -> the day is invalid, not 0/0."""
    vols, amts, _, _, closes = _test_day_arrays()
    flat = [10.0] * _N
    rows = _background(_BG_DAYS, _N, last_close=10.0) + _session(
        "2021-07-11", vols, amts, flat, flat, closes
    )
    stats = compute_valley_price_quantile_stats(_bars(rows), **_KW)
    assert (_TEST_DAY, _SYM) not in stats.index


def test_min_valley_bars_gate_rejects_a_day_with_too_few_valleys():
    """The >= min_valley_bars floor counts TRADABLE valley bars (post-guard)."""
    kw = dict(_KW, min_valley_bars=11)  # the engineered day has exactly 10
    stats = compute_valley_price_quantile_stats(
        _bars(_rows_with_prev_close(10.0)), **kw
    )
    assert (_TEST_DAY, _SYM) not in stats.index
    kw_ok = dict(_KW, min_valley_bars=10)
    stats_ok = compute_valley_price_quantile_stats(
        _bars(_rows_with_prev_close(10.0)), **kw_ok
    )
    assert (_TEST_DAY, _SYM) in stats_ok.index


def test_min_classifiable_gate_rejects_a_thin_day():
    """PR-F's >= min_classifiable floor is enforced unchanged."""
    kw = dict(_KW, min_classifiable=_N + 1)
    stats = compute_valley_price_quantile_stats(
        _bars(_rows_with_prev_close(10.0)), **kw
    )
    assert (_TEST_DAY, _SYM) not in stats.index


def test_zero_volume_valley_bar_contributes_nothing_but_is_still_classified():
    """The positive-trade guard drops a no-trade bar from the VWAP, not from the taxonomy."""
    vols, amts, highs, lows, closes = _test_day_arrays()
    # Slot 0 is a valley; zero its trade so it cannot enter the VWAP sums.
    vols[0], amts[0] = 0.0, 0.0
    rows = _background(_BG_DAYS, _N, last_close=10.0) + _session(
        "2021-07-11", vols, amts, highs, lows, closes
    )
    stats = compute_valley_price_quantile_stats(_bars(rows), **dict(_KW, min_valley_bars=9))
    # Remaining valleys: four at price 10 (amount 1000) and five at price 12 -> VWAP
    # = (4*1000 + 5*1200) / 900 = 10000/900
    expected_vwap = (4 * 1000.0 + 5 * 1200.0) / 900.0
    assert float(stats.loc[(_TEST_DAY, _SYM)]) == pytest.approx(
        (expected_vwap - _INTRADAY_LOW) / (_INTRADAY_HIGH - _INTRADAY_LOW)
    )
    # ... and the day still fails a floor of 10 tradable valley bars.
    thin = compute_valley_price_quantile_stats(_bars(rows), **dict(_KW, min_valley_bars=10))
    assert (_TEST_DAY, _SYM) not in thin.index


# --------------------------------------------------------------------------- #
# Trailing mean + valid-day floor
# --------------------------------------------------------------------------- #
def _multi_valid_day_rows(n_extra_days, quantile_targets):
    """Background + ``n_extra_days`` engineered days whose quantiles are known.

    Each engineered day reuses the standard construction but scales the two valley
    amount tiers so the day's valley VWAP lands on a chosen value; the range is always
    [8, 12] (prev_close 10 stays inside), so q = (vwap - 8) / 4.
    """
    rows = _background(_BG_DAYS, _N, last_close=10.0)
    for i, vwap in enumerate(quantile_targets[:n_extra_days]):
        day = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        vols, _, highs, lows, closes = _test_day_arrays()
        amts = [0.0] * _N
        valley_slots = [s for s in range(_N) if s not in _ERUPT]
        for s in valley_slots:
            amts[s] = 100.0 * vwap  # every valley bar priced at `vwap` -> VWAP = vwap
        for s in _ERUPT:
            amts[s] = 4000.0
        rows += _session(day, vols, amts, highs, lows, closes)
    return rows


def test_trailing_mean_averages_the_daily_quantiles():
    """The factor is the MEAN of q_day over the trailing valid days."""
    vwaps = [9.0, 10.0, 11.0]  # -> q = 0.25, 0.50, 0.75
    rows = _multi_valid_day_rows(3, vwaps)
    stats = compute_valley_price_quantile_stats(_bars(rows), **dict(_KW, min_valid_days=3))
    third_day = pd.Timestamp("2021-07-13")
    assert float(stats.loc[(third_day, _SYM)]) == pytest.approx((0.25 + 0.50 + 0.75) / 3)


def test_below_min_valid_days_is_nan():
    """Fewer than min_valid_days valid days in the window -> NaN (honest missing)."""
    rows = _multi_valid_day_rows(2, [9.0, 10.0])
    stats = compute_valley_price_quantile_stats(_bars(rows), **dict(_KW, min_valid_days=3))
    assert not np.isfinite(stats.to_numpy(dtype=float)).any()


def test_lookback_window_drops_days_beyond_the_horizon():
    """Only the most recent ``lookback_days`` VALID days enter the mean."""
    vwaps = [8.0, 12.0, 12.0]  # q = 0.0, 1.0, 1.0
    rows = _multi_valid_day_rows(3, vwaps)
    stats = compute_valley_price_quantile_stats(
        _bars(rows), **dict(_KW, lookback_days=2, min_valid_days=2)
    )
    third_day = pd.Timestamp("2021-07-13")
    assert float(stats.loc[(third_day, _SYM)]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Reversal factor: the T-1 basis (THE lookahead subtlety of this PR)
# --------------------------------------------------------------------------- #
def _closes_frame(dates, symbols, values):
    """MultiIndex(date, symbol) daily close frame."""
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    return pd.DataFrame({"close": np.asarray(values, dtype=float).reshape(-1)}, index=idx)


def test_reversal_20_is_minus_the_t_minus_1_twenty_day_return():
    """rev20(d) = -(close_{d-1}/close_{d-21} - 1), hand-checked on a known series."""
    dates = pd.bdate_range("2021-01-04", periods=25)
    # close_i = 100 + i so every 20-day span is a known ratio.
    closes = _closes_frame(dates, [_SYM], [[100.0 + i] for i in range(len(dates))])
    rev = reversal_20(closes, days=VALLEY_QUANTILE_REVERSAL_DAYS)
    d = dates[21]  # first date with both d-1 and d-21 present
    expected = -((100.0 + 20) / (100.0 + 0) - 1.0)
    assert float(rev.loc[(d, _SYM)]) == pytest.approx(expected)
    # Dates without a full 21-observation history are NaN, never fabricated.
    assert not np.isfinite(float(rev.loc[(dates[20], _SYM)]))


def test_reversal_20_ignores_day_d_close_entirely():
    """Perturbing close_d must not move rev20 AT d -- it is 15:00 information."""
    dates = pd.bdate_range("2021-01-04", periods=25)
    base = [[100.0 + i] for i in range(len(dates))]
    rev_a = reversal_20(_closes_frame(dates, [_SYM], base))
    bumped = [list(r) for r in base]
    bumped[22][0] = 9999.0  # day d itself
    rev_b = reversal_20(_closes_frame(dates, [_SYM], bumped))
    d = dates[22]
    assert float(rev_a.loc[(d, _SYM)]) == float(rev_b.loc[(d, _SYM)])


# --------------------------------------------------------------------------- #
# Reversal neutralization (cross-sectional OLS residual)
# --------------------------------------------------------------------------- #
def _panel_series(date, symbols, values, name):
    idx = pd.MultiIndex.from_arrays(
        [[pd.Timestamp(date)] * len(symbols), list(symbols)], names=["date", "symbol"]
    )
    return pd.Series(np.asarray(values, dtype=float), index=idx, name=name)


def test_residualization_hand_computed_three_symbol_cross_section():
    """OLS residual of qbar on rev20, hand-computed.

    qbar = [0.2, 0.6, 0.7], rev20 = [-0.1, 0.0, 0.1]
      x_bar = 0, y_bar = 0.5
      Sxy = (-0.1)(-0.3) + 0*(0.1) + (0.1)(0.2) = 0.05 ; Sxx = 0.01 + 0 + 0.01 = 0.02
      slope = 2.5, intercept = 0.5
      fitted = [0.25, 0.5, 0.75] -> residuals = [-0.05, +0.10, -0.05] (sum 0)
    """
    syms = ["A", "B", "C"]
    qbar = _panel_series("2023-05-10", syms, [0.2, 0.6, 0.7], "q")
    rev = _panel_series("2023-05-10", syms, [-0.1, 0.0, 0.1], "rev")
    out = residualize_on_reversal(qbar, rev, min_cross_section=3, name="f")
    assert float(out.loc[(pd.Timestamp("2023-05-10"), "A")]) == pytest.approx(-0.05)
    assert float(out.loc[(pd.Timestamp("2023-05-10"), "B")]) == pytest.approx(0.10)
    assert float(out.loc[(pd.Timestamp("2023-05-10"), "C")]) == pytest.approx(-0.05)
    assert float(out.sum()) == pytest.approx(0.0, abs=1e-12)


def test_residualization_removes_the_reversal_exposure():
    """The residual is ORTHOGONAL to rev20 -- the point of the neutralization."""
    rng = np.random.default_rng(20260720)
    syms = [f"S{i}" for i in range(30)]
    rev_vals = rng.normal(size=30)
    qbar_vals = 0.4 + 1.7 * rev_vals + 0.05 * rng.normal(size=30)  # strong exposure
    qbar = _panel_series("2023-05-10", syms, qbar_vals, "q")
    rev = _panel_series("2023-05-10", syms, rev_vals, "rev")
    assert abs(np.corrcoef(qbar_vals, rev_vals)[0, 1]) > 0.9  # before
    out = residualize_on_reversal(qbar, rev, min_cross_section=10, name="f")
    resid = out.to_numpy(dtype=float)
    assert abs(np.corrcoef(resid, rev_vals)[0, 1]) < 1e-10  # after


def test_residualization_drops_symbols_without_a_reversal_value():
    """A symbol whose rev20 is missing gets a NaN residual -- never a silent 0 fill."""
    syms = ["A", "B", "C", "D"]
    qbar = _panel_series("2023-05-10", syms, [0.2, 0.6, 0.7, 0.9], "q")
    rev = _panel_series("2023-05-10", syms, [-0.1, 0.0, 0.1, np.nan], "rev")
    out = residualize_on_reversal(qbar, rev, min_cross_section=3, name="f")
    assert not np.isfinite(float(out.loc[(pd.Timestamp("2023-05-10"), "D")]))
    # The other three are the hand-computed regression above: D did not perturb it.
    assert float(out.loc[(pd.Timestamp("2023-05-10"), "B")]) == pytest.approx(0.10)


def test_residualization_is_invariant_to_the_reversal_sign_convention():
    """Residualizing on ``rev`` and on ``-rev`` gives the SAME factor.

    Matters because the report's "正向暴露 20 日反转因子" does not say whether its 反转因子
    is the past return or its negation, and our prototype measures the exposure with the
    opposite label to the report's. An intercept OLS residual spans the same column space
    either way (span{1, x} == span{1, -x}), so the shipped factor cannot depend on which
    convention was meant -- the ambiguity is real but harmless, and this test proves it.
    """
    syms = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(11)
    qbar = _panel_series("2023-05-10", syms, rng.normal(size=12), "q")
    rev = _panel_series("2023-05-10", syms, rng.normal(size=12), "rev")
    a = residualize_on_reversal(qbar, rev, min_cross_section=10, name="f")
    b = residualize_on_reversal(qbar, -rev, min_cross_section=10, name="f")
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), atol=1e-12)


def test_residualization_below_min_cross_section_is_all_nan():
    """A thin cross-section -> the whole date is NaN (honest missing)."""
    syms = ["A", "B", "C"]
    qbar = _panel_series("2023-05-10", syms, [0.2, 0.6, 0.7], "q")
    rev = _panel_series("2023-05-10", syms, [-0.1, 0.0, 0.1], "rev")
    out = residualize_on_reversal(qbar, rev, min_cross_section=4, name="f")
    assert not np.isfinite(out.to_numpy(dtype=float)).any()


def test_residualization_degenerate_reversal_cross_section_is_nan():
    """Zero-variance rev20 cannot identify a slope -> NaN, not an unresidualized qbar."""
    syms = ["A", "B", "C"]
    qbar = _panel_series("2023-05-10", syms, [0.2, 0.6, 0.7], "q")
    rev = _panel_series("2023-05-10", syms, [0.3, 0.3, 0.3], "rev")
    out = residualize_on_reversal(qbar, rev, min_cross_section=3, name="f")
    assert not np.isfinite(out.to_numpy(dtype=float)).any()


def test_residualization_is_per_date_and_never_pools_dates():
    """Each date is regressed on ITS OWN cross-section; a second date cannot leak in."""
    syms = ["A", "B", "C"]
    q1 = _panel_series("2023-05-10", syms, [0.2, 0.6, 0.7], "q")
    r1 = _panel_series("2023-05-10", syms, [-0.1, 0.0, 0.1], "rev")
    q2 = _panel_series("2023-05-11", syms, [5.0, 9.0, 40.0], "q")
    r2 = _panel_series("2023-05-11", syms, [3.0, 1.0, -8.0], "rev")
    both = residualize_on_reversal(
        pd.concat([q1, q2]), pd.concat([r1, r2]), min_cross_section=3, name="f"
    )
    alone = residualize_on_reversal(q1, r1, min_cross_section=3, name="f")
    for s in syms:
        key = (pd.Timestamp("2023-05-10"), s)
        assert float(both.loc[key]) == pytest.approx(float(alone.loc[key]))


# --------------------------------------------------------------------------- #
# END-TO-END: the T-1 reversal basis is what the FACTOR uses (the critical test)
# --------------------------------------------------------------------------- #
def _end_to_end_inputs(n_syms=3, n_days=30):
    """Minute bars for ``n_syms`` symbols + a daily qfq close panel over the same dates.

    Each symbol gets the standard background plus engineered days with per-symbol
    quantiles, so every symbol has a finite qbar on the later dates and the
    cross-section is large enough to regress.
    """
    rows = []
    vwap_cycle = [8.5, 9.5, 10.5, 11.5]
    for k in range(n_syms):
        sym = f"S{k}.SZ"
        rows += _background(_BG_DAYS, _N, sym=sym, last_close=10.0)
        for i in range(n_days):
            day = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            vols, _, highs, lows, closes = _test_day_arrays()
            vwap = vwap_cycle[(i + k) % len(vwap_cycle)]
            amts = [0.0] * _N
            for s in range(_N):
                amts[s] = 4000.0 if s in _ERUPT else 100.0 * vwap
            rows += _session(day, vols, amts, highs, lows, closes, sym=sym)
    bars = _bars(rows)

    dates = sorted(pd.unique(bars.index.get_level_values("time").normalize()))
    syms = [f"S{k}.SZ" for k in range(n_syms)]
    rng = np.random.default_rng(7)
    vals = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(len(dates), n_syms)), axis=0))
    closes = pd.DataFrame(
        {"close": vals.reshape(-1)},
        index=pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"]),
    )
    return bars, closes, dates, syms


_E2E_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=5,
    min_valid_days=3,
    min_classifiable=1,
    min_valley_bars=1,
    min_cross_section=3,
)


def test_reversal_uses_t_minus_1_close_not_day_d():
    """THE critical lookahead test: perturbing day d's CLOSE leaves the factor at d fixed.

    The report's reversal is ``-(close_d/close_{d-20} - 1)``, which reads the 15:00 close
    on the very day we decide at 14:50. We use the T-1 basis instead, so day d's close is
    NOT an input to the factor value at d. If someone "simplifies" the implementation back
    to the report's naive form, this test fails.

    THE PERTURBATION MUST BE SINGLE-SYMBOL, and that is not a stylistic choice. Bumping
    EVERY cross-section member's ``close_d`` by the same factor ``k`` sends the naive
    regressor ``1 - close_d/close_{d-20}`` to ``k*rev - (k-1)`` -- an AFFINE
    reparametrization of x. Intercept-OLS residuals are invariant under exactly that
    (``span{1, x} == span{1, a*x + b}``), which is the very property
    ``test_residualization_is_invariant_to_the_reversal_sign_convention`` relies on. So a
    uniform bump leaves the residuals bit-identical under BOTH the T-1 and the naive form
    and can never tell them apart. Perturbing ONE symbol moves that symbol's regressor
    relative to the rest of the cross-section, changes the fitted line, and therefore
    moves every residual on the date -- under the naive form only. Verified by
    monkeypatching ``reversal_20`` to the naive ``shift(0)/shift(days)`` form: this test
    FAILS there and passes here.
    """
    bars, closes, dates, syms = _end_to_end_inputs()
    base = compute_valley_price_quantile(bars, closes, **_E2E_KW)

    d = dates[-3]  # a date with a finite factor value and a full history behind it
    bumped = closes.copy()
    # SINGLE symbol, so the change is NOT an affine map of the whole regressor vector.
    bumped.loc[(d, syms[0]), "close"] = float(bumped.loc[(d, syms[0]), "close"]) * 1.5
    after = compute_valley_price_quantile(bars, bumped, **_E2E_KW)

    on_d_before = base[base.index.get_level_values("date") == d]
    on_d_after = after[after.index.get_level_values("date") == d]
    assert on_d_before.notna().any(), "test is vacuous unless d carries a finite value"
    pd.testing.assert_series_equal(on_d_before, on_d_after)


def test_uniform_close_d_bump_cannot_distinguish_the_two_reversal_bases():
    """Documents WHY the test above must perturb a single symbol.

    A uniform bump is an affine reparametrization of the OLS regressor, so it leaves the
    residuals unchanged even under the naive close_d form. Asserting that here keeps the
    reasoning from being quietly lost: if someone later "simplifies" the critical test
    back to a uniform bump, this test explains what they broke.
    """
    syms = [f"S{i}" for i in range(12)]
    idx = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2023-05-10")] * 12, syms], names=["date", "symbol"]
    )
    rng = np.random.default_rng(4)
    qbar = pd.Series(rng.normal(size=12), index=idx)
    rev = pd.Series(rng.normal(size=12), index=idx)

    base = residualize_on_reversal(qbar, rev, min_cross_section=10, name="f")
    # k=1.5 uniform bump of close_d maps rev -> 1.5*rev - 0.5 for EVERY symbol.
    affine = residualize_on_reversal(qbar, 1.5 * rev - 0.5, min_cross_section=10, name="f")
    np.testing.assert_allclose(
        base.to_numpy(dtype=float), affine.to_numpy(dtype=float), atol=1e-12
    )

    # Moving ONE symbol is not affine on the vector, so it DOES move the residuals.
    single = rev.copy()
    single.iloc[0] = single.iloc[0] * 1.5 - 0.5
    moved = residualize_on_reversal(qbar, single, min_cross_section=10, name="f")
    assert not np.allclose(base.to_numpy(dtype=float), moved.to_numpy(dtype=float))


def test_perturbing_the_t_minus_1_close_does_move_the_factor():
    """TEETH for the test above: the closes are genuinely an input, just lagged by one day.

    Without this, the T-1 test could pass simply by the reversal never being applied.
    """
    bars, closes, dates, syms = _end_to_end_inputs()
    base = compute_valley_price_quantile(bars, closes, **_E2E_KW)

    d = dates[-3]
    d_prev = dates[-4]
    bumped = closes.copy()
    bumped.loc[(d_prev, syms[0]), "close"] = float(
        bumped.loc[(d_prev, syms[0]), "close"]
    ) * 1.5
    after = compute_valley_price_quantile(bars, bumped, **_E2E_KW)

    on_d_before = base[base.index.get_level_values("date") == d]
    on_d_after = after[after.index.get_level_values("date") == d]
    assert not on_d_before.equals(on_d_after), (
        "perturbing close_{d-1} must change rev20 at d and hence the residualized factor"
    )


def test_factor_is_residualized_not_raw_qbar():
    """The returned factor differs from the raw trailing quantile mean.

    Guards against a wiring bug where the reversal neutralization is computed but the
    RAW qbar is returned.
    """
    bars, closes, dates, syms = _end_to_end_inputs()
    raw = compute_valley_price_quantile_stats(
        bars,
        **{k: v for k, v in _E2E_KW.items() if k != "min_cross_section"},
    )
    final = compute_valley_price_quantile(bars, closes, **_E2E_KW)
    common = raw.index.intersection(final.index)
    assert len(common) > 0
    assert not np.allclose(
        raw.loc[common].to_numpy(dtype=float),
        final.loc[common].to_numpy(dtype=float),
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# PIT / leakage / isolation
# --------------------------------------------------------------------------- #
def test_bars_after_the_cutoff_are_invisible_and_the_test_has_teeth():
    """Post-14:50 bars cannot move the value; PRE-cutoff bars can (the teeth)."""
    vols, amts, highs, lows, closes = _test_day_arrays()
    rows = _background(_BG_DAYS, _N, last_close=10.0) + _session(
        "2021-07-11", vols, amts, highs, lows, closes
    )
    base = compute_valley_price_quantile_stats(_bars(rows), **_KW)

    # A wild bar at 14:55 (available 14:56) is AFTER the cutoff -> invisible.
    late = rows + [
        (pd.Timestamp("2021-07-11 14:55:00"), _SYM, 100.0, 500000.0, 9999.0, 0.01, 5000.0)
    ]
    after = compute_valley_price_quantile_stats(_bars(late), **_KW)
    assert float(after.loc[(_TEST_DAY, _SYM)]) == float(base.loc[(_TEST_DAY, _SYM)])

    # The SAME bar placed at 10:00 (visible) DOES move it -> the test is not vacuous.
    early = rows + [
        (pd.Timestamp("2021-07-11 10:00:00"), _SYM, 100.0, 500000.0, 9999.0, 0.01, 5000.0)
    ]
    moved = compute_valley_price_quantile_stats(_bars(early), **_KW)
    assert float(moved.loc[(_TEST_DAY, _SYM)]) != float(base.loc[(_TEST_DAY, _SYM)])


def test_prev_close_comes_from_the_visible_window_not_the_real_daily_close():
    """prev_close is the last bar <= 14:50 of d-1, NOT d-1's 15:00 close.

    A late bar on the PREVIOUS day (after 14:50) must not become the prev_close -- that
    is the single-visibility-definition pin.
    """
    rows = _rows_with_prev_close(10.0)
    base = compute_valley_price_quantile_stats(_bars(rows), **_KW)
    prev_day = (pd.Timestamp("2021-07-11") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with_late_close = rows + [
        (pd.Timestamp(f"{prev_day} 14:56:00"), _SYM, 100.0, 1000.0, 50.0, 50.0, 50.0)
    ]
    after = compute_valley_price_quantile_stats(_bars(with_late_close), **_KW)
    assert float(after.loc[(_TEST_DAY, _SYM)]) == float(base.loc[(_TEST_DAY, _SYM)])


def test_cross_symbol_isolation():
    """A second symbol's bars never touch the first symbol's per-symbol statistics."""
    rows_a = _rows_with_prev_close(10.0)
    alone = compute_valley_price_quantile_stats(_bars(rows_a), **_KW)

    other = []
    for r in _rows_with_prev_close(16.0):
        other.append((r[0], "600000.SH", r[2] * 7.0, r[3] * 3.0, r[4], r[5], r[6]))
    together = compute_valley_price_quantile_stats(_bars(rows_a + other), **_KW)
    assert float(together.loc[(_TEST_DAY, _SYM)]) == float(alone.loc[(_TEST_DAY, _SYM)])


def test_prev_close_does_not_cross_symbols():
    """Symbol B's previous-day close cannot become symbol A's prev_close."""
    rows_a = _rows_with_prev_close(4.0)
    alone = compute_valley_price_quantile_stats(_bars(rows_a), **_KW)
    other = [
        (r[0], "600000.SH", r[2], r[3], r[4], r[5], r[6] * 25.0)
        for r in _rows_with_prev_close(4.0)
    ]
    together = compute_valley_price_quantile_stats(_bars(rows_a + other), **_KW)
    assert float(together.loc[(_TEST_DAY, _SYM)]) == float(alone.loc[(_TEST_DAY, _SYM)])


def test_empty_bars_return_empty_schema_shaped_output():
    out = compute_valley_price_quantile_stats(empty_intraday_bars(), **_KW)
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]


def test_inputs_are_never_mutated():
    bars = _bars(_rows_with_prev_close(10.0))
    before = bars.copy(deep=True)
    compute_valley_price_quantile_stats(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_parameter_validation():
    bars = _bars(_rows_with_prev_close(10.0))
    with pytest.raises(ValueError, match="lookback_days"):
        compute_valley_price_quantile_stats(bars, **dict(_KW, lookback_days=0))
    with pytest.raises(ValueError, match="min_valley_bars"):
        compute_valley_price_quantile_stats(bars, **dict(_KW, min_valley_bars=0))
    with pytest.raises(ValueError, match="min_cross_section"):
        residualize_on_reversal(
            _panel_series("2023-05-10", ["A"], [0.1], "q"),
            _panel_series("2023-05-10", ["A"], [0.1], "r"),
            min_cross_section=1,
            name="f",
        )


# --------------------------------------------------------------------------- #
# Factor class / spec
# --------------------------------------------------------------------------- #
def test_factor_selects_the_pre_aggregated_column():
    f = ValleyPriceQuantileFactor()
    assert f.name == f"valley_price_quantile_{VALLEY_QUANTILE_LOOKBACK_DAYS}"
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-03"), "A")], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.25], "close": [10.0]}, index=idx)
    out = f.compute(panel)
    assert float(out.iloc[0]) == pytest.approx(0.25)
    assert out.name == f.name


def test_factor_raises_readably_without_the_column():
    f = ValleyPriceQuantileFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-03"), "A")], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"close": [10.0]}, index=idx))


def test_spec_declares_the_pre_registered_sign_and_the_pinned_deviations():
    spec = ValleyPriceQuantileFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.expected_ic_sign == 1
    assert spec.is_intraday is False
    assert spec.return_basis == "close_to_close"
    text = spec.description
    # The two subtleties this PR exists to get right must be ON the spec.
    assert "prev_close" in text or "previous" in text.lower()
    assert "14:50" in text
    assert "reversal" in text.lower()
    assert "d-1" in text or "T-1" in text or "t-1" in text.lower()


def test_factor_rejects_a_bad_lookback():
    with pytest.raises(ValueError, match="lookback_days"):
        ValleyPriceQuantileFactor(lookback_days=0)


# --------------------------------------------------------------------------- #
# The five already-merged factors must not drift
# --------------------------------------------------------------------------- #
def test_prior_factor_modules_are_untouched_by_this_pr():
    """PR-L must not have changed the shared classification or the five prior factors.

    A cheap structural guard: the reused entry points still exist with the same
    behaviour on a trivial input, so an accidental edit to the shared module shows up
    here as well as in the five factors' own test files.
    """
    from data.clean.intraday_ridge_return import compute_ridge_minute_return
    from data.clean.intraday_valley_ridge_vwap import compute_valley_ridge_vwap_ratio
    from data.clean.intraday_valley_vwap import compute_valley_relative_vwap

    empty = empty_intraday_bars()
    for fn in (
        compute_valley_relative_vwap,
        compute_valley_ridge_vwap_ratio,
        compute_ridge_minute_return,
    ):
        out = fn(empty)
        assert out.empty
        assert list(out.index.names) == ["date", "symbol"]

    # The shared taxonomy constants are the ones PR-F pinned (unchanged by PR-L).
    assert VOLUME_PRV_SIGMA_K == 1.0
    assert VOLUME_PRV_BASELINE_DAYS == 20
    assert VOLUME_PRV_BASELINE_MIN_OBS == 10
    assert VOLUME_PRV_MIN_CLASSIFIABLE == 100
    assert VOLUME_PRV_MIN_VALID_DAYS == 10


def test_definition_constants_are_the_pinned_values():
    assert VALLEY_QUANTILE_LOOKBACK_DAYS == 20
    assert VALLEY_QUANTILE_MIN_VALLEY_BARS == 20
    assert VALLEY_QUANTILE_REVERSAL_DAYS == 20
    assert VALLEY_QUANTILE_MIN_CROSS_SECTION == 10


def test_valley_price_quantile_by_day_is_reusable_on_a_prepared_frame():
    """The per-day seam is callable directly (what the trailing mean is built on)."""
    bars = _bars(_rows_with_prev_close(16.0))
    visible = prepare_visible_minute_bars(
        bars, extra_columns=("amount", "high", "low", "close")
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    q = valley_price_quantile_by_day(work, min_valley_bars=1, min_classifiable=1)
    assert float(q.loc[_TEST_DAY]) == pytest.approx(0.375)
