"""PR-J: VALLEY/RIDGE VWAP-RATIO factor.

Same volume classification as PR-F / PR-H / PR-I (REUSED, not re-implemented), and the
same VWAP identity as PR-I -- but the DENOMINATOR is swapped from the whole visible day
to the RIDGE bars, so the factor contrasts the two behavioural groups head-on. A "ridge"
(量岭) is an ERUPTIVE minute that is NOT an isolated peak: PR-F's internal
``eruptive & ~peak``, now exposed as a first-class ``ridge`` column. The factor is the
trailing-20-valid-day mean of ``valley VWAP / ridge VWAP``. Sign is pre-registered +1.

Hand cases build a constant BACKGROUND of prior days so the same-slot baseline is exact
(mu=100, sigma=0 -> the eruptive threshold is exactly 100), then a test day whose volumes
place valleys / ridge runs / an isolated peak at chosen slots and whose AMOUNTS are set
independently, so both VWAPs -- and therefore the ratio -- are known in closed form.
Every hand day deliberately contains an isolated PEAK whose price differs sharply from
the ridges: it must never leak into the ridge leg, which is the one way this factor can
silently degenerate into "eruptive VWAP".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_valley_ridge_vwap import (
    VALLEY_RIDGE_LOOKBACK_DAYS,
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
    compute_valley_ridge_vwap_ratio,
    valley_ridge_vwap_ratio_by_day,
)
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import ValleyRidgeVwapRatioFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, volume, amount), ...] -> normalized 1min bars.

    ``amount`` is set INDEPENDENTLY of ``volume`` -- that is the point: the per-bar price
    is ``amount / volume``. OHLC are dummy constants (this factor reads only volume +
    amount). ``normalize_intraday_bars`` sets ``available_time = bar_end + 1min``, so the
    14:50 PIT cutoff excludes any bar with ``bar_end >= 14:50``.
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
            "amount": [float(r[3]) for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, vols, amts, sym=_SYM, start="09:31:00"):
    """One session of CONSECUTIVE 1-minute bars carrying ``vols`` / ``amts``."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, v, a)
        for i, (v, a) in enumerate(zip(vols, amts))
    ]


def _background(n_days, n_slots, sym=_SYM, start_day="2021-07-01"):
    """``n_days`` prior days of flat volume-100 / amount-1000 sessions.

    Same-slot baseline becomes mu=100, sigma=0 -> the eruptive threshold is exactly 100,
    so on the test day a volume of 100 is a VALLEY and anything above erupts.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, [1000.0] * n_slots, sym=sym)
    return rows


_BG_DAYS = 10
_TEST_DAY = pd.Timestamp("2021-07-11")

# Gates small enough that the single engineered test day is the only VALID day; the
# valley-bar / ridge-bar / valid-day floors get their own dedicated tests below.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=VALLEY_RIDGE_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_valley_bars=1,
    min_ridge_bars=1,
)


# --------------------------------------------------------------------------- #
# Hand-computed VWAP ratios (2 non-trivial cases, closed-form fractions)
# --------------------------------------------------------------------------- #
# CASE A -- 16 slots.
#   ridge runs at slots (3,4) and (8,9): each bar volume 200 / amount 4000 -> price 20.
#     Every one of the four has an ERUPTIVE neighbour, so none is an isolated peak ->
#     all four are ridges. ridge VWAP = 16000 / 800 = 20.0
#   an ISOLATED eruption at slot 12 (volume 300 / amount 9000 -> price 30) IS a peak,
#     so it must be excluded from the ridge leg (the discriminating case).
#   valley bars: 11 x volume 100; six carry amount 1000 (price 10), five amount 1200
#     (price 12) -> valley VWAP = 12000 / 1100
#   ratio = (12000/1100) / 20 = 6/11 = 0.545454...
# The ridges traded far ABOVE the calm minutes, so the ratio sits below 1.
_CASE_A_N = 16
_CASE_A_RIDGES = (3, 4, 8, 9)
_CASE_A_PEAK = 12
_CASE_A_RATIO = 6.0 / 11.0


def _case_a_day():
    n = _CASE_A_N
    vols = [100.0] * n
    amts = [0.0] * n
    for s in _CASE_A_RIDGES:
        vols[s] = 200.0
        amts[s] = 4000.0
    vols[_CASE_A_PEAK] = 300.0
    amts[_CASE_A_PEAK] = 9000.0
    valley_slots = [
        s for s in range(n) if s not in _CASE_A_RIDGES and s != _CASE_A_PEAK
    ]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 6 else 1200.0
    return vols, amts


# CASE B -- 18 slots, ridge runs at (2,3), (7,8), (13,14): volume 300 / amount 2400 ->
#   price 8  -> ridge VWAP = 14400 / 1800 = 8.0
#   isolated peak at slot 16 (volume 500 / amount 25000 -> price 50), excluded.
#   valley bars: 11 x volume 100; six amount 1000, five amount 1500 -> 13500 / 1100
#   ratio = (13500/1100) / 8 = 135/88 = 1.534090...
# Here the ridges traded BELOW the calm minutes -> ratio > 1, the opposite direction from
# case A, so a sign / inversion bug cannot pass both.
_CASE_B_N = 18
_CASE_B_RIDGES = (2, 3, 7, 8, 13, 14)
_CASE_B_PEAK = 16
_CASE_B_RATIO = 135.0 / 88.0


def _case_b_day():
    n = _CASE_B_N
    vols = [100.0] * n
    amts = [0.0] * n
    for s in _CASE_B_RIDGES:
        vols[s] = 300.0
        amts[s] = 2400.0
    vols[_CASE_B_PEAK] = 500.0
    amts[_CASE_B_PEAK] = 25_000.0
    valley_slots = [
        s for s in range(n) if s not in _CASE_B_RIDGES and s != _CASE_B_PEAK
    ]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 6 else 1500.0
    return vols, amts


def test_hand_value_case_a_ratio_below_one():
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] < 1.0


def test_hand_value_case_b_ratio_above_one():
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_B_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] > 1.0


def test_isolated_peak_is_excluded_from_the_ridge_leg():
    """The ridge denominator is ``eruptive & ~peak``, NOT all eruptive bars.

    Moving ONLY the isolated peak's price must leave the factor untouched; if the
    implementation summed every eruptive bar the ratio would move. This is the single
    most important discriminator in the file.
    """
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    base = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)

    vols2, amts2 = _case_a_day()
    amts2[_CASE_A_PEAK] = 90_000.0  # price 300 instead of 30
    rows2 = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols2, amts2)
    moved = compute_valley_ridge_vwap_ratio(_bars(rows2), **_KW)

    key = (_TEST_DAY, _SYM)
    assert base.loc[key] == pytest.approx(_CASE_A_RATIO)
    assert moved.loc[key] == pytest.approx(_CASE_A_RATIO)

    # ...and the day really does contain an isolated peak (the assertion is not vacuous)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert bool(day.loc[_CASE_A_PEAK, "peak"]) is True
    assert bool(day.loc[_CASE_A_PEAK, "ridge"]) is False


def test_ridge_mask_is_eruptive_with_an_eruptive_neighbour_on_the_hand_day():
    """The ridge set on case A is EXACTLY the two adjacent-eruption runs."""
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    ridge_slots = tuple(day.index[day["ridge"].to_numpy(dtype=bool)])
    assert ridge_slots == _CASE_A_RIDGES


def test_vwap_uses_the_amount_over_volume_aggregation_identity():
    """sum(p_i * v_i) / sum(v_i) with p_i = amount_i/volume_i IS sum(amount)/sum(volume).

    The PINNED VWAP definition, recomputed the LONG way from per-bar prices for BOTH
    legs (valley bars and ridge bars) and checked against the closed-form ratio the
    factor returns -- so the identity is verified, not assumed.
    """
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    bars = _bars(rows)

    visible = prepare_visible_minute_bars(bars, extra_columns=("amount",))
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    day = work[work["trade_date"] == _TEST_DAY]

    v = day["volume"].to_numpy(dtype=float)
    a = day["amount"].to_numpy(dtype=float)
    price = a / v  # per-bar VWAP
    is_valley = day["valley"].to_numpy(dtype=bool)
    is_ridge = day["ridge"].to_numpy(dtype=bool)

    valley_long = float((price[is_valley] * v[is_valley]).sum() / v[is_valley].sum())
    assert valley_long == pytest.approx(float(a[is_valley].sum() / v[is_valley].sum()))
    assert valley_long == pytest.approx(13500.0 / 1100.0)

    ridge_long = float((price[is_ridge] * v[is_ridge]).sum() / v[is_ridge].sum())
    assert ridge_long == pytest.approx(float(a[is_ridge].sum() / v[is_ridge].sum()))
    assert ridge_long == pytest.approx(8.0)

    out = compute_valley_ridge_vwap_ratio(bars, **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(valley_long / ridge_long)


def test_moving_the_ridge_price_moves_the_ratio_inversely():
    """The ridge leg really is the DENOMINATOR (guard against a swapped ratio)."""
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    base = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)

    vols2, amts2 = _case_a_day()
    for s in _CASE_A_RIDGES:
        amts2[s] = 8000.0  # ridge price 40 instead of 20 -> denominator doubles
    rows2 = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols2, amts2)
    dearer = compute_valley_ridge_vwap_ratio(_bars(rows2), **_KW)

    key = (_TEST_DAY, _SYM)
    assert dearer.loc[key] == pytest.approx(base.loc[key] / 2.0)


def test_day_where_valleys_and_ridges_share_one_price_has_ratio_exactly_one():
    """An exact structural identity: the factor is centred on 1.

    Every bar trades at price 10, so both VWAPs are 10 whatever the volume split.
    """
    n = _CASE_A_N
    vols, _ = _case_a_day()
    amts = [10.0 * v for v in vols]  # price == 10 everywhere
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Ridge scarcity: the >= 10 tradable-ridge-bar gate (PINNED lower than the valley
# floor because ridges are structurally far rarer)
# --------------------------------------------------------------------------- #
def _ridge_runs_day(run_lengths, n_slots):
    """A day whose eruptive bars form CONSECUTIVE runs of the given lengths.

    Runs are separated by 2 mild bars, so every eruptive bar has an eruptive neighbour
    and a run of length L contributes exactly L ridge bars (no isolated peaks at all).
    Ridge bars: volume 200 / amount 4000 (price 20); valleys: volume 100 / amount 1000.
    """
    vols = [100.0] * n_slots
    amts = [1000.0] * n_slots
    slot = 1
    for length in run_lengths:
        if slot + length > n_slots:
            raise AssertionError("test day too short for the requested runs")
        for k in range(length):
            vols[slot + k] = 200.0
            amts[slot + k] = 4000.0
        slot += length + 2
    return vols, amts


def _ridge_count(vols, amts, n_slots):
    """Ridge-bar count on the engineered day, straight off the reused mask."""
    rows = _background(_BG_DAYS, n_slots) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY]
    return int(day["ridge"].sum()), int(day["peak"].sum())


_RIDGE_GATE_N = 24


def test_exactly_nine_ridge_bars_fails_the_default_ten_gate():
    vols, amts = _ridge_runs_day([2, 2, 2, 3], _RIDGE_GATE_N)  # 9 ridges
    assert _ridge_count(vols, amts, _RIDGE_GATE_N) == (9, 0)
    rows = _background(_BG_DAYS, _RIDGE_GATE_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(
        _bars(rows), **{**_KW, "min_ridge_bars": VALLEY_RIDGE_MIN_RIDGE_BARS}
    )
    assert out.dropna().empty


def test_exactly_ten_ridge_bars_passes_the_default_ten_gate():
    vols, amts = _ridge_runs_day([2, 2, 2, 2, 2], _RIDGE_GATE_N)  # 10 ridges
    assert _ridge_count(vols, amts, _RIDGE_GATE_N) == (10, 0)
    rows = _background(_BG_DAYS, _RIDGE_GATE_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(
        _bars(rows), **{**_KW, "min_ridge_bars": VALLEY_RIDGE_MIN_RIDGE_BARS}
    )
    # 14 valleys @ price 10, 10 ridges @ price 20 -> ratio exactly 0.5
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.5)


def test_ridge_bar_count_is_taken_AFTER_the_positive_trade_guard():
    """The >= min_ridge_bars gate counts ridges that actually CONTRIBUTE to the VWAP.

    Ten classifiable ridge bars, but one traded nothing -> only nine support the
    denominator, so the default gate must invalidate the day.
    """
    vols, amts = _ridge_runs_day([2, 2, 2, 2, 2], _RIDGE_GATE_N)
    # A ZERO-VOLUME bar could not erupt at all, so the untradable ridge is made with a
    # zero AMOUNT: the bar still clears the volume threshold (200 > 100) and is still
    # classified as a ridge, but it carries no traded value for the VWAP.
    amts[1] = 0.0
    assert _ridge_count(vols, amts, _RIDGE_GATE_N) == (10, 0)
    rows = _background(_BG_DAYS, _RIDGE_GATE_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(
        _bars(rows), **{**_KW, "min_ridge_bars": VALLEY_RIDGE_MIN_RIDGE_BARS}
    )
    assert out.dropna().empty
    # ...and a floor of 9 accepts the day, proving it was the COUNT that blocked it
    ok = compute_valley_ridge_vwap_ratio(_bars(rows), **{**_KW, "min_ridge_bars": 9})
    assert np.isfinite(ok.loc[(_TEST_DAY, _SYM)])


def test_day_with_no_ridge_is_invalid():
    """Only isolated peaks erupt -> no ridge denominator -> honest NaN, never 0/0."""
    n = 12
    vols = [100.0] * n
    amts = [1000.0] * n
    for s in (3, 7):  # isolated -> peaks, not ridges
        vols[s] = 200.0
        amts[s] = 4000.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.dropna().empty


def test_day_with_no_tradable_ridge_is_invalid():
    vols, amts = _case_a_day()
    for s in _CASE_A_RIDGES:
        amts[s] = 0.0  # classifiable ridges that traded no value
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.dropna().empty


def test_day_with_no_tradable_valley_is_invalid():
    vols, amts = _case_a_day()
    for s in range(_CASE_A_N):
        if s not in _CASE_A_RIDGES and s != _CASE_A_PEAK:
            vols[s] = 0.0
            amts[s] = 0.0
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.dropna().empty


def test_degenerate_bars_are_dropped_from_both_legs_without_polluting_either():
    """Untradable bars on BOTH sides of the taxonomy leave the hand value exact."""
    n = _CASE_A_N + 3
    vols, amts = _case_a_day()
    # slot 16: zero-volume valley; slot 17: zero-amount valley; slot 18: negative amount
    vols = vols + [0.0, 50.0, 50.0]
    amts = amts + [500.0, 0.0, -9_999.0]
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)


def test_valley_bar_gate_still_applies_independently_of_the_ridge_gate():
    """The two floors are separate gates; the valley one is unchanged from PR-I."""
    vols, amts = _case_a_day()  # 11 valley bars
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    bars = _bars(rows)
    ok = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "min_valley_bars": 11})
    assert ok.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)
    too_few = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "min_valley_bars": 12})
    assert too_few.dropna().empty


# --------------------------------------------------------------------------- #
# Window mechanics: the trailing mean over VALID days
# --------------------------------------------------------------------------- #
def _two_case_days():
    """Background + day1 = case A pattern, day2 = case B pattern, on a shared grid."""
    n = _CASE_B_N
    a_vols, a_amts = _case_a_day()
    pad = n - _CASE_A_N
    a_vols = a_vols + [100.0] * pad
    a_amts = a_amts + [1000.0] * pad
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", a_vols, a_amts)
    rows += _session("2021-07-12", *_case_b_day())
    return rows


def test_factor_is_the_mean_of_the_trailing_valid_day_ratios():
    bars = _bars(_two_case_days())
    ratios = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "lookback_days": 1})
    d1, d2 = pd.Timestamp("2021-07-11"), pd.Timestamp("2021-07-12")
    r1, r2 = ratios.loc[(d1, _SYM)], ratios.loc[(d2, _SYM)]
    assert r1 != pytest.approx(r2)

    out = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "lookback_days": 2})
    assert out.loc[(d1, _SYM)] == pytest.approx(r1)
    assert out.loc[(d2, _SYM)] == pytest.approx((r1 + r2) / 2.0)


def test_window_drops_days_older_than_lookback():
    rows = _two_case_days() + _session("2021-07-13", *_case_b_day())
    bars = _bars(rows)
    per_day = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "lookback_days": 1})
    d1, d2, d3 = (pd.Timestamp(f"2021-07-1{k}") for k in (1, 2, 3))
    out = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "lookback_days": 2})
    expected = (per_day.loc[(d2, _SYM)] + per_day.loc[(d3, _SYM)]) / 2.0
    assert out.loc[(d3, _SYM)] == pytest.approx(expected)
    three = compute_valley_ridge_vwap_ratio(bars, **{**_KW, "lookback_days": 3})
    assert three.loc[(d3, _SYM)] != pytest.approx(expected)
    assert np.isfinite(per_day.loc[(d1, _SYM)])


def test_invalid_days_do_not_occupy_a_slot_in_the_trailing_window():
    """A day that fails a gate is ABSENT, not NaN -- it must not shorten the window."""
    n = _CASE_B_N
    a_vols, a_amts = _case_a_day()
    a_vols = a_vols + [100.0] * (n - _CASE_A_N)
    a_amts = a_amts + [1000.0] * (n - _CASE_A_N)
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", a_vols, a_amts)
    # a day with NO ridge at all -> invalid, contributes nothing
    rows += _session("2021-07-12", [100.0] * n, [1000.0] * n)
    rows += _session("2021-07-13", *_case_b_day())
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **{**_KW, "lookback_days": 2})
    per_day = compute_valley_ridge_vwap_ratio(_bars(rows), **{**_KW, "lookback_days": 1})
    d1, d2, d3 = (pd.Timestamp(f"2021-07-1{k}") for k in (1, 2, 3))
    assert (d2, _SYM) not in per_day.index  # the flat day never produced a value
    # day 3's 2-day window pools day 1 and day 3, skipping the invalid day 2
    expected = (per_day.loc[(d1, _SYM)] + per_day.loc[(d3, _SYM)]) / 2.0
    assert out.loc[(d3, _SYM)] == pytest.approx(expected)


def test_min_valid_days_floor_returns_nan_until_enough_valid_days():
    rows = _background(_BG_DAYS, _CASE_B_N)
    for k in range(3):
        d = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        rows += _session(d, *_case_b_day())
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **{**_KW, "min_valid_days": 3})
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-11"), _SYM)])
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-12"), _SYM)])
    assert np.isfinite(out.loc[(pd.Timestamp("2021-07-13"), _SYM)])


def test_min_classifiable_gate_invalidates_thin_days():
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **{**_KW, "min_classifiable": 100})
    assert out.dropna().empty


def test_baseline_insufficient_yields_no_value():
    rows = _background(9, _CASE_A_N) + _session("2021-07-10", *_case_a_day())
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    assert out.dropna().empty


def test_default_gates_match_the_pinned_definition():
    assert VALLEY_RIDGE_LOOKBACK_DAYS == 20
    assert VALLEY_RIDGE_MIN_VALLEY_BARS == 20
    # PINNED LOWER than the valley floor: ridges are structurally far rarer (a bar must
    # erupt AND have an eruptive neighbour).
    assert VALLEY_RIDGE_MIN_RIDGE_BARS == 10
    assert VOLUME_PRV_MIN_VALID_DAYS == 10


# --------------------------------------------------------------------------- #
# PIT: no lookahead, no post-cutoff influence -- WITH TEETH
# --------------------------------------------------------------------------- #
_LATE_START = "14:50:00"
_LATE_SLOTS = 6


def _two_block_background(n_days, early_slots, sym=_SYM, start_day="2021-07-01"):
    """Background whose sessions cover BOTH the early block and the post-cutoff block.

    Giving the late slots a same-slot baseline is what puts TEETH in the leakage test:
    if the 14:50 truncation were removed, the late bars would be fully classifiable and
    would really change the factor -- so the "perturbation changes nothing" assertion is
    not passing merely because the late bars are unclassifiable noise.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * early_slots, [1000.0] * early_slots, sym=sym)
        rows += _session(
            day, [100.0] * _LATE_SLOTS, [1000.0] * _LATE_SLOTS,
            sym=sym, start=_LATE_START,
        )
    return rows


def _late_block(day, sym=_SYM):
    """A post-14:50 block with two ridge runs priced ONE (wildly off the early block)."""
    vols = [100.0] * _LATE_SLOTS
    amts = [1000.0] * _LATE_SLOTS
    for s in (0, 1, 3, 4):
        vols[s] = 500.0
        amts[s] = 500.0  # price 1
    return _session(day, vols, amts, sym=sym, start=_LATE_START)


def test_perturbing_post_1450_bars_does_not_change_factor():
    rows = _two_block_background(_BG_DAYS, _CASE_A_N)
    rows += _session("2021-07-11", *_case_a_day())
    clean = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)
    perturbed = compute_valley_ridge_vwap_ratio(
        _bars(rows + _late_block("2021-07-11")), **_KW
    )
    key = (_TEST_DAY, _SYM)
    assert clean.loc[key] == pytest.approx(_CASE_A_RATIO)
    assert perturbed.loc[key] == pytest.approx(clean.loc[key])


def test_leakage_test_has_teeth_removing_the_cutoff_changes_the_value():
    """The counter-test: with the 14:50 truncation DISABLED the value moves materially.

    Without this, the assertion above could pass for the wrong reason (post-cutoff bars
    that are inert anyway). Here the same bars are shown to carry a large, real effect
    the moment the cutoff stops hiding them -- so the truncation is doing the work.
    """
    rows = _two_block_background(_BG_DAYS, _CASE_A_N)
    rows += _session("2021-07-11", *_case_a_day())
    rows += _late_block("2021-07-11")
    bars = _bars(rows)

    truncated = compute_valley_ridge_vwap_ratio(bars, **_KW)
    untruncated = compute_valley_ridge_vwap_ratio(
        bars, decision_time="23:59:59", **_KW
    )
    key = (_TEST_DAY, _SYM)
    a, b = truncated.loc[key], untruncated.loc[key]
    assert a == pytest.approx(_CASE_A_RATIO)
    assert np.isfinite(b)
    # the late block adds four cheap ridge bars, collapsing the denominator: the value
    # must move by far more than any rounding tolerance
    assert abs(b - a) / a > 0.5
    assert b > 1.0 > a


def test_future_day_does_not_change_earlier_factor():
    base = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    a = compute_valley_ridge_vwap_ratio(_bars(base), **_KW)
    future = base + _session(
        "2021-07-12", [9_999.0] * _CASE_A_N, [1.0] * _CASE_A_N
    )
    b = compute_valley_ridge_vwap_ratio(_bars(future), **_KW)
    key = (_TEST_DAY, _SYM)
    assert np.isfinite(a.loc[key])
    assert a.loc[key] == pytest.approx(b.loc[key])


# --------------------------------------------------------------------------- #
# Per-symbol isolation
# --------------------------------------------------------------------------- #
def test_per_symbol_isolation():
    n = _CASE_B_N
    a_vols, a_amts = _case_a_day()
    a_vols = a_vols + [100.0] * (n - _CASE_A_N)
    a_amts = a_amts + [1000.0] * (n - _CASE_A_N)

    rows = _background(_BG_DAYS, n, sym="AAA.SZ")
    rows += _session("2021-07-11", a_vols, a_amts, sym="AAA.SZ")
    rows += _background(_BG_DAYS, n, sym="BBB.SZ")
    rows += _session("2021-07-11", *_case_b_day(), sym="BBB.SZ")
    out = compute_valley_ridge_vwap_ratio(_bars(rows), **_KW)

    solo_a = compute_valley_ridge_vwap_ratio(
        _bars(
            _background(_BG_DAYS, n, sym="AAA.SZ")
            + _session("2021-07-11", a_vols, a_amts, sym="AAA.SZ")
        ),
        **_KW,
    )
    assert out.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(
        solo_a.loc[(_TEST_DAY, "AAA.SZ")]
    )
    assert out.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(_CASE_B_RATIO)


# --------------------------------------------------------------------------- #
# §0 REUSE NON-DRIFT: the ridge exposure changed nothing for PR-F / PR-H / PR-I
# --------------------------------------------------------------------------- #
def test_ridge_is_exactly_eruptive_and_not_peak():
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    classifiable = work["classifiable"].to_numpy(dtype=bool)
    ridge = work["ridge"].to_numpy(dtype=bool)
    peak = work["peak"].to_numpy(dtype=bool)
    # reconstruct "eruptive" independently from the raw threshold rule
    eruptive = classifiable & (
        work["volume"].to_numpy(dtype=float) > work["thr"].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(ridge, eruptive & ~peak)
    day = work["trade_date"].to_numpy() == np.datetime64(_TEST_DAY)
    assert ridge[day].any() and peak[day].any()  # not vacuous


def test_the_three_masks_partition_the_classifiable_bars():
    """valley | peak | ridge == classifiable, pairwise disjoint.

    The structural guarantee that adding ``ridge`` did not carve anything out of the
    existing masks: every classifiable bar lands in exactly one of the three.
    """
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    classifiable = work["classifiable"].to_numpy(dtype=bool)
    valley = work["valley"].to_numpy(dtype=bool)
    peak = work["peak"].to_numpy(dtype=bool)
    ridge = work["ridge"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(valley | peak | ridge, classifiable)
    assert not (valley & peak).any()
    assert not (valley & ridge).any()
    assert not (peak & ridge).any()


def test_session_boundary_eruption_is_a_ridge_not_a_peak():
    """PINNED: an eruptive bar whose isolation is UNPROVABLE counts as a ridge.

    The first visible bar of the day has no previous neighbour, so PR-F conservatively
    refuses to call it a peak; the complement therefore makes it a ridge. Disclosed
    behaviour, locked here so it can never drift silently.
    """
    n = 12
    vols = [100.0] * n
    amts = [1000.0] * n
    vols[0] = 200.0   # first visible bar erupts -> no previous neighbour
    amts[0] = 4000.0
    vols[5] = 200.0   # an isolated eruption in the interior -> a real peak
    amts[5] = 4000.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert bool(day.loc[0, "ridge"]) is True
    assert bool(day.loc[0, "peak"]) is False
    assert bool(day.loc[5, "peak"]) is True
    assert bool(day.loc[5, "ridge"]) is False


def test_valley_and_peak_and_classifiable_unchanged_by_the_ridge_exposure():
    """The three PRE-EXISTING masks are bit-identical with the ridge column present."""
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    bars = _bars(rows)
    plain = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars).reset_index(drop=True)
    )
    carried = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars, extra_columns=("amount",)).reset_index(drop=True)
    )
    for col in ("trade_date", "bar_end", "slot", "volume", "classifiable", "valley", "peak"):
        pd.testing.assert_series_equal(carried[col], plain[col], check_names=False)


def test_volume_peak_count_unchanged_by_the_ridge_exposure():
    """PR-F's factor value is bit-identical (same hand case as the PR-I lock)."""
    from data.clean.intraday_volume_prv import compute_volume_peak_count

    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    kw = dict(
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        lookback_days=1,
        min_valid_days=1,
        min_classifiable=1,
    )
    out = compute_volume_peak_count(_bars(rows), **kw)
    # case B has exactly ONE isolated eruption (slot 16); the three adjacent runs are
    # ridges, not peaks
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(1.0)


def test_peak_interval_kurtosis_unchanged_by_the_ridge_exposure():
    """PR-H's factor value is bit-identical (the PR-H hand case, verbatim)."""
    from data.clean.intraday_peak_interval import compute_peak_interval_kurtosis

    n = 25
    vols = [100.0] * n
    for p in (1, 3, 6, 10, 15, 21, 23):
        vols[p] = 200.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, [1000.0] * n)
    out = compute_peak_interval_kurtosis(
        _bars(rows),
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        lookback_days=20,
        min_valid_days=1,
        min_classifiable=1,
        min_intervals=4,
    )
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(-1.48125)


def test_valley_relative_vwap_unchanged_by_the_ridge_exposure():
    """PR-I's factor value is bit-identical (the PR-I case-A hand value, verbatim)."""
    from data.clean.intraday_valley_vwap import compute_valley_relative_vwap

    n = 12
    erupt = (3, 7)
    vols = [100.0] * n
    amts = [0.0] * n
    valley_slots = [s for s in range(n) if s not in erupt]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 5 else 1200.0
    for s in erupt:
        vols[s] = 200.0
        amts[s] = 4000.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    out = compute_valley_relative_vwap(
        _bars(rows),
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        lookback_days=20,
        min_valid_days=1,
        min_classifiable=1,
        min_valley_bars=1,
    )
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(77.0 / 95.0)


# --------------------------------------------------------------------------- #
# The per-day ratio helper (exposed for the reuse / diagnostics path)
# --------------------------------------------------------------------------- #
def test_valley_ridge_vwap_ratio_by_day_returns_only_valid_days():
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("amount",)
        ).reset_index(drop=True)
    )
    ratio = valley_ridge_vwap_ratio_by_day(
        work, min_valley_bars=1, min_ridge_bars=1, min_classifiable=1
    )
    # the background days are unclassifiable -> not valid -> absent entirely
    assert list(ratio.index) == [_TEST_DAY]
    assert ratio.loc[_TEST_DAY] == pytest.approx(_CASE_A_RATIO)


def test_ridge_bar_counts_by_day_are_exposed_for_the_coverage_disclosure():
    """The runner must be able to REPORT the ridge-bar distribution, not just gate on it."""
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("amount",)
        ).reset_index(drop=True)
    )
    ratio, diag = valley_ridge_vwap_ratio_by_day(
        work, min_valley_bars=1, min_ridge_bars=1, min_classifiable=1,
        with_diagnostics=True,
    )
    assert ratio.loc[_TEST_DAY] == pytest.approx(_CASE_A_RATIO)
    assert int(diag.loc[_TEST_DAY, "ridge_bars"]) == 4
    assert int(diag.loc[_TEST_DAY, "valley_bars"]) == 11
    assert bool(diag.loc[_TEST_DAY, "valid"]) is True
    # every classifiable day appears in the diagnostics, valid or not
    assert len(diag) >= 1


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_valley_ridge_vwap_ratio(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "valley_ridge_vwap_ratio"


def test_input_bars_not_mutated():
    bars = _bars(
        _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    )
    before = bars.copy(deep=True)
    compute_valley_ridge_vwap_ratio(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0] * 3, [1000.0] * 3))
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, baseline_days=1)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, baseline_min_obs=1)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, sigma_k=-0.5)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, min_valid_days=0)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, min_classifiable=0)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, min_valley_bars=0)
    with pytest.raises(ValueError):
        compute_valley_ridge_vwap_ratio(bars, min_ridge_bars=0)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = ValleyRidgeVwapRatioFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "valley_ridge_vwap_ratio_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == 1  # POSITIVE (report RankIC +6.98%)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert set(spec.input_fields) == {"volume", "amount"}
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_spec_discloses_the_asymmetric_ridge_gate():
    """The scarcity-driven >= 10 ridge floor is a PINNED choice a reader must see."""
    desc = ValleyRidgeVwapRatioFactor().spec.description
    assert "14:50" in desc
    assert "RAW" in desc.upper()
    assert str(VALLEY_RIDGE_MIN_RIDGE_BARS) in desc
    assert str(VALLEY_RIDGE_MIN_VALLEY_BARS) in desc
    assert "ridge" in desc.lower() and "valley" in desc.lower()


def test_factor_subclass_window_tracks_name():
    f = ValleyRidgeVwapRatioFactor(lookback_days=10)
    assert f.name == "valley_ridge_vwap_ratio_10"
    assert f.spec.factor_id == "valley_ridge_vwap_ratio_10"


def test_factor_compute_selects_preaggregated_column():
    f = ValleyRidgeVwapRatioFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.98]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.98
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = ValleyRidgeVwapRatioFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        ValleyRidgeVwapRatioFactor(lookback_days=0)
