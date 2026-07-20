"""PR-K: RIDGE MINUTE-RETURN factor.

Same volume classification as PR-F / PR-H / PR-I / PR-J (REUSED verbatim, not
re-implemented): a "ridge" (量岭) is an ERUPTIVE minute that is NOT an isolated peak. What
is NEW here is the STATISTIC -- a RETURN rather than a count, a timing moment or a price
level. The factor is the trailing-20-valid-day SUM of each day's summed ridge-minute
returns. Sign is pre-registered -1 (the report's only negative peak/ridge/valley factor).

Hand cases build a constant BACKGROUND of prior days so the same-slot baseline is exact
(mu=100, sigma=0 -> the eruptive threshold is exactly 100), then a test day whose VOLUMES
place ridge runs / an isolated peak at chosen slots and whose CLOSES are set
INDEPENDENTLY, so every minute return -- and therefore the daily sum -- is known in closed
form. The hand closes move in halvings and doublings, so each return is exactly
representable in binary and the expected sums are exact rather than approximate.

Two properties get deliberately aggressive test data:

  * the SUM-vs-COMPOUNDING pin. At minute scale the two conventions are numerically
    indistinguishable, so a test built on realistic returns could not tell them apart. The
    hand days therefore use returns of +/-50%..+300%, where sum and Pi(1+r)-1 differ by
    more than a factor of four, and the test asserts the SUM.
  * the LEAKAGE test has TEETH: the post-14:50 block is shown to move the value materially
    the moment the truncation is disabled, so "perturbation changes nothing" cannot pass
    because the late bars were inert.

Every hand day also contains an isolated PEAK whose minute return is wildly larger than
any ridge's: it must never leak into the sum, which is the one way this factor can
silently degenerate into "eruptive-minute return".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_ridge_return import (
    RIDGE_RETURN_LOOKBACK_DAYS,
    RIDGE_RETURN_MIN_RIDGE_BARS,
    compute_ridge_minute_return,
    ridge_minute_return_by_day,
)
from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import RidgeMinuteReturnFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, close, volume), ...] -> normalized 1min bars.

    ``close`` is set INDEPENDENTLY of ``volume`` -- that is the point: the volume decides
    WHICH minutes are ridges and the close decides WHAT they returned. ``amount`` rides
    along as ``close * volume`` (this factor never reads it, but a realistic frame keeps
    the reuse locks below honest). ``normalize_intraday_bars`` sets
    ``available_time = bar_end + 1min``, so the 14:50 PIT cutoff excludes any bar with
    ``bar_end >= 14:50``.
    """
    close = [float(r[2]) for r in rows]
    volume = [float(r[3]) for r in rows]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "amount": [c * v for c, v in zip(close, volume)],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, closes, vols, sym=_SYM, start="09:31:00"):
    """One session of CONSECUTIVE 1-minute bars carrying ``closes`` / ``vols``."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, c, v)
        for i, (c, v) in enumerate(zip(closes, vols))
    ]


def _background(n_days, n_slots, sym=_SYM, start_day="2021-07-01"):
    """``n_days`` prior days of flat close-100 / volume-100 sessions.

    Same-slot baseline becomes mu=100, sigma=0 -> the eruptive threshold is exactly 100,
    so on the test day a volume of 100 is a VALLEY and anything above erupts. The flat
    closes also mean every background day's ridge-return sum is trivially zero -- and,
    having no eruption at all, no background day is ever VALID.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, [100.0] * n_slots, sym=sym)
    return rows


_BG_DAYS = 10
_TEST_DAY = pd.Timestamp("2021-07-11")

# Gates small enough that the single engineered test day is the only VALID day; the
# ridge-bar and valid-day floors get their own dedicated tests below.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=RIDGE_RETURN_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_ridge_bars=1,
)


# --------------------------------------------------------------------------- #
# Hand-computed daily sums (2 non-trivial cases, exact binary fractions)
# --------------------------------------------------------------------------- #
# CASE A -- 16 slots, POSITIVE daily sum.
#   ridge runs at (3,4) and (8,9): each bar volume 200. Every one of the four has an
#     ERUPTIVE neighbour, so none is an isolated peak -> all four are ridges.
#   closes: 100,100,100, 150, 75, 75,75,75, 300, 150, 150,150, 1500, 150,150,150
#     -> r3 = 150/100-1 = +0.5   (ridge)
#        r4 =  75/150-1 = -0.5   (ridge)
#        r8 = 300/75 -1 = +3.0   (ridge)
#        r9 = 150/300-1 = -0.5   (ridge)
#     s_day = 0.5 - 0.5 + 3.0 - 0.5 = +2.5     <- SIMPLE SUM (the pinned convention)
#     Pi(1+r)-1 = 1.5*0.5*4.0*0.5 - 1 = -0.5   <- compounding would give something else
#   an ISOLATED eruption at slot 12 (volume 300) IS a peak; its return is +9.0, so if it
#     leaked into the sum the value would be 11.5 instead of 2.5 (the discriminator).
_CASE_A_N = 16
_CASE_A_RIDGES = (3, 4, 8, 9)
_CASE_A_PEAK = 12
_CASE_A_CLOSES = [
    100.0, 100.0, 100.0, 150.0, 75.0, 75.0, 75.0, 75.0,
    300.0, 150.0, 150.0, 150.0, 1500.0, 150.0, 150.0, 150.0,
]
_CASE_A_SUM = 2.5
_CASE_A_COMPOUNDED = 1.5 * 0.5 * 4.0 * 0.5 - 1.0  # == -0.5
_CASE_A_PEAK_RETURN = 9.0


def _case_a_day():
    vols = [100.0] * _CASE_A_N
    for s in _CASE_A_RIDGES:
        vols[s] = 200.0
    vols[_CASE_A_PEAK] = 300.0
    return list(_CASE_A_CLOSES), vols


# CASE B -- 18 slots, NEGATIVE daily sum (so a sign / absolute-value bug cannot pass both).
#   ridge runs at (2,3), (7,8), (13,14); isolated peak at slot 16.
#   closes halve at each ridge except the last, which doubles:
#     r2=r3=r7=r8=r13=-0.5, r14=+1.0  -> s_day = -2.5 + 1.0 = -1.5
#     Pi(1+r)-1 = 0.5^5 * 2 - 1 = -0.9375   <- again distinguishable from the sum
#   the peak at slot 16 returns +9.0 and must be excluded.
_CASE_B_N = 18
_CASE_B_RIDGES = (2, 3, 7, 8, 13, 14)
_CASE_B_PEAK = 16
_CASE_B_CLOSES = [
    100.0, 100.0, 50.0, 25.0, 25.0, 25.0, 25.0, 12.5,
    6.25, 6.25, 6.25, 6.25, 6.25, 3.125, 6.25, 6.25, 62.5, 6.25,
]
_CASE_B_SUM = -1.5
_CASE_B_COMPOUNDED = 0.5**5 * 2.0 - 1.0  # == -0.9375


def _case_b_day():
    vols = [100.0] * _CASE_B_N
    for s in _CASE_B_RIDGES:
        vols[s] = 300.0
    vols[_CASE_B_PEAK] = 500.0
    return list(_CASE_B_CLOSES), vols


def test_hand_value_case_a_positive_sum():
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_SUM)
    assert out.loc[(_TEST_DAY, _SYM)] > 0.0


def test_hand_value_case_b_negative_sum():
    closes, vols = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_B_SUM)
    assert out.loc[(_TEST_DAY, _SYM)] < 0.0


def test_daily_aggregation_is_a_simple_sum_not_a_compounded_return():
    """THE pinned-convention discriminator: Sum(r), NOT Pi(1+r)-1.

    The report says only "累计收益" (cumulative return). Ridge minutes are NON-CONTIGUOUS
    within the day, so a holding-period/compounding reading does not apply, and the PINNED
    choice is the simple sum. At minute scale the two are numerically indistinguishable,
    so this test uses hand days with +/-50%..+300% returns where they differ by more than
    a factor of four -- if the implementation ever compounded, both assertions below fail.
    """
    for closes_vols, expected_sum, compounded in (
        (_case_a_day(), _CASE_A_SUM, _CASE_A_COMPOUNDED),
        (_case_b_day(), _CASE_B_SUM, _CASE_B_COMPOUNDED),
    ):
        closes, vols = closes_vols
        rows = _background(_BG_DAYS, len(closes)) + _session("2021-07-11", closes, vols)
        out = compute_ridge_minute_return(_bars(rows), **_KW)
        got = out.loc[(_TEST_DAY, _SYM)]
        # the two conventions really are far apart on this data (not a vacuous check)
        assert abs(expected_sum - compounded) > 0.4
        assert got == pytest.approx(expected_sum)
        assert got != pytest.approx(compounded)


def test_isolated_peak_return_is_excluded_from_the_sum():
    """The sum runs over ``eruptive & ~peak``, NOT over all eruptive bars.

    Moving ONLY the isolated peak's close must leave the factor untouched; if the
    implementation summed every eruptive bar the value would move. This is the single most
    important discriminator in the file.
    """
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    base = compute_ridge_minute_return(_bars(rows), **_KW)

    closes2, vols2 = _case_a_day()
    closes2[_CASE_A_PEAK] = 15_000.0  # peak return +99.0 instead of +9.0
    rows2 = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes2, vols2)
    moved = compute_ridge_minute_return(_bars(rows2), **_KW)

    key = (_TEST_DAY, _SYM)
    assert base.loc[key] == pytest.approx(_CASE_A_SUM)
    assert moved.loc[key] == pytest.approx(_CASE_A_SUM)

    # ...and the day really does contain an isolated peak carrying a big return, so the
    # assertion above is not vacuous
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("close",)
        ).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert bool(day.loc[_CASE_A_PEAK, "peak"]) is True
    assert bool(day.loc[_CASE_A_PEAK, "ridge"]) is False
    peak_ret = (
        day.loc[_CASE_A_PEAK, "close"] / day.loc[_CASE_A_PEAK - 1, "close"] - 1.0
    )
    assert peak_ret == pytest.approx(_CASE_A_PEAK_RETURN)


def test_ridge_set_on_the_hand_day_is_exactly_the_adjacent_eruption_runs():
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    ridge_slots = tuple(day.index[day["ridge"].to_numpy(dtype=bool)])
    assert ridge_slots == _CASE_A_RIDGES


# --------------------------------------------------------------------------- #
# The WITHIN-DAY lag (PINNED): first visible bar of each day has no return
# --------------------------------------------------------------------------- #
def test_first_visible_bar_of_the_day_never_carries_a_return():
    """A ridge at the day's first visible bar contributes NOTHING (no previous close).

    PR-F's rule makes a session-boundary eruption a RIDGE (its isolation is unprovable),
    so this is a real case: slot 0 is a ridge but has no within-day predecessor, and using
    the PREVIOUS DAY's close would be a cross-day gap the definition excludes.
    """
    n = 12
    closes = [100.0] * n
    vols = [100.0] * n
    vols[0] = 200.0  # first visible bar erupts -> ridge (no previous neighbour)
    vols[1] = 200.0  # its neighbour erupts too -> also a ridge, and this one HAS a return
    closes[1] = 150.0  # r1 = +0.5
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", closes, vols)
    bars = _bars(rows)

    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert bool(day.loc[0, "ridge"]) is True  # slot 0 IS a ridge...
    assert bool(day.loc[1, "ridge"]) is True

    out = compute_ridge_minute_return(bars, **_KW)
    # ...but only slot 1's +0.5 reaches the sum
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.5)


def test_returns_never_cross_a_day_boundary():
    """Day 2's first bar does not return against day 1's close.

    Day 2 opens at half of day 1's closing level; if the lag crossed the boundary the
    first bar would contribute -0.5 to day 2's sum.
    """
    n = 12
    day1_closes = [100.0] * n
    day2_closes = [50.0] * n
    vols = [100.0] * n
    vols[0] = vols[1] = 200.0  # both ridges on day 2; slot 1's return is exactly 0
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", day1_closes, [100.0] * n)
    rows += _session("2021-07-12", day2_closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **{**_KW, "lookback_days": 1})
    # day 2's only return-carrying ridge is slot 1 (50/50-1 = 0) -> sum exactly 0
    assert out.loc[(pd.Timestamp("2021-07-12"), _SYM)] == pytest.approx(0.0)


def test_non_positive_or_missing_previous_close_is_guarded():
    """A zero / NaN close kills its own return AND the next bar's, never an inf."""
    n = 14
    closes = [100.0] * n
    vols = [100.0] * n
    # a ridge run at (5,6) whose PREDECESSOR (slot 4) closed at zero
    closes[4] = 0.0
    closes[5] = 200.0
    closes[6] = 300.0
    vols[5] = vols[6] = 200.0
    # a second, clean ridge run at (9,10) so the day still has a finite contribution
    closes[9] = 150.0
    closes[10] = 300.0
    vols[9] = vols[10] = 200.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **_KW)
    got = out.loc[(_TEST_DAY, _SYM)]
    assert np.isfinite(got)
    # slot 5 is dropped (prev close 0); slot 6 = 300/200-1 = +0.5;
    # slot 9 = 150/100-1 = +0.5; slot 10 = 300/150-1 = +1.0  -> 2.0
    assert got == pytest.approx(2.0)


def test_within_day_gap_uses_the_previous_visible_bar_of_the_same_day():
    """PINNED, and locked so it can never drift silently.

    The lag is over the previous VISIBLE bar of the SAME day, with no exact-60s adjacency
    requirement. A bar that opens a new session block (the real one being 13:01, after the
    lunch break) therefore returns against the last bar before the gap -- a genuine price
    change of the stock over the interval, not a fabricated one. The alternative (dropping
    gap-spanning bars) would silently discard the post-lunch minute whenever it is a
    ridge. Note the same-slot baseline defuses the obvious selection worry: the 13:01 slot
    is compared against its OWN history, so a systematically busy post-lunch minute is not
    systematically eruptive.
    """
    early, late = 10, 8
    late_start = "13:01:00"
    rows = []
    for i in range(_BG_DAYS):
        day = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * early, [100.0] * early)
        rows += _session(day, [100.0] * late, [100.0] * late, start=late_start)
    # test day: flat morning, then the FIRST post-gap bar erupts together with its
    # neighbour -> both ridges; the post-gap bar returns against the morning's last close
    rows += _session("2021-07-11", [100.0] * early, [100.0] * early)
    late_closes = [120.0, 120.0] + [120.0] * (late - 2)
    late_vols = [200.0, 200.0] + [100.0] * (late - 2)
    rows += _session("2021-07-11", late_closes, late_vols, start=late_start)

    out = compute_ridge_minute_return(_bars(rows), **_KW)
    # first post-gap bar: 120/100 - 1 = +0.2 (spans the gap); its neighbour: 120/120-1 = 0
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.2)


def test_scaling_a_days_closes_leaves_the_sum_unchanged():
    """The RAW-CLOSE pin: a within-day constant adjustment factor cancels in the ratio.

    Multiplying every close of the day by a constant (what a split/dividend adjustment
    factor does, since it is constant WITHIN a day) leaves every minute return -- and
    therefore the daily sum -- exactly unchanged. This is why unadjusted closes are
    correct here, and it is a property, not an assumption.
    """
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    plain = compute_ridge_minute_return(_bars(rows), **_KW)

    scaled_closes = [c * 3.0 for c in closes]
    rows2 = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", scaled_closes, vols)
    scaled = compute_ridge_minute_return(_bars(rows2), **_KW)

    key = (_TEST_DAY, _SYM)
    assert plain.loc[key] == pytest.approx(_CASE_A_SUM)
    assert scaled.loc[key] == pytest.approx(plain.loc[key])


# --------------------------------------------------------------------------- #
# Ridge scarcity: the >= 10 return-carrying-ridge-bar gate
# --------------------------------------------------------------------------- #
_GATE_N = 24


def _one_percent_grid(n_slots):
    """Closes on which EVERY bar returns exactly +1%, so a sum reads off as a count."""
    return [100.0 * (1.01**i) for i in range(n_slots)]


def _ridge_runs_day(run_lengths, n_slots, first_slot=1):
    """A day whose eruptive bars form CONSECUTIVE runs of the given lengths.

    Runs are separated by 2 mild bars, so every eruptive bar has an eruptive neighbour and
    a run of length L contributes exactly L ridge bars (no isolated peaks at all). Closes
    are the +1% grid, so the day's sum is 0.01 * (number of RETURN-CARRYING ridge bars).
    """
    vols = [100.0] * n_slots
    slot = first_slot
    for length in run_lengths:
        if slot + length > n_slots:
            raise AssertionError("test day too short for the requested runs")
        for k in range(length):
            vols[slot + k] = 200.0
        slot += length + 2
    return _one_percent_grid(n_slots), vols


def _ridge_and_peak_count(closes, vols, n_slots):
    rows = _background(_BG_DAYS, n_slots) + _session("2021-07-11", closes, vols)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    day = work[work["trade_date"] == _TEST_DAY]
    return int(day["ridge"].sum()), int(day["peak"].sum())


def test_exactly_nine_return_carrying_ridge_bars_fails_the_default_ten_gate():
    closes, vols = _ridge_runs_day([2, 2, 2, 3], _GATE_N)  # 9 ridges
    assert _ridge_and_peak_count(closes, vols, _GATE_N) == (9, 0)
    rows = _background(_BG_DAYS, _GATE_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(
        _bars(rows), **{**_KW, "min_ridge_bars": RIDGE_RETURN_MIN_RIDGE_BARS}
    )
    assert out.dropna().empty


def test_exactly_ten_return_carrying_ridge_bars_passes_the_default_ten_gate():
    closes, vols = _ridge_runs_day([2, 2, 2, 2, 2], _GATE_N)  # 10 ridges
    assert _ridge_and_peak_count(closes, vols, _GATE_N) == (10, 0)
    rows = _background(_BG_DAYS, _GATE_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(
        _bars(rows), **{**_KW, "min_ridge_bars": RIDGE_RETURN_MIN_RIDGE_BARS}
    )
    # every bar returns +1%, and all ten ridges carry a return -> 0.10
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.10)


def test_ridge_count_is_taken_AFTER_the_return_guard():
    """The gate counts ridges that actually CONTRIBUTE a return.

    Ten classifiable ridge bars, but one of them is the day's FIRST visible bar and so has
    no return -- only nine support the sum, and the default gate must invalidate the day.
    """
    closes, vols = _ridge_runs_day([2, 2, 2, 2, 2], _GATE_N, first_slot=0)
    assert _ridge_and_peak_count(closes, vols, _GATE_N) == (10, 0)
    rows = _background(_BG_DAYS, _GATE_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(
        _bars(rows), **{**_KW, "min_ridge_bars": RIDGE_RETURN_MIN_RIDGE_BARS}
    )
    assert out.dropna().empty
    # ...and a floor of 9 accepts the day, proving it was the COUNT that blocked it
    ok = compute_ridge_minute_return(_bars(rows), **{**_KW, "min_ridge_bars": 9})
    assert ok.loc[(_TEST_DAY, _SYM)] == pytest.approx(0.09)


def test_day_with_no_ridge_is_invalid():
    """Only isolated peaks erupt -> nothing to sum -> honest absence, never a fake 0."""
    n = 12
    closes = _one_percent_grid(n)
    vols = [100.0] * n
    for s in (3, 7):  # isolated -> peaks, not ridges
        vols[s] = 200.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **_KW)
    assert out.dropna().empty


def test_min_classifiable_gate_invalidates_thin_days():
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **{**_KW, "min_classifiable": 100})
    assert out.dropna().empty


def test_baseline_insufficient_yields_no_value():
    closes, vols = _case_a_day()
    rows = _background(9, _CASE_A_N) + _session("2021-07-10", closes, vols)
    out = compute_ridge_minute_return(_bars(rows), **_KW)
    assert out.dropna().empty


def test_default_gates_match_the_pinned_definition():
    assert RIDGE_RETURN_LOOKBACK_DAYS == 20
    # Same scarcity floor PR-J pinned for its ridge leg: a bar must erupt AND fail the
    # isolation test, so ridges are structurally far rarer than valleys.
    assert RIDGE_RETURN_MIN_RIDGE_BARS == 10
    assert VOLUME_PRV_MIN_VALID_DAYS == 10


# --------------------------------------------------------------------------- #
# Window mechanics: the trailing SUM across VALID days
# --------------------------------------------------------------------------- #
def _two_case_days():
    """Background + day1 = case A pattern, day2 = case B pattern, on a shared grid."""
    n = _CASE_B_N
    a_closes, a_vols = _case_a_day()
    pad = n - _CASE_A_N
    a_closes = a_closes + [a_closes[-1]] * pad
    a_vols = a_vols + [100.0] * pad
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", a_closes, a_vols)
    rows += _session("2021-07-12", *_case_b_day())
    return rows


def test_factor_is_the_SUM_across_trailing_valid_days_not_the_mean():
    """The cross-day aggregation is a SUM (PR-J averaged; this factor accumulates)."""
    bars = _bars(_two_case_days())
    per_day = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 1})
    d1, d2 = pd.Timestamp("2021-07-11"), pd.Timestamp("2021-07-12")
    s1, s2 = per_day.loc[(d1, _SYM)], per_day.loc[(d2, _SYM)]
    assert s1 == pytest.approx(_CASE_A_SUM)
    assert s2 == pytest.approx(_CASE_B_SUM)

    out = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 2})
    assert out.loc[(d1, _SYM)] == pytest.approx(s1)
    assert out.loc[(d2, _SYM)] == pytest.approx(s1 + s2)
    # explicitly NOT the mean (the two differ here: 1.0 vs 0.5)
    assert out.loc[(d2, _SYM)] != pytest.approx((s1 + s2) / 2.0)


def test_window_drops_days_older_than_lookback():
    rows = _two_case_days() + _session("2021-07-13", *_case_b_day())
    bars = _bars(rows)
    per_day = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 1})
    d1, d2, d3 = (pd.Timestamp(f"2021-07-1{k}") for k in (1, 2, 3))
    out = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 2})
    expected = per_day.loc[(d2, _SYM)] + per_day.loc[(d3, _SYM)]
    assert out.loc[(d3, _SYM)] == pytest.approx(expected)
    three = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 3})
    assert three.loc[(d3, _SYM)] != pytest.approx(expected)
    assert np.isfinite(per_day.loc[(d1, _SYM)])


def test_invalid_days_do_not_occupy_a_slot_in_the_trailing_window():
    """A day that fails a gate is ABSENT, not zero -- it must not shorten the window."""
    n = _CASE_B_N
    a_closes, a_vols = _case_a_day()
    a_closes = a_closes + [a_closes[-1]] * (n - _CASE_A_N)
    a_vols = a_vols + [100.0] * (n - _CASE_A_N)
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", a_closes, a_vols)
    # a day with NO eruption at all -> no ridge -> invalid, contributes nothing
    rows += _session("2021-07-12", [100.0] * n, [100.0] * n)
    rows += _session("2021-07-13", *_case_b_day())
    bars = _bars(rows)
    out = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 2})
    per_day = compute_ridge_minute_return(bars, **{**_KW, "lookback_days": 1})
    d1, d2, d3 = (pd.Timestamp(f"2021-07-1{k}") for k in (1, 2, 3))
    assert (d2, _SYM) not in per_day.index  # the flat day never produced a value
    # day 3's 2-day window pools day 1 and day 3, skipping the invalid day 2
    expected = per_day.loc[(d1, _SYM)] + per_day.loc[(d3, _SYM)]
    assert out.loc[(d3, _SYM)] == pytest.approx(expected)


def test_min_valid_days_floor_returns_nan_until_enough_valid_days():
    rows = _background(_BG_DAYS, _CASE_B_N)
    for k in range(3):
        d = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        rows += _session(d, *_case_b_day())
    out = compute_ridge_minute_return(_bars(rows), **{**_KW, "min_valid_days": 3})
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-11"), _SYM)])
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-12"), _SYM)])
    assert out.loc[(pd.Timestamp("2021-07-13"), _SYM)] == pytest.approx(3 * _CASE_B_SUM)


# --------------------------------------------------------------------------- #
# PIT: no lookahead, no post-cutoff influence -- WITH TEETH
# --------------------------------------------------------------------------- #
_LATE_START = "14:50:00"
_LATE_SLOTS = 6


def _two_block_background(n_days, early_slots, sym=_SYM, start_day="2021-07-01"):
    """Background whose sessions cover BOTH the early block and the post-cutoff block.

    Giving the late slots a same-slot baseline is what puts TEETH in the leakage test: if
    the 14:50 truncation were removed, the late bars would be fully classifiable and would
    really change the factor -- so the "perturbation changes nothing" assertion is not
    passing merely because the late bars are unclassifiable noise.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * early_slots, [100.0] * early_slots, sym=sym)
        rows += _session(
            day, [100.0] * _LATE_SLOTS, [100.0] * _LATE_SLOTS,
            sym=sym, start=_LATE_START,
        )
    return rows


def _late_block(day, sym=_SYM):
    """A post-14:50 block with a ridge run whose returns are huge and NEGATIVE.

    Four consecutive eruptive minutes, each closing 90% below the previous bar, so if the
    truncation were lifted they would add 4 * (-0.9) = -3.6 to case A's +2.5 and drive the
    factor NEGATIVE. The first of them returns against the early block's last close, since
    the within-day lag spans the gap (PINNED §2).
    """
    closes = [15.0, 1.5, 0.15, 0.015, 0.015, 0.015]
    vols = [500.0, 500.0, 500.0, 500.0, 100.0, 100.0]
    return _session(day, closes, vols, sym=sym, start=_LATE_START)


def test_perturbing_post_1450_bars_does_not_change_factor():
    rows = _two_block_background(_BG_DAYS, _CASE_A_N)
    rows += _session("2021-07-11", *_case_a_day())
    clean = compute_ridge_minute_return(_bars(rows), **_KW)
    perturbed = compute_ridge_minute_return(
        _bars(rows + _late_block("2021-07-11")), **_KW
    )
    key = (_TEST_DAY, _SYM)
    assert clean.loc[key] == pytest.approx(_CASE_A_SUM)
    assert perturbed.loc[key] == pytest.approx(clean.loc[key])


def test_leakage_test_has_teeth_removing_the_cutoff_changes_the_value():
    """The counter-test: with the 14:50 truncation DISABLED the value moves materially.

    Without this, the assertion above could pass for the wrong reason (post-cutoff bars
    that are inert anyway). Here the same bars are shown to carry a large, real effect the
    moment the cutoff stops hiding them -- so the truncation is doing the work.
    """
    rows = _two_block_background(_BG_DAYS, _CASE_A_N)
    rows += _session("2021-07-11", *_case_a_day())
    rows += _late_block("2021-07-11")
    bars = _bars(rows)

    truncated = compute_ridge_minute_return(bars, **_KW)
    untruncated = compute_ridge_minute_return(bars, decision_time="23:59:59", **_KW)
    key = (_TEST_DAY, _SYM)
    a, b = truncated.loc[key], untruncated.loc[key]
    assert a == pytest.approx(_CASE_A_SUM)
    assert np.isfinite(b)
    # the late block adds two ridge bars, the second of which crashes 90% -> the value
    # must move by far more than any rounding tolerance, and flip sign
    assert abs(b - a) / abs(a) > 0.5
    assert b < 0.0 < a


def test_future_day_does_not_change_earlier_factor():
    base = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    a = compute_ridge_minute_return(_bars(base), **_KW)
    future = base + _session(
        "2021-07-12", [9_999.0] * _CASE_A_N, [9_999.0] * _CASE_A_N
    )
    b = compute_ridge_minute_return(_bars(future), **_KW)
    key = (_TEST_DAY, _SYM)
    assert np.isfinite(a.loc[key])
    assert a.loc[key] == pytest.approx(b.loc[key])


# --------------------------------------------------------------------------- #
# Per-symbol isolation
# --------------------------------------------------------------------------- #
def test_per_symbol_isolation():
    n = _CASE_B_N
    a_closes, a_vols = _case_a_day()
    a_closes = a_closes + [a_closes[-1]] * (n - _CASE_A_N)
    a_vols = a_vols + [100.0] * (n - _CASE_A_N)

    rows = _background(_BG_DAYS, n, sym="AAA.SZ")
    rows += _session("2021-07-11", a_closes, a_vols, sym="AAA.SZ")
    rows += _background(_BG_DAYS, n, sym="BBB.SZ")
    rows += _session("2021-07-11", *_case_b_day(), sym="BBB.SZ")
    out = compute_ridge_minute_return(_bars(rows), **_KW)

    solo_a = compute_ridge_minute_return(
        _bars(
            _background(_BG_DAYS, n, sym="AAA.SZ")
            + _session("2021-07-11", a_closes, a_vols, sym="AAA.SZ")
        ),
        **_KW,
    )
    assert out.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(
        solo_a.loc[(_TEST_DAY, "AAA.SZ")]
    )
    assert out.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(_CASE_B_SUM)


# --------------------------------------------------------------------------- #
# §0 REUSE NON-DRIFT: the four MERGED factors are bit-identical
# --------------------------------------------------------------------------- #
# PR-K adds no column to and changes no line of data/clean/intraday_volume_prv.py, so the
# taxonomy the four merged factors consume is untouched. These locks assert that as a
# VALUE, not as a claim: each reproduces its own module's documented hand value.
def _prv_kw(**over):
    kw = dict(
        baseline_days=VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
        sigma_k=VOLUME_PRV_SIGMA_K,
        lookback_days=20,
        min_valid_days=1,
        min_classifiable=1,
    )
    kw.update(over)
    return kw


def test_volume_peak_count_unchanged_by_pr_k():
    """PR-F's factor value is bit-identical (the PR-J lock's hand case, verbatim)."""
    from data.clean.intraday_volume_prv import compute_volume_peak_count

    closes, vols = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", closes, vols)
    out = compute_volume_peak_count(_bars(rows), **_prv_kw(lookback_days=1))
    # case B has exactly ONE isolated eruption (slot 16); the three adjacent runs are
    # ridges, not peaks
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(1.0)


def test_peak_interval_kurtosis_unchanged_by_pr_k():
    """PR-H's factor value is bit-identical (the PR-H hand case, verbatim)."""
    from data.clean.intraday_peak_interval import compute_peak_interval_kurtosis

    n = 25
    vols = [100.0] * n
    for p in (1, 3, 6, 10, 15, 21, 23):
        vols[p] = 200.0
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", [100.0] * n, vols)
    out = compute_peak_interval_kurtosis(_bars(rows), **_prv_kw(min_intervals=4))
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(-1.48125)


def test_valley_relative_vwap_unchanged_by_pr_k():
    """PR-I's factor value is bit-identical (the PR-I case-A hand value, verbatim).

    This one needs AMOUNTS set independently of the closes, so it builds its own frame
    rather than going through ``_bars`` (whose amount is close * volume).
    """
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
    bars = _amount_bars(n, vols, amts)
    out = compute_valley_relative_vwap(bars, **_prv_kw(min_valley_bars=1))
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(77.0 / 95.0)


def test_valley_ridge_vwap_ratio_unchanged_by_pr_k():
    """PR-J's factor value is bit-identical (the PR-J case-A hand value, verbatim)."""
    from data.clean.intraday_valley_ridge_vwap import compute_valley_ridge_vwap_ratio

    n = 16
    ridges, peak = (3, 4, 8, 9), 12
    vols = [100.0] * n
    amts = [0.0] * n
    for s in ridges:
        vols[s] = 200.0
        amts[s] = 4000.0
    vols[peak] = 300.0
    amts[peak] = 9000.0
    valley_slots = [s for s in range(n) if s not in ridges and s != peak]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 6 else 1200.0
    bars = _amount_bars(n, vols, amts)
    out = compute_valley_ridge_vwap_ratio(
        bars, **_prv_kw(min_valley_bars=1, min_ridge_bars=1)
    )
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(6.0 / 11.0)


def _amount_bars(n_slots, vols, amts, sym=_SYM):
    """Bars whose AMOUNT is independent of the close (for the PR-I / PR-J locks)."""
    rows = []
    for i in range(_BG_DAYS):
        day = pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)
        base = day + pd.Timedelta("09:31:00")
        for s in range(n_slots):
            rows.append((base + pd.Timedelta(minutes=s), sym, 100.0, 1000.0))
    base = _TEST_DAY + pd.Timedelta("09:31:00")
    for s in range(n_slots):
        rows.append((base + pd.Timedelta(minutes=s), sym, vols[s], amts[s]))
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


def test_the_three_masks_still_partition_the_classifiable_bars():
    """valley | peak | ridge == classifiable, pairwise disjoint (structural, unchanged)."""
    closes, vols = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", closes, vols)
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
    day = work["trade_date"].to_numpy() == np.datetime64(_TEST_DAY)
    assert ridge[day].any() and peak[day].any()  # not vacuous


def test_carrying_close_through_does_not_disturb_the_masks():
    """``extra_columns=("close",)`` only rides along; the taxonomy is untouched."""
    closes, vols = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", closes, vols)
    bars = _bars(rows)
    plain = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars).reset_index(drop=True)
    )
    carried = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars, extra_columns=("close",)).reset_index(drop=True)
    )
    for col in ("trade_date", "bar_end", "slot", "volume", "classifiable", "valley",
                "peak", "ridge"):
        pd.testing.assert_series_equal(carried[col], plain[col], check_names=False)


# --------------------------------------------------------------------------- #
# The per-day helper (exposed for the reuse / diagnostics path)
# --------------------------------------------------------------------------- #
def _work_for_case_a():
    closes, vols = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", closes, vols)
    return peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("close",)
        ).reset_index(drop=True)
    )


def test_ridge_minute_return_by_day_returns_only_valid_days():
    daily = ridge_minute_return_by_day(
        _work_for_case_a(), min_ridge_bars=1, min_classifiable=1
    )
    # the background days are unclassifiable -> not valid -> absent entirely
    assert list(daily.index) == [_TEST_DAY]
    assert daily.loc[_TEST_DAY] == pytest.approx(_CASE_A_SUM)


def test_ridge_bar_counts_by_day_are_exposed_for_the_coverage_disclosure():
    """The runner must be able to REPORT the ridge distribution, not just gate on it."""
    daily, diag = ridge_minute_return_by_day(
        _work_for_case_a(), min_ridge_bars=1, min_classifiable=1, with_diagnostics=True
    )
    assert daily.loc[_TEST_DAY] == pytest.approx(_CASE_A_SUM)
    assert int(diag.loc[_TEST_DAY, "ridge_bars"]) == 4
    # all four ridges sit past the day's first bar, so all four carry a return
    assert int(diag.loc[_TEST_DAY, "ridge_return_bars"]) == 4
    assert bool(diag.loc[_TEST_DAY, "valid"]) is True
    # every day present in the frame appears in the diagnostics, valid or not
    assert len(diag) == _BG_DAYS + 1


def test_diagnostics_separate_ridge_bars_from_return_carrying_ridge_bars():
    """The two counts must differ when a ridge opens the day (the gate uses the latter)."""
    closes, vols = _ridge_runs_day([2, 2, 2, 2, 2], _GATE_N, first_slot=0)
    rows = _background(_BG_DAYS, _GATE_N) + _session("2021-07-11", closes, vols)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("close",)
        ).reset_index(drop=True)
    )
    _, diag = ridge_minute_return_by_day(
        work, min_ridge_bars=10, min_classifiable=1, with_diagnostics=True
    )
    assert int(diag.loc[_TEST_DAY, "ridge_bars"]) == 10
    assert int(diag.loc[_TEST_DAY, "ridge_return_bars"]) == 9
    assert bool(diag.loc[_TEST_DAY, "valid"]) is False


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_ridge_minute_return(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "ridge_minute_return"


def test_input_bars_not_mutated():
    bars = _bars(
        _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    )
    before = bars.copy(deep=True)
    compute_ridge_minute_return(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0] * 3, [100.0] * 3))
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, baseline_days=1)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, baseline_min_obs=1)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, sigma_k=-0.5)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, min_valid_days=0)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, min_classifiable=0)
    with pytest.raises(ValueError):
        compute_ridge_minute_return(bars, min_ridge_bars=0)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = RidgeMinuteReturnFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "ridge_minute_return_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == -1  # NEGATIVE (report RankIC -6.29%)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert set(spec.input_fields) == {"volume", "close"}
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_spec_discloses_the_pinned_choices():
    """A reader must see the sum-vs-compounding pin, the raw closes and the 14:50 cut."""
    desc = RidgeMinuteReturnFactor().spec.description
    assert "14:50" in desc
    assert "RAW" in desc.upper()
    assert "sum" in desc.lower() and "compound" in desc.lower()
    assert str(RIDGE_RETURN_MIN_RIDGE_BARS) in desc
    assert "ridge" in desc.lower()


def test_factor_subclass_window_tracks_name():
    f = RidgeMinuteReturnFactor(lookback_days=10)
    assert f.name == "ridge_minute_return_10"
    assert f.spec.factor_id == "ridge_minute_return_10"


def test_factor_compute_selects_preaggregated_column():
    f = RidgeMinuteReturnFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [-0.031]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == -0.031
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = RidgeMinuteReturnFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        RidgeMinuteReturnFactor(lookback_days=0)
