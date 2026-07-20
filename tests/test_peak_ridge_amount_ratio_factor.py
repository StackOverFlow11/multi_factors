"""PR-M: PEAK/RIDGE AMOUNT-RATIO factor.

Same volume classification as PR-F / PR-H / PR-I / PR-J / PR-K / PR-L (REUSED, not
re-implemented). What is NEW here is twofold:

  1. The factor carries NO PRICE INFORMATION -- both legs are pure traded VALUE. The one
     way it can silently degenerate is by letting the wrong bar class into a leg, so every
     hand day deliberately contains VALLEY bars whose amounts are enormous and distinctive:
     they must never reach either leg.
  2. The AGGREGATION IS A RATIO OF 20-DAY SUMS, not the mean of daily ratios that PR-J
     used. The report specifies the two forms differently in adjacent sections (§7.2 vs
     §7.1), so ``test_factor_is_the_ratio_of_trailing_sums_not_the_mean_of_daily_ratios``
     pins the distinction with a two-day case whose two forms differ numerically.

DEFECT-INJECTION DISCIPLINE (the PR-L lesson, applied to EVERY invariance test).
A test of the shape "perturbing X leaves the factor unchanged" is worthless until the
defective implementation has been substituted and the test shown to FAIL against it --
PR-L shipped an anti-lookahead test that passed even under the buggy code because the
perturbation shape was invisible to the math. Each such test here therefore has a
``_defect_*`` context manager that monkeypatches the production seam into the specific bug
it guards, plus a companion ``test_*_has_teeth_*`` that RUNS the same assertions under the
defect and asserts they blow up. The teeth tests are part of the shipped suite, so the
guarantee cannot be quietly lost by someone "simplifying" a perturbation later.

Hand cases build a constant BACKGROUND of prior days so the same-slot baseline is exact
(mu=100, sigma=0 -> the eruptive threshold is exactly 100), then a test day whose volumes
place valleys / isolated peaks / ridge runs at chosen slots and whose AMOUNTS are set
independently, so both legs -- and therefore the ratio -- are known in closed form.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

import data.clean.intraday_amount_ratio as amount_ratio_mod
from data.clean.intraday_amount_ratio import (
    PEAK_RIDGE_LOOKBACK_DAYS,
    PEAK_RIDGE_MIN_PEAK_BARS,
    PEAK_RIDGE_MIN_RIDGE_BARS,
    compute_peak_ridge_amount_ratio,
    peak_ridge_amount_by_day,
)
from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import PeakRidgeAmountRatioFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, volume, amount), ...] -> normalized 1min bars.

    ``amount`` is set INDEPENDENTLY of ``volume``: volume drives the classification, amount
    is the factor's whole quantity. OHLC are dummy constants (this factor reads neither).
    ``normalize_intraday_bars`` sets ``available_time = bar_end + 1min``, so the 14:50 PIT
    cutoff excludes any bar with ``bar_end >= 14:50``.
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


# A VALLEY amount large enough that leaking even ONE valley bar into either leg would move
# the ratio by orders of magnitude. Used on every hand day.
_VALLEY_AMT = 500_000.0


def _background(n_days, n_slots, sym=_SYM, start_day="2021-07-01"):
    """``n_days`` prior days of flat volume-100 sessions.

    Same-slot baseline becomes mu=100, sigma=0 -> the eruptive threshold is exactly 100,
    so on a test day a volume of 100 is a VALLEY and anything above erupts.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, [_VALLEY_AMT] * n_slots, sym=sym)
    return rows


_BG_DAYS = 10
_TEST_DAY = pd.Timestamp("2021-07-11")
_DAY_1 = "2021-07-11"
_DAY_2 = "2021-07-12"

# Gates small enough that the engineered test day(s) are the only VALID days; the
# peak-bar / ridge-bar / valid-day floors get their own dedicated tests below.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=PEAK_RIDGE_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_peak_bars=1,
    min_ridge_bars=1,
)


# --------------------------------------------------------------------------- #
# Hand-computed amount ratios (2 non-trivial cases, closed-form fractions)
# --------------------------------------------------------------------------- #
# Both cases use 20 slots with the SAME geometry, so only the AMOUNTS differ:
#   ISOLATED PEAKS at slots 2, 5, 8 -- each flanked by valleys, so each is a true peak.
#   RIDGE RUNS at (11, 12) and (15, 16) -- adjacent eruptions, so none is isolated.
#   Every other slot is a VALLEY carrying the huge _VALLEY_AMT.
_N = 20
_PEAKS = (2, 5, 8)
_RIDGES = (11, 12, 15, 16)

# CASE A -- peak-heavy: peaks 1000 + 2000 + 3000 = 6000; ridges 500 + 700 + 800 + 1000 =
# 3000 -> ratio 2.0 (informed participation dominates).
_CASE_A_PEAK_AMTS = (1000.0, 2000.0, 3000.0)
_CASE_A_RIDGE_AMTS = (500.0, 700.0, 800.0, 1000.0)
_CASE_A_PEAK_SUM = 6000.0
_CASE_A_RIDGE_SUM = 3000.0
_CASE_A_RATIO = 2.0

# CASE B -- ridge-heavy: peaks 300 + 400 + 500 = 1200; ridges 1000 + 1100 + 1200 + 1300 =
# 4600 -> ratio 6/23 = 0.26087. The OPPOSITE side of 1.0 from case A, so an inverted
# numerator/denominator cannot pass both.
_CASE_B_PEAK_AMTS = (300.0, 400.0, 500.0)
_CASE_B_RIDGE_AMTS = (1000.0, 1100.0, 1200.0, 1300.0)
_CASE_B_PEAK_SUM = 1200.0
_CASE_B_RIDGE_SUM = 4600.0
_CASE_B_RATIO = 1200.0 / 4600.0


def _day(peak_amts, ridge_amts, *, peak_vol=300.0, ridge_vol=200.0):
    """Build the (vols, amts) of one hand day from its peak / ridge amounts."""
    vols = [100.0] * _N
    amts = [_VALLEY_AMT] * _N
    for slot, amt in zip(_PEAKS, peak_amts):
        vols[slot] = peak_vol
        amts[slot] = amt
    for slot, amt in zip(_RIDGES, ridge_amts):
        vols[slot] = ridge_vol
        amts[slot] = amt
    return vols, amts


def _case_a_day():
    return _day(_CASE_A_PEAK_AMTS, _CASE_A_RIDGE_AMTS)


def _case_b_day():
    return _day(_CASE_B_PEAK_AMTS, _CASE_B_RIDGE_AMTS)


def _one_day_bars(day=_DAY_1, case=_case_a_day):
    return _bars(_background(_BG_DAYS, _N) + _session(day, *case()))


def test_hand_value_case_a_ratio_above_one():
    out = compute_peak_ridge_amount_ratio(_one_day_bars(case=_case_a_day), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] > 1.0


def test_hand_value_case_b_ratio_below_one():
    out = compute_peak_ridge_amount_ratio(_one_day_bars(case=_case_b_day), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_B_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] < 1.0


def test_peak_is_the_numerator_and_ridge_the_denominator():
    """Scaling one leg must move the ratio in the DIRECTION that leg sits on.

    Doubling the peak amounts doubles the factor; doubling the ridge amounts halves it.
    A swapped numerator/denominator fails both assertions.
    """
    base = compute_peak_ridge_amount_ratio(_one_day_bars(), **_KW).loc[(_TEST_DAY, _SYM)]

    doubled_peaks = _bars(
        _background(_BG_DAYS, _N)
        + _session(
            _DAY_1,
            *_day([2 * a for a in _CASE_A_PEAK_AMTS], _CASE_A_RIDGE_AMTS),
        )
    )
    doubled_ridges = _bars(
        _background(_BG_DAYS, _N)
        + _session(
            _DAY_1,
            *_day(_CASE_A_PEAK_AMTS, [2 * a for a in _CASE_A_RIDGE_AMTS]),
        )
    )
    key = (_TEST_DAY, _SYM)
    assert compute_peak_ridge_amount_ratio(doubled_peaks, **_KW).loc[key] == pytest.approx(
        2.0 * base
    )
    assert compute_peak_ridge_amount_ratio(
        doubled_ridges, **_KW
    ).loc[key] == pytest.approx(base / 2.0)


def test_legs_by_day_match_the_hand_sums():
    """The per-day leg totals themselves, not just their ratio."""
    visible = prepare_visible_minute_bars(
        _one_day_bars(), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    legs = peak_ridge_amount_by_day(
        work, min_peak_bars=1, min_ridge_bars=1, min_classifiable=1
    )
    row = legs.loc[_TEST_DAY]
    assert row["peak_amt"] == pytest.approx(_CASE_A_PEAK_SUM)
    assert row["ridge_amt"] == pytest.approx(_CASE_A_RIDGE_SUM)


def test_ridge_run_bars_are_all_ridges_and_flanked_eruptions_are_all_peaks():
    """The hand geometry is what the REUSED classifier actually produces."""
    visible = prepare_visible_minute_bars(
        _one_day_bars(), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert list(np.flatnonzero(day["peak"].to_numpy())) == list(_PEAKS)
    assert list(np.flatnonzero(day["ridge"].to_numpy())) == list(_RIDGES)


# --------------------------------------------------------------------------- #
# THE AGGREGATION FORM: ratio of sums, NOT mean of daily ratios (report §7.2 vs §7.1)
# --------------------------------------------------------------------------- #
_TWO_DAY_RATIO_OF_SUMS = (_CASE_A_PEAK_SUM + _CASE_B_PEAK_SUM) / (
    _CASE_A_RIDGE_SUM + _CASE_B_RIDGE_SUM
)  # 7200 / 7600 = 18/19 = 0.947368...
_TWO_DAY_MEAN_OF_RATIOS = (_CASE_A_RATIO + _CASE_B_RATIO) / 2.0  # 1.130434...


def _two_day_bars():
    return _bars(
        _background(_BG_DAYS, _N)
        + _session(_DAY_1, *_case_a_day())
        + _session(_DAY_2, *_case_b_day())
    )


def test_factor_is_the_ratio_of_trailing_sums_not_the_mean_of_daily_ratios():
    """§7.2 says "20 日量峰总成交额与量岭总成交额，二者做比" -- pool first, divide once.

    The two candidate forms are numerically far apart on this pair of days (0.947 vs
    1.130) and sit on OPPOSITE sides of 1.0, so the assertion below cannot be satisfied by
    the mean-of-ratios form that PR-J's §7.1 wording specifies.
    """
    kw = dict(_KW, min_valid_days=2)
    out = compute_peak_ridge_amount_ratio(_two_day_bars(), **kw)
    value = out.loc[(pd.Timestamp(_DAY_2), _SYM)]
    assert value == pytest.approx(_TWO_DAY_RATIO_OF_SUMS)
    assert value != pytest.approx(_TWO_DAY_MEAN_OF_RATIOS)
    # And the two forms really are distinguishable here (guards the fixture itself).
    assert abs(_TWO_DAY_RATIO_OF_SUMS - _TWO_DAY_MEAN_OF_RATIOS) > 0.15


def test_busy_days_dominate_the_pool_which_is_what_a_ratio_of_sums_means():
    """A ratio of sums is AMOUNT-weighted: scaling one day's whole activity moves it.

    Multiplying day 1's BOTH legs by 10 leaves day 1's own daily ratio untouched (2.0), so
    a mean of daily ratios would not move at all. The ratio of sums must move, because day
    1 now carries ten times the weight in the pool.
    """
    kw = dict(_KW, min_valid_days=2)
    flat = compute_peak_ridge_amount_ratio(_two_day_bars(), **kw)
    scaled_rows = (
        _background(_BG_DAYS, _N)
        + _session(
            _DAY_1,
            *_day(
                [10 * a for a in _CASE_A_PEAK_AMTS],
                [10 * a for a in _CASE_A_RIDGE_AMTS],
            ),
        )
        + _session(_DAY_2, *_case_b_day())
    )
    scaled = compute_peak_ridge_amount_ratio(_bars(scaled_rows), **kw)
    key = (pd.Timestamp(_DAY_2), _SYM)
    expected = (10 * _CASE_A_PEAK_SUM + _CASE_B_PEAK_SUM) / (
        10 * _CASE_A_RIDGE_SUM + _CASE_B_RIDGE_SUM
    )
    assert scaled.loc[key] == pytest.approx(expected)
    assert scaled.loc[key] != pytest.approx(flat.loc[key])
    # Day 1's own daily ratio is unchanged at 2.0 -- read off a single-day window -- so a
    # MEAN of daily ratios would have been identical in both runs.
    solo = compute_peak_ridge_amount_ratio(
        _bars(scaled_rows), **dict(_KW, lookback_days=1, min_valid_days=1)
    )
    assert solo.loc[(pd.Timestamp(_DAY_1), _SYM)] == pytest.approx(_CASE_A_RATIO)


# --------------------------------------------------------------------------- #
# Validity gates: >=5 peak bars (PINNED, the scarce leg), >=10 ridge bars, classifiable
# --------------------------------------------------------------------------- #
def _geometry_day(n_peaks, n_ridges, n_slots=140):
    """A day with EXACTLY ``n_peaks`` isolated peaks and ``n_ridges`` ridge bars.

    Peaks are laid down first at stride 3, so each is flanked by valleys and is genuinely
    ISOLATED. Ridge RUNS follow well clear of them, separated by two valleys so adjacent
    runs cannot merge. A run of length 1 would itself be isolated and classify as a PEAK,
    so every run is >= 2 and an ODD ridge count is built from one run of 3 plus pairs.
    """
    if n_ridges == 1 or n_ridges < 0:
        raise ValueError(f"n_ridges must be 0 or >= 2; got {n_ridges}.")
    vols = [100.0] * n_slots
    amts = [_VALLEY_AMT] * n_slots
    for i in range(n_peaks):
        slot = 1 + 3 * i
        vols[slot] = 300.0
        amts[slot] = 1000.0

    runs = []
    remaining = n_ridges
    if remaining % 2 == 1:
        runs.append(3)
        remaining -= 3
    runs += [2] * (remaining // 2)

    cursor = 3 * n_peaks + 4
    for run in runs:
        for k in range(run):
            vols[cursor + k] = 200.0
            amts[cursor + k] = 500.0
        cursor += run + 2
    return vols, amts


def _geometry_counts(vols, amts, n_slots):
    """The realized (peak, ridge) bar counts of a geometry day, via the real classifier."""
    rows = _background(_BG_DAYS, n_slots) + _session(_DAY_1, vols, amts)
    visible = prepare_visible_minute_bars(
        _bars(rows), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    day = work[work["trade_date"] == _TEST_DAY]
    return int(day["peak"].sum()), int(day["ridge"].sum())


@pytest.mark.parametrize("n_peaks", [4, 5, 6])
def test_peak_bar_gate_boundary_at_five(n_peaks):
    """4 peak bars FAIL the pinned floor of 5; 5 and 6 PASS. The 4-vs-5 boundary."""
    n_slots = 140
    vols, amts = _geometry_day(n_peaks, 12, n_slots=n_slots)
    peaks, ridges = _geometry_counts(vols, amts, n_slots)
    assert peaks == n_peaks and ridges == 12  # the fixture is what it claims
    rows = _background(_BG_DAYS, n_slots) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(
        _bars(rows),
        **dict(
            _KW,
            min_peak_bars=PEAK_RIDGE_MIN_PEAK_BARS,
            min_ridge_bars=PEAK_RIDGE_MIN_RIDGE_BARS,
            min_classifiable=1,
        ),
    )
    key = (_TEST_DAY, _SYM)
    if n_peaks < PEAK_RIDGE_MIN_PEAK_BARS:
        assert key not in out.index or not np.isfinite(out.loc[key])
    else:
        assert np.isfinite(out.loc[key])


@pytest.mark.parametrize("n_ridges", [8, 9, 10, 11])
def test_ridge_bar_gate_boundary_at_ten(n_ridges):
    """9 ridge bars FAIL the floor of 10; 10 and 11 PASS."""
    n_slots = 140
    vols, amts = _geometry_day(6, n_ridges, n_slots=n_slots)
    peaks, ridges = _geometry_counts(vols, amts, n_slots)
    assert peaks == 6 and ridges == n_ridges
    rows = _background(_BG_DAYS, n_slots) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(
        _bars(rows),
        **dict(
            _KW,
            min_peak_bars=PEAK_RIDGE_MIN_PEAK_BARS,
            min_ridge_bars=PEAK_RIDGE_MIN_RIDGE_BARS,
            min_classifiable=1,
        ),
    )
    key = (_TEST_DAY, _SYM)
    if n_ridges < PEAK_RIDGE_MIN_RIDGE_BARS:
        assert key not in out.index or not np.isfinite(out.loc[key])
    else:
        assert np.isfinite(out.loc[key])


def test_bar_counts_are_taken_AFTER_the_positive_trade_guard():
    """A day with 5 peak bars, one of which traded nothing, has only 4 COUNTABLE peaks."""
    n_slots = 140
    vols, amts = _geometry_day(5, 12, n_slots=n_slots)
    peak_slots = [1 + 3 * i for i in range(5)]
    amts[peak_slots[0]] = 0.0  # traded nothing -> dropped by the guard AND uncounted
    rows = _background(_BG_DAYS, n_slots) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(
        _bars(rows),
        **dict(
            _KW,
            min_peak_bars=PEAK_RIDGE_MIN_PEAK_BARS,
            min_ridge_bars=PEAK_RIDGE_MIN_RIDGE_BARS,
            min_classifiable=1,
        ),
    )
    key = (_TEST_DAY, _SYM)
    assert key not in out.index or not np.isfinite(out.loc[key])


def test_day_with_no_ridge_is_invalid():
    vols = [100.0] * _N
    amts = [_VALLEY_AMT] * _N
    for slot, amt in zip(_PEAKS, _CASE_A_PEAK_AMTS):
        vols[slot] = 300.0
        amts[slot] = amt
    rows = _background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(_bars(rows), **_KW)
    assert (_TEST_DAY, _SYM) not in out.index


def test_day_with_no_peak_is_invalid():
    vols = [100.0] * _N
    amts = [_VALLEY_AMT] * _N
    for slot, amt in zip(_RIDGES, _CASE_A_RIDGE_AMTS):
        vols[slot] = 200.0
        amts[slot] = amt
    rows = _background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(_bars(rows), **_KW)
    assert (_TEST_DAY, _SYM) not in out.index


def test_min_classifiable_gate_invalidates_thin_days():
    out = compute_peak_ridge_amount_ratio(
        _one_day_bars(), **dict(_KW, min_classifiable=_N + 1)
    )
    assert (_TEST_DAY, _SYM) not in out.index


def test_min_valid_days_floor_returns_nan_until_enough_valid_days():
    out = compute_peak_ridge_amount_ratio(_two_day_bars(), **dict(_KW, min_valid_days=3))
    assert out.empty or not np.isfinite(out.to_numpy()).any()


def test_window_drops_days_older_than_lookback():
    """With lookback_days=1 only day d itself is pooled, so day 2 == its own ratio."""
    out = compute_peak_ridge_amount_ratio(
        _two_day_bars(), **dict(_KW, lookback_days=1, min_valid_days=1)
    )
    assert out.loc[(pd.Timestamp(_DAY_2), _SYM)] == pytest.approx(_CASE_B_RATIO)
    assert out.loc[(pd.Timestamp(_DAY_1), _SYM)] == pytest.approx(_CASE_A_RATIO)


def test_default_gates_match_the_pinned_definition():
    assert PEAK_RIDGE_LOOKBACK_DAYS == 20
    assert PEAK_RIDGE_MIN_PEAK_BARS == 5
    assert PEAK_RIDGE_MIN_RIDGE_BARS == 10
    # The PEAK floor is the LOWER one -- the REVERSE of PR-J's valley/ridge asymmetry.
    assert PEAK_RIDGE_MIN_PEAK_BARS < PEAK_RIDGE_MIN_RIDGE_BARS
    assert VOLUME_PRV_MIN_VALID_DAYS == 10
    assert VOLUME_PRV_MIN_CLASSIFIABLE == 100


# --------------------------------------------------------------------------- #
# The positive-trade guard: two-sided (dropped, AND would have mattered)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _defect_no_trade_guard(monkeypatch):
    """DEFECT: the positive-trade guard admits every bar."""
    monkeypatch.setattr(
        amount_ratio_mod,
        "_tradable_amount",
        lambda amt: np.ones(len(amt), dtype=bool),
    )
    yield


@pytest.mark.parametrize("bad", [0.0, -5000.0, float("nan"), float("inf")])
def test_non_positive_amount_is_dropped_from_the_peak_leg(bad):
    """A degenerate peak bar leaves the OTHER peaks' sum exactly intact."""
    vols, amts = _case_a_day()
    amts[_PEAKS[0]] = bad
    rows = _background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(_bars(rows), **_KW)
    expected = (_CASE_A_PEAK_SUM - _CASE_A_PEAK_AMTS[0]) / _CASE_A_RIDGE_SUM
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -5000.0, float("nan"), float("inf")])
def test_non_positive_amount_is_dropped_from_the_ridge_leg(bad):
    vols, amts = _case_a_day()
    amts[_RIDGES[0]] = bad
    rows = _background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts)
    out = compute_peak_ridge_amount_ratio(_bars(rows), **_KW)
    expected = _CASE_A_PEAK_SUM / (_CASE_A_RIDGE_SUM - _CASE_A_RIDGE_AMTS[0])
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(expected)


def test_guard_test_has_teeth_without_the_guard_the_value_changes(monkeypatch):
    """The counter-test: the dropped bar really would have mattered.

    Without this, the two tests above could pass because the bar was inert anyway. Under
    the defect the negative amount is summed into the peak leg and the ratio drops, so the
    guard is doing the work. The magnitude is chosen to keep the defective peak sum
    POSITIVE, so the day still clears the validity gate and the two values are directly
    comparable rather than one of them simply vanishing.
    """
    bad = -500.0
    vols, amts = _case_a_day()
    amts[_PEAKS[0]] = bad
    rows = _background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts)
    guarded = compute_peak_ridge_amount_ratio(_bars(rows), **_KW).loc[(_TEST_DAY, _SYM)]
    with _defect_no_trade_guard(monkeypatch):
        broken = compute_peak_ridge_amount_ratio(_bars(rows), **_KW).loc[
            (_TEST_DAY, _SYM)
        ]
    assert guarded == pytest.approx(
        (_CASE_A_PEAK_SUM - _CASE_A_PEAK_AMTS[0]) / _CASE_A_RIDGE_SUM
    )
    # The defect sums -500 in place of dropping it: 5000 -> 4500 in the numerator.
    assert broken == pytest.approx(
        (_CASE_A_PEAK_SUM - _CASE_A_PEAK_AMTS[0] + bad) / _CASE_A_RIDGE_SUM
    )
    assert broken != pytest.approx(guarded)


# --------------------------------------------------------------------------- #
# INVARIANCE 1 -- VALLEY bars reach NEITHER leg (this factor reads only eruptions)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _defect_valleys_in_the_peak_leg(monkeypatch):
    """DEFECT: the peak leg is ``peak | valley`` -- i.e. "everything that is not a ridge".

    A plausible mis-reading of the taxonomy, and the exact bug the invariance test below
    exists to catch.
    """
    real = amount_ratio_mod.peak_mask_for_symbol

    def broken(g, **kwargs):
        work = real(g, **kwargs)
        work = work.copy()
        work["peak"] = work["peak"] | work["valley"]
        return work

    monkeypatch.setattr(amount_ratio_mod, "peak_mask_for_symbol", broken)
    yield


def _valley_perturbed_bars():
    """Case A with EVERY valley bar's amount changed (peaks / ridges untouched)."""
    vols, amts = _case_a_day()
    for slot in range(_N):
        if slot not in _PEAKS and slot not in _RIDGES:
            amts[slot] = 7_777_777.0
    return _bars(_background(_BG_DAYS, _N) + _session(_DAY_1, vols, amts))


def test_perturbing_valley_amounts_does_not_change_the_factor():
    """VALLEY minutes are not part of either leg, so their traded value is irrelevant."""
    key = (_TEST_DAY, _SYM)
    clean = compute_peak_ridge_amount_ratio(_one_day_bars(), **_KW).loc[key]
    perturbed = compute_peak_ridge_amount_ratio(_valley_perturbed_bars(), **_KW).loc[key]
    assert clean == pytest.approx(_CASE_A_RATIO)
    assert perturbed == pytest.approx(clean)


def test_valley_invariance_has_teeth_under_the_valleys_in_leg_defect(monkeypatch):
    """The SAME assertions, run against the defective mask, must FAIL.

    This is the PR-L discipline: an invariance test is only evidence once the bug it
    claims to exclude has been shown to trip it.
    """
    key = (_TEST_DAY, _SYM)
    with _defect_valleys_in_the_peak_leg(monkeypatch):
        clean = compute_peak_ridge_amount_ratio(_one_day_bars(), **_KW).loc[key]
        perturbed = compute_peak_ridge_amount_ratio(_valley_perturbed_bars(), **_KW).loc[
            key
        ]
    assert clean != pytest.approx(_CASE_A_RATIO)
    assert perturbed != pytest.approx(clean)


# --------------------------------------------------------------------------- #
# INVARIANCE 2 -- PIT: post-14:50 bars cannot influence the value
# --------------------------------------------------------------------------- #
_LATE_START = "14:50:00"
_LATE_SLOTS = 8


@contextlib.contextmanager
def _defect_no_pit_cutoff(monkeypatch):
    """DEFECT: the 14:50 truncation is gone (the whole session is 'visible')."""
    real = amount_ratio_mod.prepare_visible_minute_bars

    def broken(bars, *, decision_time, extra_columns=()):
        return real(bars, decision_time="23:59:59", extra_columns=extra_columns)

    monkeypatch.setattr(amount_ratio_mod, "prepare_visible_minute_bars", broken)
    yield


def _two_block_background(n_days, early_slots, sym=_SYM, start_day="2021-07-01"):
    """Background covering BOTH the early block and the post-cutoff block.

    Giving the late slots a same-slot baseline is what puts TEETH in the leakage test: if
    the truncation were removed, the late bars would be fully CLASSIFIABLE and would really
    change the factor -- so "the perturbation changes nothing" is not passing merely
    because the late bars are unclassifiable noise.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * early_slots, [_VALLEY_AMT] * early_slots, sym=sym)
        rows += _session(
            day, [100.0] * _LATE_SLOTS, [_VALLEY_AMT] * _LATE_SLOTS,
            sym=sym, start=_LATE_START,
        )
    return rows


def _late_block(day, sym=_SYM):
    """A post-14:50 block carrying a huge RIDGE run (slots 0,1 and 4,5) and a peak at 3.

    The amounts are orders of magnitude above the early block's, so if these bars ever
    reached either leg the ratio would move enormously.
    """
    vols = [100.0] * _LATE_SLOTS
    amts = [_VALLEY_AMT] * _LATE_SLOTS
    for s in (0, 1, 4, 5):
        vols[s] = 200.0
        amts[s] = 900_000.0
    vols[3] = 300.0
    amts[3] = 400_000.0
    return _session(day, vols, amts, sym=sym, start=_LATE_START)


def _pit_pair():
    """(clean bars, bars + a post-14:50 block) sharing the same early session."""
    rows = _two_block_background(_BG_DAYS, _N) + _session(_DAY_1, *_case_a_day())
    return _bars(rows), _bars(rows + _late_block(_DAY_1))


def test_perturbing_post_1450_bars_does_not_change_the_factor():
    clean_bars, perturbed_bars = _pit_pair()
    key = (_TEST_DAY, _SYM)
    clean = compute_peak_ridge_amount_ratio(clean_bars, **_KW).loc[key]
    perturbed = compute_peak_ridge_amount_ratio(perturbed_bars, **_KW).loc[key]
    assert clean == pytest.approx(_CASE_A_RATIO)
    assert perturbed == pytest.approx(clean)


def test_pit_invariance_has_teeth_under_the_no_cutoff_defect(monkeypatch):
    """The SAME assertions, run with the 14:50 truncation removed, must FAIL."""
    clean_bars, perturbed_bars = _pit_pair()
    key = (_TEST_DAY, _SYM)
    with _defect_no_pit_cutoff(monkeypatch):
        clean = compute_peak_ridge_amount_ratio(clean_bars, **_KW).loc[key]
        perturbed = compute_peak_ridge_amount_ratio(perturbed_bars, **_KW).loc[key]
    # The late bars are inert only because they are HIDDEN; expose them and they dominate.
    assert perturbed != pytest.approx(clean)
    assert abs(perturbed - clean) / clean > 0.5


def test_public_decision_time_also_exposes_the_late_block():
    """A defect-free restatement of the same fact, via the public ``decision_time`` arg."""
    _, perturbed_bars = _pit_pair()
    key = (_TEST_DAY, _SYM)
    truncated = compute_peak_ridge_amount_ratio(perturbed_bars, **_KW).loc[key]
    untruncated = compute_peak_ridge_amount_ratio(
        perturbed_bars, decision_time="23:59:59", **_KW
    ).loc[key]
    assert truncated == pytest.approx(_CASE_A_RATIO)
    assert np.isfinite(untruncated)
    assert abs(untruncated - truncated) / truncated > 0.5


# --------------------------------------------------------------------------- #
# INVARIANCE 3 -- a FUTURE day cannot change an earlier value (trailing window)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _defect_forward_looking_window(monkeypatch):
    """DEFECT: the rolling window LEADS instead of trails (classic lookahead)."""

    def broken(legs, *, lookback_days, min_valid_days):
        ordered = legs.sort_index()
        roll = (
            ordered[::-1].rolling(lookback_days, min_periods=min_valid_days).sum()[::-1]
        )
        return roll["peak_amt"] / roll["ridge_amt"]

    monkeypatch.setattr(amount_ratio_mod, "_trailing_ratio_of_sums", broken)
    yield


def _future_pair():
    """(bars ending on day 1, the same bars plus a wildly different day 2)."""
    base = _background(_BG_DAYS, _N) + _session(_DAY_1, *_case_a_day())
    future = base + _session(_DAY_2, *_case_b_day())
    return _bars(base), _bars(future)


def test_future_day_does_not_change_the_earlier_factor():
    base_bars, future_bars = _future_pair()
    key = (_TEST_DAY, _SYM)
    a = compute_peak_ridge_amount_ratio(base_bars, **_KW).loc[key]
    b = compute_peak_ridge_amount_ratio(future_bars, **_KW).loc[key]
    assert a == pytest.approx(_CASE_A_RATIO)
    assert b == pytest.approx(a)


def test_future_day_invariance_has_teeth_under_the_forward_window_defect(monkeypatch):
    """The SAME assertions, run against a LEADING window, must FAIL."""
    base_bars, future_bars = _future_pair()
    key = (_TEST_DAY, _SYM)
    with _defect_forward_looking_window(monkeypatch):
        a = compute_peak_ridge_amount_ratio(base_bars, **_KW).loc[key]
        b = compute_peak_ridge_amount_ratio(future_bars, **_KW).loc[key]
    # Day 1's value now absorbs day 2 (case B), so adding day 2 moves it.
    assert b != pytest.approx(a)
    assert b == pytest.approx(_TWO_DAY_RATIO_OF_SUMS)


# --------------------------------------------------------------------------- #
# INVARIANCE 4 -- cross-symbol isolation
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _defect_symbol_frames_mislabeled(monkeypatch):
    """DEFECT: each symbol's LABEL is paired with a DIFFERENT symbol's bars.

    Note which bug this is and why. LITERAL pooling -- running the classifier over two
    symbols at once -- cannot happen silently: ``peak_mask_for_symbol`` unstacks by
    ``(trade_date, slot)`` and pandas raises "Index contains duplicate entries" the moment
    two symbols share a timestamp, which is a structural guarantee rather than a test
    result. The failure mode that CAN pass unnoticed is a mislabeled pairing in the split
    itself, so that is what the invariance test is made to face.
    """

    def broken(visible):
        frames = [
            (str(sym), g.reset_index(drop=True))
            for sym, g in visible.groupby("symbol", sort=True)
        ]
        rotated = frames[1:] + frames[:1]
        for (sym, _), (_, g) in zip(frames, rotated):
            yield sym, g

    monkeypatch.setattr(amount_ratio_mod, "_symbol_frames", broken)
    yield


def _two_symbol_bars():
    """AAA on case A, BBB on case B -- deliberately different ratios."""
    rows = _background(_BG_DAYS, _N, sym="AAA.SZ")
    rows += _session(_DAY_1, *_case_a_day(), sym="AAA.SZ")
    rows += _background(_BG_DAYS, _N, sym="BBB.SZ")
    rows += _session(_DAY_1, *_case_b_day(), sym="BBB.SZ")
    return _bars(rows)


def _solo_bars(sym, case):
    return _bars(
        _background(_BG_DAYS, _N, sym=sym) + _session(_DAY_1, *case(), sym=sym)
    )


def test_per_symbol_isolation():
    """Each symbol's value equals the value it has when computed entirely alone."""
    together = compute_peak_ridge_amount_ratio(_two_symbol_bars(), **_KW)
    solo_a = compute_peak_ridge_amount_ratio(_solo_bars("AAA.SZ", _case_a_day), **_KW)
    solo_b = compute_peak_ridge_amount_ratio(_solo_bars("BBB.SZ", _case_b_day), **_KW)
    assert together.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(
        solo_a.loc[(_TEST_DAY, "AAA.SZ")]
    )
    assert together.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(
        solo_b.loc[(_TEST_DAY, "BBB.SZ")]
    )
    assert together.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(_CASE_A_RATIO)
    assert together.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(_CASE_B_RATIO)


def test_isolation_has_teeth_under_the_mislabeled_frames_defect(monkeypatch):
    """The SAME assertions, run against a mislabeled split, must FAIL."""
    solo_a = compute_peak_ridge_amount_ratio(_solo_bars("AAA.SZ", _case_a_day), **_KW)
    solo_b = compute_peak_ridge_amount_ratio(_solo_bars("BBB.SZ", _case_b_day), **_KW)
    with _defect_symbol_frames_mislabeled(monkeypatch):
        together = compute_peak_ridge_amount_ratio(_two_symbol_bars(), **_KW)
    assert together.loc[(_TEST_DAY, "AAA.SZ")] != pytest.approx(
        solo_a.loc[(_TEST_DAY, "AAA.SZ")]
    )
    assert together.loc[(_TEST_DAY, "BBB.SZ")] != pytest.approx(
        solo_b.loc[(_TEST_DAY, "BBB.SZ")]
    )
    # Each name now carries the OTHER's ratio -- the swap is exact, not just noisy.
    assert together.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(_CASE_B_RATIO)
    assert together.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(_CASE_A_RATIO)


def test_literal_pooling_cannot_happen_silently(monkeypatch):
    """Records the structural guarantee the mislabel defect stands in for.

    Handing the shared classifier two symbols at once RAISES rather than returning a
    quietly-wrong frame, so cross-symbol contamination has no silent path at that layer.
    """

    def pooled(visible):
        frame = visible.reset_index(drop=True)
        for sym in sorted(set(visible["symbol"])):
            yield str(sym), frame.copy()

    monkeypatch.setattr(amount_ratio_mod, "_symbol_frames", pooled)
    with pytest.raises(ValueError, match="duplicate entries"):
        compute_peak_ridge_amount_ratio(_two_symbol_bars(), **_KW)


# --------------------------------------------------------------------------- #
# REUSE NON-DRIFT: PR-M adds nothing to the shared taxonomy
# --------------------------------------------------------------------------- #
def test_the_three_masks_still_partition_the_classifiable_bars():
    visible = prepare_visible_minute_bars(
        _one_day_bars(), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    valley = work["valley"].to_numpy(dtype=bool)
    peak = work["peak"].to_numpy(dtype=bool)
    ridge = work["ridge"].to_numpy(dtype=bool)
    classifiable = work["classifiable"].to_numpy(dtype=bool)
    assert np.array_equal(valley | peak | ridge, classifiable)
    assert not (valley & peak).any()
    assert not (valley & ridge).any()
    assert not (peak & ridge).any()


def test_peak_and_ridge_legs_use_the_shared_masks_verbatim():
    """The legs are exactly ``mask & tradable`` on the SHARED columns, nothing else."""
    visible = prepare_visible_minute_bars(
        _one_day_bars(), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    legs = peak_ridge_amount_by_day(
        work, min_peak_bars=1, min_ridge_bars=1, min_classifiable=1
    )
    amt = work["amount"].to_numpy(dtype=float)
    day = work["trade_date"].to_numpy() == np.datetime64(_TEST_DAY)
    tradable = np.isfinite(amt) & (amt > 0.0)
    expected_peak = amt[day & work["peak"].to_numpy(dtype=bool) & tradable].sum()
    expected_ridge = amt[day & work["ridge"].to_numpy(dtype=bool) & tradable].sum()
    assert legs.loc[_TEST_DAY, "peak_amt"] == pytest.approx(expected_peak)
    assert legs.loc[_TEST_DAY, "ridge_amt"] == pytest.approx(expected_ridge)


# --------------------------------------------------------------------------- #
# Diagnostics channel (the peak-scarcity disclosure)
# --------------------------------------------------------------------------- #
def test_diagnostics_expose_the_per_day_bar_counts():
    visible = prepare_visible_minute_bars(
        _one_day_bars(), decision_time="14:50:00", extra_columns=("amount",)
    )
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    legs, diag = peak_ridge_amount_by_day(
        work,
        min_peak_bars=1,
        min_ridge_bars=1,
        min_classifiable=1,
        with_diagnostics=True,
    )
    assert set(amount_ratio_mod.DIAGNOSTIC_COLUMNS) <= set(diag.columns)
    assert int(diag.loc[_TEST_DAY, "peak_bars"]) == len(_PEAKS)
    assert int(diag.loc[_TEST_DAY, "ridge_bars"]) == len(_RIDGES)
    assert bool(diag.loc[_TEST_DAY, "valid"])
    # Diagnostics cover EVERY day present, valid or not; the legs only the valid ones.
    assert len(diag) >= len(legs)


def test_diagnostics_out_does_not_change_the_factor():
    plain = compute_peak_ridge_amount_ratio(_one_day_bars(), **_KW)
    sink: list = []
    with_diag = compute_peak_ridge_amount_ratio(
        _one_day_bars(), diagnostics_out=sink, **_KW
    )
    pd.testing.assert_series_equal(plain, with_diag)
    assert sink and "symbol" in sink[0].columns


# --------------------------------------------------------------------------- #
# Edges / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_peak_ridge_amount_ratio(empty_intraday_bars(), name="f")
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "f"


def test_input_bars_not_mutated():
    bars = _one_day_bars()
    before = bars.copy(deep=True)
    compute_peak_ridge_amount_ratio(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_baseline_insufficient_yields_no_value():
    rows = _session(_DAY_1, *_case_a_day())
    out = compute_peak_ridge_amount_ratio(_bars(rows), **_KW)
    assert out.empty or not np.isfinite(out.to_numpy()).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lookback_days=0),
        dict(baseline_days=1),
        dict(baseline_min_obs=1),
        dict(sigma_k=-1.0),
        dict(min_valid_days=0),
        dict(min_classifiable=0),
        dict(min_peak_bars=0),
        dict(min_ridge_bars=0),
    ],
)
def test_bad_params_raise(kwargs):
    with pytest.raises(ValueError):
        compute_peak_ridge_amount_ratio(_one_day_bars(), **kwargs)


# --------------------------------------------------------------------------- #
# The Factor class + spec
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = PeakRidgeAmountRatioFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "peak_ridge_amount_ratio_20"
    assert spec.expected_ic_sign == 1
    assert spec.is_intraday is False
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert spec.family == "microstructure"
    assert set(spec.input_fields) == {"volume", "amount"}


def test_factor_spec_records_the_pre_registered_sign_source_and_aggregation():
    """The spec must state the report's RankIC and the ratio-of-sums form."""
    text = PeakRidgeAmountRatioFactor().spec.description
    assert "RATIO OF" in text and "SUMS" in text
    assert "峰岭成交比" in text
    assert str(PEAK_RIDGE_MIN_PEAK_BARS) in text
    assert str(PEAK_RIDGE_MIN_RIDGE_BARS) in text


def test_factor_subclass_window_tracks_name():
    assert PeakRidgeAmountRatioFactor(10).name == "peak_ridge_amount_ratio_10"
    assert PeakRidgeAmountRatioFactor(10).spec.factor_id == "peak_ridge_amount_ratio_10"


def test_factor_compute_selects_preaggregated_column():
    factor = PeakRidgeAmountRatioFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001.SZ")], names=["date", "symbol"]
    )
    panel = pd.DataFrame({factor.name: [1.25], "close": [10.0]}, index=idx)
    out = factor.compute(panel)
    assert out.name == factor.name
    assert out.iloc[0] == pytest.approx(1.25)


def test_factor_compute_missing_column_raises():
    factor = PeakRidgeAmountRatioFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001.SZ")], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        factor.compute(pd.DataFrame({"close": [10.0]}, index=idx))


@pytest.mark.parametrize("bad", [0, -1, 2.5, "20", None])
def test_factor_bad_params_raise(bad):
    with pytest.raises(ValueError):
        PeakRidgeAmountRatioFactor(bad)
