"""PR-I: VALLEY-RELATIVE VWAP factor.

Same volume classification as PR-F/PR-H (REUSED, not re-implemented), different
STATISTIC and different FAMILY: a PRICE level rather than a peak count / timing shape.
A "valley" (量谷) is a classifiable, NON-eruptive minute -- PR-F's internal ``mild``,
now exposed as a first-class ``valley`` column. The factor is the trailing-20-valid-day
mean of the daily ratio ``valley VWAP / whole-visible-day VWAP``. Sign is pre-registered
+1 (a high relative valley price = little downward over-reaction in calm minutes ->
higher forward return).

Hand cases build a constant BACKGROUND of prior days so the same-slot baseline is exact
(mu=100, sigma=0 -> the eruptive threshold is exactly 100), then a test day whose
volumes place valleys / eruptions at chosen slots and whose AMOUNTS are set
independently, so both VWAPs -- and therefore the ratio -- are known in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from data.clean.intraday_valley_vwap import (
    VALLEY_VWAP_LOOKBACK_DAYS,
    VALLEY_VWAP_MIN_VALLEY_BARS,
    compute_valley_relative_vwap,
    valley_vwap_ratio_by_day,
)
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.intraday_derived import ValleyRelativeVwapFactor
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, volume, amount), ...] -> normalized 1min bars.

    Unlike the PR-F / PR-H helpers, ``amount`` is set INDEPENDENTLY of ``volume`` --
    that is the whole point here: the per-bar price is ``amount / volume``. OHLC are
    dummy constants (this factor reads only volume + amount).
    ``normalize_intraday_bars`` sets ``available_time = bar_end + 1min``, so the 14:50
    PIT cutoff excludes any bar with ``bar_end >= 14:50``.
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

    Same-slot baseline becomes mu=100, sigma=0 -> the eruptive threshold is exactly
    100, so on the test day a volume of 100 is a VALLEY and anything above erupts.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [100.0] * n_slots, [1000.0] * n_slots, sym=sym)
    return rows


_BG_DAYS = 10
_TEST_DAY = pd.Timestamp("2021-07-11")

# Gates small enough that the single engineered test day is the only VALID day; the
# valley-bar and valid-day floors get their own dedicated tests below.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=VALLEY_VWAP_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
    min_valley_bars=1,
)


# --------------------------------------------------------------------------- #
# Hand-computed VWAP ratios (2 non-trivial cases, closed-form fractions)
# --------------------------------------------------------------------------- #
# CASE A -- 12 slots, eruptions at slots 3 and 7 (volume 200, amount 4000 -> price 20).
#   valley bars: 10 x volume 100; five carry amount 1000 (price 10), five amount 1200
#     (price 12)  -> valley VWAP = 11000 / 1000 = 11.0
#   whole visible day: sum(amount) = 11000 + 2*4000 = 19000
#                      sum(volume) = 1000 + 2*200  = 1400   -> day VWAP = 19000/1400
#   ratio = 11.0 * 1400 / 19000 = 15400/19000 = 77/95 = 0.810526...
# The eruptive minutes traded HIGHER and heavier, so the calm-minute price sits BELOW
# the day VWAP -> ratio < 1.
_CASE_A_N = 12
_CASE_A_ERUPT = (3, 7)
_CASE_A_RATIO = 77.0 / 95.0


def _case_a_day():
    vols = [100.0] * _CASE_A_N
    amts = [0.0] * _CASE_A_N
    valley_slots = [s for s in range(_CASE_A_N) if s not in _CASE_A_ERUPT]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 5 else 1200.0
    for s in _CASE_A_ERUPT:
        vols[s] = 200.0
        amts[s] = 4000.0
    return vols, amts


# CASE B -- 14 slots, eruptions at slots 2, 6, 10 (volume 300, amount 2400 -> price 8).
#   valley bars: 11 x volume 100; six carry amount 1000 (price 10), five amount 1500
#     (price 15) -> valley VWAP = 13500 / 1100
#   whole visible day: sum(amount) = 13500 + 3*2400 = 20700
#                      sum(volume) = 1100 + 3*300   = 2000  -> day VWAP = 10.35
#   ratio = (13500/1100) / 10.35 = 27000000/22770000 = 300/253 = 1.185770...
# Here the eruptions traded LOWER, so the calm-minute price sits ABOVE the day VWAP
# -> ratio > 1. Opposite direction from case A, so a sign/inversion bug cannot pass both.
_CASE_B_N = 14
_CASE_B_ERUPT = (2, 6, 10)
_CASE_B_RATIO = 300.0 / 253.0


def _case_b_day():
    vols = [100.0] * _CASE_B_N
    amts = [0.0] * _CASE_B_N
    valley_slots = [s for s in range(_CASE_B_N) if s not in _CASE_B_ERUPT]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 6 else 1500.0
    for s in _CASE_B_ERUPT:
        vols[s] = 300.0
        amts[s] = 2400.0
    return vols, amts


def test_hand_value_case_a_ratio_below_one():
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] < 1.0


def test_hand_value_case_b_ratio_above_one():
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_B_RATIO)
    assert out.loc[(_TEST_DAY, _SYM)] > 1.0


def test_vwap_uses_the_amount_over_volume_aggregation_identity():
    """sum(p_i * v_i) / sum(v_i) with p_i = amount_i/volume_i IS sum(amount)/sum(volume).

    The PINNED VWAP definition. Recomputed here the LONG way from per-bar prices, for
    BOTH legs (valley bars and the whole visible day), and checked against the closed-
    form ratio the factor returns -- so the identity is verified, not assumed.
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

    # whole visible day, the long way vs the aggregation identity
    day_long = float((price * v).sum() / v.sum())
    assert day_long == pytest.approx(float(a.sum() / v.sum()))
    assert day_long == pytest.approx(10.35)

    # valley leg, the long way vs the aggregation identity
    valley_long = float(
        (price[is_valley] * v[is_valley]).sum() / v[is_valley].sum()
    )
    assert valley_long == pytest.approx(
        float(a[is_valley].sum() / v[is_valley].sum())
    )
    assert valley_long == pytest.approx(13500.0 / 1100.0)

    # and the factor's ratio is exactly those two legs divided
    out = compute_valley_relative_vwap(bars, **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(valley_long / day_long)


def test_day_vwap_covers_the_whole_visible_day_not_only_valleys():
    """The denominator spans ALL visible bars, including the eruptive ones.

    Guard against the easy mistake of dividing the valley VWAP by itself (ratio == 1).
    """
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] != pytest.approx(1.0)
    # moving ONLY the eruptive bars' price must move the ratio (they are in the
    # denominator and nowhere else)
    vols2, amts2 = _case_a_day()
    for s in _CASE_A_ERUPT:
        amts2[s] = 8000.0  # price 40 instead of 20
    rows2 = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols2, amts2)
    out2 = compute_valley_relative_vwap(_bars(rows2), **_KW)
    assert out2.loc[(_TEST_DAY, _SYM)] < out.loc[(_TEST_DAY, _SYM)]


def test_day_with_no_eruption_has_ratio_exactly_one():
    """If NOTHING erupts, every bar is a valley and the two VWAPs are the same sum.

    An exact structural identity, useful as a scale check: the factor is centred on 1,
    and any deviation is attributable to the eruptive minutes alone.
    """
    n = 12
    amts = [1000.0, 1500.0, 800.0, 2000.0, 1200.0, 900.0] * 2  # prices vary a lot
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", [100.0] * n, amts)
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Non-positive volume / amount bars are DROPPED from the VWAP sums
# --------------------------------------------------------------------------- #
def _case_a_with_degenerate_bars():
    """Case A plus two extra slots (12, 13) that are CLASSIFIABLE VALLEYS but untradable.

    Slot 12 has volume 0 (amount 500); slot 13 has amount 0 (volume 50). Both are below
    the eruptive threshold, so the reused classifier calls them valleys -- they must be
    excluded by the POSITIVE-TRADE GUARD, not by the classification.
    """
    n = _CASE_A_N + 2
    vols, amts = _case_a_day()
    vols = vols + [0.0, 50.0]
    amts = amts + [500.0, 0.0]
    rows = _background(_BG_DAYS, n) + _session("2021-07-11", vols, amts)
    return rows


def test_degenerate_bars_are_classified_as_valleys_but_excluded_from_the_vwap():
    rows = _case_a_with_degenerate_bars()
    bars = _bars(rows)

    # they really ARE valleys under the reused classifier (so the guard is what drops
    # them -- this is not an accident of classification)
    visible = prepare_visible_minute_bars(bars, extra_columns=("amount",))
    work = peak_mask_for_symbol(visible.reset_index(drop=True))
    day = work[work["trade_date"] == _TEST_DAY].reset_index(drop=True)
    assert bool(day.loc[_CASE_A_N, "valley"]) is True      # volume 0 bar
    assert bool(day.loc[_CASE_A_N + 1, "valley"]) is True  # amount 0 bar

    # ...and the ratio is EXACTLY the clean case-A hand value: neither the numerator nor
    # the denominator was polluted.
    out = compute_valley_relative_vwap(bars, **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)


def test_negative_amount_bar_is_dropped_from_the_vwap():
    n = _CASE_A_N + 1
    vols, amts = _case_a_day()
    rows = _background(_BG_DAYS, n) + _session(
        "2021-07-11", vols + [50.0], amts + [-9_999.0]
    )
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)


def test_valley_bar_count_is_taken_AFTER_the_positive_trade_guard():
    """The >= min_valley_bars gate counts bars that actually CONTRIBUTE to the VWAP.

    Case A + 2 degenerate valleys = 12 classifiable valleys but only 10 tradable ones.
    A floor of 10 must pass; a floor of 11 must invalidate the day.
    """
    bars = _bars(_case_a_with_degenerate_bars())
    ok = compute_valley_relative_vwap(bars, **{**_KW, "min_valley_bars": 10})
    assert ok.loc[(_TEST_DAY, _SYM)] == pytest.approx(_CASE_A_RATIO)
    too_few = compute_valley_relative_vwap(bars, **{**_KW, "min_valley_bars": 11})
    assert too_few.dropna().empty


def test_day_with_no_tradable_volume_is_invalid():
    n = _CASE_A_N
    rows = _background(_BG_DAYS, n) + _session(
        "2021-07-11", [0.0] * n, [0.0] * n
    )
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.dropna().empty


def test_day_with_no_tradable_valley_is_invalid():
    # Every valley minute is untradable (volume 0); only the eruptive bars trade, so the
    # numerator has no support -> honest NaN rather than a 0/0 or a day-VWAP-only number.
    vols, amts = _case_a_day()
    for s in range(_CASE_A_N):
        if s not in _CASE_A_ERUPT:
            vols[s] = 0.0
            amts[s] = 0.0
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", vols, amts)
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.dropna().empty


# --------------------------------------------------------------------------- #
# Window mechanics: the trailing mean over VALID days
# --------------------------------------------------------------------------- #
def _two_case_days(n_slots=None):
    """Background + day1 = case A pattern, day2 = case B pattern, on a shared slot grid."""
    n = n_slots or _CASE_B_N
    a_vols, a_amts = _case_a_day()
    b_vols, b_amts = _case_b_day()
    # pad case A up to the shared grid with plain valley bars (volume 100 / amount 1000)
    pad = n - _CASE_A_N
    a_vols = a_vols + [100.0] * pad
    a_amts = a_amts + [1000.0] * pad
    rows = _background(_BG_DAYS, n)
    rows += _session("2021-07-11", a_vols, a_amts)
    rows += _session("2021-07-12", b_vols, b_amts)
    return rows


def test_factor_is_the_mean_of_the_trailing_valid_day_ratios():
    rows = _two_case_days()
    bars = _bars(rows)
    ratios = compute_valley_relative_vwap(bars, **{**_KW, "lookback_days": 1})
    d1, d2 = pd.Timestamp("2021-07-11"), pd.Timestamp("2021-07-12")
    r1 = ratios.loc[(d1, _SYM)]
    r2 = ratios.loc[(d2, _SYM)]
    assert r1 != pytest.approx(r2)  # the two engineered days really do differ

    # lookback 2 -> day 2 is the MEAN of the two daily ratios (not the last, not a sum)
    out = compute_valley_relative_vwap(bars, **{**_KW, "lookback_days": 2})
    assert out.loc[(d1, _SYM)] == pytest.approx(r1)
    assert out.loc[(d2, _SYM)] == pytest.approx((r1 + r2) / 2.0)


def test_window_drops_days_older_than_lookback():
    rows = _two_case_days()
    rows += _session("2021-07-13", *_case_b_day())
    bars = _bars(rows)
    per_day = compute_valley_relative_vwap(bars, **{**_KW, "lookback_days": 1})
    d1, d2, d3 = (pd.Timestamp(f"2021-07-1{k}") for k in (1, 2, 3))
    out = compute_valley_relative_vwap(bars, **{**_KW, "lookback_days": 2})
    # day 3 pools days 2+3 only -- day 1 has aged out of a 2-day window
    expected = (per_day.loc[(d2, _SYM)] + per_day.loc[(d3, _SYM)]) / 2.0
    assert out.loc[(d3, _SYM)] == pytest.approx(expected)
    three = compute_valley_relative_vwap(bars, **{**_KW, "lookback_days": 3})
    assert three.loc[(d3, _SYM)] != pytest.approx(expected)


def test_min_valid_days_floor_returns_nan_until_enough_valid_days():
    rows = _background(_BG_DAYS, _CASE_B_N)
    for k in range(3):
        d = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        rows += _session(d, *_case_b_day())
    out = compute_valley_relative_vwap(_bars(rows), **{**_KW, "min_valid_days": 3})
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-11"), _SYM)])
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-12"), _SYM)])
    assert np.isfinite(out.loc[(pd.Timestamp("2021-07-13"), _SYM)])


def test_min_classifiable_gate_invalidates_thin_days():
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    out = compute_valley_relative_vwap(_bars(rows), **{**_KW, "min_classifiable": 100})
    assert out.dropna().empty


def test_baseline_insufficient_yields_no_value():
    # Only 9 prior days -> fewer than baseline_min_obs (=10) same-slot observations ->
    # nothing classifiable -> no valley, no valid day, no value at all.
    rows = _background(9, _CASE_A_N) + _session("2021-07-10", *_case_a_day())
    out = compute_valley_relative_vwap(_bars(rows), **_KW)
    assert out.dropna().empty


def test_default_gates_match_the_pinned_definition():
    assert VALLEY_VWAP_LOOKBACK_DAYS == 20
    assert VALLEY_VWAP_MIN_VALLEY_BARS == 20
    assert VOLUME_PRV_MIN_VALID_DAYS == 10  # the < 10 valid days -> NaN floor


# --------------------------------------------------------------------------- #
# PIT: no lookahead, no post-cutoff influence
# --------------------------------------------------------------------------- #
def test_perturbing_post_1450_bars_does_not_change_factor():
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    a = compute_valley_relative_vwap(_bars(rows), **_KW)
    late = rows + [
        (pd.Timestamp("2021-07-11 14:50"), _SYM, 9_999.0, 9_999_999.0),
        (pd.Timestamp("2021-07-11 14:55"), _SYM, 1.0, 1.0),
        (pd.Timestamp("2021-07-11 14:56"), _SYM, 5_000.0, 1.0),
    ]
    b = compute_valley_relative_vwap(_bars(late), **_KW)
    key = (_TEST_DAY, _SYM)
    assert a.loc[key] == pytest.approx(_CASE_A_RATIO)
    assert a.loc[key] == pytest.approx(b.loc[key])


def test_future_day_does_not_change_earlier_factor():
    base = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    a = compute_valley_relative_vwap(_bars(base), **_KW)
    future = base + _session(
        "2021-07-12", [9_999.0] * _CASE_A_N, [1.0] * _CASE_A_N
    )
    b = compute_valley_relative_vwap(_bars(future), **_KW)
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
    out = compute_valley_relative_vwap(_bars(rows), **_KW)

    solo_a = compute_valley_relative_vwap(
        _bars(
            _background(_BG_DAYS, n, sym="AAA.SZ")
            + _session("2021-07-11", a_vols, a_amts, sym="AAA.SZ")
        ),
        **_KW,
    )
    assert out.loc[(_TEST_DAY, "AAA.SZ")] == pytest.approx(solo_a.loc[(_TEST_DAY, "AAA.SZ")])
    assert out.loc[(_TEST_DAY, "BBB.SZ")] == pytest.approx(_CASE_B_RATIO)


# --------------------------------------------------------------------------- #
# §0 REUSE NON-DRIFT: the exposure refactor changed nothing for PR-F / PR-H
# --------------------------------------------------------------------------- #
def test_prepare_visible_minute_bars_default_output_is_unchanged():
    """No ``extra_columns`` -> EXACTLY the historical column list, in order.

    This is the lock on the additive exposure: PR-F / PR-H call this helper with no
    extra columns, so their input frame must be byte-identical to before.
    """
    rows = _background(3, 5)
    out = prepare_visible_minute_bars(_bars(rows))
    assert list(out.columns) == [
        "symbol", "bar_end", "available_time", "volume", "trade_date", "slot"
    ]
    assert "amount" not in out.columns


def test_extra_columns_only_add_and_never_alter_the_shared_columns():
    rows = _background(3, 5) + _session("2021-07-04", *_case_a_day())
    bars = _bars(rows)
    plain = prepare_visible_minute_bars(bars)
    with_amount = prepare_visible_minute_bars(bars, extra_columns=("amount",))
    # extras ride with the other RAW columns, ahead of the derived trade_date / slot
    assert list(with_amount.columns) == [
        "symbol", "bar_end", "available_time", "volume", "amount", "trade_date", "slot"
    ]
    # ...and every shared column is untouched, row for row
    pd.testing.assert_frame_equal(with_amount[plain.columns], plain)


def test_extra_columns_rejects_unknown_and_duplicate_names():
    bars = _bars(_background(2, 3))
    with pytest.raises(ValueError, match="not present"):
        prepare_visible_minute_bars(bars, extra_columns=("nope",))
    with pytest.raises(ValueError, match="already"):
        prepare_visible_minute_bars(bars, extra_columns=("volume",))


def test_peak_and_classifiable_are_unaffected_by_carrying_amount():
    """peak / classifiable must not depend on whether ``amount`` rides along.

    Together with the two factor-level locks below, this is the numeric statement of
    "the exposure refactor is behaviour-preserving".
    """
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    bars = _bars(rows)
    plain = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars).reset_index(drop=True)
    )
    carried = peak_mask_for_symbol(
        prepare_visible_minute_bars(bars, extra_columns=("amount",)).reset_index(drop=True)
    )
    for col in ("trade_date", "bar_end", "slot", "volume", "classifiable", "peak"):
        pd.testing.assert_series_equal(carried[col], plain[col], check_names=False)


def test_valley_is_exactly_classifiable_and_not_eruptive():
    """valley == PR-F's internal ``mild`` == classifiable & ~eruptive, and never a peak."""
    vols, amts = _case_b_day()
    rows = _background(_BG_DAYS, _CASE_B_N) + _session("2021-07-11", vols, amts)
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(_bars(rows)).reset_index(drop=True)
    )
    classifiable = work["classifiable"].to_numpy(dtype=bool)
    valley = work["valley"].to_numpy(dtype=bool)
    peak = work["peak"].to_numpy(dtype=bool)
    # reconstruct "eruptive" independently from the raw threshold rule
    eruptive = classifiable & (
        work["volume"].to_numpy(dtype=float) > work["thr"].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(valley, classifiable & ~eruptive)
    # a peak is an ERUPTIVE minute, so peak and valley are disjoint
    assert not (valley & peak).any()
    # the engineered day really does contain both kinds (the assertion is not vacuous)
    day = work["trade_date"].to_numpy() == np.datetime64(_TEST_DAY)
    assert valley[day].any() and eruptive[day].any()


def test_volume_peak_count_unchanged_by_the_exposure_refactor():
    """PR-F's factor value is bit-identical whether or not amount is carried."""
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
    # three isolated eruptions with mild neighbours -> exactly three peaks
    assert out.loc[(_TEST_DAY, _SYM)] == pytest.approx(3.0)


def test_peak_interval_kurtosis_unchanged_by_the_exposure_refactor():
    """PR-H's factor value is bit-identical whether or not amount is carried."""
    from data.clean.intraday_peak_interval import compute_peak_interval_kurtosis

    # the PR-H hand case: peaks at 1, 3, 6, 10, 15, 21, 23 -> intervals [2,3,4,5,6,2]
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


# --------------------------------------------------------------------------- #
# The per-day ratio helper (exposed for the reuse / diagnostics path)
# --------------------------------------------------------------------------- #
def test_valley_vwap_ratio_by_day_returns_only_valid_days():
    rows = _background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day())
    work = peak_mask_for_symbol(
        prepare_visible_minute_bars(
            _bars(rows), extra_columns=("amount",)
        ).reset_index(drop=True)
    )
    ratio = valley_vwap_ratio_by_day(work, min_valley_bars=1, min_classifiable=1)
    # the background days are unclassifiable -> not valid -> absent entirely
    assert list(ratio.index) == [_TEST_DAY]
    assert ratio.loc[_TEST_DAY] == pytest.approx(_CASE_A_RATIO)


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_valley_relative_vwap(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "valley_relative_vwap"


def test_input_bars_not_mutated():
    bars = _bars(_background(_BG_DAYS, _CASE_A_N) + _session("2021-07-11", *_case_a_day()))
    before = bars.copy(deep=True)
    compute_valley_relative_vwap(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0] * 3, [1000.0] * 3))
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, baseline_days=1)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, baseline_min_obs=1)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, sigma_k=-0.5)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, min_valid_days=0)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, min_classifiable=0)
    with pytest.raises(ValueError):
        compute_valley_relative_vwap(bars, min_valley_bars=0)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = ValleyRelativeVwapFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "valley_relative_vwap_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == 1  # POSITIVE (report RankIC +8.69%)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert set(spec.input_fields) == {"volume", "amount"}
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_spec_discloses_pinned_interpretations():
    # The choices the report is silent on (or where we knowingly deviate) MUST be on the
    # spec a reader sees: the visible-window day VWAP deviation and the raw-price note.
    desc = ValleyRelativeVwapFactor().spec.description
    assert "14:50" in desc
    assert "RAW" in desc.upper()
    assert "amount" in desc and "volume" in desc


def test_factor_subclass_window_tracks_name():
    f = ValleyRelativeVwapFactor(lookback_days=10)
    assert f.name == "valley_relative_vwap_10"
    assert f.spec.factor_id == "valley_relative_vwap_10"


def test_factor_compute_selects_preaggregated_column():
    f = ValleyRelativeVwapFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.98]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.98
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = ValleyRelativeVwapFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        ValleyRelativeVwapFactor(lookback_days=0)
