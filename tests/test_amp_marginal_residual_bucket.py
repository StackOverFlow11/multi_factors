"""PR-E residual-bucket correctness guard: a 4-minute bar is not a 5min bar.

``amp_marginal_anomaly_vol`` is the ONE minute factor of the eleven that DERIVES
coarser (5min) bars from the 1min cache. ``resample_intraday_bars`` buckets by
``ceil(bar_end, freq)`` and reports each bucket's REAL span, so a bucket that runs
out of 1min bars is emitted as a SHORTER bar (a residual bucket) rather than
dropped. Every statistic this factor computes is bar-length sensitive
(``amp = high/low - 1``, ``r = close_t/close_{t-1} - 1``) and they are POOLED over
20 trading days, so a short bar shifts the pool's mean / std and the selected-bar
return std SYSTEMATICALLY.

That made the factor's value depend on the CALLER'S LOADING GEOMETRY:

* the legacy runner handed in WHOLE-DAY 1min bars, so the 14:46..14:50 bucket was
  complete and was then removed by the availability rule (its ``available_time``
  is 14:51 > the 14:50 cutoff) — the last visible bar of each day was the complete
  14:45 bucket;
* the D4 materializer PRE-TRUNCATES the 1min bars to ``available_time <= 14:50``
  (i.e. through the 14:49 bar), so the same bucket held four constituents, ended
  at 14:49 and passed the availability rule — every single day gained a fifth
  "5min bar" that was really four minutes long.

The fix keeps only the derived bars whose window reached its ``freq`` grid
boundary. These tests pin the three directions of that rule and, in the
whole-day / gapless case, that the filter is a NO-OP — the legacy path's values
do not move.

Definition note: the rule is about the WINDOW CLOSING, not about counting five
constituents. A bucket missing some of its minutes still closes on the grid as
long as the grid-boundary minute is present, and is kept;
``test_bucket_with_a_leading_gap_is_kept_because_its_window_still_closed`` pins
that so a future "constituent count == 5" reading cannot creep in silently.

The most visible consequence is the OPENING AUCTION: ``ceil(09:30, 5min)`` is
09:30, so that minute forms a bucket of its own with exactly ONE constituent and
is KEPT. That is one 1-minute bar pooled as a "5min bar" every trading day, in
BOTH geometries and in the frozen baseline alike. It is deliberately left alone
here -- see ``_complete_grid_bars``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import resample_intraday_bars
from data.clean.intraday_schema import normalize_intraday_bars
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AMP_ANOMALY_FREQ,
    _complete_grid_bars,
    compute_amp_marginal_anomaly_vol,
)
from factors.view_lag import minute_decision_cutoff

_SYM = "000905.SZ"
#: small gates so a compact fixture still produces a finite value; the DEFAULT
#: gates (460 / 20) are exercised by tests/test_amp_marginal_anomaly_vol_factor.py.
_GATES = {"lookback_days": 20, "min_pool": 6, "min_selected": 2}


def _rows_to_bars(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(bar_end, symbol, high, low, close), ...] -> normalized 1min bars."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": [r[4] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [100.0] * len(rows),
            "amount": [100.0] * len(rows),
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _tail_session(day: str, *, first: str = "13:31", last: str = "15:00") -> list[tuple]:
    """One session's 1min bars over ``[first, last]``, minute by minute.

    Amplitudes and closes vary bar to bar (a deterministic saw pattern seeded by
    the day) so the pooled ``|Δamp|`` has real spread and the anomaly selection is
    not degenerate; the exact numbers do not matter, only that they differ.
    """
    start = pd.Timestamp(f"{day} {first}")
    end = pd.Timestamp(f"{day} {last}")
    n = int((end - start) / pd.Timedelta(minutes=1)) + 1
    seed = int(pd.Timestamp(day).dayofyear)
    rows = []
    for i in range(n):
        amp = 0.002 + 0.001 * ((i * 7 + seed) % 11)
        close = 100.0 + 0.1 * ((i * 5 + seed) % 13)
        rows.append(
            (start + pd.Timedelta(minutes=i), _SYM, 100.0 * (1.0 + amp), 100.0, close)
        )
    return rows


def _whole_day_bars(days: list[str]) -> pd.DataFrame:
    rows: list[tuple] = []
    for day in days:
        rows.extend(_tail_session(day))
    return _rows_to_bars(rows)


def _visible_coarse_bar_ends(bars: pd.DataFrame, day: str) -> list[pd.Timestamp]:
    """The 5min ``bar_end``s the factor actually pools on ``day``, in order.

    Mirrors the factor's own front half (derive -> completeness filter ->
    availability rule) so a test can name the last bar of the day.
    """
    coarse = _complete_grid_bars(resample_intraday_bars(bars, AMP_ANOMALY_FREQ), "5min")
    cut = minute_decision_cutoff(coarse, decision_time="14:50:00")
    ends = pd.DatetimeIndex(cut["bar_end"])
    return sorted(ends[ends.normalize() == pd.Timestamp(day)])


_DAYS = ["2021-07-01", "2021-07-02", "2021-07-05"]


# --------------------------------------------------------------------------- #
# Direction 1 — the legacy (whole-day, gapless) path does not move
# --------------------------------------------------------------------------- #
def test_whole_day_gapless_bars_leave_the_completeness_filter_a_noop():
    """On whole-day gapless input EVERY derived bucket already closes on the grid.

    This is the bit-identity claim for the legacy runner path stated as a property
    of the data rather than as a diff against deleted code: if the filter removes
    nothing, it cannot have changed any legacy value.
    """
    bars = _whole_day_bars(_DAYS)
    coarse = resample_intraday_bars(bars, AMP_ANOMALY_FREQ)
    kept = _complete_grid_bars(coarse, "5min")
    # PRECONDITION: the frame is non-trivial, so "nothing was dropped" is not
    # vacuously true of an empty frame.
    assert len(coarse) == 3 * 18  # 13:35, 13:40, ... 15:00 buckets per day
    pd.testing.assert_frame_equal(kept, coarse)


def test_whole_day_last_visible_bar_of_the_day_is_the_1445_bucket():
    bars = _whole_day_bars(_DAYS)
    for day in _DAYS:
        ends = _visible_coarse_bar_ends(bars, day)
        assert ends[-1] == pd.Timestamp(f"{day} 14:45"), day
        assert ends[0] == pd.Timestamp(f"{day} 13:35"), day
        assert len(ends) == 15, day


# --------------------------------------------------------------------------- #
# Direction 2 — a caller that pre-truncates gets the SAME value (the real defect)
# --------------------------------------------------------------------------- #
def test_pretruncated_input_drops_the_1449_residual_bucket():
    bars = _whole_day_bars(_DAYS)
    pre = minute_decision_cutoff(bars, decision_time="14:50:00")
    # PRECONDITION: the pre-truncated 1min input really does end at 14:49, i.e.
    # the fixture reproduces the materializer's geometry.
    assert pd.DatetimeIndex(pre["bar_end"]).max() == pd.Timestamp("2021-07-05 14:49")
    # PRECONDITION: without the completeness filter that geometry DOES produce a
    # residual 14:49 bucket -- the thing being guarded exists in this fixture.
    raw_coarse = resample_intraday_bars(pre, AMP_ANOMALY_FREQ)
    raw_ends = pd.DatetimeIndex(raw_coarse["bar_end"])
    assert pd.Timestamp("2021-07-01 14:49") in set(raw_ends)

    for day in _DAYS:
        ends = _visible_coarse_bar_ends(pre, day)
        assert ends[-1] == pd.Timestamp(f"{day} 14:45"), day
        assert len(ends) == 15, day


def test_factor_value_is_independent_of_caller_pretruncation():
    """The load-geometry invariance the fix exists to restore, asserted bitwise."""
    bars = _whole_day_bars(_DAYS)
    pre = minute_decision_cutoff(bars, decision_time="14:50:00")
    whole = compute_amp_marginal_anomaly_vol(bars, **_GATES)
    truncated = compute_amp_marginal_anomaly_vol(pre, **_GATES)
    # PRECONDITION: a finite value exists on both sides, so equality cannot come
    # from two all-NaN series comparing equal.
    assert whole.notna().any()
    pd.testing.assert_series_equal(whole, truncated)


# --------------------------------------------------------------------------- #
# Direction 3 — a data gap makes a residual bucket too, and it goes as well
# --------------------------------------------------------------------------- #
def test_missing_data_residual_bucket_is_dropped_from_a_whole_day_frame():
    """A bucket whose trailing minutes are absent ends off-grid and is dropped.

    This is the one behavioural change the fix makes for a WHOLE-DAY caller, and
    it is deliberate: the bucket really is shorter than 5 minutes, so pooling its
    amplitude and return with full bars is the same error the 14:49 bucket was.
    """
    rows = [r for r in _tail_session("2021-07-01") if r[0].strftime("%H:%M") not in
            {"14:39", "14:40"}]
    bars = _rows_to_bars(rows)
    raw_ends = set(pd.DatetimeIndex(resample_intraday_bars(bars, AMP_ANOMALY_FREQ)["bar_end"]))
    # PRECONDITION: the gap really did produce an off-grid bucket ending 14:38.
    assert pd.Timestamp("2021-07-01 14:38") in raw_ends
    assert pd.Timestamp("2021-07-01 14:40") not in raw_ends

    kept_ends = set(
        pd.DatetimeIndex(
            _complete_grid_bars(
                resample_intraday_bars(bars, AMP_ANOMALY_FREQ), "5min"
            )["bar_end"]
        )
    )
    assert pd.Timestamp("2021-07-01 14:38") not in kept_ends
    assert pd.Timestamp("2021-07-01 14:35") in kept_ends  # untouched neighbour stays


def test_bucket_with_a_leading_gap_is_kept_because_its_window_still_closed():
    """A bucket missing its LEADING minutes still closes on the grid, so it is kept.

    Pins that the rule tests the bucket's END, not its constituent count -- a
    "count == 5" reading would silently drop these and diverge from the legacy
    path on every illiquid symbol.

    The gap is deliberately at the FRONT of the bucket (14:36/14:37 of the
    14:36..14:40 window). That is what separates this rule from a span-based one:
    a leading gap moves ``bar_start`` while leaving ``bar_end`` on the grid, so
    ``bar_end - bar_start == freq`` would reject the bucket while the real rule
    keeps it.
    """
    rows = [r for r in _tail_session("2021-07-01") if r[0].strftime("%H:%M") not in
            {"14:36", "14:37"}]
    bars = _rows_to_bars(rows)
    coarse = resample_intraday_bars(bars, AMP_ANOMALY_FREQ)
    kept = _complete_grid_bars(coarse, "5min")
    ends = set(pd.DatetimeIndex(kept["bar_end"]))
    # PRECONDITION: the 14:40 bucket really is short two constituents -- its
    # bar_start (min over the constituents) has moved from 14:35 to 14:37, so this
    # bucket genuinely has a hole rather than merely existing.
    holed = coarse[pd.DatetimeIndex(coarse["bar_end"]) == pd.Timestamp("2021-07-01 14:40")]
    assert len(holed) == 1
    assert holed["bar_start"].iloc[0] == pd.Timestamp("2021-07-01 14:37")
    assert pd.Timestamp("2021-07-01 14:40") in ends


# --------------------------------------------------------------------------- #
# Purity
# --------------------------------------------------------------------------- #
def test_complete_grid_bars_does_not_mutate_its_input():
    bars = _whole_day_bars(["2021-07-01"])
    coarse = resample_intraday_bars(minute_decision_cutoff(bars), AMP_ANOMALY_FREQ)
    before = coarse.copy(deep=True)
    out = _complete_grid_bars(coarse, "5min")
    # PRECONDITION: this call really did drop something (else purity is vacuous).
    assert len(out) < len(coarse)
    pd.testing.assert_frame_equal(coarse, before)


def test_completeness_filter_survives_an_all_residual_frame():
    """Every bucket residual -> empty coarse frame -> honest empty factor series."""
    rows = [
        (pd.Timestamp("2021-07-01 09:31"), _SYM, 101.0, 100.0, 100.5),
        (pd.Timestamp("2021-07-01 09:36"), _SYM, 101.0, 100.0, 100.5),
    ]
    bars = _rows_to_bars(rows)
    coarse = resample_intraday_bars(bars, AMP_ANOMALY_FREQ)
    assert len(coarse) == 2  # buckets 09:35 and 09:40, both ending off-grid
    assert _complete_grid_bars(coarse, "5min").empty
    out = compute_amp_marginal_anomaly_vol(bars, **_GATES)
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]


def test_derived_bar_availability_still_comes_from_the_source_maximum():
    """The completeness filter must not disturb the PIT provenance of what remains."""
    bars = _whole_day_bars(["2021-07-01"])
    kept = _complete_grid_bars(resample_intraday_bars(bars, AMP_ANOMALY_FREQ), "5min")
    row = kept[pd.DatetimeIndex(kept["bar_end"]) == pd.Timestamp("2021-07-01 14:45")]
    assert len(row) == 1
    assert row["available_time"].iloc[0] == pd.Timestamp("2021-07-01 14:46")
    assert row["bar_start"].iloc[0] == pd.Timestamp("2021-07-01 14:40")
    assert np.isfinite(float(row["high"].iloc[0]))
