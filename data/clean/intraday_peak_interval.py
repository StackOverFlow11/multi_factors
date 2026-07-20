"""Volume-peak INTERVAL-KURTOSIS factor (PR-H).

Reproduces the SECOND factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§6): "针对两个量峰之间的时间间隔计算统计指标，对过去 20 日同日前后两个量峰之间的时间间隔
分布，计算其峰度".

Same MACHINE as PR-F, different STATISTIC. The volume-peak identification is REUSED
verbatim from :mod:`data.clean.intraday_volume_prv` (``prepare_visible_minute_bars`` +
``peak_mask_for_symbol``) — same-slot strictly-prior μ+kσ eruptive/mild classification,
1-minute same-session mild-neighbour peak rule, session-boundary and missing-minute
bars are never peaks, a day is VALID iff it has enough classifiable bars. Nothing about
the taxonomy is re-implemented here, so the two factors can never drift apart; this
module only measures the GAPS between consecutive peaks and reduces them to a kurtosis.

The report is silent on two operative details; both are deliberate, DISCLOSED choices
PINNED in the task card (task_card_pr_h_*.md §1), not tuned knobs, and both are
reproduced on the factor spec:

  1. INTERVAL UNIT = TRADING MINUTES, i.e. the difference in TRADABLE SLOT POSITION
     within the day's PIT-visible bar sequence — NOT wall-clock minutes. The lunch break
     therefore costs nothing: a peak at 11:29 and a peak at 13:02 are 3 trading minutes
     apart (11:29 -> 11:30 -> 13:01 -> 13:02), not 93. Measuring wall clock would inject
     a fixed ~90-minute spike into every interval distribution that happens to straddle
     lunch and would dominate the kurtosis. A consequence, disclosed rather than
     corrected: a minute MISSING from the cache is not a tradable slot in our sequence,
     so an interval spanning it is measured one shorter (the same stance PR-F takes on
     gaps — it consults no exchange calendar).
  2. KURTOSIS = FISHER EXCESS, BIAS-CORRECTED — the ``pandas.Series.kurt()`` /
     ``scipy.stats.kurtosis(fisher=True, bias=False)`` convention (a normal sample sits
     near 0, not 3). ``excess_kurtosis`` implements it in numpy for speed and is locked
     against ``pandas.kurt()`` by test.

Factor value: ``peak_interval_kurtosis_20`` = the kurtosis of the POOLED interval
multiset over the symbol's most recent ``lookback_days`` (=20) VALID trading days
INCLUDING ``d``. A day with fewer than 2 peaks contributes ZERO intervals but is still a
valid day (it is not skipped and does not poison the window). Two NaN gates, both honest
missing rather than a fabricated number: fewer than ``min_intervals`` (=20) pooled
intervals (kurtosis is wildly unstable on small samples), or a zero-variance pool
(kurtosis is undefined). The PR-F valid-day floor is kept as well (``min_valid_days``).
Note the reused taxonomy makes two peaks 1 minute apart impossible — adjacent eruptive
minutes are RIDGES — so the smallest attainable interval is 2.

Pre-registered sign = +1 (the report's full-market RankIC is +7.19% / RankICIR 4.63,
long-short 23.3%/yr, IR 3.39, 13/13 positive years — the most stable factor in the
report). Semantics: a peaky, fat-tailed interval distribution means informed trading
arrives in BURSTS rather than spread evenly through the session -> higher future return.
NOTE the report is a MONTHLY, market-cap + industry neutral full-market series on Wind
data; our eval cell is CSI500 daily with industry + size neutralization, so its numbers
are a LOOSE reference only (the report gives no CSI500 sub-domain figure for THIS
factor, and none is invented). Raw minute volume (cached as-is) has split-day magnitude
jumps that pollute the 20-day σ; the report (Wind) does not adjust for this either, so
it is disclosed and NOT corrected.

The value at ``d`` uses only bars at dates <= d, so a factor value never sees a future
bar (invariant #1); it is a DAILY signal traded close-to-close from d+1. This module is
DATA-layer only: it does not fetch, does not touch factors / alpha / portfolio /
runtime, and never sees a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import DAILY_INDEX_NAMES, DEFAULT_DECISION_TIME
from data.clean.intraday_schema import SYMBOL_LEVEL, validate_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The peak-identification constants are IMPORTED from PR-F, never redefined.
PEAK_INTERVAL_LOOKBACK_DAYS = 20  # trailing VALID trading-day pool window, includes d
PEAK_INTERVAL_MIN_INTERVALS = 20  # min pooled intervals for a finite kurtosis

# Kurtosis needs at least 4 observations for the bias-corrected estimator to exist.
_KURTOSIS_MIN_N = 4


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def excess_kurtosis(x: np.ndarray) -> float:
    """Fisher excess, bias-corrected kurtosis of ``x`` — the ``pandas.kurt()`` convention.

    Equivalent to ``pandas.Series(x).kurt()`` and to
    ``scipy.stats.kurtosis(x, fisher=True, bias=False)`` (locked by test), computed in
    numpy because the caller evaluates it once per symbol-day::

        G2 = n(n+1)(n-1)·M4 / ((n-2)(n-3)·M2²) - 3(n-1)² / ((n-2)(n-3))

    with ``M2 = Σ(x-x̄)²`` and ``M4 = Σ(x-x̄)⁴`` (central sums, NOT means). Central sums
    are accumulated after subtracting the mean, so no catastrophic power-sum
    cancellation.

    Returns:
        The excess kurtosis, or NaN when it is undefined: fewer than 4 observations, or
        a zero-variance sample. Never raises, never returns ±inf.
    """
    n = x.size
    if n < _KURTOSIS_MIN_N:
        return float("nan")
    d = x - x.mean()
    m2 = float(np.dot(d, d))
    if not m2 > 0.0:  # zero variance (or a non-finite that made it NaN) -> undefined
        return float("nan")
    m4 = float(np.dot(d * d, d * d))
    left = n * (n + 1.0) * (n - 1.0) * m4 / ((n - 2.0) * (n - 3.0) * m2 * m2)
    adj = 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))
    return left - adj


def peak_intervals_by_day(work: pd.DataFrame) -> pd.Series:
    """Trading-minute gaps between consecutive same-day peaks, per trade date.

    ``work`` is one symbol's frame as returned by
    :func:`~data.clean.intraday_volume_prv.peak_mask_for_symbol` (sorted by
    ``(trade_date, bar_end)`` with a boolean ``peak`` column).

    The interval unit is the TRADABLE SLOT POSITION difference inside the day's visible
    bar sequence (PINNED §1 of the module docstring): position 0, 1, 2, ... is assigned
    to the day's PIT-visible bars in time order, so consecutive tradable minutes are 1
    apart REGARDLESS of the lunch break sitting between them. Intervals are never taken
    ACROSS days (the positions restart every trade date).

    Returns:
        Series indexed by ``trade_date`` (every day present in ``work``, in order) whose
        values are integer numpy arrays of that day's peak-to-peak intervals — an EMPTY
        array for a day with fewer than 2 peaks (zero intervals, still a real day).
    """
    # position within the day's visible bar sequence == "trading minute" coordinate
    pos = work.groupby("trade_date", sort=False).cumcount().to_numpy()
    peak = work["peak"].to_numpy(dtype=bool)
    dates = work["trade_date"].to_numpy()

    out_dates: list[pd.Timestamp] = []
    out_intervals: list[np.ndarray] = []
    for day, idx in pd.Series(np.arange(len(work))).groupby(dates, sort=True):
        sel = idx.to_numpy()
        day_pos = pos[sel][peak[sel]]
        out_dates.append(pd.Timestamp(day))
        out_intervals.append(np.diff(day_pos).astype(np.int64))
    return pd.Series(out_intervals, index=pd.Index(out_dates, name="trade_date"))


def _kurtosis_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
    min_intervals: int,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily peak-interval-kurtosis values for ONE symbol from its PIT-visible bars.

    Identifies the peaks with the REUSED :func:`peak_mask_for_symbol`, measures each
    valid day's peak-to-peak trading-minute intervals, then pools the trailing
    ``lookback_days`` valid days and reduces the pool to its excess kurtosis. No
    cross-symbol leakage (``g`` is one symbol's slice) and no lookahead (the baseline is
    strictly prior, the pool is trailing).
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )

    classifiable_count = work.groupby("trade_date")["classifiable"].sum()
    valid_days = classifiable_count.index[classifiable_count >= min_classifiable]
    if len(valid_days) == 0:
        return [], []

    intervals = peak_intervals_by_day(work)
    # Keep ONLY valid days, in order: a day below the classifiable floor contributes
    # nothing and does not occupy a slot in the trailing window (same rule as PR-F).
    valid = [pd.Timestamp(d) for d in sorted(valid_days)]
    per_day = [intervals.loc[d] for d in valid]

    days: list[pd.Timestamp] = []
    values: list[float] = []
    for i, day in enumerate(valid):
        window = per_day[max(0, i - lookback_days + 1) : i + 1]
        days.append(day.normalize())
        if len(window) < min_valid_days:
            values.append(float("nan"))
            continue
        pool = np.concatenate(window) if window else np.empty(0, dtype=np.int64)
        # Gate 1: too few pooled intervals -> honest NaN (kurtosis is wild on small
        # samples). Gate 2 (zero variance / n < 4) lives inside excess_kurtosis.
        if pool.size < min_intervals:
            values.append(float("nan"))
            continue
        values.append(excess_kurtosis(pool.astype(float)))
    return days, values


def compute_peak_interval_kurtosis(
    bars: pd.DataFrame,
    *,
    lookback_days: int = PEAK_INTERVAL_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_intervals: int = PEAK_INTERVAL_MIN_INTERVALS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "peak_interval_kurtosis",
) -> pd.Series:
    """PIT-safe daily "volume-peak interval kurtosis" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    identifies the volume peaks with the REUSED PR-F taxonomy, measures the
    trading-minute gaps between consecutive same-day peaks, pools the trailing
    ``lookback_days`` VALID days and returns the pool's Fisher excess (bias-corrected)
    kurtosis. See the module docstring for the LOCKED definition and the two pinned
    interpretations (interval unit, kurtosis convention).

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping
            is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window pooled for the kurtosis.
        baseline_days: strictly-prior same-slot baseline window in trading days (PR-F).
        baseline_min_obs: minimum same-slot observations for a classifiable bar (PR-F).
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ`` (PR-F).
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day is VALID iff it has at least this many classifiable bars.
        min_intervals: minimum POOLED intervals for a finite kurtosis (>= 4, the point
            below which the bias-corrected estimator does not exist).
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if baseline_days < 2:
        # Need >= 2 baseline observations so a ddof=1 std of the same-slot volume is
        # defined.
        raise ValueError(f"baseline_days must be >= 2; got {baseline_days!r}.")
    if baseline_min_obs < 2:
        raise ValueError(f"baseline_min_obs must be >= 2; got {baseline_min_obs!r}.")
    if sigma_k < 0.0:
        raise ValueError(f"sigma_k must be >= 0; got {sigma_k!r}.")
    if min_valid_days < 1:
        raise ValueError(f"min_valid_days must be >= 1; got {min_valid_days!r}.")
    if min_classifiable < 1:
        raise ValueError(f"min_classifiable must be >= 1; got {min_classifiable!r}.")
    if min_intervals < _KURTOSIS_MIN_N:
        raise ValueError(
            f"min_intervals must be >= {_KURTOSIS_MIN_N} (the bias-corrected kurtosis "
            f"does not exist below that); got {min_intervals!r}."
        )
    if len(bars) == 0:
        return _empty_series(name)

    visible = prepare_visible_minute_bars(bars, decision_time=decision_time)
    if visible.empty:
        return _empty_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _kurtosis_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_intervals=min_intervals,
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "PEAK_INTERVAL_LOOKBACK_DAYS",
    "PEAK_INTERVAL_MIN_INTERVALS",
    "compute_peak_interval_kurtosis",
    "excess_kurtosis",
    "peak_intervals_by_day",
]
