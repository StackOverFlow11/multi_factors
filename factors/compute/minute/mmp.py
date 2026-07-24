"""Minute Microstructure Pressure (MMP, I5c): the per-bar factor math (D2).

EXPLORATORY factor (never promoted; I5d's CSI500 monotonicity degraded on the
corrected engine and I5e's CSI300 generalization failed — MMP is on hold). The
math lives HERE since D2 (moved from ``data.clean.intraday_aggregate``, which
re-exports it); the daily equal-weight aggregation is consumed through
``asof_daily_features(features=["mmp_ew"])`` in the aggregate module's generic
core, which imports :func:`mmp_ew_daily` from this module.

Layering note: this module must NEVER import ``data.clean.intraday_aggregate``
— the aggregate module imports THIS one (re-export + feature hook), so an
import back would be a genuine cycle. Everything shared lives in
``data.clean.intraday_schema`` / ``factors.compute.minute.primitives``.

The window is part of the factor definition, not a tuned parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    DEFAULT_SESSION_OPEN,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)

# Minute Microstructure Pressure (MMP, I5c): rolling baseline window (prior bars
# t-MMP_LOOKBACK..t-1) and the default denominator epsilon. EXPLORATORY factor;
# the window is part of the factor definition, not a tuned parameter.
MMP_LOOKBACK = 20
DEFAULT_EPSILON = 1e-6


def compute_minute_mmp(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    lookback: int = MMP_LOOKBACK,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Per-bar Minute Microstructure Pressure ``MMP_t`` for ONE symbol/day session.

    Inputs are equal-length 1D arrays ORDERED by ``bar_end`` ascending and
    belonging to a SINGLE ``(symbol, trade_date)`` session. The rolling baselines
    use ONLY the prior ``lookback`` bars (``t-lookback..t-1``) — never bar ``t``
    itself, never a later bar, never the prior day's tail — so the first
    ``lookback`` bars have NaN ``MMP``.

        mid_t = (high_t + low_t) / 2
        S_t   = (close_t - mid_t) / mid_t                  (NaN if mid_t <= 0)
        V_t   = sqrt(volume_t / median(volume[t-lookback:t]))
                                                           (NaN if baseline <= 0 / NaN)
        B_t   = |close_t - open_t| / (high_t - low_t + epsilon)
        R_t   = (high_t - low_t) / (mean(hl[t-lookback:t]) + epsilon)
                                                           (NaN if baseline is NaN)
        MMP_t = S_t * V_t * B_t * R_t

    Invalid denominators yield NaN, never ``inf``. Pure: reads no returns / no
    future bars / no token.
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)

    hl = high - low
    mid = (high + low) / 2.0

    # Prior-`lookback` baselines: rolling over t-lookback+1..t THEN shift(1) so
    # position t holds the statistic of bars t-lookback..t-1 (excludes bar t).
    med_vol = pd.Series(volume).rolling(lookback).median().shift(1).to_numpy()
    ma_hl = pd.Series(hl).rolling(lookback).mean().shift(1).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        s_t = np.where(mid > 0.0, (close - mid) / mid, np.nan)
        ratio = np.where(med_vol > 0.0, volume / med_vol, np.nan)
        v_t = np.sqrt(ratio)
        b_t = np.abs(close - open_) / (hl + epsilon)
        r_t = hl / (ma_hl + epsilon)
        mmp = s_t * v_t * b_t * r_t
    return mmp


def in_session_bars(
    g: pd.DataFrame, trade_date: pd.Timestamp, session_open: str
) -> pd.DataFrame:
    """Bars whose ``bar_end`` is on/after ``session_open`` (the MMP window lower bound).

    The MMP daily score aggregates over ``[session_open, decision_time]`` (the upper
    bound is the available_time cutoff already applied upstream). Restricting to the
    in-session bars keeps PRE-session bars out of BOTH the rolling baseline and the
    daily mean, so the first in-session bar correctly has no prior-20 baseline.
    """
    session_start = trade_date + pd.Timedelta(session_open)
    return g[g["bar_end"] >= session_start]


def mmp_ew_daily(
    g: pd.DataFrame, epsilon: float, trade_date: pd.Timestamp, session_open: str
) -> float:
    """Equal-weight mean of valid per-minute ``MMP_t`` over one PIT-filtered group.

    ``g`` is one ``(date, symbol)`` session's visible bars, sorted by ``bar_end``;
    only the in-session bars (``bar_end >= session_open``) enter, so the rolling
    baseline starts at the session open and the first 20 in-session bars are NaN.
    Every valid minute ``MMP_t`` gets EQUAL weight (no extra volume weighting — the
    volume term already lives inside ``MMP_t``). No valid minute -> NaN.
    """
    gs = in_session_bars(g, trade_date, session_open)
    if gs.empty:
        return float("nan")
    mmp = compute_minute_mmp(
        gs["open"].to_numpy(dtype=float),
        gs["high"].to_numpy(dtype=float),
        gs["low"].to_numpy(dtype=float),
        gs["close"].to_numpy(dtype=float),
        gs["volume"].to_numpy(dtype=float),
        epsilon=epsilon,
    )
    valid = mmp[~np.isnan(mmp)]
    return float(np.mean(valid)) if valid.size else float("nan")


def mmp_valid_minute_counts(
    bars: pd.DataFrame,
    *,
    decision_time: str = DEFAULT_DECISION_TIME,
    session_open: str = DEFAULT_SESSION_OPEN,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.Series:
    """Per-``(date, symbol)`` count of valid (non-NaN) ``MMP_t`` minutes (I5c report).

    Report-only diagnostic: applies the SAME window as the daily MMP score —
    ``available_time <= trade_date + decision_time`` (upper bound) AND
    ``bar_end >= trade_date + session_open`` (lower bound) — then counts the
    in-session minutes that yielded a valid ``MMP_t`` (the first ``MMP_LOOKBACK``
    in-session bars never do). Reuses :func:`compute_minute_mmp` so there is a
    single MMP source of truth.
    """
    validate_intraday_bars(bars)
    empty = pd.Series(
        [], dtype=int,
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=DAILY_INDEX_NAMES,
        ),
    )
    if len(bars) == 0:
        return empty
    work = bars.reset_index()
    work["trade_date"] = work["bar_end"].dt.normalize()
    cutoff = work["trade_date"] + pd.Timedelta(decision_time)
    visible = work.loc[work["available_time"] <= cutoff].copy()
    if visible.empty:
        return empty
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"])
    index_tuples: list[tuple] = []
    counts: list[int] = []
    for (date, sym), g in visible.groupby(["trade_date", SYMBOL_LEVEL], sort=True):
        gs = in_session_bars(g, pd.Timestamp(date).normalize(), session_open)
        if gs.empty:
            index_tuples.append((date, str(sym)))
            counts.append(0)
            continue
        mmp = compute_minute_mmp(
            gs["open"].to_numpy(dtype=float),
            gs["high"].to_numpy(dtype=float),
            gs["low"].to_numpy(dtype=float),
            gs["close"].to_numpy(dtype=float),
            gs["volume"].to_numpy(dtype=float),
            epsilon=epsilon,
        )
        index_tuples.append((date, str(sym)))
        counts.append(int(np.count_nonzero(~np.isnan(mmp))))
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(counts, index=index, dtype=int).sort_index()


__all__ = [
    "DEFAULT_EPSILON",
    "MMP_LOOKBACK",
    "compute_minute_mmp",
    "in_session_bars",
    "mmp_ew_daily",
    "mmp_valid_minute_counts",
]
