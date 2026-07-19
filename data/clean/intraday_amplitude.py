"""Minute "ideal amplitude" factor (PR-D): trailing-window price-ranked amplitude.

Reproduces the Kaiyuan report §30 (市场微观结构系列 30) 分钟理想振幅因子 as a daily
PIT-safe column derived from 1min bars. Kept in the DATA-clean layer (like
:func:`data.clean.intraday_aggregate.compute_jump_amount_corr`) so the ``factors``
layer only SELECTS the pre-aggregated column and never fetches or sees a forward
return.

Definition (LOCKED — reproduced from the report; N/lambda/min-minutes are part of
the factor DEFINITION, not tuned knobs). For each symbol and each panel date ``d``:

  1. Take the symbol's most recent ``N`` (=10) trading days INCLUDING ``d`` (the
     symbol's own minute-trading days — mirrors the jump factor's trailing window).
  2. PIT truncation (standing authorization): keep only bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50).
     This reuses the I3 per-bar cutoff path, so EVERY day in the window is truncated
     to its own [session-open, 14:50] and post-14:50 / close data is never touched.
  3. Per-bar minute amplitude ``amp = high/low - 1``; a bar is dropped unless
     ``low > 0`` and ``high >= low``.
  4. Pool ALL surviving bars of the window into ONE set ("merged cut", not a
     per-day cut). If the pool has fewer than ``min_minutes`` (=1150 ≈ half of
     10 x ~230) valid minutes, the value is NaN (honest missing — no fabricated
     warm-up; coverage is disclosed by the runner).
  5. Rank the pooled minutes by RAW minute close (unadjusted — amplitude is a ratio
     so it needs no adjustment, and the report ranks on the raw minute price), with
     ``(close, bar_end)`` as a stable total order. With ``k = floor(lambda * n)``
     (lambda=0.25):
        V_high = mean amp of the ``k`` HIGHEST-close minutes
        V_low  = mean amp of the ``k`` LOWEST-close minutes
  6. factor(d, s) = ``V_high - V_low`` (column ``minute_ideal_amp_{N}``).

Pre-registered sign = -1 (report full-market IC -0.059 / ICIR -3.1 / rankIC -0.076).
The value at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees a
future bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.

This module is DATA-layer only: it does not fetch, does not touch factors / alpha /
portfolio / runtime, and never sees a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import DAILY_INDEX_NAMES, DEFAULT_DECISION_TIME
from data.clean.intraday_schema import SYMBOL_LEVEL, validate_intraday_bars

# Factor DEFINITION constants (report terminal parameters; NOT tuned knobs).
IDEAL_AMP_LOOKBACK_DAYS = 10  # trailing trading-day window (N), includes date d
IDEAL_AMP_LAMBDA = 0.25  # top/bottom fraction by close (lambda) that forms V_high/V_low
IDEAL_AMP_MIN_MINUTES = 1150  # minimum valid pooled minutes for a finite value


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def _rank_cut(
    closes: np.ndarray, amps: np.ndarray, bar_ends: np.ndarray, lam: float, min_minutes: int
) -> float:
    """V_high - V_low over one pooled window; NaN if too few minutes or k < 1.

    ``closes``/``amps``/``bar_ends`` are equal-length arrays for ONE pooled window.
    Ranking is a stable total order ``(close, bar_end)`` (lexsort with close as the
    primary key), so the top-k / bottom-k selection is fully deterministic even when
    two minutes share a close.
    """
    n = closes.size
    if n < min_minutes:
        return float("nan")
    k = int(np.floor(lam * n))
    if k < 1:
        return float("nan")
    order = np.lexsort((bar_ends, closes))  # close primary, bar_end tie-break
    a = amps[order]
    return float(a[-k:].mean() - a[:k].mean())


def _amplitude_for_symbol(
    g: pd.DataFrame, lookback_days: int, lam: float, min_minutes: int
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily factor values for ONE symbol from its PIT-filtered, guarded bars.

    ``g`` holds columns ``trade_date`` / ``close`` / ``amp`` / ``bar_end_ns`` for a
    single symbol. Per-day arrays are built once, then each date pools the trailing
    ``lookback_days`` days (including that date) — no cross-symbol leakage because
    ``g`` is a single symbol's slice.
    """
    days: list[pd.Timestamp] = []
    day_close: list[np.ndarray] = []
    day_amp: list[np.ndarray] = []
    day_be: list[np.ndarray] = []
    for day, sub in g.groupby("trade_date", sort=True):
        days.append(pd.Timestamp(day).normalize())
        day_close.append(sub["close"].to_numpy(dtype=float))
        day_amp.append(sub["amp"].to_numpy(dtype=float))
        day_be.append(sub["bar_end_ns"].to_numpy(dtype="int64"))

    values: list[float] = []
    for j in range(len(days)):
        lo = max(0, j - lookback_days + 1)
        closes = np.concatenate(day_close[lo : j + 1])
        amps = np.concatenate(day_amp[lo : j + 1])
        bes = np.concatenate(day_be[lo : j + 1])
        values.append(_rank_cut(closes, amps, bes, lam, min_minutes))
    return days, values


def compute_minute_ideal_amplitude(
    bars: pd.DataFrame,
    *,
    lookback_days: int = IDEAL_AMP_LOOKBACK_DAYS,
    lam: float = IDEAL_AMP_LAMBDA,
    min_minutes: int = IDEAL_AMP_MIN_MINUTES,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "minute_ideal_amp",
) -> pd.Series:
    """PIT-safe daily "minute ideal amplitude" factor from 1min ``bars``.

    See the module docstring for the LOCKED definition. The heavy per-symbol loop is
    memory-bounded (the runner feeds one symbol at a time), but this function also
    accepts a multi-symbol frame and keeps symbols strictly isolated.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``.
        lookback_days: trailing trading-day window length (part of the definition).
        lam: top/bottom close fraction defining V_high/V_low (0 < lam <= 0.5).
        min_minutes: minimum valid pooled minutes for a finite value.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if not (0.0 < lam <= 0.5):
        raise ValueError(f"lam must be in (0, 0.5]; got {lam!r}.")
    if min_minutes < 2:
        # Need at least 2 minutes so a non-empty top/bottom cut can exist.
        raise ValueError(f"min_minutes must be >= 2; got {min_minutes!r}.")
    if len(bars) == 0:
        return _empty_series(name)

    work = bars.reset_index()[
        [SYMBOL_LEVEL, "bar_end", "available_time", "high", "low", "close"]
    ].copy()
    work["trade_date"] = work["bar_end"].dt.normalize()
    # PIT truncation FIRST (per-bar timestamps): each bar's cutoff is its own
    # trade_date + decision_time, so every day is truncated to [open, 14:50].
    cutoff = work["trade_date"] + pd.Timedelta(decision_time)
    visible = work.loc[work["available_time"] <= cutoff].copy()
    if visible.empty:
        return _empty_series(name)

    low = visible["low"].to_numpy(dtype=float)
    high = visible["high"].to_numpy(dtype=float)
    guard = (low > 0.0) & (high >= low)
    visible = visible.loc[guard].copy()
    if visible.empty:
        return _empty_series(name)

    visible["amp"] = visible["high"].to_numpy(dtype=float) / visible["low"].to_numpy(
        dtype=float
    ) - 1.0
    # int64 nanoseconds for a deterministic lexsort tie-break (view avoids the
    # datetime->int astype deprecation).
    visible["bar_end_ns"] = visible["bar_end"].to_numpy(dtype="datetime64[ns]").astype(
        "int64"
    )
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _amplitude_for_symbol(g, lookback_days, lam, min_minutes)
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "IDEAL_AMP_LAMBDA",
    "IDEAL_AMP_LOOKBACK_DAYS",
    "IDEAL_AMP_MIN_MINUTES",
    "compute_minute_ideal_amplitude",
]
