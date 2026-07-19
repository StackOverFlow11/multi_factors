"""Amplitude marginal-anomaly relative-volatility factor (PR-E).

Reproduces the Changjiang high-frequency-factor series #19 (长江证券《高频因子（十九）》,
2026-06-03, reportId 5462994) "振幅边际异常相对波动因子" as a daily PIT-safe column
derived from 5min bars (themselves DERIVED from the 1min cache). Kept in the
DATA-clean layer (like :func:`data.clean.intraday_amplitude.compute_minute_ideal_amplitude`)
so the ``factors`` layer only SELECTS the pre-aggregated column and never fetches or
sees a forward return.

The source report is UNDER-SPECIFIED; the five choices below are deliberate,
DISCLOSED interpretations pinned in the task card (task_card_pr_e_*.md §0), not tuned
knobs. They are reproduced verbatim on the factor spec so a reader can see exactly
what was assumed:

  1. bar frequency = 5min, DERIVED from the 1min cache via
     :func:`data.clean.intraday_aggregate.resample_intraday_bars` (a derived bar
     inherits ``available_time = max(source_1min.available_time)`` — PIT-faithful).
  2. lookback window N = 20 trading days (the symbol's own days that have bars,
     trailing and INCLUDING ``d``).
  3. anomaly threshold = ``1{|Δamp_t| > μ + σ}`` with μ / σ = mean / ddof=1 std of
     ALL pooled ``|Δamp|`` in the 20-day pool (k = 1).
  4. weighted volatility = the ddof=1 SAMPLE std of the RETURNS on the selected
     (weight-1) bars.
  5. bar return ``r_t = close_t/close_{t-1} - 1`` and ``Δamp_t = amp_t - amp_{t-1}``
     are BOTH WITHIN-DAY lagged — each day's FIRST bar has no return / no Δamp — so
     the overnight gap never contaminates a pair.

Definition (per symbol, per panel date ``d``):

  1. Take the symbol's most recent ``N`` (=20) trading days INCLUDING ``d`` of 5min
     bars. PIT truncation (standing authorization): keep only bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50),
     so every day is truncated to its own [session-open, 14:50].
  2. Per bar: ``amp = high/low - 1`` (drop a bar unless ``low > 0`` and
     ``high >= low``). WITHIN each (symbol, day), sorted by ``bar_end``, form
     ``Δamp_t = amp_t - amp_{t-1}`` and ``r_t = close_t/close_{t-1} - 1``; the first
     surviving bar of each day has neither (no cross-day lag).
  3. Pool ALL surviving ``(|Δamp|, r)`` pairs of the ``N``-day window into ONE set.
     If the pool has fewer than ``min_pool`` (=460 ≈ 46 x 20 / 2) valid pairs, the
     value is NaN (honest missing — coverage is disclosed by the runner).
  4. Pool ``μ = mean(|Δamp|)`` and ``σ = std(|Δamp|, ddof=1)``; select the bars with
     ``|Δamp_t| > μ + k*σ`` (k = 1). If fewer than ``min_selected`` (=20) bars are
     selected, the value is NaN (an anomaly std over too few bars is unreliable).
  5. factor(d, s) = ``std(r_t | selected, ddof=1)`` (column
     ``amp_marginal_anomaly_vol_{N}``).

Pre-registered sign = +1 (the report's IC is positive across its universes:
raw CSI800 +4.47% / full-market +4.92%; market-cap + industry neutral +4.11% /
+5.56%, ICIR 65.85% / 108.71%). NOTE the report's sample is CSI800 / full-market on
a MONTHLY series; our eval cell is CSI500 daily — a LOOSE reference only. The value
at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees a future
bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.

This module is DATA-layer only: it does not fetch, does not touch factors / alpha /
portfolio / runtime, and never sees a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    resample_intraday_bars,
)
from data.clean.intraday_schema import SYMBOL_LEVEL, validate_intraday_bars

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
AMP_ANOMALY_LOOKBACK_DAYS = 20  # trailing trading-day window (N), includes date d
AMP_ANOMALY_FREQ = "5min"  # bar frequency, DERIVED from the 1min cache
AMP_ANOMALY_SIGMA_K = 1.0  # anomaly threshold multiplier k in |Δamp| > μ + k*σ
AMP_ANOMALY_MIN_POOL = 460  # minimum valid pooled (|Δamp|, r) pairs for a finite value
AMP_ANOMALY_MIN_SELECTED = 20  # minimum selected (anomaly) bars for a finite std


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def _anomaly_vol_cut(
    dabs: np.ndarray,
    ret: np.ndarray,
    min_pool: int,
    min_selected: int,
    sigma_k: float,
) -> float:
    """Selected-bar return std over one pooled window; NaN if a gate fails.

    ``dabs`` / ``ret`` are equal-length arrays of the VALID ``(|Δamp|, r)`` pairs in
    ONE pooled window (already NaN-free). Fewer than ``min_pool`` pairs -> NaN; select
    the bars whose ``|Δamp|`` exceeds ``mean + sigma_k * std(ddof=1)`` of the pooled
    ``|Δamp|``; fewer than ``min_selected`` selected -> NaN (an unreliable std).
    """
    n = dabs.size
    if n < min_pool:
        return float("nan")
    mu = float(dabs.mean())
    sigma = float(dabs.std(ddof=1))
    threshold = mu + sigma_k * sigma
    mask = dabs > threshold
    n_sel = int(np.count_nonzero(mask))
    if n_sel < min_selected:
        return float("nan")
    return float(ret[mask].std(ddof=1))


def _anomaly_vol_for_symbol(
    g: pd.DataFrame,
    lookback_days: int,
    min_pool: int,
    min_selected: int,
    sigma_k: float,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily factor values for ONE symbol from its within-day-lagged bars.

    ``g`` holds columns ``trade_date`` / ``dabs`` (``|Δamp_t|``) / ``ret`` (``r_t``)
    for a single symbol; ``dabs`` / ``ret`` are NaN on each day's first surviving bar
    (no cross-day lag). Per day the VALID (finite) pairs are collected once, then each
    date pools the trailing ``lookback_days`` days (including that date) — no
    cross-symbol leakage because ``g`` is a single symbol's slice, and no cross-day
    leakage because each day's first-bar pair is already NaN and dropped here.
    """
    days: list[pd.Timestamp] = []
    day_dabs: list[np.ndarray] = []
    day_ret: list[np.ndarray] = []
    for day, sub in g.groupby("trade_date", sort=True):
        days.append(pd.Timestamp(day).normalize())
        d = sub["dabs"].to_numpy(dtype=float)
        r = sub["ret"].to_numpy(dtype=float)
        valid = np.isfinite(d) & np.isfinite(r)
        day_dabs.append(d[valid])
        day_ret.append(r[valid])

    values: list[float] = []
    for j in range(len(days)):
        lo = max(0, j - lookback_days + 1)
        dabs = np.concatenate(day_dabs[lo : j + 1])
        ret = np.concatenate(day_ret[lo : j + 1])
        values.append(_anomaly_vol_cut(dabs, ret, min_pool, min_selected, sigma_k))
    return days, values


def compute_amp_marginal_anomaly_vol(
    bars: pd.DataFrame,
    *,
    lookback_days: int = AMP_ANOMALY_LOOKBACK_DAYS,
    min_pool: int = AMP_ANOMALY_MIN_POOL,
    min_selected: int = AMP_ANOMALY_MIN_SELECTED,
    sigma_k: float = AMP_ANOMALY_SIGMA_K,
    decision_time: str = DEFAULT_DECISION_TIME,
    freq: str = AMP_ANOMALY_FREQ,
    name: str = "amp_marginal_anomaly_vol",
) -> pd.Series:
    """PIT-safe daily "amplitude marginal-anomaly relative-volatility" factor.

    Takes normalized 1min ``bars``, DERIVES ``freq`` (default 5min) bars from them via
    :func:`resample_intraday_bars` (so a derived bar is usable only once EVERY
    constituent 1min bar is available), PIT-truncates them at ``decision_time``, forms
    the within-day ``|Δamp|`` / ``r`` pairs, and returns the trailing-``lookback_days``
    anomaly-bar return std. See the module docstring for the LOCKED definition.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping
            is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing trading-day window length (part of the definition).
        min_pool: minimum valid pooled ``(|Δamp|, r)`` pairs for a finite value.
        min_selected: minimum selected (anomaly) bars for a finite return std.
        sigma_k: anomaly threshold multiplier ``k`` in ``|Δamp| > μ + k*σ``.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        freq: derived bar frequency (default 5min); part of the definition.
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if min_pool < 2:
        # Need >= 2 pooled points so a ddof=1 std of |Δamp| is defined.
        raise ValueError(f"min_pool must be >= 2; got {min_pool!r}.")
    if min_selected < 2:
        # Need >= 2 selected bars so a ddof=1 std of their returns is defined.
        raise ValueError(f"min_selected must be >= 2; got {min_selected!r}.")
    if sigma_k < 0.0:
        raise ValueError(f"sigma_k must be >= 0; got {sigma_k!r}.")
    if len(bars) == 0:
        return _empty_series(name)

    # DERIVE the coarse (5min) bars from 1min FIRST: available_time = max source, so a
    # coarse bar enters only once all its 1min constituents are available.
    coarse = resample_intraday_bars(bars, freq)
    if len(coarse) == 0:
        return _empty_series(name)

    work = coarse.reset_index()[
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
    # Sort so the within-day lag sees bars in chronological order within each
    # (symbol, day); mergesort keeps it stable.
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")
    # Δamp and r are WITHIN-DAY lagged: grouping by (symbol, trade_date) makes each
    # day's FIRST surviving bar NaN, so no pair ever crosses the overnight gap.
    by_session = visible.groupby([SYMBOL_LEVEL, "trade_date"], sort=False)
    visible["dabs"] = by_session["amp"].diff().abs()
    visible["ret"] = by_session["close"].pct_change()

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _anomaly_vol_for_symbol(
            g, lookback_days, min_pool, min_selected, sigma_k
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "AMP_ANOMALY_FREQ",
    "AMP_ANOMALY_LOOKBACK_DAYS",
    "AMP_ANOMALY_MIN_POOL",
    "AMP_ANOMALY_MIN_SELECTED",
    "AMP_ANOMALY_SIGMA_K",
    "compute_amp_marginal_anomaly_vol",
]
