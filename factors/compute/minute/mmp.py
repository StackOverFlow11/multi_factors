"""Minute Microstructure Pressure (MMP, I5c): the per-bar factor math (D2).

EXPLORATORY factor (never promoted; I5d's CSI500 monotonicity degraded on the
corrected engine and I5e's CSI300 generalization failed — MMP is on hold). The
math lives HERE since D2 (moved from ``data.clean.intraday_aggregate``, which
re-exports it); the daily equal-weight aggregation is consumed through
``asof_daily_features(features=["mmp_ew"])`` in the aggregate module's generic
core, which imports :func:`mmp_ew_daily` from this module. D6c adds the
first-class factor surface: :func:`compute_intraday_mmp_ew` (visible-filter +
per-session groupby around the SAME ``mmp_ew_daily``) and
:class:`MmpEwFactor`, whose id is the legacy aggregate column name.

Layering note: this module must NEVER import ``data.clean.intraday_aggregate``
— the aggregate module imports THIS one (re-export + feature hook), so an
import back would be a genuine cycle. Everything shared lives in
``data.clean.intraday_schema`` / ``factors.compute.minute.primitives``.

The window is part of the factor definition, not a tuned parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    DEFAULT_SESSION_OPEN,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)
from factors.base import Factor
from factors.spec import FactorSpec, PanelField

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


#: The factor id of the MMP equal-weight session score — byte-identical to the
#: legacy I3 aggregate column name (``data.clean.intraday_aggregate._column_name``
#: derives the same string from the same constants), because every shipped report
#: is keyed on that column name verbatim. Pinned equal by test.
MMP_EW_FACTOR_NAME = f"intraday_mmp{MMP_LOOKBACK}_ew_0930_1450"


def compute_intraday_mmp_ew(
    bars: pd.DataFrame,
    *,
    decision_time: str = DEFAULT_DECISION_TIME,
    session_open: str = DEFAULT_SESSION_OPEN,
    epsilon: float = DEFAULT_EPSILON,
    name: str = MMP_EW_FACTOR_NAME,
) -> pd.Series:
    """The I3 ``mmp_ew`` session feature as a factor raw daily Series (D6c).

    Per ``(date, symbol)``: keep only the bars with
    ``available_time <= trade_date + decision_time`` (the SAME per-bar PIT
    cutoff ``data.clean.intraday_aggregate.asof_daily_features`` applies — the
    filter runs on timestamps BEFORE any daily grouping), then take the
    equal-weight mean of the session's valid per-minute ``MMP_t`` via
    :func:`mmp_ew_daily`, so the MMP math has exactly ONE definition point and
    this function adds only the visible-filter + groupby shell (the template
    ``mmp_valid_minute_counts`` already uses).

    Returns:
        ``MultiIndex(date, symbol)`` float Series (midnight-normalized dates),
        sorted, named ``name``. A session with no valid minute yields NaN.
        Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    empty = pd.Series(
        [], dtype=float, name=name,
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
    values: list[float] = []
    for (date, sym), g in visible.groupby(["trade_date", SYMBOL_LEVEL], sort=True):
        index_tuples.append((date, str(sym)))
        values.append(
            mmp_ew_daily(g, epsilon, pd.Timestamp(date).normalize(), session_open)
        )
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, dtype=float, name=name).sort_index()


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


class MmpEwFactor(Factor):
    """The I3 MMP equal-weight session score as a first-class factor (D6c).

    The factor id IS the legacy aggregate column name
    (``intraday_mmp20_ew_0930_1450``), so every report already keyed on that
    column keeps its text byte-for-byte. ``compute`` surfaces the
    pre-aggregated daily column the runner placed on the panel; the raw
    bars-based compute lives in :func:`compute_intraday_mmp_ew` (bound in
    ``factors.compute.minute.binding``).

    expected_ic_sign=+1: fixed from the I5d primary-universe (CSI500) quintile
    direction — the high-MMP group (Q5) outperformed Q1 (Q5-Q1 +17.15% on the
    corrected engine). The factor is EXPLORATORY and on hold: I5e's CSI300
    generalization was negative and I5d's monotonicity partly attributed to the
    adjustment correction itself, so the sign is a pre-registered hypothesis,
    not a validated conclusion.
    """

    name: str = MMP_EW_FACTOR_NAME

    spec = FactorSpec(
        factor_id=MMP_EW_FACTOR_NAME,
        version="1.0",
        description=(
            "Minute Microstructure Pressure, equal-weight session score: 1min "
            "bars PIT-truncated at 14:50 per bar, then the equal-weight mean of "
            "the session's valid per-minute MMP_t (rolling prior-20-bar "
            "baselines, in-session bars only). EXPLORATORY (I5c-I5e, on hold)."
        ),
        expected_ic_sign=+1,
        is_intraday=True,
        forward_return_horizon=1,
        return_basis="exec_to_exec",
        input_fields=("open", "high", "low", "close", "volume"),
        requires=_minute_requires("open", "high", "low", "close", "volume"),
        adjustment="returns_invariant",
        overnight_boundary="none",
        family="microstructure",
        # The first MMP_LOOKBACK in-session bars of every day are NaN by
        # construction — a WITHIN-day warm-up, not leading panel rows — so the
        # honest NaN rate is reported by data_coverage (jump_amount_corr
        # precedent), not hidden behind a fabricated warm-up window.
        min_history_bars=0,
        # The value at d reads ONLY d's own visible bars (per-session grouping
        # isolates the rolling baseline), so the transitive depth is the
        # materializer's floor: the signal day itself.
        lookback_depth=1,
        decision_cutoff="14:50:00",
        data_lag="1min",
        session_open="09:30:00",
        execution_model="next_minute_close",
        execution_window="[14:51:00,14:56:59]",
    )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily MMP column off ``panel``.

        The runner runs :func:`compute_intraday_mmp_ew` on the minute cache
        upstream and joins the result as ``self.name``; here we only surface
        it, so this factor does no temporal logic and cannot introduce
        lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"MmpEwFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_intraday_mmp_ew "
                f"and joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "DEFAULT_EPSILON",
    "MMP_EW_FACTOR_NAME",
    "MMP_LOOKBACK",
    "MmpEwFactor",
    "compute_intraday_mmp_ew",
    "compute_minute_mmp",
    "in_session_bars",
    "mmp_ew_daily",
    "mmp_valid_minute_counts",
]
