"""Minute -> daily PIT-safe aggregation (I3): 14:50 tail-rebalance features.

Turns normalized 1min bars (:mod:`data.clean.intraday_schema`) into a daily
``(date, symbol)`` feature table the factor layer can join. The single hard rule
(``02_minute_pit_semantics.md``): a bar may inform a decision iff
``available_time <= decision_time``. Concretely, for each bar the cutoff is its
own trading date plus ``decision_time`` (default ``14:50:00``), and ONLY bars
with ``available_time <= cutoff`` enter the aggregation.

Ordering matters and is enforced: the PIT filter runs on per-bar TIMESTAMPS
FIRST; only after filtering are the surviving bars grouped by ``(trade_date,
symbol)``. Bars are never date-normalized into a daily bucket before the cutoff
is applied — doing so would discard the very timestamps the filter needs, and a
post-14:50 bar could leak into a 14:50 decision.

There is no ``data_lag`` parameter here: ``available_time`` already bakes in the
lag (it was set to ``bar_end + data_lag`` at normalize time), so the cutoff
compares against ``available_time`` directly — passing a second lag would
double-count it.

Feature columns ENCODE the cutoff so an ambiguous name can never hide a leak:

    intraday_ret_0930_1450            close/open return, session open -> cutoff
    intraday_realized_vol_0930_1450   sqrt(sum of 1min squared log-returns)
    intraday_vwap_0930_1450           sum(amount) / sum(volume)
    intraday_last30m_ret_1420_1450    return over the last 30m before the cutoff

This module is DATA-layer only: it does not fetch, does not touch factors / alpha
/ portfolio / runtime, and never sees a token. Coarser intraday bars, if needed,
are DERIVED here from 1min via :func:`resample_intraday_bars` (never raw-fetched),
and a derived bar inherits ``available_time = max(source_1min.available_time)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_schema import (
    INTRADAY_CORE_COLUMNS,
    INTRADAY_INDEX_NAMES,
    SYMBOL_LEVEL,
    ensure_supported_freq,
    validate_intraday_bars,
)

DATE_LEVEL = "date"
DAILY_INDEX_NAMES: list[str] = [DATE_LEVEL, SYMBOL_LEVEL]

# All selectable feature keys (validated against the ``features`` argument and
# mirrored by qt.config.IntradayCfg.score_feature). Column NAMES below encode the
# cutoff and are derived from the time arguments.
INTRADAY_FEATURE_KEYS: tuple[str, ...] = (
    "ret",
    "realized_vol",
    "vwap",
    "last30m_ret",
    "mmp_ew",
)
# Default feature set when ``features`` is None — the original cheap four. ``mmp_ew``
# (the I5c rolling MMP factor) is selectable-only, so the default output columns and
# the cost of a no-args call stay EXACTLY as before (existing callers unchanged).
DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    "ret",
    "realized_vol",
    "vwap",
    "last30m_ret",
)

DEFAULT_DECISION_TIME = "14:50:00"
DEFAULT_SESSION_OPEN = "09:30:00"
DEFAULT_LAST_WINDOW_MINUTES = 30
# Minute Microstructure Pressure (MMP, I5c): rolling baseline window (prior bars
# t-MMP_LOOKBACK..t-1) and the default denominator epsilon. EXPLORATORY factor;
# the window is part of the factor definition, not a tuned parameter.
MMP_LOOKBACK = 20
DEFAULT_EPSILON = 1e-6

# Price-jump turnover-correlation factor (PR-C, Kaiyuan report §6). The daily
# value is a trailing-``JUMP_LOOKBACK_DAYS``-trading-day lagged correlation between
# the traded ``amount`` at price-JUMP minutes and the amount at the STRICTLY-next
# minute; a jump minute is one whose within-(symbol, day) amplitude z-score exceeds
# ``JUMP_Z``. These are part of the FACTOR DEFINITION (reproduced from the report),
# not tuned knobs. Requires at least ``JUMP_MIN_PAIRS`` jump-pairs in the window.
JUMP_LOOKBACK_DAYS = 20
JUMP_MIN_PAIRS = 10
JUMP_Z = 1.0
# One minute in seconds: the "strictly next minute" test. A gap != 60s (the lunch
# break, the session close, a missing bar) is NOT the next minute, so that jump
# contributes no pair — exactly the report's within-session adjacency rule.
_ONE_MINUTE_SECONDS = 60.0


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


def _hhmm(time_str: str) -> str:
    """'14:50:00' -> '1450' (label for a within-day time)."""
    total = int(pd.Timedelta(time_str).total_seconds() // 60)
    return f"{total // 60:02d}{total % 60:02d}"


def _hhmm_minus(time_str: str, minutes: int) -> str:
    """Label for ``time_str`` shifted back ``minutes`` (e.g. 14:50 - 30 -> '1420')."""
    total = int(pd.Timedelta(time_str).total_seconds() // 60) - int(minutes)
    return f"{total // 60:02d}{total % 60:02d}"


def _resolve_feature_keys(features: list[str] | None) -> list[str]:
    if features is None:
        return list(DEFAULT_FEATURE_KEYS)
    unknown = [f for f in features if f not in INTRADAY_FEATURE_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown intraday feature(s): {unknown}. "
            f"Known: {list(INTRADAY_FEATURE_KEYS)}."
        )
    return list(features)


def _column_name(
    key: str, session_open: str, decision_time: str, last_window_minutes: int
) -> str:
    o, c = _hhmm(session_open), _hhmm(decision_time)
    if key == "ret":
        return f"intraday_ret_{o}_{c}"
    if key == "realized_vol":
        return f"intraday_realized_vol_{o}_{c}"
    if key == "vwap":
        return f"intraday_vwap_{o}_{c}"
    if key == "last30m_ret":
        start = _hhmm_minus(decision_time, last_window_minutes)
        return f"intraday_last{int(last_window_minutes)}m_ret_{start}_{c}"
    if key == "mmp_ew":
        return f"intraday_mmp{MMP_LOOKBACK}_ew_{o}_{c}"
    raise ValueError(f"Unhandled intraday feature key: {key!r}.")


def _empty_daily(colnames: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    out = pd.DataFrame({c: pd.Series([], dtype=float) for c in colnames})
    out.index = index
    return out


def _compute_group(
    g: pd.DataFrame,
    decision_time: str,
    last_window_minutes: int,
    keys: list[str],
    epsilon: float,
    session_open: str,
) -> dict[str, float]:
    """Per-(date, symbol) features from already PIT-filtered, bar_end-sorted bars."""
    closes = g["close"].to_numpy(dtype=float)
    opens = g["open"].to_numpy(dtype=float)
    vols = g["volume"].to_numpy(dtype=float)
    amts = g["amount"].to_numpy(dtype=float)
    bar_end = g["bar_end"]
    trade_date = pd.Timestamp(bar_end.iloc[0]).normalize()
    cutoff = trade_date + pd.Timedelta(decision_time)
    window_start = cutoff - pd.Timedelta(minutes=int(last_window_minutes))

    first_open, last_close = opens[0], closes[-1]
    ret = (last_close / first_open - 1.0) if first_open else float("nan")

    if len(closes) >= 2 and np.all(closes > 0):
        logret = np.diff(np.log(closes))
        realized_vol = float(np.sqrt(np.sum(logret**2)))
    else:
        realized_vol = float("nan")

    tot_vol = float(np.nansum(vols))
    vwap = float(np.nansum(amts) / tot_vol) if tot_vol > 0 else float("nan")

    ref_mask = (bar_end <= window_start).to_numpy()
    if ref_mask.any():
        ref_close = closes[ref_mask][-1]
        last30 = (last_close / ref_close - 1.0) if ref_close else float("nan")
    else:
        last30 = float("nan")

    out = {
        "ret": ret,
        "realized_vol": realized_vol,
        "vwap": vwap,
        "last30m_ret": last30,
    }
    # MMP is the only rolling feature; compute it ONLY when requested (I5c).
    if "mmp_ew" in keys:
        out["mmp_ew"] = _mmp_ew_daily(g, epsilon, trade_date, session_open)
    return out


def _in_session(g: pd.DataFrame, trade_date: pd.Timestamp, session_open: str) -> pd.DataFrame:
    """Bars whose ``bar_end`` is on/after ``session_open`` (the MMP window lower bound).

    The MMP daily score aggregates over ``[session_open, decision_time]`` (the upper
    bound is the available_time cutoff already applied upstream). Restricting to the
    in-session bars keeps PRE-session bars out of BOTH the rolling baseline and the
    daily mean, so the first in-session bar correctly has no prior-20 baseline.
    """
    session_start = trade_date + pd.Timedelta(session_open)
    return g[g["bar_end"] >= session_start]


def _mmp_ew_daily(
    g: pd.DataFrame, epsilon: float, trade_date: pd.Timestamp, session_open: str
) -> float:
    """Equal-weight mean of valid per-minute ``MMP_t`` over one PIT-filtered group.

    ``g`` is one ``(date, symbol)`` session's visible bars, sorted by ``bar_end``;
    only the in-session bars (``bar_end >= session_open``) enter, so the rolling
    baseline starts at the session open and the first 20 in-session bars are NaN.
    Every valid minute ``MMP_t`` gets EQUAL weight (no extra volume weighting — the
    volume term already lives inside ``MMP_t``). No valid minute -> NaN.
    """
    gs = _in_session(g, trade_date, session_open)
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


def asof_daily_features(
    bars: pd.DataFrame,
    *,
    decision_time: str = DEFAULT_DECISION_TIME,
    session_open: str = DEFAULT_SESSION_OPEN,
    last_window_minutes: int = DEFAULT_LAST_WINDOW_MINUTES,
    features: list[str] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """PIT-safe daily features from normalized 1min ``bars``.

    For each bar the cutoff is ``bar's trade_date + decision_time``; only bars
    with ``available_time <= cutoff`` are used (the filter runs on timestamps
    BEFORE any daily grouping). Returns a ``MultiIndex(date, symbol)`` frame whose
    columns encode the cutoff. An empty input, or an input with no visible bar
    before any cutoff, yields a schema-shaped empty daily frame.
    """
    validate_intraday_bars(bars)
    keys = _resolve_feature_keys(features)
    colnames = [
        _column_name(k, session_open, decision_time, last_window_minutes)
        for k in keys
    ]

    if len(bars) == 0:
        return _empty_daily(colnames)

    work = bars.reset_index()
    work["trade_date"] = work["bar_end"].dt.normalize()
    cutoff = work["trade_date"] + pd.Timedelta(decision_time)
    # PIT filter FIRST (per-bar timestamps), THEN group by day.
    visible = work.loc[work["available_time"] <= cutoff].copy()
    if visible.empty:
        return _empty_daily(colnames)

    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"])
    index_tuples: list[tuple] = []
    data: dict[str, list[float]] = {c: [] for c in colnames}
    for (date, sym), g in visible.groupby(["trade_date", SYMBOL_LEVEL], sort=True):
        feats = _compute_group(
            g, decision_time, last_window_minutes, keys, epsilon, session_open
        )
        index_tuples.append((date, str(sym)))
        for key, col in zip(keys, colnames):
            data[col].append(feats[key])

    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.DataFrame(data, index=index)[colnames].sort_index()


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
        gs = _in_session(g, pd.Timestamp(date).normalize(), session_open)
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


def _empty_jump_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` jump-factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def compute_jump_amount_corr(
    bars: pd.DataFrame,
    *,
    lookback_days: int = JUMP_LOOKBACK_DAYS,
    min_pairs: int = JUMP_MIN_PAIRS,
    jump_z: float = JUMP_Z,
    name: str = "jump_amount_corr",
) -> pd.Series:
    """PIT-safe daily "price-jump turnover correlation" factor from 1min ``bars``.

    Reproduces the Kaiyuan report §6 factor (``价格跳跃成交额相关性``, full-A RankIC
    -10.23%): the trailing-``lookback_days``-trading-day lagged Pearson correlation
    between the traded ``amount`` at price-JUMP minutes and the amount at the
    STRICTLY-next minute. The daily value at ``(date d, symbol s)`` uses ONLY bars
    at dates ``<= d`` (the trailing window ending at d), so a factor value never
    sees a future bar (invariant #1); it is meant to trade close-to-close from d+1.

    Definition (LOCKED, per bar of one (symbol, day) session):
      * ``amplitude = (high - low) / open``           (guard open > 0, amount finite);
      * ``jump``     = within-(symbol, day) amplitude z-score (ddof=1) ``> jump_z``;
      * pair each jump minute ``t`` with the STRICTLY-next minute (same session,
        ``bar_end`` gap exactly 60s — this excludes the lunch break AND the close);
      * ``factor(s, d)`` = Pearson corr(amount[jump t], amount[t+1]) over ALL
        jump-pairs whose date is in the trailing ``lookback_days`` TRADING DAYS
        (the symbol's own minute-trading days) ending at d; NaN when fewer than
        ``min_pairs`` pairs fall in the window (or the correlation is undefined).

    Vectorized (no per-rebalance-date python loop): the trailing-window correlation
    is a rolling sum of per-day sufficient statistics (n, sum x, sum y, sum x^2,
    sum y^2, sum xy) over the trading-day axis, then Pearson's closed form. Rolling
    over ROWS of the per-day-sorted stats == a trailing window of trading days,
    because consecutive rows are consecutive trading days.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the
            grouping is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing trading-day window length (part of the definition).
        min_pairs: minimum jump-pairs in the window for a finite value.
        jump_z: within-day amplitude z-score threshold defining a jump minute.
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if min_pairs < 2:
        # Pearson correlation needs at least 2 points; below that it is undefined.
        raise ValueError(f"min_pairs must be >= 2; got {min_pairs!r}.")
    if len(bars) == 0:
        return _empty_jump_series(name)

    work = bars.reset_index()[
        [SYMBOL_LEVEL, "bar_end", "open", "high", "low", "amount"]
    ].copy()
    # Guard bad rows BEFORE anything else: a non-positive open makes the amplitude
    # meaningless and a non-finite amount would poison the correlation.
    work = work[(work["open"] > 0.0) & np.isfinite(work["amount"].to_numpy(dtype=float))]
    if work.empty:
        return _empty_jump_series(name)
    work[DATE_LEVEL] = work["bar_end"].dt.normalize()
    # Sort so the "strictly next minute" shift and the per-symbol trading-day
    # rolling both see bars in chronological order within each (symbol, day).
    work = work.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")

    work["amp"] = (work["high"] - work["low"]) / work["open"]
    by_session = work.groupby([SYMBOL_LEVEL, DATE_LEVEL], sort=False)
    mean_amp = by_session["amp"].transform("mean")
    std_amp = by_session["amp"].transform("std")  # ddof=1 (pandas default)
    zscore = (work["amp"] - mean_amp) / std_amp
    next_bar_end = by_session["bar_end"].shift(-1)
    amt_next = by_session["amount"].shift(-1)
    gap = (next_bar_end - work["bar_end"]).dt.total_seconds()
    is_jump = (zscore > jump_z) & (gap == _ONE_MINUTE_SECONDS)

    pairs = pd.DataFrame(
        {
            SYMBOL_LEVEL: work[SYMBOL_LEVEL].to_numpy(),
            DATE_LEVEL: work[DATE_LEVEL].to_numpy(),
            "x": work["amount"].to_numpy(dtype=float),
            "y": amt_next.to_numpy(dtype=float),
        }
    ).loc[is_jump.to_numpy()]

    # Per-(symbol, day) sufficient statistics of the jump-pairs.
    if pairs.empty:
        stats = pd.DataFrame(
            columns=["cnt", "sx", "sy", "sxx", "syy", "sxy"], dtype=float
        )
        stats.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=[SYMBOL_LEVEL, DATE_LEVEL],
        )
    else:
        pairs = pairs.assign(
            xx=pairs["x"] * pairs["x"],
            yy=pairs["y"] * pairs["y"],
            xy=pairs["x"] * pairs["y"],
        )
        stats = pairs.groupby([SYMBOL_LEVEL, DATE_LEVEL], sort=True).agg(
            cnt=("x", "size"),
            sx=("x", "sum"),
            sy=("y", "sum"),
            sxx=("xx", "sum"),
            syy=("yy", "sum"),
            sxy=("xy", "sum"),
        )

    # Trading-day axis = every (symbol, day) the symbol has minute bars on, so the
    # rolling window counts TRADING DAYS (days with no jump still occupy a row, as
    # zeros — they consume one of the trailing ``lookback_days`` slots).
    axis = (
        work[[SYMBOL_LEVEL, DATE_LEVEL]]
        .drop_duplicates()
        .sort_values([SYMBOL_LEVEL, DATE_LEVEL], kind="mergesort")
    )
    full_index = pd.MultiIndex.from_arrays(
        [axis[SYMBOL_LEVEL].to_numpy(), axis[DATE_LEVEL].to_numpy()],
        names=[SYMBOL_LEVEL, DATE_LEVEL],
    )
    dense = stats.reindex(full_index).fillna(0.0)
    rolled = (
        dense.groupby(level=SYMBOL_LEVEL, sort=False)
        .rolling(lookback_days, min_periods=1)
        .sum()
    )
    # groupby.rolling prepends the group key -> drop it, keep (symbol, date).
    rolled.index = rolled.index.droplevel(0)

    n = rolled["cnt"].to_numpy(dtype=float)
    sx = rolled["sx"].to_numpy(dtype=float)
    sy = rolled["sy"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n * rolled["sxy"].to_numpy(dtype=float) - sx * sy
        var_x = n * rolled["sxx"].to_numpy(dtype=float) - sx * sx
        var_y = n * rolled["syy"].to_numpy(dtype=float) - sy * sy
        den = np.sqrt(var_x * var_y)
        corr = np.where((n >= min_pairs) & (den > 0.0), cov / den, np.nan)
    corr = np.clip(corr, -1.0, 1.0)

    out = pd.Series(corr, index=rolled.index, name=name)
    out.index = out.index.set_names([SYMBOL_LEVEL, DATE_LEVEL])
    return out.reorder_levels([DATE_LEVEL, SYMBOL_LEVEL]).sort_index()


def resample_intraday_bars(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Derive coarser intraday bars from normalized 1min ``bars``.

    Coarser bars are DERIVED from 1min, never raw-fetched. Each coarse bar covers
    a ``freq``-aligned window ending at ``bar_end``; OHLC = first-open / max-high /
    min-low / last-close, volume/amount summed. Critically, the coarse bar's
    ``available_time = max(source_1min.available_time)`` — it is only usable once
    EVERY constituent 1min bar is available — and is NOT recomputed as
    ``bar_end + data_lag`` (which would understate availability). The result
    passes :func:`validate_intraday_bars`.
    """
    ensure_supported_freq(freq)
    validate_intraday_bars(bars)

    if len(bars) == 0:
        # empty 1min -> empty coarse (same schema, just relabelled freq)
        return bars.copy()

    # Bucket each 1min bar by its freq-aligned window (ceil bar_end to the grid),
    # then aggregate WITHIN the bucket from the SOURCE bars only: bar_start/bar_end/
    # available_time are min/max/max over the constituents, never the nominal grid
    # boundary. A partial bucket therefore ends at its real last 1min bar, keeping
    # available_time (= max source) >= bar_end and never claiming data it lacks.
    work = bars.reset_index().sort_values([SYMBOL_LEVEL, "bar_end"])
    work["bucket"] = work["bar_end"].dt.ceil(freq)
    grouped = work.groupby([SYMBOL_LEVEL, "bucket"], sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        bar_start=("bar_start", "min"),
        bar_end=("bar_end", "max"),
        available_time=("available_time", "max"),
    ).reset_index()
    grouped["freq"] = freq

    index = pd.MultiIndex.from_arrays(
        [grouped["bar_end"].to_numpy(), grouped[SYMBOL_LEVEL].astype(str).to_numpy()],
        names=INTRADAY_INDEX_NAMES,
    )
    ordered = [*INTRADAY_CORE_COLUMNS, "freq", "bar_start", "bar_end", "available_time"]
    out = grouped[ordered].copy()
    out.index = index
    out = out.sort_index()
    validate_intraday_bars(out)
    return out
