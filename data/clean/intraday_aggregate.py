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

# Feature keys (selectable via the ``features`` argument); column NAMES below
# encode the cutoff and are derived from the time arguments.
INTRADAY_FEATURE_KEYS: tuple[str, ...] = (
    "ret",
    "realized_vol",
    "vwap",
    "last30m_ret",
)

DEFAULT_DECISION_TIME = "14:50:00"
DEFAULT_SESSION_OPEN = "09:30:00"
DEFAULT_LAST_WINDOW_MINUTES = 30


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
        return list(INTRADAY_FEATURE_KEYS)
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
    g: pd.DataFrame, decision_time: str, last_window_minutes: int
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

    return {
        "ret": ret,
        "realized_vol": realized_vol,
        "vwap": vwap,
        "last30m_ret": last30,
    }


def asof_daily_features(
    bars: pd.DataFrame,
    *,
    decision_time: str = DEFAULT_DECISION_TIME,
    session_open: str = DEFAULT_SESSION_OPEN,
    last_window_minutes: int = DEFAULT_LAST_WINDOW_MINUTES,
    features: list[str] | None = None,
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
        feats = _compute_group(g, decision_time, last_window_minutes)
        index_tuples.append((date, str(sym)))
        for key, col in zip(keys, colnames):
            data[col].append(feats[key])

    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.DataFrame(data, index=index)[colnames].sort_index()


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
