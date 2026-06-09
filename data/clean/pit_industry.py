"""Point-in-time as-of alignment of SW industry by membership intervals (P2-3).

The correctness boundary: industry neutralization must use the industry a stock
belonged to AS OF each trade_date, not its CURRENT (latest) tag. `stock_basic.industry`
gives only the current label; broadcasting it to historical dates leaks a future
reclassification into the past (a mild membership look-ahead in the neutralization
covariate).

`asof_industry` consumes per-symbol SW-L1 membership intervals
``{symbol: [(industry_name, in_date, out_date), ...]}`` (from
:meth:`data.feed.tushare_covariates.TushareCovariatesFeed.pit_sw_l1_intervals`,
which reads tushare ``index_member_all``) and returns, for each ``(trade_date,
symbol)``, the industry whose interval ``[in_date, out_date)`` covers that date —
the most recent ``in_date`` on the (rare) overlap. ``out_date`` ``None``/``NaT`` is
an open (still-active) membership. A date before any membership, or a symbol with
no intervals, yields ``NaN`` — a disclosed data gap that the neutralizer drops
(never a silent fallback to the current tag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def asof_industry(index: pd.MultiIndex, intervals: dict) -> pd.Series:
    """Return a ``(date, symbol)`` Series of the as-of SW industry name.

    Args:
        index: target MultiIndex(date, symbol) to align onto.
        intervals: ``{symbol: [(industry_name, in_date, out_date), ...]}`` where
            ``in_date``/``out_date`` are Timestamps (``out_date`` ``None``/``NaT``
            means still active). Membership covers ``in_date <= date < out_date``,
            so on a reclassification date the NEW industry wins (PIT-safe).

    Each row carries the covering interval's industry (latest ``in_date`` on
    overlap), or ``NaN`` if no interval covers that date / the symbol is unknown.
    """
    norm = _normalize_intervals(intervals)
    dates = index.get_level_values("date")
    symbols = index.get_level_values("symbol").astype(str)

    values: list = []
    for d, s in zip(dates, symbols):
        rows = norm.get(s)
        best_in = None
        best_name = np.nan
        if rows:
            for name, in_date, out_date in rows:
                if in_date <= d and (out_date is None or d < out_date):
                    if best_in is None or in_date > best_in:
                        best_in = in_date
                        best_name = name
        values.append(best_name)
    return pd.Series(values, index=index, name="industry")


def _normalize_intervals(intervals: dict) -> dict:
    """Coerce interval dates to Timestamps; missing in_date -> Timestamp.min (open start)."""
    norm: dict[str, list] = {}
    for sym, ivs in (intervals or {}).items():
        rows = []
        for name, in_date, out_date in ivs:
            in_ts = (
                pd.Timestamp(in_date)
                if in_date is not None and not pd.isna(in_date)
                else pd.Timestamp.min
            )
            out_ts = (
                pd.Timestamp(out_date)
                if out_date is not None and not pd.isna(out_date)
                else None
            )
            rows.append((name, in_ts, out_ts))
        norm[str(sym)] = rows
    return norm
