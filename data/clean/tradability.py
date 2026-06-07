"""Enrich a market panel with tradability flag columns.

Joins the signals from :class:`~data.feed.tushare_flags.TushareFlagsFeed` onto the
canonical (date, symbol) panel as boolean columns that
:func:`universe.filters.apply_tradable_filters` consults:

  * ``suspended``      — the (date, symbol) was halted that day (suspend_d 'S').
  * ``is_st``          — the name effective on that date contains 'ST' / '*ST'.
  * ``at_up_limit``    — close is at/above the day's upper price limit (can't buy).
  * ``at_down_limit``  — close is at/below the day's lower price limit (can't sell).

Only the flags whose source is supplied are added, so callers can enrich
incrementally and the offline demo path (no flag sources) is a no-op. Pure: never
mutates the input panel.
"""

from __future__ import annotations

import pandas as pd

from data.clean.schema import validate_panel

# close within this fraction of the limit counts as "at limit".
_LIMIT_TOL = 1e-6


def enrich_tradability(
    panel: pd.DataFrame,
    *,
    suspended: set[tuple] | None = None,
    st_intervals: dict[str, list[tuple]] | None = None,
    limits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a NEW panel with the supplied tradability flag columns added."""
    validate_panel(panel)
    out = panel.copy()
    dates = out.index.get_level_values("date")
    symbols = out.index.get_level_values("symbol")

    if suspended is not None:
        out["suspended"] = [(d, s) in suspended for d, s in zip(dates, symbols)]

    if st_intervals is not None:
        out["is_st"] = [_is_st(st_intervals, s, d) for d, s in zip(dates, symbols)]

    if limits is not None and not limits.empty:
        lim = limits.copy()
        lim["date"] = pd.to_datetime(lim["date"]).dt.normalize()
        lim["symbol"] = lim["symbol"].astype(str)
        lim = lim.set_index(["date", "symbol"])
        up = lim["up_limit"].reindex(out.index)
        down = lim["down_limit"].reindex(out.index)
        out["at_up_limit"] = (out["close"] >= up - _LIMIT_TOL) & up.notna()
        out["at_down_limit"] = (out["close"] <= down + _LIMIT_TOL) & down.notna()

    return out


def _is_st(
    intervals_by_symbol: dict[str, list[tuple]], symbol: str, date: pd.Timestamp
) -> bool:
    """Is ``symbol`` ST on ``date``? Uses the effective (latest-starting) name."""
    intervals = intervals_by_symbol.get(symbol)
    if not intervals:
        return False
    covering = [
        (start, is_st)
        for start, end, is_st in intervals
        if date >= start and (end is None or date <= end)
    ]
    if not covering:
        return False
    covering.sort(key=lambda item: item[0])  # effective name = latest start
    return bool(covering[-1][1])
