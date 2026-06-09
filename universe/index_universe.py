"""PITIndexUniverse: point-in-time index membership (fixes the UNI-003 downgrade).

Given periodic constituent snapshots (from :class:`IndexConstituentsFeed`), this
universe answers ``members(date)`` with the constituents of the LATEST snapshot on
or before ``date`` — an as-of join. That is what makes it point-in-time:

  * No look-ahead: a name that only joins the index at a FUTURE snapshot is never
    returned for an earlier date.
  * No survivorship bias: a name dropped from the index later is still returned
    for the dates when it was a member, so historical backtests see the real
    investable set of the time.

It consumes membership already pulled by the data layer (constituents DataFrame);
it does not touch tushare itself, so it is unit-testable with synthetic snapshots.
"""

from __future__ import annotations

import bisect

import pandas as pd

from universe.base import Universe
from universe.filters import apply_tradable_filters


class PITIndexUniverse(Universe):
    """A point-in-time universe backed by dated index-constituent snapshots."""

    def __init__(self, constituents: pd.DataFrame, filters: dict | None = None) -> None:
        if not {"date", "symbol"}.issubset(constituents.columns):
            raise ValueError(
                "constituents must have 'date' and 'symbol' columns "
                "(use IndexConstituentsFeed.get_constituents)."
            )
        c = constituents.copy()
        c["date"] = pd.to_datetime(c["date"]).dt.normalize()
        c["symbol"] = c["symbol"].astype(str)
        # snapshot date -> sorted unique symbols on that date
        self._by_date: dict[pd.Timestamp, list[str]] = {
            date: sorted(group["symbol"].unique().tolist())
            for date, group in c.groupby("date")
        }
        self._snapshots: list[pd.Timestamp] = sorted(self._by_date)
        self._filters = dict(filters or {})

    def membership_snapshots(self) -> dict[pd.Timestamp, list[str]]:
        """Return a copy of ``{snapshot_date: [symbols]}`` (read-only reporting).

        Exposes the raw PIT snapshots so reporting/analytics can summarise the
        universe (snapshot count, distinct names, churn) WITHOUT reaching into
        private state. It returns copies, so callers cannot mutate the universe.
        """
        return {date: list(symbols) for date, symbols in self._by_date.items()}

    def members(self, date: pd.Timestamp) -> list[str]:
        """Constituents of the latest snapshot on or before ``date`` (as-of)."""
        target = pd.Timestamp(date).normalize()
        # rightmost snapshot <= target
        idx = bisect.bisect_right(self._snapshots, target) - 1
        if idx < 0:
            return []  # before the first known snapshot
        return list(self._by_date[self._snapshots[idx]])

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]:
        """PIT members tradable on ``date`` via the shared tradability filters.

        Drops missing-close names (UNI-004) and, when the matching ``filters``
        toggle is on and the flag column is present, suspended / ST / at-limit
        names (P1 — :mod:`universe.filters`).
        """
        return apply_tradable_filters(self.members(date), date, panel, self._filters)
