"""StaticUniverse: a fixed, configured stock universe (Slice 4, P0).

This is the P0 implementation of the :class:`~universe.base.Universe` port. It
holds a configured list of symbols and applies the only P0 tradability filter:
drop symbols whose ``close`` is missing/NaN on the cross-section date (UNI-004).

PIT DOWNGRADE (UNI-003) — IMPORTANT
-----------------------------------
``members(date)`` returns the configured symbol list *regardless of date*. It
does **not** reflect true historical index membership (point-in-time
constituents). Using a today-known list for past dates introduces survivorship /
look-ahead membership bias. This downgrade is intentional for P0 and MUST be
recorded in the bias audit / phase0 report; do not present this class as a real
PIT universe.
"""

from __future__ import annotations

import pandas as pd

from universe.base import Universe
from universe.filters import apply_tradable_filters


class StaticUniverse(Universe):
    """A universe with a fixed, date-independent membership list.

    Args:
        symbols: the configured universe symbols (tushare style, e.g.
            ``"000001.SZ"``). Stored as an immutable tuple; ``members`` returns a
            fresh list each call so callers cannot mutate internal state.
        filters: optional filter toggles (e.g. ``{"missing_close": True}``). Only
            the ``missing_close`` filter is active in P0; it is always applied so
            an empty/None mapping is fine. Other keys are accepted and reserved
            for P1 filters (suspended/st/limit) without changing behaviour now.
    """

    def __init__(self, symbols: list[str], filters: dict | None = None) -> None:
        self._symbols: tuple[str, ...] = tuple(str(s) for s in symbols)
        self._filters: dict = dict(filters) if filters else {}

    def members(self, date: pd.Timestamp) -> list[str]:
        """Return the configured symbols, ignoring ``date`` (PIT downgrade).

        See the module docstring: this does NOT reflect true historical index
        membership. The ``date`` argument is part of the :class:`Universe`
        contract but is deliberately unused here.
        """
        return list(self._symbols)

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]:
        """Return members tradable on ``date`` via the shared tradability filters.

        Always drops missing-close names (UNI-004); also drops suspended / ST /
        at-limit names when the matching ``filters`` toggle is on AND the panel
        carries the corresponding flag column (P1 — :mod:`universe.filters`). The
        result is a subset of ``members(date)`` preserving configured order and
        never raises on an empty/absent cross-section.
        """
        return apply_tradable_filters(self.members(date), date, panel, self._filters)
