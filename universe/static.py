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
        """Return members that are tradable on ``date`` (UNI-004 missing_close).

        A member is tradable if it has a row at ``date`` in ``panel`` with a
        non-NaN ``close``. Members absent from the panel on that date, or with a
        missing close, are excluded. The result is always a subset of
        ``members(date)`` and preserves the configured symbol order. An empty
        result (no members, or none with a valid close) is returned cleanly and
        never raises.
        """
        members = self.members(date)
        if not members:
            return []

        valid_closes = self._valid_close_symbols(date, panel)
        return [symbol for symbol in members if symbol in valid_closes]

    @staticmethod
    def _valid_close_symbols(date: pd.Timestamp, panel: pd.DataFrame) -> set[str]:
        """Symbols with a non-NaN ``close`` at ``date`` in ``panel``.

        Reads only the cross-section at ``date`` from the canonical
        MultiIndex(date, symbol) panel. Returns an empty set if the date is
        absent or the panel has no ``close`` rows there, so callers never crash.
        """
        if "close" not in panel.columns:
            raise ValueError(
                "panel is missing the 'close' column required for the "
                "missing_close tradability filter"
            )

        norm_date = pd.Timestamp(date).normalize()
        date_level = panel.index.get_level_values("date")
        cross_section = panel.loc[date_level == norm_date, "close"]
        if cross_section.empty:
            return set()

        valid = cross_section.dropna()
        symbols = valid.index.get_level_values("symbol")
        return {str(symbol) for symbol in symbols}
