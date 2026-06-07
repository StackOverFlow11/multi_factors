"""Universe port: which symbols exist and which are tradable on a given date.

The universe defines the cross-section. ``members`` is the (ideally PIT) index
membership; ``tradable`` narrows it to what can actually be traded that day
(missing price, suspended, ST, limit-locked). P0 ``StaticUniverse`` is a PIT
*downgrade* and must be documented as such (UNI-003).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Universe(ABC):
    """Abstract point-in-time stock universe."""

    @abstractmethod
    def members(self, date: pd.Timestamp) -> list[str]:
        """Return the symbols that are members of the universe on ``date``.

        For a true PIT universe this reflects historical index membership.
        StaticUniverse returns the configured list regardless of date (downgrade).
        """
        raise NotImplementedError

    @abstractmethod
    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]:
        """Return the subset of members that can be traded on ``date``.

        Args:
            date: the cross-section date.
            panel: canonical market panel (MultiIndex(date, symbol)) used to
                apply filters (missing close, suspended, ST, limit up/down).

        Returns:
            A list of tradable symbols; always a subset of ``members(date)``.
            May be empty (must not crash downstream).
        """
        raise NotImplementedError
