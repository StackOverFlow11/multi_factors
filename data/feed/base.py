"""DataFeed port: the only layer that touches a raw data source.

A DataFeed returns a market Panel and nothing else. It must NOT compute factors,
build portfolios, run backtests, or print/log any secret (tushare token).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataFeed(ABC):
    """Abstract market-data source.

    Implementations: ``DemoFeed`` (deterministic fixture), ``TushareFeed``
    (real API). Downstream layers depend on this interface, never on a concrete
    source (CLAUDE.md invariant #3).
    """

    @abstractmethod
    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Return a normalized market Panel for ``symbols`` over [start, end].

        Args:
            symbols: tushare-style codes, e.g. ["000001.SZ"].
            start, end: "YYYY-MM-DD" inclusive date bounds.
            freq: bar frequency; "D" for daily (P0). "1min" reserved (P1).

        Returns:
            A panel in canonical shape (see ``data.clean.schema``):
            MultiIndex(date, symbol) with at least CORE_COLUMNS.

        Must not:
            - compute factors / portfolios / backtests,
            - print or log the tushare token (SEC-001).
        """
        raise NotImplementedError
