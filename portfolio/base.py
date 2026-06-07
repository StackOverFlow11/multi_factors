"""PortfolioConstructor port: turn scores into target weights.

Hard boundary (CLAUDE.md invariant #3, INV-002): the portfolio layer receives
only scores / current weights / constraints. It must NOT touch a data source or
place orders. P0 builds a long-only TopN equal-weight portfolio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PortfolioConstructor(ABC):
    """Abstract portfolio constructor."""

    @abstractmethod
    def build(
        self,
        scores: pd.Series,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Build target weights from alpha scores.

        Args:
            scores: symbol-indexed alpha scores for one cross-section. NaN scores
                are ignored (PF-003).
            current_weights: optional symbol-indexed current weights (for
                turnover-aware constructors, P1). P0 may ignore it.

        Returns:
            A symbol-indexed pd.Series of target weights, summing to ~1.0 when
            any candidate exists (PF-002), long-only (no negative weights, PF-009),
            empty when there are no candidates (PF-004). The input is not mutated.
        """
        raise NotImplementedError
