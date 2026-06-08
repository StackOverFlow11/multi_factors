"""AlphaModel port: combine/predict a single score from a factor panel.

The alpha layer is the ONLY place allowed to see forward returns (for fitting
weights), and even then it must never pass them down to the factor layer
(CLAUDE.md invariant #1, ALPHA-004). ``EqualWeightAlpha`` (P0) ignores
forward_returns entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AlphaModel(ABC):
    """Abstract multi-factor combination / prediction model."""

    @abstractmethod
    def fit(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series | None = None,
    ) -> "AlphaModel":
        """Fit the model and return self (for chaining).

        Args:
            factors: MultiIndex(date, symbol) factor panel (training window).
            forward_returns: optional MultiIndex(date, symbol) future returns,
                used ONLY here to learn weights. Must never be forwarded to the
                factor layer. EqualWeightAlpha accepts ``None`` (ALPHA-003).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, factors_today: pd.DataFrame) -> pd.Series:
        """Predict a single score per symbol for one cross-section.

        Args:
            factors_today: factor rows for one date (MultiIndex or symbol-indexed).

        Returns:
            A pd.Series of scores indexed by symbol.
        """
        raise NotImplementedError
