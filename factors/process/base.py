"""FactorProcessor port: cross-sectional pre-processing of a factor panel.

Processing is always *by date* (each cross-section independently): drop missing,
z-score standardize (P0); winsorize, neutralize (P1). Output keeps the canonical
MultiIndex(date, symbol) shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FactorProcessor(ABC):
    """Abstract cross-sectional factor processor."""

    @abstractmethod
    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Transform a factor panel cross-sectionally (per date).

        Args:
            factors: MultiIndex(date, symbol) frame; columns are factor names.

        Returns:
            A new MultiIndex(date, symbol) frame, same columns, each date's
            cross-section processed independently. A single-name or zero-variance
            cross-section must not raise (return NaN or 0 by documented rule).
            The input is not mutated.
        """
        raise NotImplementedError
