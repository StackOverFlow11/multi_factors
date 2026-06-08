"""Factor port: compute a cross-sectional feature from market bars.

NO-LOOKAHEAD RULE (CLAUDE.md invariant #1, INV-001):
    A factor value at date t may use ONLY bars at dates <= t. It must never
    read future prices and must never receive forward returns. Per the fixed
    event order (see CONTRACTS.md), ``momentum_20[t] = close[t]/close[t-20] - 1``
    is acceptable because rebalancing happens AFTER the close of t and holding
    starts the next trading day.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """Abstract cross-sectional factor.

    Subclasses set the class attribute ``name`` (used as the factor-panel column)
    and implement ``compute``.
    """

    name: str

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the factor over a canonical market panel.

        Args:
            panel: MultiIndex(date, symbol) market panel with CORE_COLUMNS.

        Returns:
            A pd.Series indexed by MultiIndex(date, symbol), aligned to ``panel``,
            with ``.name == self.name``. Early dates with an insufficient window
            yield NaN. Computation must be per-symbol (no cross-symbol leakage)
            and must use only current/past bars (no lookahead).
        """
        raise NotImplementedError
