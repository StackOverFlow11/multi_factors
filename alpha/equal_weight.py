"""EqualWeightAlpha: the P0 baseline multi-factor combiner.

Score = plain row-wise mean across the factor columns. No forward returns, no
learned weights (ALPHA-001/002/003). This is deliberately the simplest possible
alpha so the end-to-end pipeline can go green before any weighting model
(ICWeightAlpha, regression) is added.

Design (documented behavior):
- Single factor column  -> the score equals that column.
- Multiple columns      -> the equal-weight (arithmetic) mean of the columns.
- NaN-by-row            -> mean over the *available* (non-NaN) factors for that
  symbol; if *all* factors are NaN for a symbol, its score is NaN. This is the
  pandas ``DataFrame.mean(axis=1)`` default (``skipna=True``) and keeps a
  partially-observed symbol scorable instead of dropping it.
- Output is a symbol-indexed ``pd.Series``. A single-date MultiIndex
  cross-section is collapsed to a plain symbol index; an already symbol-indexed
  frame is passed through.

The model is stateless: ``fit`` only validates and returns ``self`` for
chaining. It never reads ``forward_returns`` and never forwards them anywhere
(CLAUDE.md invariant #1 / ALPHA-004 boundary).
"""

from __future__ import annotations

import pandas as pd

from alpha.base import AlphaModel

_SYMBOL_LEVEL = "symbol"


class EqualWeightAlpha(AlphaModel):
    """Combine factors into one score via an equal-weight row mean."""

    def fit(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series | None = None,
    ) -> "EqualWeightAlpha":
        """Validate inputs and return self.

        ``forward_returns`` is accepted for interface compatibility but is
        ignored entirely: the equal-weight baseline needs no future data
        (ALPHA-003) and must never leak it downstream (ALPHA-004).
        """
        self._check_factor_frame(factors)
        return self

    def predict(self, factors_today: pd.DataFrame) -> pd.Series:
        """Return a symbol-indexed score = row mean across factor columns.

        Args:
            factors_today: factor rows for one cross-section. Either a
                symbol-indexed frame, or a MultiIndex(date, symbol) frame whose
                rows all belong to a single date.

        Returns:
            ``pd.Series`` of scores indexed by symbol (name ``"score"``).
        """
        self._check_factor_frame(factors_today)
        if factors_today.shape[1] == 0:
            raise ValueError(
                "EqualWeightAlpha.predict needs at least one factor column; "
                "got an empty factor frame."
            )

        # skipna=True -> average over the available factors per row; a row that
        # is all-NaN yields NaN (documented behavior).
        scores = factors_today.mean(axis=1)
        scores = self._to_symbol_index(scores)
        return scores.rename("score")

    @staticmethod
    def _check_factor_frame(factors: pd.DataFrame) -> None:
        """Fail fast with a readable error on a non-DataFrame input."""
        if not isinstance(factors, pd.DataFrame):
            raise TypeError(
                "EqualWeightAlpha expects a pandas DataFrame of factors "
                f"(columns = factor names); got {type(factors).__name__}."
            )

    @staticmethod
    def _to_symbol_index(scores: pd.Series) -> pd.Series:
        """Collapse a (date, symbol) cross-section to a plain symbol index.

        A single-date MultiIndex is reduced by dropping the date level; an
        already symbol-indexed Series is returned unchanged.
        """
        index = scores.index
        if isinstance(index, pd.MultiIndex) and _SYMBOL_LEVEL in (index.names or []):
            symbols = index.get_level_values(_SYMBOL_LEVEL)
            out = pd.Series(scores.to_numpy(), index=symbols)
            out.index.name = _SYMBOL_LEVEL
            return out
        return scores
