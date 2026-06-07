"""Portfolio constructors: turn alpha scores into target weights.

P0 ships ``TopNEqualWeight`` — long-only, equal-weight the N highest-scoring
symbols. Hard boundary (CLAUDE.md invariant #3, INV-002): this layer receives
only scores / current weights / constraints; it never touches a data source or
places orders. The input Series is never mutated (immutable style).
"""

from __future__ import annotations

import pandas as pd

from portfolio.base import PortfolioConstructor


class TopNEqualWeight(PortfolioConstructor):
    """Equal-weight the ``top_n`` highest-scoring symbols (long-only, P0).

    Behaviour:
        - drop NaN scores (PF-003);
        - select the ``top_n`` highest remaining scores (PF-001);
        - assign equal weight ``1/k`` so weights sum to 1 (PF-002), where ``k``
          is the number actually selected;
        - if fewer candidates than ``top_n`` exist, equal-weight the actual
          count (still sums to 1, PF-005);
        - if no candidates exist, return an EMPTY Series (no crash, PF-004);
        - long-only: never emit a negative weight (PF-009).
    """

    def __init__(self, top_n: int, long_only: bool = True) -> None:
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            raise ValueError(f"top_n must be an int, got {type(top_n).__name__}")
        if top_n <= 0:
            raise ValueError(f"top_n must be a positive integer, got {top_n}")
        self.top_n = top_n
        self.long_only = long_only

    def build(
        self,
        scores: pd.Series,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Build long-only equal-weight target weights from alpha scores.

        Args:
            scores: symbol-indexed alpha scores for one cross-section. NaN scores
                are ignored.
            current_weights: ignored in P0 (reserved for turnover-aware P1
                constructors); kept for interface compatibility.

        Returns:
            A symbol-indexed pd.Series of target weights summing to ~1.0 when any
            candidate exists, empty when there are none. The input is not mutated.
        """
        # Drop NaN scores without mutating the input (PF-003).
        candidates = scores.dropna()

        if candidates.empty:
            # No candidates -> empty weights (PF-004). Preserve index name/dtype.
            return pd.Series(dtype=float, index=candidates.index[:0], name="weight")

        # Select the top_n highest scores. ``nlargest`` clamps to the available
        # count, so fewer-than-N candidates are handled naturally (PF-005). It is
        # deterministic w.r.t. ties (keeps first occurrence) for reproducibility.
        selected = candidates.nlargest(self.top_n)

        k = len(selected)
        weight = 1.0 / k  # equal weight; sums to exactly 1 for k positions (PF-002)
        result = pd.Series(weight, index=selected.index, name="weight")
        result.index.name = scores.index.name
        return result
