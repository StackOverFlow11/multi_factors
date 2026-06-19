"""Cross-sectional quantile grouping primitives for the I5d grouped backtest.

These are the small, pure pieces the grouped intraday-tail backtest
(:mod:`qt.intraday_group_backtest`) layers on TOP of the existing event-driven
machinery — they decide *which names land in which group* and present one group
to the shared :class:`~runtime.backtest.engine.BacktestEngine` as an equal-weight
target, WITHOUT touching the engine, the execution feasibility, or the factor
math:

  * :func:`assign_quantile_buckets` — split one scored cross-section into N
    EQUAL-COUNT rank buckets (Q1 = lowest score, QN = highest). Rank/position
    buckets (not value cuts) so tied or degenerate scores still produce a
    deterministic, auditable assignment; too few names simply leave high groups
    empty instead of crashing.
  * :class:`GroupScores` — a ``ScoresSource`` that exposes ONE group's members to
    the engine (1.0 for a member, NaN otherwise), so the engine's selection picks
    exactly that bucket.
  * :class:`EqualWeightAll` — a constructor that equal-weights EVERY non-NaN name
    (no ``top_n`` cap): the grouped backtest holds the whole bucket.

Hard boundary preserved (CLAUDE.md invariant #1/#3): nothing here reads a forward
return, a data source, or places an order. The MMP score is computed upstream
(PIT-safe, available_time <= 14:50) and only its CROSS-SECTIONAL RANK is used to
form groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.base import PortfolioConstructor


def assign_quantile_buckets(scores: pd.Series, n_groups: int) -> dict[str, int]:
    """Assign each scored symbol to one of ``n_groups`` EQUAL-COUNT rank buckets.

    Args:
        scores: symbol-indexed scores for ONE cross-section (one rebalance date).
        n_groups: number of quantile groups (e.g. 5).

    Returns:
        ``{symbol: label}`` with ``label`` in ``1..n_groups``, where **Q1 (label
        1) is the LOWEST score** and **QN (label n_groups) is the HIGHEST**.

    Semantics:
        * NaN / non-finite scores are dropped (never assigned to a group).
        * Symbols are ordered by ``(score ascending, symbol ascending)`` and split
          BY POSITION into ``n_groups`` contiguous chunks (``np.array_split``), so
          the assignment is deterministic even with tied scores (the symbol
          tie-break decides) — a value-cut quantile could not split ties cleanly.
        * When fewer than ``n_groups`` names are scored, the lower groups fill
          first and the high groups are simply empty (no crash). Chunk sizes
          differ by at most one; the extra goes to the lower groups (the
          ``np.array_split`` convention).
    """
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1; got {n_groups}.")
    if scores is None or len(scores) == 0:
        return {}
    s = pd.Series(scores, dtype=float).dropna()
    if s.empty:
        return {}
    finite_mask = np.isfinite(s.to_numpy(dtype=float))
    s = s[finite_mask]
    if s.empty:
        return {}
    order = sorted(s.index, key=lambda sym: (float(s.loc[sym]), str(sym)))
    buckets = np.array_split(np.array(order, dtype=object), n_groups)
    out: dict[str, int] = {}
    for i, bucket in enumerate(buckets):
        for sym in bucket:
            out[str(sym)] = i + 1
    return out


class GroupScores:
    """``ScoresSource`` exposing exactly ONE quantile group to the engine.

    Bridges the precomputed per-date bucket assignment to the ``ScoresSource``
    port :class:`~runtime.backtest.engine.BacktestEngine` depends on. ``get`` marks
    a symbol with a constant score ``1.0`` iff it is a member of ``group`` on that
    date, else ``NaN`` — so an equal-weight-all constructor selects exactly the
    bucket. It only READS a precomputed rank assignment (no forward returns, no
    data source), preserving the no-lookahead boundary.
    """

    def __init__(
        self, assignments: dict[pd.Timestamp, dict[str, int]], group: int
    ) -> None:
        # assignments: normalized rebalance date -> {symbol: group label}.
        self._assignments = assignments
        self._group = int(group)

    def get(self, date: pd.Timestamp, symbols: list[str]) -> pd.Series:
        """Return symbol-indexed membership scores (1.0 member / NaN non-member)."""
        norm = pd.Timestamp(date).normalize()
        members = self._assignments.get(norm, {})
        values = [
            1.0 if members.get(str(sym)) == self._group else np.nan
            for sym in symbols
        ]
        return pd.Series(values, index=list(symbols), dtype=float)


class EqualWeightAll(PortfolioConstructor):
    """Equal-weight EVERY non-NaN name in the cross-section (no ``top_n`` cap).

    The grouped backtest holds an entire quantile bucket, so — unlike
    :class:`portfolio.construct.TopNEqualWeight` — there is no top-N selection:
    every name the :class:`GroupScores` source surfaces (score ``1.0``) is held at
    weight ``1/k``. NaN scores are dropped (non-members); an empty cross-section
    yields an empty (all-cash) target. Long-only; the input is never mutated.
    """

    def __init__(self, long_only: bool = True) -> None:
        self.long_only = bool(long_only)

    def build(
        self,
        scores: pd.Series,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Equal-weight all non-NaN names; empty target when there are none."""
        candidates = scores.dropna()
        if candidates.empty:
            return pd.Series(dtype=float, index=candidates.index[:0], name="weight")
        weight = 1.0 / len(candidates)
        result = pd.Series(weight, index=candidates.index, name="weight")
        result.index.name = scores.index.name
        return result
