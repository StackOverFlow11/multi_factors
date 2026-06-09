"""Execution port shared by backtest and live runtimes.

This is the seam that makes "backtest == live" (CLAUDE.md invariant #2, INV-003):
the same factor/alpha/portfolio code drives either a ``SimExecution`` (backtest)
or a live broker adapter, both implementing this single interface. Only the
adapter changes; the strategy code does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Execution(ABC):
    """Abstract execution port (backtest sim or live broker)."""

    @abstractmethod
    def rebalance_to(
        self,
        target_weights: pd.Series,
        date: pd.Timestamp,
        *,
        can_buy=None,
        can_sell=None,
    ) -> None:
        """Move the book toward ``target_weights`` as of ``date``.

        Args:
            target_weights: symbol-indexed target weights (sum ~1.0, long-only).
            date: the rebalance date. Per the fixed event order, this is the
                close of ``date``; the new position is held from the next trading
                day.
            can_buy / can_sell: optional per-symbol execution-feasibility maps
                (mapping/set/Series). When supplied, a backtest adapter simulates
                only the feasible fills (blocked trades carry forward) instead of
                assuming every trade executes; ``None`` -> all feasible (the
                offline/demo path is unchanged). A live adapter may use them as
                pre-trade checks or ignore them.

        A backtest adapter records turnover and trading cost here; a live adapter
        would emit orders. Returns None.
        """
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> pd.Series:
        """Return current symbol-indexed weights (after the last rebalance)."""
        raise NotImplementedError

    @abstractmethod
    def last_return(self) -> float:
        """Return the net (after-cost) portfolio return of the last holding period."""
        raise NotImplementedError


class BacktestExecution(Execution):
    """Backtest-only extension of the :class:`Execution` port.

    A historical simulator needs hooks a live broker does not: it *settles* a
    book against per-symbol holding returns and exposes the per-rebalance cost /
    turnover it just incurred. Keeping these on a sub-port (instead of the live
    ``Execution`` port) preserves "backtest == live" (CLAUDE.md invariant #2 /
    INV-003): the live port stays minimal, while the driver — which is a backtest
    concern — depends on this richer surface. A live adapter implements only the
    base ``Execution`` and is never asked to ``settle``.
    """

    @abstractmethod
    def settle(self, holding_returns: pd.Series | None) -> float:
        """Settle the current book against per-symbol gross holding returns.

        Returns the net (after-cost) return of the just-held period.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_cost(self) -> float:
        """Trading cost (turnover * fee_rate) of the most recent rebalance."""
        raise NotImplementedError

    @property
    @abstractmethod
    def last_turnover(self) -> float:
        """L1 turnover recorded by the most recent rebalance."""
        raise NotImplementedError
