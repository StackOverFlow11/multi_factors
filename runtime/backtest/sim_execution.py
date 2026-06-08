"""SimExecution: the backtest adapter for the Execution port.

This is the simulated counterpart to a live broker (CLAUDE.md invariant #2,
INV-003): identical ``rebalance_to`` / ``positions`` / ``last_return`` surface,
so the strategy code above it is unchanged between backtest and live.

Math pinned by CONTRACTS / BT-004:

    turnover = sum(|target_w - current_w|)   # full L1, aligned over the union
    cost     = turnover * fee_rate
    net_return (of a holding period) = gross_portfolio_return - cost

The gross holding return is injected (``settle``) rather than read from a data
source here, which keeps the return/cost arithmetic unit-testable in isolation
and keeps this adapter free of any market-data dependency.
"""

from __future__ import annotations

import pandas as pd

from runtime.execution import BacktestExecution


class SimExecution(BacktestExecution):
    """Simulated execution: tracks weights, turnover, trading cost, net return.

    Subclasses :class:`BacktestExecution` (itself an ``Execution``) so the
    strategy code above it is identical between backtest and live (INV-003),
    while the backtest-only ``settle`` / ``last_cost`` / ``last_turnover`` hooks
    live on the sub-port the driver depends on. Positions start empty; each
    ``rebalance_to`` records the turnover/cost of moving to the new target and
    replaces the book.
    """

    def __init__(self, fee_rate: float = 0.0, cash_return: float = 0.0) -> None:
        if fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {fee_rate!r}")
        self._fee_rate = float(fee_rate)
        self._cash_return = float(cash_return)
        self._positions: pd.Series = pd.Series(dtype=float)
        self._last_turnover: float = 0.0
        self._last_cost: float = 0.0
        self._last_return: float = 0.0

    # -- properties ------------------------------------------------------- #
    @property
    def fee_rate(self) -> float:
        return self._fee_rate

    @property
    def cash_return(self) -> float:
        return self._cash_return

    @property
    def last_turnover(self) -> float:
        """L1 turnover recorded by the most recent ``rebalance_to``."""
        return self._last_turnover

    @property
    def last_cost(self) -> float:
        """Trading cost (turnover * fee_rate) of the most recent rebalance."""
        return self._last_cost

    # -- Execution port --------------------------------------------------- #
    def rebalance_to(self, target_weights: pd.Series, date: pd.Timestamp) -> None:
        """Move the book toward ``target_weights`` as of the close of ``date``.

        Computes full-L1 turnover against the current book over the union of
        symbols, stores the resulting cost, and replaces the current positions
        with a clean copy of the target. Does not mutate the input. The new book
        is held from the next trading day (fixed event order).
        """
        target = self._clean_weights(target_weights)
        current = self._positions
        union = current.index.union(target.index)
        aligned_target = target.reindex(union, fill_value=0.0)
        aligned_current = current.reindex(union, fill_value=0.0)
        turnover = float((aligned_target - aligned_current).abs().sum())
        self._last_turnover = turnover
        self._last_cost = turnover * self._fee_rate
        # Settle to the (gross) cost-only return for this rebalance until the
        # holding period is settled; keeps last_return defined right after a
        # rebalance with no holding info yet.
        self._last_return = -self._last_cost
        self._positions = target.copy()

    def positions(self) -> pd.Series:
        """Return current symbol-indexed weights (after the last rebalance)."""
        return self._positions.copy()

    def last_return(self) -> float:
        """Net (after-cost) portfolio return of the last settled holding period."""
        return self._last_return

    # -- BacktestExecution port (backtest-only, not on the live port) ------ #
    def settle(self, holding_returns: pd.Series | None) -> float:
        """Settle the current book against per-symbol gross holding returns.

        Args:
            holding_returns: symbol-indexed gross simple returns for the period
                the current book is held. ``None``/empty -> the book is cash, so
                the gross return is ``cash_return``.

        Returns:
            Net return = gross portfolio return - cost of forming this book.
            Symbols in the book but missing from ``holding_returns`` contribute
            zero (treated as flat). Stored as ``last_return``.
        """
        book = self._positions
        if book.empty:
            gross = self._cash_return
        else:
            if holding_returns is None:
                rets = pd.Series(dtype=float)
            else:
                rets = pd.Series(holding_returns, dtype=float)
            aligned = rets.reindex(book.index).fillna(0.0)
            gross = float((book * aligned).sum())
        net = gross - self._last_cost
        self._last_return = net
        return net

    # -- internals -------------------------------------------------------- #
    @staticmethod
    def _clean_weights(weights: pd.Series | None) -> pd.Series:
        """Coerce to a clean float Series; empty/None -> empty book (cash)."""
        if weights is None:
            return pd.Series(dtype=float)
        s = pd.Series(weights, dtype=float)
        return s.dropna()
