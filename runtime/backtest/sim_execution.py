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
from runtime.fills import FillResult, simulate_fills


class SimExecution(BacktestExecution):
    """Simulated execution: tracks weights, turnover, trading cost, net return.

    Subclasses :class:`BacktestExecution` (itself an ``Execution``) so the
    strategy code above it is identical between backtest and live (INV-003),
    while the backtest-only ``settle`` / ``last_cost`` / ``last_turnover`` hooks
    live on the sub-port the driver depends on. Positions start empty; each
    ``rebalance_to`` records the turnover/cost of moving to the new target and
    replaces the book.
    """

    def __init__(self, fee_rate: float = 0.0) -> None:
        # NOTE (P2-2): idle-cash return is owned by the backtest driver (BT-007),
        # not this adapter, so SimExecution no longer takes a cash_return — settle()
        # returns the invested book's return only and the driver adds idle*cash.
        if fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {fee_rate!r}")
        self._fee_rate = float(fee_rate)
        self._positions: pd.Series = pd.Series(dtype=float)
        self._last_turnover: float = 0.0
        self._last_cost: float = 0.0
        self._last_return: float = 0.0
        self._last_fill: FillResult | None = None

    # -- properties ------------------------------------------------------- #
    @property
    def fee_rate(self) -> float:
        return self._fee_rate

    @property
    def last_turnover(self) -> float:
        """L1 turnover recorded by the most recent ``rebalance_to``."""
        return self._last_turnover

    @property
    def last_cost(self) -> float:
        """Trading cost (turnover * fee_rate) of the most recent rebalance."""
        return self._last_cost

    @property
    def last_fill(self) -> FillResult | None:
        """Feasibility diagnostics of the most recent rebalance (or None)."""
        return self._last_fill

    # -- Execution port --------------------------------------------------- #
    def rebalance_to(
        self,
        target_weights: pd.Series,
        date: pd.Timestamp,
        *,
        can_buy=None,
        can_sell=None,
    ) -> None:
        """Move the book toward ``target_weights`` as of the close of ``date``.

        Simulates only the FEASIBLE fills via :func:`runtime.fills.simulate_fills`
        (cash-coherent sell-then-buy): blocked trades carry forward and turnover/
        cost count only what actually executed. With ``can_buy``/``can_sell`` both
        ``None`` every trade is feasible, so the achieved book equals the target
        and the L1 turnover is identical to a naive rebalance (offline/demo path
        unchanged). Does not mutate the input; the new book is held from the next
        trading day (fixed event order).
        """
        target = self._clean_weights(target_weights)
        fill = simulate_fills(self._positions, target, can_buy, can_sell)
        self._last_fill = fill
        self._last_turnover = fill.executed_turnover
        self._last_cost = fill.executed_turnover * self._fee_rate
        # last_return is the cost-only return until the holding period settles.
        self._last_return = -self._last_cost
        self._positions = fill.achieved.copy()

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
            Net return of the INVESTED book = sum(book * returns) - cost. Symbols
            in the book but missing from ``holding_returns`` contribute zero (flat),
            and an empty book contributes zero invested return. The idle-cash
            return on the uninvested fraction ``1 - sum(book)`` is owned by the
            backtest driver (which honours its own ``cash_return``, BT-007), not by
            this adapter. Stored as ``last_return``.
        """
        book = self._positions
        if book.empty:
            gross = 0.0
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
