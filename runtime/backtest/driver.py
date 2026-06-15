"""BacktestDriver: the daily close-to-close backtest (compatibility wrapper).

Since I5a the backtest ledger lives in :class:`runtime.backtest.engine.BacktestEngine`
and the daily time-basis in :class:`runtime.backtest.event_models.DailyCloseEventModel`.
``BacktestDriver`` is now a thin import-compatible wrapper around that pair: same
constructor, same ``run`` / ``rebalance_dates`` / ``feasibility_log`` /
``holdings_log`` surface, byte-identical output (locked by the daily golden tests
and the phase0 regression). New code can target the engine directly; existing
callers (``qt.pipeline``, ``qt.oos_stability``, ``qt.phase2_baseline``) are
untouched.

Fixed event order (CONTRACTS §6, preserved by ``DailyCloseEventModel``):

    compute factor at close of t -> rebalance after close of t -> hold from t+1

so each rebalance is settled against the NEXT holding period's close-to-close
return (BT-003). Empty tradable universe -> the book is cash and earns
``cash_return`` (BT-007).

Output (BT-005/006): a date-indexed DataFrame with columns
``[nav, gross_return, cost, turnover, net_return]``.
"""

from __future__ import annotations

import pandas as pd

from runtime.backtest.engine import (
    BacktestEngine,
    ConstructorPort,
    ScoresSource,
    UniversePort,
)
from runtime.backtest.event_models import DailyCloseEventModel
from runtime.backtest.events import monthly_rebalance_dates, trading_calendar
from runtime.execution import BacktestExecution


class BacktestDriver:
    """Monthly-rebalanced, single-period-compounding daily backtest.

    Compatibility wrapper over :class:`BacktestEngine` + :class:`DailyCloseEventModel`.
    """

    def __init__(
        self,
        *,
        universe: UniversePort,
        scores: ScoresSource,
        constructor: ConstructorPort,
        execution: BacktestExecution,
        prices: pd.DataFrame,
        rebalance: str = "monthly",
        fee_rate: float = 0.0,
        initial_nav: float = 1.0,
        cash_return: float = 0.0,
    ) -> None:
        if rebalance != "monthly":
            raise ValueError(
                f"only 'monthly' rebalance is supported in P0, got {rebalance!r}"
            )
        if "close" not in prices.columns:
            raise ValueError("price panel must have a 'close' column")
        self._prices = prices
        # fee_rate is carried for signature compatibility; the trading cost is
        # owned by the execution adapter (SimExecution.fee_rate), as before.
        self._fee_rate = float(fee_rate)
        # Keep the execution adapter reachable on the driver (the engine holds the
        # same object), preserving the legacy attribute surface for callers that
        # inspect the post-run book.
        self._execution = execution
        self._engine = BacktestEngine(
            model=DailyCloseEventModel(prices),
            universe=universe,
            scores=scores,
            constructor=constructor,
            execution=execution,
            selection_panel=prices,
            initial_nav=initial_nav,
            cash_return=cash_return,
        )

    def rebalance_dates(self) -> list[pd.Timestamp]:
        """Last trading day of each month in the panel calendar (BT-001)."""
        return monthly_rebalance_dates(trading_calendar(self._prices))

    def run(self) -> pd.DataFrame:
        """Run the backtest and return the NAV table (BT-005/006)."""
        return self._engine.run()

    def feasibility_log(self) -> pd.DataFrame:
        """Per-settled-rebalance execution-feasibility diagnostics (date-indexed)."""
        return self._engine.feasibility_log()

    def holdings_log(self) -> pd.DataFrame:
        """Per-settled-rebalance ACHIEVED holdings (long-form date,symbol,weight,rank)."""
        return self._engine.holdings_log()
