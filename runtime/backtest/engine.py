"""BacktestEngine: the shared event-driven backtest loop (I5a).

One loop, two event models. The engine owns the *ledger* — universe selection,
feasible fills, settlement, cash, NAV compounding, and the feasibility / holdings
/ event logs — and is agnostic to whether a period is priced daily close-to-close
or intraday execution-to-execution. The time-basis differences live entirely in
the injected :class:`EventModel`:

    model.holding_periods()                 -> the rebalance schedule (anchors);
    model.holding_returns(period, symbols)  -> per-symbol gross entry->exit return;
    model.feasibility(period, symbols)      -> per-symbol (can_buy, can_sell).

This is a behaviour-preserving extraction of the legacy ``BacktestDriver`` loop:
with a :class:`~runtime.backtest.event_models.DailyCloseEventModel` the engine
reproduces the accepted monthly close-to-close NAV / cost / turnover / feasibility
/ holdings byte-for-byte (locked by tests). The same loop, given an
``IntradayTailEventModel``, prices fills at 1min execution bars and measures
exec-to-exec returns — without duplicating any fill/cash/settlement logic.

Selection vs execution are deliberately separated (the I5a contract):
``universe.tradable(date, panel)`` decides what the strategy WANTS from the daily
selection panel; ``model.feasibility`` decides what it can actually trade at the
period's execution time. The daily panel is used ONLY for selection here; the
intraday model never prices off it.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from runtime.backtest.events import HoldingPeriod
from runtime.execution import BacktestExecution

_NAV_COLUMNS = ["nav", "gross_return", "cost", "turnover", "net_return"]
_EVENT_COLUMNS = [
    "date", "decision_ts", "execution_ts", "exit_date", "next_decision_ts",
]


class ScoresSource(Protocol):
    """Minimal scores port the engine depends on."""

    def get(self, date: pd.Timestamp, symbols: list[str]) -> pd.Series: ...


class UniversePort(Protocol):
    """Minimal universe port the engine depends on."""

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]: ...


class ConstructorPort(Protocol):
    """Minimal portfolio-constructor port the engine depends on."""

    def build(
        self, scores: pd.Series, current_weights: pd.Series | None = ...
    ) -> pd.Series: ...


class EventModel(Protocol):
    """Time-basis strategy: schedule + pricing + execution feasibility."""

    def holding_periods(self) -> list[HoldingPeriod]: ...

    def holding_returns(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> pd.Series: ...

    def feasibility(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> tuple[dict, dict]: ...


class BacktestEngine:
    """Single-period-compounding backtest over a sequence of holding periods."""

    def __init__(
        self,
        *,
        model: EventModel,
        universe: UniversePort,
        scores: ScoresSource,
        constructor: ConstructorPort,
        execution: BacktestExecution,
        selection_panel: pd.DataFrame,
        initial_nav: float = 1.0,
        cash_return: float = 0.0,
    ) -> None:
        self._model = model
        self._universe = universe
        self._scores = scores
        self._constructor = constructor
        self._execution = execution
        self._panel = selection_panel
        self._initial_nav = float(initial_nav)
        self._cash_return = float(cash_return)
        self._feasibility_log: list[dict] = []
        self._holdings_log: list[dict] = []
        self._event_log: list[dict] = []

    # -- run -------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        """Run the backtest and return the date-indexed NAV table.

        Columns ``[nav, gross_return, cost, turnover, net_return]`` — identical
        to the legacy driver. Re-entrant: each run rebuilds the logs.
        """
        periods = self._model.holding_periods()
        rows: list[dict] = []
        nav = self._initial_nav
        self._feasibility_log = []
        self._holdings_log = []
        self._event_log = []
        for i, period in enumerate(periods):
            turnover, cost, gross, net = self._step(period)
            nav = nav * (1.0 + net)
            rows.append(
                {
                    "date": period.date,
                    "nav": nav,
                    "gross_return": gross,
                    "cost": cost,
                    "turnover": turnover,
                    "net_return": net,
                }
            )
            next_decision = (
                periods[i + 1].decision_ts if i + 1 < len(periods) else pd.NaT
            )
            self._record_event(period, next_decision)
        out = pd.DataFrame(rows, columns=["date", *_NAV_COLUMNS])
        return out.set_index("date")

    def _step(
        self, period: HoldingPeriod
    ) -> tuple[float, float, float, float]:
        """One rebalance: universe -> scores -> build -> feasible fill -> settle.

        Returns ``(turnover, cost, gross_return, net_return)`` for ``period``.
        """
        current = self._execution.positions()
        tradable = list(self._universe.tradable(period.date, self._panel))
        if tradable:
            scores = self._scores.get(period.date, tradable)
            target = self._constructor.build(scores, current)
        else:
            target = pd.Series(dtype=float)  # nothing to hold -> exit what we can

        symbols = sorted(set(current.index) | set(target.index))
        can_buy, can_sell = self._model.feasibility(period, symbols)
        self._execution.rebalance_to(
            target, period.date, can_buy=can_buy, can_sell=can_sell
        )

        achieved = self._execution.positions()
        holding = self._model.holding_returns(period, list(achieved.index))
        invested_net = self._execution.settle(holding)
        # The engine owns cash semantics (BT-007): the uninvested fraction earns
        # the configured cash_return, even if the execution adapter's differs.
        idle = 1.0 - float(achieved.sum())
        net = invested_net + idle * self._cash_return
        turnover = self._execution.last_turnover
        cost = self._execution.last_cost
        gross = net + cost
        self._record_feasibility(period.date, achieved)
        self._record_holdings(period.date, achieved)
        return (turnover, cost, gross, net)

    # -- feasibility log -------------------------------------------------- #
    def _record_feasibility(self, date: pd.Timestamp, achieved: pd.Series) -> None:
        fill = getattr(self._execution, "last_fill", None)
        if fill is None:
            return
        self._feasibility_log.append(
            {
                "date": date,
                "blocked_buys": len(fill.blocked_buys),
                "blocked_sells": len(fill.blocked_sells),
                "cash_constrained_buys": len(fill.cash_constrained_buys),
                "carried": len(fill.carried),
                "executed_turnover": float(fill.executed_turnover),
                "invested": float(achieved.sum()),
            }
        )

    def feasibility_log(self) -> pd.DataFrame:
        """Per-settled-rebalance execution-feasibility diagnostics (date-indexed)."""
        cols = [
            "blocked_buys", "blocked_sells", "cash_constrained_buys",
            "carried", "executed_turnover", "invested",
        ]
        if not self._feasibility_log:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="date"))
        return pd.DataFrame(self._feasibility_log).set_index("date")

    # -- holdings log ----------------------------------------------------- #
    def _record_holdings(self, date: pd.Timestamp, achieved: pd.Series) -> None:
        for sym, weight in achieved.items():
            self._holdings_log.append(
                {"date": date, "symbol": str(sym), "weight": float(weight)}
            )

    def holdings_log(self) -> pd.DataFrame:
        """Per-settled-rebalance ACHIEVED holdings (long-form date,symbol,weight,rank)."""
        cols = ["date", "symbol", "weight", "rank"]
        if not self._holdings_log:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._holdings_log).sort_values(
            ["date", "weight", "symbol"], ascending=[True, False, True]
        )
        df["rank"] = df.groupby("date").cumcount() + 1
        return df.reset_index(drop=True)[cols]

    # -- event log (I5a auditability) ------------------------------------- #
    def _record_event(
        self, period: HoldingPeriod, next_decision_ts: pd.Timestamp
    ) -> None:
        self._event_log.append(
            {
                "date": period.date,
                "decision_ts": period.decision_ts,
                "execution_ts": period.execution_ts,
                "exit_date": period.exit_date,
                "next_decision_ts": next_decision_ts,
            }
        )

    def event_log(self) -> pd.DataFrame:
        """Per-settled-period event timing (decision/execution/exit anchors)."""
        if not self._event_log:
            return pd.DataFrame(columns=_EVENT_COLUMNS).set_index("date")
        return pd.DataFrame(self._event_log).set_index("date")
