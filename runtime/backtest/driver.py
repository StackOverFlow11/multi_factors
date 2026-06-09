"""BacktestDriver: wires the strategy layers through the fixed event order.

The driver owns no strategy logic of its own; it is pure orchestration over
injected collaborators (Ports):

    universe     -> .members / .tradable(date, panel)
    scores       -> .get(date, symbols) -> symbol-indexed scores
    constructor  -> .build(scores, current_weights) -> target weights
    execution    -> BacktestExecution port (Execution + settle / last_cost /
                    last_turnover; the backtest-only sub-port)
    prices       -> canonical (date, symbol) panel with a ``close`` column

Fixed event order (CONTRACTS §6):

    compute factor at close of t -> rebalance after close of t -> hold from t+1

So each rebalance is settled against the NEXT holding period's close-to-close
return (BT-003), never the same-day return the factor already saw. Empty
tradable universe -> the book is cash and earns ``cash_return`` (BT-007).

Output (BT-005/006, framework_settings §7.9): a date-indexed DataFrame with
columns ``[nav, gross_return, cost, turnover, net_return]``.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from runtime.execution import BacktestExecution
from runtime.fills import feasibility_from_cross


class ScoresSource(Protocol):
    """Minimal scores port the driver depends on."""

    def get(self, date: pd.Timestamp, symbols: list[str]) -> pd.Series: ...


class UniversePort(Protocol):
    """Minimal universe port the driver depends on."""

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]: ...


class ConstructorPort(Protocol):
    """Minimal portfolio-constructor port the driver depends on."""

    def build(self, scores: pd.Series, current_weights: pd.Series | None = ...) -> pd.Series: ...


_NAV_COLUMNS = ["nav", "gross_return", "cost", "turnover", "net_return"]


class BacktestDriver:
    """Monthly-rebalanced, single-period-compounding backtest over a price panel."""

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
        self._universe = universe
        self._scores = scores
        self._constructor = constructor
        self._execution = execution
        self._prices = prices
        self._rebalance = rebalance
        self._fee_rate = float(fee_rate)
        self._initial_nav = float(initial_nav)
        self._cash_return = float(cash_return)
        self._feasibility_log: list[dict] = []
        self._holdings_log: list[dict] = []

    # -- calendar --------------------------------------------------------- #
    def _calendar(self) -> pd.DatetimeIndex:
        """Sorted unique trading dates present in the price panel."""
        dates = self._prices.index.get_level_values("date").unique()
        return pd.DatetimeIndex(sorted(dates))

    def rebalance_dates(self) -> list[pd.Timestamp]:
        """Last trading day of each month in the panel calendar (BT-001)."""
        cal = self._calendar()
        if len(cal) == 0:
            return []
        frame = pd.DataFrame({"date": cal})
        frame["ym"] = frame["date"].dt.to_period("M")
        last = frame.groupby("ym")["date"].max().sort_values()
        return list(last)

    # -- pricing ---------------------------------------------------------- #
    def _close_at(self, date: pd.Timestamp, symbol: str) -> float:
        """Close of ``symbol`` on ``date`` (NaN if missing)."""
        try:
            return float(self._prices.loc[(date, symbol), "close"])
        except KeyError:
            return float("nan")

    def _holding_returns(
        self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str]
    ) -> pd.Series:
        """Close-to-close gross return per symbol over the FORWARD window.

        ``start`` is the rebalance (close of t); ``end`` is the next rebalance
        date (or the final trading day). The position is held over (start, end],
        so this is the next holding period's return — never a window ending on
        ``start`` itself (BT-003). Symbols with a missing/zero start price get a
        flat (0.0) return rather than NaN, so the book stays well-defined.
        """
        out: dict[str, float] = {}
        for sym in symbols:
            start_px = self._close_at(start, sym)
            end_px = self._close_at(end, sym)
            if (
                start_px is None
                or end_px is None
                or pd.isna(start_px)
                or pd.isna(end_px)
                or start_px == 0.0
            ):
                out[sym] = 0.0
            else:
                out[sym] = end_px / start_px - 1.0
        return pd.Series(out, dtype=float)

    # -- run -------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        """Run the backtest and return the NAV table (BT-005/006)."""
        reb = self.rebalance_dates()
        cal = self._calendar()
        rows: list[dict] = []
        nav = self._initial_nav
        self._feasibility_log = []  # re-entrant: fresh per run
        self._holdings_log = []
        for i, date in enumerate(reb):
            end = reb[i + 1] if i + 1 < len(reb) else cal[-1]
            # A rebalance on the final trading day has no forward holding window
            # (end <= start) — settling it would emit a spurious zero-length
            # period (close[d]/close[d]-1 == 0). Skip it (BT-003 honesty).
            if end <= date:
                continue
            turnover, cost, gross, net = self._step(date, end)
            nav = nav * (1.0 + net)
            rows.append(
                {
                    "date": date,
                    "nav": nav,
                    "gross_return": gross,
                    "cost": cost,
                    "turnover": turnover,
                    "net_return": net,
                }
            )
        out = pd.DataFrame(rows, columns=["date", *_NAV_COLUMNS])
        return out.set_index("date")

    def _step(
        self, date: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[float, float, float, float]:
        """One rebalance: universe -> scores -> build -> feasible fill -> settle.

        Selection (``universe.tradable``) decides what the strategy WANTS; the
        execution-feasibility flags (limits / suspension / missing, read from the
        date's cross-section) decide what it can actually trade. Blocked trades
        carry forward and turnover/cost count only executed trades (no forced
        impossible trades). An empty tradable universe still routes through the
        fill so held names that turned untradeable are carried, not force-sold.

        Returns ``(turnover, cost, gross_return, net_return)`` for the period
        held from ``date`` (exclusive) to ``end`` (inclusive).
        """
        current = self._execution.positions()
        tradable = list(self._universe.tradable(date, self._prices))
        if tradable:
            scores = self._scores.get(date, tradable)
            target = self._constructor.build(scores, current)
        else:
            target = pd.Series(dtype=float)  # nothing to hold -> exit what we can

        symbols = sorted(set(current.index) | set(target.index))
        can_buy, can_sell = self._feasibility(date, symbols)
        self._execution.rebalance_to(target, date, can_buy=can_buy, can_sell=can_sell)

        achieved = self._execution.positions()
        holding = self._holding_returns(date, end, list(achieved.index))
        invested_net = self._execution.settle(holding)
        # The driver owns cash semantics (BT-007): the uninvested fraction —
        # nonzero when blocked/cash-starved buys left cash idle — earns the
        # driver's cash_return, even if the execution adapter's differs.
        idle = 1.0 - float(achieved.sum())
        net = invested_net + idle * self._cash_return
        turnover = self._execution.last_turnover
        cost = self._execution.last_cost
        gross = net + cost
        self._record_feasibility(date, achieved)
        self._record_holdings(date, achieved)
        return (turnover, cost, gross, net)

    # -- execution feasibility ------------------------------------------- #
    def _feasibility(
        self, date: pd.Timestamp, symbols: list[str]
    ) -> tuple[dict, dict]:
        """Per-symbol (can_buy, can_sell) from the date's cross-section."""
        if not symbols:
            return {}, {}
        try:
            cross = self._prices.xs(pd.Timestamp(date).normalize(), level="date")
        except KeyError:
            cross = self._prices.iloc[0:0]
        return feasibility_from_cross(cross, symbols)

    def _record_feasibility(self, date: pd.Timestamp, achieved: pd.Series) -> None:
        """Append the most recent fill's feasibility diagnostics to the log."""
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
        """Per-settled-rebalance execution-feasibility diagnostics (date-indexed).

        Columns: ``blocked_buys``, ``blocked_sells``, ``cash_constrained_buys``,
        ``carried`` (counts), ``executed_turnover``, ``invested`` (sum of achieved
        weights; ``1 - invested`` is idle cash). Aligns 1:1 with the NAV table's
        settled rebalance dates.
        """
        cols = [
            "blocked_buys", "blocked_sells", "cash_constrained_buys",
            "carried", "executed_turnover", "invested",
        ]
        if not self._feasibility_log:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="date"))
        return pd.DataFrame(self._feasibility_log).set_index("date")

    def _record_holdings(self, date: pd.Timestamp, achieved: pd.Series) -> None:
        """Record the ACHIEVED book (post-feasibility) held for this period."""
        for sym, weight in achieved.items():
            self._holdings_log.append(
                {"date": date, "symbol": str(sym), "weight": float(weight)}
            )

    def holdings_log(self) -> pd.DataFrame:
        """Per-settled-rebalance ACHIEVED holdings (long-form date,symbol,weight,rank).

        These are the ACTUAL positions held after execution feasibility — a name
        whose sell was blocked appears here carried at its old weight, and a name
        whose buy was blocked is absent. This is the auditable book, NOT the
        constructor's desired target (which can differ once a trade is blocked).
        Ranked by weight desc then symbol for determinism; aligns with the NAV /
        feasibility log's settled dates.
        """
        cols = ["date", "symbol", "weight", "rank"]
        if not self._holdings_log:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._holdings_log).sort_values(
            ["date", "weight", "symbol"], ascending=[True, False, True]
        )
        df["rank"] = df.groupby("date").cumcount() + 1
        return df.reset_index(drop=True)[cols]
