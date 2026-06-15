"""Event models for the shared backtest engine (I5a).

Each model bundles a *schedule* with a *pricing* and *feasibility* time-basis:

  * :class:`DailyCloseEventModel` — the accepted monthly close-to-close model.
    Pricing and feasibility both read the daily panel; reproduces the legacy
    ``BacktestDriver`` ledger byte-for-byte.
  * :class:`IntradayTailEventModel` — a 14:50-decision / 14:51-execution tail
    rebalance. Pricing is the 1min execution-bar price (reusing
    :func:`runtime.intraday_execution.build_execution_prices`); returns are
    exec-to-exec; a missing/NaN execution bar is an explicit block, NEVER a
    daily-close fallback.

Both schedules come from :func:`runtime.backtest.events.monthly_anchor_pairs`, so
daily and intraday rebalance on the same calendar; only the time basis differs.
The intraday model never prices off the daily panel.
"""

from __future__ import annotations

import pandas as pd

from runtime.backtest.events import (
    HoldingPeriod,
    monthly_anchor_pairs,
    trading_calendar,
)
from runtime.fills import feasibility_from_cross
from runtime.intraday_execution import (
    ExecutionFill,
    IntradayExecutionConfig,
    build_execution_prices,
)


class DailyCloseEventModel:
    """Monthly close-to-close model — the accepted daily behaviour.

    decision = execution = entry = the rebalance date (close of T); exit = the
    next rebalance date (close of T_next). Holding returns and execution
    feasibility both read the daily ``prices`` panel, exactly as the legacy
    driver did.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        if "close" not in prices.columns:
            raise ValueError("price panel must have a 'close' column")
        self._prices = prices

    def holding_periods(self) -> list[HoldingPeriod]:
        pairs = monthly_anchor_pairs(trading_calendar(self._prices))
        return [
            HoldingPeriod(
                date=date,
                entry_date=date,
                exit_date=exit_date,
                decision_ts=date,
                execution_ts=date,
                exit_execution_ts=exit_date,  # priced at the exit date's close
            )
            for date, exit_date in pairs
        ]

    def _close_at(self, date: pd.Timestamp, symbol: str) -> float:
        try:
            return float(self._prices.loc[(date, symbol), "close"])
        except KeyError:
            return float("nan")

    def holding_returns(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> pd.Series:
        """Close-to-close gross return per symbol over (entry, exit].

        Symbols with a missing/zero start price get a flat (0.0) return rather
        than NaN, so the book stays well-defined (matches the legacy driver).
        """
        start, end = period.entry_date, period.exit_date
        out: dict[str, float] = {}
        for sym in symbols:
            start_px = self._close_at(start, sym)
            end_px = self._close_at(end, sym)
            if (
                pd.isna(start_px)
                or pd.isna(end_px)
                or start_px == 0.0
            ):
                out[sym] = 0.0
            else:
                out[sym] = end_px / start_px - 1.0
        return pd.Series(out, dtype=float)

    def feasibility(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> tuple[dict, dict]:
        """Per-symbol (can_buy, can_sell) from the rebalance date's cross-section."""
        if not symbols:
            return {}, {}
        try:
            cross = self._prices.xs(
                pd.Timestamp(period.date).normalize(), level="date"
            )
        except KeyError:
            cross = self._prices.iloc[0:0]
        return feasibility_from_cross(cross, symbols)


class IntradayTailEventModel:
    """14:50-decision / 14:51-execution tail rebalance over minute bars (I5a).

    The rebalance schedule is the same monthly calendar as the daily model, but
    each period's decision is timestamped at ``decision_time`` and its execution
    at the start of the execution window. Per-symbol entry/exit prices are the
    earliest valid 1min close in the execution window (``next_minute_close``);
    holding returns are ``exec_price(exit) / exec_price(entry) - 1``.

    Minimum I5a feasibility rule (goal §5): a symbol can be traded at the
    rebalance ONLY if it has a valid (non-NaN) execution bar at the entry anchor;
    a missing/NaN execution bar blocks BOTH directions. A suspended stock has no
    minute bars and is blocked by this rule. Price-limit feasibility at execution
    time is deferred (the daily panel carries only daily-close-derived limit
    flags, which §5 forbids using for execution) and disclosed in the report.
    """

    def __init__(
        self,
        *,
        calendar_panel: pd.DataFrame,
        bars: pd.DataFrame,
        cfg: IntradayExecutionConfig | None = None,
    ) -> None:
        self._calendar_panel = calendar_panel
        self._cfg = cfg or IntradayExecutionConfig()
        pairs = monthly_anchor_pairs(trading_calendar(calendar_panel))
        self._pairs = pairs
        # Anchor dates we must price: every rebalance date AND every exit date.
        anchor_dates = sorted(
            {d for pair in pairs for d in pair},
            key=lambda t: pd.Timestamp(t),
        )
        symbols = sorted({str(s) for s in bars.index.get_level_values("symbol")})
        self._symbols = symbols
        if anchor_dates and symbols:
            prices, fills = build_execution_prices(
                bars, anchor_dates, symbols, self._cfg
            )
        else:
            prices = pd.DataFrame()
            fills = []
        self._exec_prices = prices
        self._fills = fills

    def holding_periods(self) -> list[HoldingPeriod]:
        decision = pd.Timedelta(self._cfg.decision_time)
        execution = pd.Timedelta(self._cfg.execution_window[0])
        out: list[HoldingPeriod] = []
        for date, exit_date in self._pairs:
            day = pd.Timestamp(date).normalize()
            exit_day = pd.Timestamp(exit_date).normalize()
            out.append(
                HoldingPeriod(
                    date=date,
                    entry_date=date,
                    exit_date=exit_date,
                    decision_ts=day + decision,
                    execution_ts=day + execution,
                    # the book is priced out at the exit date's execution-window
                    # start (the planned exit fill time), NOT close-to-close.
                    exit_execution_ts=exit_day + execution,
                )
            )
        return out

    def _exec_price(self, date: pd.Timestamp, symbol: str) -> float:
        try:
            return float(self._exec_prices.loc[pd.Timestamp(date).normalize(), str(symbol)])
        except (KeyError, TypeError):
            return float("nan")

    def holding_returns(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> pd.Series:
        """exec-to-exec gross return per symbol over (entry, exit].

        A symbol missing an execution price at EITHER anchor is omitted (so the
        engine's ``settle`` treats it as flat — it earns nothing for the period).
        Never substitutes a daily close.
        """
        entry, exit_ = period.entry_date, period.exit_date
        out: dict[str, float] = {}
        for sym in symbols:
            a = self._exec_price(entry, sym)
            b = self._exec_price(exit_, sym)
            if pd.notna(a) and pd.notna(b) and a != 0.0:
                out[str(sym)] = b / a - 1.0
            # else: blocked at an anchor -> omitted -> flat via settle.
        return pd.Series(out, dtype=float)

    def feasibility(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> tuple[dict, dict]:
        """Per-symbol (can_buy, can_sell): tradable iff a valid entry exec bar."""
        can_buy: dict[str, bool] = {}
        can_sell: dict[str, bool] = {}
        for sym in symbols:
            ok = pd.notna(self._exec_price(period.entry_date, sym))
            can_buy[str(sym)] = bool(ok)
            can_sell[str(sym)] = bool(ok)
        return can_buy, can_sell

    # -- diagnostics ------------------------------------------------------ #
    def execution_prices(self) -> pd.DataFrame:
        """The (date x symbol) execution-price matrix (NaN = blocked)."""
        return self._exec_prices

    def fills(self) -> list[ExecutionFill]:
        """Per-(date, symbol) execution fill log (incl. blocked reasons)."""
        return list(self._fills)

    def blocked_fills(self) -> list[ExecutionFill]:
        """Only the blocked fills (no execution bar / NaN price)."""
        return [f for f in self._fills if f.blocked]
