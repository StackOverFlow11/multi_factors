"""Intraday tail-rebalance execution semantics (I4) — SEPARATE from daily.

The daily backtest decides at T's close and measures holding returns
close(T)->close(T+1) (``runtime/backtest`` + ``settle(holding_returns)``). A
14:50 tail rebalance is a DIFFERENT event model and must NOT silently reuse that:

    signal cutoff       T 14:50:00  — the decision may use only bars with
                                       available_time <= cutoff (I3's job).
    execution timestamp T 14:51:00  — the trade fills at the NEXT minute bar
                                       (``next_minute_close`` selects WHICH bar),
                                       i.e. on data that is INTENTIONALLY after
                                       the signal cutoff. WHAT price that bar
                                       fills at is the separate
                                       ``execution_price_basis`` decision below.
    holding period      exec(T) -> exec(T_next)  — return is measured from this
                                       rebalance's execution price to the next
                                       rebalance's execution price, NEVER
                                       close-to-close.

This module is a runtime/execution skeleton: pure functions + small frozen
dataclasses turning (target weights per rebalance date, 1min bars, exec config)
into execution prices, execution-to-execution holding returns, and an explainable
blocked log. It does NOT touch factors/alpha math, the daily backtest, or the
config models; a missing minute bar / no bar in the execution window / a bar
whose configured price basis is undefined produces a BLOCKED record — never a
silent fallback to another price (bar close or daily close). Daily EOD data
(``daily_basic`` etc.) is by construction not consulted here: only intraday bars
are read, so T-day EOD values cannot leak into a 14:50 decision/execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.clean.intraday_schema import validate_intraday_bars

# Only the conservative first model is implemented; the others are declared in
# the roadmap (02_minute_pit_semantics.md) and rejected with a clear message
# until they land, so a config can never silently fall back to a wrong model.
SUPPORTED_EXECUTION_MODELS: tuple[str, ...] = ("next_minute_close",)
_FUTURE_EXECUTION_MODELS: tuple[str, ...] = ("tail_vwap", "closing_call_proxy")

# WHICH bar is selected (``execution_model``) and WHAT price that bar fills at
# (``execution_price_basis``) are separate decisions.
#   bar_vwap  — the selected bar's ``amount / volume``: the volume-weighted mean
#               of EVERY trade printed in that minute, i.e. the honest proxy for
#               an order worked across the minute, and hard to move with a single
#               small print. This is the DEFAULT.
#   bar_close — the selected bar's close: a single tick (that minute's last
#               print). Kept only to reproduce the pre-VWAP ledger bit-for-bit.
# Both are RAW unadjusted prices: the intraday cache stores raw bars, and
# ``amount``/``volume`` are raw traded value / shares, so their ratio is raw too.
# No adjustment factor is applied here or downstream of here.
PRICE_BASIS_VWAP = "bar_vwap"
PRICE_BASIS_CLOSE = "bar_close"
SUPPORTED_PRICE_BASES: tuple[str, ...] = (PRICE_BASIS_VWAP, PRICE_BASIS_CLOSE)

# blocked reasons (explainable; never a silent daily-close substitution).
REASON_NO_BAR = "no_execution_bar"
REASON_MISSING_PRICE = "missing_price"


@dataclass(frozen=True)
class IntradayExecutionConfig:
    """A strategy's tail-rebalance execution declaration.

    ``decision_time`` is the signal cutoff (T 14:50). ``data_lag`` documents the
    feature-availability lag consumed by the I3 cutoff (``available_time =
    bar_end + data_lag``); it is NOT re-applied to execution pricing. Execution
    fills at the earliest 1min bar whose ``bar_end`` falls in
    ``[execution_window_start, execution_window_end]`` (default 14:51–14:56:59),
    at that bar's ``execution_price_basis`` price (default its VWAP).
    """

    decision_time: str = "14:50:00"
    data_lag: str = "1min"
    execution_model: str = "next_minute_close"
    execution_window: tuple[str, str] = ("14:51:00", "14:56:59")
    execution_price_basis: str = PRICE_BASIS_VWAP

    def __post_init__(self) -> None:
        if self.execution_price_basis not in SUPPORTED_PRICE_BASES:
            raise ValueError(
                f"Unsupported execution_price_basis {self.execution_price_basis!r}; "
                f"supported: {SUPPORTED_PRICE_BASES}."
            )
        if self.execution_model not in SUPPORTED_EXECUTION_MODELS:
            future = (
                " (planned, not yet implemented)"
                if self.execution_model in _FUTURE_EXECUTION_MODELS
                else ""
            )
            raise ValueError(
                f"Unsupported execution_model {self.execution_model!r}{future}; "
                f"supported: {SUPPORTED_EXECUTION_MODELS}."
            )
        start = pd.Timedelta(self.execution_window[0])
        end = pd.Timedelta(self.execution_window[1])
        if start > pd.Timedelta(self.decision_time) and start <= end:
            return
        raise ValueError(
            "execution_window must satisfy decision_time < start <= end "
            f"(got decision={self.decision_time}, window={self.execution_window})."
        )


@dataclass(frozen=True)
class ExecutionFill:
    """One symbol's execution outcome on one rebalance date (immutable).

    ``exec_price`` is what the trade PAYS (the configured price basis) and is what
    the I5b price-limit gate compares. ``limit_reference_price`` is the selected
    bar's RAW close, carried only so the gate can LABEL an opened limit minute
    (closed at the limit, traded through it) for diagnostics — it never decides
    feasibility. It is ``None`` when no bar was selected or its close is NaN.
    """

    symbol: str
    date: pd.Timestamp
    exec_time: pd.Timestamp | None
    exec_price: float | None
    blocked: bool
    reason: str | None = None
    limit_reference_price: float | None = None


@dataclass(frozen=True)
class TailRebalanceResult:
    """Outcome of an execution-priced tail-rebalance simulation (immutable)."""

    period_returns: pd.Series          # entry date -> gross portfolio return
    exec_prices: pd.DataFrame          # (date x symbol) execution prices
    holding_returns: pd.DataFrame      # (entry date x symbol) exec-to-exec return
    fills: list[ExecutionFill] = field(default_factory=list)

    @property
    def blocked(self) -> list[ExecutionFill]:
        """All blocked fills (no execution bar / missing price)."""
        return [f for f in self.fills if f.blocked]


def bar_execution_price(bar: pd.Series, basis: str) -> float | None:
    """The RAW fill price of one selected 1min ``bar``, or ``None`` if undefined.

    ``bar_vwap`` (default) returns ``amount / volume`` — the volume-weighted mean
    of every trade printed in that minute. It is UNDEFINED when ``volume`` or
    ``amount`` is missing/NaN/non-finite or non-positive (a minute with no traded
    shares has no traded average price). ``None`` means exactly that: the caller
    must BLOCK the fill. It must never be softened into the bar close, and never
    into a daily close — a silent price substitution is the one degradation this
    execution layer exists to prevent.

    ``bar_close`` returns that bar's close (a single tick) and reproduces the
    pre-VWAP behaviour bit-for-bit, including its exact NaN test.
    """
    if basis == PRICE_BASIS_CLOSE:
        price = bar["close"]
        return None if pd.isna(price) else float(price)
    if basis != PRICE_BASIS_VWAP:
        raise ValueError(
            f"Unsupported execution_price_basis {basis!r}; "
            f"supported: {SUPPORTED_PRICE_BASES}."
        )
    volume = float(bar["volume"]) if pd.notna(bar["volume"]) else float("nan")
    amount = float(bar["amount"]) if pd.notna(bar["amount"]) else float("nan")
    if not (np.isfinite(volume) and np.isfinite(amount)):
        return None
    if volume <= 0.0 or amount <= 0.0:
        return None
    return amount / volume


def resolve_fill(
    symbol: str,
    date: pd.Timestamp,
    day_bars: pd.DataFrame,
    cfg: IntradayExecutionConfig,
) -> ExecutionFill:
    """Resolve the execution fill for ``symbol`` on ``date`` from its 1min bars.

    ``day_bars`` are the 1min bars for THIS symbol on THIS date (must carry
    ``bar_end`` plus the columns the price basis reads). ``next_minute_close``
    takes the EARLIEST bar whose ``bar_end`` is in the execution window; a missing
    bar, or a bar whose ``execution_price_basis`` price is undefined (NaN close /
    non-positive or non-finite volume or amount for the VWAP), yields a BLOCKED
    fill with an explainable reason — never a fallback to another price.
    """
    day = pd.Timestamp(date).normalize()
    win_start = day + pd.Timedelta(cfg.execution_window[0])
    win_end = day + pd.Timedelta(cfg.execution_window[1])

    if day_bars is None or day_bars.empty:
        return ExecutionFill(symbol, day, None, None, True, REASON_NO_BAR)
    cand = day_bars[
        (day_bars["bar_end"] >= win_start) & (day_bars["bar_end"] <= win_end)
    ].sort_values("bar_end")
    if cand.empty:
        return ExecutionFill(symbol, day, None, None, True, REASON_NO_BAR)

    bar = cand.iloc[0]  # next_minute_close: earliest available bar in the window
    bar_end = pd.Timestamp(bar["bar_end"])
    # The raw close travels with the fill regardless of the price basis: the
    # price-limit gate needs the price that is comparable to a limit price, which
    # a VWAP is not (see IntradayTailEventModel).
    raw_close = bar_execution_price(bar, PRICE_BASIS_CLOSE)
    price = bar_execution_price(bar, cfg.execution_price_basis)
    if price is None:
        # Same block path/reason as a NaN close has always taken: the bar existed
        # (exec_time is kept) but has no usable price, so the name is untradable
        # this rebalance. No other price is substituted.
        return ExecutionFill(
            symbol, day, bar_end, None, True, REASON_MISSING_PRICE, raw_close
        )
    return ExecutionFill(symbol, day, bar_end, float(price), False, None, raw_close)


def build_execution_prices(
    bars: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    symbols: list[str],
    cfg: IntradayExecutionConfig,
) -> tuple[pd.DataFrame, list[ExecutionFill]]:
    """Execution price matrix (date x symbol) + the per-(date, symbol) fill log.

    Reads ONLY intraday bars (no daily/EOD data). A blocked fill leaves a NaN in
    the matrix and an explained entry in the returned fills list.
    """
    validate_intraday_bars(bars)
    dates = [pd.Timestamp(d).normalize() for d in rebalance_dates]
    syms = [str(s) for s in symbols]

    work = bars.reset_index()
    work["date"] = work["bar_end"].dt.normalize()

    fills: list[ExecutionFill] = []
    prices = pd.DataFrame(index=pd.Index(dates, name="date"), columns=syms, dtype=float)
    for date in dates:
        on_day = work[work["date"] == date]
        for sym in syms:
            day_bars = on_day[on_day["symbol"] == sym]
            fill = resolve_fill(sym, date, day_bars, cfg)
            fills.append(fill)
            prices.loc[date, sym] = (
                np.nan if fill.exec_price is None else fill.exec_price
            )
    return prices, fills


def simulate_tail_rebalance(
    weights_by_date: dict[pd.Timestamp, pd.Series],
    bars: pd.DataFrame,
    cfg: IntradayExecutionConfig | None = None,
) -> TailRebalanceResult:
    """Simulate a tail rebalance, pricing fills at execution timestamps.

    ``weights_by_date`` maps each rebalance date to its target weights
    (symbol-indexed). For each period [T, T_next] the per-symbol holding return is
    ``exec_price(T_next) / exec_price(T) - 1`` (execution-to-execution, NOT
    close-to-close), and the gross portfolio return is ``sum_sym w_T * r``. A
    symbol missing either execution price is BLOCKED — excluded from that period's
    return (its weight earns nothing, i.e. cash) and recorded in the fills log.
    The final rebalance date has no subsequent period and no return.
    """
    cfg = cfg or IntradayExecutionConfig()
    dates = sorted(weights_by_date)
    symbols = sorted({str(s) for w in weights_by_date.values() for s in w.index})
    prices, fills = build_execution_prices(bars, dates, symbols, cfg)

    period_returns: dict[pd.Timestamp, float] = {}
    holding_rows: dict[pd.Timestamp, dict[str, float]] = {}
    for i in range(len(dates) - 1):
        entry, exit_ = dates[i], dates[i + 1]
        weights = weights_by_date[dates[i]]
        p_entry, p_exit = prices.loc[entry], prices.loc[exit_]
        gross = 0.0
        per_symbol: dict[str, float] = {}
        for sym, wt in weights.items():
            a, b = p_entry.get(str(sym)), p_exit.get(str(sym))
            if pd.notna(a) and pd.notna(b) and a != 0:
                r = float(b) / float(a) - 1.0
                per_symbol[str(sym)] = r
                gross += float(wt) * r
            # else: blocked at entry or exit -> excluded (recorded in `fills`)
        period_returns[entry] = gross
        holding_rows[entry] = per_symbol

    pr = pd.Series(period_returns, dtype=float)
    pr.index.name = "date"
    holding = pd.DataFrame(holding_rows).T if holding_rows else pd.DataFrame()
    if not holding.empty:
        holding.index.name = "date"
    return TailRebalanceResult(
        period_returns=pr,
        exec_prices=prices,
        holding_returns=holding,
        fills=fills,
    )
