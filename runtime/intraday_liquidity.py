"""Report-only intraday execution liquidity diagnostics (I5f).

For each desired rebalance trade at a 14:50 tail rebalance — the change
``target_weight - current_weight`` the engine actually planned — estimate whether
the SELECTED execution-minute 1min bar's traded ``amount`` (RMB) can absorb the
trade at a max participation rate. This is a DIAGNOSTIC ONLY: it never changes
fills, ``can_buy``/``can_sell``, blocked reasons, target weights, achieved
holdings, turnover, cost, NAV, factor scores, MMP grouping, alpha, or portfolio
construction. The intraday-tail runner builds it AFTER the backtest has settled,
from logs the backtest already produced.

Capacity model — execution-minute only (no future bar, no daily ``amount`` /
``volume`` / ``close``, no EOD proxy):

    desired_notional      = |target_weight - current_weight| * portfolio_notional
    bar_capacity_notional = execution_minute_amount * max_participation_rate
    capacity_ratio        = bar_capacity_notional / desired_notional

``capacity_ratio >= 1`` means the bar (at the participation cap) covers the desired
trade; ``< 1`` flags a potentially liquidity-constrained trade. Rules:

  * a zero desired trade (``|delta| == 0``) is not a trade and is skipped (no
    division by zero);
  * a trade whose direction was already blocked by the existing I5a/I5b execution
    feasibility (missing bar, missing price, raw ``stk_limit`` up/down limit) is
    counted as EXCLUDED and keeps its ORIGINAL block reason — it is NEVER
    reclassified as a liquidity block, and never gets a capacity ratio;
  * a missing / NaN / non-positive execution-minute ``amount`` on an otherwise
    executable trade is reported as MISSING capacity data, never inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from runtime.intraday_execution import ExecutionFill

_PLAN_COLUMNS = ["date", "symbol", "target_weight", "current_weight"]


# --------------------------------------------------------------------------- #
# Small, separable, unit-testable primitives
# --------------------------------------------------------------------------- #
def trade_direction(target_weight: float, current_weight: float) -> str | None:
    """``"buy"`` if the weight rises, ``"sell"`` if it falls, ``None`` if flat."""
    delta = float(target_weight) - float(current_weight)
    if delta > 0.0:
        return "buy"
    if delta < 0.0:
        return "sell"
    return None


def desired_trade_notional(
    target_weight: float, current_weight: float, portfolio_notional: float
) -> float:
    """``|target - current| * portfolio_notional`` — the RMB to trade this rebalance."""
    return abs(float(target_weight) - float(current_weight)) * float(portfolio_notional)


def bar_capacity_notional(
    amount: float | None, max_participation_rate: float
) -> float | None:
    """Execution-minute RMB capacity at the participation cap.

    Returns ``None`` when ``amount`` is missing / NaN / non-positive — that is
    MISSING capacity data, not zero capacity, and must never be inferred.
    """
    if amount is None:
        return None
    a = float(amount)
    if not np.isfinite(a) or a <= 0.0:
        return None
    return a * float(max_participation_rate)


def capacity_ratio(desired_notional: float, capacity_notional: float) -> float:
    """``capacity_notional / desired_notional`` (caller guarantees desired > 0)."""
    d = float(desired_notional)
    if d <= 0.0:
        raise ValueError(
            "capacity_ratio requires a positive desired_notional (a zero desired "
            "trade is not a trade and must be skipped upstream)."
        )
    return float(capacity_notional) / d


# --------------------------------------------------------------------------- #
# Result containers (immutable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LiquidityTrade:
    """One desired rebalance trade's liquidity diagnostic (immutable, report-only)."""

    date: pd.Timestamp
    symbol: str
    direction: str  # "buy" | "sell"
    desired_notional: float
    bar_capacity_notional: float | None  # None when capacity data is missing/blocked
    capacity_ratio: float | None         # None when blocked or capacity data missing
    feasibility_blocked: bool
    block_reason: str | None             # original execution block reason (kept as-is)
    missing_capacity_data: bool


@dataclass(frozen=True)
class LiquidityDiagnostics:
    """Aggregate report-only liquidity diagnostics over all desired trades."""

    portfolio_notional: float
    max_participation_rate: float
    total_desired_trades: int
    feasibility_blocked: int
    missing_capacity_rows: int
    inspected: int                       # executable trades with a usable capacity ratio
    below_capacity: int                  # inspected trades with capacity_ratio < 1
    ratio_stats: dict[str, float]        # min / p10 / median / p90 over inspected
    top_constrained: tuple[LiquidityTrade, ...]
    trades: tuple[LiquidityTrade, ...]


# --------------------------------------------------------------------------- #
# Execution-minute amount lookup (selected bar only — never a future/daily value)
# --------------------------------------------------------------------------- #
def _exec_amounts(
    bars: pd.DataFrame, fills: list[ExecutionFill]
) -> dict[tuple[pd.Timestamp, str], float]:
    """``{(norm date, symbol): execution-minute amount}`` for non-blocked fills.

    The amount is read at the EXACT selected execution bar (``bar_end ==
    exec_time``), so it is the execution-minute traded value — never a future bar,
    never a daily total. Absent rows resolve to NaN and are handled as missing data
    by the caller.
    """
    keys: list[tuple[pd.Timestamp, str]] = []
    bar_index: list[tuple[pd.Timestamp, str]] = []
    for f in fills:
        if f.blocked or f.exec_time is None:
            continue
        keys.append((pd.Timestamp(f.date).normalize(), str(f.symbol)))
        bar_index.append((pd.Timestamp(f.exec_time), str(f.symbol)))
    if not keys:
        return {}
    amounts = bars["amount"].reindex(
        pd.MultiIndex.from_tuples(bar_index, names=list(bars.index.names))
    ).to_numpy()
    return {k: amounts[i] for i, k in enumerate(keys)}


def _block_status(
    date: pd.Timestamp,
    symbol: str,
    direction: str,
    fill: ExecutionFill | None,
    up_blocked_buy_keys: set,
    down_blocked_sell_keys: set,
) -> tuple[bool, str | None]:
    """(feasibility_blocked, original_reason) for one desired trade direction.

    A missing/NaN execution bar (blocked fill) blocks BOTH directions and keeps its
    fill reason; otherwise a buy is blocked iff its (date, symbol) is in the raw
    up-limit set, a sell iff it is in the raw down-limit set. The reason is the
    ORIGINAL execution reason — never reclassified to a liquidity block.
    """
    if fill is not None and fill.blocked:
        return True, fill.reason
    key = (date, symbol)
    if direction == "buy" and key in up_blocked_buy_keys:
        return True, "up_limit"
    if direction == "sell" and key in down_blocked_sell_keys:
        return True, "down_limit"
    return False, None


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_liquidity_diagnostics(
    *,
    plan_log: pd.DataFrame,
    fills: list[ExecutionFill],
    up_blocked_buy_keys: set,
    down_blocked_sell_keys: set,
    bars: pd.DataFrame,
    portfolio_notional: float,
    max_participation_rate: float,
    top_n: int = 10,
) -> LiquidityDiagnostics:
    """Build the report-only liquidity diagnostics from settled backtest logs.

    ``plan_log`` is the engine's ``rebalance_plan_log()`` (date, symbol,
    target_weight, current_weight). For each desired trade (``|delta| > 0``): if the
    existing feasibility blocked that direction, count it as excluded (original
    reason kept); else read the selected execution-minute ``amount`` and compute the
    capacity ratio, flagging missing amount data. NEVER mutates any input or the
    backtest.
    """
    notional = float(portfolio_notional)
    part = float(max_participation_rate)
    amount_lookup = _exec_amounts(bars, fills)
    fill_map = {
        (pd.Timestamp(f.date).normalize(), str(f.symbol)): f for f in fills
    }

    trades: list[LiquidityTrade] = []
    feasibility_blocked = 0
    missing_capacity = 0
    inspected_ratios: list[float] = []

    if not plan_log.empty:
        for _, row in plan_log.iterrows():
            date = pd.Timestamp(row["date"]).normalize()
            symbol = str(row["symbol"])
            direction = trade_direction(row["target_weight"], row["current_weight"])
            if direction is None:
                continue  # zero desired trade -> not a trade -> skip (no div by zero)
            desired = desired_trade_notional(
                row["target_weight"], row["current_weight"], notional
            )
            fill = fill_map.get((date, symbol))
            blocked, reason = _block_status(
                date, symbol, direction, fill,
                up_blocked_buy_keys, down_blocked_sell_keys,
            )
            if blocked:
                feasibility_blocked += 1
                trades.append(
                    LiquidityTrade(
                        date=date, symbol=symbol, direction=direction,
                        desired_notional=desired, bar_capacity_notional=None,
                        capacity_ratio=None, feasibility_blocked=True,
                        block_reason=reason, missing_capacity_data=False,
                    )
                )
                continue
            amount = amount_lookup.get((date, symbol))
            cap = bar_capacity_notional(amount, part)
            if cap is None:
                missing_capacity += 1
                trades.append(
                    LiquidityTrade(
                        date=date, symbol=symbol, direction=direction,
                        desired_notional=desired, bar_capacity_notional=None,
                        capacity_ratio=None, feasibility_blocked=False,
                        block_reason=None, missing_capacity_data=True,
                    )
                )
                continue
            ratio = capacity_ratio(desired, cap)
            inspected_ratios.append(ratio)
            trades.append(
                LiquidityTrade(
                    date=date, symbol=symbol, direction=direction,
                    desired_notional=desired, bar_capacity_notional=cap,
                    capacity_ratio=ratio, feasibility_blocked=False,
                    block_reason=None, missing_capacity_data=False,
                )
            )

    below = sum(1 for r in inspected_ratios if r < 1.0)
    ratio_stats = _ratio_stats(inspected_ratios)
    inspected_trades = [t for t in trades if t.capacity_ratio is not None]
    top_constrained = sorted(
        inspected_trades,
        key=lambda t: (t.capacity_ratio, pd.Timestamp(t.date), t.symbol),
    )[: max(0, int(top_n))]

    return LiquidityDiagnostics(
        portfolio_notional=notional,
        max_participation_rate=part,
        total_desired_trades=len(trades),
        feasibility_blocked=feasibility_blocked,
        missing_capacity_rows=missing_capacity,
        inspected=len(inspected_ratios),
        below_capacity=below,
        ratio_stats=ratio_stats,
        top_constrained=tuple(top_constrained),
        trades=tuple(trades),
    )


def _ratio_stats(ratios: list[float]) -> dict[str, float]:
    """min / p10 / median / p90 over the inspected capacity ratios (NaN if empty)."""
    if not ratios:
        return {k: float("nan") for k in ("min", "p10", "median", "p90")}
    arr = np.asarray(ratios, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }
