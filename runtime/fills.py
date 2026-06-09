"""Execution-feasibility fill simulation (P2-2).

Turns a DESIRED target portfolio into the ACHIEVED book given per-symbol buy/sell
feasibility, simulating what a broker would actually fill. This is the seam where
"selection" (what the strategy wants) meets "execution feasibility" (what the
market lets you trade): limits, suspension, and missing prices block trades, and
the backtest must NOT silently pretend an impossible trade happened.

Cash-coherent sell-then-buy model (no leverage):

  1. ``cash = 1 - sum(current)`` (uninvested fraction).
  2. SELLS first: for each name the target reduces, if it can be sold, execute the
     reduction and add the proceeds to cash; otherwise carry the position forward
     (a blocked sell — a forced hold).
  3. BUYS next: the desired increases for buyable names are funded from available
     cash. If blocked sells starved the cash, the buys are scaled DOWN
     proportionally so the book never sums to more than 1 (no leverage). A name
     that cannot be bought is skipped (a blocked buy).
  4. Turnover counts only the trades actually executed.

Feasibility is a market reality, not a config toggle: ``can_buy`` / ``can_sell``
come from the panel flags (at-up-limit / at-down-limit / suspended / missing
close). A symbol absent from the maps defaults to feasible, so the offline demo
path (no flags) reduces to an exact ``achieved == target`` rebalance — P0/P1
behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_TOL = 1e-9


@dataclass(frozen=True)
class FillResult:
    """Outcome of one simulated rebalance (immutable)."""

    achieved: pd.Series
    executed_turnover: float
    blocked_buys: list[str] = field(default_factory=list)
    blocked_sells: list[str] = field(default_factory=list)
    cash_constrained_buys: list[str] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)

    @property
    def cash_constrained(self) -> bool:
        """True if any buyable name was scaled down for lack of cash."""
        return bool(self.cash_constrained_buys)


def _as_float_map(weights) -> dict[str, float]:
    """Coerce a Series/dict of weights to a clean ``{symbol: float}`` map."""
    if weights is None:
        return {}
    s = pd.Series(weights, dtype=float).dropna()
    return {str(k): float(v) for k, v in s.items()}


def _feasible(flag_map, symbol: str) -> bool:
    """Resolve a per-symbol bool flag; absent symbol -> feasible (True)."""
    if flag_map is None:
        return True
    if isinstance(flag_map, (set, frozenset)):
        return symbol in flag_map
    try:
        return bool(flag_map.get(symbol, True))
    except AttributeError:  # pandas Series or mapping-like without .get semantics
        return bool(flag_map[symbol]) if symbol in flag_map else True


def feasibility_from_cross(cross: pd.DataFrame, symbols) -> tuple[dict, dict]:
    """Derive per-symbol ``(can_buy, can_sell)`` from one date's cross-section.

    Feasibility is a market reality read off the panel flags, independent of the
    selection-filter config toggles:

      * a symbol with no bar that day (absent, or NaN ``close``) -> cannot trade;
      * ``suspended`` -> cannot trade either direction;
      * ``at_up_limit`` -> cannot BUY (price pinned at the ceiling);
      * ``at_down_limit`` -> cannot SELL (price pinned at the floor).

    Missing flag columns (e.g. the offline demo panel) default to not-flagged, so
    every present, non-NaN name is fully tradable and the fill reduces to an exact
    rebalance. Pure: never mutates ``cross``.
    """
    can_buy: dict[str, bool] = {}
    can_sell: dict[str, bool] = {}
    for sym in symbols:
        s = str(sym)
        if s not in cross.index:
            can_buy[s] = False
            can_sell[s] = False
            continue
        row = cross.loc[s]
        hard = bool(pd.isna(row.get("close", float("nan")))) or bool(
            row.get("suspended", False)
        )
        can_buy[s] = not (hard or bool(row.get("at_up_limit", False)))
        can_sell[s] = not (hard or bool(row.get("at_down_limit", False)))
    return can_buy, can_sell


def simulate_fills(
    current,
    target,
    can_buy=None,
    can_sell=None,
    *,
    tol: float = _TOL,
) -> FillResult:
    """Simulate feasible fills moving ``current`` toward ``target``.

    Args:
        current: symbol-indexed weights currently held (may sum to < 1; the rest
            is cash). Empty == an all-cash book.
        target: symbol-indexed desired weights (long-only; sums to ~1).
        can_buy: per-symbol buy feasibility (mapping/set/Series). Absent -> True.
        can_sell: per-symbol sell feasibility. Absent -> True.
        tol: numerical tolerance for "no change".

    Returns:
        A :class:`FillResult` with the achieved book, executed turnover, and the
        blocked-buy / blocked-sell / cash-constrained / carried diagnostics.
    """
    cur = _as_float_map(current)
    tgt = _as_float_map(target)
    symbols = sorted(set(cur) | set(tgt))

    achieved = {s: cur.get(s, 0.0) for s in symbols}
    cash = 1.0 - sum(cur.values())
    blocked_sells: list[str] = []
    blocked_buys: list[str] = []
    cash_constrained_buys: list[str] = []

    # 1) SELLS (target < current): feasible reductions free cash; blocked = hold.
    for s in symbols:
        delta = tgt.get(s, 0.0) - cur.get(s, 0.0)
        if delta < -tol:  # want to reduce / exit
            if _feasible(can_sell, s):
                achieved[s] = tgt.get(s, 0.0)
                cash += -delta  # proceeds of the executed sell
            else:
                blocked_sells.append(s)  # carried at cur[s]

    # 2) BUYS (target > current): fund from cash, scale down if starved.
    desired_buys = {
        s: tgt[s] - cur.get(s, 0.0)
        for s in symbols
        if tgt.get(s, 0.0) - cur.get(s, 0.0) > tol
    }
    buyable = {s: d for s, d in desired_buys.items() if _feasible(can_buy, s)}
    blocked_buys = [s for s in desired_buys if not _feasible(can_buy, s)]
    total_buy = sum(buyable.values())
    scale = 1.0
    if total_buy > tol and cash < total_buy - tol:
        scale = max(0.0, cash) / total_buy
    for s, d in buyable.items():
        exec_buy = d * scale
        if exec_buy > tol:
            achieved[s] = cur.get(s, 0.0) + exec_buy
            cash -= exec_buy
        # A buyable name scaled by a short cash budget (incl. fully starved -> 0)
        # is cash-constrained: this records the REASON (cash shortage); a starved
        # name also appears in `carried` below, which records the OUTCOME. The two
        # answer different questions, so the (intentional) overlap is fine.
        if scale < 1.0 - tol:
            cash_constrained_buys.append(s)

    achieved = {s: w for s, w in achieved.items() if w > tol}
    carried = [
        s for s in symbols if abs(achieved.get(s, 0.0) - tgt.get(s, 0.0)) > tol
    ]
    executed_turnover = sum(
        abs(achieved.get(s, 0.0) - cur.get(s, 0.0)) for s in symbols
    )

    book = pd.Series(achieved, dtype=float)
    book.index.name = "symbol"
    return FillResult(
        achieved=book.sort_index(),
        executed_turnover=float(executed_turnover),
        blocked_buys=sorted(blocked_buys),
        blocked_sells=sorted(blocked_sells),
        cash_constrained_buys=sorted(cash_constrained_buys),
        carried=sorted(carried),
    )
