"""Execution-feasibility fill simulation (P2-2 core, pure + network-free).

``simulate_fills`` turns a DESIRED target into an ACHIEVED book given per-symbol
buy/sell feasibility, using a cash-coherent sell-then-buy model:

  * sells execute first (feasible ones), freeing cash;
  * buys are funded from available cash, scaled down proportionally if blocked
    sells starved them (no leverage — the book never sums to > 1);
  * blocked trades carry the current position forward;
  * turnover/cost count only the trades actually executed.

These tests pin every branch the backtest driver relies on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from runtime.fills import feasibility_from_cross, simulate_fills


def _s(d: dict) -> pd.Series:
    return pd.Series(d, dtype=float)


def _cross(rows: dict) -> pd.DataFrame:
    """Build a symbol-indexed cross-section from {symbol: {col: val}}."""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "symbol"
    return frame


def test_feasibility_from_cross_directional_flags():
    cross = _cross(
        {
            "OK": {"close": 10.0, "suspended": False, "at_up_limit": False, "at_down_limit": False},
            "UP": {"close": 10.0, "suspended": False, "at_up_limit": True, "at_down_limit": False},
            "DN": {"close": 10.0, "suspended": False, "at_up_limit": False, "at_down_limit": True},
            "SUS": {"close": 10.0, "suspended": True, "at_up_limit": False, "at_down_limit": False},
            "NAN": {"close": float("nan"), "suspended": False, "at_up_limit": False, "at_down_limit": False},
        }
    )
    can_buy, can_sell = feasibility_from_cross(cross, ["OK", "UP", "DN", "SUS", "NAN", "ABSENT"])
    assert can_buy["OK"] and can_sell["OK"]
    assert not can_buy["UP"] and can_sell["UP"]      # up-limit: no buy, can sell
    assert can_buy["DN"] and not can_sell["DN"]      # down-limit: can buy, no sell
    assert not can_buy["SUS"] and not can_sell["SUS"]  # suspended: neither
    assert not can_buy["NAN"] and not can_sell["NAN"]  # no price: neither
    assert not can_buy["ABSENT"] and not can_sell["ABSENT"]  # no bar: neither


def test_feasibility_from_cross_missing_flag_columns_default_feasible():
    cross = _cross({"A": {"close": 10.0}, "B": {"close": 10.0}})
    can_buy, can_sell = feasibility_from_cross(cross, ["A", "B"])
    assert all(can_buy.values()) and all(can_sell.values())


def test_all_feasible_achieves_target_exactly():
    # demo-equivalence: every trade feasible -> achieved == target, no leverage.
    current = _s({"A": 0.5, "B": 0.5})
    target = _s({"C": 0.5, "D": 0.5})
    res = simulate_fills(current, target)
    assert res.achieved.reindex(["C", "D"]).tolist() == [0.5, 0.5]
    assert "A" not in res.achieved.index and "B" not in res.achieved.index
    assert res.executed_turnover == pytest.approx(2.0)  # sold 1.0, bought 1.0
    assert not res.blocked_buys and not res.blocked_sells


def test_first_period_from_cash_buys_full_target():
    res = simulate_fills(_s({}), _s({"A": 0.5, "B": 0.5}))
    assert res.achieved.sum() == pytest.approx(1.0)
    assert res.executed_turnover == pytest.approx(1.0)


def test_blocked_buy_at_up_limit_is_not_added():
    # want to buy C but C is at up-limit (can't buy) -> C not added; cash held.
    current = _s({})
    target = _s({"C": 0.5, "D": 0.5})
    res = simulate_fills(current, target, can_buy={"C": False})
    assert "C" not in res.achieved.index
    assert res.achieved.get("D", 0.0) == pytest.approx(0.5)
    assert res.blocked_buys == ["C"]
    assert res.achieved.sum() == pytest.approx(0.5)  # rest is cash, no leverage


def test_blocked_sell_at_down_limit_carries_position():
    # want to exit A but A is at down-limit (can't sell) -> A carried forward.
    current = _s({"A": 0.5, "B": 0.5})
    target = _s({"B": 1.0})  # exit A, double B
    res = simulate_fills(current, target, can_sell={"A": False})
    assert res.achieved.get("A", 0.0) == pytest.approx(0.5)  # forced hold
    assert res.blocked_sells == ["A"]
    assert "A" in res.carried
    # no leverage: A's blocked sell freed no cash, so B can't grow past 0.5.
    assert res.achieved.sum() == pytest.approx(1.0)
    assert res.achieved.get("B", 0.0) == pytest.approx(0.5)
    assert res.cash_constrained is True


def test_suspended_blocks_both_directions():
    current = _s({"A": 0.5, "B": 0.5})
    target = _s({"A": 1.0})  # want to increase A, exit B
    res = simulate_fills(
        current, target, can_buy={"A": False}, can_sell={"B": False}
    )
    # A can't be bought (stays 0.5), B can't be sold (stays 0.5).
    assert res.achieved.get("A", 0.0) == pytest.approx(0.5)
    assert res.achieved.get("B", 0.0) == pytest.approx(0.5)
    assert "A" in res.blocked_buys and "B" in res.blocked_sells


def test_executed_turnover_counts_only_executed_trades():
    # exit A (blocked), buy C (feasible but cash-starved by A's blocked sell).
    current = _s({"A": 1.0})
    target = _s({"C": 1.0})
    res = simulate_fills(current, target, can_sell={"A": False})
    # A can't be sold -> no cash -> C can't be bought at all.
    assert res.achieved.get("A", 0.0) == pytest.approx(1.0)
    assert res.achieved.get("C", 0.0) == pytest.approx(0.0)
    assert res.executed_turnover == pytest.approx(0.0)  # nothing executed


def test_partial_fill_when_one_of_two_sells_blocked():
    # current A,B (0.5 each); target C,D (0.5 each). A can't sell, B can.
    current = _s({"A": 0.5, "B": 0.5})
    target = _s({"C": 0.5, "D": 0.5})
    res = simulate_fills(current, target, can_sell={"A": False})
    # B sold -> 0.5 cash. Buys C,D want 1.0 total but only 0.5 cash -> scaled 0.5x.
    assert res.achieved.get("A", 0.0) == pytest.approx(0.5)  # carried
    assert res.achieved.get("C", 0.0) == pytest.approx(0.25)
    assert res.achieved.get("D", 0.0) == pytest.approx(0.25)
    assert res.achieved.sum() == pytest.approx(1.0)  # 0.5 + 0.25 + 0.25, no leverage
    assert res.cash_constrained is True


def test_unknown_symbol_defaults_to_feasible():
    # a symbol absent from can_buy/can_sell maps to feasible (True).
    res = simulate_fills(_s({}), _s({"Z": 1.0}), can_buy={"Y": False})
    assert res.achieved.get("Z", 0.0) == pytest.approx(1.0)
