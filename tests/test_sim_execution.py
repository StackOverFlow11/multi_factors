"""Tests for SimExecution (backlog Slice 9; reqs BT-004, cost/turnover).

SimExecution is the backtest adapter for the Execution port. These tests pin the
turnover/cost/return math in isolation: turnover is the full L1 distance between
target and current weights (CONTRACTS / BT-004 turnover_formula = l1), cost is
``turnover * fee_rate``, and ``last_return`` is the gross holding return net of
that cost. No other slice is imported.
"""

from __future__ import annotations

import pandas as pd
import pytest

from runtime.execution import BacktestExecution, Execution
from runtime.backtest.sim_execution import SimExecution


def test_turnover_from_empty_position_is_one_for_full_investment():
    """Empty book -> fully invested target: turnover is the full L1 sum (= 1.0)."""
    exe = SimExecution(fee_rate=0.0)
    target = pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5})
    exe.rebalance_to(target, pd.Timestamp("2024-02-01"))
    assert exe.last_turnover == pytest.approx(1.0)


def test_turnover_between_positions_is_l1_half_or_documented():
    """Turnover between two books is the documented full L1 (sum of |Δw|)."""
    exe = SimExecution(fee_rate=0.0)
    first = pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5})
    exe.rebalance_to(first, pd.Timestamp("2024-02-01"))
    # Rotate fully out of 000002 into 000003: aligned over the union of symbols
    # {1,2,3}: |0.5-0.5| + |0.5-0| + |0-0.5| = 1.0  (full L1, per CONTRACTS).
    second = pd.Series({"000001.SZ": 0.5, "000003.SZ": 0.5})
    exe.rebalance_to(second, pd.Timestamp("2024-03-01"))
    assert exe.last_turnover == pytest.approx(1.0)


def test_cost_equals_turnover_times_fee_rate():
    """cost == turnover * fee_rate (BT-004)."""
    fee_rate = 0.001
    exe = SimExecution(fee_rate=fee_rate)
    target = pd.Series({"000001.SZ": 0.6, "000002.SZ": 0.4})
    exe.rebalance_to(target, pd.Timestamp("2024-02-01"))
    assert exe.last_cost == pytest.approx(exe.last_turnover * fee_rate)
    assert exe.last_cost == pytest.approx(1.0 * fee_rate)


def test_last_return_is_net_of_cost():
    """last_return == gross holding return - cost of forming the position."""
    fee_rate = 0.001
    exe = SimExecution(fee_rate=fee_rate)
    target = pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5})
    exe.rebalance_to(target, pd.Timestamp("2024-02-01"))
    # Inject the per-symbol gross holding returns for the period that follows.
    holding_returns = pd.Series({"000001.SZ": 0.10, "000002.SZ": 0.00})
    net = exe.settle(holding_returns)
    gross = 0.5 * 0.10 + 0.5 * 0.00  # = 0.05
    expected = gross - 1.0 * fee_rate
    assert net == pytest.approx(expected)
    assert exe.last_return() == pytest.approx(expected)


def test_positions_update_after_rebalance():
    """positions() reflects the latest target after rebalance_to."""
    exe = SimExecution(fee_rate=0.0)
    assert exe.positions().empty  # starts empty
    target = pd.Series({"000001.SZ": 0.3, "000005.SZ": 0.7})
    exe.rebalance_to(target, pd.Timestamp("2024-02-01"))
    pos = exe.positions()
    assert set(pos.index) == {"000001.SZ", "000005.SZ"}
    assert pos["000001.SZ"] == pytest.approx(0.3)
    assert pos["000005.SZ"] == pytest.approx(0.7)


def test_sim_execution_is_execution_subclass():
    """SimExecution honours the Execution port (INV-003)."""
    assert issubclass(SimExecution, Execution)


def test_sim_execution_is_backtest_execution_subclass():
    """SimExecution implements the backtest sub-port AND the base port (HIGH-2).

    The driver depends on ``settle`` / ``last_cost`` / ``last_turnover``; those
    must be guaranteed by a port, not just happen to exist on SimExecution.
    """
    sim = SimExecution(fee_rate=0.0)
    assert isinstance(sim, Execution)
    assert isinstance(sim, BacktestExecution)
    assert issubclass(BacktestExecution, Execution)
    # The backtest-only hooks are abstract on the sub-port (a live adapter that
    # implements only Execution is never forced to provide them).
    for member in ("settle", "last_cost", "last_turnover"):
        assert member in BacktestExecution.__abstractmethods__


def test_live_execution_port_stays_minimal():
    """The base Execution port exposes ONLY the three live methods (HIGH-2).

    Backtest-only members (settle / last_cost / last_turnover) must NOT be
    abstract on the live port — keeping 'backtest == live' honest (INV-003).
    """
    assert Execution.__abstractmethods__ == frozenset(
        {"rebalance_to", "positions", "last_return"}
    )
    for backtest_only in ("settle", "last_cost", "last_turnover"):
        assert backtest_only not in Execution.__abstractmethods__
