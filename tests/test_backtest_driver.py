"""Tests for BacktestDriver (backlog Slice 10; reqs BT-001..007).

The driver wires injected collaborators (universe / scores source / portfolio
constructor / execution / price panel) through the fixed event order:

    compute factor at close of t -> rebalance after close of t -> hold from t+1

so the driver MUST settle each rebalance against the NEXT holding period's return
(BT-003), never the same-day return the factor already saw.

CRITICAL: no other slice is imported. All collaborators are small FAKES defined
here; only the foundation (runtime.execution, fixtures) is imported.
"""

from __future__ import annotations

import pandas as pd
import pytest

from runtime.execution import Execution
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution
from tests.fixtures.panel_factory import make_demo_panel


# --------------------------------------------------------------------------- #
# Small fakes (stand-ins for universe / scores / constructor collaborators).
# They expose only the Port methods the driver calls. Defined locally because
# the real slices are written in parallel and must not be imported.
# --------------------------------------------------------------------------- #
class FakeUniverse:
    """Returns a fixed tradable list regardless of date/panel."""

    def __init__(self, symbols: list[str]):
        self._symbols = list(symbols)

    def members(self, date: pd.Timestamp) -> list[str]:
        return list(self._symbols)

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]:
        return list(self._symbols)


class EmptyUniverse:
    """Always empty tradable set (exercises the cash path, BT-007)."""

    def members(self, date: pd.Timestamp) -> list[str]:
        return []

    def tradable(self, date: pd.Timestamp, panel: pd.DataFrame) -> list[str]:
        return []


class FakeConstructor:
    """Equal-weights whatever symbols are passed in via the scores index."""

    def build(self, scores: pd.Series, current_weights=None) -> pd.Series:
        valid = scores.dropna()
        if valid.empty:
            return pd.Series(dtype=float)
        n = len(valid)
        return pd.Series(1.0 / n, index=valid.index)


class FakeScores:
    """Scores source: equal score for the tradable symbols on each date."""

    def get(self, date: pd.Timestamp, symbols: list[str]) -> pd.Series:
        return pd.Series(1.0, index=list(symbols))


def _build_driver(universe, prices=None, fee_rate=0.0, cash_return=0.0):
    panel = make_demo_panel() if prices is None else prices
    return BacktestDriver(
        universe=universe,
        scores=FakeScores(),
        constructor=FakeConstructor(),
        execution=SimExecution(fee_rate=fee_rate),
        prices=panel,
        rebalance="monthly",
        fee_rate=fee_rate,
        initial_nav=1.0,
        cash_return=cash_return,
    )


def test_rebalance_dates_are_monthly():
    """Rebalance dates are the last trading day of each month in the calendar."""
    panel = make_demo_panel()
    driver = _build_driver(FakeUniverse(["000001.SZ"]), prices=panel)
    dates = driver.rebalance_dates()
    # 45 business days from 2024-01-01 reach 2024-03-01 -> 3 month-ends.
    months = sorted({(d.year, d.month) for d in dates})
    assert months == [(2024, 1), (2024, 2), (2024, 3)]
    # Each rebalance date must be the LAST trading day of its month in the panel.
    cal = panel.index.get_level_values("date").unique().sort_values()
    for d in dates:
        same_month = [c for c in cal if c.year == d.year and c.month == d.month]
        assert d == max(same_month)
    # All rebalance dates are real trading days in the calendar.
    assert set(dates).issubset(set(cal))


def test_backtest_outputs_nav_columns():
    """run() returns a date-indexed frame with the contract columns (BT-005/006)."""
    driver = _build_driver(FakeUniverse(["000001.SZ", "000003.SZ"]))
    out = driver.run()
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["nav", "gross_return", "cost", "turnover", "net_return"]
    assert out.index.name == "date"
    assert not out.empty
    # NAV compounds from initial_nav and is consistent with net_return.
    assert (out["nav"] > 0).all()
    prev = 1.0
    for _, row in out.iterrows():
        expected = prev * (1.0 + row["net_return"])
        assert abs(row["nav"] - expected) < 1e-9
        prev = row["nav"]


def test_backtest_uses_next_period_returns():
    """Holding return is the NEXT period's price change, not the same-day one (BT-003)."""
    panel = make_demo_panel()
    # 000001.SZ rises strictly +1/day; hold one symbol so gross == its return.
    driver = _build_driver(FakeUniverse(["000001.SZ"]), prices=panel, fee_rate=0.0)
    out = driver.run()
    cal = panel.index.get_level_values("date").unique().sort_values()
    close = panel.xs("000001.SZ", level="symbol")["close"]
    # For each rebalance row, the gross return must equal the close-to-close
    # change over the FORWARD window [rebalance_date -> next rebalance/end],
    # never a window that ends on the rebalance date itself.
    reb = list(out.index)
    for i, d in enumerate(reb):
        start_close = close.loc[d]
        if i + 1 < len(reb):
            end_date = reb[i + 1]
        else:
            end_date = max(cal)
        end_close = close.loc[end_date]
        expected = end_close / start_close - 1.0
        assert out.loc[d, "gross_return"] == pytest.approx(expected)
    # Sanity: a strictly-rising symbol earns strictly positive gross each period.
    assert (out["gross_return"] >= 0).all()


def test_backtest_handles_empty_weights():
    """Empty tradable universe -> earn cash_return, no turnover/cost (BT-007)."""
    cash = 0.001
    driver = _build_driver(EmptyUniverse(), fee_rate=0.001, cash_return=cash)
    out = driver.run()
    assert not out.empty
    assert (out["turnover"] == 0.0).all()
    assert (out["cost"] == 0.0).all()
    assert out["gross_return"].eq(cash).all()
    assert out["net_return"].eq(cash).all()
    # NAV grows by cash_return each period.
    prev = 1.0
    for _, row in out.iterrows():
        prev *= 1.0 + cash
        assert abs(row["nav"] - prev) < 1e-9


def test_backtest_applies_costs():
    """A non-zero fee_rate reduces NAV versus a zero-fee run (BT-004)."""
    universe = FakeUniverse(["000001.SZ", "000002.SZ", "000003.SZ"])
    panel = make_demo_panel()
    free = _build_driver(universe, prices=panel, fee_rate=0.0).run()
    charged = _build_driver(universe, prices=panel, fee_rate=0.01).run()
    # Costs are recorded and positive on rebalances that trade.
    assert (charged["cost"] > 0).any()
    # Net return is gross minus cost each period.
    for d in charged.index:
        assert charged.loc[d, "net_return"] == pytest.approx(
            charged.loc[d, "gross_return"] - charged.loc[d, "cost"]
        )
    # Final NAV with fees is strictly lower than without.
    assert charged["nav"].iloc[-1] < free["nav"].iloc[-1]


def test_execution_is_execution_port():
    """The injected SimExecution honours the shared Execution port (INV-003)."""
    assert isinstance(SimExecution(fee_rate=0.0), Execution)


def test_no_zero_length_holding_period_emitted():
    """A final rebalance on the last trading day emits no zero-length row (LOW).

    In the demo calendar the last month-end coincides with the last trading day,
    so its holding window (start, end] is empty (end <= start). Emitting it would
    add a spurious nav row whose gross_return is a meaningless 0.0. The driver
    must skip it, and no remaining row may have a zero-length holding window.
    """
    panel = make_demo_panel()
    cal = panel.index.get_level_values("date").unique().sort_values()
    driver = _build_driver(FakeUniverse(["000001.SZ"]), prices=panel)

    reb_all = driver.rebalance_dates()
    out = driver.run()

    # The last rebalance date equals the last trading day -> it is degenerate.
    assert reb_all[-1] == cal[-1]
    # That degenerate rebalance must NOT appear as a nav row.
    assert cal[-1] not in out.index
    # 000001.SZ rises strictly, so every EMITTED period has a positive (non-zero)
    # gross return -> no zero-length (gross == 0) holding period leaked in.
    assert (out["gross_return"] > 0).all()
