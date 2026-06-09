"""End-to-end execution-feasibility through the BacktestDriver (P2-2).

These drive a real :class:`BacktestDriver` over a tiny flagged panel and assert the
realism contract: a name at the down-limit cannot be sold (carried forward), a name
at the up-limit cannot be bought, suspended names cannot trade either way, and
turnover/cost count only the trades that actually executed — no forced impossible
trades. The per-rebalance feasibility log records the blocks.
"""

from __future__ import annotations

import pandas as pd

from portfolio.construct import TopNEqualWeight
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution

# Calendar spans Jan..early-Mar 2024 so the settled rebalances are Jan-end and
# Feb-end (the final March date is the terminal skip, BT-003).
_DATES = pd.bdate_range("2024-01-01", "2024-03-05")
_JAN, _FEB = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")


class _Universe:
    def members(self, date):
        return ["A", "B"]

    def tradable(self, date, panel):
        return ["A", "B"]


class _Scores:
    """A is top in January, B is top in February (forces an A->B switch)."""

    def get(self, date, symbols):
        d = pd.Timestamp(date).normalize()
        pref = "A" if d <= _JAN else "B"
        return pd.Series({s: (2.0 if s == pref else 1.0) for s in symbols})


def _panel(flags: dict | None = None) -> pd.DataFrame:
    rows = []
    for d in _DATES:
        for s in ("A", "B"):
            row = {
                "date": d, "symbol": s, "close": 10.0,
                "suspended": False, "at_up_limit": False, "at_down_limit": False,
            }
            if flags and (d, s) in flags:
                row.update(flags[(d, s)])
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def _driver(panel, fee_rate=0.001):
    return BacktestDriver(
        universe=_Universe(),
        scores=_Scores(),
        constructor=TopNEqualWeight(1),
        execution=SimExecution(fee_rate=fee_rate),
        prices=panel,
        rebalance="monthly",
        fee_rate=fee_rate,
        initial_nav=1.0,
    )


def test_driver_blocked_sell_at_down_limit_carries_position():
    # On Feb-end A is at the down-limit -> the A->B switch can't sell A.
    panel = _panel({(_FEB, "A"): {"at_down_limit": True}})
    driver = _driver(panel)
    driver.run()
    log = driver.feasibility_log()
    feb = log.loc[_FEB]
    assert feb["blocked_sells"] == 1          # A could not be sold
    assert feb["executed_turnover"] == 0.0    # nothing executed (no cash for B)
    assert feb["invested"] == 1.0             # A carried at full weight


def test_blocked_buy_at_up_limit_leaves_cash():
    # On Jan-end A is at the up-limit -> the initial buy of A is blocked.
    panel = _panel({(_JAN, "A"): {"at_up_limit": True}})
    driver = _driver(panel)
    driver.run()
    log = driver.feasibility_log()
    jan = log.loc[_JAN]
    assert jan["blocked_buys"] == 1
    assert jan["invested"] == 0.0             # nothing bought -> all cash
    assert jan["executed_turnover"] == 0.0


def test_suspended_blocks_trading_both_ways():
    # A held from January; on Feb-end A is suspended -> cannot exit A, cannot
    # enter B. Book is carried unchanged.
    panel = _panel({(_FEB, "A"): {"suspended": True}})
    driver = _driver(panel)
    driver.run()
    feb = driver.feasibility_log().loc[_FEB]
    assert feb["blocked_sells"] == 1
    assert feb["invested"] == 1.0
    assert feb["executed_turnover"] == 0.0


def test_no_flags_executes_full_switch():
    # Sanity: with no blocks the A->B switch fully executes (turnover 2.0).
    driver = _driver(_panel())
    driver.run()
    feb = driver.feasibility_log().loc[_FEB]
    assert feb["blocked_sells"] == 0 and feb["blocked_buys"] == 0
    assert feb["executed_turnover"] == 2.0    # sold A (1) + bought B (1)
    assert feb["invested"] == 1.0


def test_feasibility_log_aligns_with_nav_index():
    driver = _driver(_panel())
    nav = driver.run()
    log = driver.feasibility_log()
    assert list(log.index) == list(nav.index)  # settled dates only, 1:1


def test_holdings_log_reports_achieved_not_desired_target():
    # The auditability red-line (review HIGH): when A's sell is blocked at the
    # down-limit, the DESIRED target is B but the ACHIEVED book is the carried A.
    # holdings_log must report A (what was held), never B (what was wanted).
    panel = _panel({(_FEB, "A"): {"at_down_limit": True}})
    driver = _driver(panel)
    driver.run()
    h = driver.holdings_log()
    feb = h[h["date"] == _FEB]
    assert list(feb["symbol"]) == ["A"]   # carried (achieved), NOT desired B
    assert "B" not in set(feb["symbol"])
    # and it equals the execution's actual end-of-run book (Feb is the last step)
    assert set(feb["symbol"]) == set(driver._execution.positions().index)
