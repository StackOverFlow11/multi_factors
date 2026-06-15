"""Event timing primitives for the shared backtest engine (I5a).

A backtest is a sequence of *holding periods*. Each period makes the time basis
EXPLICIT and auditable so the engine never has to assume "daily close-to-close":

    decision_ts     when the strategy decides the target weights;
    execution_ts    when the (planned) fill happens;
    entry_date      the anchor date whose price enters the book;
    exit_date       the anchor date whose price closes the book;
    date            the label/rebalance date (scores, universe, NAV index).

Two event models populate these fields differently:

  * DailyCloseEventModel: decision = execution = entry = ``date`` (close of T),
    exit = the next rebalance date (close of T_next). Reproduces the accepted
    monthly close-to-close behaviour exactly.
  * IntradayTailEventModel: decision = ``date`` 14:50, execution = ``date`` 14:51,
    entry/exit anchors are still keyed by date but priced from 1min execution
    bars (exec-to-exec), NEVER daily close.

The monthly schedule helper here is the single source of truth for "which dates
are rebalance dates", shared by both models so the schedule can never silently
diverge between daily and intraday.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HoldingPeriod:
    """One settled holding period with an explicit, auditable time basis.

    ``entry_date`` / ``exit_date`` are the price anchors (a pricing provider maps
    ``(anchor_date, symbol) -> price``); ``decision_ts`` / ``execution_ts`` are the
    entry timestamps recorded for auditability, and ``exit_execution_ts`` is the
    (planned) timestamp at which the book is priced out at ``exit_date`` — so the
    holding period's time basis is fully explicit (``execution_ts`` ->
    ``exit_execution_ts``), with no need to infer it from the next period (which
    does not exist for the final period). They may carry intra-day precision.
    ``date`` is the rebalance label used for scores, universe selection, and the
    NAV index — for the daily model it equals ``entry_date``.
    """

    date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    decision_ts: pd.Timestamp
    execution_ts: pd.Timestamp
    exit_execution_ts: pd.Timestamp | None = None


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique trading dates present in a canonical (date, symbol) panel."""
    dates = prices.index.get_level_values("date").unique()
    return pd.DatetimeIndex(sorted(dates))


def monthly_rebalance_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month in ``calendar`` (BT-001), sorted.

    This is exactly the schedule the legacy ``BacktestDriver.rebalance_dates``
    produced; both event models build their holding periods from it so daily and
    intraday share one rebalance calendar.
    """
    if len(calendar) == 0:
        return []
    frame = pd.DataFrame({"date": calendar})
    frame["ym"] = frame["date"].dt.to_period("M")
    last = frame.groupby("ym")["date"].max().sort_values()
    return list(last)


def monthly_anchor_pairs(
    calendar: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``(rebalance_date, exit_date)`` pairs for the monthly schedule.

    For each monthly rebalance date the exit anchor is the NEXT rebalance date,
    except the final rebalance whose exit is the last trading day in the
    calendar. A pair whose exit ``<=`` its rebalance date (a rebalance on the
    final trading day → a zero-length forward window) is dropped, matching the
    legacy driver's ``if end <= date: continue`` (BT-003 honesty).
    """
    reb = monthly_rebalance_dates(calendar)
    if not reb:
        return []
    last_day = calendar[-1]
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i, date in enumerate(reb):
        exit_date = reb[i + 1] if i + 1 < len(reb) else last_day
        if exit_date <= date:
            continue
        pairs.append((date, exit_date))
    return pairs
