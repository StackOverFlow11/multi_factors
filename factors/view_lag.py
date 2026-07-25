"""Availability-lag transforms for the two views (factor-refactor D4, §1.3/R18).

PURE functions (no data access, no feed, no qt). Given an already-prepared
close-view daily panel or normalized 1min bars, produce the view-lagged input a
factor's ``compute`` consumes — so the SAME view-agnostic factor code (design
§1.4 mechanism 3) serves both views. The materializer composes these.

The decision(d) view is the operational information set at the 14:50 decision:

* daily fields are visible only at ``<= d-1`` EXCEPT ``open`` (fixed at 09:30,
  known by 14:50 — the ONE field-level same-day exception, R7). Concretely the
  prev-day columns are shifted forward one trading day PER SYMBOL, so the row at
  date d carries the value that was final at d-1; ``open`` is left same-day.
  This is exactly ``<= d-1`` for every prev-day field: a factor reading the
  shifted column at row d therefore uses only d-1 information for it (and a
  factor's own trailing lag composes on top consistently). The same-day set is
  DERIVED from the policy table (:func:`decision_same_day_fields`), never a
  hand-kept literal.
* minute bars: only bars with ``available_time <= d 14:50`` survive (the I3
  cutoff), filtered on per-bar timestamps BEFORE any daily grouping.

The ex-date boundary (§1.3 note 4) is a SEPARATE, type-boundary-clean module:
:func:`ex_date_mask` turns a raw ``adj_factor`` series into a BOOLEAN mask
(``af(d) != af(d-1)`` per symbol) and nothing else — the ``adj_factor`` numeric
never enters the decision-view frame, so "only judge the fact, never use the
value" is enforced by the return type, not by docstring discipline.

Leaf discipline: numpy / pandas + the pure ``data.availability_policy`` leaf and
``data.clean.intraday_schema`` constants. Never import qt, feeds, or caches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import (
    AVAILABILITY_POLICY,
    DecisionVisibility,
    RevisionHorizonSource,
)
from data.clean.intraday_schema import DEFAULT_DECISION_TIME
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL


def decision_same_day_fields() -> frozenset[str]:
    """Factor-input fields legal SAME-DAY in the decision view (policy-derived).

    Reads the §1.3 policy table: a field is decision-view same-day iff its rule
    is ``DecisionVisibility.SAME_DAY`` AND it is a factor-store input endpoint
    (``horizon_source != NOT_FACTOR_INPUT`` — stk_limit / suspend_d / namechange
    are same-day too but never feed a factor, so they are excluded). Today this
    is exactly ``{"open"}`` (market_daily's field-level exception), but it is
    derived so a policy-table change flows through automatically.
    """
    out: set[str] = set()
    for (_source, field), rule in AVAILABILITY_POLICY.items():
        if field is None:
            continue
        if (
            rule.decision_visibility is DecisionVisibility.SAME_DAY
            and rule.horizon_source is not RevisionHorizonSource.NOT_FACTOR_INPUT
        ):
            out.add(field)
    return frozenset(out)


def daily_decision_lag(
    panel: pd.DataFrame, *, same_day_fields: frozenset[str] | None = None
) -> pd.DataFrame:
    """Shift the close-view daily panel into the decision(d) view (R18 daily lag).

    Every column is shifted forward one trading day PER SYMBOL (row d gets row
    d-1's value = ``<= d-1`` visibility) EXCEPT the same-day fields (``open``),
    which are left untouched (09:30 fixed, legal at 14:50). The first row of each
    symbol has no d-1 for the shifted columns, so it becomes NaN (a decision-view
    value cannot exist on a symbol's very first date). Never mutates ``panel``.
    """
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != [
        DATE_LEVEL,
        SYMBOL_LEVEL,
    ]:
        raise ValueError(
            "daily_decision_lag expects a MultiIndex(date, symbol) panel; got "
            f"index names {list(panel.index.names)}."
        )
    same_day = decision_same_day_fields() if same_day_fields is None else same_day_fields
    ordered = panel.sort_index(kind="mergesort")
    shift_cols = [c for c in ordered.columns if c not in same_day]
    if not shift_cols:
        return ordered.copy()
    out = ordered.copy()
    shifted = ordered[shift_cols].groupby(level=SYMBOL_LEVEL, sort=False).shift(1)
    out[shift_cols] = shifted
    return out


def minute_decision_cutoff(
    bars: pd.DataFrame, *, decision_time: str = DEFAULT_DECISION_TIME
) -> pd.DataFrame:
    """Keep only bars visible at the decision cutoff (``available_time <= d cutoff``).

    The cutoff for each bar is its own trade date + ``decision_time`` (default
    14:50). The filter runs on per-bar timestamps (never date-normalized first),
    so a post-cutoff bar can never leak into a decision. Never mutates ``bars``.
    """
    if bars.empty:
        return bars.copy()
    bar_end = bars.index.get_level_values("time")
    trade_date = pd.DatetimeIndex(bar_end).normalize()
    cutoff = trade_date + pd.Timedelta(decision_time)
    visible = bars["available_time"].to_numpy() <= cutoff.to_numpy()
    return bars.loc[visible].copy()


def ex_date_mask(adj_factor: pd.Series) -> pd.Series:
    """BOOLEAN ex-date mask ``af(d) != af(d-1)`` per symbol (§1.3 note 4).

    The ONLY thing this module exposes to the materializer: the fact "day d is an
    ex-date for this symbol", never the ``adj_factor`` numeric. The first row of
    each symbol is False (no d-1 to compare). ``af`` is compared with NaN-aware
    inequality so a missing raw factor is not spuriously flagged.
    """
    if not isinstance(adj_factor.index, pd.MultiIndex) or list(
        adj_factor.index.names
    ) != [DATE_LEVEL, SYMBOL_LEVEL]:
        raise ValueError(
            "ex_date_mask expects a MultiIndex(date, symbol) adj_factor series; got "
            f"index names {list(adj_factor.index.names)}."
        )
    ordered = adj_factor.sort_index(kind="mergesort")
    prev = ordered.groupby(level=SYMBOL_LEVEL, sort=False).shift(1)
    cur_v = ordered.to_numpy(dtype=float)
    prev_v = prev.to_numpy(dtype=float)
    both_nan = np.isnan(cur_v) & np.isnan(prev_v)
    # differ, and not the (first-row) case where prev is NaN, and not both-NaN.
    changed = (~(cur_v == prev_v)) & ~np.isnan(prev_v) & ~both_nan
    return pd.Series(changed, index=ordered.index, name="ex_date")


__all__ = [
    "daily_decision_lag",
    "decision_same_day_fields",
    "ex_date_mask",
    "minute_decision_cutoff",
]
