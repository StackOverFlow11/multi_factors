"""universe.min_listing_days as a buy/selection-eligibility filter (P2-2).

Newly-listed names (age < min_listing_days as of the cross-section date) are
excluded from selection. Boundaries: age < min excluded, age == min allowed, and
a missing ``list_date`` is handled honestly — NOT silently dropped (a data gap
must not shrink the universe) — and is disclosed by the caller.
"""

from __future__ import annotations

import pandas as pd

from universe.filters import apply_tradable_filters


def _cross_panel(list_dates: dict, date: pd.Timestamp) -> pd.DataFrame:
    """One-date panel with close + per-symbol list_date (NaT where unknown)."""
    rows = []
    for sym, ld in list_dates.items():
        rows.append(
            {"date": date, "symbol": sym, "close": 10.0,
             "list_date": pd.Timestamp(ld) if ld is not None else pd.NaT}
        )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def test_age_below_min_is_excluded():
    date = pd.Timestamp("2024-03-01")
    panel = _cross_panel({"YOUNG": "2024-02-20"}, date)  # ~10 days old
    out = apply_tradable_filters(["YOUNG"], date, panel, {"min_listing_days": 60})
    assert out == []


def test_age_at_min_is_allowed():
    date = pd.Timestamp("2024-03-01")
    # exactly 60 days old -> allowed (boundary is inclusive).
    panel = _cross_panel({"BORDER": date - pd.Timedelta(days=60)}, date)
    out = apply_tradable_filters(["BORDER"], date, panel, {"min_listing_days": 60})
    assert out == ["BORDER"]


def test_age_above_min_is_allowed():
    date = pd.Timestamp("2024-03-01")
    panel = _cross_panel({"OLD": "2010-01-01"}, date)
    out = apply_tradable_filters(["OLD"], date, panel, {"min_listing_days": 60})
    assert out == ["OLD"]


def test_missing_list_date_is_kept_not_silently_dropped():
    date = pd.Timestamp("2024-03-01")
    panel = _cross_panel({"UNKNOWN": None}, date)  # NaT list_date
    out = apply_tradable_filters(["UNKNOWN"], date, panel, {"min_listing_days": 60})
    assert out == ["UNKNOWN"]  # data gap must not exclude the name


def test_zero_or_absent_min_is_noop():
    date = pd.Timestamp("2024-03-01")
    panel = _cross_panel({"YOUNG": "2024-02-29"}, date)
    assert apply_tradable_filters(["YOUNG"], date, panel, {"min_listing_days": 0}) == ["YOUNG"]
    assert apply_tradable_filters(["YOUNG"], date, panel, {}) == ["YOUNG"]


def test_no_list_date_column_is_noop():
    # panel without a list_date column at all (e.g. demo) -> filter is a no-op.
    date = pd.Timestamp("2024-03-01")
    idx = pd.MultiIndex.from_tuples([(date, "X")], names=["date", "symbol"])
    panel = pd.DataFrame({"close": [10.0]}, index=idx)
    assert apply_tradable_filters(["X"], date, panel, {"min_listing_days": 60}) == ["X"]
