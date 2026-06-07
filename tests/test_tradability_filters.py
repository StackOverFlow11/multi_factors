"""Shared tradability filter helper (suspended / ST / price-limit + missing close)."""

from __future__ import annotations

import pandas as pd
import pytest

from universe.filters import apply_tradable_filters

D = pd.Timestamp("2024-03-04")


def _panel(rows: dict) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([(D, s) for s in rows], names=["date", "symbol"])
    return pd.DataFrame([rows[s] for s in rows], index=idx)


def test_always_drops_missing_close():
    panel = _panel({"A": {"close": 10.0}, "B": {"close": float("nan")}})
    assert apply_tradable_filters(["A", "B"], D, panel, {}) == ["A"]


def test_drops_suspended_when_enabled():
    panel = _panel(
        {"A": {"close": 10.0, "suspended": False}, "B": {"close": 11.0, "suspended": True}}
    )
    assert apply_tradable_filters(["A", "B"], D, panel, {"suspended": True}) == ["A"]
    # toggle off -> both kept
    assert apply_tradable_filters(["A", "B"], D, panel, {"suspended": False}) == ["A", "B"]


def test_drops_st_when_enabled():
    panel = _panel(
        {"A": {"close": 10.0, "is_st": False}, "B": {"close": 11.0, "is_st": True}}
    )
    assert apply_tradable_filters(["A", "B"], D, panel, {"st": True}) == ["A"]


def test_drops_at_limit_when_enabled():
    panel = _panel(
        {
            "A": {"close": 10.0, "at_up_limit": False, "at_down_limit": False},
            "B": {"close": 11.0, "at_up_limit": True, "at_down_limit": False},
            "C": {"close": 9.0, "at_up_limit": False, "at_down_limit": True},
        }
    )
    assert apply_tradable_filters(["A", "B", "C"], D, panel, {"limit_up_down": True}) == ["A"]


def test_noop_when_flag_column_absent():
    # filters requested but the panel carries no flag columns (e.g. demo) -> no-op
    panel = _panel({"A": {"close": 10.0}, "B": {"close": 11.0}})
    flt = {"suspended": True, "st": True, "limit_up_down": True}
    assert apply_tradable_filters(["A", "B"], D, panel, flt) == ["A", "B"]


def test_missing_date_returns_empty():
    panel = _panel({"A": {"close": 10.0}})
    assert apply_tradable_filters(["A"], pd.Timestamp("2024-03-05"), panel, {}) == []


def test_requires_close_column():
    idx = pd.MultiIndex.from_tuples([(D, "A")], names=["date", "symbol"])
    panel = pd.DataFrame({"open": [10.0]}, index=idx)
    with pytest.raises(ValueError, match="close"):
        apply_tradable_filters(["A"], D, panel, {})
