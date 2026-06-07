"""Tests for the universe layer (Slice 4): StaticUniverse.

Covers UNI-001/002/004 plus the empty-universe edge case. The PIT downgrade
(UNI-003) is a documentation requirement, asserted here only via the docstring
contract that ``members`` ignores ``date``.
"""

from __future__ import annotations

import pandas as pd

from tests.fixtures.panel_factory import NAN_DAY, SYMBOLS, make_demo_panel
from universe.static import StaticUniverse


def _date_at(panel: pd.DataFrame, day_index: int) -> pd.Timestamp:
    """Return the ``day_index``-th distinct trade date in ``panel``."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    return dates[day_index]


def test_static_universe_members_returns_config_symbols():
    universe = StaticUniverse(symbols=SYMBOLS)
    # PIT downgrade: members ignore the date and return the configured list.
    members_early = universe.members(pd.Timestamp("2024-01-01"))
    members_late = universe.members(pd.Timestamp("2024-06-30"))
    assert members_early == list(SYMBOLS)
    assert members_late == list(SYMBOLS)


def test_tradable_excludes_missing_close():
    panel = make_demo_panel()
    universe = StaticUniverse(symbols=SYMBOLS)
    nan_date = _date_at(panel, NAN_DAY)  # 000004.SZ has a NaN close here

    tradable = universe.tradable(nan_date, panel)

    assert "000004.SZ" not in tradable
    # Every other symbol has a valid close on that date and stays tradable.
    for symbol in SYMBOLS:
        if symbol != "000004.SZ":
            assert symbol in tradable


def test_tradable_intersects_members():
    panel = make_demo_panel()
    universe = StaticUniverse(symbols=SYMBOLS)
    some_date = _date_at(panel, NAN_DAY)

    members = set(universe.members(some_date))
    tradable = set(universe.tradable(some_date, panel))

    assert tradable.issubset(members)


def test_empty_universe_is_allowed():
    panel = make_demo_panel()
    universe = StaticUniverse(symbols=[])
    some_date = _date_at(panel, 0)

    assert universe.members(some_date) == []
    # Empty universe must not crash; tradable is also empty.
    assert universe.tradable(some_date, panel) == []
