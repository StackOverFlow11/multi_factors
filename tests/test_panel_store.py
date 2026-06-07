"""Tests for PanelStore (Slice 2 store part; DATA-008/009/010).

A PanelStore persists a canonical (date, symbol) panel to parquet and reads it
back, optionally filtering by a closed date interval and/or a symbol subset. The
round-trip must preserve the panel exactly: MultiIndex(date, symbol) names, the
column set/order, dtypes, sort order, and values (NaN cells included).

Tests use ``tmp_path`` only — never a real ``artifacts/`` directory.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.clean.schema import INDEX_NAMES, validate_panel
from data.store.panel_store import PanelStore


def test_panel_store_roundtrip_preserves_panel(tmp_path, demo_panel):
    """write() then read() returns an equivalent panel (index/columns/values)."""
    store = PanelStore(root=str(tmp_path))
    store.write("daily", demo_panel)

    loaded = store.read("daily")

    # Still a valid canonical panel.
    validate_panel(loaded)
    assert list(loaded.index.names) == INDEX_NAMES

    # Same columns, same order.
    assert list(loaded.columns) == list(demo_panel.columns)

    # Index and values are equivalent (NaN cells preserved).
    pd.testing.assert_frame_equal(loaded, demo_panel, check_like=False)


def test_panel_store_filters_by_date(tmp_path, demo_panel):
    """read(start, end) returns only rows within the closed [start, end] interval."""
    store = PanelStore(root=str(tmp_path))
    store.write("daily", demo_panel)

    all_dates = demo_panel.index.get_level_values("date").unique().sort_values()
    start = all_dates[5]
    end = all_dates[10]

    loaded = store.read("daily", start=start, end=end)

    got_dates = loaded.index.get_level_values("date").unique()
    # Closed interval: both endpoints included, nothing outside.
    assert got_dates.min() == start
    assert got_dates.max() == end
    assert (got_dates >= start).all()
    assert (got_dates <= end).all()
    # Exactly the 6 days in [start, end] (indices 5..10 inclusive).
    assert len(got_dates) == 6

    # Filtering by date must not drop any symbols present in that window.
    expected = demo_panel.loc[(demo_panel.index.get_level_values("date") >= start)
                              & (demo_panel.index.get_level_values("date") <= end)]
    pd.testing.assert_frame_equal(loaded, expected, check_like=False)


def test_panel_store_filters_by_symbols(tmp_path, demo_panel):
    """read(symbols=[...]) returns only the requested symbol subset."""
    store = PanelStore(root=str(tmp_path))
    store.write("daily", demo_panel)

    wanted = ["000001.SZ", "000003.SZ"]
    loaded = store.read("daily", symbols=wanted)

    got_symbols = sorted(loaded.index.get_level_values("symbol").unique().tolist())
    assert got_symbols == sorted(wanted)

    expected = demo_panel.loc[
        demo_panel.index.get_level_values("symbol").isin(wanted)
    ]
    pd.testing.assert_frame_equal(loaded, expected, check_like=False)


def test_panel_store_string_date_filter(tmp_path, demo_panel):
    """Date filter accepts 'YYYY-MM-DD' strings, not only Timestamps."""
    store = PanelStore(root=str(tmp_path))
    store.write("daily", demo_panel)

    loaded = store.read("daily", start="2024-01-03", end="2024-01-05")
    got_dates = loaded.index.get_level_values("date").unique()
    assert got_dates.min() >= pd.Timestamp("2024-01-03")
    assert got_dates.max() <= pd.Timestamp("2024-01-05")


def test_panel_store_overwrite_guard(tmp_path, demo_panel):
    """overwrite=False on an existing file raises a readable error; True replaces."""
    store = PanelStore(root=str(tmp_path))
    store.write("daily", demo_panel)

    with pytest.raises(ValueError, match="already exists"):
        store.write("daily", demo_panel, overwrite=False)

    # Default overwrite=True succeeds and stays consistent on re-read.
    store.write("daily", demo_panel)
    loaded = store.read("daily")
    pd.testing.assert_frame_equal(loaded, demo_panel, check_like=False)


def test_panel_store_read_missing_raises(tmp_path):
    """Reading a name that was never written raises a readable error."""
    store = PanelStore(root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No stored panel"):
        store.read("does_not_exist")
