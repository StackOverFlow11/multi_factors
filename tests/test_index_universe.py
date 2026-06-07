"""PIT index universe — the core fix for survivorship / membership look-ahead."""

from __future__ import annotations

import pandas as pd
import pytest

from universe.index_universe import PITIndexUniverse


def _constituents():
    # Two snapshots. C leaves and D joins at the Feb snapshot.
    rows = [
        ("2024-01-31", "000001.SZ"),
        ("2024-01-31", "000002.SZ"),
        ("2024-01-31", "000003.SZ"),  # C: in the index in Jan, dropped in Feb
        ("2024-02-29", "000001.SZ"),
        ("2024-02-29", "000002.SZ"),
        ("2024-02-29", "000004.SZ"),  # D: only joins at the Feb snapshot
    ]
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "weight": 1.0,
        }
    )


def test_members_use_latest_snapshot_on_or_before_date():
    uni = PITIndexUniverse(_constituents())
    # mid-Feb -> the Jan snapshot is the latest on-or-before
    assert uni.members(pd.Timestamp("2024-02-15")) == ["000001.SZ", "000002.SZ", "000003.SZ"]
    # after the Feb snapshot
    assert uni.members(pd.Timestamp("2024-03-10")) == ["000001.SZ", "000002.SZ", "000004.SZ"]


def test_members_no_lookahead_into_future_snapshot():
    uni = PITIndexUniverse(_constituents())
    # 000004.SZ only joins on 2024-02-29; it must NOT appear before that snapshot
    assert "000004.SZ" not in uni.members(pd.Timestamp("2024-02-15"))


def test_members_keep_dropped_name_for_its_era():
    # survivorship: 000003.SZ left in Feb but must be a member for Jan/early-Feb
    uni = PITIndexUniverse(_constituents())
    assert "000003.SZ" in uni.members(pd.Timestamp("2024-02-01"))
    assert "000003.SZ" not in uni.members(pd.Timestamp("2024-03-01"))


def test_members_empty_before_first_snapshot():
    uni = PITIndexUniverse(_constituents())
    assert uni.members(pd.Timestamp("2024-01-01")) == []


def test_tradable_filters_missing_close():
    uni = PITIndexUniverse(_constituents())
    date = pd.Timestamp("2024-02-15")
    panel = pd.DataFrame(
        {"close": [10.0, float("nan"), 12.0]},
        index=pd.MultiIndex.from_tuples(
            [(date, "000001.SZ"), (date, "000002.SZ"), (date, "000003.SZ")],
            names=["date", "symbol"],
        ),
    )
    # 000002.SZ has a NaN close -> excluded; result stays a subset of members
    assert uni.tradable(date, panel) == ["000001.SZ", "000003.SZ"]


def test_tradable_empty_when_no_market_data_for_date():
    uni = PITIndexUniverse(_constituents())
    empty = pd.DataFrame(
        {"close": []},
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["date", "symbol"]
        ),
    )
    assert uni.tradable(pd.Timestamp("2024-02-15"), empty) == []


def test_requires_date_and_symbol_columns():
    with pytest.raises(ValueError, match="date.*symbol|constituents"):
        PITIndexUniverse(pd.DataFrame({"foo": [1]}))
