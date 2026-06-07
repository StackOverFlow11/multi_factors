"""Tests for enrich_tradability (joining flags onto the panel)."""

from __future__ import annotations

import pandas as pd

from data.clean.schema import normalize_panel
from data.clean.tradability import enrich_tradability


def _panel():
    dates = pd.bdate_range("2024-03-01", periods=3)
    rows = []
    for d in dates:
        for sym, close in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            rows.append(
                {
                    "date": d, "symbol": sym,
                    "open": close, "high": close, "low": close, "close": close,
                    "volume": 1.0, "amount": 1.0, "adj_factor": 1.0,
                }
            )
    return normalize_panel(pd.DataFrame(rows))


def test_enrich_suspended_flag():
    d = pd.Timestamp("2024-03-01")
    out = enrich_tradability(_panel(), suspended={(d, "000001.SZ")})
    assert bool(out.loc[(d, "000001.SZ"), "suspended"]) is True
    assert bool(out.loc[(d, "000002.SZ"), "suspended"]) is False


def test_enrich_is_st_from_intervals():
    intervals = {"000001.SZ": [(pd.Timestamp("2024-01-01"), None, True)]}
    out = enrich_tradability(_panel(), st_intervals=intervals)
    assert out.xs("000001.SZ", level="symbol")["is_st"].all()
    assert not out.xs("000002.SZ", level="symbol")["is_st"].any()


def test_is_st_uses_latest_starting_name():
    # ST era ends 2024-02-15; renamed non-ST after -> March is NOT ST
    intervals = {
        "000001.SZ": [
            (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-02-15"), True),
            (pd.Timestamp("2024-02-16"), None, False),
        ]
    }
    out = enrich_tradability(_panel(), st_intervals=intervals)
    assert not out.xs("000001.SZ", level="symbol")["is_st"].any()


def test_enrich_limit_flags():
    d = pd.Timestamp("2024-03-01")
    limits = pd.DataFrame(
        {
            "date": [d, d],
            "symbol": ["000001.SZ", "000002.SZ"],
            "up_limit": [10.0, 99.0],   # 000001 close==up_limit -> at_up_limit
            "down_limit": [1.0, 20.0],  # 000002 close==down_limit -> at_down_limit
        }
    )
    out = enrich_tradability(_panel(), limits=limits)
    assert bool(out.loc[(d, "000001.SZ"), "at_up_limit"]) is True
    assert bool(out.loc[(d, "000001.SZ"), "at_down_limit"]) is False
    assert bool(out.loc[(d, "000002.SZ"), "at_down_limit"]) is True


def test_enrich_does_not_mutate_input():
    panel = _panel()
    before = set(panel.columns)
    enrich_tradability(panel, suspended=set())
    assert set(panel.columns) == before
