"""Tests for the momentum_20 cross-sectional factor (Slice 5; FAC-001..004).

The no-lookahead guarantee (INV-001 / CLAUDE.md invariant #1) is the load-bearing
property: a factor value at date t may use only bars at dates <= t. These tests
also pin the FIXED formula and event order from CONTRACTS.md s6:

    momentum_20[t] = close[t] / close[t - window] - 1   (window default = 20)

Per-symbol grouping must prevent cross-symbol leakage, and early dates with an
insufficient window must yield NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.compute.momentum import MomentumFactor
from tests.fixtures.panel_factory import make_demo_panel


def _series_for(factor: pd.Series, symbol: str) -> pd.Series:
    """Slice a (date, symbol)-indexed factor Series down to one symbol, by date."""
    return factor.xs(symbol, level="symbol")


def test_momentum_output_index_matches_panel(demo_panel: pd.DataFrame) -> None:
    """Output is a MultiIndex(date, symbol) Series aligned 1:1 to the panel."""
    factor = MomentumFactor().compute(demo_panel)

    assert isinstance(factor, pd.Series)
    assert factor.name == "momentum_20"
    assert list(factor.index.names) == ["date", "symbol"]
    # Same index, same order, same length as the source panel.
    pd.testing.assert_index_equal(factor.index, demo_panel.index)


def test_momentum_window_not_enough_returns_nan(demo_panel: pd.DataFrame) -> None:
    """The first ``window`` dates of every symbol are NaN; the next is finite."""
    window = 20
    factor = MomentumFactor(window=window).compute(demo_panel)
    dates = demo_panel.index.get_level_values("date").unique().sort_values()

    rising = _series_for(factor, "000001.SZ").reindex(dates)
    # Dates 0..window-1 have no close[t-window] -> NaN.
    assert rising.iloc[:window].isna().all()
    # The first date with a full window (index == window) is finite.
    assert np.isfinite(rising.iloc[window])


def test_momentum_computed_per_symbol(demo_panel: pd.DataFrame) -> None:
    """Rising symbol is high, falling is low (negative), flat is ~0.

    Also a direct cross-symbol-leakage guard: computing the factor on the full
    panel must equal computing it on each symbol's sub-panel in isolation.
    """
    factor = MomentumFactor(window=20).compute(demo_panel)
    dates = demo_panel.index.get_level_values("date").unique().sort_values()

    rising = _series_for(factor, "000001.SZ").reindex(dates).iloc[20]
    falling = _series_for(factor, "000002.SZ").reindex(dates).iloc[20]
    flat = _series_for(factor, "000003.SZ").reindex(dates).iloc[20]

    assert rising > 0.0
    assert falling < 0.0
    assert flat == 0.0
    assert rising > flat > falling

    # No cross-symbol leakage: per-symbol isolated compute matches joint compute.
    for symbol in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        sub = demo_panel.xs(symbol, level="symbol", drop_level=False)
        isolated = MomentumFactor(window=20).compute(sub)
        joint = factor.loc[(slice(None), symbol)]
        pd.testing.assert_series_equal(
            isolated.reset_index(drop=True),
            joint.reset_index(drop=True),
            check_names=False,
        )


def test_momentum_has_no_lookahead(demo_panel: pd.DataFrame) -> None:
    """Mutating a FUTURE close leaves every earlier-date factor value unchanged.

    This is the core INV-001 guarantee: the value at date t depends only on bars
    at dates <= t, so perturbing a later bar cannot ripple backwards.
    """
    window = 20
    base = MomentumFactor(window=window).compute(demo_panel)
    dates = demo_panel.index.get_level_values("date").unique().sort_values()

    # Mutate close at the LAST date for the rising symbol (a strictly future bar
    # relative to all earlier dates). Build a fresh panel; never mutate input.
    future_date = dates[-1]
    symbol = "000001.SZ"
    mutated = demo_panel.copy(deep=True)
    mutated.loc[(future_date, symbol), "close"] = 9_999.0

    perturbed = MomentumFactor(window=window).compute(mutated)

    # Every value strictly before the mutated date must be byte-for-byte equal.
    base_earlier = base[base.index.get_level_values("date") < future_date]
    pert_earlier = perturbed[perturbed.index.get_level_values("date") < future_date]
    pd.testing.assert_series_equal(base_earlier, pert_earlier)

    # Sanity: the mutation DID change the factor at/after the mutated date,
    # otherwise the test would pass vacuously.
    changed = perturbed[(future_date, symbol)]
    assert changed != base[(future_date, symbol)]


def test_momentum_uses_previous_window() -> None:
    """Exact numeric formula check on the strictly-rising 000001.SZ.

    close = 100 + t, so momentum_20[t] = (100 + t) / (100 + t - 20) - 1.
    """
    window = 20
    panel = make_demo_panel()
    factor = MomentumFactor(window=window).compute(panel)
    dates = panel.index.get_level_values("date").unique().sort_values()
    rising_close = panel["close"].xs("000001.SZ", level="symbol").reindex(dates)
    rising_mom = factor.xs("000001.SZ", level="symbol").reindex(dates)

    for t in range(window, len(dates)):
        expected = rising_close.iloc[t] / rising_close.iloc[t - window] - 1.0
        assert rising_mom.iloc[t] == expected

    # Pin one concrete value: close[20]/close[0]-1 = 120/100 - 1 = 0.2.
    assert rising_mom.iloc[window] == 120.0 / 100.0 - 1.0


def test_momentum_name_is_window_aware() -> None:
    """A non-default window labels the factor ``momentum_<window>`` (MEDIUM-2).

    The class attribute stays the canonical default; the instance name must
    track the actual window so a window=10 config does not mislabel its column.
    """
    assert MomentumFactor.name == "momentum_20"  # class default unchanged
    assert MomentumFactor(window=20).name == "momentum_20"
    assert MomentumFactor(window=10).name == "momentum_10"

    # The computed series is named by the (window-aware) instance name.
    panel = make_demo_panel()
    series = MomentumFactor(window=10).compute(panel)
    assert series.name == "momentum_10"
