"""Slice 7 (alpha) tests: EqualWeightAlpha — ALPHA-001/002/003.

EqualWeightAlpha combines a factor panel into a single score per symbol by a
plain row-wise mean across the factor columns. It is the P0 baseline alpha:
no forward returns, no learned weights. These tests pin:

- single factor column  -> score equals that column (ALPHA-001);
- multiple columns      -> equal-weight (row) average (ALPHA-001);
- NaN-by-row behavior   -> mean over the available factors, NaN if all missing
  (documented; matches pandas skipna mean);
- ``fit`` works with ``forward_returns=None`` (ALPHA-003);
- ``predict`` returns a symbol-indexed ``pd.Series`` (ALPHA-002).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.base import AlphaModel
from alpha.equal_weight import EqualWeightAlpha
from tests.fixtures.panel_factory import SYMBOLS


def _one_date_factor_panel(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a single-cross-section factor frame: MultiIndex(date, symbol).

    ``values`` maps a factor column name to its per-symbol values (ordered like
    ``SYMBOLS``). The single date is fixed and deterministic.
    """
    date = pd.Timestamp("2024-02-01")
    index = pd.MultiIndex.from_product(
        [[date], SYMBOLS], names=["date", "symbol"]
    )
    return pd.DataFrame({col: vals for col, vals in values.items()}, index=index)


def test_equal_weight_single_factor_returns_factor() -> None:
    """One factor column -> the score equals that column, per symbol."""
    factors = _one_date_factor_panel(
        {"momentum_20": [0.5, -0.2, 0.0, 0.1, 0.9]}
    )
    model = EqualWeightAlpha().fit(factors)
    scores = model.predict(factors)

    assert isinstance(scores, pd.Series)
    assert list(scores.index) == SYMBOLS
    expected = pd.Series(
        [0.5, -0.2, 0.0, 0.1, 0.9], index=pd.Index(SYMBOLS, name="symbol")
    )
    pd.testing.assert_series_equal(
        scores, expected, check_names=False
    )


def test_equal_weight_multiple_factors_averages_columns() -> None:
    """Multiple columns -> the score is the plain row mean across columns."""
    factors = _one_date_factor_panel(
        {
            "momentum_20": [0.4, 0.0, -0.2, 1.0, 0.6],
            "volatility_20": [0.2, 0.4, 0.2, 0.0, 0.4],
        }
    )
    model = EqualWeightAlpha().fit(factors, forward_returns=None)
    scores = model.predict(factors)

    expected_values = [
        (0.4 + 0.2) / 2,
        (0.0 + 0.4) / 2,
        (-0.2 + 0.2) / 2,
        (1.0 + 0.0) / 2,
        (0.6 + 0.4) / 2,
    ]
    expected = pd.Series(
        expected_values, index=pd.Index(SYMBOLS, name="symbol")
    )
    pd.testing.assert_series_equal(scores, expected, check_names=False)


def test_equal_weight_ignores_nan_by_row() -> None:
    """NaN-by-row: mean over available factors; NaN only if all are missing.

    - 000001.SZ has one NaN of two factors -> score == the surviving factor.
    - 000004.SZ has all factors NaN        -> score is NaN.
    """
    factors = _one_date_factor_panel(
        {
            "momentum_20": [np.nan, 0.2, 0.0, np.nan, 0.6],
            "volatility_20": [0.8, 0.4, 0.2, np.nan, 0.4],
        }
    )
    model = EqualWeightAlpha().fit(factors)
    scores = model.predict(factors)

    # Partial NaN -> averages only the present factor.
    assert scores["000001.SZ"] == 0.8
    # Both factors present -> plain mean.
    assert scores["000002.SZ"] == (0.2 + 0.4) / 2
    # All factors NaN for this symbol -> NaN score.
    assert np.isnan(scores["000004.SZ"])


def test_equal_weight_does_not_require_forward_returns() -> None:
    """fit must work with no forward returns (P0 needs no future data)."""
    factors = _one_date_factor_panel(
        {"momentum_20": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )
    model = EqualWeightAlpha()

    # Both no-arg and explicit None must return self (for chaining).
    assert model.fit(factors) is model
    assert model.fit(factors, forward_returns=None) is model
    assert isinstance(model, AlphaModel)

    # And predict still produces a symbol-indexed score Series.
    scores = model.predict(factors)
    assert isinstance(scores, pd.Series)
    assert list(scores.index) == SYMBOLS
