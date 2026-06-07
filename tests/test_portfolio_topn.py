"""Tests for the TopN equal-weight portfolio constructor (Slice 8).

Covers requirements PF-001/002/003/004/005/009:
    - select the N highest scores,
    - weights sum to 1 (float tol < 1e-9),
    - NaN scores are ignored,
    - fewer candidates than N -> equal-weight the actual count (still sum 1),
    - no candidates -> empty Series (no crash),
    - long-only (no negative weights).

Uses the shared ``scores_factory`` fixture (conftest) so we don't invent demo
data. ``make_scores`` ranks 000001..000005 ascending (000005 highest) with
000004 set to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.base import PortfolioConstructor
from portfolio.construct import TopNEqualWeight

WEIGHT_TOL = 1e-9


def test_topn_selects_highest_scores(scores_factory):
    """PF-001: the top_n highest scores are the only symbols selected."""
    scores = scores_factory("2024-01-01")  # 000005>000003>000002>000001, 000004 NaN
    constructor = TopNEqualWeight(top_n=2)

    weights = constructor.build(scores)

    # Two highest non-NaN scores are 000005 (0.5) and 000003 (0.3).
    assert set(weights.index) == {"000005.SZ", "000003.SZ"}
    assert "000001.SZ" not in weights.index
    assert "000002.SZ" not in weights.index


def test_topn_weights_sum_to_one(scores_factory):
    """PF-002: selected weights sum to 1.0 within tolerance, equal weight 1/k."""
    scores = scores_factory("2024-01-01")
    constructor = TopNEqualWeight(top_n=3)

    weights = constructor.build(scores)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < WEIGHT_TOL
    # Equal weight: every selected position is 1/3.
    assert np.allclose(weights.to_numpy(), 1.0 / 3.0, atol=WEIGHT_TOL)


def test_topn_ignores_nan_scores(scores_factory):
    """PF-003: NaN-score symbols never enter the portfolio."""
    scores = scores_factory("2024-01-01")  # 000004.SZ is NaN
    # Ask for more than the number of valid candidates.
    constructor = TopNEqualWeight(top_n=5)

    weights = constructor.build(scores)

    assert "000004.SZ" not in weights.index
    # 4 valid (non-NaN) candidates remain; all selected, equal-weighted.
    assert len(weights) == 4
    assert abs(weights.sum() - 1.0) < WEIGHT_TOL


def test_topn_handles_less_than_n_candidates(scores_factory):
    """PF-005: fewer candidates than N -> equal-weight the actual count, sum 1."""
    # Only two non-NaN scores available.
    scores = pd.Series(
        {"A.SZ": 0.9, "B.SZ": 0.1, "C.SZ": np.nan}, name="score"
    )
    constructor = TopNEqualWeight(top_n=10)

    weights = constructor.build(scores)

    assert set(weights.index) == {"A.SZ", "B.SZ"}
    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < WEIGHT_TOL
    assert np.allclose(weights.to_numpy(), 0.5, atol=WEIGHT_TOL)


def test_topn_returns_empty_when_no_candidates():
    """PF-004: no candidates -> empty Series, no crash, no NaN/inf weights."""
    scores = pd.Series(
        {"A.SZ": np.nan, "B.SZ": np.nan}, name="score"
    )
    constructor = TopNEqualWeight(top_n=3)

    weights = constructor.build(scores)

    assert isinstance(weights, pd.Series)
    assert len(weights) == 0
    assert weights.sum() == 0.0  # empty sum is 0, no crash


def test_topn_returns_empty_when_input_is_empty():
    """PF-004 edge: an empty score Series yields an empty weight Series."""
    scores = pd.Series(dtype=float, name="score")
    constructor = TopNEqualWeight(top_n=3)

    weights = constructor.build(scores)

    assert isinstance(weights, pd.Series)
    assert len(weights) == 0


def test_topn_is_long_only(scores_factory):
    """PF-009: P0 portfolio has no negative weights even with negative scores."""
    scores = pd.Series(
        {"A.SZ": -0.5, "B.SZ": -0.2, "C.SZ": -0.9}, name="score"
    )
    constructor = TopNEqualWeight(top_n=2, long_only=True)

    weights = constructor.build(scores)

    # Highest (least negative) scores: A (-0.2... actually -0.5) -> A and B.
    assert (weights >= 0).all()
    assert abs(weights.sum() - 1.0) < WEIGHT_TOL


def test_topn_subclasses_constructor_and_is_pure(scores_factory):
    """Contract: TopNEqualWeight IS-A PortfolioConstructor and never mutates input."""
    scores = scores_factory("2024-01-01")
    before = scores.copy()
    constructor = TopNEqualWeight(top_n=2)

    assert isinstance(constructor, PortfolioConstructor)

    _ = constructor.build(scores)

    # Input Series is unchanged (immutable style).
    pd.testing.assert_series_equal(scores, before)
