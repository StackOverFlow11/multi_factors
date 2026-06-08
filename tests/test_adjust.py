"""Tests for front-adjustment (qfq) — the core correctness of P1 real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.adjust import front_adjust
from data.clean.schema import normalize_panel


def _raw_panel(closes, adj_factors, symbol="000001.SZ"):
    """Build a normalized single-symbol raw panel from close + adj_factor lists."""
    n = len(closes)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(closes, dtype=float)  # real prices are float, not int
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "amount": 1000.0,
            "adj_factor": pd.Series(adj_factors, dtype=float),
        }
    )
    return normalize_panel(df)


def test_front_adjust_removes_ex_dividend_gap():
    # raw close drops 10 -> 9 across an ex-date (a pure dividend, no real move).
    # adj_factor jumps so raw*adj is continuous; older dates have a smaller factor.
    raw = _raw_panel([10, 10, 9, 9, 9], [0.9, 0.9, 1.0, 1.0, 1.0])
    qfq = front_adjust(raw)
    close = qfq["close"].to_numpy()
    # the artificial -10% gap is removed -> qfq close is continuous
    assert np.allclose(close, [9, 9, 9, 9, 9])
    # raw saw a fake -10% return across the ex-date ...
    rc = raw["close"].to_numpy()
    assert np.isclose(rc[2] / rc[1] - 1, -0.1)
    # ... qfq smooths it to 0 (correct economic return)
    assert np.isclose(close[2] / close[1] - 1, 0.0)


def test_front_adjust_anchors_latest_price_unchanged():
    raw = _raw_panel([10, 10, 9, 9, 9], [0.9, 0.9, 1.0, 1.0, 1.0])
    qfq = front_adjust(raw)
    # most recent date: anchor ratio = 1, so qfq == raw there
    assert np.isclose(qfq["close"].iloc[-1], raw["close"].iloc[-1])


def test_front_adjust_returns_invariant_to_anchor():
    # same economics, adj_factor scaled by 10x -> returns must be identical
    q1 = front_adjust(_raw_panel([10, 10, 9, 9, 9], [0.9, 0.9, 1.0, 1.0, 1.0]))
    q2 = front_adjust(_raw_panel([10, 10, 9, 9, 9], [9.0, 9.0, 10.0, 10.0, 10.0]))
    r1 = q1["close"].pct_change().to_numpy()[1:]
    r2 = q2["close"].pct_change().to_numpy()[1:]
    assert np.allclose(r1, r2)


def test_front_adjust_identity_when_adj_factor_one():
    raw = _raw_panel([10, 11, 12, 13, 14], [1.0] * 5)
    pd.testing.assert_frame_equal(front_adjust(raw), raw)


def test_front_adjust_requires_adj_factor():
    raw = _raw_panel([10, 11, 12], [1.0] * 3).drop(columns=["adj_factor"])
    with pytest.raises(ValueError, match="adj_factor"):
        front_adjust(raw)


def test_front_adjust_is_per_symbol():
    a = _raw_panel([10, 10, 9], [0.9, 0.9, 1.0], "000001.SZ")
    b = _raw_panel([20, 22, 24], [1.0, 1.0, 1.0], "000002.SZ")
    raw = normalize_panel(pd.concat([a, b]).reset_index())
    qfq = front_adjust(raw)
    # symbol B (factor 1.0 throughout) is untouched -> no cross-contamination
    qb = qfq.xs("000002.SZ", level="symbol")["close"].to_numpy()
    assert np.allclose(qb, [20, 22, 24])


def test_front_adjust_does_not_mutate_input():
    raw = _raw_panel([10, 10, 9, 9, 9], [0.9, 0.9, 1.0, 1.0, 1.0])
    before = raw["close"].copy()
    front_adjust(raw)
    pd.testing.assert_series_equal(raw["close"], before)
