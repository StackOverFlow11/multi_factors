"""Tests for analytics.performance (Slice 11, req ANA-004).

Performance metrics from a nav series: annualized return, max drawdown,
volatility, Sharpe. A hand-computed drawdown series pins the math, and the
summary dict is checked for the required metric keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.performance import (
    annualized_return,
    max_drawdown,
    performance_summary,
)


def test_max_drawdown_known_series():
    """Max drawdown on a hand-computed nav series.

    nav = [1.0, 1.2, 0.6, 0.9, 1.5]
    running peak = [1.0, 1.2, 1.2, 1.2, 1.5]
    drawdown    = [0, 0, -0.5, -0.25, 0]
    The worst drawdown is -0.5 (a 50% fall from the 1.2 peak to 0.6).
    """
    nav = pd.Series([1.0, 1.2, 0.6, 0.9, 1.5])
    mdd = max_drawdown(nav)
    # documented sign convention: returned as a negative fraction
    assert np.isclose(mdd, -0.5)


def test_performance_summary_contains_required_metrics():
    """performance_summary returns at least the required metric keys.

    Required per ANA-004: annual_return, max_drawdown, volatility, sharpe.
    """
    # a gently rising nav over ~one year of business days
    dates = pd.bdate_range("2024-01-01", periods=252)
    nav = pd.Series(np.linspace(1.0, 1.2, len(dates)), index=dates)

    summary = performance_summary(nav)
    assert isinstance(summary, dict)
    for key in ("annual_return", "max_drawdown", "volatility", "sharpe"):
        assert key in summary, f"missing required metric: {key}"
        assert summary[key] is not None
        assert np.isfinite(summary[key])

    # a monotonically rising nav has no drawdown
    assert np.isclose(summary["max_drawdown"], 0.0)
    # and a positive annualized return
    assert summary["annual_return"] > 0.0


def test_monthly_nav_annualizes_at_12_not_252():
    """A 12-row monthly nav must annualize at 12/yr, not the daily 252 (HIGH-1).

    A monthly nav doubling over 12 months is ~+100%/yr. Annualizing the same nav
    at 252/yr (the daily default) explodes it absurdly — exactly the headline bug
    in the phase0 report. The cadence-correct call gives a finite, sane number.
    """
    nav = pd.Series(np.linspace(1.0, 2.0, 13))  # 12 monthly returns, doubles

    monthly = annualized_return(nav, periods_per_year=12)
    daily = annualized_return(nav, periods_per_year=252)

    assert np.isfinite(monthly)
    # ~doubling over exactly one year of months -> ~+100%, comfortably < 5000%.
    assert abs(monthly) < 50.0
    assert 0.5 < monthly < 1.5
    # The (wrong) daily annualization on the same nav explodes far past the band.
    assert daily > 50.0
