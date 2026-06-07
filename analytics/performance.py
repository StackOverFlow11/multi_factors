"""Portfolio performance metrics from a nav (or returns) series.

Implements the P0 metrics required by ANA-004: annualized return, max drawdown,
volatility, Sharpe. ``performance_summary`` bundles them into a plain dict for
the phase0 report.

Implementation note (INV-007 downgrade): these are simple, dependency-light
numpy/pandas computations, NOT quantstats/empyrical. ``quantstats`` is available
and may be used later for the richer HTML tearsheet; any such use must be
recorded in the phase0 report. The simple version is what runs in P0.

All functions are pure: inputs are never mutated.

Conventions
-----------
- nav: a level series (e.g. starts at 1.0); returns are derived as pct_change.
- max_drawdown is returned as a NEGATIVE fraction (e.g. -0.5 for a 50% fall),
  or 0.0 if the nav never falls below a prior peak.
- Annualization uses ``periods_per_year`` (default 252 trading days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_PERIODS_PER_YEAR: int = 252


def _to_returns(series: pd.Series, is_nav: bool) -> pd.Series:
    """Coerce input to a clean per-period returns series.

    If ``is_nav`` the input is a level series and returns are its pct_change;
    otherwise it is already a returns series. Leading NaN (from pct_change) and
    any other NaN are dropped so downstream stats are well defined.
    """
    s = pd.Series(series).astype(float)
    rets = s.pct_change() if is_nav else s
    return rets.dropna()


def _nav_from_input(series: pd.Series, is_nav: bool) -> pd.Series:
    """Return a nav (level) series from either a nav or a returns input."""
    s = pd.Series(series).astype(float)
    if is_nav:
        return s.dropna()
    return (1.0 + s.fillna(0.0)).cumprod()


def max_drawdown(nav: pd.Series) -> float:
    """Maximum drawdown of a nav (level) series, as a negative fraction.

    drawdown[t] = nav[t] / running_peak[t] - 1; the result is the minimum of
    that series (<= 0). A monotonically non-decreasing nav returns 0.0. An empty
    or single-point nav returns 0.0 (no drawdown observable).
    """
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return 0.0
    running_peak = s.cummax()
    drawdown = s / running_peak - 1.0
    return float(drawdown.min())


def annualized_return(nav: pd.Series, periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR) -> float:
    """Geometric annualized return of a nav (level) series.

    Uses total compounded growth over the number of return periods:
        (nav_end / nav_start) ** (periods_per_year / n_periods) - 1
    Returns 0.0 if fewer than two points exist.
    """
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return 0.0
    n_periods = len(s) - 1
    total_growth = s.iloc[-1] / s.iloc[0]
    if total_growth <= 0:
        # nav hit zero/negative: annualization undefined, report total - 1
        return float(total_growth - 1.0)
    return float(total_growth ** (periods_per_year / n_periods) - 1.0)


def volatility(
    nav: pd.Series,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    is_nav: bool = True,
) -> float:
    """Annualized volatility (std of per-period returns, scaled by sqrt(ppy)).

    Uses sample std (ddof=1). Returns 0.0 if fewer than two returns exist.
    """
    rets = _to_returns(nav, is_nav)
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    nav: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    is_nav: bool = True,
) -> float:
    """Annualized Sharpe ratio.

    mean(excess per-period return) / std(per-period return) * sqrt(ppy), where
    the per-period risk-free is ``risk_free / periods_per_year``. Returns 0.0 if
    volatility is zero or fewer than two returns exist.
    """
    rets = _to_returns(nav, is_nav)
    if len(rets) < 2:
        return 0.0
    rf_per_period = risk_free / periods_per_year
    excess = rets - rf_per_period
    std = excess.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def performance_summary(
    nav: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
    is_nav: bool = True,
) -> dict[str, float]:
    """Bundle the P0 performance metrics into a plain dict.

    Always contains at least: ``annual_return``, ``max_drawdown``,
    ``volatility``, ``sharpe``. ``is_nav=True`` treats the input as a level
    series; set ``is_nav=False`` to pass a per-period returns series instead.
    """
    nav_series = _nav_from_input(nav, is_nav)
    return {
        "annual_return": annualized_return(nav_series, periods_per_year),
        "max_drawdown": max_drawdown(nav_series),
        "volatility": volatility(nav_series, periods_per_year, is_nav=True),
        "sharpe": sharpe(nav_series, risk_free, periods_per_year, is_nav=True),
    }


__all__ = [
    "max_drawdown",
    "annualized_return",
    "volatility",
    "sharpe",
    "performance_summary",
]
