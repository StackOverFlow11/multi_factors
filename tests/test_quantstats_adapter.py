"""quantstats performance adapter (P2-4, network-free).

A thin, report-only wrapper over quantstats: it consumes the backtest's per-period
returns and produces CAGR / Sharpe / max-drawdown / volatility, tagging the
``backend`` it used. If quantstats is unavailable or raises, it must say so and
carry the simple-pandas fallback through — never silently pretend the standard
library ran.
"""

from __future__ import annotations

import math

import pandas as pd

from analytics import quantstats_adapter as qa
from analytics.quantstats_adapter import quantstats_performance


def _monthly_returns(n=12):
    idx = pd.date_range("2023-01-31", periods=n, freq="ME")
    vals = [0.02, -0.01, 0.03, -0.02, 0.01, 0.00, 0.04, -0.03, 0.02, -0.01, 0.01, 0.02][:n]
    return pd.Series(vals, index=idx, name="net_return")


def test_quantstats_performance_reports_standard_metrics():
    out = quantstats_performance(_monthly_returns(), periods_per_year=12)
    assert out["backend"] == "quantstats"
    for key in ("cagr", "sharpe", "max_drawdown", "volatility"):
        assert key in out and math.isfinite(out[key])
    assert out["max_drawdown"] <= 0.0  # drawdown is a non-positive fraction


def test_quantstats_unavailable_discloses_and_keeps_fallback(monkeypatch):
    def _raise():
        raise ImportError("quantstats not installed")

    monkeypatch.setattr(qa, "_import_quantstats", _raise)
    out = quantstats_performance(
        _monthly_returns(), periods_per_year=12,
        simple_fallback={"sharpe": 0.5, "annual_return": -0.1},
    )
    assert out["backend"] == "unavailable"  # honest: NOT "quantstats"
    assert out["sharpe"] == 0.5  # simple fallback carried through
    assert out["annual_return"] == -0.1


def test_quantstats_error_discloses_error_type(monkeypatch):
    class _Boom:
        class stats:
            @staticmethod
            def cagr(*a, **k):
                raise ValueError("degenerate series")

    monkeypatch.setattr(qa, "_import_quantstats", lambda: _Boom)
    out = quantstats_performance(_monthly_returns(), periods_per_year=12,
                                 simple_fallback={"sharpe": 0.3})
    assert out["backend"] == "error"
    assert out["error_type"] == "ValueError"  # type only, no message/secret
    assert out["sharpe"] == 0.3  # fallback preserved


def test_quantstats_empty_returns_is_safe():
    out = quantstats_performance(pd.Series([], dtype=float), periods_per_year=12)
    # empty -> either a clean unavailable/error tag or NaN metrics, never a crash
    assert out["backend"] in ("quantstats", "error", "unavailable")
