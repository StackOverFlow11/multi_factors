"""Thin, report-only quantstats adapter (P2-4).

Consumes the backtest's per-period returns and produces the standard portfolio
metrics (CAGR / Sharpe / max-drawdown / volatility) via ``quantstats``, tagging
the ``backend`` it actually used. This is REPORTING ONLY — it never feeds back
into selection / portfolio / execution, so trading results are unchanged.

Honesty contract (INV-007): if quantstats is unavailable (``ImportError``) or
raises, the result is tagged ``unavailable`` / ``error`` and the caller's simple
pandas ``simple_fallback`` is carried through — the report must never claim the
standard library ran when it did not. The ``_import_quantstats`` indirection makes
the unavailable path unit-testable without uninstalling the package.
"""

from __future__ import annotations

import math

import pandas as pd


def _import_quantstats():
    """Import quantstats (indirection so the ImportError path is testable)."""
    import quantstats

    return quantstats


def quantstats_performance(
    returns: pd.Series,
    periods_per_year: int = 252,
    simple_fallback: dict | None = None,
) -> dict:
    """Standard performance metrics from a per-period returns series.

    Args:
        returns: per-period (e.g. per-rebalance) simple returns, ideally with a
            DatetimeIndex (CAGR uses the date span).
        periods_per_year: annualization factor for Sharpe / volatility (12 for
            monthly rebalances).
        simple_fallback: the authoritative simple-pandas metrics, carried through
            verbatim when quantstats is unavailable or errors.

    Returns:
        ``{"backend": "quantstats", "cagr", "sharpe", "max_drawdown",
        "volatility"}`` on success; ``{"backend": "unavailable", **fallback}`` or
        ``{"backend": "error", "error_type", **fallback}`` otherwise.
    """
    fallback = dict(simple_fallback or {})
    try:
        qs = _import_quantstats()
    except ImportError:
        return {"backend": "unavailable", **fallback}

    try:
        r = pd.Series(returns, dtype=float).dropna()
        if len(r) < 2:
            return {
                "backend": "quantstats",
                "cagr": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"), "volatility": float("nan"),
            }
        # quantstats annualizes by ``periods`` (12 for monthly); the DEFAULT 252
        # would treat monthly returns as daily and explode CAGR. max_drawdown takes
        # the RETURNS series (it builds the equity curve itself) — passing a prebuilt
        # nav makes it read 0.
        metrics = {
            "cagr": float(qs.stats.cagr(r, periods=periods_per_year)),
            "sharpe": float(qs.stats.sharpe(r, periods=periods_per_year)),
            "volatility": float(qs.stats.volatility(r, periods=periods_per_year)),
            "max_drawdown": float(qs.stats.max_drawdown(r)),
        }
        # guard non-finite (e.g. zero-volatility degenerate input -> inf Sharpe).
        metrics = {k: (v if math.isfinite(v) else float("nan")) for k, v in metrics.items()}
        if metrics["max_drawdown"] > 0:  # drawdown is a non-positive fraction
            metrics["max_drawdown"] = -metrics["max_drawdown"]
        return {"backend": "quantstats", **metrics}
    except Exception as exc:  # noqa: BLE001 - any quantstats failure -> disclosed fallback
        return {"backend": "error", "error_type": type(exc).__name__, **fallback}


__all__ = ["quantstats_performance"]
