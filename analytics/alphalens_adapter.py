"""Thin, report-only alphalens-reloaded adapter (P2-4).

Consumes the factor Series + a wide price frame and produces the standard factor
diagnostics (IC mean / IC std / IC-IR, per-quantile mean returns) via
``alphalens``, tagging the ``backend`` it actually used. This is REPORTING ONLY —
it never feeds back into the factor / selection / portfolio, so trading results
are unchanged.

Honesty contract (INV-007): if alphalens is unavailable (``ImportError``) or
raises (small/degenerate cross-sections are common), the result is tagged
``unavailable`` / ``error`` and the caller's simple-pandas ``simple_fallback`` is
carried through — never a silent fake. The ``_import_alphalens`` indirection makes
the unavailable path testable; alphalens' ``print``/warnings are suppressed so the
CLI / report stays clean.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings

import pandas as pd


def _import_alphalens():
    """Import alphalens (indirection so the ImportError path is testable)."""
    import alphalens

    return alphalens


def alphalens_factor_metrics(
    factor: pd.Series,
    prices: pd.DataFrame,
    quantiles: int = 5,
    period: int = 1,
    simple_fallback: dict | None = None,
) -> dict:
    """Standard factor diagnostics from a factor Series + wide price frame.

    Args:
        factor: MultiIndex(date, symbol) factor values.
        prices: wide price frame, index = date, columns = symbol (e.g.
            ``panel["close"].unstack("symbol")``).
        quantiles: number of factor buckets.
        period: forward-return horizon (trading days); alphalens labels the column
            ``f"{period}D"``.
        simple_fallback: the authoritative simple-pandas IC metrics, carried
            through verbatim when alphalens is unavailable or errors.

    Returns:
        ``{"backend": "alphalens", "ic_mean", "ic_std", "ic_ir", "quantile_mean",
        "n_dates"}`` on success; ``{"backend": "unavailable", **fallback}`` or
        ``{"backend": "error", "error_type", **fallback}`` otherwise.
    """
    fallback = dict(simple_fallback or {})
    try:
        al = _import_alphalens()
    except ImportError:
        return {"backend": "unavailable", **fallback}

    try:
        col = f"{period}D"
        # alphalens uses level 0 = date, level 1 = asset; rename for clarity and
        # suppress its stdout banner / RuntimeWarnings so the CLI stays clean.
        fac = factor.copy()
        fac.index = fac.index.set_names(["date", "asset"])
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            factor_data = al.utils.get_clean_factor_and_forward_returns(
                fac, prices, quantiles=quantiles, periods=(period,), max_loss=1.0
            )
            ic = al.performance.factor_information_coefficient(factor_data)[col]
            mq = al.performance.mean_return_by_quantile(factor_data)[0][col]
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
        ic_ir = ic_mean / ic_std if (math.isfinite(ic_std) and ic_std != 0) else float("nan")
        quantile_mean = {int(q): float(v) for q, v in mq.items()}
        return {
            "backend": "alphalens",
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_ir": float(ic_ir),
            "quantile_mean": quantile_mean, "n_dates": int(len(ic)),
        }
    except Exception as exc:  # noqa: BLE001 - any alphalens failure -> disclosed fallback
        return {"backend": "error", "error_type": type(exc).__name__, **fallback}


__all__ = ["alphalens_factor_metrics"]
