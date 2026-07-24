"""D2 re-export shim — the PR-K factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the ridge minute-return
factor lives in :mod:`factors.compute.minute.ridge_minute_return`. This shim
keeps every name the pre-D2 module exported importable from its old path; it is
deleted in D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.ridge_minute_return import (
    DIAGNOSTIC_COLUMNS,
    RIDGE_RETURN_LOOKBACK_DAYS,
    RIDGE_RETURN_MIN_RIDGE_BARS,
    compute_ridge_minute_return,
    ridge_minute_return_by_day,
)

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "RIDGE_RETURN_LOOKBACK_DAYS",
    "RIDGE_RETURN_MIN_RIDGE_BARS",
    "compute_ridge_minute_return",
    "ridge_minute_return_by_day",
]
