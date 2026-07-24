"""D2 re-export shim — the PR-M factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the peak/ridge
traded-amount-ratio factor lives in
:mod:`factors.compute.minute.peak_ridge_amount_ratio`. This shim keeps every
name the pre-D2 module exported importable from its old path; it is deleted in
D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.peak_ridge_amount_ratio import (
    DIAGNOSTIC_COLUMNS,
    PEAK_RIDGE_LOOKBACK_DAYS,
    PEAK_RIDGE_MIN_PEAK_BARS,
    PEAK_RIDGE_MIN_RIDGE_BARS,
    compute_peak_ridge_amount_ratio,
    peak_ridge_amount_by_day,
)

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "PEAK_RIDGE_LOOKBACK_DAYS",
    "PEAK_RIDGE_MIN_PEAK_BARS",
    "PEAK_RIDGE_MIN_RIDGE_BARS",
    "compute_peak_ridge_amount_ratio",
    "peak_ridge_amount_by_day",
]
