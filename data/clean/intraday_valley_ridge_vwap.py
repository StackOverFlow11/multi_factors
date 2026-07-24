"""D2 re-export shim — the PR-J factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the valley/ridge
VWAP-ratio factor lives in
:mod:`factors.compute.minute.valley_ridge_vwap_ratio`. This shim keeps every
name the pre-D2 module exported importable from its old path; it is deleted in
D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.valley_ridge_vwap_ratio import (
    DIAGNOSTIC_COLUMNS,
    VALLEY_RIDGE_LOOKBACK_DAYS,
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
    compute_valley_ridge_vwap_ratio,
    valley_ridge_vwap_ratio_by_day,
)

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "VALLEY_RIDGE_LOOKBACK_DAYS",
    "VALLEY_RIDGE_MIN_RIDGE_BARS",
    "VALLEY_RIDGE_MIN_VALLEY_BARS",
    "compute_valley_ridge_vwap_ratio",
    "valley_ridge_vwap_ratio_by_day",
]
