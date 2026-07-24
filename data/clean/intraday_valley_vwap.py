"""D2 re-export shim — the PR-I factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the valley-relative
VWAP factor lives in :mod:`factors.compute.minute.valley_relative_vwap`. This
shim keeps every name the pre-D2 module exported importable from its old path;
it is deleted in D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.valley_relative_vwap import (
    VALLEY_VWAP_LOOKBACK_DAYS,
    VALLEY_VWAP_MIN_VALLEY_BARS,
    compute_valley_relative_vwap,
    valley_vwap_ratio_by_day,
)

__all__ = [
    "VALLEY_VWAP_LOOKBACK_DAYS",
    "VALLEY_VWAP_MIN_VALLEY_BARS",
    "compute_valley_relative_vwap",
    "valley_vwap_ratio_by_day",
]
