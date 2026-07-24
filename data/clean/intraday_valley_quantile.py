"""D2 re-export shim — the PR-L factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the valley
weighted-price-quantile factor lives in
:mod:`factors.compute.minute.valley_price_quantile`. This shim keeps every name
the pre-D2 module exported importable from its old path; it is deleted in D6d.
Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.valley_price_quantile import (
    VALLEY_QUANTILE_LOOKBACK_DAYS,
    VALLEY_QUANTILE_MIN_CROSS_SECTION,
    VALLEY_QUANTILE_MIN_VALLEY_BARS,
    VALLEY_QUANTILE_REVERSAL_DAYS,
    compute_valley_price_quantile,
    compute_valley_price_quantile_stats,
    residualize_on_reversal,
    reversal_20,
    valley_price_quantile_by_day,
)

__all__ = [
    "VALLEY_QUANTILE_LOOKBACK_DAYS",
    "VALLEY_QUANTILE_MIN_CROSS_SECTION",
    "VALLEY_QUANTILE_MIN_VALLEY_BARS",
    "VALLEY_QUANTILE_REVERSAL_DAYS",
    "compute_valley_price_quantile",
    "compute_valley_price_quantile_stats",
    "residualize_on_reversal",
    "reversal_20",
    "valley_price_quantile_by_day",
]
