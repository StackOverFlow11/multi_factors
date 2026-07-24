"""D2 re-export shim — the PR-D factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the minute
ideal-amplitude factor lives in
:mod:`factors.compute.minute.minute_ideal_amplitude`. This shim keeps every
name the pre-D2 module exported importable from its old path; it is deleted in
D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.minute_ideal_amplitude import (
    IDEAL_AMP_LAMBDA,
    IDEAL_AMP_LOOKBACK_DAYS,
    IDEAL_AMP_MIN_MINUTES,
    compute_minute_ideal_amplitude,
)

__all__ = [
    "IDEAL_AMP_LAMBDA",
    "IDEAL_AMP_LOOKBACK_DAYS",
    "IDEAL_AMP_MIN_MINUTES",
    "compute_minute_ideal_amplitude",
]
