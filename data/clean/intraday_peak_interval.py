"""D2 re-export shim — the PR-H factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the volume-peak
interval-kurtosis factor lives in
:mod:`factors.compute.minute.peak_interval_kurtosis`. This shim keeps every
name the pre-D2 module exported importable from its old path; it is deleted in
D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.peak_interval_kurtosis import (
    PEAK_INTERVAL_LOOKBACK_DAYS,
    PEAK_INTERVAL_MIN_INTERVALS,
    compute_peak_interval_kurtosis,
    excess_kurtosis,
    peak_intervals_by_day,
)

__all__ = [
    "PEAK_INTERVAL_LOOKBACK_DAYS",
    "PEAK_INTERVAL_MIN_INTERVALS",
    "compute_peak_interval_kurtosis",
    "excess_kurtosis",
    "peak_intervals_by_day",
]
