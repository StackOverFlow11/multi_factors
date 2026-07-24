"""D2 re-export shim — the PR-E factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the amplitude
marginal-anomaly relative-volatility factor lives in
:mod:`factors.compute.minute.amp_marginal_anomaly_vol`. This shim keeps every
name the pre-D2 module exported importable from its old path; it is deleted in
D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.amp_marginal_anomaly_vol import (
    AMP_ANOMALY_FREQ,
    AMP_ANOMALY_LOOKBACK_DAYS,
    AMP_ANOMALY_MIN_POOL,
    AMP_ANOMALY_MIN_SELECTED,
    AMP_ANOMALY_SIGMA_K,
    compute_amp_marginal_anomaly_vol,
)

__all__ = [
    "AMP_ANOMALY_FREQ",
    "AMP_ANOMALY_LOOKBACK_DAYS",
    "AMP_ANOMALY_MIN_POOL",
    "AMP_ANOMALY_MIN_SELECTED",
    "AMP_ANOMALY_SIGMA_K",
    "compute_amp_marginal_anomaly_vol",
]
