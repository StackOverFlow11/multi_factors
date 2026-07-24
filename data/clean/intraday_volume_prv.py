"""D2 re-export shim — the PR-F factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the volume-peak-count
factor lives in :mod:`factors.compute.minute.volume_peak_count` and the shared
peak/ridge/valley taxonomy in :mod:`factors.compute.minute.primitives`. This
shim keeps every name the pre-D2 module exported importable from its old path;
it is deleted in D6d. Import-only by contract (locked by the shim purity test).
"""

from factors.compute.minute.primitives import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)
from factors.compute.minute.volume_peak_count import (
    VOLUME_PRV_LOOKBACK_DAYS,
    compute_volume_peak_count,
)

__all__ = [
    "VOLUME_PRV_BASELINE_DAYS",
    "VOLUME_PRV_BASELINE_MIN_OBS",
    "VOLUME_PRV_LOOKBACK_DAYS",
    "VOLUME_PRV_MIN_CLASSIFIABLE",
    "VOLUME_PRV_MIN_VALID_DAYS",
    "VOLUME_PRV_SIGMA_K",
    "compute_volume_peak_count",
    "peak_mask_for_symbol",
    "prepare_visible_minute_bars",
]
