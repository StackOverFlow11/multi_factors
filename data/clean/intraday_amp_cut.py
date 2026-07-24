"""D2 re-export shim — the PR-G factor math moved to ``factors.compute.minute``.

Single definition point from D2 on (design v3.2 §6.4): the intraday
amplitude-cut factor lives in :mod:`factors.compute.minute.intraday_amp_cut`.
This shim keeps every name the pre-D2 module exported importable from its old
path; it is deleted in D6d. Import-only by contract (locked by the shim purity
test).
"""

from factors.compute.minute.intraday_amp_cut import (
    AMP_CUT_LAMBDA,
    AMP_CUT_LOOKBACK_DAYS,
    AMP_CUT_MIN_CROSS_SECTION,
    AMP_CUT_MIN_DAY_MINUTES,
    AMP_CUT_MIN_VALID_DAYS,
    V_MEAN_COL,
    V_STD_COL,
    combine_amp_cut_cross_section,
    compute_amp_cut_stats,
    compute_intraday_amp_cut,
)

__all__ = [
    "AMP_CUT_LAMBDA",
    "AMP_CUT_LOOKBACK_DAYS",
    "AMP_CUT_MIN_CROSS_SECTION",
    "AMP_CUT_MIN_DAY_MINUTES",
    "AMP_CUT_MIN_VALID_DAYS",
    "V_MEAN_COL",
    "V_STD_COL",
    "combine_amp_cut_cross_section",
    "compute_amp_cut_stats",
    "compute_intraday_amp_cut",
]
