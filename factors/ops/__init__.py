"""``factors.ops``: the daily-panel time-series operator library (D2, §3.2).

Pure functions over ``MultiIndex(date, symbol)`` Series with ONE shared
convention set (see :mod:`factors.ops.timeseries`): strictly per-symbol
grouping, full-window leading-NaN semantics, plain Python imports (no string
dispatch, no operator registry — design §3.2 "明确不做").
"""

from factors.ops.timeseries import (
    log_positive,
    ts_lag,
    ts_mean,
    ts_pct_change,
    ts_std,
    ts_sum,
    ts_window_return,
)

__all__ = [
    "log_positive",
    "ts_lag",
    "ts_mean",
    "ts_pct_change",
    "ts_std",
    "ts_sum",
    "ts_window_return",
]
