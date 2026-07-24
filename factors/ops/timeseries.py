"""Per-symbol time-series operators for daily ``MultiIndex(date, symbol)`` panels.

D2 (design v3.2 §3.2): the ONE home of the rolling/lag machinery the daily
factors used to inline. Three conventions, shared by every operator and locked
by tests so no factor can silently deviate:

1. **Per-symbol grouping.** Every operator groups on the ``symbol`` index level
   before any temporal step, so one symbol's history can never leak into
   another's value (red line: cross-symbol isolation).
2. **Leading-NaN semantics.** Window operators default ``min_periods`` to the
   FULL window: a warm-up row is an honest NaN, never a partial-window estimate
   that silently changes meaning across the panel head.
3. **Purity.** Operators never mutate their input and never reindex/reorder it
   beyond what the per-symbol groupby implies; callers align the result to
   their panel (``reindex``) themselves, exactly as the pre-D2 factor code did.

Scope discipline (§六.12, 不过度设计): only the operators the D2 daily-factor
rewrite actually consumes exist here. Speculative operators (``cs_rank``,
``rolling_median``, …) are added when a factor needs them, not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.schema import SYMBOL_LEVEL


def _grouped(series: pd.Series):
    """The shared per-symbol groupby (convention #1)."""
    return series.groupby(level=SYMBOL_LEVEL, group_keys=False)


def ts_lag(series: pd.Series, periods: int = 1) -> pd.Series:
    """Per-symbol backward shift: value at ``t`` becomes the value at ``t - periods``.

    Strictly backward for ``periods > 0`` (the only direction a factor may look);
    the leading ``periods`` rows of every symbol are NaN.
    """
    if not isinstance(periods, int) or periods < 1:
        raise ValueError(f"ts_lag periods must be a positive integer; got {periods!r}.")
    return series.groupby(level=SYMBOL_LEVEL).shift(periods)


def ts_window_return(series: pd.Series, window: int) -> pd.Series:
    """``series[t] / series[t - window] - 1`` per symbol (the momentum ratio).

    The leading ``window`` rows of every symbol are NaN (no full denominator).
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError(
            f"ts_window_return window must be a positive integer; got {window!r}."
        )
    return series / ts_lag(series, window) - 1.0


def ts_pct_change(series: pd.Series) -> pd.Series:
    """Per-symbol one-step simple return; each symbol's first row is NaN."""
    return _grouped(series).apply(lambda s: s.pct_change())


def ts_std(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Per-symbol rolling std (ddof=1) over ``window`` rows; full window by default."""
    if not isinstance(window, int) or window < 2:
        raise ValueError(f"ts_std window must be an integer >= 2; got {window!r}.")
    mp = window if min_periods is None else min_periods
    return _grouped(series).apply(
        lambda s: s.rolling(window, min_periods=mp).std(ddof=1)
    )


def ts_mean(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Per-symbol rolling mean over ``window`` rows; full window by default."""
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"ts_mean window must be a positive integer; got {window!r}.")
    mp = window if min_periods is None else min_periods
    return _grouped(series).apply(
        lambda s: s.rolling(window, min_periods=mp).mean()
    )


def ts_sum(series: pd.Series, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Per-symbol rolling sum over ``window`` rows; full window by default."""
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"ts_sum window must be a positive integer; got {window!r}.")
    mp = window if min_periods is None else min_periods
    return _grouped(series).apply(
        lambda s: s.rolling(window, min_periods=mp).sum()
    )


def log_positive(series: pd.Series) -> pd.Series:
    """``log(series)`` where ``series > 0``; anything else is NaN, never ``-inf``."""
    return np.log(series.where(series > 0))


__all__ = [
    "log_positive",
    "ts_lag",
    "ts_mean",
    "ts_pct_change",
    "ts_std",
    "ts_sum",
    "ts_window_return",
]
