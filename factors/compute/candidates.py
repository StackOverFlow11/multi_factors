"""P3-5 candidate factor pack: conservative, daily, PIT-safe additions.

EXPLORATORY factors used to test whether the legacy trio's weak signal was just
a too-narrow factor set (validated through the P3-4 robustness matrix; not
tuned, not a return claim). Every factor here uses ONLY data known at the trade
date and computes strictly per symbol (grouped on the ``symbol`` index level):

  * ``reversal_w``   = -(close[t] / close[t-w] - 1)  — short-horizon reversal,
    the exact negative of the momentum definition (same no-lookahead argument).
  * ``volatility_w`` = std of the trailing ``w`` daily returns (ddof=1,
    min_periods=w → the leading window is NaN, never a partial estimate).
  * ``liquidity_w``  = log of the trailing ``w``-day mean turnover ``amount``
    (non-positive means → NaN, never -inf; the panel's ``amount`` column comes
    from the bar feed, known same-day).
  * ``value_ep`` / ``value_bp`` surface a daily_basic-enriched column
    (1/pe, 1/pb placed on the panel by the pipeline's value enrichment; the
    ratios are published same-day, PIT-safe by construction). They require the
    tushare path — a demo run has no pe/pb and fails readably upstream.

The quality field ``grossprofit_margin`` is NOT here: it joins
``factors.compute.financial.SUPPORTED_FIELDS`` and rides the existing ann_date
as-of machinery unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base import Factor
from factors.compute.momentum import MomentumFactor

# daily_basic-derived value fields the pipeline can enrich + surface (P3-5).
VALUE_FIELDS: tuple[str, ...] = ("value_ep", "value_bp")


class ReversalFactor(Factor):
    """Short-horizon reversal: the exact negative of ``momentum_w``.

    Reuses :class:`MomentumFactor`'s computation (per-symbol, strictly lagged,
    no lookahead) and flips the sign, so the two definitions can never drift
    apart.
    """

    name: str = "reversal_20"

    def __init__(self, window: int = 20, price_col: str = "close") -> None:
        self._momentum = MomentumFactor(window=window, price_col=price_col)
        self.name = f"reversal_{window}"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return (-self._momentum.compute(panel)).rename(self.name)


class VolatilityFactor(Factor):
    """Trailing daily-return volatility over a fixed window (per symbol).

    ``min_periods=window`` keeps the leading window NaN (a partial-window std
    would silently change meaning across the panel head).
    """

    name: str = "volatility_20"

    def __init__(self, window: int = 20, price_col: str = "close") -> None:
        if not isinstance(window, int) or window < 2:
            raise ValueError(
                f"volatility window must be an integer >= 2, got {window!r}."
            )
        self._window = window
        self._price_col = price_col
        self.name = f"volatility_{window}"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if self._price_col not in panel.columns:
            raise ValueError(
                f"volatility factor needs a '{self._price_col}' column; panel has "
                f"{list(panel.columns)}."
            )
        price = panel[self._price_col]
        grouped = price.groupby(level="symbol", group_keys=False)
        # pct_change within each symbol (never across); rolling std needs a FULL
        # window of returns -> leading rows are NaN, all inputs are <= t.
        vol = grouped.apply(
            lambda s: s.pct_change().rolling(
                self._window, min_periods=self._window
            ).std(ddof=1)
        )
        return vol.reindex(panel.index).rename(self.name)


class LiquidityFactor(Factor):
    """Log of the trailing mean turnover ``amount`` over a fixed window.

    A simple size-of-trading liquidity proxy (its correlation with market cap
    is handled by the existing size neutralization). Non-positive rolling means
    map to NaN — never a silent ``-inf``.
    """

    name: str = "liquidity_20"

    def __init__(self, window: int = 20, amount_col: str = "amount") -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"liquidity window must be a positive integer, got {window!r}."
            )
        self._window = window
        self._amount_col = amount_col
        self.name = f"liquidity_{window}"

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if self._amount_col not in panel.columns:
            raise ValueError(
                f"liquidity factor needs an '{self._amount_col}' column (turnover "
                f"amount from the bar feed); panel has {list(panel.columns)}."
            )
        amount = panel[self._amount_col]
        grouped = amount.groupby(level="symbol", group_keys=False)
        mean_amt = grouped.apply(
            lambda s: s.rolling(self._window, min_periods=self._window).mean()
        )
        # log of a non-positive mean is undefined -> NaN (degenerate liquidity).
        safe = mean_amt.where(mean_amt > 0)
        return np.log(safe).reindex(panel.index).rename(self.name)


class ValueFactor(Factor):
    """Surface a daily_basic-enriched value column (``value_ep`` / ``value_bp``).

    The column is placed on the panel by the pipeline's value enrichment
    (1/pe, 1/pb; same-day-published ratios, PIT-safe by construction). Like the
    financial factors, this does no temporal logic of its own.
    """

    def __init__(self, field: str) -> None:
        if field not in VALUE_FIELDS:
            raise ValueError(
                f"ValueFactor field {field!r} not supported; choose one of "
                f"{VALUE_FIELDS}."
            )
        self.name = field
        self._field = field

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if self._field not in panel.columns:
            raise ValueError(
                f"ValueFactor('{self._field}') needs an enriched '{self._field}' "
                f"column on the panel (daily_basic pe/pb; tushare path only — "
                f"demo data has neither)."
            )
        return panel[self._field].rename(self.name)
