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

from data.availability_policy import DAILY_BASIC, MARKET_DAILY
from factors.base import Factor
from factors.compute.momentum import MomentumFactor
from factors.spec import FactorSpec, PanelField

# daily_basic-derived value fields the pipeline can enrich + surface (P3-5).
VALUE_FIELDS: tuple[str, ...] = ("value_ep", "value_bp")

# The daily_basic source field behind each surfaced value column (D1 requires
# declaration: the factor reads the enriched panel column, but the DATA it
# needs is the published ratio on this endpoint field).
_VALUE_SOURCE_FIELD: dict[str, str] = {"value_ep": "pe", "value_bp": "pb"}

# Per-field evaluation contract metadata for the value pack. expected_ic_sign=+1
# for both: the classic value prior (cheap earns more than expensive), and the
# ONE hypothesis this project has confirmed on independent samples — P3-5 test
# IC 3/3 positive (0.037~0.056), P3-7 SUPPORTED on 2/2 holdout cells (SSE50 +
# CSI300, magnitude decayed), P3-8 GENERALIZES to CSI500 (test IC +0.0145 /
# +0.0127). Sign confirmed; profitability at portfolio level is NOT.
_VALUE_META: dict[str, str] = {
    "value_ep": "Earnings yield 1/pe (daily_basic, published same day; pe<=0 -> NaN).",
    "value_bp": "Book yield 1/pb (daily_basic, published same day; pb<=0 -> NaN).",
}


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

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: reversal IS -momentum by construction, so its
        hypothesis is the exact negation of MomentumFactor's +1 prior (short-
        horizon losers bounce back). Project evidence: no signal — P3-5 found
        reversal_5/20 sign-flipping across cells.

        D1 declarations (derived): requires / adjustment / overnight_boundary
        are INHERITED from the underlying momentum spec — negation changes the
        sign of the value, not its inputs, its anchor dependence, or its
        overnight-boundary behaviour (this class delegates ``compute`` to
        :class:`MomentumFactor`, so the provenance is identical by
        construction).
        """
        base = self._momentum.spec
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=f"Short-horizon reversal: the exact negative of {base.factor_id}.",
            expected_ic_sign=-base.expected_ic_sign,
            is_intraday=False,
            forward_return_horizon=base.forward_return_horizon,
            return_basis=base.return_basis,
            input_fields=base.input_fields,
            requires=base.requires,
            adjustment=base.adjustment,
            overnight_boundary=base.overnight_boundary,
            family="reversal",
            min_history_bars=base.min_history_bars,
        )

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

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the low-volatility anomaly (low-vol stocks earn
        MORE than high-vol ones), i.e. the factor as defined (higher = more
        volatile) should correlate NEGATIVELY with forward returns. This is the
        project's second independently-confirmed hypothesis: P3-5 test IC 3/3
        negative (-0.044~-0.079), P3-7 SUPPORTED on 2/2 holdout cells, P3-8
        GENERALIZES to CSI500 (test IC -0.0272).

        D1 declarations (D0 pre-assignment table row 14): adjustment=
        returns_invariant — the input is per-symbol ``pct_change`` of the
        pipeline's qfq close panel (``compute`` below), so the anchor cancels
        in every return ratio (data-layer lock:
        tests/test_adjust.py::test_front_adjust_returns_invariant_to_anchor).
        overnight_boundary=none — all returns are ratios of SAME-basis qfq
        closes; in the decision view every input is <= d-1 (the market_daily
        close row of data/availability_policy.py), so no raw-price comparison
        mixes day d's new basis with old-basis history.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Trailing {self._window}-bar volatility: std (ddof=1) of daily "
                f"{self._price_col} returns over a FULL window."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._price_col,),
            requires=(PanelField(self._price_col, source=MARKET_DAILY),),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="lowvol",
            # pct_change loses row 0 and the rolling std needs ``window`` returns
            # -> the leading ``window`` rows are NaN.
            min_history_bars=self._window,
        )

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

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the illiquidity-premium prior — less-traded stocks
        earn more, so this factor (higher = MORE traded) should correlate
        NEGATIVELY with forward returns. Project evidence: P3-5 test IC 3/3
        negative but small in magnitude; P3-6 showed adding it to the
        value+lowvol subset is no free lunch (better on 1 cell, worse on 2).

        D1 declarations (derived, evidence): adjustment=none — the only input
        is the turnover ``amount`` column (``compute`` below: rolling mean of
        amount, then log); traded VALUE in RMB is not rescaled by any
        split/dividend adjustment factor, so the price channel is never
        touched. overnight_boundary=none — no price comparison exists at all,
        let alone one crossing the overnight boundary.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Log of the trailing {self._window}-bar mean turnover "
                f"'{self._amount_col}' (non-positive mean -> NaN)."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._amount_col,),
            requires=(PanelField(self._amount_col, source=MARKET_DAILY),),
            adjustment="none",
            overnight_boundary="none",
            family="liquidity",
            # rolling mean over ``window`` amounts -> first valid row is
            # ``window-1`` (no pct_change involved, unlike volatility).
            min_history_bars=self._window - 1,
        )

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


class OvernightMomentumFactor(Factor):
    """Cumulative overnight (close→open) log return over a fixed window.

        overnight_ret[t]    = log(open[t] / close[t-1])
        overnight_mom_w[t]  = sum(overnight_ret[t-w+1 .. t])

    PIT argument: open[t] is known at the t open and close[t-1] the prior
    close, so the value at t uses only <=t information (factors are computed at
    the t close). Both prices are front-adjusted by the same anchor, so the
    ratio is ex-dividend-safe. Computation is strictly per symbol (the t-1
    close never crosses symbols); non-positive prices map to NaN (never a
    silent ``-inf``); the leading window is NaN (min_periods = w full overnight
    returns, which need w+1 bars).
    """

    name: str = "overnight_mom_20"

    def __init__(
        self, window: int = 20, open_col: str = "open", close_col: str = "close"
    ) -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"overnight momentum window must be a positive integer, got {window!r}."
            )
        self._window = window
        self._open_col = open_col
        self._close_col = close_col
        self.name = f"overnight_mom_{window}"

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the overnight-return premium prior (persistent
        overnight demand keeps paying), i.e. momentum-like in direction. Project
        evidence is mixed/weak: P3-5 test IC 2/3 positive (+0.008/+0.016) and
        negative on the 2020-2022 cell.

        NOT is_intraday: it is derived from DAILY open/close bars, so it carries
        no minute decision-cutoff / execution contract.

        D1 declarations (derived, evidence): adjustment=returns_invariant —
        the value is a sum of log ratios ``log(open[t] / close[t-1])`` on the
        FRONT-ADJUSTED panel; both legs carry the same qfq anchor, which
        cancels in the ratio (class docstring: "Both prices are front-adjusted
        by the same anchor, so the ratio is ex-dividend-safe"; data-layer lock
        tests/test_adjust.py::test_front_adjust_returns_invariant_to_anchor).
        overnight_boundary=none — the ratio does SPAN the overnight gap, but
        both legs are qfq prices on one continuous basis, so no RAW-price
        comparison crosses an ex-date basis BREAK (``front_adjust`` removed
        it); the taxonomy's ``none`` is exactly "no raw-price comparison
        crosses the boundary" (data/availability_policy.py).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Sum of the last {self._window} overnight log returns "
                f"log({self._open_col}[t] / {self._close_col}[t-1])."
            ),
            expected_ic_sign=+1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._open_col, self._close_col),
            requires=(
                PanelField(self._open_col, source=MARKET_DAILY),
                PanelField(self._close_col, source=MARKET_DAILY),
            ),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="momentum",
            # row 0 has no prior close and the rolling sum needs ``window`` full
            # overnight returns -> the leading ``window`` rows are NaN.
            min_history_bars=self._window,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        missing = [c for c in (self._open_col, self._close_col) if c not in panel.columns]
        if missing:
            raise ValueError(
                f"overnight momentum factor needs {missing} column(s); panel has "
                f"{list(panel.columns)}."
            )
        open_px = panel[self._open_col]
        close_px = panel[self._close_col]
        # strictly-lagged close within each symbol; never crosses symbols.
        prev_close = close_px.groupby(level="symbol").shift(1)
        ratio = (open_px / prev_close).where((open_px > 0) & (prev_close > 0))
        overnight = np.log(ratio)
        mom = overnight.groupby(level="symbol", group_keys=False).apply(
            lambda s: s.rolling(self._window, min_periods=self._window).sum()
        )
        return mom.reindex(panel.index).rename(self.name)


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

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property because the id IS the chosen field.

        expected_ic_sign=+1 for both fields — see ``_VALUE_META`` above for the
        prior and the project's independent confirmation (P3-5/P3-7/P3-8).
        ``min_history_bars=0``: the ratio is published same-day, no warm-up.

        D1 declarations (D0 pre-assignment table rows 12-13): adjustment=none
        — the factor only surfaces the enriched 1/pe (resp. 1/pb) column: a
        published same-day ratio with no price channel, no qfq and no
        time-series logic of its own (``compute`` below is a bare column
        select). overnight_boundary=none — no raw-price comparison exists.
        ``requires`` names the daily_basic SOURCE field (pe / pb), not the
        derived panel column: what the factor needs from the data layer is the
        published ratio.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=_VALUE_META[self._field],
            expected_ic_sign=+1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._field,),
            requires=(
                PanelField(_VALUE_SOURCE_FIELD[self._field], source=DAILY_BASIC),
            ),
            adjustment="none",
            overnight_boundary="none",
            family="value",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if self._field not in panel.columns:
            raise ValueError(
                f"ValueFactor('{self._field}') needs an enriched '{self._field}' "
                f"column on the panel (daily_basic pe/pb; tushare path only — "
                f"demo data has neither)."
            )
        return panel[self._field].rename(self.name)
