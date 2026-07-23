"""FinancialFactor: a cross-sectional factor from a PIT-aligned financial column.

The financial value on the panel is ALREADY point-in-time aligned by disclosure
date (:func:`data.clean.pit_financials.asof_financials` placed it there using
``ann_date <= trade_date``), so this factor does no temporal logic — it simply
surfaces that column as the factor series. The no-look-ahead guarantee lives in
the as-of alignment, upstream; the factor layer still never sees future data.
"""

from __future__ import annotations

import pandas as pd

from data.availability_policy import FINA_INDICATOR
from factors.base import Factor
from factors.spec import FactorSpec, PanelField

# financial fields that may be requested as a factor (P1; grossprofit_margin
# joined in P3-5 as the conservative quality candidate — same ann_date as-of
# machinery, no new temporal logic).
SUPPORTED_FIELDS: tuple[str, ...] = ("roe", "netprofit_yoy", "grossprofit_margin")

# Per-field evaluation contract metadata (hypothesis fixed BEFORE any run).
# All three carry the conventional +1 prior: more profitable / faster-growing /
# higher-margin firms are hypothesized to earn higher cross-sectional returns.
# The project's own evidence is weak-to-null for roe / netprofit_yoy (P3-1: IC
# 0.0006 / 0.0001 on SSE50; P3-3/P3-4: signs flip across cells) and
# grossprofit_margin showed no signal in P3-5 — the prior is still the STATED
# hypothesis; the verdict is what checks it.
_FIELD_META: dict[str, tuple[int, str, str]] = {
    # field: (expected_ic_sign, family, description)
    "roe": (
        +1,
        "quality",
        "Return on equity (fina_indicator), PIT-aligned by ann_date disclosure.",
    ),
    "netprofit_yoy": (
        +1,
        "growth",
        "Net-profit YoY growth (fina_indicator), PIT-aligned by ann_date.",
    ),
    "grossprofit_margin": (
        +1,
        "quality",
        "Gross-profit margin (fina_indicator), PIT-aligned by ann_date.",
    ),
}


class FinancialFactor(Factor):
    """Surface a PIT-aligned financial column (e.g. ``roe``) as a factor."""

    def __init__(self, field: str = "roe") -> None:
        if field not in SUPPORTED_FIELDS:
            raise ValueError(
                f"FinancialFactor field {field!r} not supported; "
                f"choose one of {SUPPORTED_FIELDS}."
            )
        self.name = field
        self._field = field

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property because the id IS the chosen field.

        ``min_history_bars=0``: this factor does no temporal logic of its own —
        the value is already PIT-aligned upstream by ``ann_date`` as-of, so there
        is no warm-up window to exclude.

        D1 declarations (derived, evidence): adjustment=none — the factor only
        surfaces an ann_date-aligned fina_indicator column (``compute`` below
        is a bare column select); no price channel, no qfq, no time-series
        logic. overnight_boundary=none — no raw-price comparison exists, so
        nothing can cross the ex-date basis break.
        """
        sign, family, description = _FIELD_META[self._field]
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=description,
            expected_ic_sign=sign,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._field,),
            requires=(PanelField(self._field, source=FINA_INDICATOR),),
            adjustment="none",
            overnight_boundary="none",
            family=family,
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Return the as-of financial column as a MultiIndex(date, symbol) series."""
        if self._field not in panel.columns:
            raise ValueError(
                f"FinancialFactor('{self._field}') needs an as-of '{self._field}' "
                f"column on the panel. Financial factors require the tushare data "
                f"path (ann_date alignment); they are not available for demo data."
            )
        return panel[self._field].rename(self.name)
