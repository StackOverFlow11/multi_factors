"""FinancialFactor: a cross-sectional factor from a PIT-aligned financial column.

The financial value on the panel is ALREADY point-in-time aligned by disclosure
date (:func:`data.clean.pit_financials.asof_financials` placed it there using
``ann_date <= trade_date``), so this factor does no temporal logic — it simply
surfaces that column as the factor series. The no-look-ahead guarantee lives in
the as-of alignment, upstream; the factor layer still never sees future data.
"""

from __future__ import annotations

import pandas as pd

from factors.base import Factor

# financial fields that may be requested as a factor (P1).
SUPPORTED_FIELDS: tuple[str, ...] = ("roe", "netprofit_yoy")


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

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Return the as-of financial column as a MultiIndex(date, symbol) series."""
        if self._field not in panel.columns:
            raise ValueError(
                f"FinancialFactor('{self._field}') needs an as-of '{self._field}' "
                f"column on the panel. Financial factors require the tushare data "
                f"path (ann_date alignment); they are not available for demo data."
            )
        return panel[self._field].rename(self.name)
