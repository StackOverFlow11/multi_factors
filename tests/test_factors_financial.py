"""FinancialFactor surfaces a PIT-aligned financial column as a factor."""

from __future__ import annotations

import pandas as pd
import pytest

from data.clean.schema import normalize_panel
from factors.compute.financial import FinancialFactor


def _panel_with(field, values=(3.1, 5.5)):
    dates = pd.bdate_range("2024-04-22", periods=2)
    rows = []
    for d in dates:
        for sym, val in zip(("000001.SZ", "000002.SZ"), values):
            rows.append(
                {
                    "date": d, "symbol": sym,
                    "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                    "volume": 1.0, "amount": 1.0, "adj_factor": 1.0,
                    field: val,
                }
            )
    return normalize_panel(pd.DataFrame(rows))


def test_financial_factor_surfaces_column():
    series = FinancialFactor("roe").compute(_panel_with("roe"))
    assert series.name == "roe"
    assert series.xs("000001.SZ", level="symbol").iloc[0] == 3.1


def test_financial_factor_raises_when_column_absent():
    factor = FinancialFactor("roe")
    panel = _panel_with("netprofit_yoy")  # has no 'roe' column
    with pytest.raises(ValueError, match="roe"):
        factor.compute(panel)


def test_financial_factor_rejects_unsupported_field():
    with pytest.raises(ValueError, match="not supported"):
        FinancialFactor("pe_ttm")
