"""Tests for industry + market-cap covariate enrichment."""

from __future__ import annotations

import pandas as pd

from data.clean.covariates import enrich_covariates
from data.clean.schema import normalize_panel


def _panel():
    dates = pd.bdate_range("2024-03-01", periods=2)
    rows = []
    for d in dates:
        for sym, close in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            rows.append(
                {"date": d, "symbol": sym, "open": close, "high": close, "low": close,
                 "close": close, "volume": 1.0, "amount": 1.0, "adj_factor": 1.0}
            )
    return normalize_panel(pd.DataFrame(rows))


def test_enrich_industry_broadcast_per_symbol():
    out = enrich_covariates(_panel(), industry={"000001.SZ": "银行", "000002.SZ": "地产"})
    assert out.xs("000001.SZ", level="symbol")["industry"].eq("银行").all()
    assert out.xs("000002.SZ", level="symbol")["industry"].eq("地产").all()


def test_enrich_market_cap_join_by_date_symbol():
    d = pd.Timestamp("2024-03-01")
    mc = pd.DataFrame(
        {"date": [d, d], "symbol": ["000001.SZ", "000002.SZ"], "market_cap": [100.0, 200.0]}
    )
    out = enrich_covariates(_panel(), market_cap=mc)
    assert out.loc[(d, "000001.SZ"), "market_cap"] == 100.0
    assert out.loc[(d, "000002.SZ"), "market_cap"] == 200.0


def test_enrich_does_not_mutate_input():
    panel = _panel()
    before = set(panel.columns)
    enrich_covariates(panel, industry={})
    assert set(panel.columns) == before
