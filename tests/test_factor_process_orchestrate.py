"""D4 R9: the processing orchestration home (factors.process.orchestrate).

Proves the orchestrator is behavior-preserving vs a directly-constructed
ProcessingPipeline (so ``qt.pipeline._process_factors`` delegating to it changes
no number), and that covariate extraction matches the retired inline logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.process.orchestrate import covariates_from_panel, process_factor_panel
from factors.process.pipeline import ProcessingPipeline

DATES = pd.bdate_range("2024-01-02", periods=8)
SYMS = ["A", "B", "C", "D"]
IDX = pd.MultiIndex.from_product([DATES, SYMS], names=["date", "symbol"])


def _factor_panel():
    rng = np.random.RandomState(3)
    return pd.DataFrame(
        {"f1": rng.normal(size=len(IDX)), "f2": rng.normal(size=len(IDX))}, index=IDX
    )


def test_orchestrator_matches_direct_pipeline_no_neutralize():
    fp = _factor_panel()
    got = process_factor_panel(fp, drop_missing=True, standardize=True)
    direct = ProcessingPipeline(drop_missing=True, standardize=True).transform(fp)
    pd.testing.assert_frame_equal(got, direct)


def test_orchestrator_matches_direct_pipeline_with_neutralize():
    fp = _factor_panel()
    rng = np.random.RandomState(4)
    industry = pd.Series(rng.choice(["X", "Y"], size=len(IDX)), index=IDX)
    market_cap = pd.Series(np.abs(rng.normal(size=len(IDX))) + 1.0, index=IDX)
    got = process_factor_panel(
        fp, drop_missing=True, standardize=True, neutralize=True,
        industry=industry, market_cap=market_cap,
    )
    direct = ProcessingPipeline(
        drop_missing=True, standardize=True, neutralize=True,
        industry=industry, market_cap=market_cap,
    ).transform(fp)
    pd.testing.assert_frame_equal(got, direct)


def test_orchestrator_does_not_mutate_input():
    fp = _factor_panel()
    before = fp.copy()
    process_factor_panel(fp, drop_missing=True, standardize=True)
    pd.testing.assert_frame_equal(fp, before)


def test_covariates_from_panel_extracts_present_columns():
    panel = pd.DataFrame(
        {"close": 1.0, "industry": "X", "market_cap": 2.0}, index=IDX
    )
    industry, market_cap = covariates_from_panel(panel)
    assert industry is not None and market_cap is not None
    pd.testing.assert_series_equal(industry, panel["industry"])
    pd.testing.assert_series_equal(market_cap, panel["market_cap"])


def test_covariates_from_panel_returns_none_when_absent():
    panel = pd.DataFrame({"close": 1.0}, index=IDX)
    industry, market_cap = covariates_from_panel(panel)
    assert industry is None and market_cap is None
