"""ProcessingPipeline neutralization wiring (covariates required, no silent skip)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.process.pipeline import ProcessingPipeline


def _factor_panel():
    syms = [f"{i:06d}.SZ" for i in range(6)]
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-03-04")], syms], names=["date", "symbol"]
    )
    return pd.DataFrame({"f": np.arange(6, dtype=float)}, index=idx)


def test_neutralize_enabled_without_covariates_raises():
    pipe = ProcessingPipeline(drop_missing=False, standardize=False, neutralize=True)
    with pytest.raises(ValueError, match="industry/market_cap"):
        pipe.transform(_factor_panel())


def test_neutralize_runs_with_covariates():
    fp = _factor_panel()
    industry = pd.Series(["A", "A", "A", "B", "B", "B"], index=fp.index)
    mcap = pd.Series(np.exp(np.arange(6) + 20.0), index=fp.index)
    pipe = ProcessingPipeline(
        drop_missing=True, standardize=True, neutralize=True,
        industry=industry, market_cap=mcap,
    )
    out = pipe.transform(fp)
    assert isinstance(out.index, pd.MultiIndex)
    assert out["f"].notna().any()
