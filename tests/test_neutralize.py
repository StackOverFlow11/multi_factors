"""Tests for industry + size cross-sectional neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.process.neutralize import neutralize_by_date


def _cross_section(date, symbols, factor, industry, mcap):
    idx = pd.MultiIndex.from_product([[pd.Timestamp(date)], symbols], names=["date", "symbol"])
    return (
        pd.Series(factor, index=idx, dtype=float),
        pd.Series(industry, index=idx),
        pd.Series(mcap, index=idx, dtype=float),
    )


def test_residual_is_orthogonal_to_size_and_industry():
    syms = [f"{i:06d}.SZ" for i in range(12)]
    ind = ["A"] * 6 + ["B"] * 6
    log_mcap = np.linspace(20.0, 23.0, 12)
    mcap = np.exp(log_mcap)
    # factor = 3*log_mcap + industry offset (+5 / -5) + a pattern orthogonal to both
    extra = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    offset = np.where(np.array(ind) == "A", 5.0, -5.0)
    factor = 3.0 * log_mcap + offset + extra
    f, i, m = _cross_section("2024-03-04", syms, factor, ind, mcap)

    resid = neutralize_by_date(f, i, m)
    # residual must be (numerically) uncorrelated with log market cap ...
    assert abs(np.corrcoef(resid.to_numpy(), log_mcap)[0, 1]) < 1e-6
    # ... and have ~zero mean within each industry (size+industry removed)
    by_ind = resid.groupby(pd.Series(ind, index=resid.index)).mean()
    assert by_ind.abs().max() < 1e-9


def test_missing_inputs_become_nan_not_fabricated():
    syms = [f"{i:06d}.SZ" for i in range(8)]  # enough names to keep DOF > 0
    f, i, m = _cross_section(
        "2024-03-04",
        syms,
        list(range(1, 9)),
        ["A", "A", "A", "A", "B", "B", "B", "B"],
        list(np.exp(np.arange(8) + 20.0)),
    )
    i.iloc[0] = np.nan  # missing industry for one name
    resid = neutralize_by_date(f, i, m)
    assert pd.isna(resid.iloc[0])  # missing-industry name -> NaN
    assert resid.iloc[1:].notna().all()  # the other 7 (DOF > 0) get residuals


def test_degenerate_cross_section_returns_nan():
    f, i, m = _cross_section("2024-03-04", ["A.SZ", "B.SZ"], [1.0, 2.0], ["X", "Y"], [10, 20])
    resid = neutralize_by_date(f, i, m)  # only 2 names (< 3) -> NaN
    assert resid.isna().all()


def test_saturated_cross_section_returns_nan_not_zeros():
    # 3 names, 2 industries -> n_params = 1 + 2 = 3, residual DOF = 0.
    # A saturated fit would yield ~0 residuals; we must return NaN instead.
    f, i, m = _cross_section(
        "2024-03-04", ["A.SZ", "B.SZ", "C.SZ"], [1.0, 2.0, 3.0], ["X", "X", "Y"], [10, 20, 30]
    )
    resid = neutralize_by_date(f, i, m)
    assert resid.isna().all()


def test_independent_per_date():
    syms = [f"{i:06d}.SZ" for i in range(6)]
    rows_f, rows_i, rows_m = [], [], []
    for d, base in [("2024-03-04", 0.0), ("2024-03-05", 100.0)]:
        f, i, m = _cross_section(
            d, syms, np.arange(6) + base, ["A", "A", "A", "B", "B", "B"], np.exp(np.arange(6) + 20)
        )
        rows_f.append(f)
        rows_i.append(i)
        rows_m.append(m)
    factor = pd.concat(rows_f)
    resid = neutralize_by_date(factor, pd.concat(rows_i), pd.concat(rows_m))
    # both dates produce finite residuals independently
    assert resid.groupby(level="date").apply(lambda s: s.notna().any()).all()
