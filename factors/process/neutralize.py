"""Cross-sectional industry + size neutralization (per-date OLS residual).

For each date's cross-section, regress the factor on ``[log(market_cap),
one-hot(industry)]`` and keep the residual — the part of the factor NOT explained
by size or industry. This removes the (usually unwanted) systematic exposure of a
raw factor to firm size and sector, which otherwise dominates cross-sectional
ranks.

Degenerate cross-sections (fewer than 3 valid names) and rows with a missing
factor / industry / non-positive market cap return NaN rather than a fabricated
value. The regression is solved with ``numpy.linalg.lstsq`` so collinear industry
dummies (which always sum to 1) are handled without raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_NAMES = 3  # need at least this many valid names to fit a cross-section


def neutralize_by_date(
    factor: pd.Series, industry: pd.Series, market_cap: pd.Series
) -> pd.Series:
    """Return the per-date residual of ``factor`` on industry dummies + log size."""
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError(
            "neutralize_by_date expects a MultiIndex(date, symbol) factor series."
        )
    out = pd.Series(np.nan, index=factor.index, dtype=float)
    for _, idx in factor.groupby(level="date").groups.items():
        out.loc[idx] = _residual_one_date(factor.loc[idx], industry, market_cap)
    return out


def _residual_one_date(
    y: pd.Series, industry: pd.Series, market_cap: pd.Series
) -> pd.Series:
    """OLS residual for a single date's cross-section (index-aligned to ``y``)."""
    df = pd.DataFrame(
        {
            "y": y,
            "ind": industry.reindex(y.index),
            "mc": market_cap.reindex(y.index),
        }
    )
    df["lmc"] = np.where(df["mc"] > 0, np.log(df["mc"]), np.nan)
    valid = df.dropna(subset=["y", "ind", "lmc"])

    out = pd.Series(np.nan, index=y.index, dtype=float)
    if len(valid) < _MIN_NAMES:
        return out  # too few names to regress -> leave NaN (no fabrication)

    dummies = pd.get_dummies(valid["ind"], drop_first=False).astype(float)
    design = np.column_stack([valid["lmc"].to_numpy(dtype=float), dummies.to_numpy()])
    target = valid["y"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    out.loc[valid.index] = target - design @ beta
    return out
