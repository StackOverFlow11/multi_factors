"""Tests for analytics.factor (Slice 11, reqs ANA-001..003).

Analytics is the ONLY layer allowed to compute forward returns. These tests
cover cross-sectional IC, NaN-pair handling, and quantile-return shape. They use
the shared demo fixtures (no invented data) and never touch real artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.factor import (
    compute_ic,
    forward_returns,
    ic_summary,
    quantile_returns,
)
from data.clean.schema import INDEX_NAMES


def _toy_panel() -> pd.DataFrame:
    """A tiny 2-date cross-section where rank-IC is exactly known.

    Two dates, four symbols. On each date the factor and the forward return are
    PERFECTLY monotonically related, so the Spearman cross-sectional IC is +1.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    symbols = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=INDEX_NAMES)
    # factor ascending A<B<C<D on both dates
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], index=idx, name="f")
    # forward return monotonically increasing with factor (perfect rank corr)
    fwd = pd.Series([10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0], index=idx, name="r")
    return pd.DataFrame({"f": factor, "r": fwd})


def test_ic_computed_by_date_cross_section():
    """IC is computed per date as a cross-sectional rank correlation.

    With a perfectly monotone factor/return relation on each date, every
    per-date IC must be +1, indexed by date.
    """
    df = _toy_panel()
    ic = compute_ic(df["f"], df["r"], method="spearman")

    assert isinstance(ic, pd.Series)
    # one IC value per date
    expected_dates = df.index.get_level_values("date").unique()
    assert len(ic) == len(expected_dates)
    assert set(pd.to_datetime(ic.index)) == set(expected_dates)
    # perfect monotone -> IC == 1 on each date
    assert np.allclose(ic.to_numpy(), 1.0)

    summary = ic_summary(ic)
    assert "ic_mean" in summary and "ic_ir" in summary
    assert np.isclose(summary["ic_mean"], 1.0)


def test_ic_ignores_nan_pairs():
    """NaN factor/return pairs are dropped before correlating, per date.

    We blank out one symbol's factor on one date. The IC for that date must
    still be computed from the remaining valid pairs (not NaN), and the other
    date is unaffected.
    """
    df = _toy_panel()
    f = df["f"].copy()
    r = df["r"].copy()
    # NaN the factor of symbol D on date 1, and the return of symbol A on date 1
    d1 = pd.Timestamp("2024-01-01")
    f.loc[(d1, "D")] = np.nan
    r.loc[(d1, "A")] = np.nan

    ic = compute_ic(f, r, method="spearman")
    # both dates still produce a finite IC
    assert not ic.isna().any()
    # remaining valid pairs on date 1 are still perfectly monotone -> IC 1.0
    assert np.isclose(ic.loc[d1], 1.0)


def test_quantile_returns_shape():
    """quantile_returns returns one mean forward return per (date, quantile).

    Shape contract: a DataFrame whose rows are dates, whose columns are the
    quantile buckets 1..q, with mean forward return per cell. Five distinct
    symbols across two dates split into 5 quantiles -> 5 columns.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    symbols = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=INDEX_NAMES)
    factor = pd.Series([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], index=idx, name="f", dtype=float)
    fwd = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1],
        index=idx,
        name="r",
        dtype=float,
    )

    q = quantile_returns(factor, fwd, quantiles=5)
    assert isinstance(q, pd.DataFrame)
    # rows = dates, columns = 5 quantile buckets
    assert q.shape == (2, 5)
    assert set(pd.to_datetime(q.index)) == set(dates)
    # each bucket holds exactly one symbol per date here -> values are the fwd ret
    # top quantile on date 1 is symbol E (factor 5, fwd 0.5)
    top_col = q.columns[-1]
    assert np.isclose(q.loc[dates[0], top_col], 0.5)


def test_forward_returns_uses_future_close_per_symbol():
    """forward_returns is future-looking per symbol (allowed only in analytics).

    For a strictly +1/day symbol, the 1-day forward return at date t equals
    close[t+1]/close[t]-1, and the LAST date must be NaN (no future bar).
    """
    from tests.fixtures.panel_factory import make_demo_panel

    panel = make_demo_panel()
    fwd = forward_returns(panel, periods=(1, 5, 20))

    assert list(fwd.index.names) == INDEX_NAMES
    assert "forward_return_1d" in fwd.columns
    assert "forward_return_5d" in fwd.columns
    assert "forward_return_20d" in fwd.columns

    rising = fwd.xs("000001.SZ", level="symbol")["forward_return_1d"].dropna()
    closes = panel.xs("000001.SZ", level="symbol")["close"]
    # rising +1/day: close goes 100,101,...; 1d fwd ret at t = (c[t+1]-c[t])/c[t]
    expected_first = (closes.iloc[1] - closes.iloc[0]) / closes.iloc[0]
    assert np.isclose(rising.iloc[0], expected_first)

    # last available date for this symbol has no t+1 close -> NaN
    last_date = panel.index.get_level_values("date").max()
    assert np.isnan(fwd.loc[(last_date, "000001.SZ"), "forward_return_1d"])


def test_forward_returns_never_inf_on_demo_feed():
    """forward_returns contains no +/-inf over the full DemoFeed calendar (MEDIUM-1).

    A non-positive close once produced inf returns (the falling symbol crossed
    zero on the long calendar), silently polluting IC / quantile stats. The
    falling symbol now stays strictly positive and the divide masks any
    non-positive denominator to NaN, so no inf may appear.
    """
    from data.feed.demo_feed import DemoFeed

    # The example-config window is long enough that the old linear-decay falling
    # symbol would have crossed zero — exactly where inf used to leak in.
    panel = DemoFeed(calendar_start="2024-01-01").get_bars(
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"],
        "2024-01-01",
        "2024-12-31",
    )
    fwd = forward_returns(panel, periods=(1, 5, 20))
    values = fwd.to_numpy()
    assert not np.isinf(values).any(), "forward_returns leaked +/-inf"
    # The falling symbol stays strictly positive (the root-cause fix).
    assert (panel.xs("000002.SZ", level="symbol")["close"] > 0).all()
