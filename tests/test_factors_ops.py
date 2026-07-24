"""``factors.ops`` operator library tests (D2).

Locks the three shared conventions (per-symbol grouping / full-window leading
NaN / purity) AND the daily-factor rewrite equivalence: every factor that moved
its inline rolling onto ``factors.ops`` must produce bit-identical values to a
naive per-symbol loop written independently here (plain pandas arithmetic, no
ops import in the oracle).

Mutation-evidence notes (design §五.1: every invariance claim must be breakable):

* the leading-NaN tests fail if an operator's ``min_periods`` default is relaxed
  to 1 (run: edit ``ts_std`` to ``min_periods=1`` -> test_ts_std_full_window_
  leading_nan fails);
* the isolation tests fail if the groupby level is dropped (run: replace the
  per-symbol groupby in ``ts_lag`` with a bare ``shift`` -> test_ts_lag_never_
  crosses_symbols fails);
* the equivalence tests fail if any rewritten factor drifts from its pre-D2
  math (they ARE the semantic lock for commit A).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.compute.candidates import (
    LiquidityFactor,
    OvernightMomentumFactor,
    VolatilityFactor,
)
from factors.compute.momentum import MomentumFactor
from factors.ops import (
    log_positive,
    ts_lag,
    ts_mean,
    ts_pct_change,
    ts_std,
    ts_sum,
    ts_window_return,
)


def _panel(n_days: int = 30, symbols: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    """Deterministic MultiIndex(date, symbol) panel with close/open/amount."""
    rng = np.random.RandomState(7)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    index = pd.MultiIndex.from_product([dates, list(symbols)], names=["date", "symbol"])
    n = len(index)
    close = pd.Series(100.0 + np.cumsum(rng.randn(n)) * 0.5, index=index)
    return pd.DataFrame(
        {
            "close": close,
            "open": close * (1.0 + rng.randn(n) * 0.001),
            "amount": np.abs(rng.randn(n)) * 1e6 + 1e5,
        }
    )


def _series(panel: pd.DataFrame, col: str = "close") -> pd.Series:
    return panel[col]


# --------------------------------------------------------------------------- #
# ts_lag
# --------------------------------------------------------------------------- #
def test_ts_lag_is_strictly_backward_with_leading_nan():
    s = _series(_panel())
    lagged = ts_lag(s, 3)
    for sym in ("AAA", "BBB"):
        sub = s.xs(sym, level="symbol")
        lag_sub = lagged.xs(sym, level="symbol")
        assert lag_sub.iloc[:3].isna().all()
        np.testing.assert_array_equal(
            lag_sub.iloc[3:].to_numpy(), sub.iloc[:-3].to_numpy()
        )


def test_ts_lag_never_crosses_symbols():
    panel = _panel()
    s = _series(panel)
    base = ts_lag(s, 1).xs("BBB", level="symbol")
    # Perturb the OTHER symbol violently; BBB's lag must be bit-identical.
    poisoned = s.copy()
    mask = poisoned.index.get_level_values("symbol") == "AAA"
    poisoned[mask] = 1e12
    after = ts_lag(poisoned, 1).xs("BBB", level="symbol")
    pd.testing.assert_series_equal(base, after)


def test_ts_lag_rejects_non_positive_periods():
    s = _series(_panel())
    with pytest.raises(ValueError, match="positive integer"):
        ts_lag(s, 0)


# --------------------------------------------------------------------------- #
# ts_window_return / ts_pct_change
# --------------------------------------------------------------------------- #
def test_ts_window_return_matches_naive_per_symbol_loop():
    s = _series(_panel())
    out = ts_window_return(s, 5)
    for sym in ("AAA", "BBB"):
        sub = s.xs(sym, level="symbol")
        naive = sub / sub.shift(5) - 1.0
        pd.testing.assert_series_equal(
            out.xs(sym, level="symbol"), naive, check_names=False
        )


def test_ts_pct_change_first_row_per_symbol_is_nan():
    s = _series(_panel())
    out = ts_pct_change(s)
    for sym in ("AAA", "BBB"):
        sub = out.xs(sym, level="symbol")
        assert np.isnan(sub.iloc[0])
        naive = s.xs(sym, level="symbol").pct_change()
        pd.testing.assert_series_equal(sub, naive, check_names=False)


# --------------------------------------------------------------------------- #
# ts_std / ts_mean / ts_sum: full-window leading NaN + naive equivalence
# --------------------------------------------------------------------------- #
def test_ts_std_full_window_leading_nan():
    s = _series(_panel())
    out = ts_std(s, 10)
    for sym in ("AAA", "BBB"):
        sub = out.xs(sym, level="symbol")
        assert sub.iloc[:9].isna().all()
        assert sub.iloc[9:].notna().all()


def test_ts_std_matches_naive_ddof1():
    s = _series(_panel())
    out = ts_std(s, 10)
    sub = s.xs("AAA", level="symbol")
    naive = sub.rolling(10, min_periods=10).std(ddof=1)
    pd.testing.assert_series_equal(
        out.xs("AAA", level="symbol"), naive, check_names=False
    )


def test_ts_mean_and_sum_match_naive_full_window():
    s = _series(_panel())
    for op, agg in ((ts_mean, "mean"), (ts_sum, "sum")):
        out = op(s, 7)
        sub = s.xs("BBB", level="symbol")
        naive = getattr(sub.rolling(7, min_periods=7), agg)()
        pd.testing.assert_series_equal(
            out.xs("BBB", level="symbol"), naive, check_names=False
        )


def test_window_ops_reject_degenerate_windows():
    s = _series(_panel())
    with pytest.raises(ValueError):
        ts_std(s, 1)  # a ddof=1 std needs >= 2 observations
    with pytest.raises(ValueError):
        ts_mean(s, 0)
    with pytest.raises(ValueError):
        ts_sum(s, -3)
    with pytest.raises(ValueError):
        ts_window_return(s, 0)


# --------------------------------------------------------------------------- #
# log_positive
# --------------------------------------------------------------------------- #
def test_log_positive_maps_non_positive_to_nan_never_inf():
    s = pd.Series([2.0, 0.0, -1.0, np.nan, 5.0])
    out = log_positive(s)
    assert np.isclose(out.iloc[0], np.log(2.0))
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])
    assert np.isclose(out.iloc[4], np.log(5.0))
    assert not np.isinf(out.to_numpy()).any()


# --------------------------------------------------------------------------- #
# purity: operators never mutate their input
# --------------------------------------------------------------------------- #
def test_ops_do_not_mutate_input():
    s = _series(_panel())
    frozen = s.copy(deep=True)
    ts_lag(s, 2)
    ts_window_return(s, 5)
    ts_pct_change(s)
    ts_std(s, 5)
    ts_mean(s, 5)
    ts_sum(s, 5)
    log_positive(s)
    pd.testing.assert_series_equal(s, frozen)


# --------------------------------------------------------------------------- #
# Daily-factor rewrite equivalence: class output == independent naive oracle
# --------------------------------------------------------------------------- #
def test_momentum_factor_matches_naive_oracle():
    panel = _panel()
    out = MomentumFactor(window=5).compute(panel)
    price = panel["close"]
    naive = price / price.groupby(level="symbol").shift(5) - 1.0
    pd.testing.assert_series_equal(out, naive.rename("momentum_5"))


def test_volatility_factor_matches_naive_oracle():
    panel = _panel(n_days=40)
    out = VolatilityFactor(window=10).compute(panel)
    naive = panel["close"].groupby(level="symbol", group_keys=False).apply(
        lambda s: s.pct_change().rolling(10, min_periods=10).std(ddof=1)
    )
    pd.testing.assert_series_equal(out, naive.reindex(panel.index).rename("volatility_10"))


def test_liquidity_factor_matches_naive_oracle():
    panel = _panel(n_days=40)
    out = LiquidityFactor(window=10).compute(panel)
    mean_amt = panel["amount"].groupby(level="symbol", group_keys=False).apply(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    naive = np.log(mean_amt.where(mean_amt > 0))
    pd.testing.assert_series_equal(out, naive.reindex(panel.index).rename("liquidity_10"))


def test_overnight_momentum_factor_matches_naive_oracle():
    panel = _panel(n_days=40)
    out = OvernightMomentumFactor(window=10).compute(panel)
    open_px, close_px = panel["open"], panel["close"]
    prev_close = close_px.groupby(level="symbol").shift(1)
    ratio = (open_px / prev_close).where((open_px > 0) & (prev_close > 0))
    naive = np.log(ratio).groupby(level="symbol", group_keys=False).apply(
        lambda s: s.rolling(10, min_periods=10).sum()
    )
    pd.testing.assert_series_equal(
        out, naive.reindex(panel.index).rename("overnight_mom_10")
    )
