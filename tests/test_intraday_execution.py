"""Intraday tail-rebalance execution tests (I4): cutoff vs exec, exec-to-exec."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_aggregate import asof_daily_features
from data.clean.intraday_schema import normalize_intraday_bars
from runtime.intraday_execution import (
    REASON_MISSING_PRICE,
    REASON_NO_BAR,
    ExecutionFill,
    IntradayExecutionConfig,
    build_execution_prices,
    resolve_fill,
    simulate_tail_rebalance,
)


def _norm(rows, data_lag="1min"):
    """rows = [(time_str, symbol, close), ...] -> normalized 1min bars."""
    cl = [r[2] for r in rows]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": cl,
            "high": [c + 0.5 if pd.notna(c) else c for c in cl],
            "low": [c - 0.5 if pd.notna(c) else c for c in cl],
            "close": cl,
            "volume": [100.0] * len(rows),
            "amount": [10.0 * c if pd.notna(c) else c for c in cl],
        }
    )
    return normalize_intraday_bars(df, freq="1min", data_lag=data_lag)


def _day(bars, sym, date="2024-01-02"):
    w = bars.reset_index()
    w["date"] = w["bar_end"].dt.normalize()
    return w[(w["symbol"] == sym) & (w["date"] == pd.Timestamp(date))]


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_unsupported_execution_model_rejected():
    with pytest.raises(ValueError, match="execution_model"):
        IntradayExecutionConfig(execution_model="tail_vwap")


def test_bad_execution_window_rejected():
    # window start must be strictly after the decision time
    with pytest.raises(ValueError, match="execution_window"):
        IntradayExecutionConfig(execution_window=("14:49:00", "14:56:59"))


# --------------------------------------------------------------------------- #
# signal cutoff vs execution timestamp (the core separation)
# --------------------------------------------------------------------------- #
def test_signal_cutoff_and_execution_timestamp_are_separate():
    rows = [
        ("2024-01-02 09:31:00", "000001.SZ", 10.0),
        ("2024-01-02 14:49:00", "000001.SZ", 11.0),
        ("2024-01-02 14:51:00", "000001.SZ", 12.0),  # execution bar (post-cutoff)
        ("2024-01-02 14:55:00", "000001.SZ", 99.0),
    ]
    bars = _norm(rows)
    fill = resolve_fill(
        "000001.SZ", pd.Timestamp("2024-01-02"),
        _day(bars, "000001.SZ"), IntradayExecutionConfig(),
    )
    # execution INTENTIONALLY uses the 14:51 bar, which is AFTER the 14:50 cutoff
    assert not fill.blocked
    assert fill.exec_time == pd.Timestamp("2024-01-02 14:51:00")
    assert fill.exec_price == 12.0

    # the same 14:51 (and 14:55) bars are NOT visible to the 14:50 signal:
    # perturbing them leaves the I3 daily feature byte-identical.
    base = asof_daily_features(bars)
    perturbed_rows = rows[:2] + [
        ("2024-01-02 14:51:00", "000001.SZ", -123.0),
        ("2024-01-02 14:55:00", "000001.SZ", -456.0),
    ]
    perturbed = asof_daily_features(_norm(perturbed_rows))
    pd.testing.assert_frame_equal(base, perturbed)


# --------------------------------------------------------------------------- #
# next_minute_close model
# --------------------------------------------------------------------------- #
def test_next_minute_close_uses_1451_bar():
    rows = [
        ("2024-01-02 14:51:00", "000001.SZ", 10.0),
        ("2024-01-02 14:52:00", "000001.SZ", 10.5),
    ]
    fill = resolve_fill(
        "000001.SZ", pd.Timestamp("2024-01-02"),
        _day(_norm(rows), "000001.SZ"), IntradayExecutionConfig(),
    )
    assert fill.exec_time == pd.Timestamp("2024-01-02 14:51:00")
    assert fill.exec_price == 10.0


def test_missing_1451_uses_first_bar_in_window():
    rows = [
        ("2024-01-02 14:53:00", "000001.SZ", 11.0),  # no 14:51/14:52
        ("2024-01-02 14:55:00", "000001.SZ", 11.5),
    ]
    fill = resolve_fill(
        "000001.SZ", pd.Timestamp("2024-01-02"),
        _day(_norm(rows), "000001.SZ"), IntradayExecutionConfig(),
    )
    assert fill.exec_time == pd.Timestamp("2024-01-02 14:53:00")  # first in window
    assert fill.exec_price == 11.0


def test_no_bar_in_window_is_blocked():
    rows = [("2024-01-02 14:00:00", "000001.SZ", 10.0)]  # nothing in 14:51-14:56:59
    fill = resolve_fill(
        "000001.SZ", pd.Timestamp("2024-01-02"),
        _day(_norm(rows), "000001.SZ"), IntradayExecutionConfig(),
    )
    assert fill.blocked
    assert fill.reason == REASON_NO_BAR
    assert fill.exec_price is None


def test_nan_price_is_blocked_missing_price():
    rows = [("2024-01-02 14:51:00", "000001.SZ", np.nan)]
    fill = resolve_fill(
        "000001.SZ", pd.Timestamp("2024-01-02"),
        _day(_norm(rows), "000001.SZ"), IntradayExecutionConfig(),
    )
    assert fill.blocked
    assert fill.reason == REASON_MISSING_PRICE
    assert fill.exec_time == pd.Timestamp("2024-01-02 14:51:00")  # the bar existed


# --------------------------------------------------------------------------- #
# holding return = exec-to-exec, NOT close-to-close
# --------------------------------------------------------------------------- #
def test_holding_return_is_execution_to_execution_not_close():
    rows = [
        # T: exec(14:51)=10.0, daily close(15:00)=10.5
        ("2024-01-02 14:51:00", "000001.SZ", 10.0),
        ("2024-01-02 15:00:00", "000001.SZ", 10.5),
        # T_next: exec(14:51)=12.0, daily close(15:00)=11.0
        ("2024-01-03 14:51:00", "000001.SZ", 12.0),
        ("2024-01-03 15:00:00", "000001.SZ", 11.0),
    ]
    bars = _norm(rows)
    weights = {
        pd.Timestamp("2024-01-02"): pd.Series({"000001.SZ": 1.0}),
        pd.Timestamp("2024-01-03"): pd.Series({"000001.SZ": 1.0}),
    }
    res = simulate_tail_rebalance(weights, bars)
    entry = pd.Timestamp("2024-01-02")
    exec_to_exec = 12.0 / 10.0 - 1.0          # 0.20
    close_to_close = 11.0 / 10.5 - 1.0         # ~0.0476
    assert res.period_returns[entry] == pytest.approx(exec_to_exec)
    assert res.period_returns[entry] != pytest.approx(close_to_close)
    assert res.holding_returns.loc[entry, "000001.SZ"] == pytest.approx(0.20)
    # exec prices are the 14:51 bars, not the 15:00 closes
    assert res.exec_prices.loc[entry, "000001.SZ"] == 10.0


# --------------------------------------------------------------------------- #
# portfolio period return excludes blocked symbols (no silent fallback)
# --------------------------------------------------------------------------- #
def test_blocked_symbol_excluded_and_logged():
    rows = [
        # A: tradable both days (entry 10 -> exit 11, r=0.10)
        ("2024-01-02 14:51:00", "000001.SZ", 10.0),
        ("2024-01-03 14:51:00", "000001.SZ", 11.0),
        # B: entry ok at T, but NO bar in the window at T_next -> blocked at exit
        ("2024-01-02 14:51:00", "000002.SZ", 20.0),
        ("2024-01-03 14:00:00", "000002.SZ", 25.0),  # outside execution window
    ]
    bars = _norm(rows)
    weights = {
        pd.Timestamp("2024-01-02"): pd.Series({"000001.SZ": 0.5, "000002.SZ": 0.5}),
        pd.Timestamp("2024-01-03"): pd.Series({"000001.SZ": 1.0}),
    }
    res = simulate_tail_rebalance(weights, bars)
    entry = pd.Timestamp("2024-01-02")
    # only A contributes: 0.5 * 0.10 ; B excluded (its weight earns nothing)
    assert res.period_returns[entry] == pytest.approx(0.5 * 0.10)
    blocked_syms = {(f.symbol, f.reason) for f in res.blocked}
    assert ("000002.SZ", REASON_NO_BAR) in blocked_syms


def test_build_execution_prices_matrix_and_log():
    rows = [
        ("2024-01-02 14:51:00", "000001.SZ", 10.0),
        ("2024-01-02 14:00:00", "000002.SZ", 20.0),  # not in window -> blocked NaN
    ]
    bars = _norm(rows)
    prices, fills = build_execution_prices(
        bars, [pd.Timestamp("2024-01-02")], ["000001.SZ", "000002.SZ"],
        IntradayExecutionConfig(),
    )
    assert prices.loc[pd.Timestamp("2024-01-02"), "000001.SZ"] == 10.0
    assert pd.isna(prices.loc[pd.Timestamp("2024-01-02"), "000002.SZ"])
    assert any(f.blocked and f.symbol == "000002.SZ" for f in fills)
    assert all(isinstance(f, ExecutionFill) for f in fills)
