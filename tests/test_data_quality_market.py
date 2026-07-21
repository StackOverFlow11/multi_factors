"""D3 report-only daily-market quality checks — synthetic, no network/cache."""

from __future__ import annotations

import pandas as pd

from data.quality.market import (
    check_adj_factor,
    check_close_outside_range,
    check_duplicate_keys,
    check_extreme_returns,
    check_high_low_inversion,
    check_missing_dates,
    check_decreasing_adj_factor,
    check_negative_volume_amount,
    check_non_positive_ohlc,
    run_adj_factor_checks,
    run_market_checks,
)
from data.quality.report import HARD, WARNING, has_hard


def _clean(rows=None) -> pd.DataFrame:
    """A small clean daily panel (2 symbols x 3 days), no quality issues."""
    data = []
    for sym, base in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
        for i, d in enumerate(("2024-01-03", "2024-01-04", "2024-01-05")):
            px = base + i * 0.1
            data.append(
                {
                    "date": pd.Timestamp(d), "symbol": sym,
                    "open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                    "volume": 1000.0 + i, "amount": 1_000_000.0 + i,
                }
            )
    return pd.DataFrame(data if rows is None else rows)


def test_clean_panel_has_zero_hard_findings():
    findings = run_market_checks(_clean())
    assert findings == []
    assert not has_hard(findings)


def test_clean_panel_as_multiindex_also_clean():
    panel = _clean().set_index(["date", "symbol"])
    assert run_market_checks(panel) == []


def test_duplicate_keys_caught():
    df = _clean()
    dup = df.iloc[[0]].copy()
    df2 = pd.concat([df, dup], ignore_index=True)
    f = check_duplicate_keys(df2)
    assert f is not None and f.severity == HARD and f.check == "duplicate_keys"
    assert f.count == 2  # the original + the duplicate row


def test_non_positive_ohlc_caught():
    df = _clean()
    df.loc[0, "close"] = 0.0
    f = check_non_positive_ohlc(df)
    assert f is not None and f.severity == HARD
    assert f.count == 1
    assert f.examples[0]["symbol"] == "000001.SZ"


def test_high_low_inversion_caught():
    df = _clean()
    df.loc[2, "high"] = 1.0
    df.loc[2, "low"] = 5.0
    f = check_high_low_inversion(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_close_outside_range_caught():
    df = _clean()
    df.loc[1, "close"] = 999.0  # above high
    f = check_close_outside_range(df)
    assert f is not None and f.severity == HARD and f.count == 1
    assert f.examples[0]["close"] == 999.0


def test_negative_volume_amount_caught():
    df = _clean()
    df.loc[3, "volume"] = -5.0
    f = check_negative_volume_amount(df)
    assert f is not None and f.severity == HARD and f.count == 1


def test_extreme_returns_caught_as_warning():
    df = _clean()
    # 000001.SZ close jumps 10.x -> 50 on day 2 (pct_change > 0.5)
    df.loc[1, "close"] = 50.0
    df.loc[1, "high"] = 50.5
    f = check_extreme_returns(df, threshold=0.5)
    assert f is not None and f.severity == WARNING
    assert f.count >= 1
    assert "pct_change" in f.examples[0]


def test_extreme_returns_clean_when_below_threshold():
    assert check_extreme_returns(_clean(), threshold=0.5) is None


def test_adj_factor_non_positive_caught():
    adj = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
            "symbol": ["000001.SZ", "000001.SZ"],
            "adj_factor": [1.0, 0.0],  # second is invalid
        }
    )
    f = check_adj_factor(adj)
    assert f is not None and f.severity == HARD and f.count == 1
    findings = run_adj_factor_checks(adj)
    assert any(x.check == "invalid_adj_factor" for x in findings)


def test_missing_dates_caught_with_calendar():
    df = _clean()
    # drop 000001.SZ on 2024-01-04
    df = df[~((df["symbol"] == "000001.SZ") & (df["date"] == pd.Timestamp("2024-01-04")))]
    cal = ["2024-01-03", "2024-01-04", "2024-01-05"]
    f = check_missing_dates(df, cal)
    assert f is not None and f.severity == WARNING and f.count == 1
    assert f.examples[0] == {"symbol": "000001.SZ", "date": "2024-01-04"}


def test_missing_dates_none_without_calendar():
    assert check_missing_dates(_clean(), None) is None


def test_examples_bounded_to_five():
    # 8 rows all with close == 0 -> finding capped at 5 examples
    rows = [
        {
            "date": pd.Timestamp("2024-01-03"), "symbol": f"{i:06d}.SZ",
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 0.0,
            "volume": 1.0, "amount": 1.0,
        }
        for i in range(8)
    ]
    f = check_non_positive_ohlc(pd.DataFrame(rows))
    assert f.count == 8
    assert len(f.examples) == 5  # bounded


# --- decreasing adj_factor -------------------------------------------------- #
#
# A cumulative adjustment factor grows with each corporate action; it has no
# mechanism to shrink. front_adjust multiplies raw prices by it, so a decrease
# propagates into every downstream return. Fixtures below use the real defect.


def _adj(sym: str, values: list[float]) -> pd.DataFrame:
    days = pd.date_range("2024-01-03", periods=len(values), freq="D")
    return pd.DataFrame(
        {"date": days, "symbol": [sym] * len(values), "adj_factor": values}
    )


def test_decreasing_adj_factor_caught():
    # The 920627.BJ signature: the factor oscillates between two values, giving
    # alternating -57.27% / +134.04% steps. Two decreases in this window.
    f = check_decreasing_adj_factor(_adj("920627.BJ", [2.3387, 1.0, 2.3387, 1.0]))
    assert f is not None
    assert f.severity == WARNING
    assert f.count == 2
    assert f.check == "decreasing_adj_factor"
    findings = run_adj_factor_checks(_adj("920627.BJ", [2.3387, 1.0, 2.3387, 1.0]))
    assert any(x.check == "decreasing_adj_factor" for x in findings)


def test_rounding_scale_negative_steps_are_not_flagged():
    # Measured on the real CSI500 cache: ~18% of factor changes are negative and
    # essentially all sit between -0.0008% and -0.08% -- tushare rounding on
    # ex-dates, not corruption. Flagging them would bury the real defect in noise.
    assert check_decreasing_adj_factor(_adj("000001.SZ", [11.267, 11.2669, 11.2589])) is None


def test_monotonic_increasing_adj_factor_is_clean():
    assert check_decreasing_adj_factor(_adj("000001.SZ", [1.0, 1.05, 1.05, 1.30])) is None


def test_decreasing_adj_factor_does_not_bleed_across_symbols():
    # Two symbols each individually non-decreasing, but whose concatenation steps
    # DOWN at the boundary. A per-symbol check must see nothing.
    df = pd.concat([_adj("000001.SZ", [5.0, 6.0]), _adj("000002.SZ", [1.0, 1.2])])
    assert check_decreasing_adj_factor(df) is None


def test_decreasing_adj_factor_sorts_by_date_before_differencing():
    # Rows arriving out of date order must not manufacture a phantom decrease:
    # sorted, this series only ever rises.
    df = _adj("000001.SZ", [1.0, 2.0, 3.0]).iloc[::-1].reset_index(drop=True)
    assert check_decreasing_adj_factor(df) is None


def test_decreasing_adj_factor_reports_but_never_filters():
    # Report-only: the input frame is not mutated and no rows are dropped.
    df = _adj("920627.BJ", [2.3387, 1.0])
    before = df.copy(deep=True)
    check_decreasing_adj_factor(df)
    pd.testing.assert_frame_equal(df, before)
