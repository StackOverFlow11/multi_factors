"""Tests for ann_date point-in-time financial alignment (the disclosure red-line)."""

from __future__ import annotations

import pandas as pd

from data.clean.pit_financials import asof_financials


def _index(dates, symbols=("000001.SZ",)):
    return pd.MultiIndex.from_product(
        [pd.to_datetime(dates), list(symbols)], names=["date", "symbol"]
    )


def _fina():
    # prior annual (ann 2023-12-31) + Q1 (end 2024-03-31 but ANNOUNCED 2024-04-20)
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20231231", "20240420"],
            "end_date": ["20230930", "20240331"],
            "roe": [8.0, 3.1],
        }
    )


def test_asof_uses_ann_date_not_end_date():
    idx = _index(["2024-04-10", "2024-04-19", "2024-04-20", "2024-04-21"])
    out = asof_financials(idx, _fina(), ["roe"])

    def roe_on(d):
        return out.xs(pd.Timestamp(d), level="date")["roe"].iloc[0]

    # BEFORE the Q1 disclosure (ann 04-20): must still be the prior report (8.0),
    # NOT the Q1 figure (3.1) — an end_date join would wrongly leak 3.1 here.
    assert roe_on("2024-04-10") == 8.0
    assert roe_on("2024-04-19") == 8.0
    # ON/AFTER disclosure: the Q1 figure becomes visible.
    assert roe_on("2024-04-20") == 3.1
    assert roe_on("2024-04-21") == 3.1


def test_asof_carries_forward_report_disclosed_before_window():
    # a single report disclosed 2023-12-31; the whole trade window is mid-2024.
    fina = pd.DataFrame(
        {"symbol": ["000001.SZ"], "ann_date": ["20231231"],
         "end_date": ["20230930"], "roe": [8.0]}
    )
    idx = _index(["2024-06-03", "2024-06-10", "2024-06-17"])
    out = asof_financials(idx, fina, ["roe"])
    # the prior disclosed report carries forward to every trade date (no NaN gap)
    assert (out["roe"] == 8.0).all()


def test_asof_nan_before_first_disclosure():
    idx = _index(["2023-06-01"])  # before the earliest ann_date (2023-12-31)
    out = asof_financials(idx, _fina(), ["roe"])
    assert pd.isna(out["roe"].iloc[0])


def test_asof_no_future_leak_when_future_report_changes():
    idx = _index(["2024-04-19"])
    base = asof_financials(idx, _fina(), ["roe"])["roe"].iloc[0]
    # mutate the FUTURE (Q1, ann 04-20) report's value — must not change 04-19
    fina2 = _fina()
    fina2.loc[fina2["ann_date"] == "20240420", "roe"] = 999.0
    after = asof_financials(idx, fina2, ["roe"])["roe"].iloc[0]
    assert base == after == 8.0


def test_asof_dedupes_identical_disclosures():
    fina = pd.concat([_fina(), _fina()], ignore_index=True)  # duplicated rows
    idx = _index(["2024-04-21"])
    out = asof_financials(idx, fina, ["roe"])
    assert out["roe"].iloc[0] == 3.1


def _tied_fina():
    # 000001.SZ: the FY2023 annual (end 2023-12-31, roe 8.0) and the Q1-2024
    # report (end 2024-03-31, roe 3.1) are disclosed the SAME day (2024-04-20) —
    # a routine pattern (annual + Q1 share a disclosure date).
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "ann_date": ["20231231", "20240420", "20240420"],
            "end_date": ["20230930", "20231231", "20240331"],
            "roe": [9.0, 8.0, 3.1],
        }
    )


def _filler(n):
    """n unrelated symbols with one report each (varies frame size/composition)."""
    syms = [f"{i:06d}.SZ" for i in range(2, 2 + n)]
    return pd.DataFrame(
        {
            "symbol": syms,
            "ann_date": [f"2024042{i % 10}" for i in range(n)],
            "end_date": ["20240331"] * n,
            "roe": [float(i) for i in range(n)],
        }
    )


def test_same_day_disclosure_freshest_end_date_wins_across_frame_compositions():
    """Among reports sharing (symbol, ann_date) the FRESHEST period wins — and the
    pick must NOT depend on the frame's size/composition/row order.

    Regression for the unstable-dedup defect: the old single-key
    ``sort_values("ann_date")`` (quicksort, unstable) let the survivor of a tied
    (symbol, ann_date) pair flip with the surrounding frame — the same
    (trade_date, symbol) resolved to the FY annual in a small universe frame and
    to Q1 in a large one. Here every arrangement must pick the Q1 value (3.1),
    the latest end_date. Under the old code n=50 / n=200 / the reversed orders
    picked the stale annual (8.0) instead — this test is RED there.
    """
    idx = _index(["2024-04-21"])
    tie = _tied_fina()
    tie_rev = tie.iloc[::-1].reset_index(drop=True)
    arrangements = {
        "bare": tie,
        "small_frame": pd.concat([tie, _filler(5)], ignore_index=True),
        "mid_frame": pd.concat([tie, _filler(50)], ignore_index=True),
        "large_frame": pd.concat([tie, _filler(200)], ignore_index=True),
        "reversed_tie_rows": tie_rev,
        "reversed_small_frame": pd.concat([tie_rev, _filler(5)], ignore_index=True),
        "shuffled_large_frame": pd.concat([tie, _filler(200)], ignore_index=True)
        .sample(frac=1.0, random_state=7)
        .reset_index(drop=True),
    }
    for name, fina in arrangements.items():
        out = asof_financials(idx, fina, ["roe"])
        assert out["roe"].iloc[0] == 3.1, (
            f"{name}: same-day tie picked {out['roe'].iloc[0]} instead of the "
            f"freshest-period record (3.1) — the pick depends on the frame"
        )


def test_same_day_disclosure_freshest_end_date_wins_even_with_nan_fields():
    """Field coverage may differ among tied records (an older tied report can
    carry NaN where the fresher one has a value); freshest end_date still wins."""
    fina = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "ann_date": ["20240420", "20240420", "20240420"],
            "end_date": ["20230331", "20231231", "20240331"],
            "roe": [7.0, 8.0, 3.1],
            "netprofit_yoy": [None, None, 6.5],  # only the freshest record has npy
        }
    )
    idx = _index(["2024-04-21"])
    out = asof_financials(idx, fina, ["roe", "netprofit_yoy"])
    assert out["roe"].iloc[0] == 3.1
    assert out["netprofit_yoy"].iloc[0] == 6.5


def test_same_day_disclosure_without_end_date_column_still_works():
    """A fina frame lacking ``end_date`` degrades to per-(symbol, ann_date)
    dedup without raising (the tie-break simply has nothing to order by)."""
    fina = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20240420", "20240420"],
            "roe": [8.0, 3.1],
        }
    )
    idx = _index(["2024-04-21"])
    out = asof_financials(idx, fina, ["roe"])
    assert out["roe"].iloc[0] == 3.1  # stable sort keeps input order; last wins


def test_asof_multiple_symbols_independent():
    fina = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20240420", "20240101"],
            "roe": [3.1, 5.5],
        }
    )
    idx = _index(["2024-04-21"], symbols=["000001.SZ", "000002.SZ"])
    out = asof_financials(idx, fina, ["roe"])
    assert out.xs("000001.SZ", level="symbol")["roe"].iloc[0] == 3.1
    assert out.xs("000002.SZ", level="symbol")["roe"].iloc[0] == 5.5


def test_asof_raises_on_missing_field():
    import pytest

    with pytest.raises(ValueError, match="missing field"):
        asof_financials(_index(["2024-04-21"]), _fina(), ["nonexistent"])


def test_asof_aligns_multiple_fields_in_one_call():
    """P3-1: several financial fields are as-of aligned in a SINGLE pass.

    The multi-factor pipeline fetches all financial fields once and aligns them
    together; each field must independently honour ann_date <= trade_date.
    """
    fina = _fina().assign(netprofit_yoy=[12.0, -4.0])
    idx = _index(["2024-04-19", "2024-04-21"])
    out = asof_financials(idx, fina, ["roe", "netprofit_yoy"])
    # before the Q1 disclosure both fields still show the prior annual report.
    assert out["roe"].iloc[0] == 8.0
    assert out["netprofit_yoy"].iloc[0] == 12.0
    # after disclosure both switch together.
    assert out["roe"].iloc[1] == 3.1
    assert out["netprofit_yoy"].iloc[1] == -4.0
