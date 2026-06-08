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
