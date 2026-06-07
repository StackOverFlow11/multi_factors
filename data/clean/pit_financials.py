"""Point-in-time as-of alignment of financial data by disclosure date (ann_date).

The correctness red-line: a financial figure (e.g. Q1 ROE) must only become visible
AFTER it was disclosed (``ann_date``), NOT from the period end (``end_date``). For
example a Q1 report with ``end_date=2024-03-31`` is typically announced weeks later
(``ann_date=2024-04-20``); joining on ``end_date`` would leak the figure into trade
dates 04-01..04-19 that could not have known it.

``asof_financials`` attaches, for each ``(trade_date, symbol)``, the values of the
LATEST report whose ``ann_date <= trade_date`` (a backward as-of join keyed on
``ann_date``). Reports not yet disclosed are invisible; early dates with no prior
disclosure get NaN. This is the only place financial features enter the panel, and
it is structurally incapable of look-ahead.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def asof_financials(
    index: pd.MultiIndex,
    fina: pd.DataFrame,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Return a ``(date, symbol)`` frame of as-of financial ``fields``.

    Args:
        index: target MultiIndex(date, symbol) to align onto.
        fina: financial records with columns ``symbol``, ``ann_date`` (disclosure
            date; str ``YYYYMMDD`` or datetime) and the requested ``fields``.
        fields: financial column names to attach (e.g. ``["roe"]``).

    Each output row carries the field values from the latest report announced on
    or before that ``trade_date``; rows before any disclosure are NaN.
    """
    fields = list(fields)
    missing = [f for f in fields if f not in fina.columns]
    if missing:
        raise ValueError(f"financial records are missing field(s): {missing}.")

    f = fina.copy()
    f["symbol"] = f["symbol"].astype(str)
    f["ann_date"] = pd.to_datetime(f["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    f = f.dropna(subset=["ann_date"])
    # de-dup identical disclosures; keep the last record per (symbol, ann_date)
    f = f.sort_values("ann_date").drop_duplicates(["symbol", "ann_date"], keep="last")

    keys = pd.DataFrame(
        {
            "date": index.get_level_values("date"),
            "symbol": index.get_level_values("symbol").astype(str),
            "_pos": range(len(index)),
        }
    )
    keys_sorted = keys.sort_values("date")

    merged = pd.merge_asof(
        keys_sorted,
        f[["symbol", "ann_date", *fields]],
        left_on="date",
        right_on="ann_date",
        by="symbol",
        direction="backward",  # latest ann_date <= trade_date (never future)
    ).sort_values("_pos")

    out = pd.DataFrame(index=index)
    for field in fields:
        out[field] = merged[field].to_numpy()
    return out
