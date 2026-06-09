"""Shared tradability filtering for any Universe (used by static + index).

``apply_tradable_filters`` takes a candidate member list and a date cross-section
of the (enriched) panel and drops names that are not tradable that day:

  * missing close (always — UNI-004),
  * suspended    (if ``filters['suspended']`` and a ``suspended`` flag is present),
  * ST           (if ``filters['st']`` and an ``is_st`` flag is present),
  * at price limit (if ``filters['limit_up_down']`` and the limit flags are present).

Each optional filter is a no-op when its config toggle is off OR when the
corresponding flag column is absent (e.g. the offline DemoFeed carries no flags),
so demo/P0 runs are unaffected and a filter only bites once its data is wired in.
"""

from __future__ import annotations

import pandas as pd


def apply_tradable_filters(
    members: list[str],
    date: pd.Timestamp,
    panel: pd.DataFrame,
    filters: dict | None = None,
) -> list[str]:
    """Return the subset of ``members`` tradable on ``date`` (order preserved)."""
    if not members:
        return []
    if "close" not in panel.columns:
        raise ValueError("tradable() needs a panel with a 'close' column.")
    flt = filters or {}
    target = pd.Timestamp(date).normalize()
    try:
        cross = panel.xs(target, level="date")
    except KeyError:
        return []  # no market data for that date -> nothing tradable

    min_days = int(flt.get("min_listing_days") or 0)
    has_list_date = "list_date" in cross.columns

    out: list[str] = []
    for symbol in members:
        if symbol not in cross.index:
            continue
        row = cross.loc[symbol]
        if pd.isna(row["close"]):
            continue  # missing close (UNI-004, always)
        if flt.get("suspended") and bool(row.get("suspended", False)):
            continue
        if flt.get("st") and bool(row.get("is_st", False)):
            continue
        if flt.get("limit_up_down") and (
            bool(row.get("at_up_limit", False)) or bool(row.get("at_down_limit", False))
        ):
            continue
        # min_listing_days (UNI-008): exclude names younger than min_days as of
        # this date. A missing/NaT list_date is a DATA GAP, not a young name, so
        # the name is kept (never silently dropped); callers disclose the gap.
        if min_days > 0 and has_list_date:
            ld = row.get("list_date")
            if pd.notna(ld) and (target - pd.Timestamp(ld)).days < min_days:
                continue
        out.append(symbol)
    return out
