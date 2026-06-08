"""Front-adjustment (前复权 / qfq) of a raw market panel using ``adj_factor``.

tushare ``daily`` returns RAW (unadjusted) OHLC plus a separate cumulative
``adj_factor`` series. Raw prices jump artificially across ex-dividend / split
dates; that fake jump pollutes return-based factors (e.g. momentum would see a
spurious negative return on an ex-date). Front adjustment removes it.

Convention (P1):

    qfq_price[t] = raw_price[t] * adj_factor[t] / adj_factor[latest_in_window]

anchored, PER SYMBOL, to that symbol's most recent date present in the panel.

Why this anchor is safe for this framework (the batch == incremental guarantee):
    The framework's factors are RETURN based (momentum = close[t]/close[t-w] - 1).
    The anchor ``adj_factor[latest]`` is a constant per-symbol multiplier that
    CANCELS in any price ratio, so every return — and therefore every factor value
    and the backtest's holding-period return — is INVARIANT to the anchor and to
    how far the window is extended. Only the absolute adjusted price *level* shifts
    when the window grows, and nothing downstream reads the absolute level. So we
    keep the PanelStore raw (with ``adj_factor``) and front-adjust in memory after
    reading: batch and incremental runs stay consistent (no silent re-anchoring of
    persisted data).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from data.clean.schema import validate_panel

# Price columns scaled by the adjustment ratio. volume/amount are left RAW
# (adjusting volume is a separate convention we deliberately do not adopt in P0).
ADJUSTABLE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "pre_close")


def front_adjust(
    panel: pd.DataFrame,
    price_columns: Sequence[str] = ADJUSTABLE_COLUMNS,
) -> pd.DataFrame:
    """Return a NEW panel with front-adjusted (qfq) prices.

    Requires an ``adj_factor`` column (the feed must preserve it, DATA-003).
    Symbols are adjusted independently. A panel whose ``adj_factor`` is all 1.0
    (e.g. :class:`DemoFeed`) is returned unchanged (identity), so demo runs and
    P0 tests are unaffected. Pure: never mutates the input.
    """
    if "adj_factor" not in panel.columns:
        raise ValueError(
            "front_adjust requires an 'adj_factor' column on the panel; the feed "
            "must preserve it (DATA-003)."
        )
    validate_panel(panel)

    out = panel.copy()
    adj = out["adj_factor"]
    # Panel is sorted by (date, symbol); within a symbol group the rows run in
    # ascending date order, so "last" is that symbol's most recent adj_factor.
    anchor = adj.groupby(level="symbol").transform("last")
    ratio = adj / anchor

    for col in price_columns:
        if col in out.columns:
            out[col] = out[col] * ratio
    return out
