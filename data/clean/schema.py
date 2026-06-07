"""Panel schema contract for the cross-sectional multi-factor framework.

All core panels (market data, factors) share one canonical shape:

    index: MultiIndex(date, symbol)
        date   -> pandas.Timestamp, normalized to midnight (date only)
        symbol -> str, tushare style, e.g. "000001.SZ"
    sorted by (date, symbol), no duplicate (date, symbol) pairs.

This module defines that shape and the helpers to enforce it. It is pure:
every function returns a new object and never mutates its input.

IMPORTANT (design invariant): this module is panel *shape* only. It must NOT
compute forward returns or anything that touches the future — forward-return
computation lives in ``analytics`` (see CLAUDE.md invariant #1).
"""

from __future__ import annotations

import pandas as pd

# Minimal required market-panel columns.
CORE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
]

# Columns that may be present but are not required by the schema.
OPTIONAL_COLUMNS: list[str] = [
    "pre_close",
    "trade_status",
    "is_st",
    "limit_up",
    "limit_down",
    "industry",
    "market_cap",
]

# Canonical index level names.
DATE_LEVEL = "date"
SYMBOL_LEVEL = "symbol"
INDEX_NAMES: list[str] = [DATE_LEVEL, SYMBOL_LEVEL]


def _extract_date_symbol(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Pull (date, symbol) out of a frame whether they are columns or the index.

    Returns (date_series, symbol_series, payload_frame_without_keys).
    Raises a readable ValueError if neither layout is present.
    """
    # Case 1: already a (date, symbol) MultiIndex (any order / names).
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        names = list(df.index.names)
        if names == INDEX_NAMES or set(names) == set(INDEX_NAMES):
            reset = df.reset_index()
            date = reset[DATE_LEVEL]
            symbol = reset[SYMBOL_LEVEL]
            payload = reset.drop(columns=INDEX_NAMES)
            return date, symbol, payload
        # Unnamed 2-level index: assume (date, symbol) order.
        if all(n is None for n in names):
            date = df.index.get_level_values(0).to_series(index=range(len(df)))
            symbol = df.index.get_level_values(1).to_series(index=range(len(df)))
            payload = df.reset_index(drop=True)
            return date, symbol, payload

    # Case 2: date and symbol live as ordinary columns.
    if DATE_LEVEL in df.columns and SYMBOL_LEVEL in df.columns:
        reset = df.reset_index(drop=True)
        date = reset[DATE_LEVEL]
        symbol = reset[SYMBOL_LEVEL]
        payload = reset.drop(columns=INDEX_NAMES)
        return date, symbol, payload

    raise ValueError(
        "Panel must provide 'date' and 'symbol' either as a MultiIndex(date, symbol) "
        "or as two columns named 'date' and 'symbol'. "
        f"Got index names={list(df.index.names)} and columns={list(df.columns)}."
    )


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new panel in canonical shape.

    Accepts ``df`` with (date, symbol) either as columns or as a MultiIndex.
    Produces a fresh DataFrame with:
      - MultiIndex(date, symbol), date normalized to midnight Timestamp, symbol str
      - rows sorted by (date, symbol)
    Raises ValueError if any CORE_COLUMNS are missing, or on duplicate
    (date, symbol) pairs. NaN cell values are allowed (missing prices are legal).

    The input is never mutated.
    """
    date, symbol, payload = _extract_date_symbol(df)

    # Normalize key dtypes. .copy() keeps us off the caller's frame.
    date = pd.to_datetime(date).dt.normalize()
    date.name = DATE_LEVEL
    symbol = symbol.astype(str)
    symbol.name = SYMBOL_LEVEL

    payload = payload.copy()
    missing = [c for c in CORE_COLUMNS if c not in payload.columns]
    if missing:
        raise ValueError(
            f"Panel is missing required core columns: {missing}. "
            f"CORE_COLUMNS = {CORE_COLUMNS}."
        )

    out = payload.copy()
    out.index = pd.MultiIndex.from_arrays([date.to_numpy(), symbol.to_numpy()], names=INDEX_NAMES)

    if out.index.duplicated().any():
        dups = out.index[out.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "Panel has duplicate (date, symbol) index entries; each pair must be unique. "
            f"Example duplicates: {dups[:5]}."
        )

    out = out.sort_index()
    return out


def validate_panel(df: pd.DataFrame) -> None:
    """Assert that ``df`` already satisfies the canonical panel contract.

    Raises a readable ValueError if any invariant is violated. Returns None on
    success. Does not mutate or return a frame — use ``normalize_panel`` to fix
    shape.
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            "Panel index must be a 2-level MultiIndex(date, symbol); "
            f"got {type(df.index).__name__} with names={list(df.index.names)}."
        )
    if list(df.index.names) != INDEX_NAMES:
        raise ValueError(
            f"Panel index level names must be {INDEX_NAMES}; got {list(df.index.names)}."
        )

    date_level = df.index.get_level_values(DATE_LEVEL)
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        raise ValueError(
            f"Panel 'date' level must be datetime (pandas.Timestamp); got dtype {date_level.dtype}."
        )

    symbol_level = df.index.get_level_values(SYMBOL_LEVEL)
    if not all(isinstance(s, str) for s in symbol_level):
        raise ValueError("Panel 'symbol' level must contain only str values.")

    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing required core columns: {missing}.")

    if df.index.duplicated().any():
        raise ValueError("Panel has duplicate (date, symbol) index entries.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Panel index must be sorted by (date, symbol).")
