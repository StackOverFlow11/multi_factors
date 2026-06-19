"""Raw tushare endpoint parsers for the read-through cache (D2 split from
``tushare_cache.py``).

Each function turns a raw tushare frame into canonical-raw rows (``date`` as
datetime, ``symbol`` as str, native prices / weights / limits) or the endpoint's
empty schema. RAW only — no qfq, no derived flags, no PIT as-of logic; that all
stays downstream, unchanged. Behaviour is byte-identical to the pre-split
functions (the existing cache tests are the behaviour lock).
"""

from __future__ import annotations

import pandas as pd

from data.cache.tushare_specs import (
    FINA_FIELDS,
    _ADJ_COLUMNS,
    _ADJ_RENAME,
    _DAILY_BASIC_COLUMNS,
    _DAILY_COLUMNS,
    _DAILY_RENAME,
    _FINA_COLUMNS,
    _INDEX_MEMBER_COLUMNS,
    _INDEX_WEIGHT_COLUMNS,
    _NAMECHANGE_COLUMNS,
    _STK_LIMIT_COLUMNS,
    _STOCK_BASIC_COLUMNS,
    _SUSPEND_COLUMNS,
)


def _parse_daily(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``daily`` frame -> canonical-raw rows (or empty)."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_DAILY_COLUMNS)
    df = raw.rename(columns=_DAILY_RENAME).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    for col in _DAILY_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[_DAILY_COLUMNS]


def _parse_adj(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``adj_factor`` frame -> canonical-raw rows (or empty)."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_ADJ_COLUMNS)
    df = raw.rename(columns=_ADJ_RENAME).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    return df[_ADJ_COLUMNS]


def _parse_suspend(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``suspend_d`` frame -> canonical-raw rows (or empty).

    The feed queries with ``suspend_type='S'``; stored rows carry that type so
    the suspended-set the feed builds from ``(date, symbol)`` is unchanged.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_SUSPEND_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    if "suspend_type" not in df.columns:
        df["suspend_type"] = "S"
    df["suspend_type"] = df["suspend_type"].astype(str)
    return df[_SUSPEND_COLUMNS]


def _parse_stk_limit(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``stk_limit`` frame -> canonical-raw rows (or empty).

    up_limit / down_limit stay RAW price terms (the limit checks run before
    front-adjustment, as today); nothing here touches qfq.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_STK_LIMIT_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    for col in ("up_limit", "down_limit"):
        if col not in df.columns:
            df[col] = float("nan")
    return df[_STK_LIMIT_COLUMNS]


def _parse_index_weight(raw: pd.DataFrame | None, index_code: str) -> pd.DataFrame:
    """tushare ``index_weight`` frame -> canonical-raw rows (or empty).

    ``con_code`` -> symbol, ``trade_date`` -> date; ``index_code`` is fixed from
    the queried index. Raw snapshots only — the latest-snapshot as-of membership
    logic stays downstream (unchanged).
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_INDEX_WEIGHT_COLUMNS)
    df = raw.rename(columns={"con_code": "symbol", "trade_date": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    df["index_code"] = str(index_code)
    if "weight" not in df.columns:
        df["weight"] = float("nan")
    return df[_INDEX_WEIGHT_COLUMNS]


def _parse_namechange(raw: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """tushare ``namechange`` frame -> canonical-raw rows (or empty).

    ``end_date`` is NaT for an active (open) name; the feed maps NaT back to
    ``None`` when it builds the ST intervals, so the interval shape is unchanged.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_NAMECHANGE_COLUMNS)
    df = raw.copy()
    df["symbol"] = (
        df["ts_code"].astype(str) if "ts_code" in df.columns else str(symbol)
    )
    df["start_date"] = pd.to_datetime(
        df["start_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(
            df["end_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        df["end_date"] = pd.NaT
    df["name"] = df["name"].astype(str)
    return df[_NAMECHANGE_COLUMNS]


def _parse_stock_basic(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``stock_basic`` frame -> canonical-raw rows (or empty).

    Stores ``list_date`` as the raw compact string; the feed parses it to a
    Timestamp exactly as the direct path does (for the ``min_listing_days``
    selection filter). The current-tag ``industry`` is NOT stored — it must
    never re-enter neutralization (the PIT SW path replaced it).
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_STOCK_BASIC_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol"}).copy()
    df["symbol"] = df["symbol"].astype(str)
    if "list_date" not in df.columns:
        df["list_date"] = None
    df["list_date"] = df["list_date"].astype(str)
    return df[_STOCK_BASIC_COLUMNS]


def _parse_daily_basic(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``daily_basic`` frame -> canonical-raw rows (or empty).

    Stores pe / pb / total_mv RAW (published same-day, PIT-safe by construction);
    the value-ratio inversion and log-market-cap stay downstream, unchanged.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_DAILY_BASIC_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["symbol"] = df["symbol"].astype(str)
    for col in ("pe", "pb", "total_mv"):
        if col not in df.columns:
            df[col] = float("nan")
    return df[_DAILY_BASIC_COLUMNS]


def _parse_fina(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``fina_indicator`` frame -> canonical-raw SUPERSET rows (or empty).

    The stored ``date`` is the report-period ``end_date`` (the coverage axis);
    ``ann_date`` (disclosure) and ``end_date`` are kept RAW so the downstream
    ``ann_date <= trade_date`` as-of alignment is byte-identical to the direct
    path. ALL of :data:`FINA_FIELDS` are stored (missing -> NaN) so the schema is
    field-set independent.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_FINA_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol"}).copy()
    df["symbol"] = df["symbol"].astype(str)
    df["ann_date"] = df["ann_date"].astype(str) if "ann_date" in df.columns else None
    df["end_date"] = df["end_date"].astype(str) if "end_date" in df.columns else None
    df["date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    for f in FINA_FIELDS:
        if f not in df.columns:
            df[f] = float("nan")
    return df[_FINA_COLUMNS]


def _parse_index_member(raw: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """tushare ``index_member_all`` frame -> canonical-raw SW interval rows (or empty).

    Stores the L1/L2/L3 names + ``in_date``/``out_date`` (out NaT for an active
    membership). The level-name selection and the as-of interval lookup stay
    downstream (the feed builds {symbol: [(name, in, out)]}, unchanged).
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_INDEX_MEMBER_COLUMNS)
    df = raw.copy()
    df["symbol"] = (
        df["ts_code"].astype(str) if "ts_code" in df.columns else str(symbol)
    )
    for col in ("l1_name", "l2_name", "l3_name"):
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].astype(object)
    df["in_date"] = pd.to_datetime(
        df["in_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    if "out_date" in df.columns:
        df["out_date"] = pd.to_datetime(
            df["out_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        df["out_date"] = pd.NaT
    return df[_INDEX_MEMBER_COLUMNS]
