"""Endpoint metadata for the tushare read-through cache (D2 split from
``tushare_cache.py``).

Pure constants only — endpoint identifiers, canonical-raw column sets + natural
keys, the ``fina_indicator`` field superset, the index_weight paging window, and
the raw->canonical rename maps. No pandas logic, no I/O, no token. Imported by
both :mod:`data.cache.tushare_parsers` and the :class:`~data.cache.tushare_cache.
TushareCache` facade; the facade re-exports the public names (endpoint ids +
``FINA_FIELDS``) so ``from data.cache.tushare_cache import DAILY_BASIC`` etc. keep
working unchanged.
"""

from __future__ import annotations

# endpoint identifiers (also the names accepted in data.cache.force_refresh).
MARKET_DAILY = "market_daily"
ADJ_FACTOR = "adj_factor"
INDEX_WEIGHT = "index_weight"
SUSPEND_D = "suspend_d"
NAMECHANGE = "namechange"
STK_LIMIT = "stk_limit"
STOCK_BASIC = "stock_basic"
# P4-3 factor-support endpoints.
DAILY_BASIC = "daily_basic"          # dense per-symbol date-range (pe/pb/total_mv)
FINA_INDICATOR = "fina_indicator"    # per-symbol, report-period range, carries ann_date
INDEX_MEMBER_ALL = "index_member_all"  # per-symbol dimension (SW in/out intervals)

# every endpoint this cache knows (fetch_counts is seeded with all of them so a
# warm run reports an explicit 0 for an endpoint it never had to touch).
ALL_ENDPOINTS = (
    MARKET_DAILY, ADJ_FACTOR, INDEX_WEIGHT, SUSPEND_D, NAMECHANGE,
    STK_LIMIT, STOCK_BASIC, DAILY_BASIC, FINA_INDICATOR, INDEX_MEMBER_ALL,
)

# sentinel key for a global (whole-market) snapshot endpoint (stock_basic).
_GLOBAL_KEY = "__all__"

# canonical-raw column sets + natural keys stored per endpoint.
_DAILY_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
_ADJ_COLUMNS = ["date", "symbol", "adj_factor"]
_KEY_COLS = ["date", "symbol"]

_INDEX_WEIGHT_COLUMNS = ["index_code", "date", "symbol", "weight"]
_INDEX_WEIGHT_KEY = ["date", "symbol"]

_SUSPEND_COLUMNS = ["date", "symbol", "suspend_type"]
_SUSPEND_KEY = ["date", "symbol", "suspend_type"]

_STK_LIMIT_COLUMNS = ["date", "symbol", "up_limit", "down_limit"]
_STK_LIMIT_KEY = ["date", "symbol"]

_NAMECHANGE_COLUMNS = ["symbol", "start_date", "end_date", "name"]
_NAMECHANGE_KEY = ["symbol", "start_date", "end_date", "name"]

_STOCK_BASIC_COLUMNS = ["symbol", "list_date"]
_STOCK_BASIC_KEY = ["symbol"]

# P4-3 endpoints. daily_basic is a dense per-symbol date-range (like market bars);
# the stored ``date`` is the trade_date. fina_indicator's stored ``date`` is the
# REPORT-PERIOD end_date (the axis tushare filters on); ann_date is kept as a raw
# column for the downstream PIT as-of (ann_date <= trade_date) — never as the
# coverage axis. index_member_all is a per-symbol dimension of SW in/out intervals.
_DAILY_BASIC_COLUMNS = ["date", "symbol", "pe", "pb", "total_mv"]
_DAILY_BASIC_KEY = ["date", "symbol"]

# fina_indicator is field-set dependent: a per-(symbol) parquet keyed by
# (symbol, end_date, ann_date) CANNOT hold two different field sets for the same
# report (a later subset upsert would overwrite an earlier one and a covered
# interval would be reused with the wrong columns). So the cache ALWAYS fetches +
# stores the CANONICAL SUPERSET of every financial field the project supports; a
# caller (the feed) selects its requested subset on read. Coverage is therefore
# uniform (one field set) and a subset warm never blocks a later different-subset
# request. Must stay a superset of factors.compute.financial.SUPPORTED_FIELDS
# (guarded by a drift test).
FINA_FIELDS: tuple[str, ...] = ("roe", "netprofit_yoy", "grossprofit_margin")
_FINA_COLUMNS = ["date", "symbol", "ann_date", "end_date", *FINA_FIELDS]
_FINA_KEY = ["symbol", "end_date", "ann_date"]

_INDEX_MEMBER_COLUMNS = [
    "symbol", "l1_name", "l2_name", "l3_name", "in_date", "out_date"
]
_INDEX_MEMBER_KEY = ["symbol", "in_date", "out_date", "l1_name", "l2_name", "l3_name"]

# tushare per-call row cap forces index_weight to be paged in <=90-day windows.
_INDEX_WINDOW_DAYS = 90

# tushare raw -> canonical name for the columns we keep.
_DAILY_RENAME = {"ts_code": "symbol", "trade_date": "date", "vol": "volume"}
_ADJ_RENAME = {"ts_code": "symbol", "trade_date": "date"}
