"""Persistent endpoint-level raw cache for tushare (P4-1: market bars).

This package sits BELOW the feeds: it stores RAW endpoint facts (unadjusted
OHLCV/amount and raw adj_factor) keyed by their natural key, plus a coverage
ledger that records which (endpoint, key, date-range) tuples have been fetched —
so the read-through planner can fetch ONLY uncovered gaps and a repeated run
makes zero API calls for already-covered ranges.

Design invariants (see tmp/context/tushare_cache_architecture.md):
  * the durable cache stays RAW — never qfq prices, never research semantics;
  * ``front_adjust`` still runs in memory downstream, unchanged;
  * PanelStore remains a per-run artifact layer, NOT the cache source of truth;
  * the cache and its ledger never store a token or any secret-file content.

Cached endpoints: ``market_daily`` / ``adj_factor`` (P4-1); ``index_weight`` /
``suspend_d`` / ``namechange`` / ``stk_limit`` / ``stock_basic`` (P4-2);
``daily_basic`` / ``fina_indicator`` / ``index_member_all`` (P4-3, factor
support). The separate intraday 1min cache (``stk_mins``) lives in
``data.cache.intraday_*`` with its own timestamp-interval ledger.
"""

from __future__ import annotations

from data.cache.coverage import CoverageLedger
from data.cache.intraday_cache import TushareIntradayCache
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache

__all__ = [
    "CacheParquetStore",
    "CoverageLedger",
    "IntradayCoverageLedger",
    "IntradayParquetStore",
    "TushareCache",
    "TushareIntradayCache",
]
