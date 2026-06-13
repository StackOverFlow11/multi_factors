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

P4-1 implements ``market_daily`` and ``adj_factor`` only. Universe/tradability
(P4-2) and factor-support endpoints (P4-3) are deliberately out of scope here.
"""

from __future__ import annotations

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache

__all__ = ["CacheParquetStore", "CoverageLedger", "TushareCache"]
