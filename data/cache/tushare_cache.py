"""Read-through cache for tushare market endpoints (P4-1: daily + adj_factor).

``TushareCache`` turns a per-symbol requested date range into ONLY the uncovered
gaps, fetches those via a caller-supplied ``fetch`` callable (the feed wraps its
own retry/throttle there, so the cache stays transport-agnostic), upserts the
raw rows, records coverage (including empty returns), then returns the full
requested range read back from the cache.

Behaviour the P4-1 acceptance pins down:
  * full cache miss  -> fetch every symbol's full range, populate cache;
  * full cache hit   -> ZERO fetch calls;
  * partial gap      -> fetch only the missing sub-range;
  * empty return     -> still recorded as coverage (no needless refetch);
  * duplicate upsert -> one row per ``(symbol, date)``.

Stored rows are RAW and canonical-shaped (``date`` as datetime, ``symbol`` as
str, native price/volume/amount or adj_factor). Joining + ``front_adjust`` stay
downstream, unchanged. No token or secret ever reaches this layer.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import pandas as pd

from data.cache.coverage import CoverageLedger
from data.cache.intervals import merge_intervals, subtract_intervals
from data.cache.parquet_store import CacheParquetStore

_LOGGER = logging.getLogger("data.cache.tushare")

# endpoint identifiers (also the names accepted in data.cache.force_refresh).
MARKET_DAILY = "market_daily"
ADJ_FACTOR = "adj_factor"

# canonical-raw column sets stored per endpoint.
_DAILY_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
_ADJ_COLUMNS = ["date", "symbol", "adj_factor"]
_KEY_COLS = ["date", "symbol"]

# tushare raw -> canonical name for the columns we keep.
_DAILY_RENAME = {"ts_code": "symbol", "trade_date": "date", "vol": "volume"}
_ADJ_RENAME = {"ts_code": "symbol", "trade_date": "date"}

# A fetch callable: (symbol, start_compact, end_compact) -> raw tushare frame|None.
FetchOne = Callable[[str, str, str], "pd.DataFrame | None"]


def _fields_hash(columns: list[str]) -> str:
    """Stable short hash of a field set (order-independent)."""
    return hashlib.sha1(",".join(sorted(columns)).encode("utf-8")).hexdigest()[:16]


def _compact(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


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


class TushareCache:
    """Endpoint-level read-through cache for ``market_daily`` and ``adj_factor``."""

    def __init__(
        self,
        store: CacheParquetStore,
        ledger: CoverageLedger,
        *,
        refresh_recent_days: int = 14,
        force_refresh: tuple[str, ...] | list[str] = (),
        today: pd.Timestamp | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
        source_version: str | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._refresh_recent_days = int(refresh_recent_days)
        self._force_refresh = set(force_refresh or ())
        self._today = pd.Timestamp(today).normalize() if today is not None else None
        self._clock = clock or pd.Timestamp.now
        self._source_version = source_version
        # per-instance endpoint fetch counters (cache stats; one increment per
        # gap actually sent to the API). A fully-covered repeat run leaves these
        # at zero — the read-through hit rate is observable from the run log.
        self.fetch_counts: dict[str, int] = {MARKET_DAILY: 0, ADJ_FACTOR: 0}

    def stats(self) -> dict[str, int]:
        """Endpoint -> number of gap fetches sent to the API this instance."""
        return dict(self.fetch_counts)

    # -- public endpoint readers ------------------------------------------- #
    def daily_bars(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        return self._read_through(
            MARKET_DAILY, symbols, start, end, fetch, _parse_daily, _DAILY_COLUMNS
        )

    def adj_factor(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        return self._read_through(
            ADJ_FACTOR, symbols, start, end, fetch, _parse_adj, _ADJ_COLUMNS
        )

    # -- read-through engine ----------------------------------------------- #
    def _read_through(
        self,
        endpoint: str,
        symbols: list[str],
        start: str,
        end: str,
        fetch: FetchOne,
        parse: Callable[["pd.DataFrame | None"], pd.DataFrame],
        columns: list[str],
    ) -> pd.DataFrame:
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()
        fields_hash = _fields_hash(columns)
        forced = endpoint in self._force_refresh

        out: list[pd.DataFrame] = []
        n_fetches = 0
        n_covered = 0
        for symbol in symbols:
            gaps = self._gaps_for(endpoint, symbol, req_start, req_end, forced)
            if gaps:
                n_fetches += len(gaps)
            else:
                n_covered += 1
            for gap_start, gap_end in gaps:
                self._fetch_gap(
                    endpoint, symbol, gap_start, gap_end, fetch, parse, fields_hash
                )
            cached = self._store.read_symbol(endpoint, symbol)
            if not cached.empty:
                mask = (cached["date"] >= req_start) & (cached["date"] <= req_end)
                hit = cached.loc[mask, columns]
                if not hit.empty:
                    out.append(hit)
        _LOGGER.info(
            "cache %s: %d symbols, %d gap-fetches, %d fully-covered (api calls=%d)",
            endpoint, len(symbols), n_fetches, n_covered, self.fetch_counts[endpoint],
        )
        if not out:
            return pd.DataFrame(columns=columns)
        return pd.concat(out, ignore_index=True)

    def _gaps_for(self, endpoint, symbol, req_start, req_end, forced):
        """The uncovered sub-intervals to fetch (+ a forced recent tail)."""
        if forced:
            return [(req_start, req_end)]
        covered = self._ledger.covered_intervals(endpoint, symbol)
        gaps = subtract_intervals(req_start, req_end, covered)
        recent = self._recent_tail(req_start, req_end)
        if recent is not None:
            gaps = merge_intervals(gaps + [recent])
        return gaps

    def _recent_tail(self, req_start, req_end):
        """Force-refetch the recent tail within ``refresh_recent_days`` of today."""
        if self._refresh_recent_days <= 0:
            return None
        today = self._today if self._today is not None else self._clock().normalize()
        threshold = today - pd.Timedelta(days=self._refresh_recent_days)
        if req_end < threshold:
            return None  # whole request is safely historical
        return (max(req_start, threshold), req_end)

    def _fetch_gap(self, endpoint, symbol, gap_start, gap_end, fetch, parse, fields_hash):
        """Fetch one gap, upsert raw rows, record coverage (incl. empty)."""
        raw = fetch(symbol, _compact(gap_start), _compact(gap_end))
        self.fetch_counts[endpoint] = self.fetch_counts.get(endpoint, 0) + 1
        parsed = parse(raw)
        row_count = len(parsed)
        if row_count:
            self._store.upsert_symbol(endpoint, symbol, parsed, _KEY_COLS)
        self._ledger.record(
            endpoint=endpoint,
            key_type="symbol",
            key=symbol,
            start_date=gap_start,
            end_date=gap_end,
            fields_hash=fields_hash,
            row_count=row_count,
            status="ok" if row_count else "empty",
            fetched_at=self._clock(),
            source_version=self._source_version,
        )
