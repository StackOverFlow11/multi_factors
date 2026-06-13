"""Read-through cache for tushare endpoints (P4-1 market bars + P4-2 universe/
tradability + P4-3 factor-support: daily_basic / fina_indicator / index_member_all).

``TushareCache`` turns a requested range into ONLY the uncovered gaps (or a
stale snapshot), fetches those via a caller-supplied ``fetch`` callable (the
feed wraps its own retry/throttle there, so the cache stays transport-agnostic),
upserts the raw rows, records coverage (including empty / not-ready returns),
then returns the full requested data read back from the cache.

Three planning shapes share one engine:
  * dense per-symbol date-range (``market_daily``, ``adj_factor``, ``suspend_d``,
    ``stk_limit``, ``daily_basic``, ``fina_indicator``): coverage key = symbol,
    gaps subtracted per symbol, a trailing tail refetched (today-based
    ``refresh_recent_days``, or a per-endpoint range-trailing override — fina
    refetches recent report periods to catch LATE disclosures);
  * index-keyed date-range (``index_weight``): coverage key = index_code, gaps
    subtracted over the whole index, each uncovered gap paged in <=90-day
    windows (tushare's per-call row cap), raw snapshots stored;
  * snapshot / dimension (``namechange`` per-symbol, ``index_member_all``
    per-symbol, ``stock_basic`` global): no date range — refetched only when
    never fetched, stale beyond ``refresh_dimension_days``, or force-refreshed.

``fina_indicator`` is field-set dependent and ALWAYS stores the canonical
``FINA_FIELDS`` superset (one schema, one coverage), so a warm for one config's
fields never blocks another's. P4-3 also adds a ``not_ready`` pending window
(today's unpublished data is not frozen as covered).

Behaviour the acceptance pins down (each endpoint):
  * full cache miss  -> fetch the uncovered range, populate cache;
  * full cache hit   -> ZERO fetch calls;
  * partial gap      -> fetch only the missing sub-range;
  * empty return     -> still recorded as coverage (no needless refetch);
  * failed fetch     -> NOT recorded as coverage (a later run retries);
  * duplicate upsert -> one row per the endpoint's natural key.

Stored rows are RAW and canonical-shaped (``date`` as datetime, ``symbol`` as
str, native prices / weights / limits). The PIT as-of membership, raw price
limit checks, and ``front_adjust`` all stay downstream, unchanged. No token or
secret ever reaches this layer; the ledger holds endpoint metadata only.
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

# A fetch callable: (symbol, start_compact, end_compact) -> raw tushare frame|None.
FetchOne = Callable[[str, str, str], "pd.DataFrame | None"]
# A snapshot fetch: (symbol) -> raw frame|None (namechange) or () -> raw frame|None
# (stock_basic). Typed loosely; the cache only calls and parses the result.
FetchSnapshot = Callable[..., "pd.DataFrame | None"]


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


class TushareCache:
    """Endpoint-level read-through cache (market bars + universe/tradability)."""

    def __init__(
        self,
        store: CacheParquetStore,
        ledger: CoverageLedger,
        *,
        refresh_recent_days: int = 14,
        refresh_dimension_days: int = 30,
        force_refresh: tuple[str, ...] | list[str] = (),
        today: pd.Timestamp | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
        source_version: str | None = None,
        not_ready_days: int = 0,
        recent_tail_overrides: dict[str, int] | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._refresh_recent_days = int(refresh_recent_days)
        self._refresh_dimension_days = int(refresh_dimension_days)
        self._force_refresh = set(force_refresh or ())
        self._today = pd.Timestamp(today).normalize() if today is not None else None
        self._clock = clock or pd.Timestamp.now
        self._source_version = source_version
        # P4-3: how many trailing calendar days (ending today) are treated as
        # "may not be published yet". An EMPTY return inside this pending window is
        # recorded as ``not_ready`` (NOT coverage) so it is retried on a later run
        # — a 21:00 updater must not freeze today's unpublished row as covered.
        # 0 (default) disables this, keeping every existing endpoint's behaviour
        # byte-identical.
        self._not_ready_days = max(0, int(not_ready_days))
        # P4-3: per-endpoint override of the trailing-refetch window over the
        # REQUESTED range (not today-based) — e.g. fina_indicator refetches a long
        # trailing window of recent report periods to catch LATE disclosures.
        self._recent_tail_overrides = dict(recent_tail_overrides or {})
        # per-instance endpoint fetch counters (cache stats; one increment per
        # gap/window/snapshot actually sent to the API). A fully-covered repeat
        # run leaves these at zero — the read-through hit rate is observable from
        # the run log. Seeded with every known endpoint so a warm run reports an
        # explicit 0 even for endpoints it never had to touch.
        self.fetch_counts: dict[str, int] = {ep: 0 for ep in ALL_ENDPOINTS}
        # P4-3 summary counters (for the data-update summary): rows upserted and
        # not-ready (pending) records, per endpoint.
        self.written_counts: dict[str, int] = {ep: 0 for ep in ALL_ENDPOINTS}
        self.not_ready_counts: dict[str, int] = {ep: 0 for ep in ALL_ENDPOINTS}

    def stats(self) -> dict[str, int]:
        """Endpoint -> number of gap fetches sent to the API this instance."""
        return dict(self.fetch_counts)

    def update_summary(self) -> dict[str, dict[str, int]]:
        """Per-endpoint {requests, rows_written, not_ready} for the data updater."""
        return {
            ep: {
                "requests": int(self.fetch_counts.get(ep, 0)),
                "rows_written": int(self.written_counts.get(ep, 0)),
                "not_ready": int(self.not_ready_counts.get(ep, 0)),
            }
            for ep in ALL_ENDPOINTS
        }

    # -- dense per-symbol date-range endpoints ----------------------------- #
    def daily_bars(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        return self._read_through(
            MARKET_DAILY, symbols, start, end, fetch, _parse_daily,
            _DAILY_COLUMNS, _KEY_COLS,
        )

    def adj_factor(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        return self._read_through(
            ADJ_FACTOR, symbols, start, end, fetch, _parse_adj,
            _ADJ_COLUMNS, _KEY_COLS,
        )

    def suspend_d(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        """Canonical-raw suspension rows [date, symbol, suspend_type] over [start, end]."""
        return self._read_through(
            SUSPEND_D, symbols, start, end, fetch, _parse_suspend,
            _SUSPEND_COLUMNS, _SUSPEND_KEY,
        )

    def stk_limit(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        """Canonical-raw price-limit rows [date, symbol, up_limit, down_limit]."""
        return self._read_through(
            STK_LIMIT, symbols, start, end, fetch, _parse_stk_limit,
            _STK_LIMIT_COLUMNS, _STK_LIMIT_KEY,
        )

    # -- P4-3 factor-support endpoints ------------------------------------- #
    def daily_basic(
        self, symbols: list[str], start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        """Canonical-raw daily_basic rows [date, symbol, pe, pb, total_mv].

        Dense per-symbol date-range over the trade_date, exactly like the market
        bars (recent-tail refetch + not-ready pending window apply). The feed's
        ``market_cap`` / ``value_ratios`` select their columns from this one cache.
        """
        return self._read_through(
            DAILY_BASIC, symbols, start, end, fetch, _parse_daily_basic,
            _DAILY_BASIC_COLUMNS, _DAILY_BASIC_KEY,
        )

    def fina_indicator(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fetch: FetchOne,
    ) -> pd.DataFrame:
        """Canonical-raw fina_indicator rows [symbol, ann_date, end_date, *FINA_FIELDS].

        ALWAYS the SUPERSET (``FINA_FIELDS``) — the ``fetch`` closure MUST request
        every superset field, so a subset warm never blocks a later different-subset
        request (the caller selects its subset on read). Coverage is planned over
        the REPORT-PERIOD (``end_date``) range; to catch LATE disclosures of recent
        periods, a long trailing window is always refetched
        (``recent_tail_overrides[fina_indicator]``). ``ann_date`` is stored raw for
        the downstream as-of and is NEVER the coverage axis.
        """
        out = self._read_through(
            FINA_INDICATOR, symbols, start, end, fetch, _parse_fina,
            _FINA_COLUMNS, _FINA_KEY,
        )
        keep = ["symbol", "ann_date", "end_date", *FINA_FIELDS]
        present = [c for c in keep if c in out.columns]
        return out[present].reset_index(drop=True)

    def index_member_all(
        self, symbols: list[str], fetch: FetchSnapshot
    ) -> pd.DataFrame:
        """Canonical-raw SW interval rows for ``symbols`` (per-symbol dimension).

        Each symbol's membership history is refetched only when never fetched,
        stale beyond ``refresh_dimension_days``, or force-refreshed (slow-moving,
        like namechange). Returns the stored rows for the requested symbols; the
        level-name selection + as-of interval lookup stay the feed's job.
        """
        forced = INDEX_MEMBER_ALL in self._force_refresh
        fields_hash = _fields_hash(_INDEX_MEMBER_COLUMNS)
        out: list[pd.DataFrame] = []
        for sym in symbols:
            if forced or self._snapshot_stale(INDEX_MEMBER_ALL, sym):
                self._fetch_snapshot(
                    INDEX_MEMBER_ALL, sym, "symbol", lambda s=sym: fetch(s),
                    lambda raw, s=sym: _parse_index_member(raw, s),
                    _INDEX_MEMBER_KEY, fields_hash,
                )
            cached = self._store.read_symbol(INDEX_MEMBER_ALL, sym)
            if not cached.empty:
                out.append(cached[_INDEX_MEMBER_COLUMNS])
        _LOGGER.info(
            "cache %s: %d symbols (api calls=%d)",
            INDEX_MEMBER_ALL, len(symbols), self.fetch_counts[INDEX_MEMBER_ALL],
        )
        if not out:
            return pd.DataFrame(columns=_INDEX_MEMBER_COLUMNS)
        return pd.concat(out, ignore_index=True)

    # -- index-keyed date-range endpoint (index_weight, 90-day paged) ------- #
    def index_weight(
        self, index_code: str, start: str, end: str, fetch: FetchOne
    ) -> pd.DataFrame:
        """Canonical-raw index_weight snapshots [date, symbol, weight] over [start, end].

        Coverage is keyed by ``index_code`` and planned by date range; each
        uncovered gap is paged in <=90-day windows (tushare's per-call row cap).
        The latest-snapshot as-of membership stays the feed's job — this returns
        every stored snapshot in the window.
        """
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()
        forced = INDEX_WEIGHT in self._force_refresh
        fields_hash = _fields_hash(_INDEX_WEIGHT_COLUMNS)
        gaps = self._gaps_for(INDEX_WEIGHT, index_code, req_start, req_end, forced)
        for gap_start, gap_end in gaps:
            self._fetch_index_gap(index_code, gap_start, gap_end, fetch, fields_hash)
        _LOGGER.info(
            "cache %s: index=%s, %d gap-intervals (api calls=%d)",
            INDEX_WEIGHT, index_code, len(gaps), self.fetch_counts[INDEX_WEIGHT],
        )
        cached = self._store.read_symbol(INDEX_WEIGHT, index_code)
        if cached.empty:
            return pd.DataFrame(columns=["date", "symbol", "weight"])
        mask = (cached["date"] >= req_start) & (cached["date"] <= req_end)
        return cached.loc[mask, ["date", "symbol", "weight"]].reset_index(drop=True)

    # -- snapshot / dimension endpoints ------------------------------------ #
    def namechange(
        self, symbols: list[str], fetch: FetchSnapshot
    ) -> pd.DataFrame:
        """Canonical-raw namechange rows for ``symbols`` (per-symbol dimension).

        Each symbol's snapshot is refetched only when never fetched, stale beyond
        ``refresh_dimension_days``, or force-refreshed. Returns the stored rows
        for the requested symbols (the ST-interval shaping stays the feed's job).
        """
        forced = NAMECHANGE in self._force_refresh
        fields_hash = _fields_hash(_NAMECHANGE_COLUMNS)
        out: list[pd.DataFrame] = []
        for sym in symbols:
            if forced or self._snapshot_stale(NAMECHANGE, sym):
                self._fetch_snapshot(
                    NAMECHANGE, sym, "symbol", lambda s=sym: fetch(s),
                    lambda raw, s=sym: _parse_namechange(raw, s),
                    _NAMECHANGE_KEY, fields_hash,
                )
            cached = self._store.read_symbol(NAMECHANGE, sym)
            if not cached.empty:
                out.append(cached[_NAMECHANGE_COLUMNS])
        _LOGGER.info(
            "cache %s: %d symbols (api calls=%d)",
            NAMECHANGE, len(symbols), self.fetch_counts[NAMECHANGE],
        )
        if not out:
            return pd.DataFrame(columns=_NAMECHANGE_COLUMNS)
        return pd.concat(out, ignore_index=True)

    def stock_basic(self, fetch: FetchSnapshot) -> pd.DataFrame:
        """Canonical-raw stock_basic rows [symbol, list_date] (global dimension).

        One whole-market snapshot keyed by a global sentinel; refetched only when
        never fetched, stale beyond ``refresh_dimension_days``, or force-refreshed.
        The feed filters to the symbols it needs (list_date only).
        """
        forced = STOCK_BASIC in self._force_refresh
        fields_hash = _fields_hash(_STOCK_BASIC_COLUMNS)
        if forced or self._snapshot_stale(STOCK_BASIC, _GLOBAL_KEY):
            self._fetch_snapshot(
                STOCK_BASIC, _GLOBAL_KEY, "global", lambda: fetch(),
                lambda raw: _parse_stock_basic(raw), _STOCK_BASIC_KEY, fields_hash,
            )
        _LOGGER.info(
            "cache %s: global snapshot (api calls=%d)",
            STOCK_BASIC, self.fetch_counts[STOCK_BASIC],
        )
        cached = self._store.read_symbol(STOCK_BASIC, _GLOBAL_KEY)
        if cached.empty:
            return pd.DataFrame(columns=_STOCK_BASIC_COLUMNS)
        return cached[_STOCK_BASIC_COLUMNS].reset_index(drop=True)

    # -- read-through engine (dense per-symbol date-range) ----------------- #
    def _read_through(
        self,
        endpoint: str,
        symbols: list[str],
        start: str,
        end: str,
        fetch: FetchOne,
        parse: Callable[["pd.DataFrame | None"], pd.DataFrame],
        columns: list[str],
        key_cols: list[str],
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
                    endpoint, symbol, gap_start, gap_end, fetch, parse,
                    key_cols, fields_hash,
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
        return pd.concat(out, ignore_index=True).reset_index(drop=True)

    def _gaps_for(self, endpoint, symbol, req_start, req_end, forced):
        """The uncovered sub-intervals to fetch (+ a forced trailing tail)."""
        if forced:
            return [(req_start, req_end)]
        covered = self._ledger.covered_intervals(endpoint, symbol)
        gaps = subtract_intervals(req_start, req_end, covered)
        tail = self._tail_for(endpoint, req_start, req_end)
        if tail is not None:
            gaps = merge_intervals(gaps + [tail])
        return gaps

    def _tail_for(self, endpoint, req_start, req_end):
        """The trailing window always refetched for ``endpoint``.

        A per-endpoint override (``recent_tail_overrides``) refetches the trailing
        N days of the REQUESTED range — used by fina_indicator to catch LATE
        disclosures of recent report periods, independent of today. Otherwise the
        today-based recent tail (``refresh_recent_days``) applies.
        """
        override = self._recent_tail_overrides.get(endpoint)
        if override is not None:
            if override <= 0:
                return None
            return (max(req_start, req_end - pd.Timedelta(days=override - 1)), req_end)
        return self._recent_tail(req_start, req_end)

    def _recent_tail(self, req_start, req_end):
        """Force-refetch the recent tail within ``refresh_recent_days`` of today."""
        if self._refresh_recent_days <= 0:
            return None
        today = self._today if self._today is not None else self._clock().normalize()
        threshold = today - pd.Timedelta(days=self._refresh_recent_days)
        if req_end < threshold:
            return None  # whole request is safely historical
        return (max(req_start, threshold), req_end)

    def _pending_start(self):
        """First calendar day treated as 'may not be published yet' (or None)."""
        if self._not_ready_days <= 0:
            return None
        today = self._today if self._today is not None else self._clock().normalize()
        return today - pd.Timedelta(days=self._not_ready_days - 1)

    def _fetch_gap(
        self, endpoint, symbol, gap_start, gap_end, fetch, parse, key_cols, fields_hash
    ):
        """Fetch one gap, upsert raw rows, record coverage (incl. empty/not-ready)."""
        raw = fetch(symbol, _compact(gap_start), _compact(gap_end))
        self.fetch_counts[endpoint] = self.fetch_counts.get(endpoint, 0) + 1
        parsed = parse(raw)
        if len(parsed):
            self._store.upsert_symbol(endpoint, symbol, parsed, key_cols)
            self.written_counts[endpoint] = (
                self.written_counts.get(endpoint, 0) + len(parsed)
            )
        self._record_gap_coverage(
            endpoint, symbol, gap_start, gap_end, parsed, fields_hash
        )

    def _record_gap_coverage(
        self, endpoint, symbol, gap_start, gap_end, parsed, fields_hash
    ):
        """Record coverage for a fetched gap, carving out a not-ready pending tail.

        With ``not_ready_days == 0`` (default) this records ONE row for the whole
        gap (ok if any row else empty) — byte-identical to the historical
        behaviour. Otherwise the gap is split at the pending boundary: the
        historical part records ok/empty; the pending part records ``not_ready``
        when it returned no row (so a 21:00 updater never freezes today's
        unpublished data as covered).
        """
        pending_start = self._pending_start()
        if pending_start is None or gap_end < pending_start:
            self._record_one(endpoint, symbol, gap_start, gap_end, parsed,
                             "empty", fields_hash)
            return
        has_date = "date" in parsed.columns
        hist_end = pending_start - pd.Timedelta(days=1)
        if gap_start <= hist_end:
            hist = parsed[parsed["date"] <= hist_end] if has_date else parsed.iloc[0:0]
            self._record_one(endpoint, symbol, gap_start, hist_end, hist,
                             "empty", fields_hash)
        pend = parsed[parsed["date"] >= pending_start] if has_date else parsed
        self._record_one(endpoint, symbol, pending_start, gap_end, pend,
                         "not_ready", fields_hash)

    def _record_one(
        self, endpoint, symbol, start, end, rows, empty_status, fields_hash
    ):
        """Append one coverage row; ``empty_status`` is used when ``rows`` is empty."""
        n = len(rows)
        status = "ok" if n else empty_status
        if status == "not_ready":
            self.not_ready_counts[endpoint] = (
                self.not_ready_counts.get(endpoint, 0) + 1
            )
        self._ledger.record(
            endpoint=endpoint,
            key_type="symbol",
            key=symbol,
            start_date=start,
            end_date=end,
            fields_hash=fields_hash,
            row_count=n,
            status=status,
            fetched_at=self._clock(),
            source_version=self._source_version,
        )

    # -- index_weight gap fetch (paged in <=90-day windows) ---------------- #
    def _fetch_index_gap(self, index_code, gap_start, gap_end, fetch, fields_hash):
        """Page one uncovered gap into <=90-day windows, upsert, record coverage.

        Coverage is recorded for the WHOLE gap (ok if any snapshot landed, else
        empty) so a covered window never refetches; each window is one API call
        (counted), mirroring the feed's historical paging.
        """
        frames: list[pd.DataFrame] = []
        total_rows = 0
        win_start = gap_start
        while win_start <= gap_end:
            win_end = min(
                win_start + pd.Timedelta(days=_INDEX_WINDOW_DAYS - 1), gap_end
            )
            raw = fetch(index_code, _compact(win_start), _compact(win_end))
            self.fetch_counts[INDEX_WEIGHT] += 1
            parsed = _parse_index_weight(raw, index_code)
            if not parsed.empty:
                frames.append(parsed)
                total_rows += len(parsed)
            win_start = win_end + pd.Timedelta(days=1)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            self._store.upsert_symbol(
                INDEX_WEIGHT, index_code, combined, _INDEX_WEIGHT_KEY
            )
        self._ledger.record(
            endpoint=INDEX_WEIGHT,
            key_type="index_code",
            key=index_code,
            start_date=gap_start,
            end_date=gap_end,
            fields_hash=fields_hash,
            row_count=total_rows,
            status="ok" if total_rows else "empty",
            fetched_at=self._clock(),
            source_version=self._source_version,
        )

    # -- snapshot / dimension fetch (staleness policy) --------------------- #
    def _snapshot_stale(self, endpoint: str, key: str) -> bool:
        """Whether a dimension snapshot must be (re)fetched (never fetched/stale)."""
        last = self._ledger.snapshot_fetched_at(endpoint, key)
        if last is None:
            return True
        if self._refresh_dimension_days <= 0:
            return False  # only force_refresh re-pulls once present
        today = self._today if self._today is not None else self._clock().normalize()
        return (today - last.normalize()).days >= self._refresh_dimension_days

    def _fetch_snapshot(
        self, endpoint, key, key_type, fetch, parse, key_cols, fields_hash
    ):
        """Fetch one snapshot, upsert raw rows, record coverage (incl. empty)."""
        raw = fetch()
        self.fetch_counts[endpoint] = self.fetch_counts.get(endpoint, 0) + 1
        parsed = parse(raw)
        row_count = len(parsed)
        if row_count:
            self._store.upsert_symbol(endpoint, key, parsed, key_cols)
            self.written_counts[endpoint] = (
                self.written_counts.get(endpoint, 0) + row_count
            )
        self._ledger.record(
            endpoint=endpoint,
            key_type=key_type,
            key=key,
            start_date=None,
            end_date=None,
            fields_hash=fields_hash,
            row_count=row_count,
            status="ok" if row_count else "empty",
            fetched_at=self._clock(),
            source_version=self._source_version,
        )
