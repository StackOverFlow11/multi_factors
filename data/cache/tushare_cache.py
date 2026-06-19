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

D2 internal layout: endpoint constants/specs live in
:mod:`data.cache.tushare_specs`, raw endpoint parsers in
:mod:`data.cache.tushare_parsers`, and the two leaf planning helpers in
:mod:`data.cache.tushare_planning`. This module keeps the public ``TushareCache``
facade + the read-through engine, and re-exports the endpoint ids + ``FINA_FIELDS``
for backward-compatible imports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from data.cache.coverage import CoverageLedger
from data.cache.intervals import merge_intervals, subtract_intervals
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_parsers import (
    _parse_adj,
    _parse_daily,
    _parse_daily_basic,
    _parse_fina,
    _parse_index_member,
    _parse_index_weight,
    _parse_namechange,
    _parse_stk_limit,
    _parse_stock_basic,
    _parse_suspend,
)
from data.cache.tushare_planning import _compact, _fields_hash
from data.cache.tushare_specs import (
    ADJ_FACTOR,
    ALL_ENDPOINTS,
    DAILY_BASIC,
    FINA_FIELDS,
    FINA_INDICATOR,
    INDEX_MEMBER_ALL,
    INDEX_WEIGHT,
    MARKET_DAILY,
    NAMECHANGE,
    STK_LIMIT,
    STOCK_BASIC,
    SUSPEND_D,
    _ADJ_COLUMNS,
    _DAILY_BASIC_COLUMNS,
    _DAILY_BASIC_KEY,
    _DAILY_COLUMNS,
    _FINA_COLUMNS,
    _FINA_KEY,
    _GLOBAL_KEY,
    _INDEX_MEMBER_COLUMNS,
    _INDEX_MEMBER_KEY,
    _INDEX_WEIGHT_COLUMNS,
    _INDEX_WEIGHT_KEY,
    _INDEX_WINDOW_DAYS,
    _KEY_COLS,
    _NAMECHANGE_COLUMNS,
    _NAMECHANGE_KEY,
    _STK_LIMIT_COLUMNS,
    _STK_LIMIT_KEY,
    _STOCK_BASIC_COLUMNS,
    _STOCK_BASIC_KEY,
    _SUSPEND_COLUMNS,
    _SUSPEND_KEY,
)

_LOGGER = logging.getLogger("data.cache.tushare")

# A fetch callable: (symbol, start_compact, end_compact) -> raw tushare frame|None.
FetchOne = Callable[[str, str, str], "pd.DataFrame | None"]
# A snapshot fetch: (symbol) -> raw frame|None (namechange) or () -> raw frame|None
# (stock_basic). Typed loosely; the cache only calls and parses the result.
FetchSnapshot = Callable[..., "pd.DataFrame | None"]


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
