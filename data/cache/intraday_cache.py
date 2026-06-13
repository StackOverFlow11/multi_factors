"""Read-through cache for tushare ``stk_mins`` raw 1min bars (I2).

``TushareIntradayCache`` turns a requested minute window into ONLY the uncovered
trading-day gaps, fetches those via a caller-supplied ``fetch`` callable (the
feed wraps its own retry/throttle there, so the cache stays transport-agnostic),
upserts the raw bars into the month-partitioned store, records coverage in the
intraday ledger (including empty returns), then returns the requested bars read
back from the cache.

Raw intraday source of truth is ``freq="1min"`` ONLY. Coarser bars (5/15/30/60min)
are derived later from cached 1min data, never fetched/cached here — a non-1min
request fails fast (before any fetch). Natural key is ``(symbol, freq, bar_end)``.

Behaviour the acceptance pins down:
  * full cache miss   -> fetch the uncovered day-range, populate cache;
  * full cache hit    -> ZERO fetch calls;
  * partial gap       -> fetch only the missing days;
  * empty return      -> still recorded as coverage (no needless refetch);
  * failed fetch      -> NOT recorded as coverage (a later run retries);
  * duplicate upsert  -> one row per (symbol, freq, bar_end).

Stored bars are RAW (``bar_end``/OHLCV/volume/amount/``source_trade_time``/``freq``);
the derived PIT fields (``bar_start``/``available_time``) are recomputed by
``normalize_intraday_bars`` after read. No token or secret ever reaches this
layer; the ledger holds endpoint metadata only.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import pandas as pd

from data.cache.intervals import subtract_intervals
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import KEY_COLS, STORED_COLUMNS, IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, ensure_raw_intraday_freq

_LOGGER = logging.getLogger("data.cache.intraday")

ENDPOINT = "stk_mins_1min"

# Columns returned to the feed (== feed._to_canonical output, so the cached path
# normalizes to a frame byte-identical to the direct path).
READ_COLUMNS: list[str] = [
    "time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source_trade_time",
]

# 1min A-share bars are ~240 rows/trading-day; <=23 calendar days (~16 trading
# days, ~3840 rows) keeps each call well under tushare's 8000-row single-call cap.
_FETCH_WINDOW_DAYS = 23

# A fetch callable: (symbol, start_dt_str, end_dt_str) -> raw stk_mins frame|None,
# where the datetime strings are 'YYYY-MM-DD HH:MM:SS'.
FetchIntraday = Callable[[str, str, str], "pd.DataFrame | None"]


def _fields_hash(columns: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(columns)).encode("utf-8")).hexdigest()[:16]


def _parse_stk_mins(raw: pd.DataFrame | None) -> pd.DataFrame:
    """tushare ``stk_mins`` frame -> raw-canonical stored rows (or empty).

    ``ts_code`` -> symbol, ``vol`` -> volume, ``trade_time`` -> ``bar_end`` (the
    bar's END) + ``source_trade_time`` (raw string, audit). ``freq`` is fixed to
    the only raw freq, 1min. Mirrors the feed's direct ``_to_canonical`` so the
    stored bars rebuild the exact same intraday frame.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=STORED_COLUMNS)
    df = raw.rename(columns={"ts_code": "symbol", "vol": "volume"}).copy()
    df["bar_end"] = pd.to_datetime(df["trade_time"])
    df["source_trade_time"] = df["trade_time"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["freq"] = RAW_INTRADAY_FREQ
    for col in STORED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[STORED_COLUMNS]


class TushareIntradayCache:
    """Endpoint-level read-through cache for stk_mins raw 1min bars."""

    def __init__(
        self,
        store: IntradayParquetStore,
        ledger: IntradayCoverageLedger,
        *,
        force_refresh: bool = False,
        fetch_window_days: int = _FETCH_WINDOW_DAYS,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._force_refresh = bool(force_refresh)
        self._window_days = max(1, int(fetch_window_days))
        self._clock = clock or pd.Timestamp.now
        self.fetch_counts: dict[str, int] = {ENDPOINT: 0}

    def stats(self) -> dict[str, int]:
        """Endpoint -> number of gap-window fetches sent to the API this instance."""
        return dict(self.fetch_counts)

    # -- read-through API --------------------------------------------------- #
    def stk_mins_1min(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fetch: FetchIntraday,
        *,
        freq: str = RAW_INTRADAY_FREQ,
    ) -> pd.DataFrame:
        """Return raw 1min bars for ``symbols`` over [start, end] (read-through).

        ``freq`` MUST be ``"1min"`` (checked before any fetch). Returns columns
        ``READ_COLUMNS`` (``time`` == ``bar_end``); the feed normalizes them into
        the canonical intraday panel. An all-miss-empty result is an empty frame.
        """
        ensure_raw_intraday_freq(freq)
        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)
        fields_hash = _fields_hash(STORED_COLUMNS)

        out: list[pd.DataFrame] = []
        for symbol in symbols:
            for gap_start, gap_end in self._day_gaps(symbol, req_start, req_end, freq):
                self._fetch_gap(symbol, freq, gap_start, gap_end, fetch, fields_hash)
            cached = self._store.read_range(ENDPOINT, symbol, freq, req_start, req_end)
            if not cached.empty:
                hit = cached.rename(columns={"bar_end": "time"})
                out.append(hit[READ_COLUMNS])
        _LOGGER.info(
            "cache %s: %d symbols (api window-calls=%d)",
            ENDPOINT, len(symbols), self.fetch_counts[ENDPOINT],
        )
        if not out:
            return pd.DataFrame(columns=READ_COLUMNS)
        return pd.concat(out, ignore_index=True).reset_index(drop=True)

    # -- planning ----------------------------------------------------------- #
    def _day_gaps(self, symbol, req_start, req_end, freq):
        """Uncovered trading-day intervals to fetch for ``symbol``."""
        if self._force_refresh:
            return [(req_start.normalize(), req_end.normalize())]
        covered = self._ledger.covered_day_intervals(ENDPOINT, symbol, freq)
        return subtract_intervals(req_start, req_end, covered)

    def _fetch_gap(self, symbol, freq, gap_start, gap_end, fetch, fields_hash):
        """Fetch one day-gap (paged by window), upsert raw bars, record coverage.

        Coverage is recorded for the WHOLE gap ONCE, AFTER every window of the gap
        succeeds (ok if any bar landed, else empty). If a window fetch raises, the
        exception propagates and coverage is NOT recorded, so the gap is retried
        on a later run (rows already upserted are deduped idempotently).
        """
        total_rows = 0
        win_start = pd.Timestamp(gap_start).normalize()
        gap_end_day = pd.Timestamp(gap_end).normalize()
        while win_start <= gap_end_day:
            win_end = min(
                win_start + pd.Timedelta(days=self._window_days - 1), gap_end_day
            )
            raw = fetch(
                symbol,
                win_start.strftime("%Y-%m-%d 00:00:00"),
                win_end.strftime("%Y-%m-%d 23:59:59"),
            )
            self.fetch_counts[ENDPOINT] = self.fetch_counts.get(ENDPOINT, 0) + 1
            parsed = _parse_stk_mins(raw)
            if not parsed.empty:
                self._store.upsert(ENDPOINT, symbol, freq, parsed, KEY_COLS)
                total_rows += len(parsed)
            win_start = win_end + pd.Timedelta(days=1)
        self._ledger.record(
            endpoint=ENDPOINT,
            key_type="symbol",
            key=symbol,
            raw_freq=freq,
            start_time=pd.Timestamp(gap_start).normalize(),
            end_time=gap_end_day,
            fields_hash=fields_hash,
            row_count=total_rows,
            status="ok" if total_rows else "empty",
            fetched_at=self._clock(),
        )
