"""TushareIntradayFeed: 1min bars from tushare ``stk_mins`` (raw, PIT-ready).

A SEPARATE feed from :class:`data.feed.tushare_feed.TushareFeed` — it does NOT
touch the daily ``get_bars`` path and does not compute factors or aggregate to
daily. It pulls raw minute bars and maps them onto the intraday schema
(:mod:`data.clean.intraday_schema`); minute-level PIT time fields (bar_start /
bar_end / available_time) are derived there.

Token handling mirrors the other tushare feeds: read from the EXTERNAL json
config (never hardcoded, never committed), handed straight to the SDK, never
printed/logged/exposed via repr. The client is built lazily, so constructing
this class needs no network and no credentials.

Field mapping (tushare ``stk_mins`` with ``freq=1min`` -> intraday schema)::

    ts_code    -> symbol
    trade_time -> source_trade_time (kept for audit) + time (== bar_end)
    vol        -> volume
    open/high/low/close/amount -> same

The upstream ``stk_mins`` may return rows in reverse-chronological order and has
a single-call row cap; the schema normalizer sorts and deduplicates. Bulk /
multi-day windowing and a read-through cache are deferred (stages I2+); this feed
is the minimal network-free-testable raw shape. Coarser intraday bars
(5/15/30/60min) are intentionally NOT fetched here; they are derived from cached
1min bars in a later clean/resample layer.
"""

from __future__ import annotations

import pandas as pd

from data.clean.intraday_schema import (
    INTRADAY_CORE_COLUMNS,
    RAW_INTRADAY_FREQ,
    empty_intraday_bars,
    ensure_raw_intraday_freq,
    normalize_intraday_bars,
)
from data.feed.secret import read_token
from data.feed.throttle import request_with_retry

# tushare raw column -> intraday-schema column.
_FIELD_MAP: dict[str, str] = {
    "ts_code": "symbol",
    "vol": "volume",
}


class TushareIntradayFeed:
    """Loads raw 1min bars from tushare ``stk_mins``."""

    def __init__(
        self,
        secret_file: str,
        token_key: str = "tushare.token",
        rate_limit: int | None = None,
        max_retries: int = 6,
        cache=None,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        # SEC-004: per-minute call cap + retry on transient errors. stk_mins has
        # an exact per-minute ceiling that was NOT stress-tested; batch use must
        # set a conservative rate_limit (see docs/data/tushare_permissions.md).
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None  # lazily built tushare pro client
        # I2: optional read-through 1min cache (TushareIntradayCache). None keeps
        # the historical direct per-symbol fetch path EXACTLY; only an opted-in
        # caller injects one. The cache stores RAW 1min bars only; normalization
        # still runs downstream, unchanged.
        self._cache = cache

    # -- secret handling / client ------------------------------------------ #
    def _client(self):
        """Build (once) and return the tushare pro client. Lazy + no logging."""
        if self._pro is None:
            import tushare as ts

            # Do NOT log/print the token. Hand it directly to the SDK.
            self._pro = ts.pro_api(read_token(self._secret_file, self._token_key))
        return self._pro

    # -- rate limit + retry (SEC-004) -------------------------------------- #
    def _call(self, fn, **kwargs):
        """Invoke a tushare endpoint with the shared retry + per-minute throttle."""
        return request_with_retry(
            fn,
            max_retries=self._max_retries,
            rate_limit=self._rate_limit,
            **kwargs,
        )

    # -- intraday API ------------------------------------------------------- #
    def get_minutes(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        freq: str = RAW_INTRADAY_FREQ,
        data_lag: str = "1min",
    ) -> pd.DataFrame:
        """Return normalized raw 1min bars for ``symbols`` over [start, end].

        ``start``/``end`` are minute-window bounds — pass full datetime strings
        (e.g. ``"2024-01-02 09:30:00"``); a date-only value resolves to that
        day's 00:00:00. ``freq`` is kept as an explicit raw-frequency guard and
        must be ``"1min"``. Coarser bars are derived from cached 1min data.
        ``data_lag`` sets ``available_time = bar_end + data_lag`` for PIT safety.
        An empty upstream return yields a schema-shaped empty frame (not an error).
        """
        if not symbols:
            raise ValueError(
                "TushareIntradayFeed.get_minutes requires a non-empty symbol list."
            )
        ensure_raw_intraday_freq(freq)

        if self._cache is not None:
            return self._get_minutes_cached(symbols, start, end, freq, data_lag)

        pro = self._client()
        start_fmt = self._fmt_dt(start)
        end_fmt = self._fmt_dt(end)

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            raw = self._call(
                pro.stk_mins,
                ts_code=symbol,
                freq=freq,
                start_date=start_fmt,
                end_date=end_fmt,
            )
            if raw is None or len(raw) == 0:
                continue
            frames.append(self._to_canonical(raw))

        if not frames:
            return empty_intraday_bars()

        combined = pd.concat(frames, ignore_index=True)
        return normalize_intraday_bars(combined, freq=freq, data_lag=data_lag)

    # -- read-through cache path (I2) --------------------------------------- #
    def _get_minutes_cached(self, symbols, start, end, freq, data_lag) -> pd.DataFrame:
        """Build the panel from the read-through 1min cache.

        The cache fetches only uncovered trading-day gaps (a warm identical
        request makes zero ``stk_mins`` calls); the per-symbol throttle + retry
        stay HERE via the ``self._call`` closure. The cache returns the SAME
        columns as ``_to_canonical``, so after ``normalize_intraday_bars`` the
        result is byte-identical to the direct path.
        """
        combined = self._cache.stk_mins_1min(
            symbols, start, end, self._stk_mins_fetch(), freq=freq
        )
        if combined.empty:
            return empty_intraday_bars()
        return normalize_intraday_bars(combined, freq=freq, data_lag=data_lag)

    def _stk_mins_fetch(self):
        """`(symbol, start_dt, end_dt) -> raw stk_mins frame` (1min, lazy client).

        The tushare client is built lazily INSIDE the closure (on the first gap
        actually fetched), so a fully-covered warm cache run reads no token and
        constructs no client. ``start_dt``/``end_dt`` are 'YYYY-MM-DD HH:MM:SS'.
        """

        def fetch(symbol, start_dt, end_dt):
            return self._call(
                self._client().stk_mins,
                ts_code=symbol,
                freq=RAW_INTRADAY_FREQ,
                start_date=start_dt,
                end_date=end_dt,
            )

        return fetch

    # -- mapping helpers ---------------------------------------------------- #
    def _to_canonical(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rename tushare columns and expose ``time`` + audit ``source_trade_time``.

        Returns columns (time, symbol, OHLCV, source_trade_time); the schema
        normalizer derives freq/bar_start/bar_end/available_time. ``time`` keeps
        minute precision — it is NOT normalized to midnight.
        """
        df = raw.rename(columns=_FIELD_MAP).copy()
        df["source_trade_time"] = df["trade_time"].astype(str)
        df["time"] = pd.to_datetime(df["trade_time"])
        df["symbol"] = df["symbol"].astype(str)

        keep = ["time", "symbol", *INTRADAY_CORE_COLUMNS, "source_trade_time"]
        present = [c for c in keep if c in df.columns]
        return df[present]

    @staticmethod
    def _fmt_dt(value: str) -> str:
        """Format a window bound as the ``'YYYY-MM-DD HH:MM:SS'`` stk_mins expects."""
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")

    def __repr__(self) -> str:  # never leak the token
        return (
            f"TushareIntradayFeed(secret_file={self._secret_file!r}, "
            f"token_key={self._token_key!r}, rate_limit={self._rate_limit!r})"
        )
