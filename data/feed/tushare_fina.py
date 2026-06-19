"""TushareFinancialFeed: financial records with disclosure dates (ann_date).

Pulls tushare ``fina_indicator`` and returns the raw records WITH both ``ann_date``
(disclosure date) and ``end_date`` (report period). The point-in-time alignment
(``ann_date <= trade_date``) is done downstream in
:func:`data.clean.pit_financials.asof_financials`; this feed only fetches and
normalizes, it never joins to trade dates and never looks ahead.

Token is read from the external config and never printed/logged; the client is
built lazily.
"""

from __future__ import annotations

import pandas as pd

from data.cache.tushare_cache import FINA_FIELDS
from data.feed.secret import read_token
from data.feed.throttle import request_with_retry

# financial fields requested by default when a caller names none. The cache
# stores the full ``FINA_FIELDS`` superset regardless (so a subset warm never
# blocks a later different-subset request); this is only the default REQUEST set.
DEFAULT_FIELDS: tuple[str, ...] = ("roe", "netprofit_yoy")


class TushareFinancialFeed:
    """Loads ``fina_indicator`` records (with ann_date) from tushare."""

    def __init__(
        self,
        secret_file: str,
        token_key: str = "tushare.token",
        rate_limit: int | None = None,
        max_retries: int = 6,
        cache=None,
        scheduler=None,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None
        # P4-3: optional shared read-through cache. None keeps the historical
        # direct per-symbol fetch EXACTLY; the cache stores RAW fina_indicator
        # rows (with ann_date) only — the ann_date<=trade_date as-of alignment
        # stays downstream, byte-identical.
        self._cache = cache
        # D5: optional shared GlobalRateLimiter; None keeps the per-call throttle.
        self._scheduler = scheduler

    def _client(self):
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(read_token(self._secret_file, self._token_key))
        return self._pro

    def _call(self, fn, **kwargs):
        return request_with_retry(
            fn, max_retries=self._max_retries, rate_limit=self._rate_limit,
            scheduler=self._scheduler, **kwargs,
        )

    def get_fina_indicator(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return financial records whose REPORT PERIOD falls in [start, end].

        tushare ``fina_indicator`` filters ``start_date``/``end_date`` by the
        report period (``end_date``), NOT by the announcement date — every report
        whose period ends in the window is returned, regardless of when it was
        disclosed. Both ``ann_date`` (disclosure) and ``end_date`` (period) are
        returned; the point-in-time ``ann_date <= trade_date`` alignment is done
        downstream in :func:`data.clean.pit_financials.asof_financials`. This feed
        never joins to trade dates and never looks ahead.

        Output columns: ``symbol``, ``ann_date`` (str YYYYMMDD), ``end_date``, and
        the requested ``fields``.
        """
        wanted = list(fields) if fields else list(DEFAULT_FIELDS)
        col_spec = ",".join(["ts_code", "ann_date", "end_date", *wanted])

        if self._cache is not None:
            # The cache stores the CANONICAL SUPERSET (field-set independent), so a
            # warm for one config's fields never blocks another's. Fetch the
            # superset; select the requested subset on read.
            super_spec = ",".join(["ts_code", "ann_date", "end_date", *FINA_FIELDS])
            cached = self._cache.fina_indicator(
                symbols, start, end, self._fina_fetch(super_spec)
            )
            if cached.empty:
                return self._empty(wanted)
            cached = cached.copy()
            cached["symbol"] = cached["symbol"].astype(str)
            keep = ["symbol", "ann_date", "end_date", *wanted]
            return cached[[c for c in keep if c in cached.columns]].reset_index(drop=True)

        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = self._call(
                pro.fina_indicator, ts_code=sym, start_date=s, end_date=e,
                fields=col_spec,
            )
            if df is not None and len(df) > 0:
                frames.append(df)

        if not frames:
            return self._empty(wanted)

        out = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
        out["symbol"] = out["symbol"].astype(str)
        keep = ["symbol", "ann_date", "end_date", *wanted]
        return out[[c for c in keep if c in out.columns]]

    def _fina_fetch(self, col_spec: str):
        """`(symbol, s_compact, e_compact) -> raw fina_indicator frame` (period range).

        The client is built lazily inside the closure, so a fully-covered warm run
        reads no token and constructs no client. ``start_date``/``end_date`` filter
        by the REPORT PERIOD (end_date), matching the direct path exactly.
        """

        def fetch(symbol, start_compact, end_compact):
            return self._call(
                self._client().fina_indicator, ts_code=symbol,
                start_date=start_compact, end_date=end_compact, fields=col_spec,
            )

        return fetch

    @staticmethod
    def _empty(fields: list[str]) -> pd.DataFrame:
        cols = {"symbol": pd.Series([], dtype=object),
                "ann_date": pd.Series([], dtype=object),
                "end_date": pd.Series([], dtype=object)}
        for f in fields:
            cols[f] = pd.Series([], dtype=float)
        return pd.DataFrame(cols)
