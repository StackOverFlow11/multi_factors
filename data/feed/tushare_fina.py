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

from data.feed.secret import read_token
from data.feed.throttle import request_with_retry

# financial fields supported as P1 factors.
DEFAULT_FIELDS: tuple[str, ...] = ("roe", "netprofit_yoy")


class TushareFinancialFeed:
    """Loads ``fina_indicator`` records (with ann_date) from tushare."""

    def __init__(
        self,
        secret_file: str,
        token_key: str = "tushare.token",
        rate_limit: int | None = None,
        max_retries: int = 3,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None

    def _client(self):
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(read_token(self._secret_file, self._token_key))
        return self._pro

    def get_fina_indicator(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return financial records over [start, end] (filtered by ann_date).

        Output columns: ``symbol``, ``ann_date`` (str YYYYMMDD), ``end_date``, and
        the requested ``fields``. Reports announced in the window are returned; the
        as-of alignment to trade dates happens in the clean layer.
        """
        wanted = list(fields) if fields else list(DEFAULT_FIELDS)
        col_spec = ",".join(["ts_code", "ann_date", "end_date", *wanted])
        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = request_with_retry(
                pro.fina_indicator,
                max_retries=self._max_retries,
                rate_limit=self._rate_limit,
                ts_code=sym,
                start_date=s,
                end_date=e,
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

    @staticmethod
    def _empty(fields: list[str]) -> pd.DataFrame:
        cols = {"symbol": pd.Series([], dtype=object),
                "ann_date": pd.Series([], dtype=object),
                "end_date": pd.Series([], dtype=object)}
        for f in fields:
            cols[f] = pd.Series([], dtype=float)
        return pd.DataFrame(cols)
