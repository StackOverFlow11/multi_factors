"""TushareCovariatesFeed: industry + market cap for neutralization.

Provides the two cross-sectional covariates the neutralizer needs:

  * ``industry(symbols)``         -> {symbol: industry}  (tushare ``stock_basic``).
  * ``market_cap(symbols, s, e)`` -> DataFrame[date, symbol, market_cap]
        (tushare ``daily_basic.total_mv``, in 10k CNY; only the log is used, so
        units do not matter).

Caveat (disclosed in the bias audit): ``stock_basic.industry`` is the CURRENT
industry tag, not a point-in-time history, so industry neutralization carries a
mild membership-style look-ahead. Market cap is genuinely per-date. Token is read
from the external config and never printed; the client is lazy.
"""

from __future__ import annotations

import pandas as pd

from data.feed.secret import read_token
from data.feed.throttle import request_with_retry


class TushareCovariatesFeed:
    """Loads industry (stock_basic) and market cap (daily_basic) from tushare."""

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

    def _call(self, fn, **kwargs):
        return request_with_retry(
            fn, max_retries=self._max_retries, rate_limit=self._rate_limit, **kwargs
        )

    def industry(self, symbols: list[str]) -> dict[str, str]:
        """Return {symbol: industry} for ``symbols`` (current tag, not PIT)."""
        pro = self._client()
        df = self._call(pro.stock_basic, fields="ts_code,industry")
        if df is None or len(df) == 0:
            return {}
        wanted = set(map(str, symbols))
        df = df[df["ts_code"].astype(str).isin(wanted)]
        return {str(r.ts_code): r.industry for r in df.itertuples()}

    def listing_dates(self, symbols: list[str]) -> dict[str, pd.Timestamp]:
        """Return {symbol: list_date} from ``stock_basic.list_date`` (for UNI-008).

        Used by the ``min_listing_days`` selection filter. A symbol absent from
        ``stock_basic`` simply does not appear in the map (the caller treats an
        unknown listing date as a disclosed data gap, never as a young name).
        """
        pro = self._client()
        df = self._call(pro.stock_basic, fields="ts_code,list_date")
        if df is None or len(df) == 0:
            return {}
        wanted = set(map(str, symbols))
        df = df[df["ts_code"].astype(str).isin(wanted)]
        out: dict[str, pd.Timestamp] = {}
        for r in df.itertuples():
            out[str(r.ts_code)] = pd.to_datetime(
                str(r.list_date), format="%Y%m%d", errors="coerce"
            )
        return out

    def market_cap(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return DataFrame[date, symbol, market_cap] from daily_basic.total_mv."""
        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = self._call(
                pro.daily_basic,
                ts_code=sym,
                start_date=s,
                end_date=e,
                fields="ts_code,trade_date,total_mv",
            )
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame(
                {"date": pd.Series([], dtype="datetime64[ns]"),
                 "symbol": pd.Series([], dtype=object),
                 "market_cap": pd.Series([], dtype=float)}
            )
        out = pd.concat(frames, ignore_index=True).rename(
            columns={"ts_code": "symbol", "total_mv": "market_cap"}
        )
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d")
        out["symbol"] = out["symbol"].astype(str)
        return out[["date", "symbol", "market_cap"]]
