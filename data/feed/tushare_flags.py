"""TushareFlagsFeed: tradability signals (suspension, ST, price limits).

Pulls the three tushare sources that drive the tradability filters and maps them
onto shapes the :mod:`data.clean.tradability` enrichment can join to the panel:

  * ``suspended(symbols, start, end)``  -> set of (date, symbol) suspended that day
      (tushare ``suspend_d``, suspend_type 'S').
  * ``st_intervals(symbols)``           -> {symbol: [(start, end|None, is_st)]}
      (tushare ``namechange``; a name containing 'ST' / '*ST' marks the interval).
  * ``limits(symbols, start, end)``     -> DataFrame[date, symbol, up_limit, down_limit]
      (tushare ``stk_limit``).

Token is read from the external config and never printed/logged. The client is
built lazily, so constructing the feed needs no network/credentials.
"""

from __future__ import annotations

import pandas as pd

from data.feed.secret import read_token
from data.feed.throttle import request_with_retry


class TushareFlagsFeed:
    """Loads suspension / ST / price-limit signals from tushare."""

    def __init__(
        self,
        secret_file: str,
        token_key: str = "tushare.token",
        rate_limit: int | None = None,
        max_retries: int = 6,
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

    # -- suspensions (停牌) -------------------------------------------------- #
    def suspended(self, symbols: list[str], start: str, end: str) -> set[tuple]:
        """Return the set of (Timestamp date, symbol) suspended over [start, end]."""
        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        out: set[tuple] = set()
        for sym in symbols:
            df = self._call(
                pro.suspend_d, ts_code=sym, start_date=s, end_date=e, suspend_type="S"
            )
            if df is None or len(df) == 0:
                continue
            for d in df["trade_date"].astype(str):
                out.add((pd.to_datetime(d, format="%Y%m%d"), str(sym)))
        return out

    # -- ST status (namechange) -------------------------------------------- #
    def st_intervals(self, symbols: list[str]) -> dict[str, list[tuple]]:
        """Return {symbol: [(start_ts, end_ts|None, is_st_bool), ...]} from names."""
        pro = self._client()
        result: dict[str, list[tuple]] = {}
        for sym in symbols:
            df = self._call(pro.namechange, ts_code=sym)
            if df is None or len(df) == 0:
                continue
            seen: set[tuple] = set()
            intervals: list[tuple] = []
            for _, row in df.iterrows():
                start = pd.to_datetime(str(row["start_date"]), format="%Y%m%d")
                end_raw = row["end_date"]
                end = (
                    None
                    if end_raw is None or pd.isna(end_raw)
                    else pd.to_datetime(str(end_raw), format="%Y%m%d")
                )
                name = str(row["name"])
                key = (start, end, name)
                if key in seen:
                    continue
                seen.add(key)
                intervals.append((start, end, "ST" in name.upper()))
            result[str(sym)] = intervals
        return result

    # -- price limits (涨跌停) ---------------------------------------------- #
    def limits(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return DataFrame[date, symbol, up_limit, down_limit] over [start, end]."""
        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = self._call(pro.stk_limit, ts_code=sym, start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame(
                {"date": pd.Series([], dtype="datetime64[ns]"),
                 "symbol": pd.Series([], dtype=object),
                 "up_limit": pd.Series([], dtype=float),
                 "down_limit": pd.Series([], dtype=float)}
            )
        df = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        df["symbol"] = df["symbol"].astype(str)
        return df[["date", "symbol", "up_limit", "down_limit"]]
