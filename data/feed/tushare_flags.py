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
        cache=None,
        scheduler=None,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None
        # P4-2: optional shared read-through cache. None keeps the historical
        # direct per-symbol fetches EXACTLY; only an opted-in config injects one.
        # The cache stores RAW suspend_d / namechange / stk_limit rows — never a
        # derived flag as source of truth; the flag derivation stays here.
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

    # -- suspensions (停牌) -------------------------------------------------- #
    def suspended(self, symbols: list[str], start: str, end: str) -> set[tuple]:
        """Return the set of (Timestamp date, symbol) suspended over [start, end]."""
        if self._cache is not None:
            df = self._cache.suspend_d(symbols, start, end, self._suspend_fetch())
            return {
                (pd.Timestamp(d), str(sym))
                for d, sym in zip(df["date"], df["symbol"])
            }
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
        if self._cache is not None:
            frame = self._cache.namechange(symbols, self._namechange_fetch())
            return self._intervals_from_namechange(frame)
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
        if self._cache is not None:
            df = self._cache.stk_limit(symbols, start, end, self._stk_limit_fetch())
            if df.empty:
                return self._empty_limits()
            df = df.copy()
            df["symbol"] = df["symbol"].astype(str)
            return df[["date", "symbol", "up_limit", "down_limit"]].reset_index(drop=True)
        pro = self._client()
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = self._call(pro.stk_limit, ts_code=sym, start_date=s, end_date=e)
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return self._empty_limits()
        df = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        df["symbol"] = df["symbol"].astype(str)
        return df[["date", "symbol", "up_limit", "down_limit"]]

    # -- cache fetch closures (per-symbol retry/throttle stays here) -------- #
    # The tushare client is built lazily INSIDE each closure (on the first gap
    # actually fetched), so a fully-covered warm cache run reads no token and
    # constructs no client — the read-through path stays self-sufficient.
    def _suspend_fetch(self):
        """`(symbol, s_compact, e_compact) -> raw suspend_d frame` (type 'S')."""

        def fetch(symbol, start_compact, end_compact):
            return self._call(
                self._client().suspend_d, ts_code=symbol, start_date=start_compact,
                end_date=end_compact, suspend_type="S",
            )

        return fetch

    def _stk_limit_fetch(self):
        """`(symbol, s_compact, e_compact) -> raw stk_limit frame`."""

        def fetch(symbol, start_compact, end_compact):
            return self._call(
                self._client().stk_limit, ts_code=symbol, start_date=start_compact,
                end_date=end_compact,
            )

        return fetch

    def _namechange_fetch(self):
        """`(symbol) -> raw namechange frame` (dimension; no date range)."""

        def fetch(symbol):
            return self._call(self._client().namechange, ts_code=symbol)

        return fetch

    def _intervals_from_namechange(self, frame: pd.DataFrame) -> dict[str, list[tuple]]:
        """Build {symbol: [(start, end|None, is_st), ...]} from cached raw rows.

        Same dedupe (by start/end/name) and ST rule as the direct path, so the
        per-symbol interval SET is identical to a direct fetch (NaT end -> None).
        """
        result: dict[str, list[tuple]] = {}
        if frame is None or frame.empty:
            return result
        for sym, sub in frame.groupby(frame["symbol"].astype(str), sort=False):
            seen: set[tuple] = set()
            intervals: list[tuple] = []
            for _, row in sub.iterrows():
                start = pd.Timestamp(row["start_date"])
                end_raw = row["end_date"]
                end = None if pd.isna(end_raw) else pd.Timestamp(end_raw)
                name = str(row["name"])
                key = (start, end, name)
                if key in seen:
                    continue
                seen.add(key)
                intervals.append((start, end, "ST" in name.upper()))
            result[str(sym)] = intervals
        return result

    @staticmethod
    def _empty_limits() -> pd.DataFrame:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"),
             "symbol": pd.Series([], dtype=object),
             "up_limit": pd.Series([], dtype=float),
             "down_limit": pd.Series([], dtype=float)}
        )
