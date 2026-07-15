"""TushareCovariatesFeed: covariates for neutralization (+ listing dates).

Provides the cross-sectional inputs the pipeline needs:

  * ``pit_sw_intervals(symbols, level)`` -> {symbol: [(industry_name, in_date, out_date)]}
        (tushare ``index_member_all``): the POINT-IN-TIME SW industry membership history
        at ``level`` (L1/L2/L3; default L1). This is the industry source the neutralizer
        uses (P2-3): aligned as-of the trade date in
        :func:`data.clean.pit_industry.asof_industry`, so a reclassification is
        respected and no future industry leaks into the past.
  * ``market_cap(symbols, s, e)`` -> DataFrame[date, symbol, market_cap]
        (tushare ``daily_basic.total_mv``, in 10k CNY; only the log is used, so
        units do not matter). Genuinely per-date.
  * ``value_ratios(symbols, s, e)`` -> DataFrame[date, symbol, pe, pb]
        (tushare ``daily_basic``; published same-day, PIT-safe by construction;
        the value-factor inversion happens in the pipeline, P3-5).
  * ``listing_dates(symbols)``    -> {symbol: list_date}  (``stock_basic.list_date``,
        for the ``min_listing_days`` buy filter, UNI-008).
  * ``all_a_symbols()``           -> [symbol, ...]  (every listed A-share symbol from
        the same ``stock_basic`` snapshot; resolves the "all-A" data-update warm with
        no new endpoint — the snapshot is the whole listed market, not symbol-filtered).
  * ``industry(symbols)``         -> {symbol: industry}  (``stock_basic.industry``,
        the CURRENT tag). Retained as an accessor but NO LONGER wired into
        neutralization — the current tag would broadcast a future industry onto past
        dates; the PIT intervals above replace it.

Token is read from the external config and never printed; the client is lazy.
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
        max_retries: int = 6,
        cache=None,
        scheduler=None,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None
        # Optional shared read-through cache: ``listing_dates`` (stock_basic, P4-2),
        # ``market_cap`` / ``value_ratios`` (daily_basic, P4-3) and
        # ``pit_sw_intervals`` (index_member_all, P4-3) all read through it.
        # None keeps the historical direct fetch EXACTLY.
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
        if self._cache is not None:
            df = self._cache.stock_basic(self._stock_basic_fetch())
            if df.empty:
                return {}
            wanted = set(map(str, symbols))
            df = df[df["symbol"].astype(str).isin(wanted)]
            out: dict[str, pd.Timestamp] = {}
            for r in df.itertuples():
                out[str(r.symbol)] = pd.to_datetime(
                    str(r.list_date), format="%Y%m%d", errors="coerce"
                )
            return out
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

    def all_a_symbols(self) -> list[str]:
        """Return EVERY listed A-share symbol from the ``stock_basic`` snapshot.

        tushare ``stock_basic`` returns the WHOLE listed market (it is NOT filtered
        by any symbols arg), so its snapshot already carries the full listed
        A-share list. This resolves the "all-A" universe with NO new endpoint: it
        reuses the same read-through ``stock_basic`` path as ``listing_dates`` (a
        fully-covered warm run makes ~0 calls). Only the symbol column is read —
        never the current-tag industry. Returns a sorted, de-duplicated list of
        ``str`` symbols (``[]`` if the snapshot is empty).
        """
        if self._cache is not None:
            df = self._cache.stock_basic(self._stock_basic_fetch())
            col = "symbol"
        else:
            df = self._stock_basic_fetch()()
            col = "ts_code"
        if df is None or len(df) == 0 or col not in df.columns:
            return []
        return sorted({str(s) for s in df[col].tolist()})

    def _stock_basic_fetch(self):
        """`() -> raw stock_basic frame` (ts_code, list_date) — global snapshot.

        The client is built lazily inside the closure, so a fully-covered warm
        run reads no token and constructs no client.
        """

        def fetch():
            return self._call(self._client().stock_basic, fields="ts_code,list_date")

        return fetch

    _SW_LEVEL_COLUMN = {"L1": "l1_name", "L2": "l2_name", "L3": "l3_name"}

    def pit_sw_intervals(
        self, symbols: list[str], level: str = "L1"
    ) -> dict[str, list[tuple]]:
        """Return SW membership history per symbol at ``level`` (for PIT industry, UNI-010).

        ``{symbol: [(industry_name, in_date, out_date), ...]}`` from tushare
        ``index_member_all`` (SW2021), where ``industry_name`` is the L1 / L2 / L3
        name per ``level`` (default ``L1`` — the 31 broad SW sectors, standard for
        industry neutralization and DOF-safe on small cross-sections). ``in_date``/
        ``out_date`` are Timestamps; ``out_date`` is ``None`` for an active membership.
        A symbol with no SW membership row is simply absent (the caller treats an
        absent symbol as a disclosed industry data gap → NaN, never the current tag).
        One call per symbol; the payload is a handful of intervals each.
        """
        col = self._SW_LEVEL_COLUMN.get(str(level).upper())
        if col is None:
            raise ValueError(
                f"industry level must be one of {list(self._SW_LEVEL_COLUMN)}; got {level!r}."
            )
        if self._cache is not None:
            frame = self._cache.index_member_all(symbols, self._index_member_fetch())
            return self._intervals_from_member(frame, col)
        pro = self._client()
        out: dict[str, list[tuple]] = {}
        for sym in symbols:
            df = self._call(pro.index_member_all, ts_code=sym)
            if df is None or len(df) == 0 or col not in df.columns:
                continue
            rows: list[tuple] = []
            for r in df.itertuples():
                in_d = pd.to_datetime(str(r.in_date), format="%Y%m%d", errors="coerce")
                out_raw = getattr(r, "out_date", None)
                out_d = (
                    pd.to_datetime(str(out_raw), format="%Y%m%d", errors="coerce")
                    if out_raw is not None and str(out_raw) not in ("", "None", "nan")
                    else None
                )
                rows.append((getattr(r, col), in_d, out_d))
            if rows:
                out[str(sym)] = rows
        return out

    def _index_member_fetch(self):
        """`(symbol) -> raw index_member_all frame` (per-symbol dimension)."""

        def fetch(symbol):
            return self._call(self._client().index_member_all, ts_code=symbol)

        return fetch

    def _intervals_from_member(
        self, frame: pd.DataFrame, col: str
    ) -> dict[str, list[tuple]]:
        """Build {symbol: [(name, in_date, out_date|None)]} from cached raw rows.

        Same shape as the direct path: ``out_date`` NaT -> None, a symbol with no
        usable row is simply absent. The cached rows already carry datetime
        in_date/out_date, so the per-symbol interval set is identical to a direct
        per-symbol fetch.
        """
        out: dict[str, list[tuple]] = {}
        if frame is None or frame.empty or col not in frame.columns:
            return out
        for sym, sub in frame.groupby(frame["symbol"].astype(str), sort=False):
            rows: list[tuple] = []
            for r in sub.itertuples():
                name = getattr(r, col)
                in_d = pd.Timestamp(r.in_date) if pd.notna(r.in_date) else None
                out_raw = getattr(r, "out_date", None)
                out_d = pd.Timestamp(out_raw) if pd.notna(out_raw) else None
                rows.append((name, in_d, out_d))
            if rows:
                out[str(sym)] = rows
        return out

    def value_ratios(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return DataFrame[date, symbol, pe, pb] from daily_basic (P3-5).

        The ratios are published same-day (PIT-safe by construction); the
        inversion to value_ep / value_bp (with non-positive guards) happens in
        the pipeline's value enrichment, not here.
        """
        if self._cache is not None:
            df = self._cache.daily_basic(symbols, start, end, self._daily_basic_fetch())
            if df.empty:
                return self._empty_value_ratios()
            df = df.copy()
            df["symbol"] = df["symbol"].astype(str)
            return df[["date", "symbol", "pe", "pb"]].reset_index(drop=True)
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
                fields="ts_code,trade_date,pe,pb",
            )
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return self._empty_value_ratios()
        out = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol"})
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d")
        out["symbol"] = out["symbol"].astype(str)
        return out[["date", "symbol", "pe", "pb"]]

    def market_cap(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Return DataFrame[date, symbol, market_cap] from daily_basic.total_mv."""
        if self._cache is not None:
            df = self._cache.daily_basic(symbols, start, end, self._daily_basic_fetch())
            if df.empty:
                return self._empty_market_cap()
            df = df.rename(columns={"total_mv": "market_cap"}).copy()
            df["symbol"] = df["symbol"].astype(str)
            return df[["date", "symbol", "market_cap"]].reset_index(drop=True)
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
            return self._empty_market_cap()
        out = pd.concat(frames, ignore_index=True).rename(
            columns={"ts_code": "symbol", "total_mv": "market_cap"}
        )
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d")
        out["symbol"] = out["symbol"].astype(str)
        return out[["date", "symbol", "market_cap"]]

    def _daily_basic_fetch(self):
        """`(symbol, s_compact, e_compact) -> raw daily_basic frame` (pe/pb/total_mv).

        One cached daily_basic call serves BOTH value_ratios (pe/pb) and
        market_cap (total_mv); the client is built lazily inside the closure.
        """

        def fetch(symbol, start_compact, end_compact):
            return self._call(
                self._client().daily_basic, ts_code=symbol,
                start_date=start_compact, end_date=end_compact,
                fields="ts_code,trade_date,pe,pb,total_mv",
            )

        return fetch

    @staticmethod
    def _empty_value_ratios() -> pd.DataFrame:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"),
             "symbol": pd.Series([], dtype=object),
             "pe": pd.Series([], dtype=float),
             "pb": pd.Series([], dtype=float)}
        )

    @staticmethod
    def _empty_market_cap() -> pd.DataFrame:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"),
             "symbol": pd.Series([], dtype=object),
             "market_cap": pd.Series([], dtype=float)}
        )
