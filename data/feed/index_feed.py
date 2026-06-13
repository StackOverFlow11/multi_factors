"""IndexConstituentsFeed: point-in-time index membership from tushare.

``index_weight`` returns periodic snapshots of an index's constituents
(``index_code, con_code, trade_date, weight``). This feed maps them onto the
canonical ``(date, symbol)`` shape (con_code -> symbol) so the PIT universe can
answer "who was in the index AS OF date d" using the latest snapshot <= d — with
no look-ahead and no survivorship bias (a name dropped later is still present in
the earlier snapshots).

The feed only pulls and normalizes membership; it does not decide tradability or
touch portfolio logic. The token is read from the external config and never
printed/logged (same contract as :class:`TushareFeed`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.feed.throttle import request_with_retry
from data.feed.tushare_feed import _lookup_dotted

# canonical constituents columns
CONSTITUENT_COLUMNS: tuple[str, ...] = ("date", "symbol", "weight")


class IndexConstituentsFeed:
    """Pulls PIT index constituents from tushare ``index_weight``."""

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
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None  # lazily built
        # P4-2: optional shared read-through cache. None keeps the historical
        # direct-fetch (paged) behaviour EXACTLY; only an opted-in config injects
        # one. The cache stores RAW snapshots; the as-of membership stays
        # downstream, unchanged.
        self._cache = cache

    # -- secret handling (token never logged) ------------------------------- #
    def _read_token(self) -> str:
        path = Path(self._secret_file)
        if not path.exists():
            raise ValueError(
                f"Secret config file not found: {self._secret_file}. "
                f"Set data.external_secret_file to your .config.json path."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Secret config file is not valid JSON: {self._secret_file} ({exc.msg})."
            ) from None
        return _lookup_dotted(data, self._token_key)

    def _client(self):
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(self._read_token())  # token handed straight in
        return self._pro

    # tushare index_weight caps a single response at ~6000 rows; a ~300-name
    # index therefore truncates beyond ~20 snapshots and SILENTLY drops the
    # earliest dates. We page the window in chunks small enough to stay under the
    # cap so no snapshot is lost.
    _WINDOW_DAYS = 90

    # -- API ---------------------------------------------------------------- #
    def get_constituents(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """Return constituent snapshots for ``index_code`` over [start, end].

        Paged in <=90-day windows to dodge tushare's per-call row cap (otherwise a
        full-year pull silently loses the earliest snapshots). Output columns:
        ``date`` (Timestamp), ``symbol`` (str), ``weight`` (float), sorted by
        (date, symbol), de-duplicated across window boundaries. Empty
        (schema-shaped) frame if tushare returns nothing — not an error.
        """
        if self._cache is not None:
            # Read-through: the cache plans gaps by index_code and pages each
            # uncovered gap in <=90-day windows (same cap rule). It returns the
            # canonical [date, symbol, weight] snapshots; the as-of/dedupe/sort
            # finalizer below is shared with the direct path (cached == direct).
            df = self._cache.index_weight(
                index_code, start, end, self._index_weight_fetch()
            )
            return self._finalize_constituents(df)

        pro = self._client()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        win_start = start_ts
        while win_start <= end_ts:
            win_end = min(win_start + pd.Timedelta(days=self._WINDOW_DAYS - 1), end_ts)
            raw = request_with_retry(
                pro.index_weight,
                max_retries=self._max_retries,
                rate_limit=self._rate_limit,
                index_code=index_code,
                start_date=win_start.strftime("%Y%m%d"),
                end_date=win_end.strftime("%Y%m%d"),
            )
            if raw is not None and len(raw) > 0:
                frames.append(raw)
            win_start = win_end + pd.Timedelta(days=1)

        if not frames:
            return self._empty()

        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"con_code": "symbol", "trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["symbol"] = df["symbol"].astype(str)
        return self._finalize_constituents(df)

    def _index_weight_fetch(self):
        """A ``(index_code, start_compact, end_compact) -> raw frame`` closure.

        The per-call retry/throttle stays HERE (the cache is transport-agnostic);
        the cache calls this once per uncovered <=90-day window. The client is
        built lazily inside the closure, so a fully-covered warm run reads no
        token and constructs no client.
        """

        def fetch(index_code: str, start_compact: str, end_compact: str):
            return request_with_retry(
                self._client().index_weight,
                max_retries=self._max_retries,
                rate_limit=self._rate_limit,
                index_code=index_code,
                start_date=start_compact,
                end_date=end_compact,
            )

        return fetch

    def _finalize_constituents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalize: select [date, symbol, weight], dedup, sort, reset index.

        Shared by the cache and direct paths so both produce a byte-identical
        frame. ``df`` is already canonical ([date, symbol, weight] present);
        an empty frame returns the schema-shaped empty.
        """
        if df is None or df.empty:
            return self._empty()
        out = df[list(CONSTITUENT_COLUMNS)].drop_duplicates(["date", "symbol"])
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            {"date": pd.Series([], dtype="datetime64[ns]"),
             "symbol": pd.Series([], dtype=object),
             "weight": pd.Series([], dtype=float)}
        )
