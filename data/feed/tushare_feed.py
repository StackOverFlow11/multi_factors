"""TushareFeed: the real-API DataFeed (DATA-001/004, SEC-001/004).

Reads the tushare token from an EXTERNAL json config file (never hardcoded, never
committed) and maps tushare's daily fields onto the canonical panel schema. The
token is treated as a secret: it is never printed, logged, or exposed via repr.
The tushare client is built lazily so importing/constructing this class needs no
network and no credentials.

Field mapping (tushare ``daily`` -> CORE_COLUMNS):
    ts_code    -> symbol
    trade_date -> date
    open/high/low/close -> open/high/low/close
    vol        -> volume
    amount     -> amount
    adj_factor -> adj_factor   (joined from the ``adj_factor`` endpoint; 1.0 if absent)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.clean.schema import CORE_COLUMNS, normalize_panel
from data.feed.base import DataFeed
from data.feed.throttle import request_with_retry

# tushare raw column -> canonical column.
_FIELD_MAP: dict[str, str] = {
    "ts_code": "symbol",
    "trade_date": "date",
    "vol": "volume",
}


def _lookup_dotted(data: dict, dotted_key: str) -> str:
    """Resolve a dotted key (e.g. 'tushare.token') in a nested dict.

    Raises a readable ValueError if any segment is missing — the message names the
    missing key path but NEVER echoes any value (so no secret can leak).
    """
    node: object = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"Secret config is missing key '{dotted_key}'. "
                f"Expected a nested entry reachable via that dotted path."
            )
        node = node[part]
    if not isinstance(node, str) or not node:
        raise ValueError(
            f"Secret config key '{dotted_key}' must map to a non-empty string token."
        )
    return node


class TushareFeed(DataFeed):
    """A DataFeed backed by the tushare pro API.

    The token lives in an external json file; this class reads it on first use
    and hands it straight to the tushare client. It never stores the token in a
    way that a repr/log could surface.
    """

    def __init__(
        self,
        secret_file: str,
        token_key: str = "tushare.token",
        rate_limit: int | None = None,
        max_retries: int = 6,
    ) -> None:
        self._secret_file = str(secret_file)
        self._token_key = token_key
        # SEC-004: per-minute call cap (calls/min) + retry on transient errors.
        self._rate_limit = rate_limit
        self._max_retries = max(1, int(max_retries))
        self._pro = None  # lazily built tushare pro client

    # -- secret handling ---------------------------------------------------- #
    def _read_token(self) -> str:
        """Read the token from the external json config (dotted-key lookup)."""
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
        """Build (once) and return the tushare pro client. Lazy + no logging."""
        if self._pro is None:
            import tushare as ts

            token = self._read_token()
            # Do NOT log/print token. Hand it directly to the SDK.
            self._pro = ts.pro_api(token)
        return self._pro

    # -- rate limit + retry (SEC-004) --------------------------------------- #
    def _call(self, fn, **kwargs):
        """Invoke a tushare endpoint with the shared retry + per-minute throttle."""
        return request_with_retry(
            fn,
            max_retries=self._max_retries,
            rate_limit=self._rate_limit,
            **kwargs,
        )

    # -- DataFeed API ------------------------------------------------------- #
    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Return a normalized market panel for ``symbols`` over [start, end]."""
        if freq not in ("D", "1d", "daily"):
            raise ValueError(
                f"TushareFeed currently supports only daily bars (freq='D'); "
                f"got freq={freq!r}. Minute support is reserved (DATA-011)."
            )
        if not symbols:
            raise ValueError("TushareFeed.get_bars requires a non-empty symbol list.")

        pro = self._client()
        start_compact = pd.Timestamp(start).strftime("%Y%m%d")
        end_compact = pd.Timestamp(end).strftime("%Y%m%d")

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            raw = self._call(
                pro.daily,
                ts_code=symbol,
                start_date=start_compact,
                end_date=end_compact,
            )
            if raw is None or len(raw) == 0:
                continue
            adj = self._fetch_adj_factor(pro, symbol, start_compact, end_compact)
            frames.append(self._to_canonical(raw, adj))

        if not frames:
            # Empty result is not an error: build an empty canonical panel.
            return self._empty_panel()

        combined = pd.concat(frames, ignore_index=True)
        return normalize_panel(combined)

    # -- mapping helpers ---------------------------------------------------- #
    def _fetch_adj_factor(self, pro, symbol: str, start_compact: str, end_compact: str):
        """Fetch the adj_factor series for ``symbol``; tolerate absence."""
        getter = getattr(pro, "adj_factor", None)
        if getter is None:
            return None
        adj = self._call(
            getter, ts_code=symbol, start_date=start_compact, end_date=end_compact
        )
        if adj is None or len(adj) == 0:
            return None
        return adj[["ts_code", "trade_date", "adj_factor"]].copy()

    def _to_canonical(self, raw: pd.DataFrame, adj: pd.DataFrame | None) -> pd.DataFrame:
        """Rename tushare columns to canonical names and attach adj_factor."""
        df = raw.rename(columns=_FIELD_MAP).copy()

        if adj is not None:
            adj = adj.rename(columns=_FIELD_MAP)
            df = df.merge(adj, on=["symbol", "date"], how="left")
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        df["adj_factor"] = df["adj_factor"].fillna(1.0)

        # tushare trade_date is a compact 'YYYYMMDD' string; parse to datetime.
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["symbol"] = df["symbol"].astype(str)

        keep = ["date", "symbol", *CORE_COLUMNS]
        present = [c for c in keep if c in df.columns]
        return df[present]

    @staticmethod
    def _empty_panel() -> pd.DataFrame:
        """An empty but schema-shaped panel (used when tushare returns nothing)."""
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=["date", "symbol"],
        )
        return pd.DataFrame(columns=CORE_COLUMNS, index=index)

    def __repr__(self) -> str:  # never leak the token
        return (
            f"TushareFeed(secret_file={self._secret_file!r}, "
            f"token_key={self._token_key!r}, rate_limit={self._rate_limit!r})"
        )
