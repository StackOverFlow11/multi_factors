"""Month-partitioned parquet store for raw 1min bars (I2: stk_mins 1min).

The daily :class:`data.cache.parquet_store.CacheParquetStore` keeps ONE file per
``(endpoint, symbol)``. That is too coarse for multi-year 1min data (one symbol
is hundreds of thousands of rows), so the intraday store partitions by
``year``/``month`` of ``bar_end``:

    <root>/stk_mins_1min/freq=1min/symbol_prefix=000/symbol=000001.SZ/year=2024/month=01.parquet

Natural key is ``(symbol, freq, bar_end)`` with ``freq`` always ``1min``. Writes
are ATOMIC (write ``*.tmp`` then ``os.replace``) and idempotent: an upsert drops
duplicates by the natural key keeping the latest fetched row, so re-fetching an
overlapping window never doubles a ``bar_end``. A best-effort per-file lock
guards the read-modify-write.

The store holds RAW bars only (``bar_end``/OHLCV/volume/amount/``source_trade_time``/
``freq``) and never sees a token — nothing secret is ever written here. The
derived PIT fields (``bar_start``/``available_time``) are NOT stored: they are
recomputed by ``normalize_intraday_bars`` after read.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Raw-canonical columns persisted per 1min bar (natural key = symbol, freq, bar_end).
STORED_COLUMNS: list[str] = [
    "symbol",
    "bar_end",
    "source_trade_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "freq",
]
KEY_COLS: list[str] = ["symbol", "freq", "bar_end"]


def _symbol_prefix(symbol: str) -> str:
    s = str(symbol)
    return s[:3] if len(s) >= 3 else s


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples spanning [start, end]."""
    s = pd.Timestamp(start).normalize().replace(day=1)
    e = pd.Timestamp(end).normalize().replace(day=1)
    out: list[tuple[int, int]] = []
    cur = s
    while cur <= e:
        out.append((cur.year, cur.month))
        cur = (cur + pd.Timedelta(days=32)).replace(day=1)
    return out


class IntradayParquetStore:
    """Persist raw 1min bars as per-(symbol, freq, year, month) parquet files."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    # -- paths -------------------------------------------------------------- #
    def month_path(
        self, endpoint: str, symbol: str, freq: str, year: int, month: int
    ) -> Path:
        return (
            self._root
            / endpoint
            / f"freq={freq}"
            / f"symbol_prefix={_symbol_prefix(symbol)}"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month:02d}.parquet"
        )

    def _lock_path(self, endpoint: str, symbol: str, freq: str, year: int, month: int) -> Path:
        return (
            self._root / ".locks"
            / f"{endpoint}__{symbol}__{freq}__{year}-{month:02d}.lock"
        )

    # -- locking (best-effort, per file) ------------------------------------ #
    @contextmanager
    def _locked(self, endpoint, symbol, freq, year, month, timeout: float = 10.0):
        lock = self._lock_path(endpoint, symbol, freq, year, month)
        lock.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        fd = None
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass

    # -- read --------------------------------------------------------------- #
    def read_range(
        self, endpoint: str, symbol: str, freq: str, start, end
    ) -> pd.DataFrame:
        """Return stored raw bars for ``symbol`` with ``bar_end`` in [start, end].

        Reads only the month partitions the window spans; an absent month is
        simply skipped. The result is sorted by ``bar_end`` (empty if nothing
        cached). Both bounds are inclusive.
        """
        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for year, month in _months_between(req_start, req_end):
            path = self.month_path(endpoint, symbol, freq, year, month)
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=STORED_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        mask = (df["bar_end"] >= req_start) & (df["bar_end"] <= req_end)
        return df.loc[mask, STORED_COLUMNS].sort_values("bar_end").reset_index(drop=True)

    # -- upsert ------------------------------------------------------------- #
    def upsert(
        self, endpoint: str, symbol: str, freq: str, rows: pd.DataFrame, key_cols: list[str]
    ) -> int:
        """Merge ``rows`` into their month partitions, dedup by ``key_cols``.

        Returns the number of rows written across the touched months. New rows
        win on a key collision (a re-fetched overlap replaces the stale row).
        Atomic per-month write; never mutates the caller's frame.
        """
        if rows is None or rows.empty:
            return 0
        rows = rows.copy()
        rows["bar_end"] = pd.to_datetime(rows["bar_end"])
        bar_end = rows["bar_end"]
        written = 0
        for (year, month), part in rows.groupby([bar_end.dt.year, bar_end.dt.month]):
            written += self._upsert_month(
                endpoint, symbol, freq, int(year), int(month), part, key_cols
            )
        return written

    def _upsert_month(
        self, endpoint, symbol, freq, year, month, rows, key_cols
    ) -> int:
        path = self.month_path(endpoint, symbol, freq, year, month)
        with self._locked(endpoint, symbol, freq, year, month):
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, rows], ignore_index=True)
            else:
                combined = rows.copy()
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
            combined = combined.sort_values("bar_end").reset_index(drop=True)
            combined = combined.reindex(columns=STORED_COLUMNS)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, path)
            return len(combined)
