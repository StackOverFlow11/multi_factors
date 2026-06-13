"""Per-symbol parquet store for raw endpoint rows (P4-1).

One parquet file per ``(endpoint, symbol)`` under a ``symbol_prefix`` shard:

    <root>/<endpoint>/symbol_prefix=000/000001.SZ.parquet

Why per-symbol files: the feed fetches per symbol, the natural key is
``(symbol, date)``, and a per-symbol file makes upsert-by-key a trivial local
read-merge-write with no cross-symbol contention. ``symbol_prefix`` (the first
three characters) shards the directory so it never holds thousands of files.

Writes are ATOMIC (write ``*.tmp`` then ``os.replace``) and idempotent: an
upsert drops duplicates by the endpoint's natural key, keeping the latest
fetched row, so re-fetching a recent tail never doubles a ``(symbol, date)``.
A simple per-(endpoint, symbol) lock file guards the read-modify-write.

The store holds RAW rows only and never sees a token — nothing secret is ever
written here.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def _symbol_prefix(symbol: str) -> str:
    """Directory shard for a symbol (its first 3 chars, or the whole symbol)."""
    s = str(symbol)
    return s[:3] if len(s) >= 3 else s


class CacheParquetStore:
    """Persist raw endpoint rows as per-(endpoint, symbol) parquet files."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    # -- paths -------------------------------------------------------------- #
    def symbol_path(self, endpoint: str, symbol: str) -> Path:
        return (
            self._root
            / endpoint
            / f"symbol_prefix={_symbol_prefix(symbol)}"
            / f"{symbol}.parquet"
        )

    def _lock_path(self, endpoint: str, symbol: str) -> Path:
        return self._root / ".locks" / f"{endpoint}__{symbol}.lock"

    # -- locking (best-effort, per endpoint+symbol) ------------------------- #
    @contextmanager
    def _locked(self, endpoint: str, symbol: str, timeout: float = 10.0):
        lock = self._lock_path(endpoint, symbol)
        lock.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        fd = None
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    # Best-effort: a stale lock must never wedge a single-process
                    # run; proceed (atomic replace still keeps the file coherent).
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

    # -- read / upsert ------------------------------------------------------ #
    def read_symbol(self, endpoint: str, symbol: str) -> pd.DataFrame:
        """Return all cached raw rows for ``(endpoint, symbol)`` (empty if none)."""
        path = self.symbol_path(endpoint, symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def upsert_symbol(
        self,
        endpoint: str,
        symbol: str,
        rows: pd.DataFrame,
        key_cols: list[str],
    ) -> int:
        """Merge ``rows`` into the symbol's parquet, dedup by ``key_cols``.

        Returns the resulting row count. New rows win on a key collision (a
        re-fetched recent tail replaces the stale row). Atomic write; never
        mutates the caller's frame.
        """
        if rows is None or rows.empty:
            return self._row_count(endpoint, symbol)
        path = self.symbol_path(endpoint, symbol)
        with self._locked(endpoint, symbol):
            existing = self.read_symbol(endpoint, symbol)
            if existing.empty:
                combined = rows.copy()
            else:
                # new rows last so keep="last" lets a refetch overwrite.
                combined = pd.concat([existing, rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
            combined = combined.sort_values(key_cols).reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, path)
            return len(combined)

    def _row_count(self, endpoint: str, symbol: str) -> int:
        path = self.symbol_path(endpoint, symbol)
        if not path.exists():
            return 0
        return len(pd.read_parquet(path, columns=[]))
