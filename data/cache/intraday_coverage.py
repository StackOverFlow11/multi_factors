"""Intraday-aware coverage ledger for the minute cache (I2: stk_mins 1min).

The daily :class:`data.cache.coverage.CoverageLedger` tracks coverage as
``start_date``/``end_date`` (day resolution). Minute coverage is a TIMESTAMP
interval over a single raw frequency, so reusing the daily date-range columns
would silently conflate a date range with a timestamp range. This is a SEPARATE
ledger (its own parquet file) with explicit ``start_time``/``end_time`` and a
``raw_freq`` column.

Schema (one append-only row per fetch attempt):

    endpoint        stk_mins_1min
    key_type        symbol
    key             e.g. 000001.SZ
    raw_freq        always "1min" (raw intraday SoT; coarser bars are derived)
    start_time      closed interval start (Timestamp, minute precision possible)
    end_time        closed interval end (Timestamp)
    fields_hash     stable hash of the stored field set
    fetched_at      local timestamp of the fetch
    row_count       rows the endpoint returned
    status          ok | empty | failed
    schema_version  cache schema version

Only ``ok`` / ``empty`` rows count as COVERAGE (a ``failed`` fetch leaves the
range uncovered so a later run retries). The planner consumes coverage at
trading-day granularity (a minute fetch unit is a whole trading day), but the
ledger preserves the true timestamp span. The ledger NEVER stores a token or any
secret-file content — it holds endpoint metadata only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from data.cache.intervals import Interval

INTRADAY_LEDGER_COLUMNS: list[str] = [
    "endpoint",
    "key_type",
    "key",
    "raw_freq",
    "start_time",
    "end_time",
    "fields_hash",
    "fetched_at",
    "row_count",
    "status",
    "schema_version",
]

_COVERING_STATUSES = ("ok", "empty")


class IntradayCoverageLedger:
    """Append-only parquet ledger of fetched (endpoint, symbol, time-range) coverage.

    D4: mirrors :class:`data.cache.coverage.CoverageLedger` — a per-instance frame
    cache + lookup memo avoid re-reading and re-filtering the full parquet on every
    repeated ``covered_day_intervals`` lookup, invalidated on the file's mtime
    change and refreshed on writes through this instance. Public columns, parquet
    path, and coverage semantics are unchanged.
    """

    def __init__(self, root: str, schema_version: str = "v1") -> None:
        self._root = Path(root)
        self._schema_version = schema_version
        self._frame: pd.DataFrame | None = None
        self._frame_mtime: int | None = None
        self._day_memo: dict[tuple[str, str, str], list[Interval]] = {}

    @property
    def path(self) -> Path:
        return self._root / "manifest" / "coverage_intraday.parquet"

    # -- internal load path (mtime-invalidated cache) ----------------------- #
    def _read_frame(self) -> pd.DataFrame:
        """Read the ledger parquet from disk (the cache-miss path; spy target)."""
        if not self.path.exists():
            return pd.DataFrame(columns=INTRADAY_LEDGER_COLUMNS)
        return pd.read_parquet(self.path)

    def _current_mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _load(self) -> pd.DataFrame:
        """Return the cached full ledger, reloading only when the file changed."""
        mtime = self._current_mtime()
        if self._frame is None or mtime != self._frame_mtime:
            self._frame = self._read_frame()
            self._frame_mtime = mtime
            self._day_memo.clear()
        return self._frame

    # -- read --------------------------------------------------------------- #
    def read(self) -> pd.DataFrame:
        """Return the full ledger (a COPY, so callers cannot corrupt the cache)."""
        return self._load().copy()

    def covered_day_intervals(
        self, endpoint: str, key: str, raw_freq: str
    ) -> list[Interval]:
        """Covered closed DAY intervals for ``(endpoint, key, raw_freq)``.

        The stored span is a true timestamp interval; the planner consumes it at
        trading-day granularity (a minute fetch unit is a whole trading day), so
        each covered span is reported day-normalized for the day-interval algebra
        in :mod:`data.cache.intervals`. Only ``ok``/``empty`` rows count.
        """
        ledger = self._load()  # first, so an external change invalidates the memo
        memo_key = (endpoint, str(key), str(raw_freq))
        if memo_key in self._day_memo:
            return list(self._day_memo[memo_key])
        if ledger.empty:
            self._day_memo[memo_key] = []
            return []
        mask = (
            (ledger["endpoint"] == endpoint)
            & (ledger["key"] == str(key))
            & (ledger["raw_freq"] == str(raw_freq))
            & (ledger["status"].isin(_COVERING_STATUSES))
            & ledger["start_time"].notna()
            & ledger["end_time"].notna()
        )
        sub = ledger[mask]
        result = [
            (pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize())
            for s, e in zip(sub["start_time"], sub["end_time"])
        ]
        self._day_memo[memo_key] = result
        return list(result)

    # -- write -------------------------------------------------------------- #
    def _normalize_row(self, row: dict) -> dict:
        """Normalize one input row to the ledger dtypes/order (no secret fields)."""
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        return {
            "endpoint": row["endpoint"],
            "key_type": row["key_type"],
            "key": str(row["key"]),
            "raw_freq": str(row["raw_freq"]),
            "start_time": pd.Timestamp(start_time) if start_time is not None else pd.NaT,
            "end_time": pd.Timestamp(end_time) if end_time is not None else pd.NaT,
            "fields_hash": str(row["fields_hash"]),
            "fetched_at": pd.Timestamp(row["fetched_at"]),
            "row_count": int(row["row_count"]),
            "status": str(row["status"]),
            "schema_version": self._schema_version,
        }

    def record(
        self,
        *,
        endpoint: str,
        key_type: str,
        key: str,
        raw_freq: str,
        start_time,
        end_time,
        fields_hash: str,
        row_count: int,
        status: str,
        fetched_at: pd.Timestamp,
    ) -> None:
        """Append one coverage row. Delegates to :meth:`record_many` so the
        single-row path stays identical to the batch path."""
        self.record_many([
            {
                "endpoint": endpoint,
                "key_type": key_type,
                "key": key,
                "raw_freq": raw_freq,
                "start_time": start_time,
                "end_time": end_time,
                "fields_hash": fields_hash,
                "row_count": row_count,
                "status": status,
                "fetched_at": fetched_at,
            }
        ])

    def record_many(self, rows: list[dict]) -> None:
        """Append a batch of coverage rows with one normalize + one parquet write.

        Empty input is a no-op. Each row is normalized exactly like a single
        ``record``; the written parquet is reindexed to ``INTRADAY_LEDGER_COLUMNS``.
        No token / secret is ever written here. The in-process cache is refreshed to
        the just-written frame.
        """
        rows = list(rows)
        if not rows:
            return
        normalized = [self._normalize_row(r) for r in rows]
        existing = self._load()
        new_rows = pd.DataFrame(normalized)
        combined = new_rows if existing.empty else pd.concat(
            [existing, new_rows], ignore_index=True
        )
        combined = combined.reindex(columns=INTRADAY_LEDGER_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, self.path)
        self._frame = combined
        self._frame_mtime = self._current_mtime()
        self._day_memo.clear()
