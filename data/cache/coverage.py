"""Coverage ledger for the persistent cache (P4-1).

Row presence alone cannot tell "missing because not fetched" from "missing
because the source has no row" (a stock not listed / suspended / out of index on
a date legitimately has no row). The ledger records, per fetch, which
``(endpoint, key, [start_date, end_date])`` range was retrieved and with what
outcome, so the planner can compute genuine gaps.

Schema (one append-only row per fetch attempt):

    endpoint        market_daily | adj_factor | ...
    key_type        symbol | index_code | global
    key             e.g. 000001.SZ
    start_date      closed interval start (NaT for snapshot endpoints)
    end_date        closed interval end (NaT for snapshot endpoints)
    fields_hash     stable hash of the requested field set
    fetched_at      local timestamp of the fetch
    row_count       rows the endpoint returned
    status          ok | empty | failed
    schema_version  cache schema version
    source_version  optional tushare/source version string

Only ``ok`` / ``empty`` rows count as COVERAGE (a failed fetch leaves the range
uncovered so a later run retries). The ledger NEVER stores a token or any
secret-file content — it holds endpoint metadata only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from data.cache.intervals import Interval

LEDGER_COLUMNS: list[str] = [
    "endpoint",
    "key_type",
    "key",
    "start_date",
    "end_date",
    "fields_hash",
    "fetched_at",
    "row_count",
    "status",
    "schema_version",
    "source_version",
]

_COVERING_STATUSES = ("ok", "empty")


class CoverageLedger:
    """Append-only parquet ledger of fetched (endpoint, key, range) coverage.

    D4: a per-instance frame cache + lookup memos avoid re-reading and re-filtering
    the full parquet on every repeated lookup. The cache is invalidated on the
    file's mtime change (so an out-of-instance write is not served stale) and is
    refreshed in place on writes through this instance. The public columns, parquet
    path, and coverage semantics are unchanged.
    """

    def __init__(self, root: str, schema_version: str = "v1") -> None:
        self._root = Path(root)
        self._schema_version = schema_version
        # in-process cache of the full ledger frame + the mtime it was read at
        # (None mtime == file absent). Lookup results are memoized per (endpoint,
        # key) and cleared whenever the frame cache is invalidated/refreshed.
        self._frame: pd.DataFrame | None = None
        self._frame_mtime: int | None = None
        self._intervals_memo: dict[tuple[str, str], list[Interval]] = {}
        self._snapshot_memo: dict[tuple[str, str], pd.Timestamp | None] = {}

    @property
    def path(self) -> Path:
        return self._root / "manifest" / "coverage.parquet"

    # -- internal load path (mtime-invalidated cache) ----------------------- #
    def _read_frame(self) -> pd.DataFrame:
        """Read the ledger parquet from disk (the cache-miss path; spy target)."""
        if not self.path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
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
            self._intervals_memo.clear()
            self._snapshot_memo.clear()
        return self._frame

    # -- read --------------------------------------------------------------- #
    def read(self) -> pd.DataFrame:
        """Return the full ledger (a COPY, so callers cannot corrupt the cache)."""
        return self._load().copy()

    def covered_intervals(self, endpoint: str, key: str) -> list[Interval]:
        """Closed date intervals already covered for ``(endpoint, key)``.

        Only ``ok`` / ``empty`` rows count (a ``failed`` fetch is not coverage).
        Snapshot rows (NaT range) are ignored here — this is the dense-endpoint
        planner's view (P4-1 only uses dense endpoints).
        """
        ledger = self._load()  # first, so an external change invalidates the memo
        memo_key = (endpoint, key)
        if memo_key in self._intervals_memo:
            return list(self._intervals_memo[memo_key])
        if ledger.empty:
            self._intervals_memo[memo_key] = []
            return []
        mask = (
            (ledger["endpoint"] == endpoint)
            & (ledger["key"] == key)
            & (ledger["status"].isin(_COVERING_STATUSES))
            & ledger["start_date"].notna()
            & ledger["end_date"].notna()
        )
        sub = ledger[mask]
        result = [
            (pd.Timestamp(s), pd.Timestamp(e))
            for s, e in zip(sub["start_date"], sub["end_date"])
        ]
        self._intervals_memo[memo_key] = result
        return list(result)

    def snapshot_fetched_at(self, endpoint: str, key: str) -> pd.Timestamp | None:
        """Latest successful fetch time for a snapshot/dimension ``(endpoint, key)``.

        For the dimension endpoints (``stock_basic``, ``namechange``) coverage is
        not a date range but a freshness timestamp: the most recent ``ok``/
        ``empty`` ``fetched_at``. ``None`` means never fetched (must fetch). The
        caller compares against ``refresh_dimension_days`` to decide staleness.
        """
        ledger = self._load()  # first, so an external change invalidates the memo
        memo_key = (endpoint, str(key))
        if memo_key in self._snapshot_memo:
            return self._snapshot_memo[memo_key]
        result: pd.Timestamp | None = None
        if not ledger.empty:
            mask = (
                (ledger["endpoint"] == endpoint)
                & (ledger["key"] == str(key))
                & (ledger["status"].isin(_COVERING_STATUSES))
            )
            sub = ledger[mask]
            if not sub.empty:
                result = pd.Timestamp(sub["fetched_at"].max())
        self._snapshot_memo[memo_key] = result
        return result

    def last_fields_hash(self, endpoint: str) -> str | None:
        """``fields_hash`` of the most-recent (by ``fetched_at``) row for ``endpoint``.

        READ-ONLY, additive (D-series schema guard): the stored canonical-column
        hash from the last fetch of any key of ``endpoint``, or ``None`` when the
        endpoint was never recorded. Reuses the in-process frame cache; does NOT
        change ``LEDGER_COLUMNS``, the parquet path, or coverage semantics.
        """
        ledger = self._load()
        if ledger.empty:
            return None
        sub = ledger[ledger["endpoint"] == endpoint]
        if sub.empty:
            return None
        value = sub.loc[sub["fetched_at"].idxmax(), "fields_hash"]
        if pd.isna(value):  # pd.isna(None) is True — covers None / NaN / NaT
            return None
        return str(value)

    # -- write -------------------------------------------------------------- #
    def _normalize_row(self, row: dict) -> dict:
        """Normalize one input row to the ledger dtypes/order (no secret fields)."""
        start_date = row.get("start_date")
        end_date = row.get("end_date")
        return {
            "endpoint": row["endpoint"],
            "key_type": row["key_type"],
            "key": str(row["key"]),
            "start_date": pd.Timestamp(start_date) if start_date is not None else pd.NaT,
            "end_date": pd.Timestamp(end_date) if end_date is not None else pd.NaT,
            "fields_hash": str(row["fields_hash"]),
            "fetched_at": pd.Timestamp(row["fetched_at"]),
            "row_count": int(row["row_count"]),
            "status": str(row["status"]),
            "schema_version": self._schema_version,
            "source_version": row.get("source_version"),
        }

    def record(
        self,
        *,
        endpoint: str,
        key_type: str,
        key: str,
        start_date,
        end_date,
        fields_hash: str,
        row_count: int,
        status: str,
        fetched_at: pd.Timestamp,
        source_version: str | None = None,
    ) -> None:
        """Append one coverage row. Delegates to :meth:`record_many` so the
        single-row path stays identical to the batch path."""
        self.record_many([
            {
                "endpoint": endpoint,
                "key_type": key_type,
                "key": key,
                "start_date": start_date,
                "end_date": end_date,
                "fields_hash": fields_hash,
                "row_count": row_count,
                "status": status,
                "fetched_at": fetched_at,
                "source_version": source_version,
            }
        ])

    def record_many(self, rows: list[dict]) -> None:
        """Append a batch of coverage rows with one normalize + one parquet write.

        Empty input is a no-op. Each row is normalized exactly like a single
        ``record``; the written parquet is reindexed to ``LEDGER_COLUMNS``. No
        token / secret is ever written here. The in-process cache is refreshed to
        the just-written frame.
        """
        rows = list(rows)
        if not rows:
            return
        normalized = [self._normalize_row(r) for r in rows]
        existing = self._load()
        new_rows = pd.DataFrame(normalized)
        # Avoid concat-with-empty (it warns and can shift dtypes): the first
        # records ARE the new rows; later records append to a non-empty ledger.
        combined = new_rows if existing.empty else pd.concat(
            [existing, new_rows], ignore_index=True
        )
        # enforce column order / presence
        combined = combined.reindex(columns=LEDGER_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, self.path)
        # refresh the in-process cache to the just-written frame (no re-read)
        self._frame = combined
        self._frame_mtime = self._current_mtime()
        self._intervals_memo.clear()
        self._snapshot_memo.clear()
