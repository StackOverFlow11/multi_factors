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
    """Append-only parquet ledger of fetched (endpoint, key, range) coverage."""

    def __init__(self, root: str, schema_version: str = "v1") -> None:
        self._root = Path(root)
        self._schema_version = schema_version

    @property
    def path(self) -> Path:
        return self._root / "manifest" / "coverage.parquet"

    # -- read --------------------------------------------------------------- #
    def read(self) -> pd.DataFrame:
        """Return the full ledger (empty, correctly-typed frame if absent)."""
        if not self.path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        return pd.read_parquet(self.path)

    def covered_intervals(self, endpoint: str, key: str) -> list[Interval]:
        """Closed date intervals already covered for ``(endpoint, key)``.

        Only ``ok`` / ``empty`` rows count (a ``failed`` fetch is not coverage).
        Snapshot rows (NaT range) are ignored here — this is the dense-endpoint
        planner's view (P4-1 only uses dense endpoints).
        """
        ledger = self.read()
        if ledger.empty:
            return []
        mask = (
            (ledger["endpoint"] == endpoint)
            & (ledger["key"] == key)
            & (ledger["status"].isin(_COVERING_STATUSES))
            & ledger["start_date"].notna()
            & ledger["end_date"].notna()
        )
        sub = ledger[mask]
        return [
            (pd.Timestamp(s), pd.Timestamp(e))
            for s, e in zip(sub["start_date"], sub["end_date"])
        ]

    def snapshot_fetched_at(self, endpoint: str, key: str) -> pd.Timestamp | None:
        """Latest successful fetch time for a snapshot/dimension ``(endpoint, key)``.

        For the dimension endpoints (``stock_basic``, ``namechange``) coverage is
        not a date range but a freshness timestamp: the most recent ``ok``/
        ``empty`` ``fetched_at``. ``None`` means never fetched (must fetch). The
        caller compares against ``refresh_dimension_days`` to decide staleness.
        """
        ledger = self.read()
        if ledger.empty:
            return None
        mask = (
            (ledger["endpoint"] == endpoint)
            & (ledger["key"] == str(key))
            & (ledger["status"].isin(_COVERING_STATUSES))
        )
        sub = ledger[mask]
        if sub.empty:
            return None
        return pd.Timestamp(sub["fetched_at"].max())

    # -- write -------------------------------------------------------------- #
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
        """Append one coverage row (atomic). No secret is ever written here."""
        row = {
            "endpoint": endpoint,
            "key_type": key_type,
            "key": str(key),
            "start_date": pd.Timestamp(start_date) if start_date is not None else pd.NaT,
            "end_date": pd.Timestamp(end_date) if end_date is not None else pd.NaT,
            "fields_hash": str(fields_hash),
            "fetched_at": pd.Timestamp(fetched_at),
            "row_count": int(row_count),
            "status": str(status),
            "schema_version": self._schema_version,
            "source_version": source_version,
        }
        existing = self.read()
        new_row = pd.DataFrame([row])
        # Avoid concat-with-empty (it warns and can shift dtypes): the first
        # record IS the new row; later records append to a non-empty ledger.
        combined = new_row if existing.empty else pd.concat(
            [existing, new_row], ignore_index=True
        )
        # enforce column order / presence
        combined = combined.reindex(columns=LEDGER_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, self.path)
