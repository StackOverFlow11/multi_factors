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
    """Append-only parquet ledger of fetched (endpoint, symbol, time-range) coverage."""

    def __init__(self, root: str, schema_version: str = "v1") -> None:
        self._root = Path(root)
        self._schema_version = schema_version

    @property
    def path(self) -> Path:
        return self._root / "manifest" / "coverage_intraday.parquet"

    # -- read --------------------------------------------------------------- #
    def read(self) -> pd.DataFrame:
        """Return the full ledger (empty, correctly-typed frame if absent)."""
        if not self.path.exists():
            return pd.DataFrame(columns=INTRADAY_LEDGER_COLUMNS)
        return pd.read_parquet(self.path)

    def covered_day_intervals(
        self, endpoint: str, key: str, raw_freq: str
    ) -> list[Interval]:
        """Covered closed DAY intervals for ``(endpoint, key, raw_freq)``.

        The stored span is a true timestamp interval; the planner consumes it at
        trading-day granularity (a minute fetch unit is a whole trading day), so
        each covered span is reported day-normalized for the day-interval algebra
        in :mod:`data.cache.intervals`. Only ``ok``/``empty`` rows count.
        """
        ledger = self.read()
        if ledger.empty:
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
        return [
            (pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize())
            for s, e in zip(sub["start_time"], sub["end_time"])
        ]

    # -- write -------------------------------------------------------------- #
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
        """Append one coverage row (atomic). No secret is ever written here."""
        row = {
            "endpoint": endpoint,
            "key_type": key_type,
            "key": str(key),
            "raw_freq": str(raw_freq),
            "start_time": pd.Timestamp(start_time) if start_time is not None else pd.NaT,
            "end_time": pd.Timestamp(end_time) if end_time is not None else pd.NaT,
            "fields_hash": str(fields_hash),
            "fetched_at": pd.Timestamp(fetched_at),
            "row_count": int(row_count),
            "status": str(status),
            "schema_version": self._schema_version,
        }
        existing = self.read()
        new_row = pd.DataFrame([row])
        combined = new_row if existing.empty else pd.concat(
            [existing, new_row], ignore_index=True
        )
        combined = combined.reindex(columns=INTRADAY_LEDGER_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, self.path)
