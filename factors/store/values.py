"""Sparse raw factor-value storage (refactor D3, commit 3, §3.4 / R21).

One parquet file per store key, holding a ``MultiIndex(date, symbol)`` Series of
RAW factor values (processed panels never land here, red line #7):

    <root>/values/{factor_id}/{params_hash}__{code_hash}__{view}.parquet

Design choices (all locked by tests):

* **Single file, date-sorted, NOT partitioned** — a daily single-factor 5-year
  all-A panel is ~50MB; partitioning is a billion-row game (§六.6).
* **Atomic writes** (tmp + ``os.replace``) and a **per-artifact write lock**
  (``O_CREAT | O_EXCL``, mirroring ``CacheParquetStore._locked``) so a 21:00
  incremental and a concurrent research run never lose each other's rows (R21).
* **No coverage ledger** — factor values are deterministically recomputable, so a
  miss just triggers recompute; there is no ok/empty/failed state machine (§六.7).
* **A corrupt file is a MISS, not a crash** — read returns ``None`` and the caller
  recomputes (R21).
* **The DATA fingerprint is stored in the parquet FILE METADATA** (not the
  filename, design §3.4), so read-validation is self-describing and needs no
  registry lookup: a schema-version mismatch voids the WHOLE artifact (miss); a
  per-symbol adj_factor-hash mismatch (price_level only) voids THAT symbol column.

Layering: ``factors.store`` never imports ``qt`` (red line #10); this module uses
stdlib + numpy/pandas/pyarrow + the sibling ``keys``/``fingerprint`` leaves.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.store.keys import StoreKey

#: parquet file-metadata key holding the JSON data fingerprint.
_FINGERPRINT_META_KEY = b"factor_store_fingerprint"


class FactorValueStore:
    """Persist RAW factor-value Series as one parquet per :class:`StoreKey`."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    # -- paths ------------------------------------------------------------- #
    @property
    def values_root(self) -> Path:
        return self._root / "values"

    def path(self, key: StoreKey) -> Path:
        return self.values_root / key.relpath()

    def _lock_path(self, key: StoreKey) -> Path:
        safe = f"{key.factor_id}__{key.filename()}".replace("/", "_")
        return self._root / ".locks" / f"{safe}.lock"

    # -- locking (best-effort, per artifact) ------------------------------- #
    @contextmanager
    def _locked(self, key: StoreKey, timeout: float = 10.0):
        lock = self._lock_path(key)
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
                time.sleep(0.02)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass

    # -- (de)serialization ------------------------------------------------- #
    @staticmethod
    def _series_to_table(series: pd.Series, key: StoreKey, fingerprint: dict) -> pa.Table:
        frame = series.rename(key.factor_id).sort_index(kind="mergesort")
        frame = frame.reset_index()
        table = pa.Table.from_pandas(frame, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[_FINGERPRINT_META_KEY] = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
        return table.replace_schema_metadata(md)

    @staticmethod
    def _table_to_series(table: pa.Table, factor_id: str) -> pd.Series:
        frame = table.to_pandas()
        series = frame.set_index([DATE_LEVEL, SYMBOL_LEVEL])[factor_id]
        return series.rename(factor_id).sort_index(kind="mergesort")

    # -- write / upsert ---------------------------------------------------- #
    def write(self, key: StoreKey, series: pd.Series, *, fingerprint: dict) -> int:
        """Atomically (over)write the artifact for ``key``. Returns the row count."""
        with self._locked(key):
            return self._atomic_write(key, series, fingerprint)

    def upsert(self, key: StoreKey, series: pd.Series, *, fingerprint: dict) -> int:
        """Merge ``series`` into the artifact, dedup by (date, symbol), keep last.

        New rows win a key collision (a re-computed tail replaces the stored row).
        Under the per-artifact lock so a concurrent gap-fill never loses rows.
        """
        with self._locked(key):
            existing = self._read_raw(key)
            if existing is None or existing.empty:
                combined = series
            else:
                frames = pd.concat([existing, series])
                combined = frames[~frames.index.duplicated(keep="last")]
            return self._atomic_write(key, combined, fingerprint)

    def _atomic_write(self, key: StoreKey, series: pd.Series, fingerprint: dict) -> int:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self._series_to_table(series, key, fingerprint)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return table.num_rows

    # -- read -------------------------------------------------------------- #
    def _read_raw(self, key: StoreKey) -> pd.Series | None:
        """Read the stored Series, or None if the file is absent/corrupt (a MISS)."""
        path = self.path(key)
        if not path.exists():
            return None
        try:
            table = pq.read_table(path)
            return self._table_to_series(table, key.factor_id)
        except Exception:  # noqa: BLE001 - a corrupt file is a MISS, never a crash
            return None

    def read(self, key: StoreKey) -> pd.Series | None:
        """Read the stored Series with NO fingerprint validation (or None on miss)."""
        return self._read_raw(key)

    def stored_fingerprint(self, key: StoreKey) -> dict | None:
        """The data fingerprint stored in the artifact metadata (None on miss)."""
        path = self.path(key)
        if not path.exists():
            return None
        try:
            meta = pq.read_schema(path).metadata or {}
            raw = meta.get(_FINGERPRINT_META_KEY)
            return json.loads(raw.decode("utf-8")) if raw is not None else None
        except Exception:  # noqa: BLE001 - unreadable metadata => treat as a miss
            return None

    def read_valid(self, key: StoreKey, *, expected_fingerprint: dict) -> pd.Series | None:
        """Read only the values still VALID under ``expected_fingerprint``.

        * file absent / corrupt / no stored fingerprint  -> ``None`` (miss);
        * stored schema_version != expected               -> ``None`` (whole void,
          the schema/PIT machinery changed, §3.4);
        * price_level: any symbol whose stored adj_factor hash != expected is
          DROPPED (that column is void — an ex-date re-based the level); the rest
          is returned (possibly empty, a partial hit -> the caller recomputes the
          dropped symbols).
        """
        series = self._read_raw(key)
        if series is None:
            return None
        stored = self.stored_fingerprint(key)
        if stored is None:
            return None
        if stored.get("schema_version") != expected_fingerprint.get("schema_version"):
            return None  # whole artifact void
        stored_events = stored.get("adj_events")
        expected_events = expected_fingerprint.get("adj_events")
        if not expected_events and not stored_events:
            return series  # none / returns_invariant: no per-symbol validation
        # price_level: keep only symbols whose stored anchor hash still matches.
        stored_events = stored_events or {}
        expected_events = expected_events or {}
        valid_symbols = {
            sym
            for sym, digest in expected_events.items()
            if stored_events.get(sym) == digest
        }
        symbols = series.index.get_level_values(SYMBOL_LEVEL)
        return series[symbols.isin(valid_symbols)]


__all__ = ["FactorValueStore"]
