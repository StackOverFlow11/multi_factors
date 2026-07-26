"""Sparse raw factor-value storage (refactor D3, commit 3, §3.4 / R21).

One parquet file per store key, holding a ``MultiIndex(date, symbol)`` PAYLOAD of
RAW factor data (processed panels never land here, red line #7):

    <root>/values/{factor_id}/{params_hash}__{code_hash}__{view}.parquet

TWO PAYLOAD SHAPES (D4c). Almost every factor stores its VALUE: one column named
after the factor, and the Series API below is the whole story. A factor whose
value is a CROSS-SECTIONAL function of the universe cannot store its value —
the key says (factor, params, code, view) and none of those is "which universe
was loaded", so a value filled under one universe would be served to another
(measured: ``intraday_amp_cut`` filled with 12 names then read with 24 returned
24 rows instead of 48 and every shared cell differed, max|delta| 0.194). Such a
factor stores its UNIVERSE-INDEPENDENT per-symbol intermediate instead (several
columns) and the consumer runs the cross-sectional combine at read time. The
frame API carries that; the Series API is a thin wrapper over it.

The payload's COLUMNS are therefore part of what a read must validate: the
caller passes the columns it expects and a mismatch is a MISS (the artifact was
written under a different payload shape — e.g. by pre-D4c code — and must be
recomputed, never merged into or half-read). Validating the columns actually on
disk, rather than a recorded declaration, is what makes artifacts written before
this existed validate correctly.

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


def _payload_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(str(c) for c in frame.columns)


class FactorValueStore:
    """Persist a RAW factor payload as one parquet per :class:`StoreKey`."""

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
    def _frame_to_table(frame: pd.DataFrame, fingerprint: dict) -> pa.Table:
        flat = frame.sort_index(kind="mergesort").reset_index()
        table = pa.Table.from_pandas(flat, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md[_FINGERPRINT_META_KEY] = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
        return table.replace_schema_metadata(md)

    @staticmethod
    def _table_to_frame(table: pa.Table) -> pd.DataFrame:
        frame = table.to_pandas().set_index([DATE_LEVEL, SYMBOL_LEVEL])
        return frame.sort_index(kind="mergesort")

    @staticmethod
    def _as_payload(series: pd.Series, key: StoreKey) -> pd.DataFrame:
        """The single-value-column payload frame for a value Series."""
        return series.rename(key.factor_id).to_frame()

    # -- write / upsert ---------------------------------------------------- #
    def write(self, key: StoreKey, series: pd.Series, *, fingerprint: dict) -> int:
        """Atomically (over)write the VALUE artifact for ``key``. Returns rows."""
        return self.write_frame(key, self._as_payload(series, key), fingerprint=fingerprint)

    def write_frame(self, key: StoreKey, frame: pd.DataFrame, *, fingerprint: dict) -> int:
        """Atomically (over)write the payload artifact for ``key``. Returns rows."""
        with self._locked(key):
            return self._atomic_write(key, frame, fingerprint)

    def upsert(self, key: StoreKey, series: pd.Series, *, fingerprint: dict) -> int:
        """Merge a VALUE Series into the artifact, dedup by (date, symbol), keep last."""
        return self.upsert_frame(key, self._as_payload(series, key), fingerprint=fingerprint)

    def upsert_frame(self, key: StoreKey, frame: pd.DataFrame, *, fingerprint: dict) -> int:
        """Merge ``frame`` into the artifact, dedup by (date, symbol), keep last.

        New rows win a key collision (a re-computed tail replaces the stored row).
        Under the per-artifact lock so a concurrent gap-fill never loses rows.

        An artifact whose COLUMNS differ from ``frame``'s is not merged into: it
        was written under a different payload shape, so concatenating would build
        a frame that is half one shape and half the other, with NaN where each
        half lacks the other's columns. It is overwritten instead — the same
        "stale artifact => recompute" rule the fingerprint mismatch already uses.
        """
        with self._locked(key):
            existing = self._read_raw(key)
            if (
                existing is None
                or existing.empty
                or _payload_columns(existing) != _payload_columns(frame)
            ):
                combined = frame
            else:
                frames = pd.concat([existing, frame])
                combined = frames[~frames.index.duplicated(keep="last")]
            return self._atomic_write(key, combined, fingerprint)

    def _atomic_write(self, key: StoreKey, frame: pd.DataFrame, fingerprint: dict) -> int:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self._frame_to_table(frame, fingerprint)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return table.num_rows

    # -- read -------------------------------------------------------------- #
    def _read_raw(self, key: StoreKey) -> pd.DataFrame | None:
        """Read the stored payload, or None if the file is absent/corrupt (a MISS)."""
        path = self.path(key)
        if not path.exists():
            return None
        try:
            table = pq.read_table(path)
            return self._table_to_frame(table)
        except Exception:  # noqa: BLE001 - a corrupt file is a MISS, never a crash
            return None

    @staticmethod
    def _value_column(frame: pd.DataFrame | None, key: StoreKey) -> pd.Series | None:
        """The value Series of a VALUE payload; readable error for another shape.

        The Series API is only meaningful for a single-value-column payload. A
        multi-column (per-symbol intermediate) artifact reached through it is a
        wiring mistake by the caller, not a data problem, so it is loud rather
        than silently reduced to one of its columns.
        """
        if frame is None:
            return None
        columns = _payload_columns(frame)
        if columns != (key.factor_id,):
            raise ValueError(
                f"{key.factor_id}: the stored payload carries {list(columns)}, not the "
                f"single value column {key.factor_id!r}; read it with read_frame / "
                f"read_valid_frame (a cross-sectional factor stores its per-symbol "
                f"intermediate, whose combine is the consumer's job)."
            )
        return frame[key.factor_id].rename(key.factor_id)

    def read(self, key: StoreKey) -> pd.Series | None:
        """Read the stored VALUE Series with NO fingerprint validation (None on miss)."""
        return self._value_column(self._read_raw(key), key)

    def read_frame(self, key: StoreKey) -> pd.DataFrame | None:
        """Read the stored payload frame with NO validation (or None on miss)."""
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
        """Read only the VALUES still valid under ``expected_fingerprint``."""
        frame = self.read_valid_frame(
            key, expected_fingerprint=expected_fingerprint, columns=(key.factor_id,)
        )
        return self._value_column(frame, key)

    def read_valid_frame(
        self,
        key: StoreKey,
        *,
        expected_fingerprint: dict,
        columns: tuple[str, ...] | list[str],
    ) -> pd.DataFrame | None:
        """Read only the payload rows still VALID under ``expected_fingerprint``.

        * file absent / corrupt / no stored fingerprint  -> ``None`` (miss);
        * stored payload columns != ``columns``           -> ``None`` (miss: the
          artifact holds a DIFFERENT payload shape — pre-D4c value rows for a
          factor that now stores its intermediate, say — so it is recomputed, not
          half-read; D4c);
        * stored schema_version != expected               -> ``None`` (whole void,
          the schema/PIT machinery changed, §3.4);
        * price_level: any symbol whose stored adj_factor hash != expected is
          DROPPED (that column is void — an ex-date re-based the level); the rest
          is returned (possibly empty, a partial hit -> the caller recomputes the
          dropped symbols).
        """
        frame = self._read_raw(key)
        if frame is None:
            return None
        if _payload_columns(frame) != tuple(str(c) for c in columns):
            return None  # different payload shape: a miss, never a partial read
        stored = self.stored_fingerprint(key)
        if stored is None:
            return None
        if stored.get("schema_version") != expected_fingerprint.get("schema_version"):
            return None  # whole artifact void
        stored_events = stored.get("adj_events")
        expected_events = expected_fingerprint.get("adj_events")
        if not expected_events and not stored_events:
            return frame  # none / returns_invariant: no per-symbol validation
        # price_level: keep only symbols whose stored anchor hash still matches.
        stored_events = stored_events or {}
        expected_events = expected_events or {}
        valid_symbols = {
            sym
            for sym, digest in expected_events.items()
            if stored_events.get(sym) == digest
        }
        symbols = frame.index.get_level_values(SYMBOL_LEVEL)
        return frame[symbols.isin(valid_symbols)]


__all__ = ["FactorValueStore"]
