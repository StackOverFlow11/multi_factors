"""The RUN registry: append-only lineage of stored factor artifacts (D3, R22/R25).

Two registries must never be conflated (design §3.4 R25):

* the CODE registry (``factors.registry`` — name -> class, reviewed per PR); and
* the RUN registry HERE (artifact -> lineage/status, appended per run).

The run registry is a JSONL file (``<root>/registry.jsonl``). JSONL because
verdicts/status are APPEND-ONLY (#74: a record's meaning changes across contract
versions, so overwriting would drop the trajectory); each ``append`` writes ONE
line and never rewrites the file.

NO-SECRET CONTRACT (R22): a registry line stores ONLY factor_id / params hash /
code hash / view / status / endpoint NAMES / the adjustment + schema-version
fingerprint / a timestamp / an optional note. It NEVER stores a token or a
secret-file path — every free-text field is passed through
``data.quality.report.sanitize_text`` (the same redaction the cache/quality layers
use), so a stray secret in a note is redacted at the boundary, and the per-PR
secret scan stays a backstop, not the first line of defence.

``status`` is a plain STRING field (``exploratory`` | ``watch`` | ``book`` |
``retired``), not a four-state transition machine (R25): the "book" = the curated
``status == "book"`` list. Startup consistency is a single assertion (the code
registry's names must cover every run-registry factor id) — the disk is
regenerable (a miss self-heals, a stale fingerprint self-heals), so no three-way
reconcile CLI is built (§六.12).

Layering: ``factors.store`` never imports ``qt``; this uses stdlib + the
availability-policy leaf (indirectly) + ``data.quality.report.sanitize_text``
(data layer, a downward dependency).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.quality.report import sanitize_text
from factors.base import Factor
from factors.store.hashing import short_hash
from factors.store.keys import StoreKey

#: The legal status strings (R25: a string field, not a transition machine).
STATUSES: frozenset[str] = frozenset({"exploratory", "watch", "book", "retired"})

_LINE_FIELDS = (
    "factor_id",
    "params_hash",
    "code_hash",
    "view",
    "status",
    "endpoints",
    "adjustment",
    "schema_version",
    "created_utc",
    "note",
)


@dataclass(frozen=True)
class RunRecord:
    """One append-only lineage record of a stored factor artifact (no secrets)."""

    factor_id: str
    params_hash: str
    code_hash: str
    view: str
    status: str
    endpoints: tuple[str, ...]
    adjustment: str
    schema_version: str  # short digest (lineage only; the parquet metadata is authoritative)
    created_utc: str
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "params_hash": self.params_hash,
            "code_hash": self.code_hash,
            "view": self.view,
            "status": self.status,
            "endpoints": list(self.endpoints),
            "adjustment": self.adjustment,
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "note": self.note,
        }


def _now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


class RunRegistry:
    """Append-only JSONL run registry of stored factor artifacts."""

    def __init__(self, root: str | Path) -> None:
        self._path = Path(root) / "registry.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    # -- append (the only mutation; JSONL is append-only) ------------------ #
    def append(self, record: RunRecord) -> None:
        """Append ONE record as a JSON line (validated + secret-redacted)."""
        if record.status not in STATUSES:
            raise ValueError(
                f"unknown run-registry status {record.status!r}; allowed: "
                f"{sorted(STATUSES)}."
            )
        payload = record.to_dict()
        # Redact every string leaf that could carry a stray secret; benign values
        # (hashes, endpoint names, ISO timestamps) are untouched by sanitize_text.
        payload["note"] = sanitize_text(record.note) if record.note is not None else None
        payload["endpoints"] = [sanitize_text(e) for e in record.endpoints]
        line = json.dumps({k: payload[k] for k in _LINE_FIELDS}, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def append_run(
        self,
        *,
        key: StoreKey,
        factor: Factor,
        status: str,
        fingerprint: Mapping[str, object],
        note: str | None = None,
    ) -> RunRecord:
        """Build a :class:`RunRecord` from a factor + key + fingerprint, and append it."""
        endpoints = tuple(
            sorted({str(r.source) for r in (factor.spec.requires or ())})
        )
        schema = fingerprint.get("schema_version")
        record = RunRecord(
            factor_id=key.factor_id,
            params_hash=key.params_hash,
            code_hash=key.code_hash,
            view=key.view,
            status=status,
            endpoints=endpoints,
            adjustment=str(fingerprint.get("adjustment", factor.spec.adjustment)),
            schema_version=short_hash(str(schema)) if schema else "",
            created_utc=_now_utc(),
            note=note,
        )
        self.append(record)
        return record

    # -- read -------------------------------------------------------------- #
    def read_all(self) -> list[dict]:
        """All records as dicts, in append order (empty if the file is absent)."""
        if not self._path.exists():
            return []
        out: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def factor_ids(self) -> set[str]:
        """The distinct factor ids recorded so far."""
        return {rec["factor_id"] for rec in self.read_all()}


def assert_registry_consistent(registry: RunRegistry, code_registry=None) -> None:
    """Startup consistency (R25): every run-registry factor id must resolve.

    The CODE registry (name -> class) must know every factor id the run registry
    recorded; an unknown id is a readable error (a renamed/removed factor left a
    dangling artifact). Read-only — the disk is regenerable, so this only guards,
    it never rewrites.
    """
    if code_registry is None:
        from factors.registry.registry import DEFAULT_REGISTRY

        code_registry = DEFAULT_REGISTRY
    unknown: list[str] = []
    for factor_id in sorted(registry.factor_ids()):
        try:
            code_registry.resolve(factor_id)
        except ValueError:
            unknown.append(factor_id)
    if unknown:
        raise ValueError(
            f"run registry {registry.path} references factor id(s) the code "
            f"registry does not know: {unknown}. A renamed/removed factor left a "
            f"dangling artifact; add it back or prune the artifact (the store is "
            f"regenerable)."
        )


def factor_ids_in(records: Iterable[Mapping[str, object]]) -> set[str]:
    """Distinct factor ids across an iterable of records (test/helper convenience)."""
    return {str(rec["factor_id"]) for rec in records}


__all__ = [
    "STATUSES",
    "RunRecord",
    "RunRegistry",
    "assert_registry_consistent",
    "factor_ids_in",
]
