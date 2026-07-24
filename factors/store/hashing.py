"""Content-hashing primitives for the factor store (refactor D3, §3.4).

Pure, deterministic, stdlib-only helpers. Two design properties matter:

* **Checkout independence.** ``content_hash_of_labeled_files`` hashes the file
  CONTENT and a caller-supplied stable LABEL (the module dotted name), NEVER the
  filesystem path. The same code produces the same hash in the worktree, in the
  main checkout, or on another machine — a path-dependent hash would silently
  invalidate every stored value the moment the repo moved.
* **Determinism.** Every hash is version-tagged and order-independent (labeled
  files are sorted by their label), so re-hashing the same inputs is byte-stable.

Layering: this module is a leaf — stdlib only. It must never import qt, feeds,
caches, analytics, or even the rest of ``factors`` (design red line #10).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# Bump only on a deliberate, disclosed change to the hashing scheme — it would
# invalidate every stored value (that is the point of a version tag).
_HASH_VERSION = "factors.store hashing v1"

#: Default short-hash length (hex chars). 16 hex = 64 bits: collision-negligible
#: for the store's key space, and keeps filenames readable (design §3.4 keeps the
#: fingerprint OUT of the filename; params/code hashes stay short in it).
SHORT_HASH_CHARS = 16


def sha256_hex(data: bytes) -> str:
    """sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def short_hash(full_hex: str, n: int = SHORT_HASH_CHARS) -> str:
    """First ``n`` hex chars of a full digest (for compact key filenames)."""
    if not isinstance(full_hex, str) or len(full_hex) < n:
        raise ValueError(f"short_hash needs a hex digest of >= {n} chars; got {full_hex!r}.")
    return full_hex[:n]


def stable_json(obj: object) -> str:
    """Deterministic, compact JSON of ``obj`` (sorted keys, no whitespace).

    ``default=str`` coerces non-JSON leaves (e.g. numpy scalars) to text so a
    params dict never crashes serialization; the point is a STABLE identity
    string, not a round-trippable document.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(params: Mapping[str, object] | None) -> str:
    """Full sha256 hex of a factor's params dict (stable across key order).

    ``None`` and ``{}`` hash identically (an empty parameter set), so a factor
    with no params gets a stable, well-defined params hash rather than a special
    case.
    """
    payload = stable_json(dict(params or {}))
    return sha256_hex((_HASH_VERSION + "\nparams\n" + payload).encode("utf-8"))


def content_hash_of_labeled_files(items: Sequence[tuple[str, Path | str]]) -> str:
    """Hash the CONTENT of labeled files, order- and path-independent.

    ``items`` is a sequence of ``(label, path)`` where ``label`` is a STABLE
    identity for the file (its module dotted name) — the label, not the path, is
    what enters the hash, so the digest is identical regardless of where the repo
    lives on disk. Files are folded in label-sorted order with an explicit length
    delimiter, so no concatenation ambiguity can collide two different inputs.

    Duplicate labels are a readable error (they would make the fold ambiguous).
    """
    seen: set[str] = set()
    ordered = sorted(items, key=lambda t: t[0])
    digest = hashlib.sha256()
    digest.update((_HASH_VERSION + "\nfiles\n").encode("utf-8"))
    for label, path in ordered:
        if label in seen:
            raise ValueError(f"content_hash_of_labeled_files got a duplicate label {label!r}.")
        seen.add(label)
        content = Path(path).read_bytes()
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(str(len(content)).encode("ascii") + b"\0")
        digest.update(content)
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "SHORT_HASH_CHARS",
    "content_hash_of_labeled_files",
    "params_hash",
    "sha256_hex",
    "short_hash",
    "stable_json",
]
