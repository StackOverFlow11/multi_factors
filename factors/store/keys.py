"""The factor-store key: (factor_id, params, code, view) -> artifact path (D3, §3.4).

A stored value is addressed by four identity dimensions (design red line #6 —
"cache key 必须完整"):

    factor_id  — the factor panel column (also spec.factor_id)
    params     — the config params (hashed; window-named ids already fold the
                 window in, but ALL params enter the hash)
    code       — code_hash: the factor module + the enumerated shared set
    view       — decision | close (a first-class identity dimension, §1.4)

Layout (design §3.4):

    artifacts/factor_store/values/{factor_id}/{params_hash}__{code_hash}__{view}.parquet

The DATA fingerprint (schema-version + adj-factor events) is deliberately NOT in
the filename (design §3.4): it is validated on READ from the artifact metadata, so
a schema/anchor change re-validates the SAME file instead of spawning a new one.
The view is coerced through ``data.availability_policy.View`` so an unknown view is
a readable error, never a silently-new file.

Leaf: stdlib + the availability-policy enums + the sibling hashing/code_hash leaves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from data.availability_policy import View
from factors.base import Factor
from factors.store.code_hash import code_hash
from factors.store.hashing import params_hash, short_hash


@dataclass(frozen=True, slots=True)
class StoreKey:
    """The four-dimension identity of a stored factor value (immutable)."""

    factor_id: str
    params_hash: str  # short hex (SHORT_HASH_CHARS)
    code_hash: str  # short hex
    view: str  # a data.availability_policy.View value ("decision" | "close")

    def filename(self) -> str:
        """The parquet filename (the last three dimensions; factor_id is the dir)."""
        return f"{self.params_hash}__{self.code_hash}__{self.view}.parquet"

    def relpath(self) -> str:
        """Path RELATIVE to the ``values/`` root: ``{factor_id}/{filename}``."""
        return f"{self.factor_id}/{self.filename()}"


def _coerce_view(view: object) -> str:
    """Coerce to a legal View value, readably (never a silently-new file)."""
    try:
        return View(view).value  # type: ignore[arg-type]
    except ValueError:
        allowed = ", ".join(repr(v.value) for v in View)
        raise ValueError(
            f"unknown store view {view!r}; the two legal views are {allowed} "
            f"(data.availability_policy.View)."
        ) from None


def store_key(
    factor: Factor,
    *,
    view: object,
    params: Mapping[str, object] | None = None,
) -> StoreKey:
    """Build the :class:`StoreKey` for ``factor`` under ``view`` and ``params``.

    ``params`` are the config params (default ``{}``); ``code_hash`` is derived
    from the factor's module + shared set. Hashes are shortened for a readable
    filename (full digests stay in the run registry / metadata).
    """
    return StoreKey(
        factor_id=factor.name,
        params_hash=short_hash(params_hash(params)),
        code_hash=short_hash(code_hash(factor)),
        view=_coerce_view(view),
    )


__all__ = ["StoreKey", "store_key"]
