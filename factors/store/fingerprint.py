"""Data fingerprint of a stored factor value, DERIVED FROM the adjustment declaration.

Design §3.4 / D0 §1.1. A stored factor value can go stale for two reasons that the
code hash does NOT capture:

* the PIT/schema machinery in ``data.clean`` changes the ``available_time`` a bar
  is visible at — folded in as the SCHEMA-VERSION dimension (content hash of the
  two PIT-bearing modules; design §3.4 R3 "封 A8-F01 边界洞"); and
* for a ``price_level`` factor only, a corporate action re-bases the adjusted
  price level — folded in as a per-symbol ``adj_factor`` event-table hash.

The fingerprint is DERIVED FROM the factor's ``adjustment`` declaration, and the
derivation is what makes the declaration testable (D0 §1.1 / §六.17 qfq-anchor trap):

  * ``none`` / ``returns_invariant`` -> schema-version dimension only; supplying an
    adj-factor event table is a CONTRADICTION (readable error) — those factors do
    not depend on the price LEVEL, so mixing the anchor into their key would be
    wrong.
  * ``price_level`` -> schema-version dimension PLUS the per-symbol adj-factor event
    hash; OMITTING it is a readable error. The current closing 14 factors have ZERO
    price_level members, but the mechanism is NOT waived (kill file A5-F02): a
    factor mislabeled ``returns_invariant`` would silently reuse an ex-date-stale
    value, exactly the hole §六.17 closes.

Read-time use (the D3 value store):
  * schema-version mismatch  -> the WHOLE artifact is void (miss -> recompute);
  * a per-symbol adj-hash mismatch (price_level only) -> THAT symbol column is void.

Layering leaf: stdlib + numpy/pandas + the ``data.availability_policy`` enums + the
sibling ``hashing`` helper. Never qt / feeds / analytics.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from data.availability_policy import Adjustment
from factors.store.hashing import (
    content_hash_of_labeled_files,
    sha256_hex,
    stable_json,
)

#: Bump on a deliberate change to the fingerprint scheme (invalidates stored values).
FINGERPRINT_VERSION = "v1"

#: The PIT / schema-bearing ``data.clean`` modules whose content is the schema
#: dimension (design §3.4: intraday_schema + the intraday_aggregate general core).
#: Explicitly NOT the whole ``data/`` tree — that would over-invalidate on
#: data-layer churn.
_SCHEMA_MODULES: tuple[str, ...] = (
    "data.clean.intraday_schema",
    "data.clean.intraday_aggregate",
)


def _schema_module_files() -> list[tuple[str, str]]:
    import importlib
    import inspect

    out: list[tuple[str, str]] = []
    for name in _SCHEMA_MODULES:
        module = importlib.import_module(name)
        source = inspect.getsourcefile(module)
        if source is None:  # pragma: no cover - schema modules have source
            raise RuntimeError(f"cannot resolve a source file for {name!r}.")
        out.append((name, source))
    return out


def schema_version_hash() -> str:
    """Content hash of the PIT/schema-bearing ``data.clean`` modules.

    Checkout-independent (module labels + content, never paths). A change to the
    ``available_time`` formula or the general as-of aggregation core changes this,
    invalidating decision-view stored values that would otherwise be silently stale.
    """
    return content_hash_of_labeled_files(_schema_module_files())


def adj_events_hash(adj_factor: pd.Series) -> str:
    """Hash of ONE symbol's adj_factor event table, INCLUDING the anchor values.

    ``adj_factor`` is a Series indexed by date (the raw cumulative adjustment
    factors). Both the dates and the VALUES enter the hash (design §3.4 "含锚值"):
    a re-based level (an ex-date) changes the values even when the date grid is
    unchanged, so a value hash is required — a coverage/date-only hash would miss it.
    Deterministic: sorted by date, NaN payload collapsed.
    """
    if not isinstance(adj_factor, pd.Series):
        raise TypeError(f"adj_events_hash needs a pd.Series; got {type(adj_factor).__name__}.")
    series = adj_factor.sort_index(kind="mergesort")
    dates = pd.DatetimeIndex(series.index).asi8
    values = np.array(series.to_numpy(), dtype="<f8", copy=True)
    values[np.isnan(values)] = np.float64("nan")
    payload = (
        FINGERPRINT_VERSION.encode("ascii")
        + b"\nadj\n"
        + np.ascontiguousarray(dates, dtype="<i8").tobytes()
        + b"\n"
        + values.tobytes()
    )
    return sha256_hex(payload)


def data_fingerprint(
    *,
    adjustment: Adjustment | str,
    adj_events: Mapping[str, pd.Series] | None = None,
) -> dict:
    """Derive the stored value's data fingerprint FROM the adjustment declaration.

    Returns a small JSON-friendly dict stored as the artifact's metadata:
    ``{fingerprint_version, schema_version, adjustment, adj_events}`` where
    ``adj_events`` is ``None`` for none/returns_invariant and ``{symbol: hash}``
    for price_level.

    Raises (never silently) on a declaration/argument contradiction:
      * price_level without ``adj_events`` — the fingerprint would omit the anchor
        it MUST include (§六.17 qfq-anchor trap);
      * none/returns_invariant WITH ``adj_events`` — the anchor is irrelevant to
        those factors, so mixing it in is a declaration error.
    """
    adj = adjustment if isinstance(adjustment, Adjustment) else Adjustment(adjustment)
    base: dict[str, object] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "schema_version": schema_version_hash(),
        "adjustment": adj.value,
        "adj_events": None,
    }
    if adj is Adjustment.PRICE_LEVEL:
        if not adj_events:
            raise ValueError(
                "a price_level factor's data fingerprint MUST include the per-symbol "
                "adj_factor event table (design §3.4 'another adj_factor event hash'); "
                "adj_events is missing. Supply {symbol: adj_factor_series} — omitting it "
                "would let an ex-date-stale value be reused (§六.17)."
            )
        base["adj_events"] = {
            str(symbol): adj_events_hash(series)
            for symbol, series in sorted(adj_events.items(), key=lambda kv: str(kv[0]))
        }
        return base
    # none / returns_invariant: the anchor is irrelevant; supplying it is an error.
    if adj_events:
        raise ValueError(
            f"a {adj.value!r} factor does not depend on the price LEVEL, so its data "
            f"fingerprint must NOT include an adj_factor event table; adj_events was "
            f"supplied. Declare 'price_level' if it truly does (D0 §1.1)."
        )
    return base


def fingerprint_json(fingerprint: Mapping[str, object]) -> str:
    """Stable JSON string of a fingerprint (for parquet metadata storage)."""
    return stable_json(dict(fingerprint))


__all__ = [
    "FINGERPRINT_VERSION",
    "adj_events_hash",
    "data_fingerprint",
    "fingerprint_json",
    "schema_version_hash",
]
