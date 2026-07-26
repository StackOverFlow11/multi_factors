"""Factor value store (refactor D3): sparse raw factor-value persistence.

A PURE ADDITIVE layer (design §9 D3 rollback note): disabling the store falls back
to full recomputation, so nothing here is on a consumption critical path in D3 (the
D4 read-through wires it into the pipeline). The store persists RAW factor data
only (processed panels never land here as a source of truth, red line #7), addressed
by a COMPLETE key — factor_id + params + code_hash + view — with a data fingerprint
(schema version + adjustment-derived anchor events) validated on read.

What "raw factor data" is depends on the factor, and there are exactly two shapes:
its VALUE, or — for a factor whose value is a function of the loaded universe —
its universe-independent per-symbol intermediate. ``values`` documents the two and
``factors.materialize.stores_intermediate`` documents which factor gets which and
why; neither is restated here.

Layering (red line #10): ``factors.store`` never imports ``qt``.
"""

from __future__ import annotations

from factors.store.code_hash import (
    ALLOWED_INTERNAL_IMPORTS,
    code_hash,
    factor_source_files,
    module_import_violations,
    shared_set_labeled_files,
)
from factors.store.fingerprint import (
    FINGERPRINT_VERSION,
    adj_events_hash,
    data_fingerprint,
    schema_version_hash,
)
from factors.store.hashing import params_hash, short_hash
from factors.store.incremental import (
    CacheHorizonConfig,
    IncrementalResult,
    endpoint_horizons,
    factor_lookback_depth,
    overlap_window,
    tail_recompute,
)
from factors.store.keys import StoreKey, store_key
from factors.store.run_registry import (
    STATUSES,
    RunRecord,
    RunRegistry,
    assert_registry_consistent,
)
from factors.store.values import FactorValueStore

__all__ = [
    "ALLOWED_INTERNAL_IMPORTS",
    "FINGERPRINT_VERSION",
    "STATUSES",
    "CacheHorizonConfig",
    "FactorValueStore",
    "IncrementalResult",
    "RunRecord",
    "RunRegistry",
    "StoreKey",
    "adj_events_hash",
    "assert_registry_consistent",
    "code_hash",
    "data_fingerprint",
    "endpoint_horizons",
    "factor_lookback_depth",
    "factor_source_files",
    "module_import_violations",
    "overlap_window",
    "params_hash",
    "schema_version_hash",
    "shared_set_labeled_files",
    "short_hash",
    "store_key",
    "tail_recompute",
]
