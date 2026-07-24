"""Tail-recompute incremental engine (refactor D3, commit 4, §3.3 / R4 / R5).

The 21:00 factor-update story: read the store tail, recompute a bounded overlap
window, validate it bit-for-bit against the stored values, then truncate the
overlap and append. The overlap validation is the ONLY detector of a silent
upstream data revision, so it is an entry ticket, not an option (§六.14).

The overlap window (design §3.3, R4):

    w_overlap = max(W_dep, max endpoint revision horizon)

* ``W_dep`` = the factor's TRANSITIVE lookback depth (``spec.lookback_depth``) —
  NOT the headline window. A nested-lookback factor (ridge = lookback_days=20
  over baseline_days=20 => depth ~40) sized by the headline 20 systematically
  under-samples the overlap's earliest days, which then FALSELY mismatch every
  night (§六.18). The depth is HARD-REQUIRED: a missing declaration is a readable
  error, never a silent 0 or headline fallback.
* the revision horizon of each input endpoint comes from
  ``data.availability_policy.revision_horizon`` fed the LIVE cache config (R5), so
  bumping ``refresh_recent_days`` widens the overlap. Immutable endpoints
  (stk_mins 1min) contribute 0.

The recompute callable models a materializer told to emit ``[emit_start, end]``
while loading ``warmup`` (= ``W_dep``) trading days of input history before
``emit_start`` — so the entire overlap is warm and a real mismatch means a real
revision, not a warm-up artifact.

Mismatch action (§3.3 修订 2): the whole column of every mismatched symbol is
re-computed (not locally patched), classified by the factor's ``adjustment``:
``price_level`` mismatches are EXPECTED (an ex-date re-based the level — the
key/fingerprint invalidation path); ``returns_invariant`` / ``none`` mismatches
mean upstream data actually changed and are recorded LOUDLY.

Layering: ``factors.store`` never imports ``qt`` (red line #10); this uses stdlib
+ numpy/pandas + the availability-policy leaf + the sibling store modules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.availability_policy import Adjustment, revision_horizon
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.base import Factor
from factors.store.keys import StoreKey
from factors.store.values import FactorValueStore

#: recompute(emit_start, end, warmup) -> Series over [emit_start, end], having
#: loaded ``warmup`` trading days of input history before emit_start. emit_start
#: is None to compute from the very beginning (a cold store / full column).
RecomputeFn = Callable[[pd.Timestamp | None, pd.Timestamp, int], pd.Series]


@dataclass(frozen=True)
class CacheHorizonConfig:
    """The LIVE cache-config values the revision horizon is derived from (R5).

    D4-WIRING NOTE (review LOW-2): the D4 materializer MUST construct this from the
    live ``data.cache`` config — ``refresh_recent_days=cfg.data.cache.
    refresh_recent_days`` and ``fina_tail_days=cfg.data.cache.fina_tail_days`` (the
    single source unified in commit 1). The ``400`` default here is a convenience
    for tests / standalone use, NOT a value to rely on in production wiring — a
    config whose fina tail differs from 400 would silently under/over-size the
    overlap if the default were used.
    """

    refresh_recent_days: int
    fina_tail_days: int = 400

    def overrides(self) -> dict[str, int]:
        return {"fina_indicator": int(self.fina_tail_days)}


def factor_lookback_depth(factor: Factor) -> int:
    """The factor's transitive lookback depth, HARD-REQUIRED (never a silent 0)."""
    depth = factor.spec.lookback_depth
    if depth is None:
        raise ValueError(
            f"factor {factor.name!r} has no spec.lookback_depth declaration, so the "
            f"incremental overlap window cannot be sized. Declare the TRANSITIVE "
            f"lookback depth (trading days: headline + nested baselines + valid-day "
            f"slack, design §六.18) on its FactorSpec — the engine refuses to fall "
            f"back to 0 or the headline window (§3.3 R4)."
        )
    return int(depth)


def endpoint_horizons(factor: Factor, cfg: CacheHorizonConfig) -> dict[str, int | None]:
    """Per-input-endpoint revision horizon, resolved from the live cache config."""
    out: dict[str, int | None] = {}
    for req in factor.spec.requires or ():
        out[req.source] = revision_horizon(
            req.source,
            refresh_recent_days=cfg.refresh_recent_days,
            recent_tail_overrides=cfg.overrides(),
        )
    return out


def overlap_window(factor: Factor, cfg: CacheHorizonConfig) -> int:
    """``max(W_dep, max endpoint revision horizon)`` — the incremental overlap (R4)."""
    w_dep = factor_lookback_depth(factor)
    horizons = [h for h in endpoint_horizons(factor, cfg).values() if h is not None]
    return max([w_dep, *horizons])


@dataclass(frozen=True)
class IncrementalResult:
    """Immutable summary of one tail-recompute (no secrets)."""

    action: str  # "cold_full" | "appended" | "revalidated_only"
    rows_appended: int
    overlap_rows_validated: int
    overlap_window: int
    lookback_depth: int
    mismatched_symbols: tuple[str, ...] = ()
    revision_detected: bool = False  # loud for returns_invariant/none; expected for price_level
    recolumned_symbols: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def _dates(series: pd.Series) -> pd.DatetimeIndex:
    return series.index.get_level_values(DATE_LEVEL)


def _symbols(series: pd.Series) -> pd.Index:
    return series.index.get_level_values(SYMBOL_LEVEL)


def _mismatched_symbols(stored_overlap: pd.Series, new_overlap: pd.Series) -> list[str]:
    """Symbols whose overlap values differ bit-for-bit (NaN-aware) between the two."""
    old = stored_overlap.sort_index(kind="mergesort")
    new = new_overlap.reindex(old.index)
    ov = old.to_numpy(dtype=float)
    nv = new.to_numpy(dtype=float)
    # NaN-aware inequality: unequal, and not (both NaN).
    both_nan = np.isnan(ov) & np.isnan(nv)
    differs = (~(ov == nv)) & ~both_nan
    if not differs.any():
        return []
    syms = old.index.get_level_values(SYMBOL_LEVEL)[differs]
    return sorted(set(map(str, syms)))


def tail_recompute(
    store: FactorValueStore,
    key: StoreKey,
    factor: Factor,
    *,
    recompute: RecomputeFn,
    today: pd.Timestamp,
    horizon_cfg: CacheHorizonConfig,
    fingerprint: dict,
    logger: logging.Logger | None = None,
) -> IncrementalResult:
    """Incrementally update the stored artifact for ``key`` (design §3.3).

    Cold store -> a full compute. Warm store -> recompute ``[last - w_overlap,
    today]``, validate the overlap bit-for-bit, re-column any mismatched symbol
    (classified by ``adjustment``), and append the new tail.
    """
    w_dep = factor_lookback_depth(factor)
    stored = store.read(key)
    if stored is None or stored.empty:
        full = recompute(None, today, w_dep)
        store.write(key, full, fingerprint=fingerprint)
        return IncrementalResult(
            action="cold_full",
            rows_appended=len(full),
            overlap_rows_validated=0,
            overlap_window=w_dep,
            lookback_depth=w_dep,
        )

    # LOW-1: validate the stored fingerprint's SCHEMA dimension FIRST. A schema-version
    # change (the PIT/schema machinery changed the values) makes stored values stale
    # for a reason that is NOT an upstream data revision — so a mismatch is a MISS
    # (full recompute), never a (loud) returns_invariant revision. This routes through
    # the fingerprint the way ``read_valid`` does, but keeps the price_level per-symbol
    # anchor mismatch on the existing adjustment-classified overlap path below.
    stored_fp = store.stored_fingerprint(key)
    if stored_fp is None or stored_fp.get("schema_version") != fingerprint.get(
        "schema_version"
    ):
        full = recompute(None, today, w_dep)
        store.write(key, full, fingerprint=fingerprint)
        return IncrementalResult(
            action="schema_void_full",
            rows_appended=len(full),
            overlap_rows_validated=0,
            overlap_window=w_dep,
            lookback_depth=w_dep,
            notes=(
                "schema-version changed (or the stored fingerprint was absent); full "
                "recompute — this is a schema miss, NOT an upstream data revision.",
            ),
        )

    stored = stored.sort_index(kind="mergesort")
    dates = pd.DatetimeIndex(sorted(pd.unique(_dates(stored))))
    last = dates[-1]
    w = overlap_window(factor, horizon_cfg)
    emit_start = dates[-(w + 1)] if len(dates) > w else dates[0]

    recomputed = recompute(emit_start, today, w_dep).sort_index(kind="mergesort")
    rec_dates = _dates(recomputed)
    new_overlap = recomputed[rec_dates <= last]
    tail = recomputed[rec_dates > last]

    stored_overlap = stored[_dates(stored) >= emit_start]
    mismatched = _mismatched_symbols(stored_overlap, new_overlap)

    is_price_level = factor.spec.adjustment is Adjustment.PRICE_LEVEL
    notes: list[str] = []
    recolumned: tuple[str, ...] = ()

    # assemble: untouched history + recomputed overlap/tail for normal symbols.
    head = stored[_dates(stored) < emit_start]
    combined = pd.concat([head, recomputed])
    combined = combined[~combined.index.duplicated(keep="last")]

    if mismatched:
        if is_price_level:
            notes.append(
                f"price_level overlap mismatch on {len(mismatched)} symbol(s): an "
                f"ex-date re-based the level (expected; re-columning via the "
                f"fingerprint-invalidation path)."
            )
        else:
            msg = (
                f"tail-recompute: upstream data REVISION detected for factor "
                f"{factor.name!r} ({factor.spec.adjustment}) on {len(mismatched)} "
                f"symbol(s) in the overlap window — re-columning them."
            )
            notes.append(msg)
            if logger is not None:
                logger.warning(msg)
        # whole-column recompute for the mismatched symbols (not a local patch).
        full = recompute(None, today, w_dep).sort_index(kind="mergesort")
        combined = combined[~_symbols(combined).isin(mismatched)]
        full_mis = full[_symbols(full).isin(mismatched)]
        combined = pd.concat([combined, full_mis])
        recolumned = tuple(mismatched)

    combined = combined[~combined.index.duplicated(keep="last")].sort_index(kind="mergesort")
    store.write(key, combined, fingerprint=fingerprint)

    return IncrementalResult(
        action="appended" if len(tail) else "revalidated_only",
        rows_appended=len(tail),
        overlap_rows_validated=len(stored_overlap),
        overlap_window=w,
        lookback_depth=w_dep,
        mismatched_symbols=tuple(mismatched),
        revision_detected=bool(mismatched) and not is_price_level,
        recolumned_symbols=recolumned,
        notes=tuple(notes),
    )


__all__ = [
    "CacheHorizonConfig",
    "IncrementalResult",
    "RecomputeFn",
    "endpoint_horizons",
    "factor_lookback_depth",
    "overlap_window",
    "tail_recompute",
]
