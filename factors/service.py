"""Factor service: the thin read-through onto the value store (design §3.5, D4).

Two functions and one frozen data class — NOT a behavior-class stack (R25). Both
functions READ-THROUGH the value store: a hit returns the stored RAW values, a
miss triggers the ONE materializer engine (``factors.materialize``) to fill the
gap and append, then re-reads. There is no second implementation path, which is
what makes ``panel[d] ≡ cross_section(d)`` a trivial read-layer smoke and the
real guarantee ``single-point fill ≡ batch fill`` (design §3.5 P8).

Contract (all structural, not comments):
* only data with ``available_time <= decision`` is computed — the materializer
  applies the (source, view) availability lag; the service never relaxes it;
* it NEVER touches execution prices — forward returns are born only at the
  alpha/eval boundary (invariant #1); this returns raw factor VALUES only;
* it returns RAW values (processed is a consumer-side decision, red line #7);
* the view x return-basis pairing is enforced at CALL time via the D0/D1
  ``require_legal_pairing`` (decision<->exec_to_exec / close<->close_to_close);
  any other pairing is a readable error, not a doc convention.

Layering (red line #10): ``factors.service`` imports the factor layer + the pure
availability leaf only; the data access (store + providers) is INJECTED by the
consumer, so ``factors`` never carries a feed, a token, or a qt import.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from data.availability_policy import ReturnBasis, View, require_legal_pairing
from data.clean.intraday_schema import DEFAULT_DECISION_TIME
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors import registry as factor_registry
from factors.base import Factor
from factors.materialize import MaterializeSources, materialize_range
from factors.store.fingerprint import data_fingerprint
from factors.store.keys import store_key
from factors.store.values import FactorValueStore


@dataclass(frozen=True)
class DecisionPoint:
    """A decision instant: the date and the within-day cutoff (default 14:50).

    ``cutoff`` is the same string the FactorSpec intraday block declares as its
    ``decision_cutoff`` (single source); the materializer filters minute bars to
    ``available_time <= date + cutoff``.
    """

    date: pd.Timestamp
    cutoff: str = DEFAULT_DECISION_TIME


def _build_factor(factor_id: str, params_by_id: Mapping[str, Mapping[str, object]] | None) -> Factor:
    params = (params_by_id or {}).get(factor_id)
    return factor_registry.build(factor_id, params)


def _fingerprint(factor: Factor) -> dict:
    """The store data fingerprint for ``factor`` (none/returns_invariant here).

    price_level factors need the per-symbol adj_factor event table; the closing
    14 have zero such members, so ``data_fingerprint`` raises readably for them
    (never a silent wrong fingerprint) until the service wires an adj-factor
    source for that case (deferred with the ex-date masking).
    """
    return data_fingerprint(adjustment=factor.spec.adjustment)


def _ensure_coverage(
    factor: Factor,
    key,
    fingerprint: dict,
    *,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    store: FactorValueStore,
    sources: MaterializeSources,
    view: View,
    cutoff: str,
    diagnostics: list | None = None,
) -> pd.Series:
    """Read-through: fill any date the store lacks for ``factor``, return the Series.

    A miss over the requested ``dates`` materializes ``[min(missing), max(missing)]``
    (the single engine) and upserts it; already-present dates in that range are
    recomputed identically and win the dedup harmlessly. Returns the stored Series
    (possibly empty). The per-date fill (dates = one) and the batch fill (dates =
    many) leave BIT-IDENTICAL stores because the materializer's trailing trim is
    warmup-consistent (design §3.5 P8).

    ``diagnostics``: when a sink is supplied the WHOLE requested range is
    materialized even on a full store hit. The store persists VALUES, not the
    day-level gate-attrition counts a scarcity disclosure reduces, so a warm store
    can serve the values and cannot serve the disclosure. The two honest options
    were "recompute" and "publish the report without the section"; the second is
    the silent degradation this project forbids, so the cost is paid and stated.
    Values are unaffected either way (the engine is deterministic in code + data,
    which is exactly what the store key asserts).
    """
    stored = store.read_valid(key, expected_fingerprint=fingerprint)
    have = (
        pd.DatetimeIndex(pd.unique(stored.index.get_level_values(DATE_LEVEL)))
        if stored is not None and not stored.empty
        else pd.DatetimeIndex([])
    )
    missing = dates if diagnostics is not None else dates[~dates.isin(have)]
    if len(missing):
        fresh = materialize_range(
            factor,
            view=view,
            symbols=list(symbols),
            emit_start=missing.min(),
            emit_end=missing.max(),
            sources=sources,
            decision_cutoff=cutoff,
            diagnostics=diagnostics,
        )
        if not fresh.empty:
            store.upsert(key, fresh, fingerprint=fingerprint)
        stored = store.read_valid(key, expected_fingerprint=fingerprint)
    if stored is None:
        return pd.Series(
            [],
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
                names=[DATE_LEVEL, SYMBOL_LEVEL],
            ),
            dtype=float,
            name=factor.name,
        )
    return stored


def _uniform_cutoff(decisions: list[DecisionPoint]) -> str:
    cutoffs = {d.cutoff for d in decisions}
    if len(cutoffs) != 1:
        raise ValueError(
            f"panel() requires all DecisionPoints to share one cutoff; got {sorted(cutoffs)}. "
            f"Split the call per cutoff (the minute cutoff enters the store view semantics)."
        )
    return next(iter(cutoffs))


def _assemble(
    factor_ids: list[str],
    series_by_id: dict[str, pd.Series],
    dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    """Slice each factor's stored Series to (dates x universe) and stack to columns."""
    symbol_set = set(map(str, symbols))
    columns: dict[str, pd.Series] = {}
    for fid in factor_ids:
        s = series_by_id[fid]
        if s.empty:
            columns[fid] = s
            continue
        d = s.index.get_level_values(DATE_LEVEL)
        sym = s.index.get_level_values(SYMBOL_LEVEL)
        keep = d.isin(dates) & pd.Index(sym).isin(symbol_set)
        columns[fid] = s[keep]
    frame = pd.concat(columns, axis=1)
    frame.index = frame.index.set_names([DATE_LEVEL, SYMBOL_LEVEL])
    return frame.sort_index(kind="mergesort")


def panel(
    factor_ids: Iterable[str],
    universe: Iterable[str],
    decisions: list[DecisionPoint],
    *,
    store: FactorValueStore,
    sources: MaterializeSources,
    view: object = View.DECISION,
    basis: object = ReturnBasis.EXEC_TO_EXEC,
    params_by_id: Mapping[str, Mapping[str, object]] | None = None,
    diagnostics: list | None = None,
) -> pd.DataFrame:
    """Raw factor values for ``factor_ids`` over ``decisions`` x ``universe``.

    Read-through: any decision date the store lacks is materialized once (batch)
    and appended. The view x basis pairing is enforced up front.

    ``diagnostics``: optional per-day gate-attrition sink for the ONE factor whose
    scarcity disclosure needs it — see :func:`_ensure_coverage`. Requesting it for
    several factors at once is refused rather than served: the frames from two
    factors would land in one list with nothing to tell them apart, and a
    summarizer would silently reduce the mixture.
    """
    resolved_view, _ = require_legal_pairing(view, basis)
    factor_ids = list(factor_ids)
    symbols = [str(s) for s in universe]
    if not decisions:
        raise ValueError("panel() needs at least one DecisionPoint.")
    if diagnostics is not None and len(factor_ids) != 1:
        raise ValueError(
            f"panel(diagnostics=...) takes exactly ONE factor_id (got "
            f"{factor_ids}): the per-day diagnostics frames carry no factor label, "
            f"so several factors' frames in one sink would be reduced together."
        )
    cutoff = _uniform_cutoff(decisions)
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d.date).normalize() for d in decisions}))

    series_by_id: dict[str, pd.Series] = {}
    for fid in factor_ids:
        factor = _build_factor(fid, params_by_id)
        fp = _fingerprint(factor)
        key = store_key(factor, view=resolved_view.value, params=(params_by_id or {}).get(fid))
        series_by_id[fid] = _ensure_coverage(
            factor, key, fp, dates=dates, symbols=symbols, store=store,
            sources=sources, view=resolved_view, cutoff=cutoff,
            diagnostics=diagnostics,
        )
    return _assemble(factor_ids, series_by_id, dates, symbols)


def cross_section(
    factor_ids: Iterable[str],
    universe: Iterable[str],
    decision: DecisionPoint,
    *,
    store: FactorValueStore,
    sources: MaterializeSources,
    view: object = View.DECISION,
    basis: object = ReturnBasis.EXEC_TO_EXEC,
    params_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Raw factor values for ``factor_ids`` at one ``decision`` x ``universe``.

    A convenience over :func:`panel` with a single decision — the SAME read-through
    and the SAME engine, so ``panel[d] ≡ cross_section(d)`` on the same store.
    """
    return panel(
        factor_ids, universe, [decision], store=store, sources=sources,
        view=view, basis=basis, params_by_id=params_by_id,
    )


__all__ = ["DecisionPoint", "cross_section", "panel"]
