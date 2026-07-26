"""Factor service: the thin read-through onto the value store (design §3.5, D4).

Two functions and one frozen data class — NOT a behavior-class stack (R25). Both
functions READ-THROUGH the value store: a hit returns the stored RAW values, a
miss triggers the ONE materializer engine (``factors.materialize``) to fill the
gap and append, then re-reads. There is no second implementation path, which is
what makes ``panel[d] ≡ cross_section(d)`` a trivial read-layer smoke and the
real guarantee ``single-point fill ≡ batch fill`` (design §3.5 P8).

THE GAP IS A (DATE, SYMBOL) QUESTION (D4c). It used to be a date-only question,
which silently assumed every fill had covered the same universe; see
:func:`_missing_request` for what that cost. Two consequences of asking it
properly, both deliberate:

* a fill RECORDS THE CELLS IT COVERED (:func:`_record_fill_footprint`), so
  "no stored row" means "never computed" rather than doubling as "computed, no
  value" — otherwise a factor that emits nothing for 45% of the cells it is asked
  about would be re-materialized on every request;
* the SERVED shape follows, and it follows DIFFERENTLY for the two payloads, so
  it is stated per payload rather than once and wrongly:

  - VALUE payload (every factor but one): the panel now carries an explicit NaN
    row where the factor has no value, instead of no row at all. Nothing that
    reads a value is affected (NaN already meant "missing"), but a consumer that
    counts ROWS is looking at a different denominator than before — the cells
    asked about, not the cells the factor emitted.
  - INTERMEDIATE payload (the cross-sectional factor): unchanged, because the
    combine decides the served shape and it emits only the finite-pair cells.
    The NaN footprint rows are in the store and are dropped by the combine, so a
    requested name with nothing to standardize is ABSENT from the panel, not NaN
    in it (measured: 24 names + 3 with no bars -> 48 served rows, not 54).

A factor whose VALUE depends on the loaded universe cannot have that value stored
under a key that does not name the universe. It stores its per-symbol intermediate
and its cross-sectional combine runs HERE, at read-assembly, over the universe the
caller asked for (:func:`_assemble`; ``materialize.stores_intermediate``).

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
from factors.compute.minute.binding import combine_minute_stats
from factors.materialize import (
    MaterializeSources,
    materialize_intermediate_range,
    materialize_range,
    payload_columns,
    requested_universe,
    stores_intermediate,
)
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


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )


def _missing_request(
    stored: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    *,
    force_all: bool,
) -> tuple[pd.DatetimeIndex, list[str]]:
    """The requested (date, SYMBOL) cells the store does not carry (D4c).

    THE SYMBOL HALF IS THE FIX. The criterion used to ask only "which requested
    DATES are absent", which silently assumed every fill covered the same universe:
    fill a store with 12 names, then ask the SAME store for 24, and it answered
    "no missing dates" and served 12 — the other 12 vanished from the cross-section
    with no recompute, no error and no disclosure (measured: 24 rows where 48 were
    asked for, provider calls 0). A missing symbol is a sample-coverage bias and
    must be loud or filled, never dropped (the I5a red line); the design's own A1
    revision rejected "batch the callers by symbol" for exactly this reason, while
    the service was carrying the defect.

    Returns the dates to emit over and the symbols to emit FOR. Restricting the
    fill to the symbols that need one is safe because a factor value is computed
    per symbol (red line #3: cross-symbol isolation) — for a cross-sectional factor
    it is the per-symbol INTERMEDIATE that is stored, which is likewise per-symbol.

    ``force_all`` (diagnostics) re-materializes the whole request; see
    :func:`_ensure_coverage`.
    """
    if force_all or stored is None or stored.empty:
        return dates, list(symbols)
    index = stored.index
    if index.has_duplicates:  # coverage is a set question; duplicates cannot change it
        index = index[~index.duplicated()]
    present = (
        pd.Series(True, index=index)
        .unstack(level=SYMBOL_LEVEL, fill_value=False)
        .reindex(index=dates, columns=symbols, fill_value=False)
        .to_numpy(dtype=bool)
    )
    if present.all():
        return pd.DatetimeIndex([]), []
    absent = ~present
    miss_dates = dates[absent.any(axis=1)]
    miss_symbols = [s for s, need in zip(symbols, absent.any(axis=0)) if need]
    return miss_dates, miss_symbols


def _record_fill_footprint(
    fresh: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Give every cell the fill COVERED a row, so absent means NEVER COMPUTED.

    A factor does not emit a row for every (date, symbol) it is asked about — a
    suspended day, a day its own gates reject, a warm-up day below its minimum
    valid-day count. Measured on real bars (24 CSI500 names x March 2023): 0% of
    cells for seven of the minute factors, but 44.2-45.1% for the three whose
    per-day gates reject days.

    Without this, "no stored row" answers two different questions with one silence
    — "nobody computed this" and "this was computed and there is nothing" — and the
    per-(date, symbol) gap criterion, which has to treat silence as the first,
    would re-materialize those cells on EVERY request. That is not hypothetical:
    the D5b property "a warm store must not re-read minute bars" goes red for
    ``ridge_minute_return_20`` without this, and the honest fix is the engine, not
    the test (§六.16).

    So the fill writes an explicit NaN row for the cells it covered and produced
    nothing for. This adds no meaning to the store that was not already there — a
    stored NaN already meant "no finite value here", and both cases are exactly
    that. What it adds is the ability to tell computed-empty from never-computed,
    which is the question the criterion asks.

    NOTE (inherited, not introduced): the data fingerprint does not describe how
    much upstream data the cache held at fill time, so a value computed from a
    partially warmed cache is reused as-is after the cache grows. That is already
    true of every partially-computed symbol; recording empty cells extends it to
    wholly-empty ones. Refilling after a cache warm-up is a ``force`` on the cache
    side, not something the value store can infer.
    """
    grid = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates), list(symbols)], names=[DATE_LEVEL, SYMBOL_LEVEL]
    )
    if fresh.empty:
        return pd.DataFrame(float("nan"), index=grid, columns=list(columns))
    return fresh.reindex(fresh.index.union(grid))


def _materialize_payload(
    factor: Factor,
    *,
    view: View,
    symbols: list[str],
    emit_start: pd.Timestamp,
    emit_end: pd.Timestamp,
    sources: MaterializeSources,
    cutoff: str,
    diagnostics: list | None,
) -> pd.DataFrame:
    """What the store persists for ``factor`` over the window: value or intermediate."""
    if stores_intermediate(factor):
        return materialize_intermediate_range(
            factor, view=view, symbols=symbols, emit_start=emit_start,
            emit_end=emit_end, sources=sources, decision_cutoff=cutoff,
            diagnostics=diagnostics,
        )
    series = materialize_range(
        factor, view=view, symbols=symbols, emit_start=emit_start, emit_end=emit_end,
        sources=sources, decision_cutoff=cutoff, diagnostics=diagnostics,
    )
    return series.rename(factor.name).to_frame()


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
) -> pd.DataFrame:
    """Read-through: fill any (date, symbol) the store lacks; return the payload.

    A miss materializes ``[min(missing date), max(missing date)]`` for the symbols
    that need it (the single engine) and upserts it; already-present cells in that
    range are recomputed identically and win the dedup harmlessly. Returns the
    stored payload frame (possibly empty). The per-date fill (dates = one) and the
    batch fill (dates = many) leave BIT-IDENTICAL stores because the materializer's
    trailing trim is warmup-consistent (design §3.5 P8).

    The payload is the factor's VALUE, except for a factor whose value depends on
    the loaded universe — that one stores its per-symbol intermediate and the
    combine happens at assembly (see :func:`_assemble` and
    ``materialize.stores_intermediate``).

    ``diagnostics``: when a sink is supplied the WHOLE requested range is
    materialized even on a full store hit. The store persists factor data, not the
    day-level gate-attrition counts a scarcity disclosure reduces, so a warm store
    can serve the values and cannot serve the disclosure. The two honest options
    were "recompute" and "publish the report without the section"; the second is
    the silent degradation this project forbids, so the cost is paid and stated.
    Values are unaffected either way (the engine is deterministic in code + data,
    which is exactly what the store key asserts).
    """
    columns = payload_columns(factor)
    stored = store.read_valid_frame(
        key, expected_fingerprint=fingerprint, columns=columns
    )
    fill_dates, fill_symbols = _missing_request(
        stored, dates, symbols, force_all=diagnostics is not None
    )
    if fill_symbols:
        fresh = _materialize_payload(
            factor, view=view, symbols=fill_symbols, emit_start=fill_dates.min(),
            emit_end=fill_dates.max(), sources=sources, cutoff=cutoff,
            diagnostics=diagnostics,
        )
        store.upsert_frame(
            key,
            _record_fill_footprint(fresh, fill_dates, fill_symbols, columns),
            fingerprint=fingerprint,
        )
        stored = store.read_valid_frame(
            key, expected_fingerprint=fingerprint, columns=columns
        )
    if stored is None:
        return pd.DataFrame(
            {c: pd.Series([], dtype=float) for c in columns}, index=_empty_index()
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
    factors: dict[str, Factor],
    payload_by_id: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    """Slice each stored payload to (dates x universe) and stack the values.

    THE CROSS-SECTIONAL COMBINE HAPPENS HERE, once per request, on exactly the
    universe the caller asked for. That is what makes the stored artifact
    universe-independent: the store holds per-symbol intermediates, and "which
    universe standardizes them" is a property of the READ, not of the artifact —
    so the same store serves a 12-name and a 24-name request correctly, and
    neither can pollute the other (D4c / design revision A2).
    """
    symbol_set = set(map(str, symbols))
    columns: dict[str, pd.Series] = {}
    for fid, factor in factors.items():
        payload = payload_by_id[fid]
        if payload.empty:
            sliced = payload
        else:
            d = payload.index.get_level_values(DATE_LEVEL)
            sym = payload.index.get_level_values(SYMBOL_LEVEL)
            sliced = payload[d.isin(dates) & pd.Index(sym).isin(symbol_set)]
        if stores_intermediate(factor):
            columns[fid] = combine_minute_stats(factor, sliced)
        elif sliced.empty:
            columns[fid] = pd.Series([], index=_empty_index(), dtype=float, name=fid)
        else:
            columns[fid] = sliced[factor.name]
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
    # THE caller-list normalization (imported, never re-implemented): everything
    # below — the gap criterion, the fill footprint, the engine call and the
    # cross-section — reads this list, so a repeat here would be a repeat in the
    # store. See ``materialize.requested_universe`` for what a second copy costs.
    symbols = requested_universe(universe)
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

    factors: dict[str, Factor] = {}
    payload_by_id: dict[str, pd.DataFrame] = {}
    for fid in factor_ids:
        factor = _build_factor(fid, params_by_id)
        factors[fid] = factor
        fp = _fingerprint(factor)
        key = store_key(factor, view=resolved_view.value, params=(params_by_id or {}).get(fid))
        payload_by_id[fid] = _ensure_coverage(
            factor, key, fp, dates=dates, symbols=symbols, store=store,
            sources=sources, view=resolved_view, cutoff=cutoff,
            diagnostics=diagnostics,
        )
    return _assemble(factors, payload_by_id, dates, symbols)


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
