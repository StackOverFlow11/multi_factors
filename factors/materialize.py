"""The ONE vectorized factor-value engine (factor-refactor D4, design §3.5).

The materializer is the single path that turns injected data-layer inputs into
RAW factor values for a view over a date range. ``factors.service`` read-through
routes every store miss here; nothing else computes factor values. It is
factor-agnostic and view-agnostic-compute-preserving (design §1.4 mechanism 3):

* it reads the factor's ``requires`` to decide DAILY vs MINUTE input;
* it pulls inputs from INJECTED providers (layering red line #3/#10: ``factors``
  never touches a feed, a token, or qt — the consumer injects the data access);
* it applies the (source, view) availability lag from ``factors.view_lag`` (R18:
  daily prev-day shift with the field-level ``open`` exception; minute
  ``available_time <= d 14:50`` cutoff; PIT as-of is a no-op here);
* it trims the input to EXACTLY the factor's ``lookback_depth`` trailing trading
  days before the emit window, then computes and emits — so a single-date fill
  and a batch fill produce bit-identical values (design §3.5 P8);
* ``masked`` factors have their ex-date rows NaN'd via the BOOLEAN ex-date mask
  (§1.3 note 4; the ``adj_factor`` numeric never enters the frame). The closing
  14 factors are all non-masked, so this is a no-op for them (fixture-tested).

The daily factor path calls ``factor.compute`` on the lagged panel; the minute
path calls the ``factors.compute.minute.binding`` for the factor and the
cutoff-filtered bars. The forward-return boundary is elsewhere: a factor value
never sees a future return (invariant #1) — the materializer only reads history.

MINUTE LOADING IS PER-SYMBOL STREAMING (D4b). One symbol's bars are read,
reduced to that symbol's daily intermediate, and discarded before the next; the
intermediates are assembled and the factor's cross-sectional combine runs ONCE.
The all-symbol minute panel is never materialized — measured on the evaluation
plane it is 52.7 GB nominal and ~115 GB peak RSS against 56 GB available, so the
previous single-frame load could not run at all, for POOLED and BOUNDED factors
alike (``docs/factors/d5_saturation_feasibility.md``). Two consequences are
deliberate and disclosed:

* PER-SYMBOL TRIM: the trailing-trading-day trim and the pooled saturation
  criterion now read THIS SYMBOL's trading days instead of the loaded universe's
  union. Both are strictly more conservative (a symbol's own day list is a subset
  of the union, so the same "N trailing days" reaches at least as far back), and
  both remove a cross-symbol coupling — under the union rule another symbol's
  calendar could decide how much history this symbol was warmed with, which is
  itself a load-geometry dependence (red line #6). The factor math, its
  parameters and its thresholds are untouched.
* The pooled cost D4 recorded ("any requested symbol that cannot accumulate
  ``lookback_days`` valid days drags the WHOLE universe's load back to the
  declared floor") is gone: each symbol expands to its own terminal. Measured on
  the evaluation universe, 84 of 995 symbols list after the emit start and can
  never satisfy the criterion, so under the old rule the floor load was
  CONSTRUCTIVE for every pooled factor.

That same per-symbol stage is also what the STORE persists for a factor whose
value is cross-sectional (D4c): :func:`materialize_intermediate_range` stops
before the combine, and the consumer runs the combine over the universe it asked
for. See :func:`stores_intermediate` for why the alternative — a universe
dimension on the store key — was rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from data.availability_policy import STK_MINS_1MIN, OvernightBoundary, View
from data.clean.intraday_schema import DEFAULT_DECISION_TIME
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.base import Factor
from factors.compute.minute.binding import (
    combine_minute_stats,
    is_cross_sectional_minute,
    is_minute_bound,
    is_valid_day_pooled,
    minute_diagnostics_from_bars,
    minute_intermediate_columns,
    minute_raw_from_bars,
    minute_stats_from_bars,
    pooled_baseline_days,
    pooled_lookback_days,
)
from factors.store.incremental import CacheHorizonConfig
from factors.view_lag import (
    daily_decision_lag,
    ex_date_mask,
    minute_decision_cutoff,
)

#: Extra calendar days loaded before the emit window so the exact trailing-trading
#: -day trim always has enough history (covers weekends + holiday clusters + the
#: daily shift's one extra day). Generous by design: the CORRECTNESS floor is the
#: trim to ``lookback_depth`` trading days, not this buffer (design §3.5 P8).
def _load_buffer_calendar_days(warmup: int) -> int:
    return int(warmup) * 2 + 25


#: One backward-expansion step (calendar days) of the valid-day-pooled saturation
#: loop. Only a STEP SIZE (how fast the search walks back), never a correctness
#: input: termination is decided by the structural criterion in
#: ``_materialize_pooled`` or by the provider's declared earliest boundary, so a
#: too-small chunk costs iterations, never correctness.
def _saturation_chunk_calendar_days(warmup: int) -> int:
    return int(warmup) * 2 + 40


#: Hard ceiling on saturation-expansion chunks (defensive; the loop terminates
#: structurally on value-stability or provider exhaustion long before this).
_MAX_SATURATION_CHUNKS: int = 500


# --------------------------------------------------------------------------- #
# Injected data providers (the consumer wires these to feed/cache; tests fake)
# --------------------------------------------------------------------------- #
class DailyPanelProvider(Protocol):
    """Loads the CLOSE-view daily factor-input panel (front-adjusted + enriched)."""

    def daily_panel(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """MultiIndex(date, symbol) panel, values dated at their natural close date."""


class MinuteBarProvider(Protocol):
    """Loads normalized 1min bars (cache-only) for the symbols over the window.

    Providers feeding a VALID-DAY-POOLED factor must also implement
    :meth:`earliest_available` — the saturation loop needs a RELIABLE data-start
    signal. Inferring "no more history" from an unchanged row count is WRONG: a
    long mid-history gap (a suspension) yields an unchanged count while real
    earlier history still exists (the review's counterexample #2).
    """

    def minute_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """MultiIndex(time, symbol) bars (:mod:`data.clean.intraday_schema`)."""

    def earliest_available(self, symbols: list[str]) -> pd.Timestamp:
        """The earliest date for which ANY of ``symbols`` could have bars.

        The declared lower bound of the provider's data (a cache-coverage floor
        or a documented constant). Loading from it means "all history that
        exists is loaded"; a mid-history gap must NEVER be mistaken for it.
        """


class AdjFactorProvider(Protocol):
    """Loads the RAW adj_factor series (ONLY consumed as an ex-date BOOLEAN)."""

    def adj_factor(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """MultiIndex(date, symbol) raw adj_factor (never enters the factor frame)."""


@dataclass(frozen=True)
class MaterializeSources:
    """The injected data access the materializer needs (consumer-wired)."""

    daily: DailyPanelProvider | None = None
    minute: MinuteBarProvider | None = None
    adj_factor: AdjFactorProvider | None = None


# --------------------------------------------------------------------------- #
# Live-config wiring (R5 / D3-note: NEVER the CacheHorizonConfig default)
# --------------------------------------------------------------------------- #
def build_horizon_config(cache_cfg: object) -> CacheHorizonConfig:
    """Build the incremental-overlap horizon config from the LIVE cache config.

    Reads ``refresh_recent_days`` and ``fina_tail_days`` off ``cache_cfg`` (duck
    typed, so ``qt.config`` DataCacheCfg works without a factors->qt import). Both
    are REQUIRED: the ``CacheHorizonConfig`` 400 default is a test convenience and
    a config whose fina tail differs from 400 would silently mis-size the fina
    overlap window (R5) — so a missing attribute is a readable error, never the
    default.
    """
    missing = [a for a in ("refresh_recent_days", "fina_tail_days") if not hasattr(cache_cfg, a)]
    if missing:
        raise ValueError(
            f"build_horizon_config needs the live cache config's {missing} "
            f"attribute(s); refusing to fall back to the CacheHorizonConfig default "
            f"(R5: a differing fina tail would silently mis-size the overlap)."
        )
    return CacheHorizonConfig(
        refresh_recent_days=int(cache_cfg.refresh_recent_days),
        fina_tail_days=int(cache_cfg.fina_tail_days),
    )


# --------------------------------------------------------------------------- #
# Factor kind
# --------------------------------------------------------------------------- #
def is_minute_factor(factor: Factor) -> bool:
    """True iff any of the factor's declared inputs comes from the 1min endpoint."""
    return any(r.source == STK_MINS_1MIN for r in (factor.spec.requires or ()))


def stores_intermediate(factor: Factor) -> bool:
    """True iff ``factor``'s VALUE is a function of the loaded universe (D4c).

    Such a factor's value cannot be stored: the store key is (factor, params,
    code, view) and none of those says which universe produced it, so a value
    filled under one universe would be served to another. Its universe-INDEPENDENT
    per-symbol intermediate is stored instead and the cross-sectional combine runs
    at read-assembly, over the universe the READER asked for.

    Adding "which universe" to the key was considered and rejected (design
    revision A2): it would over-invalidate the ten factors whose values are
    universe-independent, and a PIT universe is a time-varying SET, so "the
    universe" of a date range is not a single value that a key dimension could
    even hold.
    """
    return is_minute_factor(factor) and is_cross_sectional_minute(factor)


def payload_columns(factor: Factor) -> tuple[str, ...]:
    """The columns of what the store persists for ``factor`` (D4c).

    The value column for almost every factor; the per-symbol intermediate's
    columns for a cross-sectional one. The reader validates the artifact on disk
    against this, so an artifact written under the other shape is a miss.
    """
    if stores_intermediate(factor):
        return minute_intermediate_columns(factor)
    return (factor.name,)


# --------------------------------------------------------------------------- #
# Trailing-trading-day trim (the P8 correctness floor)
# --------------------------------------------------------------------------- #
def _requested_universe(symbols) -> list[str]:
    """The requested symbols as strings, DE-DUPLICATED, first occurrence order.

    The engine owns this the same way ``_symbol_bars`` owns "the provider honoured
    its ``symbols`` argument" — a caller-side and a provider-side spelling of one
    invariant: a (date, symbol) cell must be produced exactly once.

    Duplicates used to be absorbed by the providers, whose ``isin`` filter is
    idempotent, so the single-frame engine returned one row per name whatever the
    caller passed. Streaming iterates the list instead, so a repeat would be
    materialized TWICE: duplicate index rows, and — for a factor with a real
    cross-sectional combine — the repeated name entering its date's cross-section
    twice, moving that date's mean and std for EVERY member. Measured on the
    streaming test fixture with two names repeated: ``volume_peak_count_20`` gained
    8 duplicate rows with unchanged values, ``intraday_amp_cut_10`` gained 8
    duplicate rows AND changed all 48 shared cells.

    De-duplicating (rather than raising) is deliberate: it reproduces exactly what
    the single-frame engine returned for the same call, which keeps D4b a change of
    memory profile and nothing else. A raise would be a NEW failure mode for input
    the previous engine accepted — a behaviour change in the opposite direction,
    and one that belongs to whoever validates a universe, not to the value engine.
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        name = str(s)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _warmup_start(dates: pd.DatetimeIndex, emit_start: pd.Timestamp, warmup: int):
    """The date ``warmup`` trading days before ``emit_start`` (or the earliest).

    ``dates`` is the sorted unique set of trading dates present in the loaded
    input. Returns the cutoff date to keep input from; if fewer than ``warmup``
    trading days precede ``emit_start`` the earliest available date is used
    (near the data start the emit rows are honestly under-warmed -> NaN, and
    single/batch stay consistent because both trim to the same earliest date).
    """
    order = dates.sort_values()
    pos = int(order.searchsorted(emit_start, side="left"))  # index of emit_start (or ins.)
    keep_idx = max(0, pos - int(warmup))
    return order[keep_idx]


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def materialize_range(
    factor: Factor,
    *,
    view: object,
    symbols: list[str],
    emit_start: pd.Timestamp,
    emit_end: pd.Timestamp,
    sources: MaterializeSources,
    decision_cutoff: str = DEFAULT_DECISION_TIME,
    warmup: int | None = None,
    diagnostics: list | None = None,
) -> pd.Series:
    """Compute ``factor``'s RAW values for ``view`` over ``[emit_start, emit_end]``.

    Loads a generous window (so the exact trailing trim always has history),
    applies the view lag, trims to ``warmup`` (= ``spec.lookback_depth``) trailing
    trading days before ``emit_start``, computes, and returns the emit-window
    rows. Single-date and batch calls give bit-identical values (design §3.5 P8).

    ``diagnostics``: OPTIONAL list the per-symbol day-level gate-attrition frames
    are appended to (emit-window rows only), for the factors that publish a
    scarcity disclosure. Default ``None`` = the D4b-verified value path, byte for
    byte: nothing extra is computed and nothing is collected. Supplying a sink
    NEVER changes a returned value — it re-runs the per-symbol math on bars already
    in memory (no extra I/O) purely to observe the day-level counts.
    """
    resolved_view = View(view)
    symbols = _requested_universe(symbols)  # one engine-level normalization
    emit_start = pd.Timestamp(emit_start).normalize()
    emit_end = pd.Timestamp(emit_end).normalize()
    w = int(factor.spec.lookback_depth) if warmup is None else int(warmup)
    if w < 1:
        raise ValueError(f"{factor.name}: warmup must be >= 1; got {w}.")
    load_start = emit_start - pd.Timedelta(days=_load_buffer_calendar_days(w))

    if is_minute_factor(factor):
        raw = _materialize_minute(
            factor, resolved_view, symbols, load_start, emit_start, emit_end, w,
            sources, decision_cutoff, diagnostics,
        )
    else:
        if diagnostics is not None:
            raise ValueError(
                f"{factor.name} is a daily factor; per-day minute diagnostics are "
                f"only published by minute factors. Refusing to return an EMPTY "
                f"sink, which a caller would read as 'no attrition'."
            )
        raw = _materialize_daily(
            factor, resolved_view, symbols, load_start, emit_start, emit_end, w, sources,
        )

    raw = _apply_ex_date_mask(
        factor, raw, resolved_view, symbols, load_start, emit_end, sources
    )
    return _restrict_emit(raw, emit_start, emit_end, factor.name)


def materialize_intermediate_range(
    factor: Factor,
    *,
    view: object,
    symbols: list[str],
    emit_start: pd.Timestamp,
    emit_end: pd.Timestamp,
    sources: MaterializeSources,
    decision_cutoff: str = DEFAULT_DECISION_TIME,
    warmup: int | None = None,
    diagnostics: list | None = None,
) -> pd.DataFrame:
    """``factor``'s UNIVERSE-INDEPENDENT per-symbol intermediate over the window.

    Same engine, same loading geometry and the same trailing trim as
    :func:`materialize_range` — it simply stops one step earlier, before the
    cross-sectional combine. Only a factor that :func:`stores_intermediate` has a
    reason to call this: for every other factor the intermediate IS the value, so
    asking for it here would create a second name for one thing.

    A ``masked`` factor is refused rather than served: the ex-date mask belongs to
    the VALUE (it NaNs out a computed value on its ex-date), and the value does not
    exist yet at this point, so an intermediate could only carry the mask by
    applying it to the wrong object. No closing factor is masked, so this refuses
    a case that does not exist rather than guessing at it.
    """
    if not stores_intermediate(factor):
        raise ValueError(
            f"{factor.name} stores its VALUE, not an intermediate: its value does "
            f"not depend on the loaded universe, so materialize_range is the one "
            f"engine entry point for it."
        )
    if factor.spec.overnight_boundary is OvernightBoundary.MASKED:
        raise NotImplementedError(
            f"{factor.name} is both cross-sectional and overnight_boundary='masked'. "
            f"The ex-date mask applies to the VALUE, which the intermediate does not "
            f"carry, so storing the intermediate would silently drop the mask. No "
            f"closing factor is in this combination; wiring it needs the mask moved "
            f"to the read-assembly combine, deliberately and with its own test."
        )
    resolved_view = View(view)
    symbols = _requested_universe(symbols)
    emit_start = pd.Timestamp(emit_start).normalize()
    emit_end = pd.Timestamp(emit_end).normalize()
    w = int(factor.spec.lookback_depth) if warmup is None else int(warmup)
    if w < 1:
        raise ValueError(f"{factor.name}: warmup must be >= 1; got {w}.")
    load_start = emit_start - pd.Timedelta(days=_load_buffer_calendar_days(w))
    return _minute_intermediate(
        factor, resolved_view, symbols, load_start, emit_start, emit_end, w, sources,
        decision_cutoff, diagnostics,
    )


def _materialize_daily(
    factor, view, symbols, load_start, emit_start, emit_end, warmup, sources,
) -> pd.Series:
    if sources.daily is None:
        raise ValueError(
            f"{factor.name} is a daily factor but no DailyPanelProvider was injected."
        )
    panel = sources.daily.daily_panel(list(symbols), load_start, emit_end)
    if panel.empty:
        return _empty_series(factor.name)
    if view is View.DECISION:
        panel = daily_decision_lag(panel)  # prev-day shift, open same-day (R18)
    trimmed = _trim_daily(panel, emit_start, warmup)
    return factor.compute(trimmed).rename(factor.name)


def _materialize_minute(
    factor, view, symbols, load_start, emit_start, emit_end, warmup, sources,
    decision_cutoff, diagnostics=None,
) -> pd.Series:
    """STREAMING minute materialization (D4b): one symbol's bars at a time.

    Loads ONE symbol's minute history, reduces it to that symbol's small daily
    intermediate, and DISCARDS the bars before the next symbol — the multi-year
    all-symbol minute panel is never materialized. That is a structural
    requirement, not an optimization: the evaluation plane's single frame is
    52.7 GB nominal / ~115 GB peak RSS against 56 GB of memory, so the
    whole-universe load cannot run at all (``docs/factors/d5_saturation_feasibility.md``).

    THE CUT IS BEFORE THE CROSS-SECTION. The per-symbol intermediates are
    assembled into the full-universe panel and the factor's combine runs ONCE on
    it, so ``intraday_amp_cut``'s date-wise z-score still sees the whole covered
    universe (a per-WHOLE-FACTOR split would hand it a one-symbol cross-section,
    which its ``AMP_CUT_MIN_CROSS_SECTION`` gate turns entirely into NaN —
    measured, and never worked around by relaxing that definition constant).

    Two load geometries, unchanged in meaning from the single-frame engine:
    bounded factors keep the fixed trailing-trading-day trim; valid-day-POOLED
    factors expand to saturation. Both are now decided PER SYMBOL — see
    :func:`_pooled_symbol_stats` for why that is the same terminal, not a
    weakening.
    """
    if not is_minute_bound(factor):
        # Readable error (e.g. valley_price_quantile needs the daily panel too).
        if sources.minute is None:
            raise ValueError(
                f"{factor.name} is a minute factor but no MinuteBarProvider was injected."
            )
        return minute_raw_from_bars(factor, sources.minute.minute_bars([], load_start, emit_end))
    stats = _minute_intermediate(
        factor, view, symbols, load_start, emit_start, emit_end, warmup, sources,
        decision_cutoff, diagnostics,
    )
    if stats.empty:
        return _empty_series(factor.name)
    return combine_minute_stats(factor, stats)


def _minute_intermediate(
    factor, view, symbols, load_start, emit_start, emit_end, warmup, sources,
    decision_cutoff, diagnostics=None,
) -> pd.DataFrame:
    """The assembled PER-SYMBOL intermediate, emit-restricted, BEFORE the combine.

    This is the universe-INDEPENDENT half of a minute factor: each symbol's rows
    are produced from that symbol's bars alone. The materialized value is
    ``combine_minute_stats`` of it; the store persists this frame directly for a
    cross-sectional factor (D4c), so the two callers share one engine rather than
    a second per-symbol loop.
    """
    if sources.minute is None:
        raise ValueError(
            f"{factor.name} is a minute factor but no MinuteBarProvider was injected."
        )
    pooled = is_valid_day_pooled(factor)
    pooled_kw = {}
    if pooled:
        pooled_kw = {
            "floor": _provider_earliest(sources.minute, list(symbols), factor),
            "baseline_days": pooled_baseline_days(factor),
            "lookback_days": pooled_lookback_days(factor),
        }

    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        sym = str(symbol)
        if pooled:
            stats = _pooled_symbol_stats(
                factor, view, sym, emit_start, emit_end, warmup, sources,
                decision_cutoff, diagnostics=diagnostics, **pooled_kw,
            )
        else:
            stats = _bounded_symbol_stats(
                factor, view, sym, load_start, emit_start, emit_end, warmup, sources,
                decision_cutoff, diagnostics=diagnostics,
            )
        if stats is None or stats.empty:
            continue
        # Restrict to the emit window BEFORE assembling: the combine is a per-date
        # reduction (binding CONTRACT), so a kept date's value is unaffected, and
        # the assembled panel stays daily-sized.
        parts.append(_restrict_emit_frame(stats, emit_start, emit_end))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=list(minute_intermediate_columns(factor)))
    return pd.concat(parts).sort_index(kind="mergesort")


def _symbol_bars(provider, symbol: str, start, end) -> pd.DataFrame:
    """One symbol's bars from ``provider``, RE-FILTERED to that symbol.

    The re-filter is not redundant belt-and-braces: streaming makes the engine's
    correctness depend on the provider honouring its ``symbols`` argument, and a
    provider that ignores it (returning the whole store) would silently emit every
    other symbol's rows once PER STREAMED SYMBOL — duplicated index entries, an
    inflated cross-section, and for a pooled factor a saturation decision taken on
    the wrong symbol's valid days. The engine owns that invariant rather than
    trusting an injected object with it.
    """
    bars = provider.minute_bars([symbol], start, end)
    if bars.empty:
        return bars
    mine = pd.Index(bars.index.get_level_values(SYMBOL_LEVEL)) == symbol
    return bars if bool(mine.all()) else bars[mine]


def _collect_diagnostics(factor, work, emit_start, emit_end, diagnostics) -> None:
    """Append ``factor``'s emit-window per-day diagnostics for ONE symbol, if asked.

    ``work`` is the SAME cutoff-filtered, trimmed frame the value path used, so the
    counts describe the days the values were computed from — not a differently
    loaded window. Restricting to the emit window matches the disclosure's meaning
    ("the days this run scored"); the warm-up days ahead of it were loaded to make
    those days computable and were never part of the reported denominator.
    """
    if diagnostics is None:
        return
    frame = minute_diagnostics_from_bars(factor, work)
    if frame.empty:
        return
    index = pd.DatetimeIndex(frame.index)
    within = (index >= emit_start) & (index <= emit_end)
    kept = frame[within]
    if not kept.empty:
        diagnostics.append(kept)


def _bounded_symbol_stats(
    factor, view, symbol, load_start, emit_start, emit_end, warmup, sources,
    decision_cutoff, *, diagnostics=None,
) -> pd.DataFrame | None:
    """One BOUNDED minute factor's per-symbol intermediate over the fixed window.

    Load one extra calendar day past ``emit_end`` so bar cutoffs on ``emit_end``
    resolve, then apply the same trailing-trading-day trim as the single-frame
    engine. The trim's day list is now THIS SYMBOL's trading days rather than the
    loaded universe's union — see the module docstring's PER-SYMBOL TRIM note.
    """
    bars = _symbol_bars(
        sources.minute, symbol, load_start, emit_end + pd.Timedelta(days=1)
    )
    if bars.empty:
        return None
    if view is View.DECISION:
        bars = minute_decision_cutoff(bars, decision_time=decision_cutoff)
    bars = _trim_minute(bars, emit_start, warmup)
    _collect_diagnostics(factor, bars, emit_start, emit_end, diagnostics)
    return minute_stats_from_bars(factor, bars)


def _pooled_symbol_stats(
    factor, view, symbol, emit_start, emit_end, warmup, sources, decision_cutoff,
    *, floor, baseline_days, lookback_days, diagnostics=None,
) -> pd.DataFrame | None:
    """Saturation-expanding load for ONE symbol of a valid-day-pooled factor.

    STRUCTURAL saturation criterion (NOT a value-stability fixed point — that was
    disproved: one expansion chunk landing entirely inside a long no-bar gap
    leaves the emit values unchanged and would falsely terminate).

    The locking argument (why the criterion is SUFFICIENT):

    * Expanding the load backward can only change the classification of days
      within ``baseline_days`` trading days of the loaded window's start — the
      rolling baseline takes the ``baseline_days`` most recent STRICTLY-PRIOR
      same-slot observations, and once at full depth no earlier history can move
      it. Locking is therefore POSITION-MONOTONE: **drop the first
      ``baseline_days`` trading days of the loaded window and every later day's
      classification is FINAL.**
    * The trailing pool takes only the most recent ``lookback_days`` VALID days.
      So if the LOCKED sub-window already contains >= ``lookback_days`` final
      valid days at or before the earliest emit date, the pool for every emit
      date is drawn entirely from final days and can never reach back into the
      unlocked head — further history cannot change any emit value.

    Termination: the criterion above, OR the load start reaching the provider's
    DECLARED earliest boundary (then all existing history is loaded and the value
    is well-defined). A mid-history gap is never mistaken for the data start
    (that is exactly the row-count inference this replaced).

    Valid days are counted as the factor's own OUTPUT dates: a pooled factor
    emits one row per valid day, so output dates ARE the valid days — the count
    is read off the engine's own result rather than a re-implementation of each
    factor's validity rule, and a gap simply contributes no output dates (the
    count stalls and the loop keeps expanding, which is the desired behaviour).

    PER SYMBOL, and why that is the SAME terminal (D4b, not a weakening):

    * The criterion's whole content is "further history cannot change any emit
      value", and it is evaluated on the symbol's own valid days and its own
      trailing pool. So a symbol that satisfies it at depth D has, BY THE
      CRITERION, the same values it would have at any deeper load — including the
      universe-wide floor the single-frame engine used to drag everyone to.
    * The locked head is now dropped from THIS symbol's loaded days rather than
      from the loaded UNION. The union is a superset, so its ``baseline_days``-th
      day is at or before the symbol's — i.e. the per-symbol locked window is at
      or inside the union one, and the criterion is at least as hard to satisfy.
      Per-symbol therefore loads at least as deep; it never terminates earlier.
    * The requested-symbol VETO (below) still applies with this symbol as the
      requested symbol: a symbol with no output at the current depth counts zero
      valid days and cannot terminate its own loop. What it can no longer do is
      drag 994 other symbols to the floor with it, which is the entire cost D4's
      docstring recorded as "a per-symbol saturation start is the D5 optimization".
    """
    chunk = pd.Timedelta(days=_saturation_chunk_calendar_days(warmup))
    end = emit_end + pd.Timedelta(days=1)
    load_start = emit_start - chunk
    for _ in range(_MAX_SATURATION_CHUNKS):
        at_floor = load_start <= floor
        if at_floor:
            load_start = floor
        bars = _symbol_bars(sources.minute, symbol, load_start, end)
        if bars.empty:
            if at_floor:
                return None
            load_start = load_start - chunk
            continue
        work = bars
        if view is View.DECISION:
            work = minute_decision_cutoff(work, decision_time=decision_cutoff)
        stats = minute_stats_from_bars(factor, work)
        if at_floor or _pooled_pool_saturated(
            combine_minute_stats(factor, stats), work, emit_start,
            baseline_days=baseline_days, lookback_days=lookback_days,
            symbols=[symbol],
        ):
            # Only the TERMINAL load is observed: the intermediate expansion steps
            # are the search, not the computation, and reporting their day counts
            # would inflate the disclosure with days that were re-derived deeper.
            _collect_diagnostics(factor, work, emit_start, emit_end, diagnostics)
            return stats
        load_start = load_start - chunk
    # Never silently return an UNSATURATED result (red line #9: no silent
    # degradation). Unreachable in practice — 500 chunks is >160 years of history
    # — so hitting it means the loop is not converging, which must be loud.
    raise RuntimeError(
        f"{factor.name}: pooled saturation did not converge after "
        f"{_MAX_SATURATION_CHUNKS} expansion chunks (expanded back to "
        f"{load_start.date()} for symbol {symbol}, declared floor "
        f"{floor.date()}). Refusing to return an unsaturated value, which would "
        f"depend on load geometry."
    )


def _provider_earliest(provider, symbols, factor) -> pd.Timestamp:
    """The provider's DECLARED earliest-available date (never inferred)."""
    getter = getattr(provider, "earliest_available", None)
    if getter is None:
        raise ValueError(
            f"{factor.name} is a valid-day-pooled factor, so its saturation load "
            f"needs a RELIABLE data-start signal, but the minute provider "
            f"({type(provider).__name__}) does not implement earliest_available(). "
            f"Inferring the data start from an unchanged row count is unsound — a "
            f"long mid-history no-bar gap (a suspension) looks identical to it. "
            f"Declare the provider's earliest available date."
        )
    return pd.Timestamp(getter(list(symbols))).normalize()


def _pooled_pool_saturated(
    full: pd.Series,
    bars: pd.DataFrame,
    emit_start: pd.Timestamp,
    *,
    baseline_days: int,
    lookback_days: int,
    symbols: list[str],
) -> bool:
    """The structural criterion (see :func:`_materialize_pooled` for the argument).

    True iff, for EVERY REQUESTED symbol, the LOCKED sub-window (the loaded
    trading days after dropping the first ``baseline_days``) holds at least
    ``lookback_days`` valid days at or before ``emit_start`` — valid days being
    the factor's own output dates.

    ITERATING THE REQUESTED SYMBOLS, NOT THE OUTPUT SYMBOLS, IS LOAD-BEARING
    (review HIGH): a symbol that produces ZERO rows in the current load window (a
    long suspension, thin coverage) is absent from the output, so iterating the
    output would give it NO VETO — yet it is exactly the symbol that a deeper load
    would unlock. The consequence was worse than a wrong value: a single-date fill
    dropped such a name from the cross-section entirely while a batch fill kept it
    (measured: volume_peak_count_20 single ``rows=0`` vs batch ``rows=10``), which
    is the silent name-dropping the I5a red line forbids (a missing symbol is a
    sample-coverage bias, and must be loud). Counting an absent symbol as 0 valid
    days vetoes termination, so the loop expands until it too is saturated or the
    declared floor is reached.

    D4b: ``symbols`` is now the ONE symbol whose load depth is being decided (the
    caller streams). The veto is unchanged in meaning — a symbol with no output at
    the current depth still counts zero valid days and still cannot terminate ITS
    OWN expansion — and its blast radius shrinks: it no longer forces the other 994
    symbols to the floor. The failure mode the veto was added for (a name silently
    missing from one fill and present in the other) is additionally foreclosed by
    construction now, because a symbol's own load window always covers the whole
    emit window, so a symbol that emits on an emit date always has bars there.
    The COST D4 recorded (any thin symbol dragging the whole universe to the
    declared floor) is therefore retired; see :func:`_pooled_symbol_stats` for why
    the per-symbol terminal is the same terminal, and note that the
    ``intraday_amp_cut`` coupling D4 flagged is handled by cutting BEFORE its
    cross-sectional combine rather than around the whole factor.
    """
    loaded_days = pd.DatetimeIndex(
        pd.unique(pd.DatetimeIndex(bars.index.get_level_values("time")).normalize())
    ).sort_values()
    if len(loaded_days) <= baseline_days:
        return False
    locked_from = loaded_days[baseline_days]  # first day with a FINAL classification

    out_dates = full.index.get_level_values(DATE_LEVEL)
    out_symbols = pd.Index(full.index.get_level_values(SYMBOL_LEVEL))
    for symbol in symbols:  # the REQUESTED universe, not just what produced rows
        mine = out_dates[out_symbols == str(symbol)]
        final_valid = mine[(mine >= locked_from) & (mine <= emit_start)]
        if len(final_valid) < lookback_days:
            return False
    return True


def _trim_daily(panel: pd.DataFrame, emit_start: pd.Timestamp, warmup: int) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.unique(panel.index.get_level_values(DATE_LEVEL)))
    keep_from = _warmup_start(dates, emit_start, warmup)
    return panel[panel.index.get_level_values(DATE_LEVEL) >= keep_from]


def _trim_minute(bars: pd.DataFrame, emit_start: pd.Timestamp, warmup: int) -> pd.DataFrame:
    bar_dates = pd.DatetimeIndex(bars.index.get_level_values("time")).normalize()
    trading = pd.DatetimeIndex(pd.unique(bar_dates))
    keep_from = _warmup_start(trading, emit_start, warmup)
    return bars[bar_dates >= keep_from]


def _apply_ex_date_mask(factor, raw, view, symbols, load_start, emit_end, sources):
    """NaN out ex-date rows for a ``masked`` factor (§1.3 note 4).

    Non-masked factors (all 14 closing) skip this entirely — the adj_factor
    provider is not even consulted, so no ex-date machinery runs for them.
    """
    if factor.spec.overnight_boundary is not OvernightBoundary.MASKED:
        return raw
    if view is not View.DECISION or raw.empty:
        return raw
    if sources.adj_factor is None:
        raise ValueError(
            f"{factor.name} declares overnight_boundary='masked' but no "
            f"AdjFactorProvider was injected to resolve ex-dates."
        )
    af = sources.adj_factor.adj_factor(list(symbols), load_start, emit_end)
    mask = ex_date_mask(af).reindex(raw.index, fill_value=False)
    out = raw.copy()
    out[mask.to_numpy()] = float("nan")
    return out


def _restrict_emit(raw, emit_start, emit_end, name) -> pd.Series:
    if raw.empty:
        return _empty_series(name)
    dates = raw.index.get_level_values(DATE_LEVEL)
    within = (dates >= emit_start) & (dates <= emit_end)
    return raw[within].sort_index(kind="mergesort").rename(name)


def _restrict_emit_frame(stats: pd.DataFrame, emit_start, emit_end) -> pd.DataFrame:
    """Emit-window slice of a per-symbol intermediate (safe: per-date combine)."""
    dates = stats.index.get_level_values(DATE_LEVEL)
    return stats[(dates >= emit_start) & (dates <= emit_end)]


def _empty_series(name: str) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )
    return pd.Series([], index=index, dtype=float, name=name)


def make_recompute_fn(
    factor: Factor,
    *,
    view: object,
    symbols: list[str],
    sources: MaterializeSources,
    data_start: pd.Timestamp,
    decision_cutoff: str = DEFAULT_DECISION_TIME,
) -> Callable[[pd.Timestamp | None, pd.Timestamp, int], pd.Series]:
    """A ``RecomputeFn`` for the D3 tail-recompute engine (design §3.3).

    ``recompute(emit_start, end, warmup)`` -> the factor's raw Series over
    ``[emit_start, end]`` having loaded ``warmup`` trailing trading days of input
    (``emit_start is None`` -> from ``data_start``). The same trailing-trim floor
    as ``materialize_range`` keeps the incremental overlap bit-identical to a full
    column (the D3 batch=incremental invariant).
    """

    def _recompute(emit_start, end, warmup) -> pd.Series:
        start = data_start if emit_start is None else emit_start
        return materialize_range(
            factor,
            view=view,
            symbols=list(symbols),
            emit_start=start,
            emit_end=end,
            sources=sources,
            decision_cutoff=decision_cutoff,
            warmup=warmup,
        )

    return _recompute


__all__ = [
    "AdjFactorProvider",
    "DailyPanelProvider",
    "MaterializeSources",
    "MinuteBarProvider",
    "build_horizon_config",
    "is_minute_factor",
    "make_recompute_fn",
    "materialize_intermediate_range",
    "materialize_range",
    "payload_columns",
    "stores_intermediate",
]
