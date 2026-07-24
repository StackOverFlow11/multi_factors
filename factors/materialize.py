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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN, OvernightBoundary, View
from data.clean.intraday_schema import DEFAULT_DECISION_TIME
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.base import Factor
from factors.compute.minute.binding import (
    is_minute_bound,
    is_valid_day_pooled,
    minute_raw_from_bars,
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


#: One backward-expansion chunk (calendar days) for the valid-day-pooled
#: saturation loop. Generous vs the baseline depth so a single no-change across a
#: chunk implies saturation (the boundary days gained far more than baseline_days
#: of prior history, so their classification is stable). See ``_materialize_pooled``.
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
    """Loads normalized 1min bars (cache-only) for the symbols over the window."""

    def minute_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """MultiIndex(time, symbol) bars (:mod:`data.clean.intraday_schema`)."""


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


# --------------------------------------------------------------------------- #
# Trailing-trading-day trim (the P8 correctness floor)
# --------------------------------------------------------------------------- #
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
) -> pd.Series:
    """Compute ``factor``'s RAW values for ``view`` over ``[emit_start, emit_end]``.

    Loads a generous window (so the exact trailing trim always has history),
    applies the view lag, trims to ``warmup`` (= ``spec.lookback_depth``) trailing
    trading days before ``emit_start``, computes, and returns the emit-window
    rows. Single-date and batch calls give bit-identical values (design §3.5 P8).
    """
    resolved_view = View(view)
    emit_start = pd.Timestamp(emit_start).normalize()
    emit_end = pd.Timestamp(emit_end).normalize()
    w = int(factor.spec.lookback_depth) if warmup is None else int(warmup)
    if w < 1:
        raise ValueError(f"{factor.name}: warmup must be >= 1; got {w}.")
    load_start = emit_start - pd.Timedelta(days=_load_buffer_calendar_days(w))

    if is_minute_factor(factor):
        raw = _materialize_minute(
            factor, resolved_view, symbols, load_start, emit_start, emit_end, w,
            sources, decision_cutoff,
        )
    else:
        raw = _materialize_daily(
            factor, resolved_view, symbols, load_start, emit_start, emit_end, w, sources,
        )

    raw = _apply_ex_date_mask(
        factor, raw, resolved_view, symbols, load_start, emit_end, sources
    )
    return _restrict_emit(raw, emit_start, emit_end, factor.name)


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
    factor, view, symbols, load_start, emit_start, emit_end, warmup, sources, decision_cutoff,
) -> pd.Series:
    if sources.minute is None:
        raise ValueError(
            f"{factor.name} is a minute factor but no MinuteBarProvider was injected."
        )
    if not is_minute_bound(factor):
        # Readable error (e.g. valley_price_quantile needs the daily panel too).
        return minute_raw_from_bars(factor, sources.minute.minute_bars([], load_start, emit_end))
    # Load one extra calendar day past emit_end so bar cutoffs on emit_end resolve.
    # POOLED (valid-day trailing window): the pool counts VALID days, so its
    # CALENDAR depth is data-dependent and UNBOUNDED — a fixed trim would make the
    # value depend on load geometry (review HIGH / red line #6). Load by
    # SATURATION-EXPANSION instead (below); bounded factors keep the fixed trim.
    if is_valid_day_pooled(factor):
        return _materialize_pooled(
            factor, view, list(symbols), emit_start, emit_end, warmup, sources, decision_cutoff,
        )
    bars = sources.minute.minute_bars(
        list(symbols), load_start, emit_end + pd.Timedelta(days=1)
    )
    if bars.empty:
        return _empty_series(factor.name)
    if view is View.DECISION:
        bars = minute_decision_cutoff(bars, decision_time=decision_cutoff)
    bars = _trim_minute(bars, emit_start, warmup)
    return minute_raw_from_bars(factor, bars)


def _materialize_pooled(
    factor, view, symbols, emit_start, emit_end, warmup, sources, decision_cutoff,
) -> pd.Series:
    """Saturation-expanding load for a valid-day-pooled factor (design §3.3, review HIGH).

    Expands the load backward in chunks and recomputes the EMIT-range values until
    they stop changing (saturated: adding more history no longer alters the pool
    composition or the boundary classifications) OR the provider returns no more
    earlier bars (the real data start — the structural terminal). The chunk is
    generous vs ``baseline_days`` so one no-change implies saturation. The returned
    values are load-geometry-free: the finite<->NaN divergence between a single-date
    fill and a batch fill is eliminated (the residual across the two fills is only
    the pandas-accumulation float-reorder, JC1 <= 1e-12, because the two fills may
    terminate at different — but each individually saturated — load starts).
    """
    chunk = pd.Timedelta(days=_saturation_chunk_calendar_days(warmup))
    end = emit_end + pd.Timedelta(days=1)
    load_start = emit_start - chunk
    prev_emit: pd.Series | None = None
    prev_nbars = -1
    for _ in range(_MAX_SATURATION_CHUNKS):
        bars = sources.minute.minute_bars(symbols, load_start, end)
        nbars = len(bars)
        if nbars == 0:
            return _empty_series(factor.name)
        work = bars
        if view is View.DECISION:
            work = minute_decision_cutoff(work, decision_time=decision_cutoff)
        emit = _restrict_emit(
            minute_raw_from_bars(factor, work), emit_start, emit_end, factor.name
        )
        if prev_emit is not None and _pooled_emit_saturated(prev_emit, emit):
            return emit
        if nbars == prev_nbars:  # provider exhausted -> real data start reached
            return emit
        prev_emit, prev_nbars = emit, nbars
        load_start = load_start - chunk
    return prev_emit if prev_emit is not None else _empty_series(factor.name)


def _pooled_emit_saturated(prev: pd.Series, cur: pd.Series, tol: float = 1e-12) -> bool:
    """True iff the emit-range values stopped changing across a chunk expansion.

    Compares on the union index (NaN where absent): the NaN mask must match
    EXACTLY (a finite<->NaN flip is a real pool change, not saturation) and the
    finite values must agree within ``tol`` (a chunk changes the array length, so
    the pandas rolling-sum accumulation reorders finite values at ~1e-15 even when
    the pool composition is stable — that float-reorder is not a pool change).
    """
    index = prev.index.union(cur.index)
    a = prev.reindex(index).to_numpy(dtype=float)
    b = cur.reindex(index).to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    finite = ~np.isnan(a)
    if not finite.any():
        return True
    return bool(np.allclose(a[finite], b[finite], rtol=0.0, atol=tol))


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
    "materialize_range",
]
