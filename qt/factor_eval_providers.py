"""Real-cache provider wiring for the unified factor-eval runner (D5 C4, commit 1).

The materializer (``factors.materialize``) is factor-agnostic and takes its data
access INJECTED (layering red line #3/#10: ``factors`` never touches a feed, a
token, or qt). This module is the qt side of that injection for the evaluation
plane: it wires the persistent tushare caches and the pipeline's daily panel
into the provider protocols, and assembles the service bundle the unified
exec-only ``FactorEvalRunner`` (the next C4 commit) consumes.

* :class:`CacheMinuteProvider` — cache-only 1min bars, MOVED here from
  ``qt.factor_hotpath_smoke`` (single source; the smoke and the probes import
  it from here now). Zero live calls: ``IntradayParquetStore.read_range`` has
  no fetch closure, so a missing month is an empty read, never a warm.
* :class:`DailyEvalPanelProvider` — serves the pipeline's loaded daily panel
  through the ``DailyPanelProvider`` protocol.
* :func:`build_eval_service` / :class:`EvalServiceBundle` — the one wiring of
  cache -> universe -> panel -> enrichments -> store + sources, in the SAME
  call order the legacy eval runners used (``qt/eval_jump_amount_corr.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import READ_COLUMNS
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import (
    RAW_INTRADAY_FREQ,
    empty_intraday_bars,
    normalize_intraday_bars,
)
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.materialize import MaterializeSources
from factors.store import FactorValueStore
from qt.pipeline import (
    _build_cache,
    _build_universe,
    _load_panel,
    _log_run_cache_stats,
    _maybe_enrich_covariates,
    _maybe_enrich_value,
)

if TYPE_CHECKING:
    from qt.config import RootConfig

#: The intraday cache's DECLARED earliest bar date (measured on the real cache:
#: several CSI500 names carry 1min bars from 2015-01-05). The pooled saturation
#: loop needs a declared floor; it must never infer one from row counts.
CACHE_MINUTE_DATA_START = "2015-01-05"

#: The factor value store's runtime artifact root (design §3.4 R22: runtime
#: artifacts live under artifacts/, gitignored, never in the source tree).
DEFAULT_STORE_ROOT = "artifacts/factor_store"


class CacheMinuteProvider:
    """Cache-only MinuteBarProvider: per-symbol read + normalize, zero live calls."""

    def __init__(self, root: str) -> None:
        self._store = IntradayParquetStore(root)
        self.calls = 0
        self.live_calls = 0  # provably 0 — read_range has no fetch closure

    def earliest_available(self, symbols):
        """The cache's DECLARED minute-data floor (measured: bars from 2015-01-05).

        Declared, never inferred from row counts — a long mid-history no-bar gap
        (a suspension) is indistinguishable from exhaustion by row count, which
        is exactly the unsound signal the pooled saturation loop refuses.
        """
        return pd.Timestamp(CACHE_MINUTE_DATA_START)

    def minute_bars(self, symbols, start, end):
        self.calls += 1
        if not symbols:
            return empty_intraday_bars()
        parts = []
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for sym in symbols:
            part = self._store.read_range(INTRADAY_ENDPOINT, sym, RAW_INTRADAY_FREQ, s, e)
            if part.empty:
                continue
            parts.append(
                normalize_intraday_bars(
                    part.rename(columns={"bar_end": "time"})[READ_COLUMNS], freq=RAW_INTRADAY_FREQ
                )
            )
        if not parts:
            return empty_intraday_bars()
        return pd.concat(parts).sort_index(kind="mergesort")


class DailyEvalPanelProvider:
    """``DailyPanelProvider`` over the pipeline's loaded evaluation panel.

    CLOSE-VIEW, NOT LAGGED — and that is exactly right: the materializer's own
    daily path applies ``factors.view_lag.daily_decision_lag`` for the decision
    view (the prev-day shift with the field-level ``open`` exception, R18), so
    the provider must hand it values dated at their natural close date. A
    pre-lagged panel would be shifted TWICE. ``qt.pipeline._load_panel``'s
    product (raw bars enriched with tradability flags, front-adjusted in
    memory) is precisely such an un-lagged close-view panel, so this provider
    is a thin window/symbol slicer over it.

    WINDOW SEMANTICS: the panel covers the configured ``[data.start, data.end]``
    window (the same window the legacy runners loaded). A materializer load
    request reaching before the panel's first date is served what exists —
    the trailing-trading-day trim then treats the panel's left edge like the
    data start (honest under-warm NaN), which reproduces the legacy runners'
    warmup geometry rather than silently inventing deeper history.
    """

    def __init__(self, panel: pd.DataFrame) -> None:
        self._panel = panel

    def daily_panel(self, symbols, start, end):
        if self._panel.empty:
            return self._panel
        dates = self._panel.index.get_level_values(DATE_LEVEL)
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        out = self._panel[mask]
        keep = [str(s) for s in symbols]
        syms = out.index.get_level_values(SYMBOL_LEVEL)
        return out[syms.isin(keep)]


@dataclass(frozen=True)
class EvalServiceBundle:
    """The wired evaluation-plane service inputs (immutable, no secrets).

    ``store``   — the factor value store (design §3.4 R22 root).
    ``sources`` — the materializer's injected data access (daily + minute).
    ``panel``   — the enriched close-view daily panel (book factors +
                  neutralization covariates ride on it, as in the legacy runners).
    ``symbols`` — the union of historical constituents from ``_build_universe``.
    ``cache``   — the shared read-through cache (None when caching is disabled).

    The universe object itself is deliberately NOT carried yet: this bundle has
    no consumer until the unified runner lands (the next C4 commit), and what
    it needs beyond these five fields is that commit's to add.
    """

    store: FactorValueStore
    sources: MaterializeSources
    panel: pd.DataFrame
    symbols: list[str]
    cache: object


def build_eval_service(
    cfg: RootConfig,
    logger: logging.Logger,
    *,
    value_factors=(),
    store_root: str = DEFAULT_STORE_ROOT,
) -> EvalServiceBundle:
    """Wire the evaluation service bundle from the live config.

    SAME CALL ORDER as the legacy eval runners (``qt/eval_jump_amount_corr.py``):
    shared cache -> PIT universe -> raw panel -> value (pe/pb) enrichment ->
    neutralization covariates -> one cache-stats log line. ``value_factors``
    are the factors needing ``daily_basic`` pe/pb (the book's value_ep /
    value_bp); empty means the enrichment is a no-op passthrough.

    The minute provider reads ``cfg.data.cache.root_dir`` directly: it is
    cache-only by construction (never a live call), so it is wired whether or
    not the read-through cache is enabled.
    """
    cache = _build_cache(cfg)
    _universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    panel = _maybe_enrich_value(cfg, panel, symbols, list(value_factors), logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)
    sources = MaterializeSources(
        daily=DailyEvalPanelProvider(panel),
        minute=CacheMinuteProvider(cfg.data.cache.root_dir),
    )
    return EvalServiceBundle(
        store=FactorValueStore(store_root),
        sources=sources,
        panel=panel,
        symbols=list(symbols),
        cache=cache,
    )


__all__ = [
    "CACHE_MINUTE_DATA_START",
    "DEFAULT_STORE_ROOT",
    "CacheMinuteProvider",
    "DailyEvalPanelProvider",
    "EvalServiceBundle",
    "build_eval_service",
]
