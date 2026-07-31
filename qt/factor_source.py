"""The pipeline's factor-VALUE source: one wiring of the factor service (D6a).

Every daily runner used to get its factor values from ``factor.compute(panel)``
directly. That was the second factor-sourcing path the refactor exists to kill
(design decision 3), so the values now come from ``factors.service`` — the store
read-through onto the ONE materializer engine. This module is the qt side of
that wiring for the daily close-view plane: the store policy, the daily panel
provider, and the served-panel assembly.

VIEW. The daily runners decide at the close of ``d`` and hold from ``d+1``, so
their pairing is ``close`` x ``close_to_close`` — the other of the two legal
pairings (``data.availability_policy.require_legal_pairing`` enforces it at the
service boundary). Under the close view the materializer applies NO availability
lag, which is why routing these runners through the service is a change of PATH
and not of VALUES.

STORE POLICY (the reason it exists is the reason it is enforced structurally).
A stored factor value is addressed by ``(factor_id, params, code, view)`` plus a
data fingerprint that covers the PIT/schema modules — nothing in that identity
says WHICH DATA produced the value. For a real market source that is sound and
deliberate (design §3.4): the same symbol-date bar is the same fact whoever asks
for it. For a SYNTHETIC source it is false, and demonstrably so:
``data.feed.demo_feed`` builds each price path from ``arange(n)`` counted off
``data.start``, so the demo close for one (date, symbol) CHANGES when the
configured window start changes, while the store key cannot see ``data.start``
at all.

    A value whose identity the key cannot capture is, by definition, not
    storable.

So the answer here is not a wider key, it is REFUSAL: a synthetic source gets an
ephemeral per-run store and :func:`factor_store_root` raises rather than hand one
a persistent root. This is not a convention the callers must remember — there is
no code path from a synthetic config to a durable artifact.

What refusing costs: a demo run recomputes its factors every time. That is
exactly what it did before the service existed, so nothing regresses.

What a provenance dimension on the key WOULD additionally buy — telling two REAL
vintages apart when a cache backfill changes the bytes while the ``data.clean``
module hashes do not — is a pre-existing D3 property, not something this step
introduces, and is registered as a structural follow-up rather than done here.

Layering: this module imports the factor layer and the config leaf only; it must
NOT import ``qt.pipeline`` (which imports this), and it carries no feed and no
token.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.availability_policy import ReturnBasis, View
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors import service as factor_service
from factors.base import Factor
from factors.materialize import MaterializeSources
from factors.store import FactorValueStore

#: Sources whose bars are FABRICATED by the process that reads them. Their
#: values are not a function of (symbol, date) alone, so they are not storable
#: under a key that names neither the source nor the window (see module docstring).
SYNTHETIC_SOURCES: frozenset[str] = frozenset({"demo"})

#: Sources whose bars are observed market facts: the same (symbol, date) bar is
#: the same fact for every caller, which is what makes a shared store sound.
PERSISTABLE_SOURCES: frozenset[str] = frozenset({"tushare"})

#: Directory name of the factor value store under a run's ``output.root_dir``.
FACTOR_STORE_DIRNAME = "factor_store"


def known_sources() -> frozenset[str]:
    """Every source this module has classified (synthetic + persistable)."""
    return SYNTHETIC_SOURCES | PERSISTABLE_SOURCES


def config_declared_sources() -> frozenset[str]:
    """The sources ``qt.config`` actually accepts (the drift-guard reference).

    Read off the ``DataCfg.source`` Literal rather than restated, so the pair of
    sets above can be checked against the config surface instead of trusted.
    """
    from qt.config import DataCfg

    return frozenset(typing.get_args(DataCfg.model_fields["source"].annotation))


def is_synthetic_source(source: str) -> bool:
    """True iff ``source`` fabricates its bars; readable error if unclassified.

    An unknown source is refused rather than assumed real: guessing would mean
    silently granting a durable store to data nobody has reasoned about, which
    is precisely the failure this module exists to make impossible (red line #9).
    """
    name = str(source)
    if name in SYNTHETIC_SOURCES:
        return True
    if name in PERSISTABLE_SOURCES:
        return False
    raise ValueError(
        f"data.source {name!r} has no factor-store storability classification. "
        f"Add it to qt.factor_source.SYNTHETIC_SOURCES (fabricated bars -> "
        f"ephemeral store) or PERSISTABLE_SOURCES (observed market facts -> "
        f"shared store); known: {sorted(known_sources())}."
    )


def factor_store_root(cfg) -> str:
    """The PERSISTENT factor-store root for ``cfg`` (raises for a synthetic source).

    Derived from ``output.root_dir`` rather than configured separately: a run
    that redirects its outputs (every test does) must take its factor store with
    it, and one fewer knob is one fewer thing to point at the wrong tree.

    Asking this of a synthetic-source config is a readable error, not a fallback
    — see the module docstring for why such values are not storable at all.
    """
    if is_synthetic_source(cfg.data.source):
        raise ValueError(
            f"data.source={cfg.data.source!r} fabricates its bars, so its factor "
            f"values have no persistent store root: the demo close for a (date, "
            f"symbol) is built from a counter that starts at data.start, so the "
            f"same key would address different values for different windows. Use "
            f"open_factor_value_store(cfg), which gives a synthetic run an "
            f"ephemeral store."
        )
    return str(Path(cfg.output.root_dir) / FACTOR_STORE_DIRNAME)


@contextmanager
def open_factor_value_store(cfg, logger: logging.Logger | None = None):
    """Yield the factor value store for ``cfg``: shared for real, ephemeral for demo.

    The synthetic branch's store lives in a temporary directory that is removed
    on exit, so a demo run leaves no artifact any later run could be served. The
    real branch uses :func:`factor_store_root`. Either way the ONE materializer
    engine behind the service is the same — what differs is only whether its
    output outlives the run.
    """
    synthetic = is_synthetic_source(cfg.data.source)
    if synthetic:
        with TemporaryDirectory(prefix="qt-factor-store-ephemeral-") as tmp:
            if logger is not None:
                logger.info(
                    "factor store: EPHEMERAL (data.source=%s fabricates its bars; "
                    "its values are not addressable by the store key, so they are "
                    "never persisted). Factors are recomputed every run.",
                    cfg.data.source,
                )
            yield FactorValueStore(tmp)
        return
    root = factor_store_root(cfg)
    if logger is not None:
        logger.info("factor store: shared read-through at %s", root)
    yield FactorValueStore(root)


class DailyEvalPanelProvider:
    """``DailyPanelProvider`` over an already-loaded daily panel.

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


def decision_points(panel: pd.DataFrame) -> list[factor_service.DecisionPoint]:
    """One DecisionPoint per trading day present in ``panel``.

    The within-day cutoff is the ``DecisionPoint`` default; it selects minute
    bars and is inert on this close-view daily plane, where no factor reads a
    minute endpoint. (``qt.factor_eval_runner._decisions`` is the same reduction
    on the evaluation plane; converging them means touching the eval path, which
    this step deliberately leaves frozen — D6c.)
    """
    dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(d).normalize() for d in panel.index.get_level_values(DATE_LEVEL)})
    )
    return [factor_service.DecisionPoint(d) for d in dates]


@dataclass(frozen=True)
class ServedFactorValues:
    """One service read, reduced to the panel's own grid (+ what that dropped)."""

    frame: pd.DataFrame
    served_rows: int
    footprint_rows_dropped: int


def factor_values(
    factors: Sequence[Factor],
    panel: pd.DataFrame,
    symbols: Iterable[str],
    *,
    store: FactorValueStore,
    params_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> ServedFactorValues:
    """Raw values for ``factors`` over ``panel``'s grid, from the factor service.

    Close view x close-to-close basis (the legal pairing for a runner that
    decides at the close of ``d`` and holds from ``d+1``).

    WHY THE RESULT IS REINDEXED TO ``panel.index``. The service answers a
    (date, symbol) question and therefore returns a row for every cell it was
    ASKED about, carrying an explicit NaN where the factor has no value (D4c's
    fill footprint, which is what lets a warm store tell computed-empty from
    never-computed). An index universe's symbol list is the union of historical
    constituents, so that grid is strictly larger than the loaded panel.

    HOW MUCH larger, and where that number comes from: 2.2% for CSI300 and 4.0%
    for CSI500. Those are SCOUTING-PHASE measurements, not products of this
    change and not reproducible from anything it emits — they were read off two
    local (gitignored) daily panels, ``artifacts/data/i5e_csi300_daily.parquet``
    (569,736 rows over 1,210 dates x 481 symbols) and
    ``artifacts/data/d1_panel_freeze_daily.parquet`` (1,158,912 over 1,210 x
    996), as ``dates * symbols / rows - 1``. They size the effect; they are not
    evidence about this code. The phase2 panel this step actually reconciles on
    is exactly dense (68 x 241 = 16,388), so its footprint count is zero — the
    non-zero case is covered by ``tests/test_factor_source.py`` instead.

    Every one of those extra rows is all-NaN, so no VALUE changes either way — but a consumer
    that counts rows (``per_factor[...]["coverage"]`` is ``notna().mean()``) is
    looking at a different denominator, and a published diagnostic must not move
    because the values arrived by a different route (red line #8). The count of
    rows this drops is returned so the reduction is disclosed rather than
    invisible.

    A cell present in the panel but ABSENT from the service's answer is a loud
    error: it would mean the request and the panel disagreed about the grid, and
    a silently short cross-section is the sample-coverage bias the I5a red line
    forbids.
    """
    factor_ids = [f.name for f in factors]
    served = factor_service.panel(
        factor_ids,
        symbols,
        decision_points(panel),
        store=store,
        sources=MaterializeSources(daily=DailyEvalPanelProvider(panel)),
        view=View.CLOSE,
        basis=ReturnBasis.CLOSE_TO_CLOSE,
        params_by_id=dict(params_by_id or {}),
    )
    unserved = panel.index.difference(served.index)
    if len(unserved):
        raise ValueError(
            f"the factor service returned no row for {len(unserved)} (date, symbol) "
            f"cell(s) the market panel carries, e.g. {list(unserved[:3])}. The "
            f"request grid and the panel must agree; serving the panel's factor "
            f"values short would drop names from a cross-section silently."
        )
    return ServedFactorValues(
        frame=served.reindex(panel.index),
        served_rows=int(len(served)),
        footprint_rows_dropped=int(len(served) - len(panel.index)),
    )


__all__ = [
    "FACTOR_STORE_DIRNAME",
    "PERSISTABLE_SOURCES",
    "SYNTHETIC_SOURCES",
    "DailyEvalPanelProvider",
    "ServedFactorValues",
    "config_declared_sources",
    "decision_points",
    "factor_store_root",
    "factor_values",
    "is_synthetic_source",
    "known_sources",
    "open_factor_value_store",
]
