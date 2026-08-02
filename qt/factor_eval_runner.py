"""run-factor-eval: the UNIFIED exec-only factor evaluation runner (D5 C4, commit 2).

ONE runner body for the eleven minute-derived factors, replacing the eleven
near-identical ``qt/eval_*.py`` main flows (the difference catalogue
``docs/factors/d5_runner_difference_catalogue.md`` §四 lists what was already
identical; §二/§三 list the four extension points this runner parameterizes):

1. factor construction + params — the registry's pre-registered defaults
   (``factors.registry.build``); the legacy runners passed the same values
   explicitly.
2. the post-loop cross-sectional finalizer — the D4c read-assembly combine,
   which ``factors.service`` runs over exactly the requested universe
   (``intraday_amp_cut``'s z-score happens there, not in a runner).
3. the add-Section coverage disclosures — ``qt.factor_eval_disclosures``
   (mechanism A diagnostics sink through ``factors.service.panel``'s
   ``diagnostics=``; the sink re-materializes the request so a warm store can
   still serve the disclosure — see ``factors.service._ensure_coverage``;
   mechanism B — valley_price_quantile's NeutralizationCoverage — is reduced
   from the stored intermediate + the reversal + the served residual AFTER the
   panel read, see ``_summarize_subject_neutralization``).
4. metric keys — ``qt.exec_basis_eval``'s extraction.

The subject factor's VALUES come from the factor SERVICE
(``factors.service.panel``, decision view x exec_to_exec basis), not from a
per-runner minute loader: the store read-through + the per-symbol streaming
materializer are the single engine (D4/D4b/D4c), which is what retires the
second factor-sourcing path (design decision 3). The evaluation itself reuses
``qt.exec_basis_eval.run_exec_basis_evaluation`` verbatim — the SAME evaluator,
the SAME two book settings, the SAME 14:51-VWAP execution anchor.

EXEC-ONLY: this runner never produces close_to_close artifacts. Its EvalConfig
declares ``view="decision", return_basis="exec_to_exec"`` EXPLICITLY — the
EvalConfig default is still the legacy close pairing, and an exec-only runner
must not inherit it.

BOOK MODES (``--book-mode``):

* ``close``   — faithfully replicates the legacy book path: direct
  ``Factor.compute`` on the close-view enriched panel + one
  ``_process_factors`` (template ``qt/eval_jump_amount_corr.py``). This is the
  live defect design §1.1 records (close(d) book values against an exec
  holding window opening at 14:51(d)); the artifact says ``book_view=close``.
* ``decision`` (default) — the book comes through the SAME service panel as
  the subject, so the materializer's ``daily_decision_lag`` applies to it;
  the artifact says ``book_view=decision``.

The book is the frozen confirmed trio value_ep / value_bp / volatility_20
(lead ruling Q1). Catalogue BUG 5 closure: a config ``factors:`` block that
disagrees with that effective book is a READABLE ERROR, never silently
ignored; the honest declaration is ``factors: []``.

ARTIFACT ISOLATION: the report stem is ``factor_eval_{factor_id}`` (vs the
legacy ``eval_{name}``), so new and legacy artifacts coexist and can never
overwrite each other (the compare_postmerge lesson). In ``close`` book mode
the with-book artifacts gain a ``_bookclose`` suffix for the same reason.

RUN REGISTRY (D7-PR0): each run is appended to the run registry
(``factors.store.run_registry``) under the factor store root — view=decision,
basis=exec_to_exec, status from the explicit mapping in
``qt.factor_eval_registry`` (book member -> ``book``; Watch -> ``watch``;
Reject / INSUFFICIENT-DATA -> ``exploratory``; Adopt is unreachable and
refused). At startup the runner seeds-or-asserts the book
(:func:`qt.factor_eval_registry.sync_book_registry`): an empty registry is
seeded from ``BOOK_IDS``; a non-empty one must agree with it exactly, or the
run fails readably before any evaluation work.

``valley_price_quantile`` IS served (PR-C4b): its per-symbol ``raw_qbar``
intermediate rides the same store read-through as every other factor, and its
cross-sectional reversal neutralization runs at the service's read-assembly
over exactly the requested universe (the binding's DECLARED daily combine
input). Its NeutralizationCoverage disclosure is catalogue §三 mechanism B —
no diagnostics sink — so the runner reduces it from the STORED intermediate +
the reversal + the served residual AFTER the panel read
(:func:`_summarize_subject_neutralization`), never from a second engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics.eval import EvalConfig
from data.availability_policy import ReturnBasis, View
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors import registry as factor_registry
from factors import service as factor_service
from factors.compute.minute.binding import (
    RAW_QBAR_COL,
    has_minute_diagnostics,
    is_minute_bound,
)
from factors.compute.minute.valley_price_quantile import (
    VALLEY_QUANTILE_MIN_CROSS_SECTION,
    VALLEY_QUANTILE_REVERSAL_DAYS,
    reversal_20,
)
from factors.spec import FactorSpec
from factors.store import RunRegistry, data_fingerprint, store_key
from qt.config import RootConfig, load_config
from qt.exec_basis_eval import ExecBasisEvaluation, run_exec_basis_evaluation
from qt.factor_eval_disclosures import (
    NEUTRALIZATION_SECTION_NAME,
    disclosure_binding_for,
    publishes_neutralization_disclosure,
    summarize_neutralization,
    to_section,
)
from qt.factor_eval_providers import EvalServiceBundle, build_eval_service
from qt.factor_eval_registry import status_for_run, sync_book_registry
from qt.pipeline import _make_logger, _process_factors

_LOGGER_NAME = "qt.factor_eval_runner"

#: The frozen confirmed book (lead ruling Q1) — the same trio the eleven legacy
#: runners hardcoded in ``_build_book_factors`` (11 AST-identical bodies).
BOOK_IDS: tuple[str, ...] = ("value_ep", "value_bp", "volatility_20")

#: What the effective book IS, for the catalogue BUG 5 consistency check: a
#: config ``factors:`` block must either be empty (``factors: []``) or declare
#: exactly this — anything else would sit in the file looking live while the
#: runner never reads it.
_EFFECTIVE_BOOK: dict[str, dict[str, object]] = {
    "value_ep": {},
    "value_bp": {},
    "volatility_20": {"window": 20, "price_col": "close"},
}

BOOK_MODES: tuple[str, ...] = (View.DECISION.value, View.CLOSE.value)


def _build_book_factors() -> list:
    """Instantiate the confirmed book (value_ep / value_bp / volatility_20)."""
    from factors.compute.candidates import ValueFactor, VolatilityFactor

    return [
        ValueFactor("value_ep"),
        ValueFactor("value_bp"),
        VolatilityFactor(window=20),
    ]


# --------------------------------------------------------------------------- #
# Config gates (the eleven identical bodies, collapsed once — catalogue C1/C2)
# --------------------------------------------------------------------------- #
def _check_preconditions(cfg: RootConfig) -> None:
    """Fail readably if the config cannot drive a real, cache-only CSI500 eval."""
    if cfg.data.source != "tushare":
        raise ValueError(
            "run-factor-eval needs data.source='tushare' (real cached A-share "
            f"data); got {cfg.data.source!r}."
        )
    if not cfg.data.cache.enabled:
        raise ValueError(
            "run-factor-eval needs data.cache.enabled=true (it reads the "
            "persistent tushare cache and never warms live)."
        )
    if cfg.universe.type != "index":
        raise ValueError(
            "run-factor-eval needs universe.type='index' (PIT membership, e.g. "
            f"000905.SH for CSI500); got {cfg.universe.type!r}."
        )
    if not cfg.processing.neutralize.enabled:
        raise ValueError(
            "run-factor-eval expects processing.neutralize.enabled=true "
            "(industry + size neutralization, matching the report's neutral "
            "column and the EvalConfig declaration)."
        )


def _check_config_book(cfg: RootConfig) -> None:
    """Catalogue BUG 5 closure: the config may not declare a book the run ignores.

    The legacy configs carry a ``factors:`` block naming the book, but no runner
    ever read ``cfg.factors`` — harmless only while the two agree. Here the
    subject comes from ``--factor`` and the book is fixed, so a non-empty block
    must match the effective book EXACTLY; a mismatch is a readable error, and
    the honest declaration is ``factors: []``.
    """
    declared = {f.name: dict(f.params) for f in cfg.factors if f.enabled}
    if not declared:
        return
    if declared != _EFFECTIVE_BOOK:
        raise ValueError(
            "run-factor-eval does not read config 'factors:' — the subject comes "
            "from --factor and the book is fixed at value_ep / value_bp / "
            "volatility_20(window=20, price_col=close). The config declares "
            f"{sorted(declared)}, which disagrees; remove the block (declare "
            "'factors: []') or make it match exactly, so the file cannot "
            "silently describe a factor set the run never used (catalogue BUG 5)."
        )


def _build_eval_config(cfg: RootConfig) -> EvalConfig:
    """The per-run EvalConfig — the catalogue C1 17 shared kwargs + exec identity.

    The 17 kwargs are the ones the eleven legacy ``_build_eval_config`` bodies
    shared character-for-character. This runner is EXEC-ONLY, so the identity
    is declared explicitly: ``view="decision", return_basis="exec_to_exec"``
    (the EvalConfig default is still the legacy close pairing and must not be
    inherited here). ``book_view`` is NOT set on this config — the no-book and
    with-book runs carry DIFFERENT book views, so the identity is derived per
    run by ``qt.exec_basis_eval.exec_identity`` (which also re-derives the
    subject view from the measured cutoff-safety fact).
    """
    if cfg.oos is None:
        raise ValueError(
            "run-factor-eval requires an 'oos' section (split_date) so the "
            "Predictive axis has an out-of-sample split to assess; add e.g. "
            "oos: {split_date: '2024-01-01'}."
        )
    return EvalConfig(
        universe=cfg.universe.index_code or cfg.universe.type,
        universe_is_pit=cfg.universe.type == "index",
        start=cfg.data.start,
        end=cfg.data.end,
        is_exploratory=True,
        post_hoc_selected=False,
        rebalance="daily",
        n_quantiles=int(cfg.analytics.quantiles),
        cost_scenarios=(1.0, 2.0, 4.0),
        oos_split=cfg.oos.split_date,
        # winsorize is a P0 no-op in this codebase -> declare None (nothing clipped).
        winsorize=None,
        standardize="zscore" if cfg.processing.standardize.enabled else None,
        neutralization=("industry", "size") if cfg.processing.neutralize.enabled else (),
        industry_level=cfg.processing.neutralize.industry_level,
        tuned=False,
        # We evaluate ONE pre-registered factor whose sign came from the report
        # (not a screen of our own); the report's own screen is a caveat noted in
        # the run's prose, not our multiple-testing background.
        n_factors_screened=1,
        data_snapshot_id=cfg.data.cache.root_dir,
        view=View.DECISION.value,
        return_basis=ReturnBasis.EXEC_TO_EXEC.value,
    )


# --------------------------------------------------------------------------- #
# The evaluation flow
# --------------------------------------------------------------------------- #
def _decisions(panel: pd.DataFrame) -> list[factor_service.DecisionPoint]:
    """One DecisionPoint per daily-panel trading day (14:50 default cutoff)."""
    dates = pd.Index(
        pd.unique(panel.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    ).sort_values()
    return [factor_service.DecisionPoint(d) for d in dates]


def _load_subject_raw(
    bundle: EvalServiceBundle,
    factor_id: str,
    decisions: list[factor_service.DecisionPoint],
    *,
    diagnostics: list | None,
) -> pd.DataFrame:
    """The subject's raw values from the factor SERVICE (the single engine).

    Decision view x exec_to_exec basis (the pairing is enforced at the service
    boundary). The store read-through fills any (date, symbol) it lacks via the
    ONE materializer; the cross-sectional combine runs at read-assembly over
    exactly this universe (D4c).
    """
    return factor_service.panel(
        [factor_id],
        bundle.symbols,
        decisions,
        store=bundle.store,
        sources=bundle.sources,
        view=View.DECISION,
        basis=ReturnBasis.EXEC_TO_EXEC,
        diagnostics=diagnostics,
    )


def _summarize_subject_neutralization(
    bundle: EvalServiceBundle,
    factor,
    subject_raw: pd.DataFrame,
    decisions: list[factor_service.DecisionPoint],
):
    """valley_price_quantile's NeutralizationCoverage (catalogue §三 mechanism B).

    No diagnostics sink exists for this disclosure: it is reduced AFTER the
    panel read from the three panels the legacy runner also used —

    * raw      — the STORED per-symbol ``raw_qbar`` intermediate, read back
      through :func:`factors.service.stored_payload` (the same store the value
      read just came from, sliced to exactly this request — never a second
      engine, never a recomputation);
    * rev      — ``reversal_20`` on the bundle's close-view daily panel (NO new
      data source). This is the legacy runner's exact computation, and the C4b
      binding pins it bit-for-bit equal to what the service's read-assembly
      combine actually applied (``reversal_20_shifted`` on the decision-lagged
      panel);
    * residual — the served subject values (pre-processing), i.e. the combine's
      output over exactly this universe.
    """
    payload = factor_service.stored_payload(
        factor.name,
        bundle.symbols,
        decisions,
        store=bundle.store,
        sources=bundle.sources,
        view=View.DECISION,
        basis=ReturnBasis.EXEC_TO_EXEC,
    )
    raw = payload[RAW_QBAR_COL]
    rev = reversal_20(bundle.panel[["close"]], days=VALLEY_QUANTILE_REVERSAL_DAYS)
    return summarize_neutralization(
        raw,
        rev,
        subject_raw[factor.name],
        min_cross_section=VALLEY_QUANTILE_MIN_CROSS_SECTION,
    )


def _load_book_raw(
    bundle: EvalServiceBundle,
    decisions: list[factor_service.DecisionPoint],
    *,
    book_mode: str,
) -> pd.DataFrame:
    """The book's raw values, in one of the two declared modes.

    ``close`` replicates the legacy runners exactly (direct compute on the
    close-view enriched panel); ``decision`` takes the book through the same
    service panel as the subject, so the materializer's decision-view lag
    applies (the §1.1 close-view book defect does not exist on this path).
    """
    if book_mode == View.CLOSE.value:
        return pd.concat(
            [f.compute(bundle.panel).rename(f.name) for f in _build_book_factors()],
            axis=1,
        )
    return factor_service.panel(
        list(BOOK_IDS),
        bundle.symbols,
        decisions,
        store=bundle.store,
        sources=bundle.sources,
        view=View.DECISION,
        basis=ReturnBasis.EXEC_TO_EXEC,
    )


def _coverage_split(
    subject_raw: pd.DataFrame, factor_id: str, symbols: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(symbols with >= 1 finite raw value, requested symbols with none)."""
    by_symbol = subject_raw[factor_id].notna().groupby(level=SYMBOL_LEVEL).any()
    covered = {str(s) for s in by_symbol[by_symbol].index}
    empty = sorted(set(map(str, symbols)) - covered)
    return tuple(sorted(covered)), tuple(empty)


#: ``book_mode='close'`` artifact isolation: the with-book artifacts are
#: written straight to ``..._exec_with_book_bookclose.*``. The no-book
#: artifacts carry no book at all, so they keep the shared stem; the sanity
#: report is book-independent for the same reason.
#:
#: D5 C5 F5: this used to be a POST-HOC ``os.replace`` of files first written
#: under the shared ``_exec_with_book`` stem — which silently destroyed the
#: DECISION-book artifacts of any earlier run, because the close run wrote
#: over them before moving them aside. Running the two book modes in the order
#: decision -> close therefore left no decision artifacts at all, and the
#: reconcile's reports leg died on a bare FileNotFoundError. The suffix now
#: travels INTO the write-out layer, so each book mode only ever writes its
#: own file names and the run order stops mattering.
_BOOKCLOSE_SUFFIX = "_bookclose"


def _with_book_suffix(book_mode: str) -> str:
    """The with-book artifact suffix for a book mode (see ``_BOOKCLOSE_SUFFIX``)."""
    return _BOOKCLOSE_SUFFIX if book_mode == View.CLOSE.value else ""


# --------------------------------------------------------------------------- #
# Result container + the full glue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorEvalResult:
    """Immutable summary of one run-factor-eval run."""

    config: RootConfig
    spec: FactorSpec
    factor_id: str
    book_mode: str
    requested_symbols: int
    covered_symbols: int
    empty_symbols: int
    factor_rows: int
    minute_live_calls: int
    coverage: object | None  # a qt.factor_eval_disclosures coverage, when published
    exec_basis: ExecBasisEvaluation
    registry_status: str  # the status this run appended to the run registry (D7-PR0)
    log_path: Path
    elapsed: float


def run_factor_eval(
    config_path: str, factor_id: str, *, book_mode: str = View.DECISION.value
) -> FactorEvalResult:
    """Run the unified exec-only evaluation of ``factor_id`` (cache-only)."""
    if book_mode not in BOOK_MODES:
        raise ValueError(
            f"run-factor-eval --book-mode must be one of {BOOK_MODES}; got "
            f"{book_mode!r}."
        )
    cfg = load_config(config_path)
    _check_preconditions(cfg)
    _check_config_book(cfg)
    factor = factor_registry.build(factor_id)
    spec = factor.spec
    eval_cfg = _build_eval_config(cfg)

    log_path = Path(cfg.output.log_dir) / f"factor_eval_{factor_id}.log"
    logger = _make_logger(log_path, name=_LOGGER_NAME)
    started = time.monotonic()
    logger.info(
        "eval config: %s rebalance=daily oos_split=%s factor=%s book_mode=%s",
        eval_cfg.universe, eval_cfg.oos_split, factor_id, book_mode,
    )

    # The book's value factors need daily_basic pe/pb on the panel in BOTH book
    # modes (in 'decision' mode the materializer's daily provider serves this
    # same enriched panel to ValueFactor.compute).
    bundle = build_eval_service(cfg, logger, value_factors=_build_book_factors())

    # D7-PR0 startup gate: seed-or-assert the run registry's book against the
    # code-declared BOOK_IDS BEFORE any evaluation work (an empty registry is
    # seeded from BOOK_IDS; a non-empty one must already agree with it).
    registry = RunRegistry(bundle.store.root)
    book_state = sync_book_registry(registry, book_ids=BOOK_IDS)
    logger.info("run registry book %s: %s", book_state, sorted(BOOK_IDS))

    decisions = _decisions(bundle.panel)

    # Subject: the service panel. The diagnostics sink rides along only for the
    # factors that publish a day-level gate-attrition disclosure (catalogue §三
    # mechanism A); asking for it re-materializes the request (a warm store can
    # serve values but not the disclosure), which is the disclosed cost.
    binding = disclosure_binding_for(factor)
    if binding is None and is_minute_bound(factor) and has_minute_diagnostics(factor):
        raise ValueError(
            f"{factor_id}: the minute binding publishes per-day diagnostics but "
            "qt.factor_eval_disclosures has no summarizer for them — a disclosure "
            "without a home must be loud, never silently reduced away."
        )
    sink: list | None = [] if binding is not None else None
    subject_raw = _load_subject_raw(bundle, factor_id, decisions, diagnostics=sink)
    covered, empty = _coverage_split(subject_raw, factor_id, bundle.symbols)

    coverage = None
    extra_sections = []
    if binding is not None:
        coverage = binding.summarize(sink)
        logger.info("%s", coverage.render())
        extra_sections.append(to_section(binding.section_name, coverage))
    elif publishes_neutralization_disclosure(factor):
        # Catalogue §三 mechanism B: no sink — reduce the disclosure from the
        # stored intermediate + the reversal + the served residual, AFTER the
        # panel read. The cost is the store re-read of a payload already
        # materialized by the subject read (a warm-store read, zero minute
        # bars); the COLD fill behind that payload is vpq's real cost — it is
        # pooled + cross-sectional + daily-bound, so a cold store means the
        # saturation load (the declared 2015-01-05 floor), disclosed here and
        # visible in the run log's elapsed time, same as every other factor's
        # cold fill.
        coverage = _summarize_subject_neutralization(
            bundle, factor, subject_raw, decisions
        )
        logger.info("%s", coverage.render())
        extra_sections.append(to_section(NEUTRALIZATION_SECTION_NAME, coverage))

    subject_processed = _process_factors(
        cfg, subject_raw[[spec.factor_id]], bundle.panel
    )
    factor_series = subject_processed[spec.factor_id]
    book_processed = _process_factors(
        cfg, _load_book_raw(bundle, decisions, book_mode=book_mode), bundle.panel
    )

    stem = f"factor_eval_{factor_id}"
    exec_basis = run_exec_basis_evaluation(
        factor_series,
        spec,
        eval_cfg,
        book_processed,
        cfg=cfg,
        panel=bundle.panel,
        symbols=bundle.symbols,
        logger=logger,
        report_dir=Path(cfg.output.report_dir),
        stem=stem,
        book_view=book_mode,
        with_book_suffix=_with_book_suffix(book_mode),
        extra_sections=extra_sections or None,
    )

    logger.info(
        "verdict exec no-book: %s (predictive=%s); with-book: %s (incremental=%s)",
        exec_basis.no_book_metrics["deployment"],
        exec_basis.no_book_metrics["predictive"],
        exec_basis.with_book_metrics["deployment"],
        exec_basis.with_book_metrics["incremental"],
    )

    # D7-PR0 run-registry append: the WITH-BOOK deployment drives the status
    # (it is the three-axis verdict — Predictive + Incremental + Tradable; the
    # no-book run leaves Incremental NOT_ASSESSED). Both deployments ride in the
    # note. The key/fingerprint mirror the subject's decision-view service read
    # (params=None, the registry-default params this runner always uses).
    status = status_for_run(
        factor_id, exec_basis.with_book_metrics["deployment"], book_ids=BOOK_IDS
    )
    record = registry.append_run(
        key=store_key(factor, view=View.DECISION.value, params=None),
        factor=factor,
        status=status,
        fingerprint=data_fingerprint(adjustment=spec.adjustment),
        note=(
            f"run-factor-eval factor={factor_id} book_mode={book_mode} "
            f"universe={eval_cfg.universe} "
            f"window={eval_cfg.start}..{eval_cfg.end} "
            f"oos_split={eval_cfg.oos_split} "
            f"verdict_no_book={exec_basis.no_book_metrics['deployment']} "
            f"verdict_with_book={exec_basis.with_book_metrics['deployment']} "
            f"status={status}"
        ),
    )
    logger.info("run registry: appended factor=%s status=%s", record.factor_id, status)

    return FactorEvalResult(
        config=cfg,
        spec=spec,
        factor_id=factor_id,
        book_mode=book_mode,
        requested_symbols=len(bundle.symbols),
        covered_symbols=len(covered),
        empty_symbols=len(empty),
        factor_rows=int(len(factor_series)),
        minute_live_calls=int(getattr(bundle.sources.minute, "live_calls", 0)),
        coverage=coverage,
        exec_basis=exec_basis,
        registry_status=status,
        log_path=log_path,
        elapsed=time.monotonic() - started,
    )


__all__ = [
    "BOOK_IDS",
    "BOOK_MODES",
    "FactorEvalResult",
    "run_factor_eval",
]
