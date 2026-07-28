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
   still serve the disclosure — see ``factors.service._ensure_coverage``).
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

RunRegistry appends are a DELIBERATE DEFERRAL to D7's governance surface: this
runner writes its artifacts but does not register runs.

``valley_price_quantile`` is NOT served: its raw compute also needs the DAILY
close panel (reversal neutralization), so it has no minute binding yet — that
lands in PR-C4b. Asking for it is a readable error, never a silent mis-compute.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from analytics.eval import EvalConfig
from data.availability_policy import ReturnBasis, View
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors import registry as factor_registry
from factors import service as factor_service
from factors.compute.minute.binding import has_minute_diagnostics, is_minute_bound
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.spec import FactorSpec
from qt.config import RootConfig, load_config
from qt.exec_basis_eval import ExecBasisEvaluation, run_exec_basis_evaluation
from qt.factor_eval_disclosures import disclosure_binding_for, to_section
from qt.factor_eval_providers import EvalServiceBundle, build_eval_service
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


def _check_subject_supported(factor) -> None:
    """The one subject this runner cannot serve yet (PR-C4b), as a readable error."""
    if isinstance(factor, ValleyPriceQuantileFactor):
        raise ValueError(
            f"run-factor-eval cannot serve {factor.name!r} yet: its raw compute "
            "also needs the DAILY close panel (its reversal neutralization), so "
            "it has no minute binding in factors.compute.minute.binding — the "
            "binding lands in PR-C4b. Asking for it here must fail loudly, never "
            "silently mis-compute."
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


def _apply_bookclose_suffix(
    exec_basis: ExecBasisEvaluation, report_dir: Path, stem: str
) -> ExecBasisEvaluation:
    """``book_mode='close'`` artifact isolation: ``..._exec_with_book_bookclose.*``.

    The no-book artifacts carry no book at all, so they keep the shared stem;
    the sanity report is book-independent for the same reason.
    """
    mapping = {
        "with_book_md": report_dir / f"{stem}_exec_with_book_bookclose.md",
        "with_book_json": report_dir / f"{stem}_exec_with_book_bookclose.json",
        "with_book_dashboard": report_dir
        / f"{stem}_exec_with_book_bookclose_dashboard.png",
    }
    for field_name, dst in mapping.items():
        os.replace(getattr(exec_basis, field_name), dst)
    return replace(exec_basis, **mapping)


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
    _check_subject_supported(factor)
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
        extra_sections=extra_sections or None,
    )
    if book_mode == View.CLOSE.value:
        exec_basis = _apply_bookclose_suffix(
            exec_basis, Path(cfg.output.report_dir), stem
        )

    logger.info(
        "verdict exec no-book: %s (predictive=%s); with-book: %s (incremental=%s)",
        exec_basis.no_book_metrics["deployment"],
        exec_basis.no_book_metrics["predictive"],
        exec_basis.with_book_metrics["deployment"],
        exec_basis.with_book_metrics["incremental"],
    )

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
        log_path=log_path,
        elapsed=time.monotonic() - started,
    )


__all__ = [
    "BOOK_IDS",
    "BOOK_MODES",
    "FactorEvalResult",
    "run_factor_eval",
]
