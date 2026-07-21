"""Evaluate a minute-derived factor a SECOND time, on the ``exec_to_exec`` basis.

The eleven minute-factor runners each already evaluate their factor twice
(no-book / with-book) against ``close_to_close`` daily returns. This module adds
the execution-anchored pair: the SAME factor values, the SAME evaluator, the SAME
two book settings — only the forward return changes, from
``close(t+h)/close(t)`` to the 14:51 VWAP anchor described in
:mod:`qt.exec_forward_returns`.

Everything is deliberately additive:

  * the existing ``{stem}_no_book`` / ``{stem}_with_book`` artifacts are the
    close-to-close control and are NOT touched — the new reports are written as
    ``{stem}_exec_no_book`` / ``{stem}_exec_with_book``;
  * the factor is computed ONCE by the runner and evaluated four times;
  * the frozen evaluator is used as-is. The exec context supplies
    ``forward_returns`` and deliberately omits ``price_panel``, so if the returns
    were ever absent the evaluator's guard raises instead of quietly measuring a
    close-to-close return under an ``exec_to_exec`` label.

DISCLOSURE. The execution parameters, the coverage loss by cause, the measured
``stk_mins`` live-call count and the sanity-check headline travel INSIDE each exec
report via ``EvalContext.execution_capacity`` — the contract's documented
pass-through for "execution facts measured elsewhere". The two keys that axis
reads for a verdict (``tradable`` / ``capacity_sufficient``) are deliberately NOT
supplied: they are not measured here, so the Tradable axis stays NOT_ASSESSED,
exactly as it is on the close-to-close runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics.eval import (
    EvalConfig,
    EvalContext,
    FactorEvalReport,
    StandardFactorEvaluator,
)
from analytics.eval.figures import render_factor_dashboard
from analytics.factor import forward_returns as _close_forward_returns
from data.clean.schema import DATE_LEVEL
from factors.spec import FactorSpec
from qt.config import RootConfig
from qt.exec_basis_sanity import ExecBasisSanity, check_exec_basis, render_sanity_report
from qt.exec_forward_returns import (
    ExecBasisParams,
    build_exec_price_panel,
    coverage_loss,
    exec_forward_returns,
    intraday_spec_variant,
)


@dataclass(frozen=True)
class ExecBasisEvaluation:
    """The exec-basis half of a factor's evaluation (immutable)."""

    spec: FactorSpec
    params: ExecBasisParams
    artifact_path: Path
    artifact_key: str
    artifact_reused: bool
    minute_live_calls: int
    coverage: dict[str, object]
    sanity: ExecBasisSanity
    sanity_report_path: Path
    no_book: FactorEvalReport
    with_book: FactorEvalReport
    no_book_md: Path
    no_book_json: Path
    with_book_md: Path
    with_book_json: Path
    no_book_dashboard: Path
    with_book_dashboard: Path
    no_book_metrics: dict
    with_book_metrics: dict
    elapsed: float


def _flatten_coverage(coverage: dict[str, object]) -> dict[str, object]:
    """Flatten the coverage dict so every number renders as its own report row."""
    out: dict[str, object] = {}
    for key, value in coverage.items():
        if isinstance(value, dict):
            for cause, count in value.items():
                out[f"coverage_{key}_{cause}"] = count
        else:
            out[f"coverage_{key}"] = value
    return out


def build_disclosure(
    params: ExecBasisParams,
    coverage: dict[str, object],
    sanity: ExecBasisSanity,
    *,
    horizon: int,
    artifact_path: Path,
    artifact_key: str,
    artifact_reused: bool,
    minute_live_calls: int,
    sanity_report_path: Path,
) -> dict[str, object]:
    """The facts every exec report must carry (task card §2.4).

    Deliberately free of ``tradable`` / ``capacity_sufficient``: this run measures
    a RETURN BASIS, not fill feasibility or capacity, and a verdict axis must not
    move because a disclosure was added.
    """
    disclosure: dict[str, object] = {
        "return_basis": "exec_to_exec",
        "forward_return_horizon_periods": horizon,
        "decision_cutoff": params.decision_cutoff,
        "data_lag": params.data_lag,
        "session_open": params.session_open,
        "execution_model": params.execution_model,
        "execution_window": (
            f"[{params.execution_window[0]},{params.execution_window[1]}]"
        ),
        "execution_price_basis": params.execution_price_basis,
        "execution_price_definition": "selected 1min bar amount / volume (RAW)",
        "price_adjustment_applied": (
            "(raw_exec*adj_factor)_exit / (raw_exec*adj_factor)_entry - 1"
        ),
        "execution_parameter_source": params.source,
        "stk_mins_live_calls": minute_live_calls,
        "exec_price_artifact": str(artifact_path),
        "exec_price_artifact_key": artifact_key,
        "exec_price_artifact_reused": artifact_reused,
        "sanity_report": str(sanity_report_path),
        "missing_policy": (
            "no bar / undefined VWAP / unusable adj_factor -> NaN, counted by cause; "
            "never a bar-close, daily-close or adj_factor=1.0 fallback"
        ),
    }
    disclosure.update(_flatten_coverage(coverage))
    disclosure.update(sanity.headline())
    return disclosure


def _write_report(report: FactorEvalReport, report_dir: Path, stem: str) -> tuple[Path, Path]:
    md_path = report_dir / f"{stem}.md"
    json_path = report_dir / f"{stem}.json"
    md_path.write_text(report.render(), encoding="utf-8")
    json_path.write_text(report.to_json(), encoding="utf-8")
    return md_path, json_path


def _extract_metrics(report: FactorEvalReport) -> dict:
    """Headline verdict + gated metrics, mirroring the runners' ``extract_metrics``."""
    verdict = report.require_verdict()
    sections = report.by_name()

    def payload(name: str) -> dict:
        return dict(getattr(sections.get(name), "payload", {}) or {})

    pred = payload("predictive_power")
    coverage = payload("data_coverage")
    incr = payload("purity")
    return {
        "deployment": verdict.verdict,
        "predictive": verdict.predictive.verdict,
        "incremental": verdict.incremental.verdict,
        "tradable": verdict.tradable.verdict,
        "ic_mean": pred.get("ic_mean"),
        "ic_ir": pred.get("ic_ir"),
        "ic_win_rate": pred.get("ic_win_rate"),
        "ic_nw_t": pred.get("ic_nw_t"),
        "settled_rebalances": coverage.get("settled_rebalances"),
        "effective_samples": coverage.get("effective_samples"),
        "span_days": coverage.get("span_days"),
        "incremental_ic_ir": incr.get("incremental_ic_ir"),
        "incremental_ic_mean": incr.get("incremental_ic_mean"),
    }


def run_exec_basis_evaluation(
    factor_panel: pd.Series | pd.DataFrame,
    spec: FactorSpec,
    eval_cfg: EvalConfig,
    book: pd.DataFrame,
    *,
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    logger,
    report_dir: Path,
    stem: str,
    force_rebuild: bool = False,
) -> ExecBasisEvaluation:
    """Build the exec-to-exec returns, sanity-check them, evaluate twice, report.

    ``panel`` is the front-adjusted daily panel (it carries the raw ``adj_factor``
    the adjustment identity needs and defines the shared price grid); ``spec`` is
    the factor's DAILY spec, from which the intraday variant is derived.
    """
    started = time.monotonic()
    params = ExecBasisParams.from_config(cfg)
    exec_spec = intraday_spec_variant(spec, params)
    horizon = spec.forward_return_horizon

    prices = build_exec_price_panel(
        cfg, panel, symbols, params, logger, force_rebuild=force_rebuild
    )

    factor = (
        factor_panel
        if isinstance(factor_panel, pd.Series)
        else factor_panel[spec.factor_id]
    )
    dates = pd.Index(
        pd.unique(factor.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    ).sort_values()

    exec_returns = exec_forward_returns(prices.adjusted_price, dates, horizon)
    # The control series, derived by the SAME boundary the close_to_close run uses,
    # restricted to the same grid first — so the comparison below is like-for-like.
    on_grid = panel[panel.index.get_level_values(DATE_LEVEL).isin(dates)]
    close_returns = _close_forward_returns(on_grid, periods=(horizon,))[
        f"forward_return_{horizon}d"
    ]
    coverage = coverage_loss(
        exec_returns,
        close_returns,
        prices.frame["status"],
        horizon,
    )
    logger.info(
        "exec basis coverage: %d/%d pairs measurable vs close_to_close "
        "(lost %d = %.2f%%; no_bar=%d bad_vwap=%d bad_adj_factor=%d; %d symbols)",
        coverage["exec_to_exec_measurable_pairs"],
        coverage["close_to_close_measurable_pairs"],
        coverage["lost_pairs"],
        coverage["lost_pairs_pct_of_close_to_close"],
        coverage["lost_pairs_by_cause"]["no_bar"],
        coverage["lost_pairs_by_cause"]["bad_vwap"],
        coverage["lost_pairs_by_cause"]["bad_adj_factor"],
        coverage["distinct_symbols_affected"],
    )

    sanity = check_exec_basis(
        prices.frame,
        exec_returns,
        close_returns,
        params,
        cfg.data.cache.root_dir,
        horizon,
    )
    logger.info(
        "exec basis sanity: corr median=%.4f (p10=%.4f p90=%.4f over %d dates), "
        "exec std=%.4f vs close std=%.4f, hand-check max diff=%.3e",
        sanity.corr_median, sanity.corr_p10, sanity.corr_p90, sanity.corr_dates,
        sanity.exec_stats["std"], sanity.close_stats["std"],
        sanity.hand_check_max_abs_diff,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    sanity_path = report_dir / f"{stem}_exec_basis_sanity.md"
    sanity_path.write_text(
        render_sanity_report(
            sanity,
            params,
            coverage,
            key=prices.key,
            artifact_path=str(prices.path),
            horizon=horizon,
            minute_live_calls=prices.minute_live_calls,
        ),
        encoding="utf-8",
    )

    disclosure = build_disclosure(
        params,
        coverage,
        sanity,
        horizon=horizon,
        artifact_path=prices.path,
        artifact_key=prices.key,
        artifact_reused=prices.reused,
        minute_live_calls=prices.minute_live_calls,
        sanity_report_path=sanity_path,
    )

    evaluator = StandardFactorEvaluator()
    # NOTE: no price_panel. The exec basis is not derivable from a close panel, and
    # withholding it means an absent forward_returns raises instead of silently
    # falling back to close_to_close under an exec_to_exec label.
    ctx_no_book = EvalContext(
        forward_returns=exec_returns,
        universe_symbols=tuple(symbols),
        fee_rate=float(cfg.cost.fee_rate),
        execution_capacity=disclosure,
    )
    report_no_book, ir_no_book = evaluator.evaluate_with_ir(
        factor_panel, exec_spec, eval_cfg, ctx_no_book
    )
    ctx_with_book = EvalContext(
        forward_returns=exec_returns,
        universe_symbols=tuple(symbols),
        fee_rate=float(cfg.cost.fee_rate),
        known_factors=book,
        execution_capacity=disclosure,
    )
    report_with_book, ir_with_book = evaluator.evaluate_with_ir(
        factor_panel, exec_spec, eval_cfg, ctx_with_book
    )

    nb_md, nb_json = _write_report(report_no_book, report_dir, f"{stem}_exec_no_book")
    wb_md, wb_json = _write_report(
        report_with_book, report_dir, f"{stem}_exec_with_book"
    )
    nb_png = render_factor_dashboard(
        report_no_book, ir_no_book, report_dir / f"{stem}_exec_no_book_dashboard.png"
    )
    wb_png = render_factor_dashboard(
        report_with_book, ir_with_book, report_dir / f"{stem}_exec_with_book_dashboard.png"
    )
    return ExecBasisEvaluation(
        spec=exec_spec,
        params=params,
        artifact_path=prices.path,
        artifact_key=prices.key,
        artifact_reused=prices.reused,
        minute_live_calls=prices.minute_live_calls,
        coverage=coverage,
        sanity=sanity,
        sanity_report_path=sanity_path,
        no_book=report_no_book,
        with_book=report_with_book,
        no_book_md=nb_md,
        no_book_json=nb_json,
        with_book_md=wb_md,
        with_book_json=wb_json,
        no_book_dashboard=nb_png,
        with_book_dashboard=wb_png,
        no_book_metrics=_extract_metrics(report_no_book),
        with_book_metrics=_extract_metrics(report_with_book),
        elapsed=time.monotonic() - started,
    )


def _fmt(value: object, spec: str = ".4f") -> str:
    """Format a metric a Skipped section may legitimately leave absent.

    The print happens AFTER the reports are written and outside the command's
    error handling, so a None must render as "n/a" rather than raise a TypeError
    that would bury four finished reports under a traceback.
    """
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def format_exec_basis_line(result: ExecBasisEvaluation) -> str:
    """The CLI's exec-basis summary — same shape as the close_to_close lines."""
    nb, wb = result.no_book_metrics, result.with_book_metrics
    cov = result.coverage
    causes = cov["lost_pairs_by_cause"]
    return (
        f"exec-to-exec basis (14:51 {result.params.execution_price_basis}, adjusted; "
        f"stk_mins_live_calls={result.minute_live_calls}):\n"
        f"  coverage loss vs close_to_close: {cov['lost_pairs']} pairs "
        f"({_fmt(cov['lost_pairs_pct_of_close_to_close'], '.2f')}%) — "
        f"no_bar={causes['no_bar']} bad_vwap={causes['bad_vwap']} "
        f"bad_adj_factor={causes['bad_adj_factor']}, "
        f"{cov['distinct_symbols_affected']} symbols; "
        f"corr vs close median={_fmt(result.sanity.corr_median)}\n"
        f"  exec no-book: {nb['deployment']} (predictive={nb['predictive']}) "
        f"ic_mean={_fmt(nb['ic_mean'])} ic_ir={_fmt(nb['ic_ir'], '.3f')}\n"
        f"  exec with-book: {wb['deployment']} (incremental={wb['incremental']}) "
        f"incr_ic_ir={_fmt(wb['incremental_ic_ir'], '.3f')}\n"
        f"  exec reports: {result.no_book_md} | {result.with_book_md}\n"
        f"  sanity: {result.sanity_report_path}"
    )


__all__ = [
    "ExecBasisEvaluation",
    "build_disclosure",
    "format_exec_basis_line",
    "run_exec_basis_evaluation",
]
