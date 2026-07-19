"""Command-line entry point for the Phase 0 framework.

Subcommands:
    validate-config      --config PATH  Load + validate a YAML config.
    run-phase0           --config PATH  Run the full end-to-end pipeline + report.
    run-phase2-baseline  --config PATH  Run the small-scale REAL (tushare) baseline.
    fetch-data           --config PATH  Stage helper (runs the pipeline; see note).
    compute-factors      --config PATH  Stage helper (runs the pipeline; see note).
    run-backtest         --config PATH  Stage helper (runs the pipeline; see note).

The CLI is intentionally thin: orchestration lives in :mod:`qt.pipeline`. Errors
are reported as readable one-line messages (CLI-003), never raw tracebacks. Run
as ``python -m qt.cli <subcommand> --config <path>``.

Note on stage helpers: P0 keeps a single reproducible spine. The fetch-data /
compute-factors / run-backtest sub-commands run that spine and report the stage
of interest, rather than persisting partial cross-process state. ``run-phase0``
is the canonical end-to-end command (CLI-002).
"""

from __future__ import annotations

import argparse
import sys

from qt.config import ConfigError, load_config


def _cmd_validate_config(args: argparse.Namespace) -> int:
    """Load and validate the config; print OK / error and return an exit code."""
    try:
        load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def _run_pipeline_cmd(config: str, stage: str) -> int:
    """Shared runner for run-phase0 and the stage helper sub-commands."""
    # Imported lazily so ``validate-config`` stays light and import errors in a
    # heavy slice never break config validation.
    from qt.pipeline import run_phase0

    try:
        result = run_phase0(config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if stage == "fetch-data":
        print(f"OK fetch-data: {result.panel_rows} rows -> {result.data_path}")
    elif stage == "compute-factors":
        print(f"OK compute-factors: {result.factor_name} -> {result.factor_path}")
    elif stage == "run-backtest":
        annual = result.performance.get("annual_return", float("nan"))
        print(f"OK run-backtest: annual_return={annual:.4f}, nav rows={len(result.nav_table)}")
    else:
        print(
            f"OK run-phase0: ic_mean={result.ic_mean:.4f}, "
            f"annual_return={result.performance.get('annual_return', float('nan')):.4f}\n"
            f"report: {result.report_path}"
        )
    return 0


def _cmd_run_phase0(args: argparse.Namespace) -> int:
    """Run the full end-to-end pipeline and write the phase0 report (CLI-002)."""
    return _run_pipeline_cmd(args.config, "run-phase0")


def _cmd_run_phase2_baseline(args: argparse.Namespace) -> int:
    """Run the small-scale REAL-data (tushare) reproducibility baseline + report."""
    # Imported lazily so validate-config / demo runs never import the heavy
    # real-data baseline module.
    from qt.phase2_baseline import run_phase2_baseline

    try:
        result = run_phase2_baseline(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK run-phase2-baseline: symbols={result.panel_symbols}, "
        f"rebalances={len(result.rebalance_dates)}, ic_mean={result.ic_mean:.4f}, "
        f"annual_return={result.performance.get('annual_return', float('nan')):.4f} "
        f"({result.elapsed_seconds:.1f}s)\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_phase3_oos(args: argparse.Namespace) -> int:
    """Run the OOS stability validation (equal_weight vs ic_weighted) + report."""
    from qt.oos_stability import run_phase3_oos

    try:
        result = run_phase3_oos(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    eq = result.performance["equal_weight"]["test"].get("annual_return", float("nan"))
    ic = result.performance["ic_weighted"]["test"].get("annual_return", float("nan"))
    print(
        f"OK run-phase3-oos: split={result.split_date.date()}, "
        f"test annual eq={eq:.4f} ic={ic:.4f}, fallbacks={result.n_fallback} "
        f"({result.elapsed_seconds:.1f}s)\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_phase3_robustness(args: argparse.Namespace) -> int:
    """Run the robustness matrix (universes x windows OOS cells) + report."""
    from qt.robustness import run_phase3_robustness

    try:
        result = run_phase3_robustness(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK run-phase3-robustness: cells={result.summary['n_cells']}, "
        f"skipped={len(result.skipped_cells)}, "
        f"ic_beats_eq_test={result.summary['ic_beats_eq_test']}/"
        f"{result.summary['n_cells']} ({result.elapsed_seconds:.1f}s)\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_phase3_subset(args: argparse.Namespace) -> int:
    """Run the subset-validation matrix (factor groups x cost scenarios) + report."""
    from qt.subset_validation import run_phase3_subset

    try:
        result = run_phase3_subset(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    n_groups = len(result.summary.get("groups", {}))
    verdict_note = ""
    if result.verdicts:
        statuses = ", ".join(
            f"{label}={v['status']}" for label, v in result.verdicts.items()
        )
        verdict_note = f"\nindependent verdicts: {statuses}"
    print(
        f"OK run-phase3-subset: cells={result.summary['n_cells']}, "
        f"groups={n_groups}, scenarios={len(result.scenario_fees)}, "
        f"skipped={len(result.skipped_cells)} ({result.elapsed_seconds:.1f}s)"
        f"{verdict_note}\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_phase_i5a_intraday(args: argparse.Namespace) -> int:
    """Run the I5a intraday tail-rebalance architecture smoke + report."""
    from qt.intraday_tail_framework import run_phase_i5a_intraday

    try:
        result = run_phase_i5a_intraday(args.config)
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    final_nav = (
        result.nav_table["nav"].iloc[-1] if not result.nav_table.empty else float("nan")
    )
    n_blocked = sum(result.blocked_fill_counts.values())
    print(
        f"OK run-phase-i5a-intraday: periods={len(result.nav_table)}, "
        f"covered={result.covered_symbols}/{result.requested_symbols}, "
        f"stk_mins_live_calls={result.minute_live_calls}, blocked_fills={n_blocked}, "
        f"final_nav={final_nav:.6f}\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_phase_i5d_intraday_groups(args: argparse.Namespace) -> int:
    """Run the I5d MMP quintile grouped intraday-tail backtest + report/figures."""
    from qt.intraday_group_backtest import run_phase_i5d_intraday_groups

    try:
        result = run_phase_i5d_intraday_groups(args.config)
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    navs = " ".join(
        f"Q{g.group}={g.metrics['final_nav']:.4f}" for g in result.groups
    )
    print(
        f"OK run-phase-i5d-intraday-groups: groups={result.n_groups}, "
        f"rebalances={result.rebalance_count}, "
        f"covered={result.covered_symbols}/{result.requested_symbols}, "
        f"stk_mins_live_calls={result.minute_live_calls}, "
        f"fee_rate={result.config.cost.fee_rate}\n"
        f"final_nav: {navs}\n"
        f"report: {result.report_path}"
    )
    return 0


def _cmd_run_eval_jump_amount_corr(args: argparse.Namespace) -> int:
    """Run the two real jump-amount-corr factor evaluations (cache-only) + reports."""
    from qt.eval_jump_amount_corr import run_eval_jump_amount_corr

    try:
        result = run_eval_jump_amount_corr(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    nb, wb = result.no_book_metrics, result.with_book_metrics
    print(
        f"OK run-eval-jump-amount-corr: covered={result.covered_symbols}/"
        f"{result.requested_symbols}, stk_mins_live_calls={result.minute_live_calls}, "
        f"factor_rows={result.factor_rows} ({result.elapsed:.1f}s)\n"
        f"no-book: {nb['deployment']} (predictive={nb['predictive']}) "
        f"ic_mean={nb['ic_mean']:.4f} ic_ir={nb['ic_ir']:.3f} N_eff={nb['effective_samples']:.1f}\n"
        f"with-book: {wb['deployment']} (incremental={wb['incremental']}) "
        f"incr_ic_ir={wb['incremental_ic_ir']:.3f}\n"
        f"reports: {result.reports.no_book_md} | {result.reports.with_book_md}\n"
        f"dashboards: {result.reports.no_book_dashboard} | "
        f"{result.reports.with_book_dashboard}"
    )
    return 0


def _cmd_run_eval_minute_ideal_amplitude(args: argparse.Namespace) -> int:
    """Run the two real minute-ideal-amplitude factor evaluations (cache-only) + reports."""
    from qt.eval_minute_ideal_amplitude import run_eval_minute_ideal_amplitude

    try:
        result = run_eval_minute_ideal_amplitude(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    nb, wb = result.no_book_metrics, result.with_book_metrics
    print(
        f"OK run-eval-minute-ideal-amplitude: covered={result.covered_symbols}/"
        f"{result.requested_symbols}, stk_mins_live_calls={result.minute_live_calls}, "
        f"factor_rows={result.factor_rows} ({result.elapsed:.1f}s)\n"
        f"no-book: {nb['deployment']} (predictive={nb['predictive']}) "
        f"ic_mean={nb['ic_mean']:.4f} ic_ir={nb['ic_ir']:.3f} N_eff={nb['effective_samples']:.1f}\n"
        f"with-book: {wb['deployment']} (incremental={wb['incremental']}) "
        f"incr_ic_ir={wb['incremental_ic_ir']:.3f}\n"
        f"reports: {result.reports.no_book_md} | {result.reports.with_book_md}\n"
        f"dashboards: {result.reports.no_book_dashboard} | "
        f"{result.reports.with_book_dashboard}"
    )
    return 0


def _cmd_data_update(args: argparse.Namespace) -> int:
    """Warm/update the tushare caches (P4-3); never runs a backtest."""
    from qt.data_updater import format_summary, run_data_update

    try:
        result = run_data_update(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = (
        f"OK data-update: {len(result.endpoints)} endpoints, "
        f"{len(result.symbols)} symbols ({result.elapsed_seconds:.1f}s)\n"
        f"{format_summary(result)}"
    )
    # D3b: only surfaced when the report-only quality hook is enabled; with the
    # default (disabled) hook the output is materially unchanged.
    if result.quality_report_path is not None:
        out += (
            f"\nquality: findings={result.quality_findings_count} "
            f"hard={result.quality_hard_count} "
            f"report={result.quality_report_path}"
        )
    print(out)
    return 0


def _cmd_data_backfill(args: argparse.Namespace) -> int:
    """Backfill the tushare caches over the WIDE historical window (PR-2); no backtest."""
    from qt.data_backfill import format_summary, run_data_backfill

    try:
        result = run_data_backfill(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK data-backfill: {result.universe_size} symbols in "
        f"{result.n_batches} batch(es), failed_batches={result.failed_batches} "
        f"({result.elapsed_seconds:.1f}s)\n"
        f"{format_summary(result)}"
    )
    return 0


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    """Stage helper: run the spine and report the data-fetch stage."""
    return _run_pipeline_cmd(args.config, "fetch-data")


def _cmd_compute_factors(args: argparse.Namespace) -> int:
    """Stage helper: run the spine and report the factor-compute stage."""
    return _run_pipeline_cmd(args.config, "compute-factors")


def _cmd_run_backtest(args: argparse.Namespace) -> int:
    """Stage helper: run the spine and report the backtest stage."""
    return _run_pipeline_cmd(args.config, "run-backtest")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="qt",
        description="A-share cross-sectional multi-factor framework (Phase 0).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-config", help="Validate a YAML config file.")
    p_validate.add_argument("--config", required=True, help="Path to the YAML config.")
    p_validate.set_defaults(func=_cmd_validate_config)

    p_run = sub.add_parser("run-phase0", help="Run the end-to-end Phase 0 pipeline.")
    p_run.add_argument("--config", required=True, help="Path to the YAML config.")
    p_run.set_defaults(func=_cmd_run_phase0)

    p_p2 = sub.add_parser(
        "run-phase2-baseline",
        help="Run the small-scale REAL-data (tushare) reproducibility baseline.",
    )
    p_p2.add_argument("--config", required=True, help="Path to the YAML config.")
    p_p2.set_defaults(func=_cmd_run_phase2_baseline)

    p_oos = sub.add_parser(
        "run-phase3-oos",
        help="Run the REAL-data OOS stability validation (equal_weight vs ic_weighted).",
    )
    p_oos.add_argument("--config", required=True, help="Path to the YAML config.")
    p_oos.set_defaults(func=_cmd_run_phase3_oos)

    p_rob = sub.add_parser(
        "run-phase3-robustness",
        help="Run the REAL-data robustness matrix (universes x windows OOS cells).",
    )
    p_rob.add_argument("--config", required=True, help="Path to the YAML config.")
    p_rob.set_defaults(func=_cmd_run_phase3_robustness)

    p_sub = sub.add_parser(
        "run-phase3-subset",
        help="Run the REAL-data subset validation (factor groups x cost scenarios).",
    )
    p_sub.add_argument("--config", required=True, help="Path to the YAML config.")
    p_sub.set_defaults(func=_cmd_run_phase3_subset)

    p_du = sub.add_parser(
        "data-update",
        help="Warm/update the tushare raw caches (P4-3); no backtest.",
    )
    p_du.add_argument("--config", required=True, help="Path to the YAML config.")
    p_du.set_defaults(func=_cmd_data_update)

    p_bf = sub.add_parser(
        "data-backfill",
        help="Backfill the tushare raw caches over the WIDE history (PR-2); no backtest.",
    )
    p_bf.add_argument("--config", required=True, help="Path to the YAML config.")
    p_bf.set_defaults(func=_cmd_data_backfill)

    p_i5a = sub.add_parser(
        "run-phase-i5a-intraday",
        help="Run the I5a intraday tail-rebalance architecture smoke (minute cache).",
    )
    p_i5a.add_argument("--config", required=True, help="Path to the YAML config.")
    p_i5a.set_defaults(func=_cmd_run_phase_i5a_intraday)

    p_i5d = sub.add_parser(
        "run-phase-i5d-intraday-groups",
        help="Run the I5d MMP quintile grouped intraday-tail backtest (5 groups).",
    )
    p_i5d.add_argument("--config", required=True, help="Path to the YAML config.")
    p_i5d.set_defaults(func=_cmd_run_phase_i5d_intraday_groups)

    p_jac = sub.add_parser(
        "run-eval-jump-amount-corr",
        help="Run the first real factor evaluation (jump-amount-corr, CSI500, cache-only).",
    )
    p_jac.add_argument("--config", required=True, help="Path to the YAML config.")
    p_jac.set_defaults(func=_cmd_run_eval_jump_amount_corr)

    p_mia = sub.add_parser(
        "run-eval-minute-ideal-amplitude",
        help="Run the minute-ideal-amplitude factor evaluation (CSI500, cache-only).",
    )
    p_mia.add_argument("--config", required=True, help="Path to the YAML config.")
    p_mia.set_defaults(func=_cmd_run_eval_minute_ideal_amplitude)

    for name, func, help_text in (
        ("fetch-data", _cmd_fetch_data, "Run the spine, report data fetch."),
        ("compute-factors", _cmd_compute_factors, "Run the spine, report factor compute."),
        ("run-backtest", _cmd_run_backtest, "Run the spine, report backtest."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="Path to the YAML config.")
        p.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
