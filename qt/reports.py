"""Markdown report writers for the Phase 0 run (ANA-005, INV-006/007).

``write_phase0_summary`` renders the end-to-end run result into
``artifacts/reports/phase0_summary.md``. It only WRITES under the configured
output dirs (SEC-003) and is fully regenerable from the same config (INV-006).

The summary always contains an explicit DOWNGRADES section (INV-007) so the
static-universe PIT downgrade, the daily-data limitation, and any simple-vs-
alphalens / simple-vs-quantstats fallback are disclosed in the report itself.

``write_bias_audit`` and ``write_runbook`` / ``write_test_report`` emit the
repo-root delivery docs from the framework spec §10.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # avoid a circular import at runtime (pipeline imports reports)
    from qt.phase2_baseline import Phase2Result
    from qt.pipeline import Phase0Result


def _fmt(value: float, pct: bool = False) -> str:
    """Format a float for the report (NaN/inf-safe; optional percent)."""
    if value is None or not math.isfinite(value):  # NaN or +/-inf
        return "n/a"
    return f"{value * 100:.2f}%" if pct else f"{value:.4f}"


def _quantile_table(q_returns: pd.DataFrame) -> str:
    """Render mean-over-time quantile returns as a small markdown table."""
    if q_returns is None or q_returns.empty:
        return "_(no quantile data)_"
    means = q_returns.mean(axis=0)
    header = "| Quantile | Mean forward return |\n|---|---|\n"
    rows = "".join(f"| {int(q)} | {_fmt(float(v), pct=True)} |\n" for q, v in means.items())
    return header + rows


def _err_suffix(d: dict) -> str:
    """' (error_type=X)' when a standard-analytics backend errored, else ''."""
    if (d or {}).get("backend") == "error" and d.get("error_type"):
        return f" (error_type={d['error_type']})"
    return ""


def standard_analytics_block(std_performance: dict, std_factor: dict) -> str:
    """Render the report-only standard-library cross-check (P2-4).

    Shows the quantstats performance + alphalens factor metrics WHEN those backends
    ran, and otherwise discloses the backend (``unavailable`` / ``error`` / ``skipped``)
    so the report never implies a standard library ran when it did not. These never
    replace the authoritative simple metrics shown elsewhere.
    """
    lines: list[str] = []
    qb = (std_performance or {}).get("backend", "n/a")
    if qb == "quantstats":
        sp = std_performance
        lines.append(
            f"- **quantstats** performance: CAGR **{_fmt(sp.get('cagr', float('nan')), pct=True)}** · "
            f"Sharpe **{_fmt(sp.get('sharpe', float('nan')))}** · "
            f"maxDD **{_fmt(sp.get('max_drawdown', float('nan')), pct=True)}** · "
            f"vol **{_fmt(sp.get('volatility', float('nan')), pct=True)}**\n"
        )
    else:
        lines.append(
            f"- quantstats performance: **backend={qb}**{_err_suffix(std_performance)} "
            f"— the simple metrics above are authoritative (not faked).\n"
        )
    ab = (std_factor or {}).get("backend", "n/a")
    if ab == "alphalens":
        sf = std_factor
        qm = sf.get("quantile_mean", {}) or {}
        qstr = ", ".join(f"Q{k} {_fmt(float(v), pct=True)}" for k, v in sorted(qm.items()))
        lines.append(
            f"- **alphalens** factor: IC mean **{_fmt(sf.get('ic_mean', float('nan')))}** · "
            f"IC-IR **{_fmt(sf.get('ic_ir', float('nan')))}** "
            f"({sf.get('n_dates', 0)} dates) · quantile mean fwd-ret: {qstr or '_n/a_'}\n"
        )
    else:
        lines.append(
            f"- alphalens factor: **backend={ab}**{_err_suffix(std_factor)} "
            f"— the simple IC above is authoritative (not faked).\n"
        )
    return "".join(lines)


def alpha_model_block(
    alpha_summary: dict,
    alpha_weights: pd.DataFrame | None,
    rebalance_dates: tuple | None = None,
) -> str:
    """Render the active alpha model + walk-forward weight disclosure (P3-2).

    equal_weight -> one line (no trained weights to disclose). ic_weighted ->
    hyper-params, training coverage (fallback count), and the per-rebalance
    EFFECTIVE weights table (a fallback row shows the equal weights actually
    used, flagged). Always states this is NOT a tuned-performance claim.
    """
    model = (alpha_summary or {}).get("model", "?")
    if model != "ic_weighted" or alpha_weights is None:
        return (
            f"- active alpha model: **`{model}`** — equal-weight mean of the "
            f"processed factor columns; no future data, no trained weights.\n"
        )
    s = alpha_summary
    n_dates, n_fallback = s.get("n_dates", 0), s.get("n_fallback", 0)
    lines = [
        "- active alpha model: **`ic_weighted`** (walk-forward rolling-IC weights)\n",
        f"- params: window=`{s.get('window')}` trading days · "
        f"min_periods=`{s.get('min_periods')}` · horizon=`{s.get('horizon')}`d · "
        f"mode=`{s.get('mode')}`\n",
        "- lookahead boundary: a (factor[t], fwd[t]) pair enters a date d's "
        "weights only once REALIZED (t + horizon <= d, trading days); "
        "insufficient history falls back to equal weight.\n",
        f"- training coverage: **{n_dates - n_fallback}/{n_dates}** scored dates "
        f"used trained weights (**{n_fallback}** equal-weight fallback"
        f"{'s' if n_fallback != 1 else ''}; coverage "
        f"{_fmt(s.get('trained_coverage', float('nan')), pct=True)})\n",
        "- _Weights are L1-normalized and sign-preserving (negative-IC factor -> "
        "negative weight). This is a fixed, disclosed recipe — NOT a tuned-"
        "performance claim._\n\n",
    ]
    show = alpha_weights
    if rebalance_dates:
        wanted = [d for d in rebalance_dates if d in alpha_weights.index]
        show = alpha_weights.loc[wanted] if wanted else alpha_weights.iloc[0:0]
        lines.append("Effective weights at each SETTLED rebalance date:\n\n")
    else:
        lines.append("Effective weights (all scored dates summarized below):\n\n")
        show = alpha_weights.iloc[0:0]  # phase0 keeps it short: summary only
    if not show.empty:
        factor_cols = [c for c in show.columns if c != "fallback"]
        header = "| Date | " + " | ".join(f"`{c}`" for c in factor_cols) + " | fallback |\n"
        header += "|" + "---|" * (len(factor_cols) + 2) + "\n"
        body = ""
        for date, row in show.iterrows():
            cells = " | ".join(_fmt(float(row[c])) for c in factor_cols)
            body += f"| {_date_str(date)} | {cells} | {'YES' if row['fallback'] else 'no'} |\n"
        lines.append(header + body)
    return "".join(lines)


def _combo_label(alpha_summary: dict) -> str:
    """Label the combo-score rows by the ACTIVE alpha model (P3-2)."""
    model = (alpha_summary or {}).get("model", "equal_weight")
    return (
        "combo score (ic-weighted, walk-forward)"
        if model == "ic_weighted"
        else "combo score (equal-weight)"
    )


def per_factor_ic_table(
    per_factor: dict, combo: dict, combo_label: str = "combo score (equal-weight)"
) -> str:
    """Per-factor + combo-score IC table (simple implementation, P3-1).

    One row per RAW factor (with its non-NaN coverage) plus the COMBO row — the
    processed score the backtest actually trades (label names the active alpha,
    P3-2). The combo has no raw-coverage notion (post drop_missing), shown '—'.
    """
    header = "| Factor | coverage | IC mean | IC IR |\n|---|---|---|---|\n"
    rows = ""
    for name, m in (per_factor or {}).items():
        rows += (
            f"| `{name}` | {_fmt(m.get('coverage', float('nan')), pct=True)} | "
            f"{_fmt(m.get('ic_mean', float('nan')))} | "
            f"{_fmt(m.get('ic_ir', float('nan')))} |\n"
        )
    c = combo or {}
    rows += (
        f"| **{combo_label}** | — | {_fmt(c.get('ic_mean', float('nan')))} | "
        f"{_fmt(c.get('ic_ir', float('nan')))} |\n"
    )
    return header + rows


def per_factor_quantiles_block(
    per_factor: dict, combo: dict, combo_label: str = "combo score (equal-weight)"
) -> str:
    """Quantile-return tables per factor plus the combo score (P3-1)."""
    parts: list[str] = []
    for name, m in (per_factor or {}).items():
        parts.append(f"### `{name}`\n\n")
        parts.append(_quantile_table(m.get("quantile_returns")) + "\n")
    parts.append(f"### {combo_label}\n\n")
    parts.append(_quantile_table((combo or {}).get("quantile_returns")) + "\n")
    return "".join(parts)


def render_phase0_summary(result: "Phase0Result") -> str:
    """Build the phase0 summary markdown string (pure; no I/O)."""
    cfg = result.config
    perf = result.performance
    lines: list[str] = []
    lines.append("# Phase 0 Summary\n")
    lines.append(f"Project: **{cfg.project.name}**\n")

    lines.append("## Config echo\n")
    lines.append(
        f"- data: source=`{cfg.data.source}`, freq=`{cfg.data.freq}`, "
        f"window=`[{cfg.data.start}, {cfg.data.end}]`\n"
        f"- universe: type=`{cfg.universe.type}`, "
        f"symbols={list(cfg.universe.symbols)}\n"
        f"- factors (active): `{list(result.factor_names)}` "
        f"(primary: `{result.factor_name}`)\n"
        f"- alpha: `{cfg.alpha.model}`\n"
        f"- portfolio: `{cfg.portfolio.constructor}`, top_n=`{cfg.portfolio.top_n}`\n"
        f"- backtest: rebalance=`{cfg.backtest.rebalance}`, "
        f"event_order=`{cfg.backtest.event_order}`\n"
        f"- cost: fee_rate=`{cfg.cost.fee_rate}`, slippage=`{cfg.cost.slippage_rate}`\n"
    )

    lines.append("## Alpha model\n")
    lines.append(alpha_model_block(result.alpha_summary, result.alpha_weights))
    if result.alpha_weights is not None and not result.alpha_weights.empty:
        w = result.alpha_weights
        factor_cols = [c for c in w.columns if c != "fallback"]
        means = ", ".join(
            f"`{c}` {_fmt(float(w.loc[~w['fallback'], c].mean()) if (~w['fallback']).any() else float('nan'))}"
            for c in factor_cols
        )
        lines.append(f"- mean trained weights over scored dates: {means}\n")

    lines.append("## Data shape\n")
    lines.append(
        f"- panel rows: **{result.panel_rows}**\n"
        f"- symbols: **{result.panel_symbols}**\n"
    )

    lines.append("## Factor IC\n")
    lines.append(
        f"- IC mean: **{_fmt(result.ic_mean)}** (primary `{result.factor_name}`)\n"
        f"- IC_IR (mean/std): **{_fmt(result.ic_ir)}**\n\n"
    )
    lines.append(per_factor_ic_table(
        result.per_factor, result.combo_analytics, _combo_label(result.alpha_summary)
    ))

    lines.append("## Quantile returns\n")
    lines.append(_quantile_table(result.quantile_returns) + "\n")
    if len(result.factor_names) > 1:
        lines.append("\n")
        lines.append(per_factor_quantiles_block(
            result.per_factor, result.combo_analytics, _combo_label(result.alpha_summary)
        ))

    lines.append("## Portfolio performance\n")
    lines.append(
        f"- annual return: **{_fmt(perf.get('annual_return', float('nan')), pct=True)}**\n"
        f"- max drawdown: **{_fmt(perf.get('max_drawdown', float('nan')), pct=True)}**\n"
        f"- volatility: **{_fmt(perf.get('volatility', float('nan')), pct=True)}**\n"
        f"- sharpe: **{_fmt(perf.get('sharpe', float('nan')))}**\n"
    )

    lines.append("## Turnover & cost\n")
    lines.append(
        f"- average turnover (per rebalance): **{_fmt(result.avg_turnover)}**\n"
        f"- total cost drag: **{_fmt(result.cost_drag, pct=True)}**\n"
    )

    lines.append("## Standard analytics (alphalens / quantstats cross-check)\n")
    lines.append(
        "_Report-only standard-library metrics; the simple metrics above remain the "
        "authoritative backtest result (these never alter trading)._\n"
    )
    lines.append(standard_analytics_block(result.std_performance, result.std_factor))

    lines.append("## DOWNGRADES (INV-007 — must be disclosed)\n")
    lines.append(
        "This Phase 0 run intentionally uses simplified / downgraded components. "
        "Each is recorded here so no downgrade is hidden:\n"
    )
    for item in result.downgrades:
        lines.append(f"- {item}\n")

    lines.append("\n## Artifacts\n")
    lines.append(
        f"- data: `{result.data_path}`\n"
        f"- factors: `{result.factor_path}`\n"
        f"- report: `{result.report_path}`\n"
        f"- log: `{result.log_path}`\n"
    )
    return "".join(lines)


def write_phase0_summary(result: "Phase0Result") -> Path:
    """Render and write the phase0 summary; return the path written (SEC-003)."""
    target = result.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_phase0_summary(result), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Phase 2-1 real-data baseline report (qt.phase2_baseline).
# --------------------------------------------------------------------------- #
_PHASE2_REQUIRED_SECTIONS = (
    "## Data window",
    "## Universe / PIT membership",
    "## Alpha model",
    "## Financial ann_date coverage",
    "## Tradability filter hits",
    "## Execution feasibility",
    "## Rebalance dates",
    "## Holdings per period",
    "## Turnover & cost",
    "## Factor IC",
    "## Quantile returns",
    "## Portfolio performance",
    "## Standard analytics",
    "## DOWNGRADES",
)


def phase2_baseline_required_sections() -> tuple[str, ...]:
    """Section headers the phase2 baseline report MUST contain (single source)."""
    return _PHASE2_REQUIRED_SECTIONS


def _date_str(value: object) -> str:
    """Format a timestamp-ish value as YYYY-MM-DD (or 'n/a')."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)


def _universe_block(summ: dict) -> str:
    """Render the universe / PIT membership summary (loaded vs in-window)."""
    if summ.get("pit"):
        return (
            f"- type: **index** (point-in-time, survivorship-safe)\n"
            f"- loaded snapshots: **{summ.get('n_loaded_snapshots', 0)}** "
            f"({_date_str(summ.get('loaded_first'))} → {_date_str(summ.get('loaded_last'))}; "
            f"includes the ~370-day pre-start lookback for as-of safety)\n"
            f"- in-window membership updates: **{summ.get('n_window_snapshots', 0)}** "
            f"(as-of anchor at start: {_date_str(summ.get('anchor_snapshot'))})\n"
            f"- distinct names in-window: **{summ.get('distinct_names_in_window', 0)}**\n"
            f"- per-snapshot size: {summ.get('min_size', 0)}–{summ.get('max_size', 0)}\n"
            f"- avg churn per in-window update: **{summ.get('avg_churn_in', 0.0):.1f} in / "
            f"{summ.get('avg_churn_out', 0.0):.1f} out**\n"
        )
    return (
        f"- type: **{summ.get('type', '?')}** (NOT point-in-time — survivorship/"
        f"look-ahead membership downgrade)\n"
        f"- symbols: **{summ.get('distinct_names_in_window', 0)}**\n"
    )


def _coverage_block(overall: float, by_reb: pd.DataFrame) -> str:
    """Render the ann_date as-of financial coverage diagnostic."""
    head = f"- overall as-of coverage (non-NaN rows): **{_fmt(overall, pct=True)}**\n\n"
    if by_reb is None or by_reb.empty:
        return head + "_(no rebalance cross-sections)_\n"
    table = "| Rebalance date | members | covered | coverage |\n|---|---|---|---|\n"
    for _, r in by_reb.iterrows():
        table += (
            f"| {_date_str(r['date'])} | {int(r['n_members'])} | "
            f"{int(r['n_covered'])} | {_fmt(float(r['coverage']), pct=True)} |\n"
        )
    return head + table


def _financial_coverage_block(financial_coverage: dict) -> str:
    """Render the per-field ann_date coverage (P3-1): one subsection per field.

    Each field is labelled by its role — a TRADED financial factor vs a pure
    diagnostic — so a reader can never mistake a data-quality lens for a signal
    (or vice versa).
    """
    if not financial_coverage:
        return "_(no financial fields)_\n"
    parts: list[str] = []
    for field, info in financial_coverage.items():
        role = (
            "TRADED financial factor in this run (ann_date PIT-aligned)"
            if info.get("is_factor")
            else "diagnostic only — NOT an alpha factor in this run"
        )
        parts.append(f"### `{field}` — {role}\n\n")
        parts.append(
            _coverage_block(
                info.get("overall", float("nan")), info.get("by_rebalance")
            )
        )
        parts.append("\n")
    return "".join(parts)


def _tradability_block(hits: pd.DataFrame) -> str:
    """Render the tradability filter-hit funnel."""
    candidates = hits.attrs.get("candidates", 0)
    tradable = hits.attrs.get("tradable", 0)
    head = (
        f"- member-days evaluated (over rebalance dates): **{candidates}**\n"
        f"- tradable after filters: **{tradable}**\n\n"
    )
    table = "| Filter reason | hits (member-days dropped) |\n|---|---|\n"
    for reason, row in hits.iterrows():
        table += f"| {reason} | {int(row['hits'])} |\n"
    return head + table


def _feasibility_block(log: pd.DataFrame) -> str:
    """Render the direction-aware execution-feasibility funnel (P2-2)."""
    if log is None or log.empty:
        return "_(no settled rebalances)_\n"
    tot_bb = int(log["blocked_buys"].sum())
    tot_bs = int(log["blocked_sells"].sum())
    tot_cc = int(log["cash_constrained_buys"].sum())
    avg_inv = float(log["invested"].mean())
    head = (
        f"_Direction-aware fills: at-up-limit blocks buys, at-down-limit blocks "
        f"sells, suspended/missing blocks both; blocked trades carry forward and "
        f"turnover/cost count only executed trades._\n\n"
        f"- total blocked buys: **{tot_bb}** · blocked sells: **{tot_bs}** · "
        f"cash-constrained buys: **{tot_cc}**\n"
        f"- avg invested fraction (1 − idle cash): **{_fmt(avg_inv)}**\n\n"
    )
    table = (
        "| Rebalance date | blocked_buys | blocked_sells | carried | exec_turnover | invested |\n"
        "|---|---|---|---|---|---|\n"
    )
    for date, r in log.iterrows():
        table += (
            f"| {_date_str(date)} | {int(r['blocked_buys'])} | {int(r['blocked_sells'])} | "
            f"{int(r['carried'])} | {_fmt(float(r['executed_turnover']))} | "
            f"{_fmt(float(r['invested']))} |\n"
        )
    return head + table


def _holdings_block(holdings: pd.DataFrame) -> str:
    """Render per-period holdings (complete; one line per rebalance date)."""
    if holdings is None or holdings.empty:
        return "_(no holdings — universe was empty every rebalance)_\n"
    lines: list[str] = []
    for date, block in holdings.groupby("date", sort=True):
        syms = list(block.sort_values("rank")["symbol"])
        k = len(syms)
        w = block["weight"].iloc[0] if k else float("nan")
        lines.append(f"- **{_date_str(date)}** ({k} names, w≈{_fmt(float(w))}): {', '.join(syms)}\n")
    return "".join(lines)


def _per_period_table(nav: pd.DataFrame) -> str:
    """Render per-rebalance turnover / cost / net return."""
    if nav is None or nav.empty:
        return "_(no settled periods)_\n"
    table = "| Rebalance date | turnover | cost | net return | nav |\n|---|---|---|---|---|\n"
    for date, r in nav.iterrows():
        table += (
            f"| {_date_str(date)} | {_fmt(float(r['turnover']))} | "
            f"{_fmt(float(r['cost']), pct=True)} | {_fmt(float(r['net_return']), pct=True)} | "
            f"{_fmt(float(r['nav']))} |\n"
        )
    return table


def render_phase2_baseline(result: "Phase2Result") -> str:
    """Build the phase2 real-baseline markdown (pure; no I/O, no secrets).

    The tushare token / secret file is never echoed; only non-sensitive config
    (window, universe type, factor) and computed diagnostics are written.
    """
    cfg = result.config
    perf = result.performance
    lines: list[str] = []
    lines.append("# Real-data Reproducibility Baseline\n")
    lines.append(
        f"Project: **{cfg.project.name}** · source: **{cfg.data.source}** · "
        f"ran in **{result.elapsed_seconds:.1f}s**\n"
    )
    lines.append(
        "\n> Small-scale REAL (tushare) baseline that runs the existing pipeline "
        "spine end-to-end. No parameter search, not a performance claim — it "
        "validates the real-data plumbing and is fully reproducible from the "
        "config below.\n"
    )

    lines.append("\n## Config echo\n")
    lines.append(
        f"- universe: type=`{cfg.universe.type}`, index_code=`{cfg.universe.index_code}`, "
        f"top_n=`{cfg.portfolio.top_n}`\n"
        f"- factors (active): `{list(result.factor_names)}` "
        f"(primary for the std cross-check: `{result.factor_name}`) · "
        f"neutralize=`{cfg.processing.neutralize.enabled}`\n"
        f"- filters: `{cfg.universe.filters.model_dump()}`\n"
        f"- backtest: rebalance=`{cfg.backtest.rebalance}`, fee_rate=`{cfg.cost.fee_rate}`\n"
    )

    lines.append("\n## Data window\n")
    lines.append(
        f"- configured: `[{cfg.data.start}, {cfg.data.end}]`\n"
        f"- panel calendar: {_date_str(result.first_trade_date)} → "
        f"{_date_str(result.last_trade_date)} (**{result.trade_days}** trading days)\n"
        f"- panel rows: **{result.panel_rows}**, symbols: **{result.panel_symbols}**\n"
    )

    lines.append("\n## Universe / PIT membership\n")
    lines.append(_universe_block(result.universe_summary))
    if result.list_date_total:
        missing = result.list_date_total - result.list_date_known
        lines.append(
            f"- min_listing_days `list_date` coverage (this run): "
            f"**{result.list_date_known}/{result.list_date_total}** known, "
            f"**{missing}** missing (kept as a disclosed data gap, never excluded)\n"
        )
    if math.isfinite(result.industry_pit_coverage):
        level = result.config.processing.neutralize.industry_level
        lines.append(
            f"- neutralization industry is **point-in-time SW-{level}** (as-of trade date; "
            f"level configurable, default L1); PIT coverage this run: "
            f"**{_fmt(result.industry_pit_coverage, pct=True)}** of (date, symbol) rows. "
            f"Names with no SW history get NaN (dropped by the neutralizer) — never a "
            f"current-tag fallback.\n"
        )

    lines.append("\n## Alpha model\n")
    lines.append(
        alpha_model_block(
            result.alpha_summary, result.alpha_weights, result.rebalance_dates
        )
    )
    if (result.alpha_summary or {}).get("model") == "ic_weighted":
        lines.append(
            "\n_Comparable EQUAL-WEIGHT baseline: the same universe / window / "
            "factors under `config/phase3_real_multifactor.yaml` "
            "(alpha.model=equal_weight) — rerun it for a side-by-side; this "
            "report makes no tuned-performance claim._\n"
        )

    lines.append("\n## Financial ann_date coverage\n")
    lines.append(
        "_Per-field ann_date as-of coverage; each field is labelled TRADED factor "
        "vs diagnostic-only below._\n\n"
    )
    lines.append(_financial_coverage_block(result.financial_coverage))

    lines.append("\n## Tradability filter hits\n")
    lines.append(_tradability_block(result.tradability_hits))

    lines.append("\n## Execution feasibility\n")
    lines.append(_feasibility_block(result.feasibility_log))

    lines.append("\n## Rebalance dates\n")
    reb = ", ".join(_date_str(d) for d in result.rebalance_dates) or "_(none)_"
    lines.append(
        f"- **{len(result.rebalance_dates)}** settled rebalance dates (held + "
        f"settled; these drive holdings / filter hits / coverage below): {reb}\n"
    )
    lines.append(
        f"- candidate rebalance dates (last trading day of each month): "
        f"**{len(result.candidate_rebalance_dates)}**\n"
    )
    if result.skipped_terminal_dates:
        skipped = ", ".join(_date_str(d) for d in result.skipped_terminal_dates)
        lines.append(
            f"- skipped (terminal date(s) with no forward holding period, BT-003): "
            f"{skipped}\n"
        )

    lines.append("\n## Holdings per period\n")
    lines.append(
        "_ACHIEVED holdings (the actual book held after execution feasibility), NOT "
        "the constructor's desired target: a name whose sell was blocked appears "
        "carried here, a name whose buy was blocked is absent. Settled dates only, "
        "so they match the NAV/turnover table 1:1._\n\n"
    )
    lines.append(_holdings_block(result.holdings))

    lines.append("\n## Turnover & cost\n")
    lines.append(
        f"- average turnover (per rebalance): **{_fmt(result.avg_turnover)}**\n"
        f"- total cost drag: **{_fmt(result.cost_drag, pct=True)}**\n\n"
    )
    lines.append(_per_period_table(result.nav_table))

    lines.append("\n## Factor IC\n")
    lines.append(
        f"- IC mean: **{_fmt(result.ic_mean)}** (primary `{result.factor_name}`)\n"
        f"- IC_IR (mean/std): **{_fmt(result.ic_ir)}**\n\n"
    )
    lines.append(per_factor_ic_table(
        result.per_factor, result.combo_analytics, _combo_label(result.alpha_summary)
    ))

    lines.append("\n## Quantile returns\n")
    lines.append(per_factor_quantiles_block(
        result.per_factor, result.combo_analytics, _combo_label(result.alpha_summary)
    ))

    lines.append("\n## Portfolio performance\n")
    lines.append(
        f"- annual return: **{_fmt(perf.get('annual_return', float('nan')), pct=True)}**\n"
        f"- max drawdown: **{_fmt(perf.get('max_drawdown', float('nan')), pct=True)}**\n"
        f"- volatility: **{_fmt(perf.get('volatility', float('nan')), pct=True)}**\n"
        f"- sharpe: **{_fmt(perf.get('sharpe', float('nan')))}**\n"
    )

    lines.append("\n## Standard analytics (alphalens / quantstats cross-check)\n")
    lines.append(
        "_Report-only standard-library metrics; the simple metrics above remain the "
        "authoritative backtest result (these never alter trading)._\n\n"
    )
    lines.append(standard_analytics_block(result.std_performance, result.std_factor))

    lines.append("\n## DOWNGRADES (INV-007 — must be disclosed)\n")
    for item in result.downgrades:
        lines.append(f"- {item}\n")

    lines.append("\n## Artifacts\n")
    lines.append(f"- report: `{result.report_path}`\n- log: `{result.log_path}`\n")
    return "".join(lines)


def write_phase2_baseline_summary(result: "Phase2Result") -> Path:
    """Render and write the phase2 baseline report; return the path (SEC-003)."""
    target = result.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_phase2_baseline(result), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Phase 3-3 OOS stability report (qt.oos_stability).
# --------------------------------------------------------------------------- #
def _oos_perf_table(performance: dict) -> str:
    """Both models × both subperiods performance table."""
    header = (
        "| Model | Period | annual | vol | sharpe | maxDD | avg turnover | rebalances |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for model in ("equal_weight", "ic_weighted"):
        for period in ("train", "test"):
            p = (performance.get(model) or {}).get(period) or {}
            rows += (
                f"| `{model}` | {period} | "
                f"{_fmt(p.get('annual_return', float('nan')), pct=True)} | "
                f"{_fmt(p.get('volatility', float('nan')), pct=True)} | "
                f"{_fmt(p.get('sharpe', float('nan')))} | "
                f"{_fmt(p.get('max_drawdown', float('nan')), pct=True)} | "
                f"{_fmt(p.get('avg_turnover', float('nan')))} | "
                f"{int(p.get('n_rebalances', 0))} |\n"
            )
    return header + rows


def _oos_ic_table(ic_stats: dict, sign_consistency: dict) -> str:
    """Per-series IC stability table (train + test rows, sign-consistency flag)."""
    header = (
        "| Series | Period | IC mean | IC IR | hit rate | n | sign consistency |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for series, stats in ic_stats.items():
        flag = "YES" if sign_consistency.get(series) else "NO"
        for period in ("train", "test"):
            p = stats.get(period) or {}
            rows += (
                f"| `{series}` | {period} | {_fmt(p.get('ic_mean', float('nan')))} | "
                f"{_fmt(p.get('ic_ir', float('nan')))} | "
                f"{_fmt(p.get('hit_rate', float('nan')), pct=True)} | "
                f"{int(p.get('n', 0))} | "
                f"{flag if period == 'train' else ''} |\n"
            )
    return header + rows


def render_oos_stability(result) -> str:
    """Build the phase3 OOS stability markdown (pure; no I/O, no secrets)."""
    cfg = result.config
    lines: list[str] = []
    lines.append("# Phase 3-3 — OOS Stability Validation (equal_weight vs ic_weighted)\n")
    lines.append(
        f"Project: **{cfg.project.name}** · source: **{cfg.data.source}** · "
        f"ran in **{result.elapsed_seconds:.1f}s**\n"
    )
    lines.append(
        "\n> **This is NOT a return claim.** A small-sample stability check on one "
        "index over two years: it compares the same three factors under "
        "equal-weight vs walk-forward IC-weighted combination across a train/test "
        "split. Subperiod metrics carry wide uncertainty.\n"
    )

    lines.append("\n## Split boundaries\n")
    lines.append(
        f"- split_date: **{_date_str(result.split_date)}** (train = "
        f"[data.start, split), test = [split, data.end])\n"
        f"- train: **{_date_str(result.train_start)} → {_date_str(result.train_end)}** "
        f"({result.n_train_days} trading days)\n"
        f"- test: **{_date_str(result.test_start)} → {_date_str(result.test_end)}** "
        f"({result.n_test_days} trading days)\n"
        f"- semantics: evaluation is WALK-FORWARD (rolling subperiod) — weights at "
        f"any date d use only observations REALIZED by d (t + horizon <= d), so no "
        f"test-period forward return reaches a train-period computation (locked by "
        f"tests). The split is an accounting boundary for the statistics below.\n"
        f"- performance slicing is HOLDING-WINDOW aware: a rebalance row's return "
        f"covers [that rebalance, the next one], so train rows must have their "
        f"holding END on/before the split and test rows their holding START on/"
        f"after it. IC stats are sliced by the realization date (t + horizon) the "
        f"same way.\n"
    )
    if result.boundary_dates:
        bd = ", ".join(_date_str(d) for d in result.boundary_dates)
        lines.append(
            f"- straddling rebalance(s) excluded from BOTH subperiods (holding "
            f"window crosses the split): {bd}\n"
        )

    lines.append("\n## Config echo\n")
    lines.append(
        f"- universe: type=`{cfg.universe.type}`, index_code=`{cfg.universe.index_code}`, "
        f"top_n=`{cfg.portfolio.top_n}`\n"
        f"- factors: `{list(result.factor_names)}` · "
        f"neutralize=`{cfg.processing.neutralize.enabled}`\n"
        f"- ic_weighted params: {result.alpha_summary}\n"
    )

    lines.append("\n## Subperiod performance (same data, same rules, two alphas)\n")
    lines.append(_oos_perf_table(result.performance))

    lines.append("\n## IC stability (raw factors + both combo scores)\n")
    lines.append(
        "_hit rate = share of positive daily ICs; sign consistency = train and "
        "test mean ICs share one nonzero sign._\n\n"
    )
    lines.append(_oos_ic_table(result.ic_stats, result.sign_consistency))

    lines.append("\n## Weight stability (ic_weighted)\n")
    lines.append(
        f"- scored dates: **{result.n_scored}** · equal-weight fallbacks: "
        f"**{result.n_fallback}**\n"
    )
    for reason, count in (result.fallback_reasons or {}).items():
        lines.append(f"  - fallback reason ×{count}: {reason}\n")
    flips = ", ".join(f"`{k}` {v}" for k, v in (result.sign_flips or {}).items())
    lines.append(
        f"- sign flips between consecutive TRAINED rebalance weights "
        f"(fallback rows excluded): {flips or '_n/a_'}\n\n"
    )
    w = result.weights_at_rebalances
    if w is not None and not w.empty:
        factor_cols = [c for c in w.columns if c != "fallback"]
        lines.append("Effective weights at each settled rebalance date "
                     "(ic_weighted leg):\n\n")
        header = "| Date | period | " + " | ".join(f"`{c}`" for c in factor_cols)
        header += " | fallback |\n|" + "---|" * (len(factor_cols) + 3) + "\n"
        body = ""
        # period labels match the PERFORMANCE slicing: a straddling rebalance
        # (holding window crosses the split) is labelled boundary, not train.
        boundary = {pd.Timestamp(d) for d in (result.boundary_dates or ())}
        for date, row in w.iterrows():
            ts = pd.Timestamp(date)
            if ts in boundary:
                period = "boundary"
            else:
                period = "test" if ts >= result.split_date else "train"
            cells = " | ".join(_fmt(float(row[c])) for c in factor_cols)
            body += (
                f"| {_date_str(date)} | {period} | {cells} | "
                f"{'YES' if row['fallback'] else 'no'} |\n"
            )
        lines.append(header + body)

    lines.append("\n## DOWNGRADES / caveats (INV-007 — must be disclosed)\n")
    for item in result.downgrades:
        lines.append(f"- {item}\n")

    lines.append("\n## Artifacts\n")
    lines.append(f"- report: `{result.report_path}`\n- log: `{result.log_path}`\n")
    return "".join(lines)


def write_oos_stability_summary(result) -> Path:
    """Render and write the OOS stability report; return the path (SEC-003)."""
    target = result.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_oos_stability(result), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Phase 3-4 robustness matrix report (qt.robustness).
# --------------------------------------------------------------------------- #
def _matrix_summary_block(summary: dict) -> str:
    """Cross-cell stability matrix: which findings hold across cells."""
    n = int(summary.get("n_cells", 0))
    lines = [
        f"- cells aggregated: **{n}**\n",
        f"- ic_weighted beats equal_weight on TEST annual return in "
        f"**{summary.get('ic_beats_eq_test', 0)}/{n}** cells\n\n",
        "| Series | test IC > 0 | train→test sign consistent | test IC by cell |\n"
        "|---|---|---|---|\n",
    ]
    for name, s in (summary.get("series") or {}).items():
        by_cell = ", ".join(
            f"{label}: {_fmt(v)}" for label, v in (s.get("test_ic_by_cell") or {}).items()
        )
        lines.append(
            f"| `{name}` | {s.get('test_ic_positive', 0)}/{s.get('n_cells', 0)} | "
            f"{s.get('sign_consistent', 0)}/{s.get('n_cells', 0)} | {by_cell} |\n"
        )
    return "".join(lines)


def render_robustness_matrix(result) -> str:
    """Build the phase3 robustness-matrix markdown (pure; no I/O, no secrets)."""
    cfg = result.config
    lines: list[str] = []
    lines.append("# Phase 3-4 — Robustness Matrix "
                 "(universes × windows, equal_weight vs ic_weighted)\n")
    lines.append(
        f"Project: **{cfg.project.name}** · source: **{cfg.data.source}** · "
        f"ran in **{result.elapsed_seconds:.1f}s**\n"
    )
    lines.append(
        "\n> **This is NOT a return claim and not a tuned result.** It re-runs the "
        "P3-3 OOS stability check (same three factors, same rules, walk-forward "
        "weights, holding-window subperiod slicing) on every universe × window "
        "cell, to test whether the single-sample conclusions generalize. "
        "Small-sample caveats apply to every cell.\n"
    )

    lines.append("\n## Cells\n")
    lines.append(
        "| Cell (universe \\| window) | window | split | runtime |\n|---|---|---|---|\n"
    )
    for label, cell in result.cells.items():
        runtime = result.cell_runtimes.get(label, float("nan"))
        lines.append(
            f"| `{label}` | {_date_str(cell.train_start)} → "
            f"{_date_str(cell.test_end)} | {_date_str(cell.split_date)} | "
            f"{runtime:.0f}s |\n"
        )
    if result.skipped_cells:
        sk = ", ".join(f"`{s}`" for s in result.skipped_cells)
        lines.append(
            f"\n- **skipped cells (disclosed, runtime budget — coverage is "
            f"reduced, not hidden):** {sk}\n"
        )

    lines.append("\n## Cross-cell stability summary\n")
    lines.append(_matrix_summary_block(result.summary))

    for label, cell in result.cells.items():
        lines.append(f"\n## Cell `{label}`\n")
        lines.append(
            f"- train: {_date_str(cell.train_start)} → {_date_str(cell.train_end)} "
            f"({cell.n_train_days}d) · test: {_date_str(cell.test_start)} → "
            f"{_date_str(cell.test_end)} ({cell.n_test_days}d) · split "
            f"{_date_str(cell.split_date)}\n"
        )
        if cell.boundary_dates:
            bd = ", ".join(_date_str(d) for d in cell.boundary_dates)
            lines.append(
                f"- boundary rebalance(s) excluded from both subperiods "
                f"(holding window straddles the split): {bd}\n"
            )
        lines.append("\n### Subperiod performance\n")
        lines.append(_oos_perf_table(cell.performance))
        lines.append("\n### IC stability\n")
        lines.append(_oos_ic_table(cell.ic_stats, cell.sign_consistency))
        flips = ", ".join(f"`{k}` {v}" for k, v in (cell.sign_flips or {}).items())
        lines.append(
            f"\n### Weight stability (ic_weighted)\n"
            f"- scored dates: **{cell.n_scored}** · equal-weight fallbacks: "
            f"**{cell.n_fallback}**\n"
            f"- sign flips between consecutive trained rebalance weights: "
            f"{flips or '_n/a_'}\n"
        )

    lines.append("\n## DOWNGRADES / caveats (INV-007 — must be disclosed)\n")
    # matrix-level scope FIRST: the disclosures below must never read as a
    # single-universe run (the matrix spans several universes × windows).
    run_labels = ", ".join(f"`{label}`" for label in result.cells)
    universes = sorted({label.split("|", 1)[0] for label in result.cells})
    windows = sorted({label.split("|", 1)[1] for label in result.cells})
    skipped = ", ".join(f"`{s}`" for s in result.skipped_cells) or "none"
    lines.append(
        f"- MATRIX SCOPE: run cells: {run_labels}; skipped cells: {skipped}; "
        f"universes covered: {universes}; windows covered: {windows}. "
        "Universe-specific disclosures below are the UNION over all run cells "
        "(each universe's PIT-membership line appears once — never only the "
        "first cell's; see the Cells section for per-cell identity).\n"
    )
    seen: set[str] = set()
    for cell in result.cells.values():
        for item in cell.downgrades:
            if item not in seen:
                seen.add(item)
                lines.append(f"- {item}\n")
    lines.append(
        "- MATRIX CAVEAT: per-cell metrics remain SMALL-SAMPLE (each cell = one "
        "index over one window, ~22 rebalances); the matrix spans multiple "
        "universes × windows to test REPEATABILITY — the cross-cell summary "
        "shows which findings repeat, not that any of them is tradable. NOT a "
        "return claim.\n"
    )

    lines.append("\n## Artifacts\n")
    lines.append(f"- report: `{result.report_path}`\n- log: `{result.log_path}`\n")
    return "".join(lines)


def write_robustness_matrix_summary(result) -> Path:
    """Render and write the robustness-matrix report; return the path (SEC-003)."""
    target = result.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_robustness_matrix(result), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Phase 3-6 subset validation report (qt.subset_validation).
# --------------------------------------------------------------------------- #
def _subset_perf_table(performance: dict) -> str:
    """Scenario × model × period performance table (with cost metrics)."""
    header = (
        "| Scenario | Model | Period | annual | vol | sharpe | maxDD | "
        "avg turnover | total cost | cost drag (ann.) | rebalances |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for scn, models in performance.items():
        for model in ("equal_weight", "ic_weighted"):
            for period in ("train", "test"):
                p = (models.get(model) or {}).get(period) or {}
                rows += (
                    f"| `{scn}` | `{model}` | {period} | "
                    f"{_fmt(p.get('annual_return', float('nan')), pct=True)} | "
                    f"{_fmt(p.get('volatility', float('nan')), pct=True)} | "
                    f"{_fmt(p.get('sharpe', float('nan')))} | "
                    f"{_fmt(p.get('max_drawdown', float('nan')), pct=True)} | "
                    f"{_fmt(p.get('avg_turnover', float('nan')))} | "
                    f"{_fmt(p.get('total_cost', float('nan')), pct=True)} | "
                    f"{_fmt(p.get('cost_drag_annual', float('nan')), pct=True)} | "
                    f"{int(p.get('n_rebalances', 0))} |\n"
                )
    return header + rows


def _subset_group_summary_block(glabel: str, gsum: dict, level: str = "###") -> str:
    """One group's cross-cell summary (combo stability + cost ladder)."""
    n = int(gsum.get("n_cells", 0))
    lines = [
        f"\n{level} Group `{glabel}` across cells\n",
        f"- ic_weighted beats equal_weight on TEST annual return at the BASE "
        f"scenario in **{gsum.get('ic_beats_eq_test_base', 0)}/{n}** cells\n\n",
        "| Combo series | test IC > 0 | train→test sign consistent | "
        "test IC by cell |\n|---|---|---|---|\n",
    ]
    for series, s in (gsum.get("combo") or {}).items():
        by_cell = ", ".join(
            f"{label}: {_fmt(v)}"
            for label, v in (s.get("test_ic_by_cell") or {}).items()
        )
        lines.append(
            f"| `{series}` | {s.get('test_ic_positive', 0)}/{n} | "
            f"{s.get('sign_consistent', 0)}/{n} | {by_cell} |\n"
        )
    ladder = gsum.get("ic_test_annual_by_scenario") or {}
    if ladder:
        lines.append(
            "\nCost ladder — ic_weighted TEST annual return by scenario "
            "(same trades, scaled fee):\n\n"
        )
        lines.append("| Scenario | " + " | ".join(
            f"`{c}`" for c in next(iter(ladder.values()), {})
        ) + " |\n")
        lines.append("|---|" + "---|" * len(next(iter(ladder.values()), {})) + "\n")
        for scn, by_cell in ladder.items():
            cells_fmt = " | ".join(
                _fmt(v, pct=True) for v in by_cell.values()
            )
            lines.append(f"| `{scn}` | {cells_fmt} |\n")
    return "".join(lines)


def render_subset_validation(result) -> str:
    """Build the phase3 subset-validation markdown (pure; no I/O, no secrets)."""
    cfg = result.config
    lines: list[str] = []
    lines.append("# Phase 3-6 — Value+LowVol Subset Re-check + Cost Sensitivity "
                 "(factor groups × cost scenarios)\n")
    lines.append(
        f"Project: **{cfg.project.name}** · source: **{cfg.data.source}** · "
        f"ran in **{result.elapsed_seconds:.1f}s**\n"
    )
    lines.append(
        "\n> **This is NOT a return claim and not a tuned result.** It compares "
        "configured FACTOR GROUPS head-to-head on the same robustness matrix "
        "(same data, same rules, equal_weight vs walk-forward ic_weighted per "
        "group) and repeats every backtest under scaled trading-cost scenarios. "
        "POST-HOC SELECTION applies: the value+lowvol subset was chosen AFTER "
        "seeing the P3-5 results on these same windows — this quantifies "
        "RELATIVE robustness and cost sensitivity, not independent confirmation.\n"
    )

    lines.append("\n## Factor groups (compared head-to-head)\n")
    lines.append("| Group | n | factors |\n|---|---|---|\n")
    seen_groups: dict[str, tuple] = {}
    for cell in result.cells.values():
        for glabel, g in cell.groups.items():
            seen_groups.setdefault(glabel, tuple(g.get("factors", ())))
    for glabel, factors_t in seen_groups.items():
        flist = ", ".join(f"`{f}`" for f in factors_t)
        lines.append(f"| `{glabel}` | {len(factors_t)} | {flist} |\n")
    lines.append(
        "\n_Each group is re-processed independently from one shared raw factor "
        "panel (drop_missing applies per group), then run through the SAME "
        "equal_weight vs ic_weighted comparison and OOS slicing._\n"
    )

    lines.append("\n## Cost scenarios\n")
    lines.append("| Scenario | fee multiplier | effective fee_rate |\n|---|---|---|\n")
    multipliers = {
        s.label: s.fee_multiplier
        for s in (cfg.subset_validation.cost_scenarios if cfg.subset_validation else [])
    }
    for scn_label, fee in (result.scenario_fees or {}).items():
        base_tag = " (base)" if scn_label == result.base_scenario else ""
        mult = multipliers.get(scn_label, float("nan"))
        lines.append(f"| `{scn_label}`{base_tag} | {mult:g} | {fee:.6g} |\n")
    lines.append(
        "\n_Scenarios scale `cost.fee_rate` ONLY: scores and fills never see the "
        "fee, so trades and turnover are identical across scenarios — only the "
        "cost line (and net return) changes (locked by tests)._\n"
    )

    lines.append("\n## Cells\n")
    cell_samples = getattr(result, "cell_samples", None) or {}
    lines.append(
        "| Cell (universe \\| window) | sample | window | split | runtime |\n"
        "|---|---|---|---|---|\n"
    )
    for label, cell in result.cells.items():
        runtime = result.cell_runtimes.get(label, float("nan"))
        sample = cell_samples.get(label, "screened")
        lines.append(
            f"| `{label}` | {sample} | {_date_str(cell.train_start)} → "
            f"{_date_str(cell.test_end)} | {_date_str(cell.split_date)} | "
            f"{runtime:.0f}s |\n"
        )
    if result.skipped_cells:
        sk = ", ".join(f"`{s}`" for s in result.skipped_cells)
        lines.append(
            f"\n- **skipped cells (disclosed, runtime budget — coverage is "
            f"reduced, not hidden):** {sk}\n"
        )

    verdicts = getattr(result, "verdicts", None) or {}
    if verdicts:
        lines.append("\n## Independent holdout verdict\n")
        lines.append(
            "_Derived from INDEPENDENT cells ONLY (declared in "
            "`subset_validation.independent_cells`; their data took no part in "
            "factor screening — screened cells never enter this section). A "
            "hypothesis HOLDS iff the factor's mean IC carries the pre-declared "
            "expected sign in BOTH subperiods of the holdout cell (both postdate "
            "the screening). Settled rebalances below `min_rebalances` yield "
            "INSUFFICIENT-DATA. This is a factual IC sign check — NOT a return "
            "claim._\n"
        )
        for label, v in verdicts.items():
            lines.append(f"\n### `{label}` — **{v['status']}**\n")
            lines.append(
                f"- sample size: **{v['n_settled']}** settled rebalances "
                f"(train+test) vs required minimum **{v['min_rebalances']}**\n"
                f"- {v['reason']}\n\n"
            )
            lines.append(
                "| Hypothesis factor | expected sign | train IC | test IC | "
                "holds (train) | holds (test) | holds (BOTH) |\n"
                "|---|---|---|---|---|---|---|\n"
            )
            for name, f in (v.get("factors") or {}).items():
                lines.append(
                    f"| `{name}` | {f['expected']} | {_fmt(f['train_ic'])} | "
                    f"{_fmt(f['test_ic'])} | {'YES' if f['holds_train'] else 'NO'} | "
                    f"{'YES' if f['holds_test'] else 'NO'} | "
                    f"{'**YES**' if f['holds'] else 'NO'} |\n"
                )

    lines.append("\n## Cross-cell summary by group\n")
    sample_summaries = getattr(result, "sample_summaries", None) or {}
    if "independent" in sample_summaries:
        lines.append(
            "_Summaries are computed PER SAMPLE CLASS — independent holdout "
            "cells are never averaged with screened (post-hoc) cells._\n"
        )
        for cls, title in (("independent", "Independent holdout cells"),
                           ("screened", "Screened (post-hoc) cells")):
            if cls not in sample_summaries:
                continue
            cls_summary = sample_summaries[cls]
            lines.append(f"\n### {title}\n")
            lines.append(
                f"- cells aggregated: **{int(cls_summary.get('n_cells', 0))}**\n"
            )
            for glabel, gsum in (cls_summary.get("groups") or {}).items():
                lines.append(_subset_group_summary_block(glabel, gsum, level="####"))
    else:
        lines.append(
            f"- cells aggregated: **{int(result.summary.get('n_cells', 0))}**\n"
        )
        for glabel, gsum in (result.summary.get("groups") or {}).items():
            lines.append(_subset_group_summary_block(glabel, gsum))

    for label, cell in result.cells.items():
        lines.append(f"\n## Cell `{label}`\n")
        lines.append(
            f"- train: {_date_str(cell.train_start)} → {_date_str(cell.train_end)} "
            f"({cell.n_train_days}d) · test: {_date_str(cell.test_start)} → "
            f"{_date_str(cell.test_end)} ({cell.n_test_days}d) · split "
            f"{_date_str(cell.split_date)}\n"
        )
        if cell.boundary_dates:
            bd = ", ".join(_date_str(d) for d in cell.boundary_dates)
            lines.append(
                f"- boundary rebalance(s) excluded from both subperiods "
                f"(holding window straddles the split): {bd}\n"
            )
        lines.append(
            "\n### Raw factor IC (per-column, group-independent — the no-drift "
            "cross-check vs the P3-5 report)\n"
        )
        lines.append(_oos_ic_table(cell.raw_ic_stats, cell.raw_sign_consistency))
        for glabel, g in cell.groups.items():
            flist = ", ".join(f"`{f}`" for f in g.get("factors", ()))
            lines.append(f"\n### Group `{glabel}` — {flist}\n")
            lines.append("\n#### Subperiod performance × cost scenarios\n")
            lines.append(_subset_perf_table(g.get("performance") or {}))
            lines.append("\n#### Combo IC stability\n")
            lines.append(_oos_ic_table(
                g.get("combo_ic_stats") or {}, g.get("combo_sign_consistency") or {}
            ))
            flips = ", ".join(
                f"`{k}` {v}" for k, v in (g.get("sign_flips") or {}).items()
            )
            lines.append(
                f"\n#### Weight stability (ic_weighted)\n"
                f"- scored dates: **{g.get('n_scored', 0)}** · equal-weight "
                f"fallbacks: **{g.get('n_fallback', 0)}**\n"
                f"- sign flips between consecutive trained rebalance weights: "
                f"{flips or '_n/a_'}\n"
            )

    lines.append("\n## DOWNGRADES / caveats (INV-007 — must be disclosed)\n")
    run_labels = ", ".join(f"`{label}`" for label in result.cells)
    universes = sorted({label.split("|", 1)[0] for label in result.cells})
    windows = sorted({label.split("|", 1)[1] for label in result.cells})
    skipped = ", ".join(f"`{s}`" for s in result.skipped_cells) or "none"
    samples_note = "; ".join(
        f"`{label}` = {cell_samples.get(label, 'screened')}" for label in result.cells
    )
    lines.append(
        f"- MATRIX SCOPE: run cells: {run_labels}; skipped cells: {skipped}; "
        f"universes covered: {universes}; windows covered: {windows}; "
        f"sample classes: {samples_note}. "
        "Universe-specific disclosures below are the UNION over all run cells.\n"
    )
    seen: set[str] = set()
    for cell in result.cells.values():
        for item in cell.downgrades:
            if item not in seen:
                seen.add(item)
                lines.append(f"- {item}\n")
    lines.append(
        "- COMPARISON CAVEAT: groups are compared on the SAME overlapping "
        "windows the P3-5 candidates were screened on (POST-HOC selection); "
        "per-cell metrics remain SMALL-SAMPLE. The summary shows which group "
        "holds up RELATIVELY and how fast costs erode each — NOT a return "
        "claim, NOT independent confirmation.\n"
    )

    lines.append("\n## Artifacts\n")
    lines.append(f"- report: `{result.report_path}`\n- log: `{result.log_path}`\n")
    return "".join(lines)


def write_subset_validation_summary(result) -> Path:
    """Render and write the subset-validation report; return the path (SEC-003)."""
    target = result.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_subset_validation(result), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Repo-root delivery docs (framework spec §10). These describe the run; they
# carry no secrets and are regenerable.
# --------------------------------------------------------------------------- #
_BIAS_AUDIT_SECTIONS = (
    "未来函数 / lookahead",
    "PIT 成分股",
    "可交易过滤",
    "ann_date 财务对齐",
    "复权",
    "交易成本",
)


def render_bias_audit() -> str:
    """Build the BIAS_AUDIT.md markdown (Slice 13; all required sections)."""
    return (
        "# Bias Audit (Phase 0)\n\n"
        "本文件记录 P0 框架对各类偏差/未来函数的处理状态与降级。每个小节标注当前"
        "状态(已处理 / 降级 / 待办)。\n\n"
        "## 未来函数 / lookahead\n\n"
        "- 状态: **已处理(P0)**。\n"
        "- `momentum_20[t] = close[t] / close[t-window] - 1`,严格只用 t 及之前"
        "的收盘价(`groupby(symbol).shift(window)`)。\n"
        "- 事件顺序固定:在 t 收盘计算因子,t 收盘后调仓,从 t+1 持有。回测用"
        "**下一持有期**的收益结算,绝不使用因子已经看见的当日收益。\n"
        "- forward returns 只在 `analytics/` 计算,因子层永远拿不到未来收益"
        "(INV-001)。\n"
        "- **alpha 层 walk-forward 权重训练(P3-2,`alpha.model: ic_weighted`)**:"
        "alpha 层是**唯一**允许看 forward returns 的层,且只用于拟合因子权重——"
        "训练严格 walk-forward:对每个打分日 d,(factor[t], fwd_h[t]) 对只有在"
        "**已实现**(按交易日序 `t + h <= d`,h 日 forward return 在 t+h 才实现)"
        "时才进入 d 的权重;扰动任何未实现的 forward return 不改变 d 的权重"
        "(扰动测试锁定)。窗口 rolling(默认,保守)或 expanding;历史不足"
        "(任一因子有效已实现 IC < min_periods)→ 该日**退回等权**并计数披露;"
        "权重 L1 归一化、保留符号。固定配方、不调参,非收益声明。factors 层"
        "边界不变:forward returns 由 pipeline 在 alpha 边界计算、只传 "
        "`alpha.fit`,因子计算在其之前完成且永不接触。\n\n"
        "## PIT 成分股\n\n"
        "- 状态: **PIT 已实现(P1) / StaticUniverse 为离线降级**。\n"
        "- `PITIndexUniverse`(`universe.type=index`)用 tushare `index_weight` 的"
        "历史快照做 as-of 成分:`members(date)` 取 ≤date 的最近快照,绝不用未来快照"
        "(UNI-009)。被剔除的票在其在册期内仍是成员 —— 无幸存者偏差、无成分前视。\n"
        "- pipeline 构建 index universe 时会额外向回看 370 天成分快照,确保回测从两次"
        "成分调整中间开始时,起始日也能取到“开始日前最近快照”,而不是错误空仓。\n"
        "- 实证:沪深300 2024 全年 24 个快照、328 个不同成分(每快照 300),28 进"
        "28 出;`000069.SZ` 在 06-03 在册、06-28 已剔,各按其时代正确归属。\n"
        "- 数据坑:`index_weight` 单次约 6000 行上限,长窗口会**静默丢最早快照**;"
        "feed 已分 90 天窗口分页拉取规避。\n"
        "- `StaticUniverse`(`universe.type=static`,demo/离线用)成分与日期无关"
        "(UNI-003),是**降级**:存在幸存者 / 成分前视偏差,仅供无网络的 demo 跑通,"
        "并在 `phase0_summary.md` 的 DOWNGRADES 小节显式记录。\n\n"
        "## 可交易过滤\n\n"
        "- 状态: **停牌 / ST / 涨跌停已实现(P1)**。\n"
        "- `missing_close`(总是开):截面日 `close` 为 NaN 的标的不可交易(UNI-004)。\n"
        "- 统一在 `universe.filters.apply_tradable_filters` 按 `UniverseFilters` 开关"
        "执行;flag 由 `data.clean.tradability.enrich_tradability` 从 tushare "
        "`suspend_d` / `namechange` / `stk_limit` 富化到 panel(StaticUniverse 与 "
        "PITIndexUniverse 共用)。demo 无 flag 数据时各过滤自动 no-op。\n"
        "- **ST(UNI-006)**:`namechange` 名称区间含 'ST'/'*ST' 即标记,按 date 取"
        "生效名称(实证:`000005.SZ` 2024 全程 ST,正确剔除)。\n"
        "- **涨跌停(UNI-007)**:用**未复权 raw close** 与当日 raw `up_limit`/"
        "`down_limit` 比较,标记 `at_up_limit`/`at_down_limit`(qfq 复权价仅用于因子/"
        "回测收益;flag 富化在 front_adjust **之前**完成,故比较的是同口径 raw 价)。"
        "实证:`000005.SZ` 2024-02-01 触跌停。\n"
        "- **方向感知执行(UNI-007 / P2-2,已实现)**:选股层(`apply_tradable_filters`)"
        "与执行层(`runtime.fills.simulate_fills`)**拆分**。执行可行性按 panel flag 实时"
        "判定、与选股 toggle 无关:`at_up_limit` 挡**买入/加仓**,`at_down_limit` 挡"
        "**卖出/减仓**,`suspended`/缺收盘价双向挡。被挡的交易 **carry forward**(绝不"
        "强行成交不可能的单),现金一致 sell-then-buy(卖在前释放现金、买在后,现金不足"
        "按比例部分成交,**无杠杆**),换手/成本只算**实际成交**,闲置现金按 `cash_return` "
        "计息。demo 无 flag → 全可成交 → P0/P1 数字不变。每个调仓期的 blocked buys/sells/"
        "carried/executed turnover 记入回测 feasibility log,phase2 报告有专门小节。\n"
        "- **停牌(UNI-005)**:`suspend_d` 标记停牌日。**实测发现**:tushare 全天停牌"
        "当日**无 bar** → 已被 `missing_close` 剔除,故显式 suspended flag 与之重叠;"
        "其价值在盘中停牌(`suspend_timing`)或会给停牌日 bar 的数据源,属防御性。\n"
        "- 退市 / 无数据标的(如 `000003.SZ`)同样表现为不在 panel 而被剔除。PIT 历史"
        "成分见上节。\n"
        "- **`universe.min_listing_days`(UNI-008,P2-2)**:作为**买入/选股资格**过滤。"
        "**真实路径已执行**——从 tushare `stock_basic.list_date` 富化每只票上市日,某调仓日"
        "`age < min_listing_days` 的新上市标的剔除(边界 `age == min` 放行);**缺 list_date "
        "视为数据缺口,保留并披露**,绝不静默剔除。**demo 路径无上市日 → 仍 no-op(显式披露"
        "的降级)**,不伪造上市日。\n\n"
        "## ann_date 财务对齐\n\n"
        "- 状态: **已实现(P1)**。\n"
        "- 财务因子(`roe` / `netprofit_yoy`)经 `data.clean.pit_financials.asof_financials` "
        "按披露日 `ann_date` 做 backward as-of 对齐:每个 trade_date 只取 "
        "`ann_date <= trade_date` 的最近一期报告,**绝不按 `end_date`(报告期末)join**"
        "(DATA-012)。\n"
        "- 拉取窗口向回看约 16 个月(`start` 之前),确保回测 `start` 前已披露的上一期"
        "财报在集合内、能 as-of **carry forward** 到早期交易日,避免早期 NaN 缺口。\n"
        "- 实证:平安银行 2024 Q1(end_date 2024-03-31)披露日 ann_date 2024-04-20;"
        "as-of roe 在 04-19 仍是上一期年报值(10.2436),04-22 才切到 Q1(3.1176)——"
        "晚于报告期末约 3 周,证明无未来披露泄漏。\n"
        "- 财务因子仅在 tushare 数据路径可用;demo 无披露日,配置财务因子 + demo 源"
        "会报可读错误,**不伪造财务**。\n"
        "- **多因子(P3-1)**:多个财务字段(如 roe + netprofit_yoy)**一次 fetch、"
        "一次 as-of 对齐**(同一 `asof_financials` 调用,逐字段独立遵守 "
        "`ann_date <= trade_date`),无每因子重复拉取;财务字段可作为**被交易的"
        "因子**进入组合(不再只是诊断),报告按字段披露 TRADED vs diagnostic 角色"
        "与覆盖率。多因子合成是处理后(z-score/中性化)各列的**等权平均**"
        "(EqualWeightAlpha)——无 learned weights、不看 forward returns、不调参;"
        "`drop_missing` 要求该日该票**所有**启用因子齐备,缺任一因子即从该截面剔除"
        "(显式约定,绝不在部分数据上打分)。\n\n"
        "## 复权\n\n"
        "- 状态: **前复权已实现(P1)**。\n"
        "- panel 始终携带 `adj_factor` 列(DemoFeed 中恒为 1.0)。`data/clean/adjust.py` "
        "的 `front_adjust` 用 `adj_factor` 做前复权(qfq),在 pipeline 读盘后、因子"
        "计算前于内存中应用(DATA-003)。\n"
        "- 约定:按 symbol 锚定窗口内最新日 "
        "(`qfq = raw × adj_factor / adj_factor[latest]`)。锚定项在任何价格比值中"
        "约掉,故所有收益率 / 因子值对锚定与扩窗都不变 —— PanelStore 保持 raw"
        "(+adj_factor),复权在内存做,batch≡incremental 一致。\n"
        "- 实证:平安银行 2024-06-14 除权,raw 当日 -5.74%(分红跳空),qfq +0.99%"
        "(真实涨跌);momentum_20 因此最多变动 6.77pp。demo(adj=1.0)下为恒等。\n\n"
        "## 交易成本\n\n"
        "- 状态: **已处理(P0)**。\n"
        "- 成本 = L1 换手 × `fee_rate`;`turnover = sum(|target_w - current_w|)`,"
        "在 symbol 并集上对齐计算。\n"
        "- 每个调仓期 `net_return = gross_return - cost`,成本拖累在"
        "`phase0_summary.md` 中汇总(BT-004)。slippage 参数已预留。\n"
        "- **结算价缺失约定(P0 降级)**:若持仓标的在持有期末(end)的 `close` "
        "为 NaN(停牌 / 缺数据),回测以 0.0(持平)记其该期收益,而非剔除或用"
        "最近可得价结算。该约定在此显式披露(INV-007);P1 接入真实停牌/退市"
        "处理后改进结算逻辑。\n\n"
        "## 中性化\n\n"
        "- 状态: **行业 + 市值中性化已实现(P1)**。\n"
        "- `factors.process.neutralize.neutralize_by_date`:每个 date 截面把因子对 "
        "`[log(market_cap), one-hot(industry)]` 做 OLS,取残差,移除规模与行业暴露。"
        "缺行业 / 市值,或**残差自由度 ≤ 0**(名称数 ≤ 1+行业数,饱和拟合会给出无意义"
        "的伪 0 残差)时返回 **NaN**,绝不静默乱算;`processing.neutralize` 开启但协变量"
        "缺失(如 demo 路径)直接报可读错误。\n"
        "- 实证:12 只票横跨 4 行业(2024-09-30),corr(原始 momentum, log市值) "
        "= -0.617 → 中性化后 -0.000,各行业残差均值 ≈ 0,确认规模/行业暴露被移除。\n"
        "- **行业 PIT(UNI-010,P2-3,已实现)**:行业协变量从 `stock_basic.industry` 的"
        "**当前**标签,升级为按 trade_date **as-of** 的历史申万行业。`data.feed."
        "tushare_covariates.pit_sw_intervals(symbols, level)` 读 `index_member_all` 拿每股 "
        "SW 行业的 `in_date`/`out_date` 区间;`data.clean.pit_industry.asof_industry` 按"
        "`[in_date, out_date)` 覆盖该日的区间取行业(改分类日**新行业**生效,PIT-safe,绝不用"
        "未来行业)。起始日前已有的成分能 carry forward 到窗口开始。\n"
        "- **SW 层级可配置**:`processing.neutralize.industry_level`(L1/L2/L3,**默认 L1**"
        "=31 宽板块,行业中性化业界标准,小截面自由度更稳)。**实测**:旧 tag 年化 −17.6%、"
        "SW-L1 −10.2%、SW-L2 −9.3% —— L1≈L2,−17.6→−10 的大跳是 **tushare→SW 分类切换**"
        "(补 PIT 的必然代价:只有 SW 有 in/out 历史可 PIT 化,旧 tag 无法),**与粒度无关**;"
        "报告披露实际 level 与覆盖率。\n"
        "- **缺失处理(不静默退回 current)**:无 SW 历史的票 → 行业 **NaN**,被 neutralize "
        "按截面丢弃;每次运行的 **PIT 行业覆盖率**在 `phase2_real_baseline.md` 披露。市值 "
        "`daily_basic.total_mv` 为逐日真值。\n"
    )


def write_bias_audit(repo_root: Path) -> Path:
    """Write BIAS_AUDIT.md at the repo root and return its path."""
    target = Path(repo_root) / "BIAS_AUDIT.md"
    target.write_text(render_bias_audit(), encoding="utf-8")
    return target


def bias_audit_required_sections() -> tuple[str, ...]:
    """The section titles the bias audit must contain (Slice 13 contract)."""
    return _BIAS_AUDIT_SECTIONS
