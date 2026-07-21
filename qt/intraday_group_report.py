"""Markdown report + PNG figures for the I5d MMP quintile grouped backtest.

Kept in a sibling module (cohesion / small files) and decoupled from the run
logic: every function takes already-computed results and only READS them, so there
is no import cycle with :mod:`qt.intraday_group_backtest`. The report makes the
exploratory framing, the Q1/Q5 direction, the ``fee_rate`` and the cache-only
minute provenance explicit, and embeds the three required figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qt.intraday_group_figures import (
    METRIC_COLUMNS,
    plot_group_metrics,
    plot_quintile_nav,
    plot_spread_curve,
)


def _write_figures(
    figure_dir: Path,
    groups: tuple,
    per_period: pd.Series,
    cumulative: pd.Series,
    n_groups: int,
) -> dict[str, Path]:
    """Render the three required PNGs (NAV curves / spread / metric bars)."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    nav_by_group = {g.group: g.nav_table for g in groups}
    metrics = pd.DataFrame(
        {g.group: {c: g.metrics.get(c, float("nan")) for c in METRIC_COLUMNS}
         for g in groups}
    ).T
    nav_path = plot_quintile_nav(
        nav_by_group, figure_dir / "mmp_quintile_nav.png", n_groups
    )
    spread_path = plot_spread_curve(
        cumulative,
        figure_dir / f"mmp_q{n_groups}_minus_q1_spread.png",
        low_label="Q1",
        high_label=f"Q{n_groups}",
    )
    metrics_path = plot_group_metrics(
        metrics, figure_dir / "mmp_quintile_metrics.png", n_groups
    )
    return {"nav": nav_path, "spread": spread_path, "metrics": metrics_path}


def _fmt(v: float, pct: bool = False, nd: int = 4) -> str:
    """Format a float, NaN-safe; ``pct`` renders as a percentage."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NaN"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{nd}f}"


def _intro_lines(result) -> list[str]:
    """Title + exploratory framing + study design."""
    cfg = result.config
    title = cfg.output.intraday_report_title or "Phase I5d — MMP Quintile 5Y Group Backtest"
    t = title.strip()
    h1 = t if t.startswith("#") else f"# {t}"
    lines = [
        h1,
        "",
        "**This is an EXPLORATORY grouped factor analysis of the I5c MMP "
        "minute-derived daily score — NOT a performance claim and NOT parameter "
        "tuning.** On each monthly rebalance date the cross-section is ranked by the "
        "PIT-safe daily score and split into equal-count quantile groups; each group "
        "is run as its own long-only equal-weight portfolio through the SAME "
        "event-driven intraday-tail model (14:50 decision / 14:51 execution, "
        "exec-to-exec returns) with I5b raw `stk_limit` execution feasibility ON and "
        "trading cost `fee_rate` applied. No parameter, group count, cost, universe, "
        "or window was tuned from results.",
        "",
        "## Study design",
        "",
        f"- window: `{cfg.data.start}` → `{cfg.data.end}`",
        f"- universe: `{cfg.universe.type}` `{cfg.universe.index_code or ''}` "
        f"(requested {result.requested_symbols} distinct constituents)",
        f"- rebalance: monthly — {result.rebalance_count} settled periods",
        f"- score column: `{result.score_feature}` (key=`{result.score_feature_key}`)",
        f"- groups: `analytics.quantiles={result.n_groups}` equal-count rank buckets, "
        "**Q1 = lowest MMP score, "
        f"Q{result.n_groups} = highest**",
        f"- decision_time `{cfg.intraday.decision_time}` / execution_window "
        f"`[{cfg.intraday.execution_window[0]}, {cfg.intraday.execution_window[1]}]`; "
        "returns are execution-to-execution, NEVER close-to-close",
        f"- execution_price_basis: `{cfg.intraday.execution_price_basis}` "
        "(`bar_vwap` = the selected 1min bar's amount/volume, RAW unadjusted; "
        "`bar_close` = that bar's single closing tick)",
        f"- trading cost: `cost.fee_rate={cfg.cost.fee_rate}`, "
        f"`slippage_rate={cfg.cost.slippage_rate}` (cost line = turnover × fee_rate "
        "inside SimExecution; no extra ad-hoc cost layer)",
        "",
    ]
    return lines


def _mmp_and_grouping_lines(result) -> list[str]:
    return [
        "## MMP factor & group assignment",
        "",
        "Per 1min bar `t`: `mid=(high+low)/2`; `S=(close-mid)/mid`; "
        "`V=sqrt(volume/median(vol[t-20:t]))`; `B=|close-open|/(high-low+eps)`; "
        "`R=(high-low)/(mean(hl[t-20:t])+eps)`; **`MMP_t = S*V*B*R`** (`eps=1e-6`). "
        "The daily score `intraday_mmp20_ew_0930_1450` is the EQUAL-WEIGHT mean of "
        "valid `MMP_t` over the in-session bars visible at the 14:50 cutoff "
        "(`[session_open, decision_time]`); rolling baselines use only the prior 20 "
        "in-session bars (first 20 are NaN). The MMP math is UNCHANGED from I5c.",
        "- **group rule**: drop NaN/non-finite scores, sort by `(score asc, symbol "
        "asc)`, split BY RANK into equal-count buckets. `Q1` = lowest score, "
        f"`Q{result.n_groups}` = highest. Equal-count (not value-cut) buckets keep "
        "tied/degenerate scores deterministic; a date with too few valid names "
        "leaves the high groups empty (no crash).",
        "- **no lookahead**: grouping uses only the daily PIT universe ∩ "
        "minute-covered names ∩ valid MMP score; forward returns NEVER enter the "
        "assignment.",
        "",
    ]


def _provenance_lines(result) -> list[str]:
    lines = [
        "## Minute-cache coverage & data provenance",
        "",
        f"- requested symbols: {result.requested_symbols}; minute-cache fully "
        f"covered: {result.covered_symbols}; excluded (uncovered): "
        f"{len(result.uncovered_symbols)}",
        f"- anchor dates requiring minute bars: {result.anchor_dates} (rebalance ∪ "
        "exit dates only — the full multi-year minute history is never loaded)",
        f"- raw rows loaded: {result.raw_rows}; normalized rows used: "
        f"{result.normalized_rows}",
        f"- **stk_mins live API calls during this run: {result.minute_live_calls}** "
        "(cache-only, read-only; a miss is never a silent warm/backfill)",
        f"- elapsed: {result.elapsed:.1f}s",
    ]
    if result.uncovered_symbols:
        shown = ", ".join(result.uncovered_symbols[:20])
        more = (
            "" if len(result.uncovered_symbols) <= 20
            else f" (+{len(result.uncovered_symbols) - 20} more)"
        )
        lines.append(
            f"- **excluded (uncovered) symbols** ({len(result.uncovered_symbols)}): "
            f"{shown}{more}"
        )
        lines.append(
            "  - ⚠️ dropping uncovered names trades full-window completion for a "
            "potential coverage bias (the realized cross-section omits names with "
            "no cached minute history). Disclosed, not silent; the 5-year WINDOW is "
            "NOT shortened."
        )
    sc = result.score_coverage
    lines.append(
        f"- daily MMP score panel: {sc['rows']} (date,symbol) rows; valid "
        f"{sc['valid']}; NaN {sc['nan']}."
    )
    lines.append("")
    return lines


def _feasibility_lines(result) -> list[str]:
    cfg = result.config
    lines = ["## Execution-time price-limit feasibility (I5b)", ""]
    if not result.price_limit_check:
        lines.append(
            "- **disabled**: feasibility is the base bar-exists rule only "
            "(missing/NaN execution bar blocks both directions)."
        )
        lines.append("")
        return lines
    cov = result.limit_coverage
    lines.extend([
        "- **enabled** (`intraday.price_limit_check=true`): a buy is blocked at the "
        "raw upper limit and a sell at the raw lower limit, comparing the selected "
        "execution-minute **raw** 1min close to the raw `stk_limit` band "
        "(RAW-vs-RAW; never qfq / daily close / a daily-close-derived flag).",
        f"- limit tolerance: `{cfg.intraday.limit_tolerance}`; "
        f"require_price_limit_coverage: `{cfg.intraday.require_price_limit_coverage}`",
        f"- limit coverage over rebalance anchors (shared across groups): required "
        f"{cov.get('required', 0)} (date, symbol) pairs; present "
        f"{cov.get('present', 0)}; missing {cov.get('missing', 0)}",
        f"- **stk_limit cache gap-fetches this run: {result.stk_limit_gap_fetches}** "
        "(read through the existing P4 daily cache — never a minute/stk_mins fetch).",
        "",
        "Per-group blocked buy/sell counts (a fresh model per group, so counts do "
        "NOT double-count across groups):",
        "",
        "| group | up-limit blocked buys | down-limit blocked sells | unchecked limit rows |",
        "|---|---|---|---|",
    ])
    for g in result.groups:
        lines.append(
            f"| Q{g.group} | {g.up_limit_blocked_buys} | "
            f"{g.down_limit_blocked_sells} | {g.missing_limit_rows} |"
        )
    lines.append("")
    return lines


def _performance_lines(result) -> list[str]:
    lines = [
        "## Per-group NAV / performance",
        "",
        "| group | final NAV | annual | vol | Sharpe | maxDD | mean turnover | "
        "total cost | avg holdings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in result.groups:
        m = g.metrics
        tag = " (low)" if g.group == 1 else (" (high)" if g.group == result.n_groups else "")
        lines.append(
            f"| Q{g.group}{tag} | {_fmt(m['final_nav'])} | "
            f"{_fmt(m['annual_return'], pct=True)} | {_fmt(m['volatility'], pct=True)} "
            f"| {_fmt(m['sharpe'])} | {_fmt(m['max_drawdown'], pct=True)} | "
            f"{_fmt(m['mean_turnover'])} | {_fmt(m['total_cost'])} | "
            f"{_fmt(m['avg_holdings'], nd=1)} |"
        )
    lines.append("")
    lines.append(
        "- turnover/cost count the ACHIEVED book after feasible fills (price-limit "
        "blocked + missing-bar blocked names earn nothing, never a daily-close "
        "fallback); idle cash earns `cash_return`."
    )
    lines.append("")
    return lines


def _spread_lines(result) -> list[str]:
    n = result.n_groups
    s = result.spread_summary
    mono = result.monotonicity
    lines = [
        f"## Q{n}−Q1 synthetic spread & monotonicity (report-only)",
        "",
        f"The Q{n}−Q1 spread is the per-period difference of the Q{n} (high) and Q1 "
        "(low) group NET returns; the cumulative curve is their running sum. It is a "
        "**synthetic long-only leg difference, NOT a separately executed "
        "dollar-neutral portfolio** (no long-short execution model is run).",
    ]
    if s:
        lines.append(
            f"- mean per-period Q{n}−Q1 net return: {_fmt(s.get('mean_per_period'), pct=True)} "
            f"over {int(s.get('n_periods', 0))} periods; cumulative (sum): "
            f"{_fmt(s.get('total'), pct=True)}."
        )
    if mono:
        lines.append(
            f"- group monotonicity (Spearman of group index 1..{n} vs metric): "
            f"annual return {_fmt(mono.get('annual_spearman'))}, final NAV "
            f"{_fmt(mono.get('final_nav_spearman'))} "
            "(+1 = perfectly increasing Q1→QN, −1 = perfectly decreasing)."
        )
    lines.append(
        "- **report-only**: returns are read here for analytics only and never feed "
        "the factor/alpha; a single overlapping window is far too little to infer "
        "factor quality."
    )
    lines.append("")
    return lines


def _per_date_lines(result) -> list[str]:
    n = result.n_groups
    header = "| date | n_scored | " + " | ".join(f"Q{i}" for i in range(1, n + 1)) + \
        " | score mean | std | p10 | p50 | p90 |"
    sep = "|---|---|" + "---|" * n + "---|---|---|---|---|"
    lines = [
        "## Per-rebalance group sizes & score distribution",
        "",
        header,
        sep,
    ]
    for r in result.per_date_rows:
        sizes = " | ".join(str(x) for x in r["sizes"])
        lines.append(
            f"| {pd.Timestamp(r['date']).date()} | {r['n_scored']} | {sizes} | "
            f"{_fmt(r['mean'], nd=6)} | {_fmt(r['std'], nd=6)} | {_fmt(r['p10'], nd=6)} "
            f"| {_fmt(r['p50'], nd=6)} | {_fmt(r['p90'], nd=6)} |"
        )
    lines.append("")
    return lines


def _figure_lines(result) -> list[str]:
    report_dir = result.report_path.parent
    lines = ["## Figures", ""]
    captions = {
        "nav": "NAV curves for each quantile group (Q1 = lowest MMP score, "
               f"Q{result.n_groups} = highest).",
        "spread": f"Cumulative synthetic Q{result.n_groups}−Q1 net-return spread "
                  "(long-only leg difference).",
        "metrics": "Grouped bars: annualized return, max drawdown, mean turnover, "
                   "total cost per group.",
    }
    for key in ("nav", "spread", "metrics"):
        path = result.figure_paths.get(key)
        if path is None:
            continue
        rel = Path(path).relative_to(report_dir).as_posix()
        lines.append(f"**{captions[key]}**")
        lines.append("")
        lines.append(f"![{key}]({rel})")
        lines.append("")
    return lines


def _limitations_lines(result) -> list[str]:
    return [
        "## Limitations (explicit)",
        "",
        "- **EXPLORATORY grouped factor analysis, NOT a performance claim**: one "
        "factor, one overlapping 5-year window, one universe. No tuning, no "
        "robustness matrix, no learned / IC-weighted alpha.",
        "- **coverage bias**: uncovered minute names are dropped (disclosed above); "
        "the realized cross-section is the covered subset.",
        "- **execution feasibility** models price-limit + bar-existence only; no "
        "partial-fill / liquidity / volume cap at the execution minute. Suspended "
        "names have no minute bar and are blocked by the missing-bar rule; explicit "
        "ST status is not consulted at 14:50.",
        f"- **Q{result.n_groups}−Q1 spread is synthetic** (a long-only leg "
        "difference), not a separately executed dollar-neutral book.",
        "",
    ]


def _write_report(result) -> None:
    """Write the I5d grouped-backtest markdown report (with embedded figures)."""
    path = result.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += _intro_lines(result)
    lines += _mmp_and_grouping_lines(result)
    lines += _provenance_lines(result)
    lines += _performance_lines(result)
    lines += _spread_lines(result)
    lines += _feasibility_lines(result)
    lines += _per_date_lines(result)
    lines += _figure_lines(result)
    lines += _limitations_lines(result)
    lines.append(f"_elapsed: {result.elapsed:.1f}s_")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
