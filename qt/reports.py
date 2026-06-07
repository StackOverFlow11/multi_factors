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
        f"- factor: `{result.factor_name}`\n"
        f"- alpha: `{cfg.alpha.model}`\n"
        f"- portfolio: `{cfg.portfolio.constructor}`, top_n=`{cfg.portfolio.top_n}`\n"
        f"- backtest: rebalance=`{cfg.backtest.rebalance}`, "
        f"event_order=`{cfg.backtest.event_order}`\n"
        f"- cost: fee_rate=`{cfg.cost.fee_rate}`, slippage=`{cfg.cost.slippage_rate}`\n"
    )

    lines.append("## Data shape\n")
    lines.append(
        f"- panel rows: **{result.panel_rows}**\n"
        f"- symbols: **{result.panel_symbols}**\n"
    )

    lines.append("## Factor IC\n")
    lines.append(
        f"- IC mean: **{_fmt(result.ic_mean)}**\n"
        f"- IC_IR (mean/std): **{_fmt(result.ic_ir)}**\n"
    )

    lines.append("## Quantile returns\n")
    lines.append(_quantile_table(result.quantile_returns) + "\n")

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
        "(INV-001)。\n\n"
        "## PIT 成分股\n\n"
        "- 状态: **降级(P0)**。\n"
        "- 使用 `StaticUniverse`:成分与日期无关,返回配置中的固定 symbol 列表,"
        "**不是**真正的 point-in-time 历史成分(UNI-003)。\n"
        "- 因此存在幸存者偏差 / 成分前视偏差。该降级是 P0 有意为之,并在"
        "`phase0_summary.md` 的 DOWNGRADES 小节显式记录。P1 接入"
        "`index_weight` 历史成分后修复。\n\n"
        "## 可交易过滤\n\n"
        "- 状态: **部分处理(P0)**。\n"
        "- P0 仅实现 `missing_close` 过滤:截面日 `close` 为 NaN 的标的不可交易"
        "(UNI-004)。\n"
        "- 停牌 / ST / 涨跌停过滤接口已预留(`UniverseFilters`),但 P0 未实现"
        "(降级),P1 补齐。\n"
        "- `universe.min_listing_days` 已在配置中(默认 60),但 P0 **未执行**"
        "(no-op,降级):新上市标的不会被剔除。该空操作在此显式披露(INV-007),"
        "P1 接入上市日期后强制执行。\n\n"
        "## ann_date 财务对齐\n\n"
        "- 状态: **不适用 / 待办(P0)**。\n"
        "- P0 只用行情动量因子,不使用任何财务数据,因此本期不存在财务前视风险。\n"
        "- 一旦引入财务因子,财务特征必须按披露日 `ann_date` 对齐,不能早于披露日"
        "出现(DATA-012,P1)。\n\n"
        "## 复权\n\n"
        "- 状态: **保留 adj_factor(P0)**。\n"
        "- panel 始终携带 `adj_factor` 列(DemoFeed 中恒为 1.0)。P0 未做完整前复权"
        "重算价格序列;真实 tushare 接入时需用 `adj_factor` 做前复权"
        "(DATA-003)。该降级在报告中说明。\n\n"
        "## 交易成本\n\n"
        "- 状态: **已处理(P0)**。\n"
        "- 成本 = L1 换手 × `fee_rate`;`turnover = sum(|target_w - current_w|)`,"
        "在 symbol 并集上对齐计算。\n"
        "- 每个调仓期 `net_return = gross_return - cost`,成本拖累在"
        "`phase0_summary.md` 中汇总(BT-004)。slippage 参数已预留。\n"
        "- **结算价缺失约定(P0 降级)**:若持仓标的在持有期末(end)的 `close` "
        "为 NaN(停牌 / 缺数据),回测以 0.0(持平)记其该期收益,而非剔除或用"
        "最近可得价结算。该约定在此显式披露(INV-007);P1 接入真实停牌/退市"
        "处理后改进结算逻辑。\n"
    )


def write_bias_audit(repo_root: Path) -> Path:
    """Write BIAS_AUDIT.md at the repo root and return its path."""
    target = Path(repo_root) / "BIAS_AUDIT.md"
    target.write_text(render_bias_audit(), encoding="utf-8")
    return target


def bias_audit_required_sections() -> tuple[str, ...]:
    """The section titles the bias audit must contain (Slice 13 contract)."""
    return _BIAS_AUDIT_SECTIONS
