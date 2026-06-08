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
        "实证:`000005.SZ` 2024-02-01 触跌停。当前选股层对两个方向都剔除;**方向感知**"
        "(买入只看涨停、持有跌停不强卖)属执行层,后续细化。\n"
        "- **停牌(UNI-005)**:`suspend_d` 标记停牌日。**实测发现**:tushare 全天停牌"
        "当日**无 bar** → 已被 `missing_close` 剔除,故显式 suspended flag 与之重叠;"
        "其价值在盘中停牌(`suspend_timing`)或会给停牌日 bar 的数据源,属防御性。\n"
        "- 退市 / 无数据标的(如 `000003.SZ`)同样表现为不在 panel 而被剔除。PIT 历史"
        "成分见上节。\n"
        "- `universe.min_listing_days` 已在配置中(默认 60),但仍 **未执行**(no-op,"
        "降级):新上市标的不会被剔除。显式披露(INV-007),后续接上市日期后强制。\n\n"
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
        "会报可读错误,**不伪造财务**。\n\n"
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
        "- **降级**:行业来自 `stock_basic.industry` 的**当前**行业标签,非按历史时点,"
        "故行业中性化带有轻微成分前视(市值 `daily_basic.total_mv` 为逐日真值)。"
        "PIT 行业历史是后续项,此降级在此显式披露(INV-007)。\n"
    )


def write_bias_audit(repo_root: Path) -> Path:
    """Write BIAS_AUDIT.md at the repo root and return its path."""
    target = Path(repo_root) / "BIAS_AUDIT.md"
    target.write_text(render_bias_audit(), encoding="utf-8")
    return target


def bias_audit_required_sections() -> tuple[str, ...]:
    """The section titles the bias audit must contain (Slice 13 contract)."""
    return _BIAS_AUDIT_SECTIONS
