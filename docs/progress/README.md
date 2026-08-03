# 进度档案索引

仓库根 `CLAUDE.md`（= `AGENTS.md`）曾内联全部阶段日志（~78k tokens，每个会话都被完整加载）。
2026-07-30 把历史搬到本目录，**逐字搬移、一字未改**；`CLAUDE.md` 只留每个会话都需要的东西
（定位 / 架构 / 不变量 / 环境 / secret 纪律 / 开发约定 / 当前状态 / 方法论 / 本索引）。

## 怎么用

- **不要整读。** 这些文件按阶段切分，用 `grep` 定位：`grep -rn "关键词" docs/progress/`。
- 找某个 PR 干了什么：先查 [`pr_ledger.md`](pr_ledger.md)（PR 号 → 一句话内容），再到对应档案里 grep。
- 每条记录都保留了**真实 run 的数字、被推翻的假设、评审揪出的缺陷、以及自己犯过的错**——
  这些负面记录是刻意留下的，**不要因为"看起来不好看"就删**。
- 惯例：**文档是地图，`git log` / `gh pr list` 是地形**；冲突以后者为准。

## 档案

| 文件 | 覆盖范围 | PR |
|---|---|---|
| [`pr_ledger.md`](pr_ledger.md) | 全部 PR 编号 → 内容映射（含 **#38 保持 OPEN** 的说明） | #1–#107 |
| [`01_phase0_3_daily_pipeline.md`](01_phase0_3_daily_pipeline.md) | 7 层骨架 / Phase 0 MVP / Phase 1 偏差边界（复权·PIT 成分·可交易·`ann_date`·中性化）/ Phase 2 真实基准与执行真实性与 PIT 行业与标准分析 / Phase 3 多因子 baseline·walk-forward IC 加权·OOS·robustness matrix·候选因子组·子集复检与成本敏感性·独立样本验证·CSI500 泛化 | #1–#21 |
| [`02_phase4_cache_and_intraday_i1_i4.md`](02_phase4_cache_and_intraday_i1_i4.md) | 端点级持久化 raw 缓存（行情 / universe+tradability / 因子支撑端点）+ `data-update` 增量暖跑；分钟 pipeline I1–I4（1min feed·月分区缓存·PIT 日频聚合·尾盘执行骨架） | #23–#31 |
| [`03_intraday_i5.md`](03_intraday_i5.md) | I5a 事件驱动回测重构 / I5b 执行期涨跌停 / I5c MMP 分钟因子 / I5d 五分位 / I5e CSI300 泛化（负结果）/ I5f 流动性诊断 | #33–#55 |
| [`04_data_layer.md`](04_data_layer.md) | D1 契约文档 + token 去重 / D2 cache 拆分 / D3+D3b report-only 质量层 / D4 ledger 批量写 / D5 并发+全局限频器 / schema 注册表与 drift 守卫 / 全A 增量 auto-warm 与历史回填 | #41–#61 |
| [`05_minute_factors_eval_contract.md`](05_minute_factors_eval_contract.md) | `analytics/eval/` 三轴判决契约（v0.1→v0.9，含两处判定门缺陷修复）+ 十一个分钟因子复现总表与三条循环级结论 | #63–#74 |
| [`06_execution_basis_and_corrections.md`](06_execution_basis_and_corrections.md) | 尾盘成交价改 bar VWAP / 分钟持有期收益复权 / 报告披露修正 / `adj_factor` 下降检查 / 日频除权正确性审计（GENUINE）/ I5d·I5e·I5f 在修正引擎上重跑 / 十一因子改 14:51 VWAP exec-to-exec 基准 / 闸门描述单一来源 | #75–#82 |
| [`07_factor_layer_refactor.md`](07_factor_layer_refactor.md) | 因子层深度重构 D0 纸面契约 → D1 registry+面板冻结 → D2 primitives → D3 store → D4/D4b/D4c → D5a/D5b/C4/C4b，含每一步的评审发现与实施期修订 A1/A2/A3 | #84–#107 |
| [`08_gates_disclosures_roadmap.md`](08_gates_disclosures_roadmap.md) | 归档时点（`main` @ `c9cc12e`）的质量门快照、完整披露清单、路线图原文 | — |

## 未在 git 里的权威文档（只在本机）

`.gitignore` 排除了 `tmp/`（仅保留 `tmp/framework/architecture.html`），所以下面这些**不在版本控制里**：

- `archive/tmp/design/HANDOFF_2026-07-30_claude_code.md` —— 当前唯一有效的执行交接（重构 D5 C5；2026-08-03 随 PR #137 归档进 `archive/`）
- `archive/tmp/context/cc_handoff_20260730_d5_c5/HANDOFF.md` —— F1–F5 的逐格实测证据（同上，已归档）
- `tmp/design/factor_refactor_design_v3.md` —— 重构设计权威（红线 #1–#11 / 四腿对账 / D0–D7）
- `tmp/design/RESULTS_post_pr75_2026-07-21.md`、`tmp/design/FINDING_adj_factor_seam_2026-07-21.md` —— 修正引擎重跑与除权审计的原始记录
