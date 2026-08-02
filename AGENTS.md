# Quantitative_Trading — A股截面多因子框架

## 项目定位
- A股 **截面多因子选股**（cross-sectional multi-factor）框架。**不做择时**。
- 路径：研究/回测优先 → 渐进到实盘。
- 构建方式：成熟工具 + **自建因子层**为核心。
- 完整架构设计见 [`tmp/framework/architecture.html`](tmp/framework/architecture.html)。本文件是精简操作版，**细节以该文档为准**。

## 架构：7 层
```
data → universe → factors(特征) → alpha(合成/预测) → portfolio(+risk约束) → runtime(backtest|live) → analytics
```

| 层 | 职责 |
|---|---|
| `data/` | 采集(feed) / 清洗复权·对齐披露日(clean) / 面板存储(store) |
| `universe/` | PIT 成分股 + 可交易过滤(停牌/涨跌停/ST) |
| `factors/` | 截面因子 计算(compute) / 预处理三件套(process) / 落盘(store) |
| `alpha/` | 多因子合成/预测（独立成层） |
| `portfolio/` | 组合构建 + 风控**事前约束**(construct.py / risk.py) |
| `runtime/` | 回测 与 实盘 统一：driver + execution，两套实现(backtest/live) |
| `analytics/` | alphalens(因子检验) + quantstats(组合绩效) |

## 不可违反的设计不变量（写代码必须守）
1. **factors 永不碰未来收益**；只有 `alpha` 层才用未来收益拟合权重（防未来函数边界）。
2. **回测即实盘**：`runtime` 的 backtest/live 是同接口两实现；`factors/alpha/portfolio` 层两边复用、一行不改。
3. **分层解耦**：factor 不碰数据源；portfolio 不碰下单。坏味道：策略里直接 `pro.daily(...)`。
4. 决策是**截面的**（每个调仓日横向排序选股）；时序只用于算因子。

## 致命陷阱（correctness 红线，详见 architecture.html §8）
未来函数 · PIT 成分股 · 可交易过滤 · 财务按 `ann_date` 披露日对齐 · 前复权 · batch≡incremental 一致性 · 行业+市值中性化 · 交易成本/换手 · 过拟合(样本内外+IC稳定性)。

## 环境
- **框架/测试环境**：conda `quant_mf`（Py 3.12）。运行 Phase 0、pytest、ruff、CLI 时用绝对路径 python：
  `/home/shaofl/Development/env_tools/envs/quant_mf/bin/python`
- **数据拉取环境**：conda `data_fetch`（Py 3.12）。仅用于独立数据抓取/交互查数；在非交互 shell 里**用绝对路径 python**，不靠 activate：
  `/home/shaofl/Development/env_tools/envs/data_fetch/bin/python`

## 数据：tushare
- **token**：`/home/shaofl/Projects/financial_projects/.config.json`（key `tushare.token`）。
  ⚠️ **绝不打印、绝不写进 repo、绝不 commit。** 代码里从该文件读取，不硬编码。
- **权限**：实测充足——个股日线 / 分钟(`stk_mins`) / 复权(`adj_factor`) / 成分股(`index_weight`) / 申万行业(`index_classify`) / 财务含`ann_date`(`income`,`fina_indicator`) 全可取。分钟级可直接上，无需先退回日线。
- **MCP（可选开发工具）**：`financial_projects/.mcp.json` 有 tushare MCP，仅供开发期交互查数；**从 `financial_projects/` 启动 claude 才加载**。
- **数据层 ETL 一律用 Python SDK（批量/增量），不要建在 MCP 上。** 注意 tushare 各接口有每分钟调用上限，批量拉取需限流+重试。

## 技术选型
| 用途 | 选型 |
|---|---|
| 数据处理 | pandas（+ polars 可选）· numpy |
| 存储 | parquet（分钟级按 symbol/year 分区）· DuckDB |
| 因子检验 | alphalens-reloaded |
| 绩效 | quantstats |
| 回归/合成 | statsmodels · scikit-learn |
| 组合优化(后期) | cvxpy · riskfolio-lib |
| 实盘下单(后期) | vnpy / miniqmt(QMT) —— **仅作下单通道，不当回测引擎** |
| 配置/测试 | pydantic-settings + YAML · pytest |

## 开发约定
- **交流中文**；代码/注释/commit message 用**英文**。
- **Git**：feature 分支 + PR；commit 用 conventional 格式、**无 attribution**（不加 Co-Authored-By）。**代码 PR 与进度文档 PR 分开**，实现 PR 绝不碰 `CLAUDE.md` / `AGENTS.md`；改完 `CLAUDE.md` 必须 `cp CLAUDE.md AGENTS.md` 并 `cmp` 验证逐字节相同。仓库**无 CI**，合并用 `gh pr merge N --merge`。历史 PR 台账（#1–#107）见 [`docs/progress/pr_ledger.md`](docs/progress/pr_ledger.md)；**PR #38 永远保持 OPEN、不许动**。
- **不过度设计**：按路线图 MVP 先打通一条端到端链路，再加层（architecture.html §11，Phase 0→3）。
- **secrets** 一律走外部 `.config.json`；repo `.gitignore` 已排除数据产物(`*.parquet`等)、缓存、`tmp/`（仅留架构文档）。
- 文件小而专（<800 行），immutable 优先。

## 当前状态（截面写于 2026-08-02，均为实跑验证）

| 项 | 值 |
|---|---|
| `main` | 下表全部 gate 实测于 `bb59fdf`（PR #127） |
| `pytest -p no:warnings` | **2796 passed + 1 skipped**（主 checkout 口径；⚠️ 不要再传 `-q`：`pyproject.toml` 已含 `addopts="-q"`，叠成 `-qq` 会吞掉摘要行） |
| `ruff check .` | clean |
| phase0 锚（`run-phase0 --config config/example.yaml`） | **ic 0.9600 / annual 0.8408**（自 P0 起从未变过，任何改动都不许动它） |
| `qt.exec_baseline_freeze --verify` | **77/77 @ `45c14aa`** |
| `validate-config` | 37/37（32 + 5 个 cached twin 配置） |
| 唯一 OPEN 的 PR | #38（历史遗留，保持 OPEN） |

**当前主线 = 因子层深度重构（设计 v3.2，D0–D7）。D5 已全部收口**：C5 四腿对账 33 格全绿（#113），**C6 完成（#115 改造 + #116 删除）——11 个旧 runner 归零，决策③「杀第二条因子取数路径」收尾**，三个重生成工具退役为 verify-only、D1 基线 frozen-forever、校验已接进 C5 harness 的三个读入点。**D6a 已收口（#118）**：phase0/phase2 切 FactorService，真实 phase2 逐格 `max_abs_diff=0.0`、锚未动（R10：新锚 D6d 才转正）；顺带堵死一个**已经在发生**的红线 #6 违反（demo 值写进共享 store）。**D6b 已收口（#122–#124）**：oos/subset/robustness 切 FactorService，**先抓 legacy 再切**（D6a-5 教训）——PR-1 基线工程（capture harness + cached twins + `d6b_phase3` 冻结基线）中抓出 fina 并列披露 tie-break 缺陷（#122 单独 PR）与 fina tail 泄漏（`fina_tail_days: 0` 封），PR-2 切换后 S1 冷/S2 暖 vs 冻结基线**全叶子 0**。**D6c 已收口（#126/#127）**：intraday 两 runner（tail + group）切 FactorService——PR-1 先把 `mmp_ew`/`ret` 注册为正式因子（factor_id=legacy 列名保报告逐字节；binding 显式条目致全部已存因子 key 一次性失效已披露）并冻结 `d6c_i5` 基线（**score 腿 6/6 max_abs_diff=0.0**，新因子 ≡ legacy hook 的真数据值级证据在切换前就落盘），PR-2 切换后 S1 冷/S2 暖 × 6 配置 vs L1 **12/12 n_diffs=0**、执行面零改动。**下一步 = D6d**（删 shim/死 helper/aggregate 因子数学 + 新锚转正，4 PR 切分已定），然后 D7 收官全量重评估（**exec-only**）。D6c 登记三条 follow-up（详见 `docs/progress/07_factor_layer_refactor.md` D6c 条）：store key 无 cutoff 维（现由 runner guard 挡住，结构性修复留 D6d/D7）、未注册 score_feature 报错前置 fast-fail、`ret` 数学留 aggregate（R14 通用核）。

> **接手必读（唯一入口）**：[`tmp/design/HANDOFF_2026-07-30_claude_code.md`](tmp/design/HANDOFF_2026-07-30_claude_code.md)
> ——含当前截面、F1–F5 裁定速查表、六步路线、派发/评审/限额的具体操作法、陷阱清单。
> 伴侣文档 `tmp/context/cc_handoff_20260730_d5_c5/HANDOFF.md`（**动手修 F1–F5 前必须整读其 §2**，逐格实测证据在那里）。
> 设计权威 `tmp/design/factor_refactor_design_v3.md`（红线 #1–#11、四腿对账、D0–D7、实施期修订 A1/A2/A3）。
> ⚠️ **`tmp/` 被 `.gitignore` 排除**——这三份文档只在本机，不在 git 里。

## 常设授权（owner 明确给过，不必逐项再确认）

- **git 自主推进**（2026-07-27）：commit / push / 开 PR / merge 不必逐项确认，推进到重构完成或新指令为止。
- **14:50 截断**（2026-07-19）：全日数据的分钟因子一律截断到 14:50，授权直接改，不必逐因子再问。
- **C6 退役重生成能力**（2026-07-28）：删旧 runner 的同时把 `panel_freeze` / `panel_reconcile` / `hand_anchors_engine_values` 改为 verify-only，D1 基线 frozen-forever。

## 工作制度：双轨验收

- lead 只做调度与统筹（任务书 / 派发 / 裁决 / 合并 / 写进度文档）；**实现与评审派不同的 subagent**，评审是**对抗性**的（自写 fixture、自跑 mutation，**不复核对方的自述**）。这套制度累计抓出 **6+ 个真 HIGH**，多数是"测试全绿但结论错"。
- **验收双轨**：评审裁定 **+ lead 独立重跑 gates**；**绝不采信 subagent 自报的数字**。
- 评审的裁定只覆盖它实际看到的 commit；后落的 commit 要么补一轮评审，要么在 PR 正文里**显式声明未经评审**。
- 具体派发形态、限额中断处理、worktree 陷阱见 handoff §4/§5。

## 方法论（比任何单条代码都值钱）

1. **每个 mutation 都要先断言它真的改变了目标。** 本项目累计 **9 例"不可能失败的测试"**，其中三次出在同一天同一份工作里，**没有一次是被测试抓住的，全是被"与预期矛盾的观察"抓住的**。评审提出的建议本身也需要 mutation 证据。
2. **正则断言不了"没有别的句子这么说"，而"根本没有别的句子"可以。** 措辞/事实只写一遍（author-once），其余每处都去组合它；扫描守卫只是网，且**守卫的射程必须写进它自己的 docstring**。一个只扫你已经修过的地方的守卫，永远只能确认你已经知道的事。
3. **失配摧毁意义 → 拦；失配只带来已披露、方向已知的偏 → 记。**
4. **未编目的差异算失败**（设计 §六.5）；**性质测试跑不过时修引擎、不改测试**（§六.16）。
5. **"预期全绿却没绿"时先怀疑预测本身**——已有两次源头是上一份交接的预测写错了，不是引擎 bug。
6. **从一个样本推总体会造出假警报**；**均值与中位差一个数量级时，先去找那一行坏数据，别先给分布编故事**。
7. **"报告没说出关于它自己出处的该说的话"与"声明了它没做的检查"是同一类缺陷**——已出现三次。
8. **卡住就停下上报，别夹带。** 正因如此，一个已发布结论的更正才没被混进重构 PR。
9. 研究侧：**IC 强 ≠ 可交易**（11 个分钟因子里 1× 费率下净多空为正的只有 2 个，差距几乎全部来自换手）；**非业绩声明**是所有因子结论的默认前缀。

## 绝不可覆盖的 artifact

`artifacts/refactor_baseline/{panels, panels_d2, exec_baseline, pr_c_cutoff_fix, d6b_phase3, d6c_i5}` —— 后续一切对账的唯一参照物，**无法从当前代码再生**，只读。所有真实 run 一律 **cache-only**（日志须 `stk_mins_live_calls=0`）。

## 进度档案（完整历史；**按需 grep，不要整读**）

本文件曾内联全部阶段日志（~78k tokens，每个会话都被完整加载）。历史已逐字搬进 `docs/progress/`，**内容一字未改**。

> **新阶段的详细记录写进 `docs/progress/`（PR 号追加进 `pr_ledger.md`），不要写回本文件。**
> 本文件只维护三处随进度变化的内容：**当前状态**表、**当前有效的披露与限制**、**路线图**。

| 档案 | 覆盖 |
|---|---|
| [`01_phase0_3_daily_pipeline.md`](docs/progress/01_phase0_3_daily_pipeline.md) | Phase 0 MVP / Phase 1 偏差边界 / Phase 2 执行真实性 + PIT 行业 / Phase 3 多因子·IC 加权·OOS·robustness·候选因子·子集·独立样本·CSI500（PR #1–#21） |
| [`02_phase4_cache_and_intraday_i1_i4.md`](docs/progress/02_phase4_cache_and_intraday_i1_i4.md) | 端点级持久化缓存 P4-1/4-2/4-3 + 分钟 pipeline I1–I4（PR #23–#31） |
| [`03_intraday_i5.md`](docs/progress/03_intraday_i5.md) | 事件驱动回测 / 执行期涨跌停 / MMP 因子与五分位 / CSI300 泛化负结果 / 流动性诊断（PR #33–#55） |
| [`04_data_layer.md`](docs/progress/04_data_layer.md) | 数据层 D1–D5、schema 注册表与 drift 守卫、全A auto-warm 与历史回填（PR #41–#61） |
| [`05_minute_factors_eval_contract.md`](docs/progress/05_minute_factors_eval_contract.md) | `analytics/eval/` 三轴判决契约 v0.1→v0.9 + 十一因子复现总表与结论（PR #63–#74） |
| [`06_execution_basis_and_corrections.md`](docs/progress/06_execution_basis_and_corrections.md) | VWAP 成交 / 分钟收益复权 / 披露修正 / `adj_factor` 检查 / 日频除权审计 / exec-to-exec 改基准（PR #75–#82） |
| [`07_factor_layer_refactor.md`](docs/progress/07_factor_layer_refactor.md) | 因子层重构 D0–D5 C4b 全部细节与评审发现（PR #84–#107） |
| [`08_gates_disclosures_roadmap.md`](docs/progress/08_gates_disclosures_roadmap.md) | 归档时点的质量门快照 / 完整披露清单 / 路线图原文 |

其它权威文档：`docs/data/data_layer_contracts.md`（cache vs store 契约）、`docs/factors/`（D0 契约 / D1 面板冻结 manifest / D5 差异编目 / 性质测试映射表 / C5 审计报告骨架）、`docs/ops/data_update_schedule.md`。

## 当前有效的披露与限制

- **研究侧结论一律 EXPLORATORY、非业绩声明。** 日线 only；demo 路径非真数据。
- 旧三因子（momentum_20 / roe / netprofit_yoy）**无信号**；**value/低波**获得独立样本**符号级**确认（P3-7 SSE50/CSI300 量级衰减、P3-8 CSI500 泛化更强），但**组合级盈利能力仍未确立**（组间排名跨 cell 翻转）。
- 十一个分钟因子**全部封顶 Watch，无一到 Adopt**；1× 费率下净多空为正的只有 2 个（其一扛不住 2×）。
- **MMP 暂缓推进，理由在修正引擎上比过去更强**：I5d 的 CSI500 完美单调性退化（Spearman +1.0→+0.9，Q5 让位 Q4，四档归因把效应归到**复权修正本身**——原结论实质是"把除权下跌记成亏损"的产物）；I5e 的 CSI300 负结果不变且更负。**在出现 disjoint 窗口 / 第三 universe 的正向证据前，不当稳健信号推进。**
- I5f 容量诊断已按**真实 100 万 RMB 名义**重写（below-1.0x 由 20 笔变 **0** 笔，最紧 1.22×）；此前"10M 名义下约半数不足"的说法**已作废**。执行模型仍无 partial-fill / volume-cap。
- **日频除权审计已完成：GENUINE，无 vintage 接缝**（评估池暴露 1/603,258，不重述任何因子结果）；分钟路径的复权缺陷已在 PR #75 修。
- ⚠️ **PR #79 的 code review 未返回、靠自查合并**，其"未独立核"清单（五条 mutation 声称 / 十一个 runner 改动是否机械等同 / `execution_capacity` 透传）仍待补检——若查出问题，重述的是 **exec 基准**数字，close-to-close 未被触碰。
- schema drift 守卫**仍未接** live `data-update` 暖跑路径（当前只接 pipeline/回测）；分钟 schema 守卫未做。
- HTML compendium（`artifacts/reports/factor_compendium.html`）**还读 v0.8 结果**，待按最终结果修订。
- 全A 盘后自动拉取：**建设侧完成**（增量 auto-warm + systemd default-off + 历史回填），**live rollout 是用户 Stage-1**（先 `data-backfill` → 手动观察一次增量 warm → 才 enable timer）。

## 路线图

1. **因子层重构收尾（当前主线）**：~~F1 (#112)~~ → ~~C5 audit (#113)~~ → ~~C5 进度文档 (#114)~~ → ~~C6 (#115+#116)~~ → ~~D6a (#118)~~ → ~~D6b (#122–#124)~~ → ~~D6c (#126/#127)~~ → **D6d（下一步：删 shim/死 helper/aggregate 因子数学 + 新锚转正，4 PR 切分已定）** → D7 收官全量重评估（**exec-only**）。
2. **因子研究**：十一因子无一到 Adopt ⇒ 下一步优先做**换手/成本敏感的组合层验证**与更长 holdout，而不是继续堆新因子；补 PR #79 的 review；修订 HTML compendium。
3. **可选、须单独显式 goal**：I5g（强制 partial-fill / volume-cap）；数据层 D6（`PanelStore` append/partition，仅当因子研究需要可复用派生面板时才启动）；schema 守卫接 live `data-update`。
