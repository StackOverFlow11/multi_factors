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
- **权限**：实测充足——个股日线 / 分钟(`stk_mins`) / 复权(`adj_factor`) / 成分股(`index_weight`) / 申万行业历史(`index_member_all`/`index_classify`) / 财务含`ann_date`(`income`,`fina_indicator`) 全可取。分钟级可直接上，无需先退回日线。
- **MCP（可选开发工具）**：`financial_projects/.mcp.json` 有 tushare MCP，仅供开发期交互查数；**从 `financial_projects/` 启动 Codex 才加载**。
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
- **Git**：feature 分支 + PR。**PR #1（P0+P1）、#2（P2-1）、#3（P2-2）、#4（进度文档）、#5（P2-3,默认 SW-L1 可配置）、#6（进度文档）、#7（P2-4）均已 merge 到 `main`。** commit 用 conventional 格式，**无 attribution**（不加 Co-Authored-By）。
- **不过度设计**：按路线图 MVP 先打通一条端到端链路，再加层（architecture.html §11，Phase 0→3）。
- **secrets** 一律走外部 `.config.json`；repo `.gitignore` 已排除数据产物(`*.parquet`等)、缓存、`tmp/`（仅留架构文档）。
- 文件小而专（<800 行），immutable 优先。

## 当前进度
- ✅ 7 层骨架 + 架构文档（`main`）
- ✅ **Phase 0 MVP**（PR #1）：DemoFeed → PanelStore → StaticUniverse → momentum_20 → zscore → EqualWeightAlpha → TopN 等权 → 月度回测（成本/换手）→ IC/绩效报告，单命令可复现。
- ✅ **Phase 1 偏差边界**（PR #1，全部真数据实证）：
  - 前复权（qfq；store 存 raw，内存复权 → batch≡incremental 安全）
  - PIT 指数成分（`index_weight` as-of，survivorship-safe；370 天 pre-start 回看 + 90 天分页）
  - 可交易过滤（停牌 / ST / 涨跌停；**涨跌停用未复权 raw close** 比 `stk_limit`）
  - 财务 `ann_date` 披露日 as-of（绝不按 end_date；500 天 lookback carry forward）
  - 行业 + 市值中性化（按 date 截面 OLS 残差；欠定/无自由度截面 → NaN）
  - 路径感知降级披露（demo/static vs tushare/index/ann_date，绝不把 demo 当真实验证）
- ✅ 真数据实证（tushare，非 CI）：复权除权日 raw−5.74%→qfq+0.99% / CSI300 全年 24 快照 328 名换手 / ann_date Q1 延后至 04-20 / 中性化 corr −0.617→0。详见 `BIAS_AUDIT.md`、`artifacts/reports/phase1_summary.md`。
- ✅ **Phase 2-1 真实数据可复现基准**（PR #2 已 merge）：新 run mode `run-phase2-baseline` + `config/phase2_real_baseline.yaml`（上证50 `000016.SH`，2023-07~2024-06）。复用 P0/P1 全套机器，不扩因子、不调参。一次真实跑约 11 min，输出 `artifacts/reports/phase2_real_baseline.md`（gitignored）：数据窗口 / PIT 成分摘要 / ann_date 覆盖率 / 可交易过滤 / 每期持仓 / 换手成本 / IC / 绩效 / 全部 P2 降级。
- ✅ **Phase 2-2 执行真实性**（PR #3 已 merge）：拆分 selection（选谁）与 execution feasibility（能否成交）。
  - `runtime/fills.py::simulate_fills` 方向感知执行：涨停挡买 / 跌停挡卖 / 停牌·缺收盘双向挡；按 panel flag 实时判定，与选股 toggle 无关。
  - 现金一致 sell-then-buy：卖在前释放现金、买在后，现金不足按比例部分成交 → 无杠杆；被挡交易 carry forward；换手/成本只算实际成交；闲置现金按 driver 的 `cash_return` 计息。
  - `universe.min_listing_days` 真实路径已执行（`stock_basic.list_date` 富化，买入资格过滤，边界 age==min 放行，缺 list_date 保留并披露）；demo 无上市日 → 披露 no-op。
  - 回测 `feasibility_log()` 记录 blocked buys/sells/carried/executed turnover/invested；phase2 报告有 **Execution feasibility** 小节。
- ✅ **Phase 2-3 历史 PIT 行业**（**PR #5 已 merge**,默认 SW-L1 可配置）：把行业中性化协变量从 `stock_basic.industry` 当前标签升级为按 trade_date as-of 的历史申万行业。
  - `tushare_covariates.pit_sw_intervals(symbols, level)` 读 `index_member_all` 的 `in_date`/`out_date` 区间；`data/clean/pit_industry.py::asof_industry` 按 `[in_date, out_date)` 覆盖该日取行业，改分类日新行业生效，PIT-safe，起始日前成分 carry forward 到窗口开始。
  - **SW 层级可配置** `processing.neutralize.industry_level`(L1/L2/L3,**默认 L1**=31 宽板块,中性化标准 + 小截面自由度更稳)。**关键实证**:旧 tag 年化 −17.6% / SW-L1 −10.2% / SW-L2 −9.3% → **L1≈L2**,−17.6→−10 大跳主因是 **tushare→SW 分类切换**(补 PIT 必然:只有 SW 有 in/out 历史可 PIT 化,旧 tag 无法),**与粒度无关**。(曾误判粒度、默认 L2,经 L2 实测推翻,改默认 L1。)
  - 不静默退回 current：无 SW 历史的票 → 行业 NaN，被 neutralize 按截面丢弃；每次运行 **实际 level + PIT 覆盖率**进 phase2 报告。
  - 真实 baseline(SW-L1,默认):`run-phase2-baseline` OK,symbols=68,settled=11,**PIT SW-L1 coverage 98.53%**(67/68),IC 0.0083,**annual −10.19%**。
- ✅ **Phase 2-4 标准分析集成**（**PR #7 已 merge 到 `main`**）：alphalens-reloaded + quantstats 接入报告(**report-only cross-check**)。
  - `analytics/alphalens_adapter.py`(IC mean/IR + 分位均值)+ `analytics/quantstats_adapter.py`(CAGR/Sharpe/maxDD/vol)薄 adapter,各带 `backend` 字段;`_import_*` 间接层使 import-missing 路径可测;alphalens stdout/warnings 已抑制。
  - **简版仍权威**(驱动回测 + cross-check);依赖不可用/报错 → 报告披露 backend(unavailable/error,只记异常类型),**绝不静默假装**。
  - **不改 alpha/portfolio/runtime/fills/universe**:P0/P2 交易数字不变(demo ic 0.96/annual 0.84 不变;alphalens IC=简版 IC 吻合),只新增 **Standard analytics** 报告段。
- 🔧 **Phase 3-1 首个真实多因子 baseline**（`p3-multifactor-financial-baseline` 分支,代劳待验收）：pipeline 消费**全部 enabled factors**（曾只用第一个）;`config/phase3_real_multifactor.yaml` = momentum_20 + roe + netprofit_yoy（SSE50 同窗口,与 phase2 可比;不调参、无 learned weights、非收益承诺）。
  - 财务字段**一次 fetch + 一次 as-of 对齐**;合成仍 EqualWeightAlpha 等权;`drop_missing` 要求全因子齐备（披露）;demo+财务因子仍可读报错;重名因子报错。
  - 报告：active factor list / per-factor coverage+IC+分位 / combo score 诊断 / 财务 coverage 按字段（TRADED vs diagnostic）;`output.baseline_report_name` 分离 phase3 报告。
  - 真实结果:多因子 annual **−9.05%**（单因子 −10.19%）;momentum_20 IC 0.0083（跨 run 一致）/ roe 0.0006 / netprofit_yoy 0.0001 / combo −0.0038;财务 ann_date 覆盖 100%;PIT SW-L1 98.53%。回归不破:phase2 rerun −10.19%/0.0083 不变,demo 0.96/0.84 不变。
- ✅ 当前质量门：`pytest -p no:cacheprovider` **249 passed**；`ruff` clean；`validate-config`（demo + `example_tushare.yaml` + `phase2_real_baseline.yaml` + `phase3_real_multifactor.yaml`）OK；`run-phase0`（demo）OK。
- ⚠️ 剩余（已显式披露）：日线 only、demo 路径非真数据。
- 路线图下一步：alpha 加权合成（IC 加权/回归）/ 分钟级（architecture.html §11）。
