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
- **Git**：feature 分支 + PR。**PR #1（P0+P1）、#2（P2-1）、#3（P2-2）均已 merge 到 `main`**。commit 用 conventional 格式，**无 attribution**（不加 Co-Authored-By）。
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
- ✅ **Phase 2-1 真实数据可复现基准**（**PR #2 已 merge 到 `main`**）：新 run mode `run-phase2-baseline` + `config/phase2_real_baseline.yaml`（上证50 `000016.SH`，2023-07~2024-06）。**复用 P0/P1 全套机器，不扩因子、不调参**。一次真实跑 ~11min（68 成分 / 25 loaded 快照 / in-window distinct 60 / 11 settled 调仓，候选 12 末日跳过），输出 `artifacts/reports/phase2_real_baseline.md`（gitignored）：数据窗口 / PIT 成分摘要(loaded vs in-window) / ann_date 覆盖率(100%) / 可交易过滤命中(首命中互斥) / 每期持仓 / 换手成本 / IC(≈0.008) / 绩效(年化−17.6%，**亏损动量基准，非业绩声明**) / 全部 P2 降级。诊断只读、demo 源拒绝、token 不入报告。
- ✅ **Phase 2-2 执行真实性**（PR #3 已 merge 到 `main`）：**拆分 selection（选谁）与 execution feasibility（能否成交）**。
  - 方向感知执行 `runtime/fills.py::simulate_fills`：涨停挡买 / 跌停挡卖 / 停牌·缺收盘双向挡;按 panel flag 实时判定,与选股 toggle 无关。
  - **现金一致 sell-then-buy**:卖在前释放现金、买在后,现金不足按比例部分成交 → **无杠杆**;被挡交易 carry forward;换手/成本只算实际成交;闲置现金按 driver 的 `cash_return` 计息（BT-007）。
  - `universe.min_listing_days` **真实路径已执行**（`stock_basic.list_date` 富化,买入资格过滤,边界 age==min 放行,缺 list_date 保留并披露）;demo 无上市日 → 披露 no-op。
  - 回测 `feasibility_log()`：每调仓期 blocked buys/sells/carried/executed turnover/invested;phase2 报告新增 **Execution feasibility** 小节。
  - **保不变量**:demo 无 flag → 全可成交 → P0/P1 数字不变;无未来函数、PIT/ann_date/real-demo 分离不变。
- ✅ **Phase 2-3 历史 PIT 行业**（PR #5,**默认 SW-L1 可配置版**,等验收）：把行业中性化协变量从 `stock_basic.industry` **当前**标签升级为按 trade_date **as-of** 的历史申万行业。
  - `tushare_covariates.pit_sw_intervals(symbols, level)` 读 `index_member_all` 取每股 SW 行业 `in_date`/`out_date` 区间;`data/clean/pit_industry.py::asof_industry` 按 `[in_date,out_date)` 覆盖该日取行业(改分类日新行业生效,PIT-safe;起始日前成分 carry forward 到窗口开始)。
  - **SW 层级可配置** `processing.neutralize.industry_level`(L1/L2/L3,**默认 L1**=31 宽板块,中性化标准 + 小截面自由度更稳)。**关键实证**:旧 tag 年化 −17.6% / SW-L1 −10.2% / SW-L2 −9.3% → **L1≈L2**,大跳的主因是 **tushare→SW 分类切换**(补 PIT 的必然:只有 SW 有 in/out 历史可 PIT 化,旧 tag 无法),**与粒度无关**。(注:曾误以为是粒度、默认 L2,经 L2 实测推翻,改默认 L1。)
  - **不静默退回 current**:无 SW 历史的票 → 行业 NaN,被 neutralize 按截面丢弃(neutralize 数学不变,本就丢 NaN 行);每次运行 **实际 level + PIT 覆盖率**进 phase2 报告。
- ✅ 质量门：`pytest` **223 passed**（P0=95 / P1=77 / P2-1=15 / P2-2=22 / P2-3=14）；`ruff` clean；`validate-config`（demo + `example_tushare.yaml` + `phase2_real_baseline.yaml`）+ `run-phase0`（demo）均 OK。
- ⚠️ 剩余 P2（已显式披露）：日线 only、简版 IC/绩效未走 alphalens/quantstats、demo 路径非真数据。
- 路线图下一步：财务因子组合 / 分钟级 / alphalens·quantstats 接入（architecture.html §11）。
