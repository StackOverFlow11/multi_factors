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
- **Git**：feature 分支 + PR。**PR #1（P0+P1）、#2（P2-1）、#3（P2-2）、#4（进度文档）、#5（P2-3）、#6（进度文档）、#7（P2-4）、#8（进度文档）、#9（P3-1）、#10（进度文档）、#11（P3-2）、#12（P3-3）、#13（进度文档）、#14（P3-4）、#15（进度文档）、#16（P3-5）、#17（进度文档）、#18（P3-6）、#19（P3-7）、#20（进度文档）、#21（P3-8）、#22（进度文档）、#23（P4-1）、#24（进度文档）、#25（P4-2）、#28（tushare 权限/限频探测 + capability registry）、#29（I1–I4 分钟级 intraday pipeline，4 commit 一 PR）、#30（进度文档）、#31（P4-3 因子支撑端点缓存 + 21:00 data updater，2 commit 一 PR）、#33（P-I5a 事件驱动回测架构重构 + opt-in 分钟尾盘 event model，4 commit 一 PR）、#35（P-I5b 分钟尾盘执行期 raw stk_limit 涨跌停可行性，4 commit 一 PR）、#37（P-I5c MMP 分钟因子端到端 opt-in alpha）、#39（P-I5d MMP 五分位分组回测 standalone，含 I5c plumbing）均已 merge 到 `main`**。commit 用 conventional 格式，**无 attribution**（不加 Co-Authored-By）。
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
- ✅ **Phase 2-3 历史 PIT 行业**（**PR #5 已 merge 到 `main`**,默认 SW-L1 可配置）：把行业中性化协变量从 `stock_basic.industry` **当前**标签升级为按 trade_date **as-of** 的历史申万行业。
  - `tushare_covariates.pit_sw_intervals(symbols, level)` 读 `index_member_all` 取每股 SW 行业 `in_date`/`out_date` 区间;`data/clean/pit_industry.py::asof_industry` 按 `[in_date,out_date)` 覆盖该日取行业(改分类日新行业生效,PIT-safe;起始日前成分 carry forward 到窗口开始)。
  - **SW 层级可配置** `processing.neutralize.industry_level`(L1/L2/L3,**默认 L1**=31 宽板块,中性化标准 + 小截面自由度更稳)。**关键实证**:旧 tag 年化 −17.6% / SW-L1 −10.2% / SW-L2 −9.3% → **L1≈L2**,大跳的主因是 **tushare→SW 分类切换**(补 PIT 的必然:只有 SW 有 in/out 历史可 PIT 化,旧 tag 无法),**与粒度无关**。(注:曾误以为是粒度、默认 L2,经 L2 实测推翻,改默认 L1。)
  - **不静默退回 current**:无 SW 历史的票 → 行业 NaN,被 neutralize 按截面丢弃(neutralize 数学不变,本就丢 NaN 行);每次运行 **实际 level + PIT 覆盖率**进 phase2 报告。
- ✅ **Phase 2-4 标准分析集成**（**PR #7 已 merge 到 `main`**）：alphalens-reloaded + quantstats 接入报告(**report-only cross-check**)。
  - `analytics/alphalens_adapter.py`(IC mean/IR + 分位均值)+ `analytics/quantstats_adapter.py`(CAGR/Sharpe/maxDD/vol)薄 adapter,各带 `backend` 字段。
  - **简版 numpy/pandas 仍权威**(驱动回测 + cross-check);依赖不可用/报错 → 报告显式披露 backend(unavailable/error,只记异常**类型**不记消息),**绝不静默假装用了标准库**。
  - **不改 alpha/portfolio/runtime/fills/universe**:P0/P2 交易数字不变(demo ic 0.96/annual 0.84 不变;实测 alphalens IC=简版 IC 完全吻合),只新增 **Standard analytics** 报告段。
- ✅ **Phase 3-1 首个真实多因子 baseline**（**PR #9 已 merge 到 `main`**）：pipeline 从"只用第一个 enabled factor"升级为**消费全部 enabled factors**;新增 `config/phase3_real_multifactor.yaml`（momentum_20 + roe + netprofit_yoy,SSE50 同窗口,与 phase2 可比）。**不调参、无 learned weights、非收益承诺**。
  - `_build_factors` 全量实例化（配置序;重名报错）;财务字段**一次 fetch + 一次 as-of 对齐**（逐字段独立守 `ann_date <= trade_date`,无每因子重复拉取）;demo+财务因子仍可读报错。
  - 合成仍 `EqualWeightAlpha`（处理后各列等权平均,不看 forward returns）;`drop_missing` 要求该日该票**所有**因子齐备（显式披露）。
  - 报告增强：active factor list / per-factor coverage+IC+分位 / **combo score** 诊断 / 财务 coverage **按字段**披露（TRADED vs diagnostic 角色标注）;`output.baseline_report_name` 使 phase3 报告独立于 phase2。
  - **真实结果**（SSE50 2023-07~2024-06,~14min）:多因子 annual **−9.05%**（单因子 −10.19%）;per-factor IC: momentum_20 0.0083（与 phase2 run 完全一致,跨 run 一致性实证）/ roe 0.0006 / netprofit_yoy 0.0001;combo IC −0.0038;财务两字段 ann_date 覆盖 **100%**;PIT SW-L1 98.53%。财务因子在该小截面短窗口 IC≈0,照实披露——这是 plumbing 验证。
  - **回归不破**:phase2 单因子真实 rerun annual −10.19% / IC 0.0083 不变;demo ic 0.96/annual 0.84 不变。
- ✅ **Phase 3-2 walk-forward IC 加权 alpha**（**PR #11 已 merge 到 `main`**）：新增 `alpha/ic_weight.py::RollingICWeightAlpha`（`alpha.model: ic_weighted`）;EqualWeightAlpha 仍默认 + 回归基线。**不调参、非收益声明**。
  - **Lookahead 边界（测试锁定）**：训练严格 walk-forward——(factor[t], fwd_h[t]) 只有**已实现**（交易日序 `t+h <= d`）才进日 d 的权重;**扰动未实现 forward returns 权重不变**（扰动测试）+ `t+h` 切片精确边界测试。forward returns 由 pipeline 在 alpha 边界计算、只传 `alpha.fit`,factors 层照旧绝不接触（不变量 #1）。
  - rolling（默认保守,窗口=60 交易日,min_periods=20）/ expanding 可配;历史不足 → 该日**退回等权**（与 EqualWeightAlpha 逐 bit 一致）并计数披露;权重 L1 归一化、保留符号（负 IC 因子负权重）。
  - 报告新增 **Alpha model** 必含小节：active model / 超参 / 训练覆盖率 + fallback 次数 / 每调仓日生效权重表（fallback 行标注）/ 非调参声明 + 等权基线对比指引。
  - `config/phase3_real_ic_weighted.yaml`：与 phase3_real_multifactor **唯一差异是 alpha.model**（universe/window/因子/中性化全同,直接可比）。
  - **真实结果**（SSE50 2023-07~2024-06,~14min）:annual **−3.57%**（等权 −9.05% / 单因子 −10.19%）,maxDD −12.93%,训练覆盖 **201/221**（20 个 fallback 全在窗口攒满前,90.95%）。⚠️ 优于等权**不是**业绩声明——单年窗口 + 权重逐期翻号（如 momentum_20 从 −0.58 到 +0.36）正是小样本不稳定的体现,照实披露。
  - **回归不破**:phase3 等权真实 rerun annual −9.05% / IC 0.0083 不变;demo equal_weight ic 0.96/annual 0.84 不变（测试锁定）。
- ✅ **Phase 3-3 OOS 稳定性验证**（**PR #12 已 merge 到 `main`**,含 review 两 HIGH+一 LOW 修复）：**报告型验证层**,不加新 alpha 复杂度、不改 portfolio/execution/factor math。新 run mode `run-phase3-oos`（`qt/oos_stability.py`）+ `config/phase3_real_oos_stability.yaml`（SSE50 扩到 **2 年** 2022-07~2024-06,split 2023-07-01 → train 1y / test 1y,test 年=旧 baseline 窗口可对照）。
  - **一次数据加载、同一 processed 因子面板、两次回测**（equal_weight vs ic_weighted）;所有诊断按 split 切段（子段 nav 重新归一,绝不跨段串味）。
  - **边界语义（测试锁定）**:walk-forward(rolling subperiod)——任何日期的权重只用该日已实现观测(`t+h <= d`);**扰动 split 后全部 forward returns,train 期所有日期权重逐 bit 不变**(split 无泄漏测试);不用 freeze-at-split(那是新 alpha 模式,超范围)。**绩效切片按持有窗口**(train 行持有期 end≤split、test 行 start≥split,跨界调仓从两段排除并披露;IC 按实现日 t+h 切)——绝不按 signal date 单切(review HIGH 修复:旧切法把跨界持有期的 test 收益记进 train);runner 强制 `alpha.model: ic_weighted`(否则假对比,可读报错)。
  - 报告 `phase3_oos_stability.md`:split 边界+跨界行披露/分期绩效(annual/vol/sharpe/maxDD/turnover)/逐序列 IC 分期(mean/IR/hit rate/sign consistency)/权重稳定性(每期权重含 train-test 标注、trained 行 sign flips、fallback 次数+原因)/小样本 caveat。
  - **真实结果（关键发现,~16min,77 成分/2 年,持有窗口切片）**:三个原始因子 train→test **IC 全部翻号**(momentum −0.023→+0.006 / roe −0.029→+0.007 / np_yoy −0.011→+0.005,sign consistency 全 NO),hit rate 46~53%≈抛硬币;权重 23 期 sign flips 7/3/4;绩效 eq train −11.92%/test −5.27%,ic train −8.31%/test −2.70%(跨界行 2023-06-30 排除;修切片前 train 被 test 期收益污染到 −6.81%/−1.69%,修正幅度本身就是边界 bug 的实证)。**结论:ic_weighted 两段都略好但 IC≈0 且翻号——P3-2 单年跑赢不可外推,这正是本验证层要拿到的证据;非收益声明。**
  - **回归不破**:phase3 equal_weight rerun −9.05%/0.0083、ic_weighted rerun −3.57% 均不变;demo 0.96/0.84 不变;secret scan 报告 0 处 token/config.json。
- ✅ **Phase 3-4 robustness matrix**（**PR #14 已 merge 到 `main`**,含 review MEDIUM 修复:矩阵 DOWNGRADES 全 universe 披露）：把 P3-3 OOS 检验批量跑到 **universe × window 矩阵**上,回答 P3-2/P3-3 结论是否 SSE50 小样本偶然。不加因子/alpha、不改 portfolio/execution/factor math。
  - 新 run mode `run-phase3-robustness`（`qt/robustness.py`）+ `config/phase3_real_robustness_matrix.yaml`（SSE50+CSI300 × 2020-2022/2022-2024 两 fold;`skip_cells` 显式跳过 CSI300×2020-2022——runtime 预算,报告披露,绝不静默缩覆盖）。
  - **每个 cell 逐字复用 P3-3 cell 核心**（`_run_oos_cell` 重构抽出,单 run 行为不变;持有窗口切片/实现日 IC 切片/walk-forward/ic_weighted guard 全继承）;cell config 仅替换 universe/窗口/split/output_name（重过全套 pydantic 验证,parquet 不互覆盖）;跨 cell 汇总严格按 cell 归属（测试锁定不串数）。
  - **真实矩阵结果**（3 cells,总 2h:960s/934s/5301s,CSI300 399 名）:
    - **P3-3 不漂移 smoke ✓**:SSE50|2022-2024 cell 逐数复现 P3-3 报告（−11.92%/−5.27%/−8.31%,boundary 2023-06-30）。
    - 原始因子 train→test sign consistency:momentum 1/3 / roe 0/3 / np_yoy 0/3——**IC 翻号跨 cell 普遍,P3-3 结论非偶然**。
    - `combo_ic_weighted` 是唯一跨 cell 稳定序列（test IC 3/3 正 + 3/3 sign consistent）但量级 ≈0.004~0.008 极小。
    - **关键负面发现**:ic_weighted 组合绩效不稳健——train→test 翻车形态出现在 **3 cells 中的 2 个**:SSE50|2020-2022 train **+7.09%→test −7.85%**、CSI300|2022-2024 train **+11.07%→test −17.74%**（sharpe +0.59→−1.01;CSI300 换手 1.2~1.6 成本拖累更重）。**凡 ic_weighted 样本内显著为正,样本外即翻负——典型过拟合签名;P3-2 的"跑赢"不能外推;微弱正 IC ≠ 净值赢。非收益声明。**
  - secret scan 报告 0 处;demo 0.96/0.84 不变。
- ✅ **Phase 3-5 factor candidate pack**（**PR #16 已 merge 到 `main`**,**EXPLORATORY**）：保守日频 PIT-safe 候选因子组,经 P3-4 矩阵原样复检"弱信号是否只是因子集太窄"。不动 alpha/portfolio/execution/OOS/robustness、不调参。
  - 新因子（`factors/compute/candidates.py`,数学/前导 NaN/未来扰动不变/跨 symbol 隔离/不可变全测试锁定）:`reversal_5/20`(=−momentum 复用同机器)/`volatility_20`(滚动收益 std,满窗)/`liquidity_20`(log 滚动均额,非正→NaN)/`overnight_mom_20`(Σlog(open_t/close_{t-1}) 20 日,open 当日开盘已知,prev close 不跨 symbol)/`value_ep`/`value_bp`(daily_basic pe/pb **一次 fetch** 取倒数,≤0→NaN,当日发布 PIT-safe)/`grossprofit_margin`(进 SUPPORTED_FIELDS,走既有 ann_date as-of)。
  - dispatch（registry）扩展;**窗口名/params 不一致 → 可读报错**(绝不静默错标列);旧配置逐 bit 复现(demo 0.96/0.84 不变)。
  - `config/phase3_real_factor_candidates.yaml`:旧三 + 候选八 = 11 因子,同 P3-4 矩阵形状(CSI300×2020-2022 显式 skip 披露);**旧 vs 新一跑读出**(raw IC 按列独立)。
  - **真实矩阵（11 因子版 3 cells / 2h;旧三因子+前轮候选 per-cell IC 逐数不变=不漂移 smoke ✓;secret scan 0;一次 CSI300 网络故障后整 run 重跑成功）**:
    - **"因子集太窄"假设获得支持**:`value_ep`/`value_bp` test IC **3/3 正、0.037~0.056**;`volatility_20` **3/3 负、−0.044~−0.079**(低波方向稳定)——比旧三因子(|IC|≤0.015)高一个量级且方向跨 cell 一致。liquidity 3/3 负但量级小;`overnight_mom_20` test IC 2/3 正(+0.008/+0.016,2020-22 为负)中等偏弱;reversal/grossprofit_margin 翻号无信号。
    - `combo_ic_weighted` test IC **3/3 正 + 3/3 sign consistent**(0.0253/0.0012/0.0395)——walk-forward IC 加权吃到了新信号;11 因子**等权** combo 反被翻号因子稀释(IC 0/3 正,CSI300 test **−25.2%**)——等权对因子集质量敏感。
    - 组合绩效:ic test −2.21%/−5.02%/−2.76%(**3/3 cells 跑赢等权**;CSI300 train +23.11%→test −2.76%,train→test 衰减仍显著)。
    - ⚠️ **EXPLORATORY,非收益声明**:value/低波在 2020-2024 A 股是已知强势 regime;三 cells 窗口重叠非独立样本;未调参、未做成本敏感性。下一步候选:value+lowvol 子集独立复检。
- ✅ **Phase 3-6 value+lowvol 子集复检 + 成本敏感性**（**PR #18 已 merge 到 `main`**，**EXPLORATORY**）：把 P3-5 发现的 value/低波信号做**组间对照**（同一矩阵、同 footing），并补三档交易成本场景。**report-only：不加 alpha、不调参、不动 portfolio/execution/OOS 切片/robustness 聚合；`_run_oos_cell` 一字未动**。
  - 新 run mode `run-phase3-subset`（`qt/subset_validation.py`）+ `config/phase3_real_subset_costs.yaml`：4 组（legacy_trio / full_pack=11 因子 / value_lowvol=value_ep+value_bp+volatility_20 / value_lowvol_liq=+liquidity_20，/goal 的"可选"直接实测）× 3 成本场景（base ×1 / 2x ×2 / high_cost ×4，**必须有 multiplier=1.0 base 锚**，config 校验强制）× P3-4 矩阵 3 cells（CSI300×2020-2022 skip 照旧披露）。
  - **机制（测试锁定）**：每 cell 一次共享加载 + raw factor panel（与 OOS cell 同调用序）→ 每组从 raw panel **独立重处理**（`drop_missing` 按组——切 processed 列会错因为 drop_missing 跨列；全列组与旧处理 bitwise 一致）→ 每组 eq vs walk-forward ic_weighted × 每场景回测。成本场景只乘 `cost.fee_rate`：scores/fills 不见 fee → **trades/turnover/gross 跨场景严格不变，只动成本线**；`_run_backtest_for` 增加 default-preserving `fee_rate` 参数（None==旧行为，bitwise 测试锁定）。
  - **三层不漂移对账（真实 run 全过）**：① raw 因子 IC **66/66 行**（11 因子×train/test×3 cells）与 P3-5 报告逐数一致；② `legacy_trio`@base 逐数复现 P3-3/P3-4（ic test −7.85%/−2.70%/−17.74%，train +7.09%/−8.31%/+11.07%）；③ `full_pack`@base 逐数复现 P3-5（ic test −2.21%/−5.02%/−2.76%，combo IC 0.0253/0.0012/0.0395，eq test −4.30%/−6.83%/−25.16%）——按组重处理 ≡ 把该组当配置因子集单跑。
  - **真实结果（3 cells×4 组×3 场景，~2.1h）**：
    - **子集假设在净值层不成立**：value_lowvol ic test base **−4.49%/−8.66%/−5.84%** vs full_pack **−2.21%/−5.02%/−2.76%**——full_pack 在 **3/3 cells × 全部场景**净值更优。IC 层子集略稳：value_lowvol combo_ic test IC 3/3 正（0.0245/**0.0107**/0.0387），量级与 full_pack 相当且中间 cell 远离 0（full_pack 0.0012≈0）——**IC 稳定性与净值排序不同向**，照实披露。
    - value_lowvol_liq（+liquidity_20）：CSI300 IC 最高（0.0427）、SSE50|2020-2022 单 cell 最佳（−2.48%），但 2022-2024 两 cell 均劣于 value_lowvol——**加 liquidity 非免费午餐**，"可选"问题用数据回答。
    - 等权稀释复确认：full_pack eq CSI300 −25.16% vs value_lowvol eq −14.97%（子集等权耐稀释）；ic beats eq（base）：trio 2/3、full_pack 3/3、value_lowvol 3/3、liq 2/3。
    - **成本敏感性（新维度）**：base→2x→high_cost **全组全 cell 单调恶化**（无例外）；4× fee 退化 1.7~4.4pp 年化——legacy_trio CSI300 最重（−17.74%→−22.13%，高换手代价），value_lowvol 低换手（0.41~0.74/月）退化 ~1.8~2.4pp；算术年化 drag base ~0.5-0.9% / high_cost ~1.4-4.8%；turnover 跨场景不变实测 ✓。
    - ⚠️ **全组 test annual 均为负——非收益声明**；**POST-HOC 选择已披露**（子集是在同一批窗口上看完 P3-5 结果后选的，本 run 只量化相对稳健性+成本敏感性，**不是独立确认**——独立确认需真正新窗口/新 universe）。
  - secret scan 报告+日志 0 处（token 值/"token"/".config.json"）；demo 0.96/0.84 不变。
- ✅ **Phase 3-7 独立样本验证**（**PR #19 已 merge 到 `main`**，曾 stacked 在 P3-6 分支上、PR #18 先合后顺序合入。**EXPLORATORY**，含 review 2 HIGH 修复）：把 value/低波发现从"同窗口 POST-HOC 对照"推进到真正独立 holdout。不加因子/alpha、不调参；P3-6 group/cost 逻辑与配置不变（旧 27 测试原样通过）。
  - **review 2 HIGH 修复**：P3-7 报告曾沿用 P3-6 标题/"not independent confirmation"开场白/"SAME overlapping windows"收尾 caveat，与自身 verdict 节矛盾 → `render_subset_validation` 标题/框架/caveat 与 `_subset_downgrades` 按是否含 independent cells 分支（P3-6 配置原文不变，回归测试锁定，+3 测试）；artifact 按用户指示**不重跑矩阵**，以确定性脚本把 5 处与数据无关的静态文案行替换为新渲染器的精确输出（新字符串从新代码路径提取、非手敲；diff 仅 5 行、全部数字未动、secret scan 0）。
  - **机制（测试锁定）**：独立性是**人的声明**——`subset_validation.independent_cells` 显式列 cell（必须引用矩阵已声明 cell、**禁止与 skip_cells 冲突**——"声明独立验证却不跑"是配置错误；未声明一律默认 screened，保守）；`hypotheses`（因子→期望 IC 符号）**run 前固定**于 config（value_ep+/value_bp+/volatility_20−，源自 P3-5 筛选）；`min_rebalances`（默认 8）不足 → **INSUFFICIENT-DATA**（样本量必披露，绝不静默通过）。verdict=事实性符号检查（HOLDS=期望符号在 holdout **双子期**都成立；SUPPORTED/PARTIAL/NOT SUPPORTED；NaN/缺列不成立）；**分类汇总绝不混**（summarize_by_sample 按 sample class 各自聚合；verdict 节只读独立 cells）。
  - 真实 config：screened anchor SSE50|2022-2024（必须逐数复现 P3-6=run 内不漂移锚）+ 独立 SSE50|2024-2026、CSI300|2024-2026（**2024-07-01→2026-05-31 后于全部筛选**，split 2025-07-01；数据实测到 2026-06）；CSI300|2022-2024 skip 披露。
  - **运维实证**：两次真实 run 死于瞬时 ConnectionError（旧默认 3 次重试≈3s 容忍）→ 默认重试预算 **3→6**（≈23s 容忍，成功路径不变，测试锁定）→ 第三次 ~2.2h 跑通。⚠️ P3-7 报告与 P3-6 同名（`phase3_subset_validation.md`），后跑覆盖前跑（regenerable，数字都在进度文档）。
  - **anchor 不漂移 ✓**：raw IC 22/22 ≡ P3-5 矩阵报告；组级 base 年化逐数 ≡ P3-6。
  - **真实结果（3 cells/~2.2h，独立结论只来自 2 个 holdout cells）**：
    - **独立 verdict：2/2 cells SUPPORTED**（各 21 settled rebalances vs min 8）——value_ep IC +0.0322/+0.0134（SSE50 holdout train/test）与 +0.0245/+0.0072（CSI300）；value_bp +0.0379/+0.0033 与 +0.0310/+0.0041；volatility_20 −0.0320/−0.0120 与 −0.0373/−0.0164。**P3-5 的符号假设在未参与筛选的样本上全部成立**。
    - ⚠️ **量级显著衰减**：筛选期 |IC| 0.037~0.079 → holdout 后段 0.003~0.016（value_bp 后段≈0）——符号存活、强度衰减（部分向零回归），照实披露。
    - combo_ic_weighted 独立 test IC **8/8 正**（4 组×2 cells，0.0060~0.0238）——walk-forward IC 加权在 holdout 上仍有效。
    - **组合净值仍未确立**：独立 base ic test——legacy_trio **+1.15%/+8.13%**（2/2 正，恰是此前实证"无信号"的组！）、full_pack +7.30%/−1.93%、value_lowvol +1.72%/−4.68%、liq +3.53%/−0.39%——组间排名跨 cell 翻转,~21 rebalances 小样本,**IC 符号确认 ≠ 组合盈利**；成本阶梯仍全单调。非收益声明。
  - secret scan 报告+日志 0 处；demo 0.96/0.84 不变。
- ✅ **Phase 3-8 CSI500 独立泛化检验**（**PR #21 已 merge 到 `main`**，**EXPLORATORY**）：P3-7 的符号级结论是否泛化到筛选 universe 之外——新增独立 cell **CSI500（000905.SH）|2024-2026**（universe+时间双独立）。**机器零改动**复用 P3-7 层（同组/同成本场景/同假设，不加因子不调参）；唯一代码变更 `output.subset_report_name`（沿 `baseline_report_name` 先例，默认 None 保持旧文件名 bitwise，测试锁定）——P3-8 报告独立成文件，**不再覆盖已验收的 P3-7 artifact**。
  - cell 设计：SSE50|2022-2024（screened 锚，须 ≡P3-6/P3-7）+ SSE50|2024-2026（independent 锚，须 ≡P3-7 verdict 数字）+ CSI500|2024-2026（主问题）；CSI500|2022-2024 skip 披露（~645 名×2y 再加 ~3h 不值）。数据可行性先实测：index_weight 月度 500 名快照到 2026-05-29 含 2024-06-28 pre-start 锚。
  - **真实 run（3 cells/~3.55h，CSI500 735 distinct 名主导；一次跑通）双锚对账 ✓**：screened raw IC 22/22 ≡ P3-5 报告；independent SSE50 verdict IC 逐数 ≡ P3-7（+0.0322/+0.0134、+0.0379/+0.0033、−0.0320/−0.0120）——复现性二度确认。
  - **CSI500 verdict：SUPPORTED**（21 settled vs min 8）：value_ep **+0.0083/+0.0145**（train/test）、value_bp **+0.0230/+0.0127**、volatility_20 **−0.0350/−0.0272**——三假设双子期全保持。**且衰减更小**：CSI500 test 子期量级（0.0127~0.0272）明显高于 SSE50/CSI300 holdout 的后段（0.003~0.016）——value/低波信号在中盘股上更强，**P3-7 结论泛化成立（GENERALIZES，未减弱）**。
  - combo_ic_weighted CSI500 test IC 全 4 组正（0.0243/0.0294/0.0286/0.0285，为三个 holdout cells 最高）；**组合净值 CSI500 base 全组正且 4× 高成本仍全正**（trio +17.80%→+10.57% / full_pack +7.83%→+2.95% / value_lowvol +4.19%→+1.14% / liq +4.02%→+0.84%）——首个全成本阶梯为正的 cell。⚠️ 仍然诚实：trio +17.80% 又是组间排名跨 cell 翻转的例证（中盘动量 regime）；单窗口 ~21 调仓小样本；**非收益声明**。
  - **报告标题配置化**（`output.subset_report_title`，沿 `subset_report_name` 先例）：P3-8 报告 H1 自报 study 名「Phase 3-8 — CSI500 Independent Generalization Check」而非机器默认 P3-7 标签（review 同类 stale-wording 问题修复，测试断言首行非 P3-7；P3-6/P3-7 配置不设此项、保持 sample-aware 默认，回归锁定）；文案/分区检查全过（无 stale P3-6 措辞、verdict 节只含独立 cells）；secret scan 报告+日志 0 处；demo 0.96/0.84 不变。
- ✅ **Phase 4-1 持久化 Tushare 行情缓存**（**PR #23 已 merge 到 `main`**，daily + adj_factor）：feed 之下的 **endpoint 级 raw 缓存**——真实 run 不再每次重抓全量日线+复权因子。**默认 disabled（向后兼容，旧配置行为一字不变）**；opt-in 后行情走 read-through，只有未覆盖的日期区间打 API。
  - **缓存只存 raw**（未复权 OHLCV/amount + 原始 adj_factor），绝不存 qfq、绝不存任何 secret；`front_adjust` 仍在内存跑，公式/时机零改动；`PanelStore` 仍是 per-run artifact，**不是**缓存 SoT。
  - `data/cache/`：`intervals`（闭区间日历算法做 gap 规划——按"日历区间是否抓过"而非"行是否存在"判定，正确区分"未抓"vs"源本无行"）/ `parquet_store`（按 `(endpoint,symbol)` 分文件存 `symbol_prefix` 分片，原子 upsert 按 `(date,symbol)` 去重，latest 胜）/ `coverage`（append-only ledger，11 列含 status ok/empty/failed，**只 ok/empty 算覆盖**，failed 留待重试）/ `tushare_cache`（read-through：只抓 gap、upsert、记 coverage 含空返回、再从缓存读全区间；`refresh_recent_days` 重抓近端 tail；`force_refresh` 整端点重拉；per-run 抓取计数 stats 日志）。
  - `TushareFeed.get_bars`：`cache=None` → 旧直抓路径**逐字不变**；`cache` 注入 → read-through 后用**与直抓完全相同的 join+select**，故 `front_adjust` 前面板逐字节一致（qfq 等价单测锁定）；per-symbol 限流/重试仍留在 feed 的 `_call` 闭包里。`_build_market_cache` 仅在 `data.cache.enabled` 时接线。
  - **配置**：`data.cache`（enabled=False / root_dir=`artifacts/cache/tushare/v1`（gitignored）/ refresh_recent_days=14 / force_refresh=[]）；校验拒绝负刷新窗/空 root/未知键；全部旧配置仍 validate。
  - **真实 smoke（phase2 baseline；非缓存参照 ref / 缓存冷 cold / 缓存暖 warm 三轮）**：
    - **三轮 report 指标完全一致** ✓（IC 0.0083 / annual −10.19% / maxDD −16.52% / vol 16.59% / sharpe −0.5703 / turnover 1.0818 / cost 1.19%，REF==COLD==WARM）——qfq 等价单测已证 cached==direct 数据层逐字节一致，真实 run 三轮一致是端到端实证。
    - **warm 轮零市场端点调用**：cold 后 coverage ledger = 68 market_daily + 68 adj_factor 行（68 成分全 ok）；warm 后**仍 68+68 不变**（零新 coverage 行 = 零 gap-fetch = 零 daily/adj 调用）。
    - wall：ref 998s / cold 1025s / warm **734s**（暖跑省 ~290s = 行情抓取部分；index_weight/财务/covariates 仍 live——P4-1 只缓存行情，诚实标注）。
    - secret scan：缓存 parquet + ledger 0 处 token / `.config.json`；ledger 列只有端点元数据。
  - **缓存命中直接可见（review follow-up）**：`TushareFeed.cache_stats()` 暴露各端点 gap-fetch 计数，`_load_panel` 经 run-scoped logger 打 `data cache: market_daily_gap_fetches=N adj_factor_gap_fetches=M`——冷跑非零、暖跑 0/0（warm rerun 实证 run_phase2_baseline.log 含 `0/0`，secret scan 0）。
- **不变量守住**：factor/alpha/portfolio/execution/OOS 切片/report 全不动；`artifacts/data/{output_name}.parquet` 不当 SoT。范围克制：P4-1 只 market_daily+adj_factor，其余端点（index_weight/daily_basic/fina_indicator/...）P4-2/P4-3 再缓存。
- ✅ **Phase 4-2 持久化 Tushare universe + tradability 缓存**（**PR #25 已 merge 到 `main`**，index_weight + suspend_d + namechange + stk_limit + stock_basic）：把 P4-1 端点级 raw 缓存扩到 universe/可交易性端点——真实 run 不再每次重抓成分股/停牌/ST/涨跌停/上市日。**默认仍 disabled（向后兼容，旧配置行为一字不变）**；opt-in 后这五端点走 read-through。
  - **三种规划形态共用一引擎**：① dense per-symbol 日期区间（`suspend_d`/`stk_limit`，复用 P4-1 gap + recent-tail）；② index_code 维日期区间（`index_weight`，coverage key=index_code，gap 内仍 **90 天分页**，raw 快照入库）；③ snapshot 维度（`namechange` per-symbol、`stock_basic` 全局 sentinel），用 `refresh_dimension_days`（默认 30）staleness + force_refresh（`CoverageLedger.snapshot_fetched_at` 给新鲜度判定）。
  - **语义全保**：`index_weight` 仍 PIT/as-of（370 天 pre-start lookback 进缓存请求区间，latest-snapshot-on-or-before 仍在 feed/universe）；`stk_limit` 仍 **raw price**（限价检查在 front-adjust 前，不碰 qfq）；`stock_basic` 只取 list_date 供 `min_listing_days`（缺失仍 kept+披露），current-tag `industry` 绝不入缓存/中性化；`suspend_d`/`namechange`/ST 区间形状不变；**缓存只存 raw 端点事实，不存派生 flag 作 SoT**。
  - **三 feed 接线**：`IndexConstituentsFeed`/`TushareFlagsFeed`/`TushareCovariatesFeed` 各加 `cache=None` 注入；cache present → read-through + **共享 finalizer**（cached==direct，限价 frame `assert_frame_equal`、ST 区间集合相等、suspend set / listing dict 相等单测锁定）；`cache=None` → 直抓路径**逐字不变**（旧 feed 测试原样过）。per-symbol 限流/重试仍在各 feed 的 `_call` 闭包里（缓存 transport-agnostic）。
  - **coverage ledger 复用 + 扩展**：11 列不变；**empty 算覆盖、failed 不算**（fetch 抛错则不记 coverage、留待重试，测试锁定）；ledger 列只有端点元数据，无 token/secret。
  - **单一共享 cache 贯穿 4 runner**（run_phase0/phase2/oos/subset）：`_build_cache(cfg)` 建一个实例线穿 `_build_universe`/`_load_panel`/`_enrich_tradability`/`_maybe_enrich_listing`，跑完所有缓存端点后 `_log_run_cache_stats` 打**一行** 7 端点统计；P4-1 的 market 行前缀保留（旧统计测试过）。
  - **配置**：`data.cache.refresh_dimension_days`（默认 30，>=0 校验）；旧配置全 validate（含未知键拒绝）。
  - **真实 smoke（phase2 baseline，fresh temp root，cold→warm；fresh root 不动已 merge 的 v1 缓存）**：
    - **cold 行**：`market_daily=68 adj_factor=68 index_weight=9 suspend_d=68 namechange=68 stk_limit=68 stock_basic=1`（全非零）；coverage = market/adj/namechange/stk_limit 各 68 ok + index_weight **1 ok**（整 gap 一行，内部 9 窗分页）+ stock_basic 1 ok + **suspend_d 68 empty**（SSE50 大盘股该窗口无停牌 → 空返回算覆盖，warm 不重抓）。
    - **warm 行：7 端点全 0**；coverage ledger **零新行**；report 指标与 cold/P4-1 cached baseline **逐数一致**（IC 0.0083 / annual −10.19% / maxDD −16.52% / vol 16.59% / sharpe −0.5703 / turnover 1.0818 / cost 1.19%）——cached==direct 二度端到端实证。
    - wall：cold **960s** / warm **366s**（暖跑省 ~594s = universe+tradability+market 抓取；`daily_basic`(market_cap)/`index_member_all`(pit_sw) 仍 live——P4-2 不缓存这俩，P4-3 再说，诚实标注）。
    - secret scan：缓存 parquet + ledger + 日志 + 报告 0 处 token 值 / `.config.json`；ledger 无 token 列。
  - **不变量守住**：factor/alpha/portfolio/execution/OOS 切片/report/`front_adjust` 全不动；`artifacts/data/*.parquet` 不当 SoT。范围克制：`daily_basic`/`fina_indicator`/`index_member_all` 仍留 P4-3。
- ✅ **分钟级 intraday pipeline I1–I4**（**PR #29 已 merge 到 `main`**，4 commit 一 PR；**全程与日频链路解耦——`factors`/`alpha`/`portfolio`、日频 `runtime/backtest`、日频 `TushareFeed`/`TushareCache`、全部 `config/` 零改动**，Phase 0/2/3 数字不变）。本地验收文档在 `tmp/context/intraday_pit_checkpoints/stage_i{1,2,3,4}_acceptance.md`（gitignored）。
  - **核心架构决策**：raw intraday SoT = **stk_mins `1min` only**；5/15/30/60min 是从缓存 1min **派生**的视图，不作独立 raw 上游产品抓取；日频 `D` 仍独立保留（不被分钟重采样替代）。三时间戳贯穿全程严格分离：**signal cutoff**（T 14:50，特征只用 `available_time<=cutoff`）/ **execution timestamp**（T 14:51）/ **holding period**（exec→next-exec，绝不 close-to-close）。
  - **I1（feat data）** 1min raw feed + PIT schema：`data/clean/intraday_schema.py`（独立 `MultiIndex(time,symbol)`，分钟精度不归零；`bar_end=trade_time`/`bar_start=bar_end-freq`/`available_time=bar_end+data_lag`；`RAW_INTRADAY_FREQ`+`ensure_raw_intraday_freq` 钉死 1min）+ `data/feed/tushare_intraday.py`（`TushareIntradayFeed.get_minutes`，`vol→volume`/`ts_code→symbol`/`trade_time→time/bar_end`，非 1min 在建 SDK client 前即拒，token 不入库不打印）。
  - **I2（feat cache）** stk_mins 1min read-through cache：`data/cache/intraday_coverage.py`（**timestamp-interval ledger**，`raw_freq/start_time/end_time`，不复用日频 date 语义；ok/empty 算覆盖 failed 不算）+ `intraday_parquet_store.py`（**月分区** `stk_mins_1min/freq=1min/symbol_prefix/symbol/year/month.parquet`，原子幂等 upsert by `(symbol,freq,bar_end)`，只存 raw）+ `intraday_cache.py`（`TushareIntradayCache.stk_mins_1min`：交易日 gap 规划复用 `intervals.py`，≤23 日窗分页 <8000 行 cap，1min-only guard）。接入 `TushareIntradayFeed` 可选 cache path（直抓路径字节级不变）。**冷写/暖零调用/partial 只补缺口/empty 记录/failed 重试/cached==direct 规范化后逐字节等价** 全测试锁定；**日频 `TushareCache` 零改动**（独立类）。
  - **I3（feat data）** 分钟→日频 PIT 聚合：`data/clean/intraday_aggregate.py::asof_daily_features`（默认 `decision_time=14:50:00`；**先 `available_time<=cutoff` 逐 bar 过滤再按日 groupby**，分钟时间戳绝不先归零进日频）。cutoff 编码列名 `intraday_ret_0930_1450`/`intraday_realized_vol_0930_1450`/`intraday_vwap_0930_1450`/`intraday_last30m_ret_1420_1450`。`resample_intraday_bars` 派生粗 bar `available_time=max(source_1min.available_time)`。泄漏测试（扰动 14:50 后 bar → 特征不变）+ 可见性排除测试锁定。
  - **I4（feat runtime）** 尾盘调仓 execution 骨架：`runtime/intraday_execution.py::simulate_tail_rebalance`（`next_minute_close` 模型：14:51 或窗口内最早 bar 成交；持有期收益 `exec(T)→exec(T_next)` 非收盘；缺 bar/NaN/无窗口 bar → 可解释 blocked，**不静默用日收盘替代**；只读分钟 bar 无 EOD 泄漏）。**独立函数,未接 config/pipeline**；日频 `close_to_next_period` 不变。`tail_vwap`/`closing_call_proxy` config 层拒绝（注明 future，后者需 `stk_auction_*` 权限当前无）。
  - **下一步**：执行真实性先于研究因子 → I5b 已补执行期涨跌停可行性（见下）；之后 I5c 再把 EXPLORATORY 分钟因子作为真实 opt-in alpha 端到端接 I2→I3→I4→event engine（报告披露 cutoff/lag/execution_model/window），日频回归不破。
- ✅ **P4-3 因子支撑端点缓存 + 21:00 data updater**（**PR #31 已 merge 到 `main`**，daily_basic + fina_indicator + index_member_all 进既有日频 `TushareCache`；新增独立 `data-update` CLI 增量暖跑，**不跑 factor/alpha/portfolio/backtest、不写 PanelStore**，日频回测仍自走 read-through 补缺口）。本地验收：`tmp/context/intraday_pit_checkpoints/stage_p4_3_data_updater_acceptance.md` + `tmp/context/session_handoff_20260613/p4_3_codex_acceptance.md`（codex 复核 PASS）。
  - **daily_basic**（dense，pe/pb/total_mv，一次缓存调用喂 market_cap+value_ratios）；**index_member_all**（per-symbol SW in/out 维度，staleness 刷新）。
  - **fina_indicator 字段集无关**（codex acceptance blocker 修复）：cache **永远存 canonical `FINA_FIELDS` superset**（roe/netprofit_yoy/grossprofit_margin），feed 读时选子集——一个配置 warm 不会阻塞另一配置（光加 fields_hash 会在 upsert 时互相覆盖）；drift 测试守 `FINA_FIELDS ⊇ financial.SUPPORTED_FIELDS`；coverage 仍按 report-period end_date + 长 trailing tail 抓晚披露，ann_date 作 raw 不当覆盖轴。
  - **not_ready pending window**：今日(`not_ready_days`)空返回记 `not_ready`(非 coverage) 次跑重试，跨界 gap 拆分；`not_ready_days=0`(默认) 行为逐字不变。per-endpoint trailing tail + summary 计数。
  - cached==direct 实证（market_cap/value_ratios/pit_sw/fina-as-of）；`cache=None`→direct 逐字不变；缓存只存 raw（无 qfq/因子/token）；日频/分钟各自 ledger/store。**日频回测数学/factor/alpha/portfolio/runtime 零改动，Phase 0/2/3 不变**。
- ✅ **P-I5a 事件驱动回测架构重构 + opt-in 分钟尾盘 event model**（**PR #33 已 merge 到 `main`**，架构/框架 PR，非研究结果；含 review 两修复）：把回测层重构成**共享事件驱动核心**，日频 close-to-close 与分钟尾盘成为**同一 achieved-book ledger 上的两个 event model**，零重复 fill/cash/settle。本地验收：`tmp/context/session_handoff_20260615/stage_i5a_acceptance.md`。
  - **核心**（`runtime/backtest/`）：`events.py`（`HoldingPeriod` 显式时间基 + 共享月度日历）/ `engine.py`（`BacktestEngine` 单循环：universe→scores→构建→可行成交 via `SimExecution`/`simulate_fills`→settle→NAV/feasibility/holdings/**event** 日志）/ `event_models.py`（`DailyCloseEventModel` 逐字节复刻日频 + `IntradayTailEventModel` 14:50 决策/14:51 成交/exec-to-exec，复用 `intraday_execution.build_execution_prices`）。`BacktestDriver` 改为 engine+DailyCloseEventModel 的**薄 wrapper**（−237/+167），`pipeline`/`oos`/`phase2` 三 runner 一行未改。
  - **config**：opt-in `intraday`（enabled/decision_time/data_lag/session_open/execution_model/execution_window/require_cache_coverage/missing_execution）+ `backtest.event_order` Literal；`intraday_tail_rebalance` 必须 `intraday.enabled=true`（root validator），旧配置全 validate，非法 model/window 可读报错。
  - **分钟尾盘语义**：决策只用 `available_time ≤ 14:50` 的 bar；成交取执行窗口 `[14:51,14:56:59]` 内最早 1min 收盘；持有期收益 `exec(T_next)/exec(T)−1`（**绝非 close-to-close**）；**缺/NaN 执行 bar = 显式 block，绝不退回日收盘**；turnover/cost/holdings 按 achieved book；闲置现金 `cash_return`。事件时间戳全程可审计。
  - **新 CLI `run-phase-i5a-intraday`** + `config/phase_i5a_intraday_tail_framework.yaml`（SSE50 SH/SZ smoke）：日频面板/universe 走既有 P4 缓存；分钟 bar **只读既有 intraday 缓存**（缺即 hard blocker，绝不静默暖跑 → **零 `stk_mins` live call**）；PIT-safe 分数 = I3 `intraday_ret_0930_1450` 特征（纯框架验证，非业绩声明）。
  - **真实 smoke（SSE50，2026-03-03→06-12，~180s）**：`periods=3 / covered=58/58 / stk_mins_live_calls=0 / blocked_fills=0`；event 表显示 14:50 决策 / 14:51 执行 / **actual exec bar 范围** / 真实 exit 执行锚；secret scan 0。
  - **review 两修复**：① `require_cache_coverage=true` 下**任一** symbol 缺即 loud fail（原仅全缺才失败 → 静默丢名引入样本覆盖偏差）；② `HoldingPeriod.exit_execution_ts` 显式化，event_log + 报告显示 **planned vs ACTUAL 执行 bar 时间**（14:51 缺、14:52 成交可见），末期不再 NaT。
  - **不变量守住**：daily close-to-close 行为不漂移（engine==driver 黄金测试 + phase0 `ic 0.9600/annual 0.8408` 不变）；无重复 ledger；缺 bar 绝不日收盘兜底；turnover/cost 按 achieved。
- ✅ **P-I5b 分钟尾盘执行期价格涨跌停可行性**（**PR #35 已 merge 到 `main`**，执行真实性硬化 PR，非研究 alpha；含 review LOW 修复）：补 P-I5a 显式 deferred 的限制——在「有效 1min 执行 bar」规则之上，用 raw `stk_limit` 对**选定执行分钟的 raw 1min 收盘**做方向感知闸门。本地验收：`tmp/context/session_handoff_20260615/stage_i5b_acceptance.md`。
  - **语义（测试锁定）**：`exec ≥ up_limit − tol → can_buy=False`（涨停**只挡买**）/ `exec ≤ down_limit + tol → can_sell=False`（跌停**只挡卖**）/ 缺·NaN 执行 bar 仍**先于**限价逻辑挡两方向；**RAW-vs-RAW**——raw 1min 收盘（intraday 缓存存未复权）vs raw `stk_limit`，绝不碰 qfq/日收盘/`at_up_limit·at_down_limit` 日收盘派生 flag。`can_buy/can_sell` 喂既有 `simulate_fills` achieved-book ledger（turnover/cost/holdings 按实际成交）。
  - **缺限价行绝不静默当通过**：严格 `require_price_limit_coverage=true` 在**模型构造期 raise**（出结果前列缺失 pair）；宽松 `false` 计数披露 unchecked 并退回 bar-exists 规则。诊断（涨停挡买/跌停挡卖/unchecked/coverage）按 `(date,symbol)` 幂等，重复 `feasibility()` 不重计。
  - **接线**：`IntradayCfg.{price_limit_check（默认 false）/require_price_limit_coverage/limit_tolerance（≥0 校验）}` + `OutputCfg.intraday_report_name`（I5b 报告独立成文件，不覆盖已验收 I5a artifact，沿 baseline_report_name 先例）；runner 经 `TushareFlagsFeed(cache=cache).limits()` 走既有 P4 read-through 缓存取 raw 限价（**不加新端点**），报告 H1/intro 按 check 自报 I5b（修同类 stale-title）。
  - **不变量守住**：`BacktestEngine`/`BacktestDriver`/`DailyCloseEventModel`/`fills`/`execution` **零改动**（限价 map 仅 check 开时构建，默认 false → `feasibility()` 逐字退回 I5a bar-exists 规则）；日频 close-to-close 不漂移。
  - **真实 smoke（SSE50 2026-03-03→06-12，3 次跑 NAV 全 `0.976448` 确定性）**：`covered=58/58 / stk_mins_live_calls=0`；限价覆盖 **174/174**；**1 笔涨停挡买 / 0 笔跌停挡卖** → NAV 由 I5a 的 0.952207 变到 **0.976448**（2026-04-30 一只 14:51 封涨停的票被正确挡买，该期只持 9 只——I5a 只查 bar 存在会「买」进涨停板，**这正是 I5b 补的执行真实性，非业绩声明**）；`stk_limit` gap-fetches=58（冷暖都 58：窗口尾端距今 3 天落在 `refresh_recent_days=14` 内，P4 按策略重抓 tail，据实报告，非 minute fetch）；secret scan 报告+日志+58 缓存 parquet **0 处**。
- ✅ **P-I5c MMP 分钟因子端到端 opt-in alpha**（**PR #37 已 merge 到 `main`**，**EXPLORATORY**，非业绩声明）：把首个 EXPLORATORY 分钟因子作为真实 opt-in alpha 端到端接 I2→I3→I4→event engine，日频回归不破。
  - **MMP（Minute Microstructure Pressure）**（`data/clean/intraday_aggregate.py`，公式/PIT/前导 NaN/跨 symbol 隔离全测试锁定）：逐 1min bar `mid=(high+low)/2`、`S=(close−mid)/mid`、`V=√(volume/median(vol[t-20:t]))`、`B=|close−open|/(high−low+eps)`、`R=(high−low)/(mean(hl[t-20:t])+eps)`、`MMP_t=S·V·B·R`（eps=1e-6）；日频分数 = 在场 bar `[session_open, decision_time]` 等权均值，rolling baseline 用前 20 个在场 bar（首 20 NaN）。**先 `available_time≤cutoff` 过滤再聚合**（PIT-safe，绝不先归零）。
  - **可配置 `intraday.score_feature`**（默认沿用，`mmp_ew` opt-in）+ 可配置报告标题（修同类 stale-title）；`config/phase_i5c_mmp_minute_factor.yaml`。decision feature→alpha→portfolio→event engine 全程复用既有机器，报告披露 cutoff/lag/execution_model/window。**日频 close-to-close 零漂移**（phase0 `0.9600/0.8408` 不变）。
- ✅ **P-I5d MMP 五分位分组回测**（**PR #39 已 merge 到 `main`**，**standalone against `main`**，**EXPLORATORY，非业绩声明、非调参**）：把 I5c MMP 日频分数 `intraday_mmp20_ew_0930_1450` 做 5 等额分位分组回测——每月调仓日按 PIT-safe 分数横截面排名切 `analytics.quantiles=5` 等数桶（Q1 最低 / Q5 最高），每组作独立 long-only 等权组合走**同一** `BacktestEngine`+`IntradayTailEventModel`+`SimExecution(fee_rate=0.001)`，I5b raw `stk_limit` 执行期涨跌停 ON。引擎/执行可行性/MMP 因子数学全不变,仅新增分组 + per-group 编排。本地验收：`tmp/context/session_handoff_20260615/stage_i5d_acceptance.md`。
  - **新增**（机器零改动复用 I5a/I5b/I5c）：`qt/intraday_groups.py`（等额 rank 桶 / `GroupScores` / `EqualWeightAll`）、`qt/intraday_group_backtest.py`（cache-only anchor-date-sliced 分钟加载、一份共享 exec-price 矩阵跨 N 个 fresh per-group 模型、QN−Q1 合成 spread）、`qt/intraday_group_figures.py`/`intraday_group_report.py`（NAV/spread/metric 图 + 报告）、`config/phase_i5d_mmp_quintile_5y.yaml`；唯一 runtime 改动 `IntradayTailEventModel` 可选 `precomputed_prices`（默认 None → I5a/I5b 逐字节不变）。
  - **真实 run（CSI500 `000905.SH`，2021-06-01→2026-05-31，59 月调仓，~95min）**：`covered=892/995`（**103 只分钟未覆盖成分被 drop 并披露，5 年窗口不缩短**）/ `stk_mins_live_calls=0`（cache-only）/ `fee_rate=0.001`；final NAV Q1→Q5 **0.9822 / 1.0423 / 1.0495 / 1.1275 / 1.1577**（**单调 Spearman 1.0000**，等额 rank 桶对 MMP 肥尾稳健）；合成 Q5−Q1 spread +0.29%/期、累计 **+17.15%**；限价覆盖 52,584/52,584（严格 `require_price_limit_coverage=true` 过）。
  - ⚠️ **caveat（保留，不移除）**：单因子/单重叠 5 年窗口/单 universe,无调参/无 robustness/无 learned 权重;`covered=892/995`(103 drop);`stk_mins_live_calls=0`;**Q5−Q1 是合成 long-only 腿差,非单独执行的 dollar-neutral 组合**;执行模型仍缺 partial-fill / liquidity / volume-cap;**Q5>Q1 不作业绩声明**。这是项目首个正向 intraday 信号,下一步需独立泛化(CSI300/中证1000 或 disjoint 窗口,机器冻结)。
  - **PR 处置**：#37（I5c）随 #39 落地后其 head 成 `main` 祖先,GitHub **自动 merged**(非本次手动 merge);#38（i5d stacked on i5c,base 非 main）已被 #39 **完全取代**(零 remaining commit/file diff vs main),保持 **OPEN 待清理**(未获显式授权关闭)。
- ✅ 质量门：`pytest` **577 passed**（P0=97 / P1=78 / P2-1=22 / P2-2=22 / P2-3=14 / P2-4=8 / P3-1=10 / P3-2=18 / P3-3=16 / P3-4=15 / P3-5=22 / P3-6=27 / P3-7=25+1 throttle / P3-8=8 / P4-1=28 / P4-2=17 / **intraday I1 schema=14 + feed=9 / I2 cache=10 / I3 aggregate=13 / I4 execution=10 / P4-3 cache=10 + updater=7 / P-I5a event-backtest=19 / P-I5b exec-feasibility=16 / P-I5c mmp-minute=18 / P-I5d mmp-quintile=19**）；`ruff` clean；`validate-config`（全部 17 配置含 data_update + phase_i5a + phase_i5b + phase_i5c + phase_i5d）+ `run-phase0`（demo）均 OK（**ic 0.9600 / annual 0.8408 不变**）。
- ⚠️ 剩余（已显式披露）：日线 only、demo 路径非真数据、旧三因子无信号（P3-3/P3-4 实证;但其组合在 2024-2026 holdout 上 SSE50/CSI300/CSI500 全正、CSI500 高达 +17.8%——小样本/regime 翻转的持续例证）;value/低波信号获得**独立样本符号级确认**（P3-7 SSE50/CSI300 量级衰减;P3-8 CSI500 泛化成立且更强），组合级盈利能力仍未确立（排名跨 cell 翻转）;subset 报告文件名已可配置（P3-8 起不再互覆盖）。
- 路线图下一步：**I5d 独立泛化**（MMP 五分位单调性是项目首个正向 intraday 信号,但只一个重叠 5 年窗口/一个 universe → 在第二个 universe（CSI300/中证1000）和/或 disjoint 窗口上机器冻结复跑分组回测,使单调性可归因于因子而非该 regime,沿 P3-7/P3-8 独立确认先例）；**I5b/执行 follow-up**（执行期 feasibility 可再扩 partial-fill / liquidity / volume cap，size-aware 读数前补）；**P4-3 follow-up**（fina 按 ann_date 建披露日历 ledger 以去掉启发式 tail / updater 加 stk_mins 历史回填 / data-update summary 落 cache-stats artifact）；**数据层 D1+**（cache/store 契约已文档化,token 解析已收敛,后续 D2+ 再做 cache 内部拆分/schema registry/data-quality/并发）；研究侧：更长 holdout 滚动复检 / 成本模型细化。
