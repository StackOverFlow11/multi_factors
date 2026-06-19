# 数据层契约（Data Layer Contracts）

> 目的：把 **`cache`（缓存）** 与 **`store`（面板存储）** 的边界写成可提交的契约，
> 让后续改动有明确不变量可守。本文件只描述 **当前已实现** 的行为，不预告未实现的阶段。

已实现：**D1**（边界文档化 + 低风险 token 解析去重）、**D2**（`TushareCache` endpoint specs/parsers 拆分，公开缓存行为不变）、**D3**（report-only `data/quality/` 数据质量层）、**D3b**（默认关闭的 `data-update` 质量报告钩子，report-only）、**D4**（coverage ledger `record_many` 批量写 + 进程内查找缓存，公开路径/列/语义不变）、**D5**（opt-in 有界并发的 updater/缓存暖跑取数 + 单一全局限频器，默认串行）。**D6 尚未实现**，本文件不声称其存在。

---

## 1. `cache` vs `store` —— 两层职责互不替代

| 层 | 路径 | 是什么 | 不是什么 |
|---|---|---|---|
| **缓存（cache）** | `data/cache/` | **可复用的 endpoint 级 raw 缓存**，是行情/成分/可交易性/财务等原始事实的 source of truth（SoT） | 不是派生数据（不存 qfq、不存因子、不存任何 flag 派生值） |
| **面板存储（store）** | `data/store/panel_store.py`（`PanelStore`） | **单次 run 的 canonical 面板 artifact**（per-run），供该次回测/分析消费 | **不是 raw data lake，不是缓存的 SoT**；`artifacts/data/*.parquet` 绝不当缓存权威 |

要点：

- **缓存是跨 run 复用的原始数据底座**；`PanelStore` 是某一次 run 内组装好的面板快照。
  两者方向不同：缓存向上游（API）负责，`PanelStore` 向下游（本次回测）负责。
- **缓存只存 raw**。`data/cache/` 下的持久化行（`parquet_store.py` / `intraday_parquet_store.py`）
  一律是未派生的原始 endpoint 事实；覆盖账本（`coverage.py` / `intraday_coverage.py`）只记
  endpoint 元数据，**不含 token、不含 secret**。
- 日频缓存（`tushare_cache.py`）与分钟缓存（`intraday_cache.py`）各自独立的 ledger / store，互不串用。

## 2. 缓存只存 raw —— 所有派生/对齐/校验留在下游

以下逻辑 **不属于缓存层**，必须留在缓存的下游（feed/clean/universe/factors/alpha/portfolio/runtime）：

- **前复权 `front_adjust`**：缓存存未复权 OHLCV + 原始 `adj_factor`；qfq 在内存按现有公式/时机计算。
- **PIT 指数成分**：`index_weight` raw 快照入缓存；as-of（latest snapshot ≤ d）在 feed/universe 下游判定。
- **PIT 申万行业**：`index_member_all` 的 in/out 区间 raw 入缓存；按 trade_date as-of 取行业在下游。
- **财务 `ann_date` 披露日对齐**：财务字段 raw 入缓存；逐字段 `ann_date ≤ trade_date` 的 as-of 在下游。
- **raw 涨跌停可行性检查**：`stk_limit` raw 价入缓存；限价闸门用 raw（绝不碰 qfq）在执行层判定。
- **factors / alpha / portfolio / runtime 的全部数学**：均在缓存下游，缓存不参与。

换言之：**缓存换不换、暖不暖，下游数学逐字节不变**。这是缓存层 opt-in、默认 disabled 的前提。

## 3. `data-update` 只暖缓存，不跑研究

- `data-update` CLI（`qt/data_updater.py::run_data_update`）**只做 endpoint 级增量暖缓存**。
- 它 **不跑 factor / alpha / portfolio / backtest，不写 `PanelStore`**。
- 真实回测仍各自走 read-through，按需补自己的缺口；`data-update` 只是把常用 endpoint 提前填好。

## 4. D1 / D2 / D3 / D3b / D4 / D5 已做什么；D6+ 范围（未实现）

**D1**（行为零改动）做两件低风险事：**把上述边界写成本契约文档**，以及 **把 `TushareFeed` /
`IndexConstituentsFeed` 里重复的 token 解析收敛到共享的 `data/feed/secret.py::read_token`**
（其余 feed 早已使用该共享读取器）。

**D2**（行为保持型重构，公开缓存语义不变）把 `data/cache/tushare_cache.py` 的内部拆成小文件:
endpoint 常量/specs → `data/cache/tushare_specs.py`、raw endpoint 解析器 → `data/cache/tushare_parsers.py`、
两个叶子规划 helper（`_fields_hash`/`_compact`）→ `data/cache/tushare_planning.py`;`TushareCache` 仍是
公开门面（方法/签名/gap 规划/分页/staleness/coverage 语义全不变),并 re-export endpoint ids + `FINA_FIELDS`
保持向后兼容导入。

**D3**（report-only 数据质量层，库 + 测试）新增独立的 `data/quality/` 包,在接入处附近**只报告**可疑的上游
日频行情 / `adj_factor` / 1min 分钟数据,**绝不**过滤行、修复值、改 qfq、改 cache coverage、或动 feed/factor/
alpha/portfolio/runtime。纯函数:输入 DataFrame → findings(含 dataset/check 元数据 + bounded 样本),输入永不被
改;findings/渲染报告携带 redaction guard,不含 token/secret 路径/无界 dump。

**D3b**（默认关闭的 `data-update` 质量报告钩子）把 D3 质量层接入运维：新增严格的 `data_update.quality`
配置(`enabled` 默认 **false**,所以所有现存配置行为不变)。开启后,`data-update` 在它 **已经暖好的**帧
(market bars + 1min 分钟,**不额外打 API**)上跑 D3 的**结构**检查,并把确定性 Markdown 报告写到
`output.report_dir`(clean 也写)。可检查端点仅限 updater 以帧形式加载的结构面:`market_daily` / `adj_factor`
/ `stk_mins_1min`;报告只含 bounded/redacted findings + 窗口/符号**数量**,不含 secret 路径/token/无界 symbol
dump。**report-only**:不过滤/修复/改数据,不让 job 失败,不改 cache coverage / 每端点请求 summary,不动
feed/factor/alpha/portfolio/runtime/backtest 语义(日频 close-to-close 不漂移)。不合成交易日历(仅结构检查;
缺日期/缺分钟的 D3 检查在 D3b 关闭)。

**D4**（行为保持的缓存内部扩展）给日频 `CoverageLedger` 与分钟 `IntradayCoverageLedger` 加 `record_many(...)`
批量追加(一次归一化 + 一次原子 parquet 写;空输入 no-op;每行与单行 `record` 逐字归一),`record(...)` 委托给
`record_many([row])`;并加进程内帧缓存 + 按 `(endpoint, key[, raw_freq])` 的查找 memo,让重复 lookup 不再重读重
过滤整张 parquet(按文件 mtime 失效,外部写绝不被当陈旧服务;`read()` 返回 copy 防污染)。`TushareCache` 的
not-ready 拆分用**一次** `record_many` 写其 1–2 行覆盖。**公开方法/构造、parquet 路径(`manifest/coverage.parquet`
/ `coverage_intraday.parquet`)、`LEDGER_COLUMNS`/`INTRADAY_LEDGER_COLUMNS` 名序、coverage 语义(仅 ok/empty 算
覆盖;failed/not_ready 不算;snapshot 取最新成功 fetched_at)全不变**。

**D5**（opt-in 有界并发 + 单一全局限频器）给 `data-update` / 缓存暖跑的取数阶段加可选并发。新增
`data/feed/scheduler.py::GlobalRateLimiter`(线程安全 ticket 限频器,锁内预定下一时隙、锁外 sleep,N 个 worker 汇入
**一个**每分钟预算 —— 配额绝不按线程倍增;`monotonic`/`sleep` 可注入,只见整数预算,无 token/secret);
`request_with_retry(..., scheduler=...)` 每次 attempt(含重试)前 acquire 全局时隙,且不再做 per-call rate sleep。
**默认 `data_update.concurrency.max_workers=1` 保持全串行**(所有现存配置照常 validate、行为字节级不变)。
`max_workers>1` 时:**只有日频密集 per-symbol 取数**(`market_daily`/`adj_factor`/`suspend_d`/`stk_limit`/
`daily_basic`/`fina_indicator`)走有界线程池,**store upsert + coverage ledger 写仍在主线程按计划序**(成功 gap 先
durable 再抛首个失败,失败 gap 不记覆盖、可重试);**snapshot、`index_weight` 90 天分页、intraday 缓存在 D5 仍串行**
(并发开启时它们仍共享同一限频器)。`cache` 仍是 raw endpoint SoT、`PanelStore` 仍是 per-run 面板 artifact,缓存/
store 边界不变;factor/alpha/portfolio/runtime/backtest 数学不变。

以下明确 **仍未实现**（D6+,本文件 **不声称已实现**）：

- endpoint schema registry（改运行时 dispatch 语义）；
- `PanelStore` 的 append/partition / 可选物化派生面板存储（**D6**,仅当因子研究需要可复用派生面板时才启动）。
