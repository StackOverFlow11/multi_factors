# 数据层契约（Data Layer Contracts）

> 目的：把 **`cache`（缓存）** 与 **`store`（面板存储）** 的边界写成可提交的契约，
> 让后续改动有明确不变量可守。本文件只描述 **当前已实现** 的行为，不预告未实现的阶段。

本文件对应 **D1**（边界文档化 + 低风险 token 解析去重）。**D2–D6 尚未实现**，本文件不声称其存在。

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

## 4. 不属于 D1 的范围（后续阶段，未实现）

以下明确 **不在 D1**，本文件 **不声称已实现**：

- 数据质量校验（data-quality validator）；
- 并发 / 线程池 / 异步抓取（concurrency）；
- `TushareCache` 内部拆分、endpoint schema registry、`CoverageLedger` 存储格式变更、
  `PanelStore` 的 append/partition 特性。

D1 仅做两件低风险事：**把以上边界写成本契约文档**，以及 **把 `TushareFeed` / `IndexConstituentsFeed`
里重复的 token 解析收敛到共享的 `data/feed/secret.py::read_token`**（其余 feed 早已使用该共享读取器）。
数据行为零改动。
