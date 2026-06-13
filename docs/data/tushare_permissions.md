# Tushare 接口权限 / 限频实测

> 目的：区分 **有没有权限** 与 **能不能批量用**。MCP 单发探测只能证明权限（调一次通），
> 证明不了限频；**本表以 Python SDK 实测为准**，MCP 仅作辅助参考。

## 探测元信息

| 项 | 值 |
|---|---|
| 探测日期 | **2026-06-13** |
| Token 来源 | 外部 config：`financial_projects/.config.json`（key `tushare.token`）——**值绝不入库/不打印** |
| SDK | `tushare` **1.4.29**（`pro.query`，conda env `data_fetch`）|
| 方法 | 每接口 **1 次最小查询**（单只 `000001.SZ` + 单日/单报告期；`index_weight` 用单月区间）。**不暴力压测限频**；限频以 SDK 实际返回的「X 次/单位时间」为准 |
| 原始结果 | `artifacts/permissions/tushare_probe_20260613.json`（gitignored，无 token）|

> **更新 2026-06-13（同日复测）**：`stk_mins` 的「1 次/小时」限频**已放开**——SDK 连续 3 次小查询
> （不同交易日窗口、间隔 ~4s）**3/3 全通、约 12s 内完成**。判定从 `rate_limited` 改为
> `authorized`（exact 每分钟上限未压测）。分钟级数据现在**数据层可行**（路线图 architecture.html §11）。
> `hm_detail` 本轮**未复测**，其「1 次/小时」结论维持原状。

### 状态枚举

| status | 含义 |
|---|---|
| `authorized` | 有权限，本次返回了数据 |
| `empty_authorized` | **有权限**，但该样本（这只票/这天）无行——不是无权限 |
| `rate_limited` | 有权限，但触发限频（**关键：能调≠能批量**）|
| `no_permission` | 无权限（40203），需单独/更高档开通 |
| `transient_error` | 网络等瞬时错误（本次无）|

⚠️ **限频未逐个压测**：返回数据的接口只证明「这一次能调」，**未测每分钟/每天上限**。首轮探测时
`stk_mins`、`hm_detail` 因本会话先前 MCP 探测已各消耗掉额度，SDK 复测暴露 `1 次/小时`——
但 **`stk_mins` 同日再复测 3/3、限制已放开**（见上方更新说明），`hm_detail` 未复测、维持 1 次/小时。
其余 `authorized` 接口的批量额度**尚未核实**，接入前需各自做限频体检。

## 总表（SDK 实测）

### 项目数据层（P0–P4-2 已在用，且真实 run 已大规模拉取 = 批量可用已证）

| 接口 | 用途 / 项目相关 | probe params | SDK 结果 | 权限 | 批量可用 | PIT timing | 备注 |
|---|---|---|---|---|---|---|---|
| `daily` | P0 OHLCV（raw，下游 qfq） | ts+d | authorized (1 行) | ✅ | **已证**（真实 run） | T 日收盘后可得 T | P4-1 已缓存 |
| `adj_factor` | P1 复权因子 | ts+d | authorized (1) | ✅ | **已证** | T 日 | P4-1 已缓存 |
| `index_weight` | P1 PIT 成分 | 300+月区间 | authorized (600) | ✅ | **已证** | as-of 快照（latest ≤ date）| P4-2 已缓存；90 天分页 |
| `suspend_d` | P2-2 停牌 | ts+月区间 | empty_authorized (0) | ✅ | **已证** | T 日 flag | 样本无停牌≠无权限；P4-2 已缓存 |
| `namechange` | P2-2 ST 区间 | ts | authorized (8) | ✅ | **已证** | dimension（in/out 区间）| P4-2 已缓存 |
| `stock_basic` | P2-2 list_date（min_listing_days）| ts | authorized (1) | ✅ | **已证** | dimension（全局快照）| P4-2 已缓存 |
| `stk_limit` | P2-2 **raw** 涨跌停价 | ts+d | authorized (1) | ✅ | **已证** | T 日 **raw 价**（front-adjust 前比对）| P4-2 已缓存 |
| `daily_basic` | P2-3/P3-5 pe/pb/total_mv | ts+d | authorized (1) | ✅ | **已证** | **当日发布，same-day PIT-safe** | P4-3 待缓存 |
| `fina_indicator` | P3-1 roe/np_yoy（ann_date as-of）| ts+period | authorized (1) | ✅ | **已证** | **按 ann_date 披露后才可用** | 单次最多 100 行；P4-3 待缓存 |
| `index_member_all` | P2-3 PIT SW 行业 | ts | authorized (1) | ✅ | **已证** | in/out 区间 as-of | P4-3 待缓存 |
| `income` | 财务源（ann_date）| ts+period | authorized (1) | ✅ | 同 fina | 按 ann_date 披露 | 目前未进因子，留作财务源 |

### EXPLORATORY 候选源（权限有，但**批量额度未核实**——接入前各自限频体检）

| 接口 | 用途 / 项目相关 | probe params | SDK 结果 | 权限 | 批量可用 | PIT timing | 备注 |
|---|---|---|---|---|---|---|---|
| `stk_factor` | 技术因子（MACD/KDJ/RSI/BOLL…）| ts+d | authorized (1) | ✅ | **未压测** | T 日（由 EOD 计算）| 候选 |
| `stk_factor_pro` | 技术因子专业版（数百列，bfq/qfq/hfq）| ts+d | authorized (1) | ✅ | **未压测** | T 日 | 候选 ⭐ |
| `cyq_perf` | 每日筹码及胜率 | ts+d | authorized (1) | ✅ | **未压测** | ⚠️ **盘后 18–19 点更新 → T 日盘中/收盘不可用，T+1 才 PIT-safe** | 候选 ⭐ |
| `cyq_chips` | 每日筹码分布 | ts+d | authorized (104) | ✅ | **未压测** | ⚠️ **盘后 18–19 点** | 候选 ⭐ |
| `moneyflow` | 个股资金流（大小单）| ts+d | authorized (1) | ✅ | **未压测** | ⚠️ **盘后** | 候选 |
| `moneyflow_dc` | 东财个股资金流 | ts+d | authorized (1) | ✅ | **未压测** | ⚠️ **盘后**（数据起 2023-09-11）| 候选 ⭐ |
| `moneyflow_ths` | 同花顺个股资金流 | ts+d | empty_authorized (0) | ✅ | **未压测** | ⚠️ **盘后** | 样本空≠无权限 |
| `moneyflow_hsgt` | 北向/南向资金 | d | authorized (1) | ✅ | **未压测** | ⚠️ **盘后** | 候选 |
| `limit_list_d` | 涨跌停/炸板统计 | d+U | authorized (52) | ✅ | **未压测** | ⚠️ **盘后**（数据起 2020）| 候选 ⭐ |
| `top_list` | 龙虎榜明细 | ts+d | empty_authorized (0) | ✅ | **未压测** | ⚠️ **盘后** | 样本空（000001 当日未上榜）|
| `block_trade` | 大宗交易 | ts+d | empty_authorized (0) | ✅ | **未压测** | ⚠️ **盘后** | 样本空 |
| `margin_detail` | 两融明细 | ts+d | authorized (1) | ✅ | **未压测** | ⚠️ **盘后** | 参考 |
| `stk_surv` | 机构调研记录 | ts+月区间 | authorized (1) | ✅ | **未压测** | 按 surv_date（事件日）| 另类数据 |
| `stk_mins` | **分钟线** 1/5/15/30/60min | ts+小窗 | authorized (6) — **复测 3/3** | ✅ | **限频已放开**（每分钟上限未压测）| intraday（无 PIT 滞后）| ⭐ 路线图分钟级现在数据层可行；接 ETL 仍需缓存层+throttle |

### 限频到不可批量 / 无权限

| 接口 | 用途 | SDK 结果 | 判定 | 说明 |
|---|---|---|---|---|
| `hm_detail` | 游资交易每日明细 | `频率超限(1次/小时)` | ⚠️ **callable but NOT batch-viable** | 限 **1 次/小时**；本轮未复测 |
| `stk_auction_o` | 开盘集合竞价 | `没有接口访问权限(40203)` | ❌ **no_permission** | 集合竞价家族需**单独开通**，5000 仍不够 |
| `stk_auction_c` | 收盘集合竞价 | `没有接口访问权限(40203)` | ❌ **no_permission** | 同上 |

## PIT 安全提醒（接候选源前必读）

- **盘后数据**（`cyq_*` / `moneyflow*` / `limit_list_d` / `top_list` / `hm_detail` / `block_trade` /
  `margin_detail`）：T 日数值 **T 日收盘后**（部分晚到 18–19 点）才有。**绝不能用 T 日的该值参与 T 日
  收盘选股**——会引入未来函数。只能当作「T 日已实现」、在 **T+1 调仓**时使用。接入时按 `factors` 层
  不变量（factor 不碰未来收益）+ 现有 `ann_date` as-of 那套纪律处理。
- **当日发布且 PIT-safe**：`daily_basic`（pe/pb 基于当日收盘，已按 same-day 使用）。
- **披露日对齐**：`fina_indicator` / `income` 按 `ann_date` 才可用（已实现）。
- 任何候选源进 P3 矩阵仍走 **EXPLORATORY + 独立样本** 那套（POST-HOC 披露、跨 cell 复检）。

## 结论摘要

- **5000 积分基本打通全部「特色数据」权限档**（筹码、技术因子 pro、资金流、龙虎榜、涨跌停统计、
  机构调研…），相比 3000 解锁一大批候选源——但**权限 ≠ 批量可用**。
- **需购买 / 无权限**：`stk_auction_o`、`stk_auction_c`（集合竞价家族）。
- **分钟线已放开**：`stk_mins` 同日复测 3/3，「1 次/小时」限制已解除——分钟级研究数据层现在可行
  （每分钟上限未压测；接 ETL 仍要走缓存层 + throttle）。
- **仍限 1 次/小时、不可批量**：`hm_detail`（游资明细，本轮未复测）。
- **未压测限频（authorized 但批量额度未知）**：全部 EXPLORATORY 候选源（含已放开的 `stk_mins`）——接入前各做一次小批量限频体检
  （项目已有 throttle+retry，正好测每接口 `X 次/分`）。
- **不要让生产代码依赖本表自动跳过**：本表与 `data/feed/tushare_capabilities.py` 仅供人审阅 / 规划；
  运行时是否有权限由实际 API 返回决定，权限/积分会变。
