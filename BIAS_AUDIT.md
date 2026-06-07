# Bias Audit (Phase 0)

本文件记录 P0 框架对各类偏差/未来函数的处理状态与降级。每个小节标注当前状态(已处理 / 降级 / 待办)。

## 未来函数 / lookahead

- 状态: **已处理(P0)**。
- `momentum_20[t] = close[t] / close[t-window] - 1`,严格只用 t 及之前的收盘价(`groupby(symbol).shift(window)`)。
- 事件顺序固定:在 t 收盘计算因子,t 收盘后调仓,从 t+1 持有。回测用**下一持有期**的收益结算,绝不使用因子已经看见的当日收益。
- forward returns 只在 `analytics/` 计算,因子层永远拿不到未来收益(INV-001)。

## PIT 成分股

- 状态: **PIT 已实现(P1) / StaticUniverse 为离线降级**。
- `PITIndexUniverse`(`universe.type=index`)用 tushare `index_weight` 的历史快照做 as-of 成分:`members(date)` 取 ≤date 的最近快照,绝不用未来快照(UNI-009)。被剔除的票在其在册期内仍是成员 —— 无幸存者偏差、无成分前视。
- pipeline 构建 index universe 时会额外向回看 370 天成分快照,确保回测从两次成分调整中间开始时,起始日也能取到“开始日前最近快照”,而不是错误空仓。
- 实证:沪深300 2024 全年 24 个快照、328 个不同成分(每快照 300),28 进28 出;`000069.SZ` 在 06-03 在册、06-28 已剔,各按其时代正确归属。
- 数据坑:`index_weight` 单次约 6000 行上限,长窗口会**静默丢最早快照**;feed 已分 90 天窗口分页拉取规避。
- `StaticUniverse`(`universe.type=static`,demo/离线用)成分与日期无关(UNI-003),是**降级**:存在幸存者 / 成分前视偏差,仅供无网络的 demo 跑通,并在 `phase0_summary.md` 的 DOWNGRADES 小节显式记录。

## 可交易过滤

- 状态: **停牌 / ST / 涨跌停已实现(P1)**。
- `missing_close`(总是开):截面日 `close` 为 NaN 的标的不可交易(UNI-004)。
- 统一在 `universe.filters.apply_tradable_filters` 按 `UniverseFilters` 开关执行;flag 由 `data.clean.tradability.enrich_tradability` 从 tushare `suspend_d` / `namechange` / `stk_limit` 富化到 panel(StaticUniverse 与 PITIndexUniverse 共用)。demo 无 flag 数据时各过滤自动 no-op。
- **ST(UNI-006)**:`namechange` 名称区间含 'ST'/'*ST' 即标记,按 date 取生效名称(实证:`000005.SZ` 2024 全程 ST,正确剔除)。
- **涨跌停(UNI-007)**:close 触及当日 `up_limit`/`down_limit` 标记 `at_up_limit`/`at_down_limit`(实证:`000005.SZ` 2024-02-01 触跌停)。当前选股层对两个方向都剔除;**方向感知**(买入只看涨停、持有跌停不强卖)属执行层,后续细化。
- **停牌(UNI-005)**:`suspend_d` 标记停牌日。**实测发现**:tushare 全天停牌当日**无 bar** → 已被 `missing_close` 剔除,故显式 suspended flag 与之重叠;其价值在盘中停牌(`suspend_timing`)或会给停牌日 bar 的数据源,属防御性。
- 退市 / 无数据标的(如 `000003.SZ`)同样表现为不在 panel 而被剔除。PIT 历史成分见上节。
- `universe.min_listing_days` 已在配置中(默认 60),但仍 **未执行**(no-op,降级):新上市标的不会被剔除。显式披露(INV-007),后续接上市日期后强制。

## ann_date 财务对齐

- 状态: **已实现(P1)**。
- 财务因子(`roe` / `netprofit_yoy`)经 `data.clean.pit_financials.asof_financials` 按披露日 `ann_date` 做 backward as-of 对齐:每个 trade_date 只取 `ann_date <= trade_date` 的最近一期报告,**绝不按 `end_date`(报告期末)join**(DATA-012)。
- 实证:平安银行 2024 Q1(end_date 2024-03-31)披露日 ann_date 2024-04-20;as-of roe 在 04-19 仍是上一期年报值(10.2436),04-22 才切到 Q1(3.1176)——晚于报告期末约 3 周,证明无未来披露泄漏。
- 财务因子仅在 tushare 数据路径可用;demo 无披露日,配置财务因子 + demo 源会报可读错误,**不伪造财务**。

## 复权

- 状态: **前复权已实现(P1)**。
- panel 始终携带 `adj_factor` 列(DemoFeed 中恒为 1.0)。`data/clean/adjust.py` 的 `front_adjust` 用 `adj_factor` 做前复权(qfq),在 pipeline 读盘后、因子计算前于内存中应用(DATA-003)。
- 约定:按 symbol 锚定窗口内最新日 (`qfq = raw × adj_factor / adj_factor[latest]`)。锚定项在任何价格比值中约掉,故所有收益率 / 因子值对锚定与扩窗都不变 —— PanelStore 保持 raw(+adj_factor),复权在内存做,batch≡incremental 一致。
- 实证:平安银行 2024-06-14 除权,raw 当日 -5.74%(分红跳空),qfq +0.99%(真实涨跌);momentum_20 因此最多变动 6.77pp。demo(adj=1.0)下为恒等。

## 交易成本

- 状态: **已处理(P0)**。
- 成本 = L1 换手 × `fee_rate`;`turnover = sum(|target_w - current_w|)`,在 symbol 并集上对齐计算。
- 每个调仓期 `net_return = gross_return - cost`,成本拖累在`phase0_summary.md` 中汇总(BT-004)。slippage 参数已预留。
- **结算价缺失约定(P0 降级)**:若持仓标的在持有期末(end)的 `close` 为 NaN(停牌 / 缺数据),回测以 0.0(持平)记其该期收益,而非剔除或用最近可得价结算。该约定在此显式披露(INV-007);P1 接入真实停牌/退市处理后改进结算逻辑。
