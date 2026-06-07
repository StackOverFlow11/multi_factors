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

- 状态: **部分处理(P0)**。
- P0 仅实现 `missing_close` 过滤:截面日 `close` 为 NaN 的标的不可交易(UNI-004)。
- 停牌 / ST / 涨跌停过滤接口已预留(`UniverseFilters`),但 P0 未实现(降级),P1 补齐。
- 退市 / 无数据标的(如 `000003.SZ`,2024 已退市)表现为不在 panel 中而被隐式剔除。PIT 历史成分已由 `PITIndexUniverse` 处理(见上节);停牌期间的显式停牌标记过滤仍是后续项。
- `universe.min_listing_days` 已在配置中(默认 60),但 P0 **未执行**(no-op,降级):新上市标的不会被剔除。该空操作在此显式披露(INV-007),P1 接入上市日期后强制执行。

## ann_date 财务对齐

- 状态: **不适用 / 待办(P0)**。
- P0 只用行情动量因子,不使用任何财务数据,因此本期不存在财务前视风险。
- 一旦引入财务因子,财务特征必须按披露日 `ann_date` 对齐,不能早于披露日出现(DATA-012,P1)。

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
