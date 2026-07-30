# D5 C5 —— 四腿全量对账审计报告

> **状态**：**已填充**（2026-07-30）。11 因子 × panels/reports/anchors 全量对账已跑完，
> 结果在 §三/§四/§五。填充纪律（已执行）：每处差异必须落在 §二 预登记清单或 §二之三 的
> 本轮新增登记内；**未编目的差异 = 验收不通过**（设计 v3.2 §五腿 3：「允许差异，但每处
> 差异必须归因到具名原因；变好的差异优先怀疑泄漏」）。**类外差异为 0。**
> ⚠️ 本轮的每一条新增登记都不是"放宽"，而是**原登记表建立在一个不完整的证据基上**——
> 详见 §六.6/6.7 与编目 §七之七/§七之八。
> **执行环境**：11 因子 × {no_book, with_book} 统一 exec-only runner
> （`qt/factor_eval_runner.py`）+ 对账 harness（`qt/factor_eval_reconcile.py` 三模式）。

---

## 一、范围与方法

### 1.1 四腿定义（设计 v3.2 §五）

| 腿 | 内容 | 本步载体 |
|---|---|---|
| 腿 1 性质测试搬迁 | 旧 runner 测试的每条性质在新套件有映射，覆盖数非降（R16） | `docs/factors/d5_property_test_migration_map.md`（59/59 有映射，0 静默丢失） |
| 腿 2 手算锚 | 每因子 ≥5 行从 raw parquet 徒手复算，不 import 新引擎；warmup 区差异归 `warmup_left_extension`，jump 非 warmup 行必须 reconcile（证明 service 携带截断定义） | harness `--mode anchors` vs `artifacts/refactor_baseline/hand_anchors_d2.json` |
| 腿 3 语义对账 | 新 artifact vs 冻结 exec artifact **值级**比对（IC/ICIR/分位价差/verdict；逐字节已知不成立，见 §二 #1）；差异按白名单分类，聚合路径之外任何变化即失败 | harness `--mode reports`（JSON 叶子 + MD 行级配对，strict/no-book 与 bookclose 双闸） |
| 腿 4 面板腿 | service 面板（decision 视图 × exec_to_exec）vs D1 冻结面板**逐格**（1e-12；NaN 集合差异单独分类）；jump 走 `pr_c_cutoff_fix` 参照物 | harness `--mode panels`（具名类：`warmup_left_extension` / `float_reordering_tail` / `threshold_flip_tail` / `threshold_flip_contamination`） |

### 1.2 硬闸门与基线纪律

- **77/77 硬闸门**：三个模式入口都先跑 `verify_all()`，冻结 exec 基线字节不可达或被改
  是**硬错误不是 skip**（没有基线就没有对账；「对账通过但没有基线」是
  compare_postmerge 空对账形态）。
- **只读冻结基线**：对账只读 D5a 冻结的 77 份（`artifacts/refactor_baseline/exec_baseline/`，
  sha256 manifest 入 git），**绝不读 live 路径**——新 runner 写同名文件，跑一次就毁掉
  唯一基线。
- **cache-only**：每次 service 取数后断言 `stk_mins_live_calls == 0`，非零即 ABORT。
- **bounded `warmup_left_extension` 的 cell ceiling 是 backstop，不是主闸门（措辞已修正）**：
  ceiling =（w−1 个 warmup 交易日）×（冻结面板 symbol 数）是该判据下 warmup 格数的
  **结构上限**——只要 `_in_warmup` 的**日期边界判据本身**没坏，它在**数学上就不可能被
  超过**。所以它**既不是量级闸门也不是密度闸门**，而是"日期边界判据自身有没有坏"的
  **兜底**；**只有在另一个守卫已经失效时它才会响**。
  实测占用率 91.4%–91.5%（8198/8955、17289/18905、17272/18905）**不是"还有 8.5% 余量"
  的意思**：占用率 =（warmup 区内真有差异的票数）/（全部票数），结构上本就该接近 1，
  它既不是警报也不是头寸。
  ⚠️ **C4a 评审 LOW-1 的量级缺口仍然敞着**：bounded `warmup_left_extension` 类**至今
  没有任何量级上限**，本轮实测落在 warmup 类里的 max rel 高达 **1.996**（符号翻转）。
  区外仍由 1e-12 把门，区内不设限是既有裁定（两种加载几何在该区本来就合法地不同）——
  **但读者不许把上面那个 ceiling 当成填补了这个缺口**。

## 二、预登记漂移清单（对账白名单）

以下来源：编目 §七 / §七之二 / §七之三 / §七之四 / §七之五
（`docs/factors/d5_runner_difference_catalogue.md`）+ 交接 §3
（`tmp/design/HANDOFF_2026-07-26_refactor_execution.md`）。**不在本表内的差异一律
unclassified → FAIL。**

| # | 名称 | 边界 | 出处 |
|---|---|---|---|
| 1 | 契约 v1.0/v1.1 artifact 新增（`spec` 16→20 键 / `eval_config` +3 键 / 顶层 `eval_contract_version` + `corrections` / MD +4 行 provenance + 每条更正 +1 行） | 全为**新增**、零删除、零数值变化；逐字节对账已知不成立，只成立值级 + 结构性断言 | 编目 §七、§七之二；交接 §3 #1 |
| 2 | exec 侧 no_book 与 with_book 的 `eval_config` 首次合法不同（`book_view`: null vs run 模式） | 仅此一键；不得当作两格之间的回归 | 编目 §七之二 ⚠️；交接 §3 #2 |
| 3 | served-panel NaN 足迹（D4c）：行集变密、按行数算分母的覆盖率变，读值不受影响 | 额外行必须全 NaN；有限值的新行只许落在 warmup 边界内 | 交接 §3 #3 |
| 4 | `warmup_left_extension`（panels/anchors 通用，三方向） | bounded：冻结网格前 w−1 个交易日（w=`lookback_depth`），cell 数 ≤ (w−1)×|symbols|（本步新增 ceiling）；pooled：早区 [2021-07-01, 2021-10-31] + 按月计数非递增，豁免月 = 冻结面板**首个 finite 值所在月**（结构性锚定；全 NaN 冻结面板不豁免） | 编目 §七之三根因一、§七之四 #2、§七之五裁定 1；交接 §3 #4② |
| 5 | `float_reordering_tail` | rel ≤ 5e-12 或 abs ≤ 1e-12；cap：bars-only ≤101 cell，截面 ≤1,000 cell | 编目 §七之三根因二、§七之五裁定 3 |
| 6 | `threshold_flip_tail`（count 因子） | 幅度恰 ±1 count、rel ≤ 1e-2、≤25 cell；实测 600623.SH 2023-06-15..07-14 | 编目 §七之三根因二 |
| 7 | `threshold_flip_contamination`（仅截面因子） | 窗口 [2023-06-01, 2023-07-14]；直接 symbol 600623.SH ≤2e-04，其余 ≤1e-06，总 ≤20,000 cell；逐格校验，越界即 unclassified | 编目 §七之五裁定 2 |
| 8 | `warmup_aggregate_effect`（reports 腿） | 仅聚合叶子（`sections[*]` / `verdict.reasons[*]` / `verdict.axes.*`）且**数字承载**；纯标签翻转（PASS→FAIL 无数字）仍失败 | 编目 §七之三 reports #2 |
| 9 | `registered_sanity_stem_rename` | sanity 报告名 `eval_<name>_` → `factor_eval_<factor_id>_`，双侧以 `_exec_basis_sanity.md` 收尾 | 编目 §七之三 reports #4 |
| 10 | `registered_run_order_artifact` | `exec_price_artifact_reused` False→True（同会话 run 序）；反向不登记 | 编目 §七之三 reports #5 |
| 11 | jump 更正效应（`registered_correction_effect`） | 仅当新 JSON 带 `corrections` 结构化承载（契约 v1.1）；其值级验证在 panels 腿（`pr_c_cutoff_fix` 参照物）与 anchors 腿（非 warmup 行必须 reconcile），不在 reports 腿 | 编目 §七之三 reports #6、§七之二 ✅；交接 §3 #4③ |
| 12 | vpq `neutralization_coverage` add-Section | 整个 subtree 为新增；按 section **名字**匹配（`REGISTERED_EXTRA_SECTIONS`），绝不按裸下标 | 编目 §七之四 #1 |
| 13 | `book_view_effect`（with_book(decision) 格） | decision 书 vs 冻结 close 书的 Incremental 轴数值差：**全量报告、不设闸**；闸在 `_bookclose` 格（见 §四） | harness 设计（a)/(b) 分解；交接 §3 D5 C4 ⚠️ |
| 14 | `hand_anchors_d2.json` 报 `all_ok_frozen14: False` | 70 行 5 处失配**全部且仅有 jump**（手算已截断 vs 冻结 `panels_d2` 未截断）——正确信号非回归 | 交接 §3 #5 |

## 二之三、本轮新增的登记项（§二 清单的增补）

以下六条是 C5 修复轮新增的具名登记。**每一条都配有正反向测试（越界必 FAIL）与 mutation
证据**，出处见编目 §七之七 / §七之八。**#17 / #19 / #20 带反棘轮约束：不得再次放宽——
再有新成员出现即说明该类在描述残差而非机制，停下上报。**

| # | 名称 | 边界 | 出处 |
|---|---|---|---|
| 15 | `threshold_flip_contamination` bars-only 臂 | 同窗同 symbol；**相对**界按因子（kurtosis ≤5e-3 / relative_vwap ≤1e-5）；≤25 cell/因子 | 编目 §七之七 F2 |
| 16 | `warmup_left_extension` 新方向 `frozen_finite_new_nan` | 仅 valid-day pooled 因子 + 早区窗内 | 编目 §七之七 F3① |
| 17 | `warmup_sparse_valid_day_tail` | 仅 pooled；[2021-11-01, 2021-11-15]；**9 票白名单即成员**（**枚举类 + 必要条件守卫，不是机制类**：密度守卫在 **p25**、18/18 通过，但该谓词单独允许 **1,250** 对 ≈ 本类 **69 倍** ⇒ 约束成员集但不决定它）；≤20 cell/因子；**反棘轮 + 禁止照密度推导新成员** | 编目 §七之七 F3② |
| 18 | 绝对 float-dust 前移到区域分支之前 | `abs ≤ 1e-12` 按机制分类，不按位置 | 编目 §七之七 F4 |
| 19 | 诊断 sink 披露节（三个因子） | `old=None` 纯新增 + verdict 标签不变 + **无索引位移**，逐 grid 复验 **12/12**（4 因子 × 3 grid）；**反棘轮** | 编目 §七之八 A |
| 20 | `spec.description` 的 D2 provenance 改写 | **逐对精确枚举**（3 因子）；其它一律 FAIL；不 bump 版本、不触发更正承载 | 编目 §七之八 B |

## 二之二、C5 首轮全量跑的原始结果（pre-fix）—— 誊自日志，**耐久化**

> **为什么誊在这里**：这些数字原本只存在于 `artifacts/logs/`（gitignored、只在本机），
> 而 C5 的重跑会把同名日志**全部覆盖**——`qt.pipeline._make_logger` 以 truncate 模式打开。
> 本项目已经因为 `/tmp` 被清理丢过一次全量跑的总日志。**报告是耐久载体，日志不是。**
> 下表是 F1/F2/F3/F4/F5 全部修复**之前**的状态，B 阶段的结果表（§三）与它对照阅读。

**运行环境**：11 因子 × {panels, reports, anchors}，2026-07-28 01:50→05:52 一轮 sweep，
cache-only（每行 `live_calls=0`）。**⚠️ 该轮跑在 PR #109 之前的判据上**（见 §六.6）。

### panels 腿（`unclassified` 与 `ok` 为判定值，其余为分类计数）

| 因子 | frozen / new | warmup | float_tail | flip | contam | footprint | **unclassified** | max_rel | **ok** |
|---|---|---|---|---|---|---|---|---|---|
| minute_ideal_amp_10 | 1159263 / 1205160 | 8198 | 0 | 0 | 0 | 45897 | **0** | 1.996e+00 | **True** |
| jump_amount_corr_20 | 1159263 / 1205160 | 17289 | 101 | 0 | 0 | 45897 | **0** | 1.787e+00 | **True** |
| volume_peak_count_20 | 1149313 / 1205160 | 34441 | 0 | 20 | 0 | 46788 | **0** | 8.453e-01 | **True** |
| valley_price_quantile_20 | 1146878 / 1205160 | 24568 | 883 | 0 | 19177 | 58282 | **0** | 1.999e+00 | **True** |
| amp_marginal_anomaly_vol_20 | 1159263 / — | — | — | — | — | — | **729029** | — | **False** (F1) |
| peak_interval_kurtosis_20 | 1149313 / 1205160 | 35152 | 0 | 0 | 0 | 46790 | **20** | 1.233e+00 | **False** (F2) |
| valley_relative_vwap_20 | 1146878 / 1205160 | 35205 | 0 | 0 | 0 | 49235 | **20** | 8.041e-03 | **False** (F2) |
| valley_ridge_vwap_ratio_20 | 591524 / 1205160 | 28161 | 0 | 0 | 0 | 608309 | **20** | 1.293e-02 | **False** (F3) |
| ridge_minute_return_20 | 588536 / 1205160 | 27811 | 0 | 0 | 0 | 611318 | **20** | 1.999e+00 | **False** (F3) |
| intraday_amp_cut_10 | 1154288 / 1158842 | 8221 | 758 | 0 | 17 | 0 | **0** | 1.976e+00 | **False** (F4，按月计数非单调) |
| peak_ridge_amount_ratio_20 | 579030 / 1205160 | 27896 | 0 | 0 | 0 | 620916 | **30** | 7.360e-01 | **False** (F3 第三个因子) |

**出处（逐条）**：8 行誊自幸存日志原文（快照 `tmp/context/cc_c5_audit/c5_run_logs_preserved/`，
mtime 保留）；`peak_interval_kurtosis_20` 与 `intraday_amp_cut_10` 两行的日志**已被本轮
判据验证重跑覆盖**，其值出自 ① 覆盖前的逐字抓取与 ② 用同一批 served 面板在 **main 判据**
下的离线独立重算——两者一致；`amp_marginal_anomaly_vol_20` 一行的日志已被 F1 修复重跑
覆盖，其值出自编目 §七之六（该行本就是 F1 的证据，非本轮新测）。

**交叉印证**：离线独立重算（不读日志，直接对 dump 的 served 面板跑 main 判据）在 **9 个
非 amp_marginal 因子上逐格复现上表**——这使离线重算可以独立当证据基用，而不必信任日志。

### anchors 腿

| 因子 | rows | ok | warmup | failed | 判定 |
|---|---|---|---|---|---|
| minute_ideal_amp_10 / jump / volume_peak / ridge / valley_ridge / peak_ridge | 5 | 3 | 2 | 0 | **True** |
| valley_price_quantile_20 / valley_relative_vwap_20 / intraday_amp_cut_10 | 5 | 4 | 1 | 0 | **True** |
| peak_interval_kurtosis_20 | 5 | 2 | 3 | 0 | **True** |
| amp_marginal_anomaly_vol_20 | 5 | — | — | **3** | **False** (F1；出处编目 §七之六) |

### reports 腿

**11/11 全部 rc=1，且每一个都是 F5**——`{stem}_exec_with_book.json` 被 close 模式的
事后改名销毁，`run_reports_mode` 抛 `FileNotFoundError`。**这不是判定结果**：该腿在 C5
首轮**没有对任何因子产生过一个判定**（详见 §六.7）。

## 三、每因子结果表

**运行环境**：11 因子 × {no_book, with_book} × {decision, close} eval（22 次 rc 全 0）+
三模式对账，cache-only（`stk_mins_live_calls` 非零命中 **0**）。panels 与 anchors 出自 phase B
全量跑；reports 出自其后的 reports-only 重跑（改动经 AST 逐函数比对**只触及 reports 腿**，
panels/anchors 的入口函数逐字节未变——见 §六.9）。

### 三之1 panels 腿（全部 `unclassified=0`）

| 因子 | warmup | sparse_tail | float_tail (cap) | flip | contam | footprint | unclassified |
|---|---|---|---|---|---|---|---|
| jump_amount_corr_20 | 17289 | 0 | 101 (101) | 0 | 0 | 45897 | **0** |
| minute_ideal_amp_10 | 8198 | 0 | 0 (101) | 0 | 0 | 45897 | **0** |
| amp_marginal_anomaly_vol_20 | 17272 | 0 | 0 (101) | 0 | 0 | 45897 | **0** |
| volume_peak_count_20 | 34441 | 0 | 0 (101) | 20 | 0 | 46788 | **0** |
| intraday_amp_cut_10 | 8198 | 0 | 798 (1000) | 0 | 0 | 0 | **0** |
| peak_interval_kurtosis_20 | 35152 | 0 | 0 (101) | 0 | 20 | 46790 | **0** |
| valley_relative_vwap_20 | 35205 | 0 | 0 (101) | 0 | 20 | 49235 | **0** |
| valley_ridge_vwap_ratio_20 | 28168 | 13 | 0 (101) | 0 | 0 | 608309 | **0** |
| ridge_minute_return_20 | 27818 | 13 | 0 (101) | 0 | 0 | 611318 | **0** |
| valley_price_quantile_20 | 24531 | 0 | 925 (1000) | 0 | 19172 | 58282 | **0** |
| peak_ridge_amount_ratio_20 | 27907 | 19 | 0 (101) | 0 | 0 | 620916 | **0** |

### 三之2 anchors 腿（全部 `failed=0`）

| 因子 | rows | ok | warmup | failed |
|---|---|---|---|---|
| jump_amount_corr_20 | 5 | 3 | 2 | **0** |
| minute_ideal_amp_10 | 5 | 3 | 2 | **0** |
| amp_marginal_anomaly_vol_20 | 5 | 4 | 1 | **0** |
| volume_peak_count_20 | 5 | 3 | 2 | **0** |
| intraday_amp_cut_10 | 5 | 4 | 1 | **0** |
| peak_interval_kurtosis_20 | 5 | 2 | 3 | **0** |
| valley_relative_vwap_20 | 5 | 4 | 1 | **0** |
| valley_ridge_vwap_ratio_20 | 5 | 3 | 2 | **0** |
| ridge_minute_return_20 | 5 | 3 | 2 | **0** |
| valley_price_quantile_20 | 5 | 4 | 1 | **0** |
| peak_ridge_amount_ratio_20 | 5 | 3 | 2 | **0** |

### 三之3 reports 腿（三格；全部 rc=0、零类外叶子）

| 因子 | rc | 登记类计数（no_book / bookclose / decision） |
|---|---|---|
| jump_amount_corr_20 | **0** | 108 diffs / 124 diffs / 124 diffs |
| minute_ideal_amp_10 | **0** | 121 diffs / 122 diffs / 137 diffs |
| amp_marginal_anomaly_vol_20 | **0** | 121 diffs / 122 diffs / 137 diffs |
| volume_peak_count_20 | **0** | 122 diffs / 137 diffs / 138 diffs |
| intraday_amp_cut_10 | **0** | 126 diffs / 127 diffs / 142 diffs |
| peak_interval_kurtosis_20 | **0** | 127 diffs / 142 diffs / 143 diffs |
| valley_relative_vwap_20 | **0** | 128 diffs / 144 diffs / 144 diffs |
| valley_ridge_vwap_ratio_20 | **0** | 163 diffs / 179 diffs / 179 diffs |
| ridge_minute_return_20 | **0** | 158 diffs / 172 diffs / 174 diffs |
| valley_price_quantile_20 | **0** | 105 diffs / 121 diffs / 121 diffs |
| peak_ridge_amount_ratio_20 | **0** | 163 diffs / 179 diffs / 179 diffs |

**总计：33 格（11 因子 × 三腿）全部 rc=0**；panels 的 `unclassified` 与 anchors 的 `failed`
全为 0，reports 三格零类外叶子。

### 三之四、贴边的边界（登记为脆弱点）

反棘轮裁定使这些"紧"是**有意的牙**——但读者得知道它们有多紧：

| 边界 | 上限 | 实测 | 余量 |
|---|---|---|---|
| `jump_amount_corr_20` float_tail cap | 101 | **101** | **0 格** |
| `warmup_sparse_valid_day_tail` 窗口右端 | 2021-11-15 | 该日**恰有 1 格** | **0 天** |
| `warmup_sparse_valid_day_tail` cell cap | 20 | **19**（peak_ridge） | **1 格** |
| **sparse tail 密度守卫（p25）** | **门槛 36 天** | **35 天**（`300857.SZ` × peak_ridge） | **1 天** |
| bars-only flip contamination cap | 25 | 20 | 5 格 |
| kurtosis 相对界 | 5e-3 | 2.904e-3 | 1.72× |
| relative_vwap 相对界 | 1e-5 | 1.530e-6 | 6.54× |

**前三行是零/近零余量**：
- **jump 的 101/101**：该 cap 当初按这个因子的观测值标定，**再多一格就 FAIL**。本轮判据改动没有
  往里加格（它那 101 格全在 warmup 区外，F4 前移一格未动）；但下次任何触及 rolling 求和顺序的
  改动都可能顶破它，而那时失败信息会是"cap 超了"，**不是"哪里错了"**。
- **sparse-tail 窗口右端 0 天余量**：窗口收在 2021-11-15，**恰恰因为该日有且只有 1 格**
  （`688183.SH`）。窗口不是留了余地画的，是**贴着最后一格画的**。
- **sparse-tail cap 只剩 1 格**：19/20。
- **sparse-tail 的密度守卫也只剩 1 天**：门槛是各因子的 p25 = 36 个发行日，而最贴边的成员
  `300857.SZ`（在 `peak_ridge_amount_ratio_20` 上）发行 **35** 天。⚠️ **这道门红了就是
  STOP-AND-REPORT，不许把门槛放回中位**——那正是反棘轮明令禁止的动作，**而它红的时候恰恰
  最诱人**（中位门会放行，于是"改一个数就绿了"）。这道门是本轮返工**自己新加的**，性质与
  上面三条零/近零余量完全相同，所以一并登记在此。

这三处**都不许因为"贴边"而放宽**——反棘轮的裁定正是"顶破了就停下上报"，贴边意味着**下一次
触发会很快到来且必须被当真**。

## 四、with_book 差异分解（逐指标）

分解口径：**A** = `no_book`（引擎效应，先对干净的）· **B** = `_bookclose`（legacy-faithful
close 书 + 新引擎，与 A 同口径可比）· **C** = decision 书 · **C−B = 书视图修正**。
Δ 一律相对**冻结基线的同名格**。

> ⚠️ **参照物只有冻结 exec 基线**：C5 首轮 reports 腿对 11 个因子一个判定都没产生过
> （全是 F5），所以本表**不是**与上一轮 C5 的对比。见 §六.7。

| 因子 | 指标 | 冻结 no_book | **A** 新 no_book (Δ) | 冻结 with_book | **B** bookclose (Δ) | **C** decision (Δ) | **C−B**（书视图修正） |
|---|---|---|---|---|---|---|---|
| jump_amount_corr_20 | IC mean | -0.030840 | -0.030387 (+0.000453) | -0.030840 | -0.030387 (+0.000453) | -0.030387 (+0.000453) | **+0.000000** |
| jump_amount_corr_20 | ICIR | -0.425348 | -0.423262 (+0.002086) | -0.425348 | -0.423262 (+0.002086) | -0.423262 (+0.002086) | **+0.000000** |
| jump_amount_corr_20 | NW-t | -15.191082 | -15.123855 (+0.067227) | -15.191082 | -15.123855 (+0.067227) | -15.123855 (+0.067227) | **+0.000000** |
| jump_amount_corr_20 | N_eff | 1162.206183 | 1164.076757 (+1.870574) | 1162.206183 | 1164.076757 (+1.870574) | 1164.076757 (+1.870574) | **+0.000000** |
| jump_amount_corr_20 | incr. ICIR | n/a | n/a (—) | -0.300601 | -0.299010 (+0.001591) | -0.295001 (+0.005600) | **+0.004009** |
| jump_amount_corr_20 | **verdict** | Watch | Watch (same) | Watch | Watch (same) | Watch (same) | 同 |
| jump_amount_corr_20 | **三轴标签** | PASS/NOT_/NOT_ | PASS/NOT_/NOT_ (same) | PASS/PASS/NOT_ | PASS/PASS/NOT_ (same) | PASS/PASS/NOT_ (same) | 同 |
| minute_ideal_amp_10 | IC mean | -0.029644 | -0.029508 (+0.000136) | -0.029644 | -0.029508 (+0.000136) | -0.029508 (+0.000136) | **+0.000000** |
| minute_ideal_amp_10 | ICIR | -0.473361 | -0.471797 (+0.001564) | -0.473361 | -0.471797 (+0.001564) | -0.471797 (+0.001564) | **+0.000000** |
| minute_ideal_amp_10 | NW-t | -16.476304 | -16.393063 (+0.083241) | -16.476304 | -16.393063 (+0.083241) | -16.393063 (+0.083241) | **+0.000000** |
| minute_ideal_amp_10 | N_eff | 1205.000000 | 1209.000000 (+4.000000) | 1205.000000 | 1209.000000 (+4.000000) | 1209.000000 (+4.000000) | **+0.000000** |
| minute_ideal_amp_10 | incr. ICIR | n/a | n/a (—) | -0.260375 | -0.260375 (+0.000000) | -0.257123 (+0.003252) | **+0.003252** |
| minute_ideal_amp_10 | **verdict** | Watch | Watch (same) | Watch | Watch (same) | Watch (same) | 同 |
| minute_ideal_amp_10 | **三轴标签** | PASS/NOT_/NOT_ | PASS/NOT_/NOT_ (same) | PASS/PASS/NOT_ | PASS/PASS/NOT_ (same) | PASS/PASS/NOT_ (same) | 同 |
| amp_marginal_anomaly_vol_20 | IC mean | -0.042601 | -0.042341 (+0.000260) | -0.042601 | -0.042341 (+0.000260) | -0.042341 (+0.000260) | **+0.000000** |
| amp_marginal_anomaly_vol_20 | ICIR | -0.446484 | -0.443614 (+0.002870) | -0.446484 | -0.443614 (+0.002870) | -0.443614 (+0.002870) | **+0.000000** |
| amp_marginal_anomaly_vol_20 | NW-t | -16.154643 | -16.132296 (+0.022347) | -16.154643 | -16.132296 (+0.022347) | -16.132296 (+0.022347) | **+0.000000** |
| amp_marginal_anomaly_vol_20 | N_eff | 1116.017042 | 1115.866033 (-0.151009) | 1116.017042 | 1115.866033 (-0.151009) | 1115.866033 (-0.151009) | **+0.000000** |
| amp_marginal_anomaly_vol_20 | incr. ICIR | n/a | n/a (—) | -0.311985 | -0.311985 (+0.000000) | -0.304787 (+0.007198) | **+0.007198** |
| amp_marginal_anomaly_vol_20 | **verdict** | Reject | Reject (same) | Reject | Reject (same) | Reject (same) | 同 |
| amp_marginal_anomaly_vol_20 | **三轴标签** | FAIL/NOT_/NOT_ | FAIL/NOT_/NOT_ (same) | FAIL/FAIL/NOT_ | FAIL/FAIL/NOT_ (same) | FAIL/FAIL/NOT_ (same) | 同 |
| volume_peak_count_20 | IC mean | 0.018587 | 0.018475 (-0.000112) | 0.018587 | 0.018475 (-0.000112) | 0.018475 (-0.000112) | **+0.000000** |
| volume_peak_count_20 | ICIR | 0.224592 | 0.223694 (-0.000898) | 0.224592 | 0.223694 (-0.000898) | 0.223694 (-0.000898) | **+0.000000** |
| volume_peak_count_20 | NW-t | 7.758584 | 7.765443 (+0.006859) | 7.758584 | 7.765443 (+0.006859) | 7.765443 (+0.006859) | **+0.000000** |
| volume_peak_count_20 | N_eff | 1055.757320 | 1068.478323 (+12.721003) | 1055.757320 | 1068.478323 (+12.721003) | 1068.478323 (+12.721003) | **+0.000000** |
| volume_peak_count_20 | incr. ICIR | n/a | n/a (—) | 0.120357 | 0.121800 (+0.001443) | 0.112905 (-0.007452) | **-0.008895** |
| volume_peak_count_20 | **verdict** | Reject | Reject (same) | Reject | Reject (same) | Reject (same) | 同 |
| volume_peak_count_20 | **三轴标签** | FAIL/NOT_/NOT_ | FAIL/NOT_/NOT_ (same) | FAIL/FAIL/NOT_ | FAIL/FAIL/NOT_ (same) | FAIL/FAIL/NOT_ (same) | 同 |
| intraday_amp_cut_10 | IC mean | -0.035941 | -0.035806 (+0.000135) | -0.035941 | -0.035806 (+0.000135) | -0.035806 (+0.000135) | **+0.000000** |
| intraday_amp_cut_10 | ICIR | -0.454801 | -0.453101 (+0.001700) | -0.454801 | -0.453101 (+0.001700) | -0.453101 (+0.001700) | **+0.000000** |
| intraday_amp_cut_10 | NW-t | -15.653555 | -15.615023 (+0.038532) | -15.653555 | -15.615023 (+0.038532) | -15.615023 (+0.038532) | **+0.000000** |
| intraday_amp_cut_10 | N_eff | 959.099290 | 955.010692 (-4.088598) | 959.099290 | 955.010692 (-4.088598) | 955.010692 (-4.088598) | **+0.000000** |
| intraday_amp_cut_10 | incr. ICIR | n/a | n/a (—) | -0.258500 | -0.258500 (+0.000000) | -0.262065 (-0.003565) | **-0.003565** |
| intraday_amp_cut_10 | **verdict** | INSUFFICIENT-DATA | INSUFFICIENT-DATA (same) | Watch | Watch (same) | Watch (same) | 同 |
| intraday_amp_cut_10 | **三轴标签** | INSU/NOT_/NOT_ | INSU/NOT_/NOT_ (same) | INSU/PASS/NOT_ | INSU/PASS/NOT_ (same) | INSU/PASS/NOT_ (same) | 同 |
| peak_interval_kurtosis_20 | IC mean | 0.000030 | -0.000078 (-0.000108) | 0.000030 | -0.000078 (-0.000108) | -0.000078 (-0.000108) | **+0.000000** |
| peak_interval_kurtosis_20 | ICIR | 0.000815 | -0.002072 (-0.002887) | 0.000815 | -0.002072 (-0.002887) | -0.002072 (-0.002887) | **+0.000000** |
| peak_interval_kurtosis_20 | NW-t | 0.033214 | -0.084467 (-0.117681) | 0.033214 | -0.084467 (-0.117681) | -0.084467 (-0.117681) | **+0.000000** |
| peak_interval_kurtosis_20 | N_eff | 1190.000000 | 1209.000000 (+19.000000) | 1190.000000 | 1209.000000 (+19.000000) | 1209.000000 (+19.000000) | **+0.000000** |
| peak_interval_kurtosis_20 | incr. ICIR | n/a | n/a (—) | 0.045255 | 0.042346 (-0.002909) | 0.035443 (-0.009812) | **-0.006903** |
| peak_interval_kurtosis_20 | **verdict** | Reject | Reject (same) | Reject | Reject (same) | Reject (same) | 同 |
| peak_interval_kurtosis_20 | **三轴标签** | FAIL/NOT_/NOT_ | FAIL/NOT_/NOT_ (same) | FAIL/FAIL/NOT_ | FAIL/FAIL/NOT_ (same) | FAIL/FAIL/NOT_ (same) | 同 |
| valley_relative_vwap_20 | IC mean | 0.033656 | 0.033325 (-0.000331) | 0.033656 | 0.033325 (-0.000331) | 0.033325 (-0.000331) | **+0.000000** |
| valley_relative_vwap_20 | ICIR | 0.526483 | 0.519995 (-0.006488) | 0.526483 | 0.519995 (-0.006488) | 0.519995 (-0.006488) | **+0.000000** |
| valley_relative_vwap_20 | NW-t | 17.580989 | 17.495268 (-0.085721) | 17.580989 | 17.495268 (-0.085721) | 17.495268 (-0.085721) | **+0.000000** |
| valley_relative_vwap_20 | N_eff | 822.812580 | 829.057881 (+6.245301) | 822.812580 | 829.057881 (+6.245301) | 829.057881 (+6.245301) | **+0.000000** |
| valley_relative_vwap_20 | incr. ICIR | n/a | n/a (—) | 0.348870 | 0.348049 (-0.000821) | 0.350587 (+0.001717) | **+0.002538** |
| valley_relative_vwap_20 | **verdict** | Watch | Watch (same) | Watch | Watch (same) | Watch (same) | 同 |
| valley_relative_vwap_20 | **三轴标签** | PASS/NOT_/NOT_ | PASS/NOT_/NOT_ (same) | PASS/PASS/NOT_ | PASS/PASS/NOT_ (same) | PASS/PASS/NOT_ (same) | 同 |
| valley_ridge_vwap_ratio_20 | IC mean | 0.034667 | 0.034648 (-0.000019) | 0.034667 | 0.034648 (-0.000019) | 0.034648 (-0.000019) | **+0.000000** |
| valley_ridge_vwap_ratio_20 | ICIR | 0.439482 | 0.440794 (+0.001312) | 0.439482 | 0.440794 (+0.001312) | 0.440794 (+0.001312) | **+0.000000** |
| valley_ridge_vwap_ratio_20 | NW-t | 14.350975 | 14.629276 (+0.278301) | 14.350975 | 14.629276 (+0.278301) | 14.629276 (+0.278301) | **+0.000000** |
| valley_ridge_vwap_ratio_20 | N_eff | 815.607462 | 842.525194 (+26.917732) | 815.607462 | 842.525194 (+26.917732) | 842.525194 (+26.917732) | **+0.000000** |
| valley_ridge_vwap_ratio_20 | incr. ICIR | n/a | n/a (—) | 0.295070 | 0.295828 (+0.000758) | 0.296398 (+0.001328) | **+0.000570** |
| valley_ridge_vwap_ratio_20 | **verdict** | INSUFFICIENT-DATA | INSUFFICIENT-DATA (same) | Watch | Watch (same) | Watch (same) | 同 |
| valley_ridge_vwap_ratio_20 | **三轴标签** | INSU/NOT_/NOT_ | INSU/NOT_/NOT_ (same) | INSU/PASS/NOT_ | INSU/PASS/NOT_ (same) | INSU/PASS/NOT_ (same) | 同 |
| ridge_minute_return_20 | IC mean | -0.032373 | -0.031852 (+0.000521) | -0.032373 | -0.031852 (+0.000521) | -0.031852 (+0.000521) | **+0.000000** |
| ridge_minute_return_20 | ICIR | -0.397967 | -0.391910 (+0.006057) | -0.397967 | -0.391910 (+0.006057) | -0.391910 (+0.006057) | **+0.000000** |
| ridge_minute_return_20 | NW-t | -13.690742 | -13.552495 (+0.138247) | -13.690742 | -13.552495 (+0.138247) | -13.552495 (+0.138247) | **+0.000000** |
| ridge_minute_return_20 | N_eff | 1058.906714 | 1059.087960 (+0.181246) | 1058.906714 | 1059.087960 (+0.181246) | 1059.087960 (+0.181246) | **+0.000000** |
| ridge_minute_return_20 | incr. ICIR | n/a | n/a (—) | -0.150500 | -0.150937 (-0.000437) | -0.155878 (-0.005378) | **-0.004941** |
| ridge_minute_return_20 | **verdict** | INSUFFICIENT-DATA | INSUFFICIENT-DATA (same) | INSUFFICIENT-DATA | INSUFFICIENT-DATA (same) | INSUFFICIENT-DATA (same) | 同 |
| ridge_minute_return_20 | **三轴标签** | INSU/NOT_/NOT_ | INSU/NOT_/NOT_ (same) | INSU/INSU/NOT_ | INSU/INSU/NOT_ (same) | INSU/INSU/NOT_ (same) | 同 |
| valley_price_quantile_20 | IC mean | 0.026937 | 0.027028 (+0.000091) | 0.026937 | 0.027028 (+0.000091) | 0.027028 (+0.000091) | **+0.000000** |
| valley_price_quantile_20 | ICIR | 0.424804 | 0.426259 (+0.001455) | 0.424804 | 0.426259 (+0.001455) | 0.426259 (+0.001455) | **+0.000000** |
| valley_price_quantile_20 | NW-t | 13.358750 | 13.417417 (+0.058667) | 13.358750 | 13.417417 (+0.058667) | 13.417417 (+0.058667) | **+0.000000** |
| valley_price_quantile_20 | N_eff | 761.173410 | 764.155900 (+2.982490) | 761.173410 | 764.155900 (+2.982490) | 764.155900 (+2.982490) | **+0.000000** |
| valley_price_quantile_20 | incr. ICIR | n/a | n/a (—) | 0.309642 | 0.311285 (+0.001643) | 0.305919 (-0.003723) | **-0.005366** |
| valley_price_quantile_20 | **verdict** | Watch | Watch (same) | Watch | Watch (same) | Watch (same) | 同 |
| valley_price_quantile_20 | **三轴标签** | PASS/NOT_/NOT_ | PASS/NOT_/NOT_ (same) | PASS/PASS/NOT_ | PASS/PASS/NOT_ (same) | PASS/PASS/NOT_ (same) | 同 |
| peak_ridge_amount_ratio_20 | IC mean | 0.041017 | 0.041098 (+0.000081) | 0.041017 | 0.041098 (+0.000081) | 0.041098 (+0.000081) | **+0.000000** |
| peak_ridge_amount_ratio_20 | ICIR | 0.510337 | 0.513597 (+0.003260) | 0.510337 | 0.513597 (+0.003260) | 0.513597 (+0.003260) | **+0.000000** |
| peak_ridge_amount_ratio_20 | NW-t | 16.523066 | 16.884838 (+0.361772) | 16.523066 | 16.884838 (+0.361772) | 16.884838 (+0.361772) | **+0.000000** |
| peak_ridge_amount_ratio_20 | N_eff | 933.277963 | 958.265755 (+24.987792) | 933.277963 | 958.265755 (+24.987792) | 958.265755 (+24.987792) | **+0.000000** |
| peak_ridge_amount_ratio_20 | incr. ICIR | n/a | n/a (—) | 0.308282 | 0.315855 (+0.007573) | 0.321288 (+0.013006) | **+0.005433** |
| peak_ridge_amount_ratio_20 | **verdict** | Watch | Watch (same) | Watch | Watch (same) | Watch (same) | 同 |
| peak_ridge_amount_ratio_20 | **三轴标签** | PASS/NOT_/NOT_ | PASS/NOT_/NOT_ (same) | PASS/PASS/NOT_ | PASS/PASS/NOT_ (same) | PASS/PASS/NOT_ (same) | 同 |

**分解读数**：

1. **A 与 B 逐指标同号同量级**：IC / ICIR / NW-t / N_eff 四个指标在 A 与 B 两格的 Δ
   **完全相同**（书不进入这些指标的计算，44/44 取值逐格相同），差异全部来自已登记的
   `warmup_aggregate_effect`——即 panels 腿那些 warmup 格的下游聚合效应。
2. **书视图的爆炸半径被限制住了**：artifact 层 C−B 非零的叶子共 **174 个**，且**全部**落在
   `sections[3]`（purity / incremental）+ `book_view` + verdict reasons ——
   **174/174 通过**。这是这张表真正说得出的事：**换书视图动到的东西，恰好只在它应该动到的
   地方**。
3. **verdict 与三轴标签：11/11 因子、每一格全部未变**（A、B、C 三格各自与冻结同名格比较）。
   174 个叶子的书视图修正**没有翻动任何一个 Incremental 轴标签**。

> ### ⚠️ 一条曾被写成检验、实为**恒真**的推论（留档，不悄悄换掉）
>
> 本节初版在读数 2 里写过：「**若引擎回归藏在别的指标里，上表会显示 C−B 在那些行非零**」。
> **这句话是假的**，评审用 mutation 直接演示：把 `ic_mean` 在 **A / B / C 三本书里同时 ×1.5**，
> **C−B 仍精确为 0.0**。机制很直白——A/B/C 在 IC/ICIR/NW-t/N_eff 上**取值完全相同**（书对这些
> 指标**构造性免疫**），所以**引擎回归会让 A、B、C 一起动，然后在 C−B 里对消**。C−B 检不出
> 引擎回归，一格都检不出。
>
> 同一句里的「C−B 非零恰好 11 格 / 57 行」也不该当成事实陈述：**11 与 57 是这张表自己选了哪些
> 行的结果**，artifact 层的真实数字是上面的 **174** 个叶子。
>
> **真正检出引擎回归的是别的东西**：`no_book` 与 `bookclose` 两格在 **strict 模式**下对冻结基线
> 的逐叶子闸门（未登记的变化即 FAIL）。**它存在、而且通过了**——本 PR 的引擎结论站在它上面，
> **不站在 C−B 上**。
>
> **审计结论不受影响，错的是归因**：数字一个没变，变的是"哪个装置在把关"。留档而不是改写，
> 与 `compare_postmerge.py` 那次空对账同一处理——**被证否的推理留在案上**。

## 五、判定

| 因子 | 腿 1 性质映射 | 腿 2 anchors | 腿 3 reports | 腿 4 panels | 判定 |
|---|---|---|---|---|---|
| jump_amount_corr_20 | 59/59 | PASS (failed=0) | PASS (rc=0) | PASS (unclassified=0) | **PASS** |
| minute_ideal_amp_10 | 59/59 | PASS | PASS | PASS | **PASS** |
| amp_marginal_anomaly_vol_20 | 59/59 | PASS | PASS | PASS | **PASS**（F1 修复后） |
| volume_peak_count_20 | 59/59 | PASS | PASS | PASS | **PASS** |
| intraday_amp_cut_10 | 59/59 | PASS | PASS | PASS | **PASS**（F4 判据修正后） |
| peak_interval_kurtosis_20 | 59/59 | PASS | PASS | PASS | **PASS**（F2 + 成因 B 登记后） |
| valley_relative_vwap_20 | 59/59 | PASS | PASS | PASS | **PASS**（同上） |
| valley_ridge_vwap_ratio_20 | 59/59 | PASS | PASS | PASS | **PASS**（F3 + 成因 A/B 登记后） |
| ridge_minute_return_20 | 59/59 | PASS | PASS | PASS | **PASS**（F3 + 成因 A） |
| valley_price_quantile_20 | 59/59 | PASS | PASS | PASS | **PASS** |
| peak_ridge_amount_ratio_20 | 59/59 | PASS | PASS | PASS | **PASS**（F3 重新推导 + 成因 A） |

（腿 1 的 59/59 来自 `docs/factors/d5_property_test_migration_map.md`，本步未改动它。）

**整体结论：11 因子四腿全过，C5 通过** ⇒ 满足 C6（删除 11 个旧 runner）的前置条件之一。
每一处差异都落在 §二 的预登记清单或 §二之三 的本轮新增登记内，**类外差异为 0**。

⚠️ **本判定覆盖的是"新引擎与冻结基线的差异是否全部可归因"，不是因子的研究结论。**
十一个因子的 verdict 与三轴标签本轮**一个都没变**，研究侧结论（全部封顶 Watch、无一 Adopt）
不受本步影响。

## 六、已知局限（登记，不阻塞）

1. **C4a 评审 LOW-1（已处置）**：bounded `warmup_left_extension` 类原无 cell 数上限，
   warmup 区内任意幅度错误都被吞（暴露面 ~1.6% 网格）。本准备步已加结构性 ceiling
   （§1.2）；**类内幅度仍不设限**是既有裁定——C5 用途（几何对账）下可接受，因为两种
   加载几何在该区本来就合法地不同；区外由 1e-12 把门。
2. **C4a 评审 LOW-2**：reports 腿 `warmup_aggregate_effect` 的「数字承载」是 digit OR
   规则——散文理由行的语义可变（改了词但含数字仍放行）；verdict 标签等无数字翻转
   仍严格失败。判定阅读时应以 panels 腿为值级闸门（reports 腿是聚合层）。
3. **NIT 登记**：C4a NIT-1/NIT-2（装饰性，不影响判定）。
4. **既有披露缺陷（非本次回归，见交接 §4）**：六个 JSON 字段被 `MAX_VALUE_CHARS=200`
   静默截断（census 守卫登记为 frozen set）；9/11 dashboard FACTOR DEFINITION 带叠字
   （限行 + 省略标记已落地，C5 重渲染后残留形态需目视确认）。
5. **census 守卫 cwd 陷阱**：读真实语料的 census 测试在无语料的 worktree 里静默
   skip——全量对账的 gate 行必须报「语料可达/不可达」两组数字，别只看总数。
6. **C5 首轮全量跑执行在 PR #109 之前的判据上 ⇒ cell ceiling 一次都没上过场**（登记，
   不补跑）。`warmup_ceiling=` 字段由 `35de346`（2026-07-28 **02:08**）引入，而 sweep 跑在
   **01:50→05:52**；**幸存的 8 条日志行没有一条带这个字段**（带的三条全是 07-30 的重跑）
   ⇒ 整轮 sweep 用的是 #109 之前的分类器。**影响已结清**：ceiling 只对 bounded 因子有效，
   在当前 HEAD（ceiling 生效）上重算，三个 bounded 因子全过——8198/8955、17289/18905、
   17272/18905。**没有藏住任何失败。** 不补跑的理由：B 阶段本来就在当前 HEAD 上重跑全部
   11 个因子，ceiling 届时真正上场。但**首轮那一列 panels 结果是另一套判据产出的**，
   §二之二 的表要这么读。
7. **reports 腿在 C5 首轮零观测**（登记）：11 个 `rc=1` **全部是 F5**（缺输入文件抛异常），
   没有一个是判定结果。⇒ 该腿至今只被真正观测过**一个**因子（`amp_marginal_anomaly_vol_20`，
   F1 修复后重跑 rc=0，六格差异全落在已登记类内）。**B 阶段是这一腿的首次观测**；§四的
   A/B/C 分解因此**只有冻结基线一个参照物**（见 §四的告示）。B 中若 reports 出现类外差异：
   **停下上报，不扩类**。
8. **jump 与 `with_book(decision)` 两格对"值改动"实际不设闸**（背景，**非本 PR 引入**，登记
   即可、本 PR 不改）：jump 的 reports 腿 593 个 diff 里 **518 个**归入既有登记 #11 的
   `registered_correction_effect`（契约 v1.1 的更正承载一旦存在，值差异即被接受）；
   `with_book(decision)` 格同理是 report-only（登记 #13，`book_view_effect` 全量报告不设闸）。
   **结构性增删（未登记的新增/删除、未登记的 section）仍然设闸。** 读判定时应以
   `no_book` / `bookclose` 两个 strict 格 + panels 腿为值级依据。
9. **`_make_logger` 是 truncate 模式 —— 重跑会静默销毁同名日志的唯一副本**（登记为已知
   运维风险，**本 PR 不改**：那是行为改动，属另一个 PR）。本轮已被它咬过一次：为验证判据
   改动而重跑 `intraday_amp_cut_10` / `peak_interval_kurtosis_20` 的 panels 腿，覆盖掉了这
   两条 C5 原始日志行（值经"覆盖前逐字抓取 + 离线独立重算"双路复原，见 §二之二 出处）。
   与 D5a 抓到的"跑一次就毁掉唯一冻结基线"是**同一族失效模式**：**产物是唯一副本，而写它
   的动作没有意识到这一点**。缓解：`tmp/context/cc_c5_audit/c5_run_logs_preserved/` 存了
   B 之前的快照，且 §二之二 已把数字誊进**被提交的**报告。
