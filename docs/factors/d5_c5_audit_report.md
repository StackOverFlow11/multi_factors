# D5 C5 —— 四腿全量对账审计报告

> **状态**：**骨架**。结果区全部 TBD，由全量对账（11 因子 × panels/reports/anchors）
> 跑完后填充。填充纪律：每处差异必须落在 §二 预登记清单的具名条目内；**未编目的
> 差异 = 验收不通过**（设计 v3.2 §五腿 3：「允许差异，但每处差异必须归因到具名原因；
> 变好的差异优先怀疑泄漏」）。
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
- **harness 加固（本准备步）**：bounded `warmup_left_extension` 类加**结构性 cell 数
  ceiling** =（w−1 个 warmup 交易日）×（冻结面板 symbol 数）——该类按定义不可能超过
  此界，越界即 FAIL（C4a 评审 LOW-1；类内**幅度仍不设限**的既有裁定不变，理由：两种
  加载几何在该区本来就合法地不同，区外仍由 1e-12 把门）。

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

## 三、每因子结果表（TBD）

填写口径：panels = 具名类 cell 计数 + unclassified 计数（必须为 0）；reports =
no_book / with_book(decision) / with_book(bookclose) 三格各自的登记类计数与
unregistered 计数（strict 格必须为 0）；anchors = ok/warmup/failed 行数（failed 必须
为 0）。`stk_mins_live_calls` 每因子必须为 0。

| 因子 | panels: warmup / float_tail / flip_tail / flip_contam / footprint / unclassified / ceiling | reports: no_book | reports: with_book(decision) | reports: with_book(bookclose) | anchors: ok / warmup / failed | live_calls |
|---|---|---|---|---|---|---|
| jump_amount_corr_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| minute_ideal_amp_10 | TBD | TBD | TBD | TBD | TBD | TBD |
| amp_marginal_anomaly_vol_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| volume_peak_count_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| intraday_amp_cut_10 | TBD | TBD | TBD | TBD | TBD | TBD |
| peak_interval_kurtosis_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| valley_relative_vwap_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| valley_ridge_vwap_ratio_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| ridge_minute_return_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| valley_price_quantile_20 | TBD | TBD | TBD | TBD | TBD | TBD |
| peak_ridge_amount_ratio_20 | TBD | TBD | TBD | TBD | TBD | TBD |

## 四、with_book 差异分解表（TBD）

交接 §3 的硬要求：with_book 的差异必须可分解、**整体归因不可接受**（「with_book 变了
是因为书视图改了」会把引擎回归藏在 intended change 后面）。三分量：

- **A = no_book 引擎效应**：no_book 格的登记类差异（干净的「有没有弄坏东西」测试，
  先对干净）。
- **B = close 书引擎效应**：`with_book(bookclose)` 格（legacy-faithful close 书，strict）
  的登记类差异——引擎在带书条件下的效应，与 A 同口径可比。
- **C−B = 书视图修正**：`with_book(decision)` 格的 `book_view_effect` 叶子（decision 书
  vs close 书的预期差异，全量列出不设闸）——即 §1.1 活缺陷（close(d) 书）的修正幅度。

| 因子 | A（no_book 各类计数） | B（bookclose 各类计数） | C−B（book_view_effect 叶子数 / 涉及轴） | 分解闭合？（B 超 A 的部分是否全部具名） |
|---|---|---|---|---|
| jump_amount_corr_20 | TBD | TBD | TBD | TBD |
| minute_ideal_amp_10 | TBD | TBD | TBD | TBD |
| amp_marginal_anomaly_vol_20 | TBD | TBD | TBD | TBD |
| volume_peak_count_20 | TBD | TBD | TBD | TBD |
| intraday_amp_cut_10 | TBD | TBD | TBD | TBD |
| peak_interval_kurtosis_20 | TBD | TBD | TBD | TBD |
| valley_relative_vwap_20 | TBD | TBD | TBD | TBD |
| valley_ridge_vwap_ratio_20 | TBD | TBD | TBD | TBD |
| ridge_minute_return_20 | TBD | TBD | TBD | TBD |
| valley_price_quantile_20 | TBD | TBD | TBD | TBD |
| peak_ridge_amount_ratio_20 | TBD | TBD | TBD | TBD |

## 五、判定（TBD）

| 因子 | 腿 1 性质映射 | 腿 2 anchors | 腿 3 reports | 腿 4 panels | 判定 |
|---|---|---|---|---|---|
| （11 行，每行 PASS/FAIL + 一句话归因） | 59/59（本准备步已立） | TBD | TBD | TBD | TBD |

**整体结论**：TBD（全 11 因子四腿全过才进入 C6 删除旧 runner；任一不过则旧 runner
不删、迭代新 runner——设计 v3.2 §九 D5 行回滚条款）。

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
