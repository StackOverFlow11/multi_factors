# D5 C2 —— 11 个 eval runner 的差异编目（统一前置）

> **状态**：D5 交付物 C2。设计出处 v3.2 §六.5：11 个 runner 的差异**先编目再合并**，
> 合并后按 §五语义对账逐项归因——**「未编目的差异」算失败**。
> 本表是 **C5 对账的差异白名单唯一来源**：对账中出现表外差异 = 验收不通过。
> 编目方法：全 11 份 AST body 比对 + 11 份 config 展平 YAML 全量比对 + 逐个分歧区精读。
> 路径均相对仓库根；行号为编目时实测。

**总量**：11 个 runner 共 6550 行（496–749 行/个）。

---

## 一、结论：只需要四个扩展点

| # | 扩展点 | 为什么不能做成固定 body |
|---|---|---|
| 1 | **因子构造 + per-symbol reducer，kwargs 不透明（2–9 个）** | 各因子预注册参数个数与名字都不同；且 reducer 返回**三种形态** |
| 2 | **可选的 post-loop 截面 finalizer** | 2 个因子的值**不是**逐 symbol 产出的；其中 1 个还要日频面板 |
| 3 | **可选的 add-Section 覆盖披露** | 4 个 runner 各有一份**形状不同**的稀缺性披露，且这是**故意的** |
| 4 | **metric 键集** | 三档 19/27/29，但**与因子无关**（见下），可收敛 |

其余全部可归一（§四 C1–C13）。

## 二、(a) 真参数 —— 必须作为 runner 参数存活

### 2.1 per-symbol reduction 形态（**最承重的一条**）

统一 runner **不能假设 per-symbol reducer 返回因子值**：

| 形态 | runner 数 | 成员 |
|---|---|---|
| 纯 series | 6 | jump_amount_corr / minute_ideal_amplitude / amp_marginal_anomaly_vol / volume_peak_count / peak_interval_kurtosis / valley_relative_vwap |
| series + diagnostics sink | 3 | valley_ridge_vwap_ratio / ridge_minute_return / peak_ridge_amount_ratio |
| **stats frame + 强制截面 finalizer** | 2 | **intraday_amp_cut / valley_price_quantile** —— 循环**根本不产出因子** |

### 2.2 post-loop 截面 finalizer（2 个）

- `qt/eval_intraday_amp_cut.py:172-177` → `combine_amp_cut_cross_section(stats, min_cross_section=…)`（研报第 4 步截面 z-score）。其覆盖率判定测的是 `V_MEAN_COL` **与** `V_STD_COL` 双 finite（`:155`），不是"series 非 NaN"。
- `qt/eval_valley_price_quantile.py:281-284` → `reversal_20(panel[["close"]])` + `residualize_on_reversal(raw, rev, min_cross_section=…)`。

> ⚠️ 与 D5 饱和探测（`docs/factors/d5_saturation_feasibility.md`）**直接耦合**：
> `intraday_amp_cut` 的截面步正是 per-symbol 流式化切口必须落在其**之前**的原因。

### 2.3 需要日频面板的 loader（1 个）

`_load_valley_price_quantile_panel(cfg, symbols, spec, panel, logger, *, …)`
（`qt/eval_valley_price_quantile.py:202-207`，调用点 `:598`）——**唯一** 5 位置参数 loader，
其余 10 个都是 4 个。reversal 取自前复权面板 `panel['close']` 的 T-1。
→ **reducer 签名必须能接住日频面板。**

### 2.4 因子构造参数（全部为模块常量，无一 config 驱动；kwargs 个数 2–9，不能做固定列表）

| runner | Factor 类 | lookback | 额外 loader kwargs |
|---|---|---|---|
| jump_amount_corr | JumpAmountCorrFactor | 20 | `min_pairs=10` |
| minute_ideal_amplitude | MinuteIdealAmplitudeFactor | 10 | `lam=0.25, min_minutes=1150` |
| amp_marginal_anomaly_vol | AmpMarginalAnomalyVolFactor | 20 | `min_pool=460, min_selected=20, sigma_k=1.0` |
| volume_peak_count | VolumePeakCountFactor | 20 | PRV-5 |
| intraday_amp_cut | IntradayAmpCutFactor | 10 | `lam=0.20, min_day_minutes=100, min_valid_days=6, min_cross_section=10` |
| peak_interval_kurtosis | PeakIntervalKurtosisFactor | 20 | PRV-5 + `min_intervals=20` |
| valley_relative_vwap | ValleyRelativeVwapFactor | 20 | PRV-5 + `min_valley_bars=20` |
| valley_ridge_vwap_ratio | ValleyRidgeVwapRatioFactor | 20 | PRV-5 + `min_valley_bars=20, min_ridge_bars=10` |
| ridge_minute_return | RidgeMinuteReturnFactor | 20 | PRV-5 + `min_ridge_bars=10` |
| valley_price_quantile | ValleyPriceQuantileFactor | 20 | PRV-5 + `min_valley_bars=20, min_cross_section=10, reversal_days=20` |
| peak_ridge_amount_ratio | PeakRidgeAmountRatioFactor | 20 | PRV-5 + `min_peak_bars=5, min_ridge_bars=10` |

PRV-5 = `baseline_days=20, baseline_min_obs=10, sigma_k=1.0, min_valid_days=10, min_classifiable=100`
（`factors/compute/minute/primitives.py:55-59`，8 个 runner 共享）。
**这些是预注册定义，参数化保留、不统一**（v3.2 §〇 总原则）。

## 三、(b) 真异构披露 —— 走 add-Section 扩展点

4 个 runner 各有一份稀缺性/中性化披露，**无共享基类**，4 个独立 frozen dataclass 各带 `render()`。
两种机制：

| runner | 类 | 定义行 | 机制 |
|---|---|---|---|
| valley_ridge_vwap_ratio | `RidgeCoverage` | `:110-233` | A：循环中 diagnostics sink（`:310`） |
| ridge_minute_return | `RidgeReturnCoverage` | `:120-253` | A：sink（`:328`） |
| peak_ridge_amount_ratio | `PeakCoverage` | `:122-243` | A：sink（`:320`） |
| valley_price_quantile | `NeutralizationCoverage` | `:121-182` | B：循环后从已装配面板算（`:285`），无 sink |

**它们不是同一对象换字段名**——共有 `symbol_days / classifiable_days / valid_days /
days_below_classifiable_gate / min_classifiable`，其余各不相同：

- `RidgeCoverage`：`ridge_percentiles / ridge_mean / valley_median`；两道 gate；反事实 `valid_days_at_valley_floor`
- `RidgeReturnCoverage`：另**追踪两个计数** `ridge_bars_mean/median`；额外属性 `return_guard_attrition`；反事实 floor `_COMPARISON_FLOOR = 20`（**硬编码**对齐 PR-J）
- `PeakCoverage`：反事实 floor `_COUNTERFACTUAL_PEAK_FLOOR = PEAK_RIDGE_MIN_RIDGE_BARS`（**派生**，非硬编码）
- `NeutralizationCoverage`：形状完全不同——`raw_rows/rev_rows/residual_rows/dates_total/dates_residualized/cross_section_min,median,max/raw_rev_spearman_mean`

**披露措辞**（4 条 format 模板，同时进 run log 与 stdout）：
`eval_valley_ridge_vwap_ratio.py:151-168` / `eval_ridge_minute_return.py:173-189` /
`eval_peak_ridge_amount_ratio.py:164-177` / `eval_valley_price_quantile.py:293-301`。
各类 docstring 均强调：**报出的 floor 必须是本次 run 实际施加的那个**，不是模块默认值。

**判定 (b) 而非 (c)**：这些是各因子**自己 gate 的实测事实**，量纲与语义都不同，
强行统一 = 丢信息。这正是 §3.6 的 add-Section 扩展点（"may ADD sections but never
drop a mandatory one"），**不是第二套 runner**（kill 档案 A8-F03）。

**一处归一**：PR-L 在 `qt/cli.py:525-530` 用**内联 f-string** 重渲染，而其三个兄弟走
`.render()`（`:436` / `:479` / `:570`）→ 统一走 `.render()`。

## 四、全 11 份完全一致（可做固定 body，无参数）

| 轴 | 值 | 证据 |
|---|---|---|
| **horizon** | `h = 1`，且**根本不是 runner 关切** | 来自 `FactorSpec.forward_return_horizon`，仅在共享模块 `qt/exec_basis_eval.py:213` 消费。**不要加 horizon 旋钮** |
| **`execution_capacity` 透传** | **零个 runner 设置**；仅共享模块 `qt/exec_basis_eval.py:305`(no_book) / `:315`(with_book)，同一 `disclosure` dict（`:97-151`） | 该 dict **故意省略** `tradable`/`capacity_sufficient` 两个 Tradable 轴会读的键 → 该轴恒 NOT_ASSESSED，模块 docstring `:21-27` 明写 |
| **因子簿** | `ValueFactor("value_ep"), ValueFactor("value_bp"), VolatilityFactor(window=20)` | `eval_jump_amount_corr.py:480-488`，**11 份 AST body 逐字节相同** |
| **费率/成本** | `fee_rate = cfg.cost.fee_rate`（配置 0.001）；`cost_scenarios = (1.0, 2.0, 4.0)` 11 份硬编码相同 | `qt/exec_basis_eval.py:304, 313` |
| **分钟读取窗口** | 11 份相同：`[data.start, data.end 23:59:59]` | **无 runner 向前扩窗做 warm-up** → 前 lookback_days 结构性 NaN（与 D4 饱和工作直接相关） |
| **process/中性化** | 因子一次 + 因子簿一次 `_process_factors`；`neutralization=("industry","size"), standardize="zscore", winsorize=None` | 唯一例外是 PR-L 的 reversal 残差化，发生在 loader **内部**、`_process_factors` **之前** |
| **报告/图输出** | `{stem}_no_book|_with_book` .md+.json + 2 PNG；exec 侧同形 | `eval_jump_amount_corr.py:221-228`；`qt/exec_basis_eval.py:321-330` |
| **`_exec_basis_sanity.md`** | **仅**共享模块 `qt/exec_basis_eval.py:271-283` 产出，11 份无差异，无 runner 代码触碰 | 渲染器 `qt/exec_basis_sanity.py:355+` |
| **config** | **D1 manifest 的声称经复核成立**：11 份展平 YAML 共 56 个 union 键，**恰有两个不同**：`project.name` 与 `data.output_name` | 其余逐值相同（CSI500 / 2021-07-01..2026-06-30 / split 2024-01-01 / fee 0.001 / quantiles 5 / top_n 50 / L1 / cache root / factors 列表） |

**`execution_capacity` 附注**：CLAUDE.md 把「`execution_capacity` 透传是否在作者自带测试之外
仍保持 Tradable 轴 `NOT_ASSESSED`」列为 **#79 未独立核**三项之一。本次编目**关闭了这一项的
结构面**：透传是**单一共享代码路径、无 runner 变体、且不含任何能动 verdict 的键**，
故该性质**按构造**不可能逐因子变化。（另两项仍未核。）

### metric 键集：三档，但**与因子无关**

19 键（8 个）/ 27 键（ridge_minute_return，`:490-519`）/ 29 键（valley_price_quantile `:436-465`、
peak_ridge_amount_ratio `:488-517`）。**多出的键全部取自每次 run 都产出的 `return_risk` /
`stability_cost` payload**——即「PR-K 之后开始多报几个」的**报告选择**，不是因子性质。
→ **收敛到 29 键超集 减去 `aligned_spread_by_cost`**（该键不存在，见 BUG 2）。

## 五、(c) 无意义漂移 —— 已判定归一，逐条记录

| # | 漂移 |
|---|---|
| C1 | `_build_eval_config` 11 份不同 body、语义全同；差别仅 `oos is None` 错误串里的 runner 名。17 个 EvalConfig kwargs 逐字符相同 |
| C2 | `_check_preconditions` 同上，同 4 项检查，差别仅嵌入的 runner 名 |
| C3 | 11 个 result dataclass 名不同，14–16 字段相同 |
| C4 | 11 个私有 load-dataclass 名不同 |
| C5 | 局部变量命名 `jump_raw` / `amp_raw` / `factor_raw` |
| C6 | 11 个 `_load_*_panel` 函数名不同，11 行骨架相同 |
| C7 | 日志措辞（`rows read` 换行位置；`symbols with a value` vs `with a stat`） |
| C8 | `logger.info("eval config: …")` 单行 vs 折行 |
| C9 | `_write_report` 声明单行 vs 四行折行，body 相同 |
| C10 | docstring 88 vs 90 列重排——**任何两份 byte-diff 的主体其实是这个** |
| C11 | 仅 4 个大 runner import numpy（只被覆盖率 summarizer 用） |
| C12 | `qt/cli.py:206-580` 11 份 `_cmd_run_eval_*`，同 try/except + f-string |
| C13 | 两份相同 `_fmt` helper：`qt/cli.py:395-409` 与 `qt/exec_basis_eval.py:355-367` |

## 六、发现的缺陷（**已标记，未修**）

> 按 lead 指令「flag separately，do not fix」。以下 BUG 1/2 由本 agent **独立复核确认**
> （非仅采信编目），复核方式记在各条。

### BUG 1（CRITICAL，用户可见）—— runner 描述了一个 **PR #74 已经修掉**的缺陷

**声称**（`qt/eval_ridge_minute_return.py:477-481`，并在 `:49-56`、`:705-708` 复述）：
> "The frozen layer's `aligned_spread_*` computes `sign * (gross - cost)` … costs are added
> back rather than deducted … **The frozen layer is not patched**"

**实际**（`analytics/eval/standard.py:419`，本 agent 直接读取确认）：
```
aligned_base = sign * gross - base_cost
```
上方注释以**过去时**描述旧缺陷（"The pre-v0.8 form `sign * net` expanded, at sign=-1, to
`-gross + cost`"）——即 PR #74 的 v0.8 修复**已在位**。

**且它每次 PR-K run 都打到 stdout**（`qt/cli.py:443`
"net long-short by cost (aligned_spread_* UNRELIABLE at sign=-1)"）并写进 run log
（`eval_ridge_minute_return.py:708`）。另有四处断言同一死声称的否定形式：
`qt/cli.py:484-485`、`:535-536`、`eval_valley_price_quantile.py:57-59` 与 `:422-424`、
`eval_peak_ridge_amount_ratio.py:475`。

**祖先关系已验**：`08829bd`（v0.8 cost-correct aligned spread）**既是** HEAD 祖先、
**也是** `36c8c85`（最后触碰 ridge runner 的 commit）的祖先 → **该文字在最后一次被编辑时
就已经是陈旧的**。

**这正是 #76/#78/#82 的失效模式**：行为改了、措辞没改。按项目自定规则，修法是
**author-once（只写一遍、其余组合引用）**，不是再复制一份正确措辞。
→ **归属**：本条与 D5 统一 runner 天然同批（11 份措辞坍缩成一份），但**它是独立缺陷，
不因 D5 而存在**；若 D5 拆期，本条应单独修。

### BUG 2（HIGH）—— `aligned_spread_by_cost` 被提取，而 frozen 层从不产出它

`eval_valley_price_quantile.py:461` 与 `eval_peak_ridge_amount_ratio.py:513` 都做
`dict(ret_risk.get("aligned_spread_by_cost", {}) or {})`，但 `analytics/` 全仓**不存在**该键
（只有 `aligned_spread_annual_return|sharpe|volatility|max_drawdown|sortino|final_nav`
与 `net_long_short_by_cost`）。

**本 agent 独立复核**：直接读**冻结的 exec artifact** 的 `return_risk` payload，
两个因子的 `aligned_spread_by_cost` 均 **不存在**；实际 `aligned_*` 键恰为上述六个。
→ 两个 runner 报的**永远是 `{}`**。

**唯一"守卫"是一个不可能失败的测试**：
`tests/test_eval_valley_price_quantile_runner.py:485`
`assert isinstance(m["aligned_spread_by_cost"], dict)` —— `{}` 是 dict，**无条件通过**。
正是项目方法论 ① 点名的形态。

### BUG 3（HIGH）—— `qt/eval_peak_ridge_amount_ratio.py` **完全没有 runner 测试**

11 个里 10 个有 `tests/test_eval_<name>_runner.py`；PR-M 没有
（`grep -rn "qt.eval_peak_ridge_amount_ratio" tests/` 无命中）。
后果：`PeakCoverage` + `summarize_peak_coverage`（`:122-243`，**122 行披露逻辑**）
**零测试覆盖**，而三个兄弟 summarizer 都有反事实 + 空帧测试。
`tests/test_peak_ridge_amount_ratio_factor.py` 只覆盖因子数学。

### BUG 4（MEDIUM）—— PR-M 的 run log 比终端输出更贫

它提取全 29 键、CLI 也打印 net-spread/pearson/monotonicity（`qt/cli.py:486-489`），
但不像 PR-K（`:708`）/ PR-L（`:654,:660`）那样发对应 `logger.info`。

### BUG 5（LOW）—— config 声明了一份 runner 根本不读的因子簿

每份 config 都有 `factors:[value_ep,value_bp,volatility_20]`，与硬编码
`_build_book_factors()` 相同，但**无 runner 读 `cfg.factors`**。今天两者一致故无害，
但**改 config 会静默无效**。同形：`backtest.rebalance='monthly'` 而
`_build_eval_config` 硬编码 `rebalance='daily'`；整个 `backtest:/portfolio:/alpha:` 块对这些 run 惰性。

### BUG 6（LOW / 近失）—— 重名测试函数，**目前**两份都被收集

`test_summarize_ridge_coverage_handles_no_frames` 同时定义于
`tests/test_eval_ridge_minute_return_runner.py:246` 与
`tests/test_eval_valley_ridge_vwap_ratio_runner.py:232`。
**因为 `tests/__init__.py` 存在**，两份都被收集（实测该两文件收 10+7=17）。
→ ⚠️ **对 C6 的直接警告**：若统一 runner 的测试整合过程中删掉 `tests/__init__.py`，
这条会立刻变成静默丢测试。

### BUG 7（LOW）—— `eval_ridge_minute_return.py:471` 自称"Beyond the fields PR-C..PR-J surfaced"，
但 PR-J（valley_ridge_vwap_ratio）属 19 键基础档，它点名的对照集在自己的边界上就是错的。

## 六之二、C6 爆炸半径：谁 import 这 11 个 runner（**删之前必须处置**）

严格说这不是「runner 之间的差异」，但删除动作的影响面必须落在纸面上——与 BUG 6
（`tests/__init__.py` 收集陷阱）同一接口。**非测试模块共四个**（实测
`grep -rln "qt\.eval_" --include=*.py | grep -v ^./tests/`）：

| 模块 | 用途 | C6 处置 |
|---|---|---|
| `qt/cli.py` | 11 个 `_cmd_run_eval_*` 子命令（§五 C13） | 随统一 runner 收敛为一个子命令 |
| **`qt/panel_freeze.py`** | **生产 D1 冻结基线的工具**——调各 runner 私有 `_load_*_panel` + `_build_book_factors`，「零公式重抄」正是靠 import 它们实现的 | **删 runner 会打断「重新生成 / 重新验证 D1 基线」的能力** |
| **`qt/panel_reconcile.py`** | D2 逐格对账工具（同上依赖） | 同上 |
| **`qt/hand_anchors_engine_values.py`** | D2 手算锚的引擎侧取值 | 同上 |

⚠️ **后三个全是 D1/D2 的验收工具**，而 **D5 面板腿的比较对象正是 `panel_freeze.py` 的产物**。
按设计 v3.2 §五第 4 腿的 **provenance 规则**，基线只许从钉住的 pre-D2 SHA 重新生成——
所以严格说 C6 之后从**当前树**重跑它们本来就不合法。但「工具还在、只是不该从这里跑」与
「工具已被删、想复核也无从下手」是两回事：**这与 C1 存在的理由是同一种不对称损失**
（复核能力一旦丢失，代价远大于保留一份不再调用的代码）。

**处置要求（C6 执行时逐条落实，不得默认删掉了事）**：① 三个工具**要么保留**（连同它们
import 的 runner loader，即便统一 runner 已上线）、**要么显式记录**「D1/D2 基线自此不可从
本仓再生，只能依赖已冻结的 artifact + manifest 哈希」并让 lead 知情裁定；
② 无论哪种，`docs/factors/d1_panel_freeze_manifest.md` 的「重跑命令」一节都会变成陈旧描述，
**必须同批更新**——否则就是本项目 #76/#78/#82 那条形态的又一次复发（文档教人跑一条已经
跑不了的命令）。

## 七、与 C5 对账的接口

**本表 §二/§三 = 允许出现的差异白名单；§四 = 必须逐值一致的项；§五 = 已判定归一（对账中
不应产生数值差异）。**

另有一条**已知且必然**的 artifact 差异，来自 D1/D3 而非 D5，**必须计入白名单**
（本 agent 独立复核确认）：

> **`spec` 块键数 16 → 20**。冻结 artifact 的 `spec` 有 16 键，当前代码产出 20 键，
> 新增恰为 `requires` / `adjustment` / `overnight_boundary` / `lookback_depth`，**零删除**。
> 成因：`render.py:156` 的 `sanitize_payload(vars(report.spec))` 会把 FactorSpec 的**任何**
> 新字段自动带进 JSON，而这四个字段是 PR #86（契约 v1.0）与 PR #91（v1.1 `lookback_depth`）
> 加的。`requires` 经 `clean_value` 的 `str()` 兜底渲染成 **repr 字符串列表**。
> **属预期漂移，非回归**——但若不预先登记，会在对账时被当成 D5 引入的差异去追。

### 七之二、C3（评估契约 v1.0）引入的 artifact 漂移 —— 同样预先登记

C3 升级 `analytics/eval` 契约至 v1.0（版本陈述见 `analytics/eval/contract.py` 的模块
docstring，按 #74 先例）。它改变 artifact 的**三处且仅三处**，全部为**新增**，零删除、
零数值变化：

| # | 位置 | 变化 | 成因 |
|---|---|---|---|
| 1 | JSON `eval_config` 块 | **+3 键** `view` / `return_basis` / `book_view` | `report_to_dict` 的 `sanitize_payload(vars(report.cfg))` 自动带上 `EvalConfig` 的新字段（与 §七 的 `spec` 16→20 完全同一机制） |
| 2 | JSON 顶层 | **+1 键** `eval_contract_version` | 显式写入：一个 verdict 只有对着产生它的契约版本才可解释（#74 的教训） |
| 3 | Markdown `## 0. Header & Provenance` | **+4 行** `evaluation contract` / `requires (endpoint inputs)` / `adjustment / overnight boundary` / `lookback depth ...` | R24 的身份字段 + D1 契约 v1.0/v1.1 的三个声明维，从 `vars(spec)` 的 repr 转述升级为具名行 |
| 4 | JSON 顶层 | **+1 键** `corrections`（契约 **v1.1**，见下） | 更正是**结构化事实**，必须在**有长度上限的自由文本通道之外**传达 |
| 5 | Markdown `## 0.` + dashboard PNG | 每条更正 **+1 行** provenance 行；PNG 头行 **+1 个** `CORRECTED vX -> vY` 标记（无更正的因子零新增） | 与 #4 同一结构化元组派生（author once），三个呈现面不可能互相矛盾 |
| 6 | dashboard PNG FACTOR DEFINITION 带 | 描述被**限行**（`DEFINITION_MAX_LINES=4`），超出部分标注省略并指向 Markdown | **修既有缺陷**，见下 |

**#4/#5 的成因（契约 v1.1）**：`sanitize_payload` 对**任何**导出字符串封顶 200 字符
（`MAX_VALUE_CHARS`）并追加 `...[truncated]`。这是**通用行为、非某条路径特有**——实测
已发布的 **44/44** 份 eval JSON 的 `spec.description` 全部被截断（共 218 个
`[truncated]` 标记）。所以写进 `description` 的更正只到得了 Markdown 与 PNG，在 JSON ——
**汇总层读的那一份**——里被切掉。v1.1 把更正改为 `FactorSpec.corrections`
（`FactorCorrection` 元组），经 `corrections_record` 在**封顶通道之外**导出（仍脱敏），
且超长在 spec 构造期 **raise 而非裁剪**。

**#6 的成因（既有缺陷，非本次回归）**：FACTOR DEFINITION 带把描述锚在 y=0.62（va=top）、
metadata 行锚在 y=0.20/0.10（va=bottom），**中间没有任何约束**，长描述直接**画在 metadata
上面**，两行都糊。真实渲染几何实测：**11 个分钟因子里 7 个重叠**
（`valley_price_quantile_20` 超出 **296 px** / 25 行对 4 行槽位）。
`analytics/eval/figures.py` **在 git 里只有一个 commit**，所以这不是新引入的——**每一份已
产出的 dashboard 都是这样画的**，包括冻结的 22 张。限行 + 显式省略标记严格优于覆盖绘制
（后者两段文字同时不可读且不声明）。守卫 `tests/test_definition_band_layout.py` 用**真实
dashboard 几何**渲染后比对 Text 的 bbox，**几何断言而非看图**。

⚠️ **`book_view` 使 exec 侧 no_book 与 with_book 两份 artifact 的 `eval_config` 块首次不同**（`null` vs `"close"`）。这是**有意**的：一次 with-book 评估携带**两个**信息集（候选因子的与因子簿的），一个 `view` 字段表达不了；设计 §1.1 记录的因子簿 close-view 活缺陷要到 D7 才关，在那之前诚实的 artifact 就该写 `view=decision, book_view=close`。对账时**不要**把这一处不同当作 no_book/with_book 之间的回归。

✅ **已解除（解除于截断修正 PR，本节保留作记录）**：`jump_amount_corr_20` 的 exec 评估曾经 loud raise（`qt.exec_basis_eval.exec_identity`）——它的值是 close 视图（compute 无 14:50 截断，实测），(close, exec_to_exec) 非法配对。那是**故意的阻断**：在该因子被截断修正之前，它没有诚实的 exec artifact，而写一份声明 `view=decision` 的假 artifact 正是本字段要防的事。

独立的 correctness-fix PR 已把该因子截断到 14:50（用与十个兄弟相同的 `visible_minute_frame`），并在**同一个 PR 里**从 `factors.compute.minute.binding.NOT_DECISION_CUTOFF_SAFE` 移除该条 + 用旧 runner 重跑重述了它的 exec verdict（Watch/Watch 不变，数字见 `docs/factors/pr_c_cutoff_fix_reference_panel.md` §六），所以不存在「因子已修好、评估仍被拦」的中间态。`NOT_DECISION_CUTOFF_SAFE` 现在为空集，且这个空集是**被测量的**：`tests/test_decision_cutoff_visibility.py` 把十个 bars-bound 因子全部纳入 clean 参数化——**它正是逼出这次删除的东西**（截断落地后它红了，并在失败信息里给出「drop it from … and re-state its verdict」）。

⚠️ 连带影响（D5b 评审的 NIT，自动消失）：那个阻断点在 runner 里偏晚（跑完逐票分钟聚合、写完两份 close_to_close 报告之后才 raise）。deny list 条目移除后该路径不再 raise，此 NIT 无需单独处理。

⚠️ **C5 面板腿注意**：该因子的 D1 冻结面板（`artifacts/refactor_baseline/panels/jump_amount_corr_20.parquet`）是用**未截断**定义冻的，仍按原样保留不动；对账请改用 `artifacts/refactor_baseline/pr_c_cutoff_fix/`（同一条旧 runner 取值路径 + 截断输入）。provenance 与「为什么两份并存」见 `docs/factors/pr_c_cutoff_fix_reference_panel.md`。

**没有变的**（`tests/test_eval_contract_v1.py` 逐值钉住）：`VerdictThresholds` 的**每一个**
默认门（`min_abs_icir=0.30` / `min_incremental_abs_icir=0.15` /
`min_monotonicity_spearman=0.0` / 三段样本门），以及三轴判决 / 非对称门 /
unknown-never-convicts / N_eff CI / exploratory 封顶 Watch 的全部规则。设计 §十 R24 明令
**禁止**在被评判的 run 上调门；门槛标定要单独预注册 run。

⚠️ 因此 C5 对账在**逐字节**层面对 22 份 md/json 一定不成立，能成立的是 **IC / ICIR /
分位价差 / verdict 的值级对账**加上「JSON 差异恰为上表三项 + §七 的 spec 四键」这个
**结构性**断言。把「逐字节相同」当作 C5 的通过条件会让人在这里误判为回归。

### 七之三、C4 首轮对账（3 因子 × 3 模式）暴露的判据缺口 —— 补登记（2026-07-27）

C4 harness（`qt/factor_eval_reconcile.py`）首轮对账 minute_ideal_amp_10 / jump_amount_corr_20 /
volume_peak_count_20 三模式全红。独立调查（证据脚本 `/tmp/q1_dump.py` / `/tmp/q1_twogeom.py` /
`/tmp/q1b_tail.py`，首轮日志 `/tmp/c4_reconcile_run.log`）把**全部**差异归到两个根因，均为
**加载几何/浮点层面、非取值 bug**。本节把判据补全为具名类，每条都有边界与反例约束
（类外差异仍非零退出——未编目的差异算失败）。

**认知修正（本节最重要的一条）**：C4 handoff §3 ① 曾预测「bounded 因子 vs D1 baseline 预期
**零差异**，因为旧 runner 也逐票读全窗口」。**这个预测错了**：旧 runner 的「全窗口」左端同样
锚在 `data.start`（2021-07-01）——即 **anchor 截断**；新 materializer 的饱和加载左延到分钟缓存
真实起点（2015-01-05）。首轮「预期全绿却没绿」的源头是这个预测，**不是引擎 bug**。

**根因一：warmup 左延漂移（设计内）**，具名类 `warmup_left_extension`，bounded 与 pooled 通用，
覆盖**全部三个方向**（frozen-NaN→new-finite、finite→finite 即 partial pool→full pool、
new-only finite 行——旧 runner 网格根本没发射的那些行）：

| 形态 | 边界（实测精确） | 首轮实测 |
|---|---|---|
| bounded（minute_ideal_amp_10 / jump_amount_corr_20） | 冻结网格**前 w−1 个交易日**（w=该因子 `lookback_depth`；第 w 日已完全 warmup）。边界取**网格左端**而非 per-symbol 首日：旧 runner 对每个 symbol 的加载都锚在 `data.start`，只有评估窗左端欠 warmup；晚上市/停牌票的首日两种几何看到同一批 bar，不可能有差异 | NaN→finite 3644 + 180；finite→finite 4554 + 17109 |
| pooled（valid-day 池化，volume_peak_count_20） | 早区窗 [2021-07-01, 2021-10-31]，且**按月计数非递增**（违反即失败） | NaN→finite 8189；finite→finite 17193（07-01..08-24 按月衰减）；new-only finite 9059（07-01..07-14，旧网格未发射） |

anchors 腿同判据：hand 侧按旧几何算、service 侧按新几何算，失败行全在 warmup 区。原 harness
只给 pooled 备了早区类、bounded 没有 → **判据不对称**（minute_ideal_amp 与 jump 的首轮 anchors
失败即此）。bounded 的 warmup 边界与 panels 腿相同（冻结网格前 w−1 个交易日）；jump 的
**非** warmup 行仍必须 reconcile（它证明 service 携带截断定义，首轮 3 个 random 行 rel ~1e-15
已绿），warmup 区外的 jump mismatch 仍 FAIL。

**根因二：两个浮点/阈值尾部（非 bug）**，各带**双边界**（不调大全局容差，保住对真回归的牙）：

| 类 | 机制 | 边界 | 实测 |
|---|---|---|---|
| `float_reordering_tail` | rolling 相关求和顺序差异（JC1 已裁 1e-12 为可归因浮点重排地板，这是地板之上的实测尾部） | rel ≤ **5e-12** 且 cell 数 ≤ **101**（超任一即失败） | jump 101 cell，5 票散在，rel 1.0e-12–2.9e-12 |
| `threshold_flip_tail` | rolling σ 浮点噪声（~4e-10）× 整数成交量恰压阈值 → 计数翻转 | 幅度**恰 ±1 count**、rel ≤ 1e-2、cell 数 ≤ **25**（超任一即失败） | volume_peak 20 cell，600623.SH 2023-06-15..07-14 每天恰 −1（整数成交量 13300.0） |

**reports 腿两处口径修正 + 三个 JSON 登记项**：

1. **MD 按 key 配对成行级 change 再分类**。裸行集合差把每个值变化报成一对（一条 removal +
   一条同 `- key:` 头的 addition）——首轮 83–96 条「removal」全是这种幻影。配对后跟随与 JSON
   相同的数值归因梯。配对变化的分类：**数字承载**（任一侧含数字，散文行的数字可能在冒号
   **之前**）才进 `warmup_aggregate_effect`；纯标签翻转（verdict PASS→FAIL 不含数字）仍
   unregistered 失败。**配对是两遍的**：先按精确 `- key:` 头配对，再对剩余行按**数字归一化**
   的头配对（实测必要：incremental 轴理由行的变化数字在首个冒号之前，精确头永远配不上；
   且该行头长 ~103 字符，配对 key 上限须 ≥200——改了散文**词**而非数字的行归一化后仍配不上，
   照失败，有反向测试锁定）。
2. `warmup_aggregate_effect`（JSON 与 MD 共用）：聚合指标叶子（`sections[*]` /
   `verdict.reasons[*]` / `verdict.axes.*`）的数值/散文变化是 panels 腿已登记 warmup 差异的
   **下游**（聚合一个 warmup cell 变了的面板；panels 腿才是值级闸门）。**牙仍在**：聚合路径
   之外（verdict 标签、spec、eval_config、criteria）任何变化仍失败；无数字的标签翻转仍失败。
3. `spec.requires[0..n]` 由精确匹配改**前缀**匹配（§七 已登记 spec 16→20 键，`requires` 是
   repr 字符串**列表**，展平后是索引叶子）。
4. `registered_sanity_stem_rename`：`sections[5].payload.sanity_report`（及 MD 同名行）的值从
   `eval_<name>_exec_basis_sanity.md` 改为 `factor_eval_<factor_id>_exec_basis_sanity.md`——
   runner 有意的防碰撞改名，双侧都以 `_exec_basis_sanity.md` 收尾才认。
5. `registered_run_order_artifact`：`sections[5].payload.exec_price_artifact_reused`
   False→True——同会话 run-order 产物（首个 run 建 exec 价格 artifact，后续 run 复用）。
   **反向（True→False）不登记，仍失败**。
6. jump 的 `spec.description` 与 `sections[7].payload.factor_version`（连同原已特判的
   `spec.version`）纳入 `registered_correction_effect`——仅当新 JSON 带 `corrections`
   结构化更正承载（契约 v1.1）时成立。

### 七之四、C4b（vpq 绑定 + runner 启用）引入的 artifact 漂移 —— 预登记（2026-07-28）

1. **`valley_price_quantile_20` 的 reports 腿：新增 add-Section `neutralization_coverage`**。
   旧 runner 对 NeutralizationCoverage 只 **log + 进 result dataclass**（§三 机制 B），从未
   add-Section；统一 runner 把它经 §3.6 扩展点装配进报告。冻结 exec artifact 没有这一节，
   故新 artifact 的 JSON `sections[8].*` 全部叶子与 MD 的整节渲染行（`## + neutralization_coverage`
   标题、note 行、9 个 payload `- key:` 行及分节空行）是**新增**。登记方式：按 section **名字**
   匹配（`qt/factor_eval_reconcile.py::REGISTERED_EXTRA_SECTIONS`，JSON 侧读新 JSON 该下标
   section 的 `name` 字段核对，**绝不按裸下标放行**；MD 侧前缀从 dataclass 字段派生，改名即红）。
   任何其他新增 section 不登记、仍失败。
2. **vpq 的 panels 腿预期差异类**：与 pooled 因子同口径——`warmup_left_extension`（旧 runner
   左端锚 `data.start` 的 anchor 截断 vs materializer 饱和左延到 2015-01-05；早区
   [2021-07-01, 2021-10-31] 三方向 + 按月非递增）。vpq 的反转中性化在旧几何是
   `reversal_20`（未滞后面板）、新几何是 `reversal_20_shifted`（决策滞后面板），两者
   **逐位等价**（C4b 双侧钉死），故平移面板本身**不产生**类外差异；左端 22 个交易日窗口的
   零 NaN 差异已在 fixture 层钉死。若真实缓存对账出现类外差异，如实全报。

### 七之五、C4b vpq 真实对账（2026-07-28 实测）——panels 腿差异分解与 anchors 修复登记

真实对账（cache-only，`stk_mins_live_calls=0`；store 由本次 run 冷填，pooled 饱和加载
1132s）：reports 4/4 OK；anchors 修复后 5/5（ok=4 + warmup=1）；**panels FAIL，差异已全量
分解如下（三块，类外 0 未解释）——具名类边界**已由 lead 裁定（2026-07-28）并落地**，
三条裁定与边界参数见本节末**：

1. **warmup 块（24,516 格，全 finite→finite，2021-07-30→2021-09-30）**：即登记的
   `warmup_left_extension`（anchor 截断 vs 饱和左延）。**但 vpq 的逐月计数结构性地
   从部分月开始**（2021-07: 902 → 2021-08: 19,975 → 2021-09: 3,639）——冻结面板的残差
   要到 ~07-29 才存在（qbar 需 ≥10 有效日 + rev20 需 21 收盘），7 月天然非满月，pooled
   规则「按月非递增」在首月必假。monotonicity 起点需按因子推迟（如从首个满月起算）。
2. **600623.SH 分类翻转块（19,143 格，2023-06-01→2023-07-14）**：直接受影响行只有
   **600623.SH 一只、30 个交易日**（max |Δ|=1.60e-04）；其余 19,113 格（959 票 ≤ 5.5e-07）
   是该股输入变化经**逐日 OLS 系数**污染全截面的二阶效应。与 C4 首轮登记的
   `threshold_flip_tail`（volume_peak_count，**同一 symbol 同一窗口** 600623.SH
   2023-06-15..07-14）同族：D2 迁移前后 PRV 分类在临界 bar 上的翻转（rolling-sigma 浮点
   噪声 × 整数成交量），在 vpq 身上表现为 valley VWAP→q_day→qbar 的连续值变化 +
   截面污染，而非 ±1 计数。qbar 本身新旧几何**逐位一致**（max 1.1e-16，旧锚 vs 饱和
   三票抽样 141 格）；日频/分钟缓存 ledger 自冻结（2026-07-24）前零写入（日频 max
   fetched_at 2026-07-18、分钟 2026-07-16），**输入侧无变化**——差异源是冻结面板由
   pre-D2 旧代码路径产生、当前引擎是迁移后 primitives。
3. **浮点尘（~232 格）**：rel>5e-12 中 abs ≤ 1e-8 的部分（残差量级 ~1e-5，rel 判据在
   近零值上失真，abs 实为 ~1e-16 机器精度）；加上 rel ≤ 5e-12 的 707 格**超过**
   `FLOAT_TAIL_MAX_CELLS`=101 的既有上限。`float_reordering_tail` 的 cap 是按 jump 实测
   标定的，对截面 OLS 残差（近零值多、格子多）系统性偏紧——需要 abs 下限或按因子标定。

**anchors 模式修复（已落地）**：截面因子（`stores_intermediate`）的 served 值是请求
universe 的函数，只请求锚行 symbol（5 只 < `min_cross_section`=10）会全 NaN——改为
**按 config 全 universe 服务再查锚格**（per-symbol 因子保持锚-only 请求）。实测：修复前
4/5 行 service=NaN（假 FAIL），修复后 4 行精确对上（2 行 rel=0.0、2 行 ~1e-15）+ 1 行
warmup_end（2021-07-30，rel 2.61e-01，属登记 warmup 类）。**intraday_amp_cut 有同样性质，
C5 全量对账会踩到同一处**——修复是通用的。

**第三条路（shifted vs legacy rev）在真实全 universe 上逐位一致**：served（引擎 shifted
路径）vs `residualize_on_reversal(qbar, reversal_20(未滞后面板))`，1,135,926 格
max|diff|=0.0、NaN 集合互差为 0——平移面板左端效应在真实缓存上**不存在**，与 fixture
层钉死一致。

**lead 三裁定（2026-07-28，已落地 `qt/factor_eval_reconcile.py` + 正反向测试）**：

1. **pooled 月单调性起点**：`warmup_left_extension` 的 pooled 按月非递增检查豁免**一个
   结构性锚定的部分月**——豁免月 = 冻结面板上该因子**首个 finite 值所在的月**（残差/值
   存在性起始的那个部分月，不参与单调性判定；该月差异仍必须在早区窗内且方向合规），
   豁免月之后仍强制非递增，违反即失败。*理由*：vpq 冻结面板的残差 ~07-29 才存在
   （qbar 需 ≥10 有效日 + rev20 需 21 收盘），首月结构性必为部分月，对它施加非递增是
   判一个构造上必假的命题。**锚定必须是结构性的（评审 LOW-1 修正）**：初版实现是位置性
   的——豁免「第一个**有** warmup diff 的月」，评审实测「豁免月零 diff + 随后 Aug→Sep
   增长形」被放行，裁定理由被绕过；改为从**冻结网格**读首个 finite 值月后，该 probe
   反转（FAIL）。冻结面板全 NaN（无 finite 值）→ 不豁免（保守方向）。
2. **新具名类 `threshold_flip_contamination`（仅截面因子）**：窗口
   [2023-06-01, 2023-07-14]；直接 symbol（600623.SH）|diff| ≤ **2e-04**；其余 symbol
   污染格 |diff| ≤ **1e-06**；总 cell 数 ≤ **20,000**。这是本类全部参数，逐格校验，
   任何越界即 unclassified → FAIL。*理由*：与 `threshold_flip_tail` 同 symbol 同窗口
   同族（PRV 临界 bar 分类翻转 × rolling-sigma 浮点噪声 × 整数成交量），vpq 表现为
   连续值（600623.SH 30 日 max 1.60e-04）+ 逐日截面 OLS 系数污染全截面（19,113 格
   ≤5.5e-07）；非输入侧变化——两缓存 ledger 自冻结前零写入、qbar 新旧几何逐位一致
   （max 1.1e-16）、同 symbol+窗口翻转已对 volume_peak 登记过。
3. **float 尾判据修两处**：① 增加 abs 下限——|diff| ≤ **1e-12** 一律计 float dust
   （近零残差上 rel 判据失真，实测 abs ~1e-16）；② cap 分档——bars-only 因子 ≤101
   （不变），**截面因子 ≤1,000**（实测 707 + 余量）。全局容差不动。*理由*：101 的 cap
   按 jump（bars-only）标定，对近零值多、格子多的截面 OLS 残差系统性偏紧；abs 下限
   把「rel 在近零值上失真」从 cap 压力里剥离，两处都不放宽对真回归的牙。

### 七之六、C5 F1（`amp_marginal_anomaly_vol_20` 5min 残桶）——**真引擎缺陷，已修，永不作为具名类**

**结论先行（给做 C5 audit 的人）**：C5 全量对账里 `amp_marginal_anomaly_vol_20` 的
panels 腿 **729,029 格类外 finite-vs-finite**，**不是**一个待登记的差异类，而是一个
**真实引擎缺陷**，已由独立 correctness PR（`fix/amp-marginal-residual-bucket`）消除。
**不要为它注册任何具名类、不要为它调任何边界参数。** 修复后实测
`unclassified=0`，且 `float_tail=0 / threshold_flip=0 / flip_contamination=0`
——它**没有**靠任何容差类兜底。

**机制**：本因子是 11 个分钟因子里**唯一派生 5min bar** 的。
`resample_intraday_bars` 按 `ceil(bar_end, freq)` 分桶，但 emit 的是桶的**真实跨度**
（`bar_end` = 最后一个成分的 bar_end），所以成分不齐的桶被 emit 成一根**更短的 bar**
（残桶），而不是被丢弃。

**残桶只有两个成因，且到不了同一批调用方**：

1. **截断**：materializer 预截到 `available_time <= 14:50`（到 14:49 那根）⇒
   `[14:46,14:50]` 桶只剩 4 个成分、`bar_end=14:49` 不在网格上 ⇒ **每个交易日一个**
   ⇒ **这就是 729,029 的全部来源**。
2. **session 内数据洞**：桶的尾部分钟缺失，同样收在网格外。**这是唯一能在不截断的情况下
   发生的成因。**

legacy / 冻结几何喂的是**整日** bar ⇒ 成因 1 **不可能**发生 ⇒ 它的暴露只剩成因 2。
成因 2 实测为空：**全量普查**（`tmp/context/f1_residual_census_full.py`，网格直接取自
**冻结面板自身的 (date, symbol) 对**、无抽样）——**995/995 只、1,159,263 对**
（**恰等于 `rows_frozen`**，即普查网格与冻结面板逐格重合）、**残桶 0 个**，缓存 1min bar
在每个 session 内网格连续。两个成因都排除 ⇒ 过滤器在 legacy 路径上是 **no-op** ⇒
**冻结 exec 基线与 D1 冻结面板一格未动**。两方独立测得同一结果（实现方本脚本 / 评审自写
脚本），数字逐项一致。

> ⚠️ **早期版本的普查射程被夸大过，记录在案**：初版 `f1_residual_probe.py` 的 `LIMIT=600`
> 是在**全部 5,782 个缓存 symbol 目录**上随机抽样，与 995 只评估 universe 的交集只有
> **102 只（10.3%）**，却被写成"评估窗内 591 只"——读者会读成"995 里的 591"。**这是承重
> 前提（legacy 路径 no-op 的论据），射程写错等于论据落空**，故改为上面的全量普查。
>
> ⚠️ **普查只量成因 2**（它对输入不施加任何 cutoff）。若把它读成"残桶总数为 0"，就会和
> 729,029 直接矛盾——后者是成因 1，在一个该普查根本没看的几何里。**任何时候证据和已知
> 事实打架，先怀疑证据在量的是别的东西。**
>
> **panels 腿本身已 subsume 这个普查**（995 只 × 1,191 个非 warmup 日全部卡在 1e-12）；
> 普查的**独特价值是它把另外 19 个 warmup 日也覆盖了**——那 19 天恰是 panels 腿唯一不施加
> 1e-12 的区域。

**修法**：`factors/compute/minute/amp_marginal_anomaly_vol.py::_complete_grid_bars`，
判据 `bar_end == bar_end.dt.ceil(freq)`（emit 的 bar_end 与桶键相等 ⟺ 窗口闭合，精确），
在 compute 里 resample 之后立刻应用。**`resample_intraday_bars` 未改**（通用 data-layer
原语，残桶行为由 `tests/test_intraday_aggregate.py` 直接覆盖；"完整桶"是**本因子定义**的
性质）；**也没有在 materializer 打补丁**（那会把因子定义散到调用方，正是本缺陷的形状）。

**逐格实证**（`tmp/context/f1_ab_geometry.py`，同一批真实缓存 bar，000008.SZ × 61 日，
cache-only）：

| | 整日几何（legacy） | 预截几何（materializer） |
|---|---|---|
| 修复前 | `0.004156551477727235` | `0.0041410493817719074` |
| 修复后 | `0.004156551477727235` | `0.004156551477727235` |

修复前 **51/61 日不同、max\|diff\| 1.42e-04**（值量级 ~4e-3，**有方向**）；修复后
**0/61 不同**，两种几何收敛到**冻结那一侧**。

**修复后真实对账（cache-only，`stk_mins_live_calls=0`，`covered=995/996`）**：

- **panels rc=0**：`frozen=1159263 new=1205160 equal=1822 within_tol=1140169
  warmup=17272 float_tail=0 threshold_flip=0 flip_contamination=0
  nan_footprint=45897 unclassified=0 max_rel_diff=9.030e-01`；
  warmup by direction `{nan_to_finite: 9109, finite_to_finite: 8163}`。
- **anchors rc=0**：`5 rows, ok=4 warmup=1 failed=0`；此前 FAIL 的 3 行现在**精确
  `rel=0.00e+00`**，剩下 1 行是登记的 `warmup_left_extension`。
- **reports rc=0**：6 份 artifact，差异**全部落在已登记类内**
  （`registered_addition` / `warmup_aggregate_effect` / `registered_sanity_stem_rename`
  / `book_view_effect`），**无一处 `spec.*` 值变化**——实测 frozen 18 个 `spec.*` 叶子、
  new 24 个，两侧都存在的 **18 个值全同**（含 `spec.version` 与 `spec.description`）、
  0 处删除；**6 处差异是纯新增**（`spec.adjustment` / `spec.lookback_depth` /
  `spec.overnight_boundary` / `spec.requires[0..2]`），逐条 `_is_registered_addition
  == True`，属 D1/D5b 契约的**预登记新增，非本 PR 引入**。

**不 bump `spec.version`、不加 `FactorCorrection`（已裁定）**：该字段的语义是
「**已发布**的值被取代」。#103（jump）是**已发布 artifact 本身就脏**（旧 runner 没做
截断）⇒ 必须承载更正；F1 相反，**已发布/冻结的那一侧是对的**（整日几何 = 定义忠实的
路径），缺陷是重构过程中新引擎引入、**从未发布**，修复是让新引擎**回到已发布值**。
声明一次没发生的 supersession 会往每份 artifact 里写一句假话。连带：**不动
`spec.description`** ⇒ reports 腿**不出 `spec.*` 值变化**（仅上述 6 处预登记新增）⇒
**不需要 jump 式注册路由**。

**但确实被污染过的东西要点名**（免得读成"从来没出过错"）：C5 全量跑写出的 live
`artifacts/reports/factor_eval_amp_marginal_anomaly_vol_20_exec_*` **是脏的**，由缺陷
引擎产出，已被本次重跑覆盖。**冻结 exec 基线与十一因子 v0.9 表不受影响。** 缺陷的足迹
**不小**——它触到了几乎每一个 emitted cell；秩 IC 逐日封顶会让聚合看起来动得很小，那是
**稀释不是无害**。

**verdict 重述（真跑出来的，非推定）**：`Reject / Reject` **未变**，三轴标签全未变。
IC −0.042601 → −0.042341、ICIR −0.446484 → −0.443614、N_eff 1116.017 → 1115.866、
NW-t −16.1546 → −16.1323、periods 1199 → 1209；增量 ICIR：`_bookclose`
**逐位复现冻结的 −0.311985**，decision 书 −0.304787（intended book-view change）。
⚠️ **这些微小位移不是 F1 修复造成的**——修复是把 served 值搬回冻结那一侧；位移是已登记的
`warmup_left_extension` 聚合效应（评估日 1199→1209）。

**判据"放行"了什么：09:30 集合竞价 bar —— 已知，且刻意不改**。判据判的是**网格边界那一
分钟在不在场**，不是成分数。`ceil(09:30, 5min) = 09:30` ⇒ session 首分钟**自成一桶、恰
1 个成分**、收在网格上 ⇒ **保留**。即**每个交易日都有一根 1 分钟 bar 被当 5min bar 池化**
（实测 600000.SH 2023-06：20/20 天各恰有一个被保留的 <5 成分桶，且**全是 09:30**；另一条
独立路径的交叉印证：全量普查在前 200 只上数到 **233,225** 个 (symbol, day) 对，与评审在同
样 200 只上数到的 <5 成分"完整"桶数**同为 233,225** ⇒ 每 (symbol, 交易日) 恰一个）。

这与上面"部分桶不该被池化"的理由**确实张力**，但**必须原样保留**：两种几何都保留它、
**冻结基线也保留它** ⇒ 丢掉它会**移动已发布值**，那是**另一个 correctness 问题、要另开 PR
带自己的证据**，不能夹带进本次修复。**后人不要当 bug 顺手"修"了。**

**本过滤器的射程（写进它自己的 docstring）**：等值判据只在 `freq` 网格与 A 股 session 对齐
时有意义。实测 `{1, 5, 15, 30}min` **一格不丢**；**`60min` 每天丢一桶**——`ceil(11:30,
60min) = 12:00`，故 11:01–11:30 收在网格外，**整个上午收盘半小时会被每日静默丢弃**
（实测 600000.SH 2023-06：60min coarse 100 → kept 80，丢的 `bar_end` 全是 `11:30`）。
今日仅为**潜在**：`freq` 默认取模块常量 `AMP_ANOMALY_FREQ`（`"5min"`），两个调用点都用默认。

**缺陷幅度（比初版呈现的更重）**：初版 PR 引用的例子（000008.SZ，61 日窗口，max\|diff\|
1.42e-04）偏小。在**完整评估窗口**上重测 `600000.SH`（1,213 个日期，cache-only）：修复前
**567/1,213 日不同**、max\|diff\| **4.14e-04**、**max rel `1.456776e-01`**；修复后
**0/1,213 日不同**、max rel **0.0**。评审独立复测（另写脚本、用 `git show main:` 把修复
前后两个模块载进同一进程喂逐字节相同输入）得 max rel **1.457e-01**、且**整日几何 5/5
逐位相等**——"已发布值不移动"由此**直接证得，不靠推理**。

### 七之七、C5 全量对账（11 因子 × 三模式）暴露的五类失败 —— 具名类扩展登记（2026-07-30）

C5 全量跑（`artifacts/logs/factor_eval_reconcile_*.log`，2026-07-28 那批）暴露五类失败
F1–F5。F1 是**真引擎缺陷**、已由独立 correctness PR 消除（见 §七之六，**永不作为具名类**）；
F5 是 harness/runner 的 artifact 销毁缺陷（修码，见本节末）。**只有 F2/F3/F4 是判据缺口**，
本节登记它们的边界与越界反例。

**方法**：先把 11 个因子的 served 面板一次性 dump 下来（`tmp/context/cc_c5_audit/`，
cache-only、`live_calls=0`），再离线对**同一批 served 值**跑改动前后的分类器——同一份输入、
两套判据，差异才可归因到判据本身而不是两次引擎运行。

#### F2 —— `threshold_flip_contamination` 的 bars-only 臂（新增）

与 §七之五裁定 2 的截面臂**同一物理事件**：600623.SH 2023-06-15 13:07 一根
`volume=13300.0` 整数 bar 压在 same-slot μ+σ 阈值上，pandas rolling 浮点累加**路径依赖
加载起点**（thr 差 ~4e-13）⇒ `vol > thr` 翻转。bars-only 因子**没有逐日 OLS 去传播它**，
所以只有**直接受影响的那只 symbol** 能动；窗口内任何**别的** symbol 仍然掉进后面的通用尾部，
量级不够就照样 unclassified 失败。

| 参数 | 值 | 实测 |
|---|---|---|
| 窗口 | [2023-06-01, 2023-07-14]（**与截面臂同窗，未改**） | 实际落点 2023-06-15..07-14 |
| symbol | 仅 600623.SH（**与截面臂同 symbol，未改**） | 两因子的 20 格全在这一只 |
| 判据 | **相对**（截面臂是绝对） | 见下 |
| `peak_interval_kurtosis_20` | rel ≤ **5e-3** | 实测 max rel **2.904e-3**（max abs 1.07e-2） |
| `valley_relative_vwap_20` | rel ≤ **1e-5** | 实测 max rel **1.530e-6**（恒定绝对偏移） |
| cell 数 | ≤ **25**/因子 | 实测各 **20** 格 |

**为什么 bars-only 臂用相对判据而截面臂用绝对**：这两个因子的值尺度互不相干——kurtosis 的
20 格 |diff| 到 **1.07e-2**（若沿用截面臂 2e-04 的绝对界会全数越界），relative_vwap 的
|diff| 只有 1.53e-6。在其中一个上标定的绝对界对另一个没有意义，故按因子登记**相对**界。
**未登记的因子拿不到这个类**（jump 在同窗同量级仍 unclassified，有反向测试）。

#### F3 —— anchor 截断的两个新表象

① **`warmup_left_extension` 方向集新增 `frozen_finite_new_nan`**，**仅** valid-day pooled
因子、**仅**早区窗内。机制：这些因子只在有效日发行、且需要 `min_valid_days` 个有效日，
anchor 边缘两种几何**积累到的有效日个数**可以不同（实证 688276.SH 2021-08-20：旧锚 10 个
有效日 → 出值，饱和加载 8 个 → NaN）。bounded 因子没有这个计数闸门，其 finite→NaN
**仍然**是 unclassified（反向测试锁定）；pooled 因子在早区窗**之外**的 finite→NaN 同样仍失败。
实测：ridge/valley_ridge 各 7 格、peak_ridge 11 格，全在 2021-07-28..08-23（早区窗内）。

② **新具名类 `warmup_sparse_valid_day_tail`**：同一 anchor 截断在**稀疏发行**的 valid-day
pooled 因子上的长尾——有效日少的票把早区差异**带出早区窗**，落在 2021 年 11 月上旬一小簇。

| 参数 | 值 | 实测 |
|---|---|---|
| 因子形态 | 仅 valid-day pooled | bounded 因子拿不到（反向测试） |
| 窗口 | [2021-11-01, **2021-11-12**] | ridge / valley_ridge 落点即此 |
| symbol 白名单 | 000034.SZ / 000402.SZ / 000999.SZ / 002375.SZ / 002653.SZ / 688183.SH | 两因子各 5 只，并集 6 只 |
| 幅度 | **不设界**（与 warmup 同理由：两种加载几何在该处本就合法地不同） | ridge 最大 rel **1.96**（符号翻转） |
| cell 数 | ≤ **20**/因子 | 实测各 **13** 格 |

**机制的可证伪印证**：该类涉及的 (因子, symbol) 对**全部 18/18** 落在该因子自身发行密度的
中位数**以下**（窗口 [2021-07-01, 2021-11-30]，中位 40–41 个发行日）。更强的一条：
600906.SH 在 peak_ridge 上发行 29 日（低于中位）而在 ridge 上 42 日（**高于**中位）——
它也**只**出现在 peak_ridge 的簇里。稀疏性是逐 (因子, symbol) 的，受影响集合随之而变。
对照组 `volume_peak_count_20`（非稀疏 pooled，发行 90% 的日子）：同样这 9 只票**一格不差**。

#### F4 —— 绝对 float-dust 谓词前移到所有区域分支之前

`|diff| ≤ 1e-12`（§七之五裁定 3 的绝对臂）原先排在 `_in_warmup` **之后**，于是恰好落在
warmup 区的几格机器精度尘埃被计成 warmup cell，其**按月计数**把 pooled 非递增闸门顶翻——
实测 `intraday_amp_cut_10` 月计数 `{07: 8198, 08: 11, 09: 3, 10: 9}`，3→9 判非单调，而
8/9/10 月那 23 格**全部** `abs ≤ 1e-12`，与 2021-11 至 2026-06 每月都有的稳态尾部同一总体。

裁定：**按机制识别，不按位置**——绝对臂前移到梯子最前（紧跟 1e-12 相对容差之后）。
它同时先于截面污染窗口，这是有意的：实测 `intraday_amp_cut_10` 那 17 格"污染"**全部**是尘埃，
现在如实归入 float 尾。相对臂（rel ≤ 5e-12）**仍在原位**（区域分支之后），未动。

**回归实测（同一批 served 值，改动前 → 改动后）**：

| 因子 | warmup | 月计数 | float_tail（cap） | contamination | ok |
|---|---|---|---|---|---|
| `intraday_amp_cut_10` | 8221 → **8198** | `{07:8198, 08:11, 09:3, 10:9}` → **`{07:8198}`** | 758 → **798**（1000） | 17 → **0** | False → **True** |
| `valley_price_quantile_20` | 24568 → **24531** | `{07:902, 08:19980, 09:3667, 10:19}` → `{07:902, 08:19980, 09:3649}` | 883 → **925**（1000） | 19177 → 19172 | True → True |
| `jump_amount_corr_20` | 17289（不变） | — | **101 → 101**（cap 101，**恰在界上、未被顶破**） | 0 → 0 | True → True |
| 其余 8 个因子 | 不变 | 不变 | 0 → 0 | 不变 | 不变 |

尘埃只在两个因子身上落进过区域分支；两者改动后都**远在** float cap 之内。**恰在 cap 上的
jump 一格未增**（它的 101 格全在 warmup 区外）。

#### F5 —— runner 的 artifact 销毁（修码，不是具名类）

`qt/factor_eval_runner.py::_apply_bookclose_suffix` 原先**先**用共享 stem 写
`{stem}_exec_with_book.*`、**再** `os.replace` 移到 `_bookclose` ⇒ 按 decision→close 顺序跑完，
decision 的 with-book 三件套被**覆盖后移走**，全 11 因子的 `exec_with_book.*` 在 C5 全量跑里
全部消失，reconcile reports 腿死在裸 `FileNotFoundError`。修法：后缀**传进写出层**
（`run_exec_basis_evaluation(..., with_book_suffix=)`，默认 `""` ⇒ 既有调用方逐字节不变），
**不再事后移动**；reconcile reports 模式增加前置检查，缺文件时给出"先跑
`run-factor-eval --book-mode decision`"的可读错误并列出缺了哪几个。半写的 `_bookclose`
配对（只有 json 或只有 md）同样是可读错误——那说明 close 模式跑到一半死了。

#### 本节未覆盖、需 lead 另行裁定的一处

`peak_ridge_amount_ratio_20` 的 panels 腿**在 C5 全量跑里就是 FAIL（unclassified=30）**，
而交接文档 §2 的结果总表把它记成 panels=0（通过）——**文档记错，日志为准**
（`artifacts/logs/factor_eval_reconcile_peak_ridge_amount_ratio_20.log`，
`unclassified=30 ... ok=False`）。它属于 F3 同族的**第三个**因子：11 格 ffnn（已由 ① 吸收）
+ 19 格 11 月簇，其中 13 格落在 ② 的窗口/白名单内、**剩 6 格在界外**（3 只未登记的
symbol 共 5 格 + 688183.SH 在 2021-11-15，晚窗口末日一个交易日）。按 ② 现行参数它仍 FAIL。
**未自行放宽**——见 C5 审计报告与交接记录。
