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
