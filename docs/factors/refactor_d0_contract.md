# 因子层重构 D0 纸面契约：分类学预指派 + 性质清单双向编目

> **状态**：D0 交付物（设计 v3.2 §九 D0 行）。纸面契约，不含任何引擎/materializer/store 实现。
> **上游**：`tmp/design/factor_refactor_design_v3.md`（v3.2 定稿，本文引用其 §1.3 / §3.4 / §五 /
> §九）；修订来龙去脉见 `tmp/design/review_round3_v32_revision_list.md`（R2/R5/R6/R7/R8/R12/R15）。
> **author-once 纪律**：§1.3 policy 表与两个分类学的值域**单一来源是代码常量**
> `data/availability_policy.py`。本文只引用常量名与语义，不复述数值表——重复即是 #76/#78/#82
> 那一类"行为变了而措辞没变"缺陷的温床。
> **基线**：`main` @ `1f9d85e`。文中全部 `file:line` 与 `test 文件::测试名` 均在该基线上实读核对。

---

## 一、分类学语义（值域引用 `data/availability_policy.py`）

两个分类学是将来 `FactorSpec` 的必填声明字段（D1 落地），各三值、封闭。值域枚举：

- `Adjustment`：`NONE` / `RETURNS_INVARIANT` / `PRICE_LEVEL`
- `OvernightBoundary`：`NONE` / `CROSSED_DISCLOSED` / `MASKED`

两轴管**不同的事**，不许混谈（kill 档案 A2-F01 的裁决边界）：

| 轴 | 管什么 | 不管什么 |
|---|---|---|
| `adjustment` | 因子存储值对**复权锚**的依赖 → 决定 store 数据指纹怎么派生（§3.4）：锚一动，缓存值是否作废 | 不管除权日当天的跨隔夜基准断裂 |
| `overnight_boundary` | decision(d) 视图下 `adj_factor ≤ d−1` 造成的**除权日基准断裂**是否进入因子值（§1.3 注） | 不管缓存陈旧性；也不管 volume 通道的拆股污染（那是钉死的因子定义，见 §二注 3） |

### 1.1 `adjustment` 三值语义与可测试性

（语义组合引用 v3.2 §3.4；值 = `Adjustment` 枚举成员。）

- **`NONE`** — 因子完全不碰价格通道（volume / amount / 当日发布比值）。锚扰动对它天然不可见。
  **可测性**：锚扰动不变测试同样适用且应当 by construction 通过；none 与 returns_invariant 的
  判据是**输入清单是否含价格通道**（D1 起可由 `requires` 声明静态检查，不必逐因子跑扰动）。
- **`RETURNS_INVARIANT`** — 碰价格，但只以**锚在比值中相消**的方式碰（同日比值 / qfq 比值）。
  **可测性（§3.4 明文）**：必须通过「扰动锚 → 值不变」性质测试。
- **`PRICE_LEVEL`** — 依赖复权后价格**水平**；store 指纹必须另掺 per-symbol `adj_factor`
  事件表 hash（含锚值）。**可测性**：收官集零成员**不豁免**——机制用 fixture 因子测试
  （kill 档案 A5-F02：砍掉此分支 = 声明 price_level 的因子静默拿错指纹，恰是 §六.17 要封的洞）。

数据指纹派生规则（§3.4）：`NONE`/`RETURNS_INVARIANT` → date-grid coverage + schema-version 维；
`PRICE_LEVEL` → 另掺 adj_factor 事件表 hash。**声明错误的失效形态**：把 price_level 错标
returns_invariant → 除权后陈旧值当新值复用（§六.17 qfq 锚陷阱）——这正是声明必须可测试的原因。

### 1.2 `overnight_boundary` 三值语义与可测试性

（语义组合引用 v3.2 §1.3 除权日边界注 1–6；值 = `OvernightBoundary` 枚举成员。）

- **`NONE`** — 因子值中没有任何 raw 价格比较跨越隔夜边界。
  **可测性（注 1）**：「扰动 d−1 收盘/af → 当日值不变」。**本编目的口径澄清（判读，D2 落测试
  时须按此实现）**：该扰动应实现为**历史基准统一重标度**——对 symbol 的全部 `< d` raw 价格乘
  常数 λ（模拟 d 为除权日）→ 日 d 因子值不变。不能判读为「扰动 d−1 的任意单个数」：任何多日
  回看因子（如 PR-K 的 20 日基线）都依赖 d−1 的数据，逐点扰动全军覆没，该轴便失去判别力；
  而统一重标度恰好检验「基准断裂是否进入值」——PR-K docstring 的自证
  （`data/clean/intraday_ridge_return.py:39-42`："no return ever straddles an ex-date
  boundary"）正是对这个不变性的主张。
- **`CROSSED_DISCLOSED`**（R6 新增第三态）— 钉死定义**故意**跨隔夜、ex-date 值按定义保留、
  偏差**已测量披露**。三要件缺一即不构成本态。对这样的因子声明 `NONE` → 性质测试必红；声明
  `MASKED` → 删掉定义故意保留的值 = 定义变更（撞 §〇 总原则/红线 #8）。
  **可测性**：正向对照测试（扰动跨界输入 → 值**必须动**）+ 披露测量存在性。既有实例：
  `tests/test_valley_price_quantile_factor.py::test_perturbing_the_t_minus_1_close_does_move_the_factor`。
- **`MASKED`** — materializer 统一在 ex-date 上把该 (d, symbol) 置 NaN 走既有缺失路径。
  **披露必须含分布**（按月 / 按因子 / 每期截面缩减量）+ mean_ci 压紧口径 caveat（注 3）。
  **收官 14 因子零成员**；对既有因子施加 masked = 定义变更单独立项，materializer 不得静默施加。
  **可测性**：机制测试（fixture 因子 + 屏蔽分布披露断言），与 price_level 同一"零成员不豁免"纪律。

**ex-date 判定结构化（注 4）**：判定式 `af(d) ≠ af(d−1)` 由专职模块执行，向 decision-view
materializer **只暴露布尔**——af(d) 数值不进 decision-view 帧（合法性论证见注 5：「今天是除权
日」这一事实现实中开盘前公开，af 序列比较只是对该公开事实的代理重构）。

---

## 二、收官 14 因子预指派表（纸面预注册；D1 落成 spec 字段）

判据（§一）：`adjustment` 看碰不碰价格通道、以何种方式碰；`overnight_boundary` 看 raw 价格
比较是否跨隔夜。证据列为**实读**的 docstring/代码自证行（`main` @ `1f9d85e`）。

| # | 因子 | adjustment | overnight_boundary | 证据（file:line） |
|---|---|---|---|---|
| 1 | `jump_amount_corr` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_aggregate.py:420`（jump = within-(symbol, day) amplitude z-score，amplitude=(high−low)/open 同日比值）；`:421-422`（配对严格限同 session 相邻分钟，跨午休/收盘即无对）；相关额为 amount（基准无关） |
| 2 | `minute_ideal_amplitude` | `RETURNS_INVARIANT` | **`CROSSED_DISCLOSED`（候选）⚠️ 与设计锚点分歧，见注 1** | `data/clean/intraday_amplitude.py:24-27`（"Rank the pooled minutes by RAW minute close (unadjusted — amplitude is a ratio so it needs no adjustment, and the report ranks on the raw minute price)"——amp 是同日比值，但**排序键是跨 10 日池化的 raw 价格水平**） |
| 3 | `amp_marginal_anomaly_vol` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_amp_anomaly.py:25-26`（Δamp 与 r "are BOTH WITHIN-DAY lagged … the overnight gap never contaminates a pair"） |
| 4 | `volume_peak_count` | `NONE` | `NONE` | `data/clean/intraday_volume_prv.py:52-55`（纯 volume 通道；"Raw minute volume (cached as-is) has magnitude jumps across split days … we disclose it and do NOT correct it"——见注 3） |
| 5 | `intraday_amp_cut` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_amp_cut.py:27-28`（r "WITHIN-DAY lagged … no overnight gap; PR-E precedent"；amp = high/low−1 同日比值） |
| 6 | `peak_interval_kurtosis` | `NONE` | `NONE` | `data/clean/intraday_peak_interval.py:8-9`（峰识别 verbatim 复用 PR-F 的 volume 机器）+ `:20-21`（间隔 = 交易分钟位置差，无价格输入）+ `:51-53`（拆股 σ 污染披露同 PR-F——见注 3） |
| 7 | `valley_relative_vwap` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_valley_vwap.py:33-35`（"Both legs are sums over the SAME trading day, and a split/dividend adjustment factor is constant within a day, so it cancels exactly in the ratio"） |
| 8 | `valley_ridge_vwap_ratio` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_valley_ridge_vwap.py:42-45`（同日双腿之比，同一相消论证） |
| 9 | `ridge_minute_return` | `RETURNS_INVARIANT` | `NONE` | `data/clean/intraday_ridge_return.py:39-42`（"a split/dividend adjustment factor is constant WITHIN a day, so it cancels exactly in `close_t / close_{t-1}`; … no return ever straddles an ex-date boundary"） |
| 10 | `valley_price_quantile` | `RETURNS_INVARIANT` | **`CROSSED_DISCLOSED`** | 跨界+测量披露：`data/clean/intraday_valley_quantile.py:56-72`（prev_close 跨隔夜、ex-date 单侧拉高 hi、实测 0.73% 真 ex-date / ~0.21% 真 ex-date 且 range-distorted、"DISCLOSED here rather than silently corrected"）+ surface 披露 `factors/compute/intraday_derived.py:1124-1134`；returns_invariant 证据：`intraday_valley_quantile.py:49-55` + `:403-404`（rev20 用 qfq 收盘**因为**比值跨 20 日，锚在比值中相消；分钟腿全同日） |
| 11 | `peak_ridge_amount_ratio` | `NONE` | `NONE` | `data/clean/intraday_amount_ratio.py:53-57`（"``amount`` is traded VALUE in RMB, which no split or dividend adjustment factor rescales … there is nothing to cancel"——零价格含量，两条免责一次占全） |
| 12 | `value_ep` | `NONE` | `NONE` | `factors/compute/candidates.py:44`（1/pe，daily_basic 当日发布比值）+ `:296-301`（ValueFactor 只 surface 富化列，无价格通道、无 qfq、无时序逻辑） |
| 13 | `value_bp` | `NONE` | `NONE` | `factors/compute/candidates.py:45` + `:296-301`（同上，1/pb） |
| 14 | `volatility_20` | `RETURNS_INVARIANT` | `NONE` | `factors/compute/candidates.py:144-151`（per-symbol `pct_change` 后滚动 std——输入是 pipeline 的 qfq close 面板，锚在每个收益比值中相消；数据层锚不变性已有测试 `tests/test_adjust.py::test_front_adjust_returns_invariant_to_anchor`）；decision 视图下全部输入 ≤ d−1（`data/availability_policy.py` 的 market_daily close 行），不与 d 日盘中新基准混算 |

**统计**：`RETURNS_INVARIANT` 9 个 / `NONE` 5 个 / `PRICE_LEVEL` 0 个（零成员，fixture 测试
义务见 §一.1）；`overnight_boundary` `NONE` 12 个 / `CROSSED_DISCLOSED` 1 个 + 1 个候选 /
`MASKED` 0 个。

**注 1（⚠️ 实读与设计锚点的分歧，本表最重要的一条 judgment call）**：设计 v3.2 只点名
`valley_price_quantile` 一个 `crossed_disclosed`（R6："masked 在收官 14 因子中零成员——全部
日内复权抵消或 returns_invariant"）。实读 `data/clean/intraday_amplitude.py:20-27` 发现
`minute_ideal_amplitude` 的构造是：把 10 日窗口的全部分钟 bar 池化成一个集合，**按 raw 分钟
close（价格水平）排序**取 top/bottom 25% 的均 amp 之差。amp 本身是同日比值（基准无关），但
**排序键把不同交易日的 raw 价格水平放进同一个序**——若窗口内含除权日，除权前后的 bar 位于
不同基准，池化排序的交错即被基准断裂扭曲（与 PR-L 的 prev_close 同构：钉死定义故意用 raw、
报告 Wind 口径如此、修它 = 定义变更）。按 §一.2 的统一重标度测试，它声明 `NONE` 必红。
**故预指派 `CROSSED_DISCLOSED`，但三要件缺第三件**：跨界事实成立、按定义保留成立，
**偏差测量披露目前缺失**（无 PR-L 式的 0.73%/0.21% 测量）。**D1 把该声明落成 spec 字段前，
必须补一次 PR-L 式的 ex-date 偏差测量**（10 日池窗口 × CSI500 评估网格上，真 ex-date 落入
池窗口的天数占比 + 排序扰动的实际量级），否则该声明不完整。若主会话裁定改判 `NONE`，须同时
接受 D2 的统一重标度性质测试对它预期为红（known-red），并单独立项处理——两条路都已摆明，
本表按 R6 对 PR-L 的同构处理逻辑取前者。

**注 2（`valley_price_quantile` 的两轴为何不同值）**：它的 `adjustment` 是
`RETURNS_INVARIANT`（rev20 是 qfq 比值、锚相消；分钟腿 raw 且当日自足——**锚扰动**不动它），
而 `overnight_boundary` 是 `CROSSED_DISCLOSED`（prev_close 与当日 bar 的**基准断裂**动它）。
这正是两轴必须分开声明的实例：管缓存陈旧性的轴与管除权日断裂的轴在同一个因子上取不同值。

**注 3（volume 通道拆股污染，两个 volume 因子共有，不进两轴）**：`volume_peak_count` 与
`peak_interval_kurtosis` 的 20 日同 slot σ 基线跨拆股日会被 raw volume 的量级跳变污染
（`intraday_volume_prv.py:52-55` / `intraday_peak_interval.py:51-53` 均披露且**故意不修**，
与研报 Wind 口径对齐）。kill 档案 A2-F01 已裁决：这不是 adjustment 轴的第四形态（该轴管缓存
陈旧性，raw 分钟 volume 写后不变、无陈旧路径），也不改 overnight_boundary 的 `NONE`（该轴管
**价格**基准断裂；§一.2 的重标度测试只动价格，volume 因子不读价格、必然通过）。修它 = 定义
变更夹带，禁止。此注存在的目的：防止后来者"发现"该污染后误以为分类学对它失明。

---

## 三、性质清单双向编目（D0 核心交付）

### 3.0 编目范围与方法

- 性质清单 = v3.2 §五.1 的 11 类（原文钉死，含 R15 补入的 purity）。
- **方向一**（性质 → 既有测试）：每类性质列出 `tests/` 中现存的承载测试（`文件::测试名`，
  全部经 grep 实证存在于 `main` @ `1f9d85e`，无一编造）。重复模式以「模式 × 文件数 + 文件列表」
  紧凑记法列出。
- **方向二**（既有测试 → 性质）：把 leakage/mutation/invariance/purity 类扫描命中逐条归类；
  归不进任何类的单列 §3.3「清单外」——那是清单遗漏的信号，照实列出。
- 扫描模式：`lookahead|future|perturb|leak|not_mutated|does_not_mutate|ignores_future|
  isolation|per_symbol|cross_symbol|cutoff|1450|visible|nan|sign|invariant`。

### 3.1 方向一：11 类性质 → 既有测试

#### P1 未来扰动不变（扰动 date > d 的数据 → 因子值逐 bit 不变；红线 #2）

因子层（12）：

- `tests/test_factors_momentum.py::test_momentum_has_no_lookahead`
- `tests/test_candidate_factors.py::test_reversal_ignores_future_bars`
- `tests/test_candidate_factors.py::test_overnight_mom_ignores_future_bars`
- `tests/test_jump_amount_corr_factor.py::test_pit_perturbing_future_bars_does_not_change_factor_at_d`
- `test_future_day_does_not_change_earlier_factor` × 5 文件：`test_volume_peak_count_factor.py`、
  `test_peak_interval_kurtosis_factor.py`、`test_ridge_minute_return_factor.py`、
  `test_valley_relative_vwap_factor.py`、`test_valley_ridge_vwap_ratio_factor.py`
- `tests/test_peak_ridge_amount_ratio_factor.py::test_future_day_does_not_change_the_earlier_factor`
- `tests/test_intraday_amp_cut_factor.py::test_amp_cut_future_day_does_not_change_earlier_factor`
- `tests/test_i5c_mmp_minute_factor.py::test_i5c_future_bars_do_not_change_earlier_mmp`

邻层（不随因子层重写迁移，但同性质）：

- `tests/test_pit_financials.py::test_asof_no_future_leak_when_future_report_changes`（data 层 ann_date as-of）
- `tests/test_index_universe.py::test_members_no_lookahead_into_future_snapshot`（universe 层）
- `tests/test_i5d_mmp_quintile.py::test_future_exit_bar_cannot_change_grouping`（runtime/分组层）

teeth（本性质自带的 mutation 证据测试）：

- `tests/test_peak_ridge_amount_ratio_factor.py::test_future_day_invariance_has_teeth_under_the_forward_window_defect`

#### P2 cutoff 可见性（`available_time ≤ 14:50` 逐 bar 过滤；扰动 post-cutoff bar → 值不变）

- `tests/test_intraday_aggregate.py::test_post_cutoff_bars_do_not_leak`（任务书锚点）
- `tests/test_intraday_aggregate.py::test_all_bars_after_cutoff_returns_empty`
- `tests/test_intraday_aggregate.py::test_ret_value_open_to_last_visible`
- `test_perturbing_post_1450_bars_does_not_change_factor` × 7 文件（精确同名）：
  `test_amp_marginal_anomaly_vol_factor.py`、`test_minute_ideal_amplitude_factor.py`、
  `test_peak_interval_kurtosis_factor.py`、`test_ridge_minute_return_factor.py`、
  `test_valley_relative_vwap_factor.py`、`test_valley_ridge_vwap_ratio_factor.py`、
  `test_volume_peak_count_factor.py`；变体 2：
  `test_intraday_amp_cut_factor.py::test_amp_cut_perturbing_post_1450_bars_does_not_change_factor`、
  `test_peak_ridge_amount_ratio_factor.py::test_perturbing_post_1450_bars_does_not_change_the_factor`
- `tests/test_valley_price_quantile_factor.py::test_bars_after_the_cutoff_are_invisible_and_the_test_has_teeth`
- `tests/test_valley_price_quantile_factor.py::test_prev_close_comes_from_the_visible_window_not_the_real_daily_close`
- `test_pit_truncation_excludes_post_1450_bars` × 2：`test_amp_marginal_anomaly_vol_factor.py`、
  `test_minute_ideal_amplitude_factor.py`
- `tests/test_i5a_event_backtest.py::test_perturb_post_decision_bars_leave_decision_feature_unchanged`（runtime 层）
- `tests/test_i5c_mmp_minute_factor.py::test_i5c_post_cutoff_bars_do_not_change_daily_mmp`
- `tests/test_i5d_mmp_quintile.py::test_post_cutoff_bar_cannot_change_grouping`（runtime/分组层）

teeth：

- `test_leakage_test_has_teeth_removing_the_cutoff_changes_the_value` × 2：
  `test_ridge_minute_return_factor.py`、`test_valley_ridge_vwap_ratio_factor.py`
- `tests/test_peak_ridge_amount_ratio_factor.py::test_pit_invariance_has_teeth_under_the_no_cutoff_defect`

#### P3 跨 symbol 隔离（红线 #3 的一半：一个 symbol 的数据绝不进另一个 symbol 的值）

- `test_per_symbol_isolation` × 9 文件（精确同名）：`test_amp_marginal_anomaly_vol_factor.py`、
  `test_jump_amount_corr_factor.py`、`test_minute_ideal_amplitude_factor.py`、
  `test_peak_interval_kurtosis_factor.py`、`test_peak_ridge_amount_ratio_factor.py`、
  `test_ridge_minute_return_factor.py`、`test_valley_relative_vwap_factor.py`、
  `test_valley_ridge_vwap_ratio_factor.py`、`test_volume_peak_count_factor.py`
- `tests/test_valley_price_quantile_factor.py::test_cross_symbol_isolation` +
  `::test_prev_close_does_not_cross_symbols`
- `tests/test_intraday_amp_cut_factor.py::test_amp_cut_per_symbol_isolation_at_cut_level`
- `tests/test_i5c_mmp_minute_factor.py::test_i5c_multi_symbol_isolation`
- 日频：`tests/test_factors_momentum.py::test_momentum_computed_per_symbol`、
  `tests/test_candidate_factors.py::test_volatility_is_per_symbol_no_cross_leakage`、
  `tests/test_candidate_factors.py::test_overnight_mom_prev_close_never_crosses_symbols`
- 数据层（邻层）：`tests/test_adjust.py::test_front_adjust_is_per_symbol`

teeth：

- `tests/test_peak_ridge_amount_ratio_factor.py::test_isolation_has_teeth_under_the_mislabeled_frames_defect`

#### P4 前导 NaN 窗口（warm-up 期诚实 NaN，不许部分窗静默改语义）

- `tests/test_factors_momentum.py::test_momentum_window_not_enough_returns_nan`
- `test_min_valid_days_floor_returns_nan_until_enough_valid_days` × 5 文件：
  `test_peak_interval_kurtosis_factor.py`、`test_peak_ridge_amount_ratio_factor.py`、
  `test_ridge_minute_return_factor.py`、`test_valley_relative_vwap_factor.py`、
  `test_valley_ridge_vwap_ratio_factor.py`
- `tests/test_volume_peak_count_factor.py::test_valid_day_floor_returns_nan_until_enough_valid_days`
- `tests/test_volume_peak_count_factor.py::test_baseline_insufficient_yields_no_value` +
  `::test_current_day_volume_not_in_its_own_baseline`（strictly-prior 基线边界）
- `tests/test_valley_price_quantile_factor.py::test_first_day_of_the_window_has_no_prev_close_and_is_invalid` +
  `::test_below_min_valid_days_is_nan`
- `tests/test_ridge_minute_return_factor.py::test_first_visible_bar_of_the_day_never_carries_a_return`
  （within-day 首 bar 无 lag 的窗口边界）

#### P5 NaN 策略（诚实缺失门：数据不足 → NaN 并披露，绝不造数）

- 因子门（min_* 家族）：`tests/test_jump_amount_corr_factor.py::test_min_pairs_gate_returns_nan`、
  `tests/test_minute_ideal_amplitude_factor.py::test_min_minutes_gate_default_1150_returns_nan_on_small_pool`、
  `tests/test_amp_marginal_anomaly_vol_factor.py::test_min_pool_gate_default_460_returns_nan_on_small_pool` +
  `::test_min_selected_gate_returns_nan_when_too_few_anomalies`、
  `tests/test_peak_interval_kurtosis_factor.py::test_too_few_pooled_intervals_is_nan` +
  `::test_zero_variance_pool_is_nan`、
  `test_min_classifiable_gate_invalidates_thin_days` × 6 文件（`test_volume_peak_count_factor.py`、
  `test_peak_interval_kurtosis_factor.py`、`test_peak_ridge_amount_ratio_factor.py`、
  `test_ridge_minute_return_factor.py`、`test_valley_relative_vwap_factor.py`、
  `test_valley_ridge_vwap_ratio_factor.py`）
- 截面门：`tests/test_intraday_amp_cut_factor.py::test_amp_cut_cross_section_below_min_is_all_nan` +
  `::test_amp_cut_cross_section_degenerate_zero_variance_is_nan`、
  `tests/test_valley_price_quantile_factor.py::test_residualization_below_min_cross_section_is_all_nan` +
  `::test_residualization_degenerate_reversal_cross_section_is_nan` +
  `::test_residualization_drops_symbols_without_a_reversal_value`
- 非法输入 → NaN 不 crash：`tests/test_candidate_factors.py::test_liquidity_nonpositive_amount_is_nan_not_crash` +
  `::test_overnight_mom_nonpositive_prices_are_nan_not_inf`
- 评估层同纪律（邻层，保留资产）：`tests/test_factor_eval_standard.py::test_all_nan_and_empty_cross_sections_are_nan_not_zero` +
  `::test_degenerate_series_give_nan_cis_never_a_fabricated_number` +
  `::test_newey_west_t_degenerate_inputs_are_nan_not_zero`

#### P6 符号约定（`expected_ic_sign` 预注册、方向对齐判定；红线 #1 的评估侧）

- `tests/test_factor_eval_contract.py::test_expected_ic_sign_must_be_plus_or_minus_one` +
  `::test_expected_ic_sign_accepts_both_directions` +
  `::test_independently_confirmed_signs_match_the_project_evidence` +
  `::test_reversal_sign_is_the_exact_negation_of_momentum` +
  `::test_negative_sign_factor_is_judged_in_its_own_direction` +
  `::test_the_aligned_monotonicity_ci_swaps_min_and_max_at_sign_minus_one`
- `tests/test_factor_eval_standard.py::test_hypothesis_win_rate_is_relative_to_the_expected_sign`
- `tests/test_eval_valley_price_quantile_runner.py::test_spec_sign_is_positive_so_aligned_spread_is_not_mis_signed`
- `tests/test_valley_price_quantile_factor.py::test_spec_declares_the_pre_registered_sign_and_the_pinned_deviations`

#### P7 视图隔离（view 进身份；decision/close 信息集不混）——部分前身 + 核心 NET-NEW

现状：view 尚不存在为一等身份，无直接测试。**前身测试**（守住了同一语义边界的局部）：

- `tests/test_i5a_event_backtest.py::test_perturb_daily_close_does_not_change_intraday_returns`
  （decision/exec 路径不读日收盘——视图隔离在 runtime 的雏形）
- `tests/test_i5b_execution_feasibility.py::test_i5b_daily_close_perturbation_does_not_change_feasibility`
- `tests/test_intraday_execution.py::test_signal_cutoff_and_execution_timestamp_are_separate`
- `tests/test_exec_forward_returns.py::test_exec_basis_artifact_is_reused_and_rekeyed_on_a_parameter_change`
  （key 完整性——「视图进 key」的先例机制）
- `tests/test_exec_basis_eval.py::test_exec_basis_eval_shares_one_artifact_across_factors`

**核心机制（view 字段 / 配对合法性 / store key 含 view）为 NET-NEW** → 见 §3.4。

#### P8 单点填 ≡ 批量填（`cross_section` 逐日 gap-fill ≡ `panel` 一次 gap-fill，store 逐 bit 相等）

因子层无既有测试（store 尚不存在）→ **NET-NEW**（§3.4）。邻层前身（同一不变量思想在数据层的
既有承载，供 D4 移植参照）：数据缓存的 cold/warm/partial 一致性测试族（`tests/test_data_feed.py`
等 P4-1/P4-2 缓存测试，cached==direct 逐字节；此处不逐条列——它们守的是数据层，不是因子 store）。

#### P9 adjustment 声明可测性（§3.4：returns_invariant 扰锚不变 / price_level fixture 指纹）

因子层无既有测试（声明字段尚不存在）→ **NET-NEW**（§3.4）。邻层锚（同一数学事实的既有测试，
D3 写因子级测试时的参照）：

- `tests/test_adjust.py::test_front_adjust_returns_invariant_to_anchor`（数据层：qfq 收益对锚不变）
- `tests/test_adjust.py::test_front_adjust_anchors_latest_price_unchanged` + `::test_front_adjust_removes_ex_dividend_gap`
- `tests/test_exec_forward_returns.py::test_exec_basis_adjustment_is_invariant_to_a_common_rescale`（exec 收益层）
- `tests/test_exec_vwap_basis.py::test_adjustment_does_not_leak_into_the_raw_price_limit_gate`
  （runtime 层：复权**不得**渗入 raw-vs-raw 闸门——本性质的镜像边界）

#### P10 overnight_boundary 声明可测性（§1.3 注 1/2：none 重标度不变 / crossed 正向对照 + 披露）

`crossed_disclosed` 侧已有**正向对照**（PR-L 专属，证明跨界事实存在且测试有牙）：

- `tests/test_valley_price_quantile_factor.py::test_perturbing_the_t_minus_1_close_does_move_the_factor`
- `tests/test_valley_price_quantile_factor.py::test_uniform_close_d_bump_cannot_distinguish_the_two_reversal_bases`
- `tests/test_valley_price_quantile_factor.py::test_reversal_20_ignores_day_d_close_entirely`
  （rev20 的 T−1 边界——跨界的合法信息侧）

`NONE` 侧（统一基准重标度 → 当日值不变，覆盖 12 个声明 none 的因子）**无任何既有测试** →
**NET-NEW**（§3.4）。

#### P11 purity 无副作用（compute 纯函数：不改输入、不写盘、不读全局；红线 #3）

- `test_input_bars_not_mutated` × 9 文件（精确同名，R15 所指家族）：
  `test_amp_marginal_anomaly_vol_factor.py`、`test_jump_amount_corr_factor.py`、
  `test_minute_ideal_amplitude_factor.py`、`test_peak_interval_kurtosis_factor.py`、
  `test_peak_ridge_amount_ratio_factor.py`、`test_ridge_minute_return_factor.py`、
  `test_valley_relative_vwap_factor.py`、`test_valley_ridge_vwap_ratio_factor.py`、
  `test_volume_peak_count_factor.py`
- 变体：`tests/test_intraday_amp_cut_factor.py::test_amp_cut_input_bars_not_mutated` +
  `::test_amp_cut_stats_input_not_mutated_by_combine`、
  `tests/test_valley_price_quantile_factor.py::test_inputs_are_never_mutated`
- 邻层同纪律：`tests/test_adjust.py::test_front_adjust_does_not_mutate_input`（data）、
  `tests/test_intraday_schema.py::test_normalize_does_not_mutate_input`（data）、
  `tests/test_covariates_enrich.py::test_enrich_does_not_mutate_input` +
  `tests/test_tradability_enrich.py::test_enrich_does_not_mutate_input`（enrich）、
  `tests/test_factor_eval_standard.py::test_build_does_not_mutate_the_caller_s_panels`（eval）、
  `tests/test_ic_weight_alpha.py::test_inputs_are_not_mutated`（alpha）

### 3.2 方向二：既有测试 → 性质（扫描命中归类核对）

扫描命中的 leakage/mutation/invariance/purity 类测试逐条归类结果：**全部因子层命中均已归入
P1–P6 / P11**（见 §3.1 各条），邻层命中按层标注归入或列入清单外。方向二没有发现"因子层测试
归不进 11 类"的情况；发现的清单外条目全部来自**邻层**或**互补性质**，如下节。

### 3.3 清单外（归不进 11 类的扫描命中——清单遗漏信号，照实列出）

1. **secret 泄漏守卫**（扫描词 `leak` 命中，但守的是 secret 不是未来数据）：
   `tests/test_oos_stability.py::test_render_oos_report_leaks_no_secret`、
   `tests/test_robustness_matrix.py::test_render_matrix_report_leaks_no_secret`、
   `tests/test_subset_validation.py::test_render_subset_report_leaks_no_secret`、
   `tests/test_independent_validation.py::test_render_independent_result_leaks_no_secret`、
   `tests/test_phase2_baseline.py::test_render_does_not_leak_secret_file_or_token`、
   `tests/test_phase0_pipeline.py::test_phase0_standard_analytics_no_secret_leak`、
   `tests/test_tushare_intraday_feed.py::test_no_token_leak`。
   **信号**：「no-secret」是贯穿 renderer/feed/store 的横切性质，§五.1 清单没有它——D3 的
   运行注册表 no-secret 契约（R22）与 D5 统一 runner 的报告都需要它，建议 D3/D5 验收显式
   继承该测试族的模式（per-PR secret scan 只是兜底）。
2. **边界互补测试**（性质的"另一半"：边界之外**必须**看得见未来/必须动）：
   `tests/test_analytics_factor.py::test_forward_returns_uses_future_close_per_symbol`
   （forward returns 在 alpha/eval 边界**就该**用未来收盘——不变量 #1 的补集）、
   `tests/test_i5a_event_backtest.py::test_perturb_execution_bar_changes_intraday_returns`
   （执行 bar 扰动**必须**改变收益——证明不变性测试不是恒真）。
   **信号**：清单的 11 类全是"必须不变"；「必须变」的正向对照是 mutation 证据纪律的组成部分，
   D2 重写性质测试时应成对迁移（§3.5）。
3. **alpha 层 walk-forward 性质**（同思想、不同层，不随因子层迁移）：
   `tests/test_ic_weight_alpha.py::test_perturbing_unrealized_future_does_not_change_weights`、
   `tests/test_oos_stability.py::test_perturbing_post_split_returns_leaves_train_weights_unchanged`、
   `tests/test_ic_weight_alpha.py::test_realization_cutoff_is_exact_t_plus_h`。
4. **数据层运维守卫**（扫描命中，性质属数据层契约）：
   `tests/test_data_backfill.py::test_future_start_raises_not_silent_noop`、
   `tests/test_data_quality_market.py::test_decreasing_adj_factor_does_not_bleed_across_symbols`、
   `tests/test_intraday_schema.py::test_schema_can_represent_future_derived_coarser_bars`。

### 3.4 NET-NEW 章（「既有测试」格为空的性质 → 钉归属 D 格）

| 性质 | NET-NEW 内容 | 归属 D 格（v3.2 §九） |
|---|---|---|
| **P9 adjustment 声明可测性** | 因子级「扰锚 → 值不变」测试（9 个 returns_invariant 因子）+ `PRICE_LEVEL` fixture 因子指纹测试（零成员不豁免，A5-F02）+ `NONE` 的输入清单静态检查 | **D3**（store/指纹格；R15 明文钉 D3） |
| **P10 overnight_boundary 声明可测性** | `NONE` 侧统一基准重标度测试（12 因子；口径澄清见 §一.2）+ `CROSSED_DISCLOSED` 的披露测量存在性断言（含 §二注 1 的 `minute_ideal_amplitude` 补测量）+ `MASKED` 机制 fixture | **D2**（因子迁移格；R15 明文钉 D2） |
| **P8 单点填 ≡ 批量填** | 两个冷 store 分别经 `cross_section` 逐日 / `panel` 一次 gap-fill 填成 → 内容逐 bit 相等；mutation：tail 拼接少取一行必红（§3.5 契约的可失败表述） | **D4**（v3.2 §九 D4 验收行已钉） |
| **P7 视图隔离**（核心机制） | view 进 spec / store key / artifact 身份；`require_legal_pairing` 在构造期强制（D0 已交纸面常量 `data/availability_policy.py::require_legal_pairing`）；「扰动另一视图的输入 → 本视图值不变」 | **D1**（spec/config 配对可读报错，§九 D1 验收行）+ **D3**（key 含 view）+ **D4**(配对报错测试重申) |

其余七类（P1–P6、P11）方向一均非空：迁移语义为 §五.1 的「性质测试搬迁」——在新引擎上**重写**
并重跑 mutation 证据，不是沿用旧文件（D2 起逐格执行，D5 按 R16 覆盖数非降门把关）。

### 3.5 mutation 证据纪律（引用 §五，本编目的执行前提）

每条性质在新引擎重写时，**mutation 证据必须实跑且打中所主张的那个具体性质**（§五.1 /
§六.10）：不可能失败的测试在本项目已有四个在案反例（PR-L 旗舰反泄漏测试的仿射变换盲区、
`compare_postmerge.py` 空对账、I5b fixture 的 VWAP≡close、恒真 panel≡cross_section）。本编目
中 §3.1 各 teeth 测试（`*_has_teeth_*` 家族）与 §3.3 类 2 的正向对照，就是该纪律在既有套件里
的存量形态——迁移时**成对搬**：不变性测试与它的"必须变"对照一起走。性质测试在新引擎上失败时
**修引擎不改测试**；性质本身被证明错了，单独记录并走定义变更流程（§五.1，红线 #8）。

---

## 附：本编目的统计

- 方向一（条数 = 本文实际列出的 `文件::测试名`，重复模式按文件数展开）：
  P1 = 12 因子层 + 3 邻层 + 1 teeth；P2 = 17 因子层（3 aggregate + 9 post-1450 扰动 +
  2 valley_price 可见性 + 2 truncation + 1 i5c）+ 2 runtime 层 + 3 teeth；
  P3 = 13 分钟因子层 + 3 日频 + 1 数据层 + 1 teeth；P4 = 12；P5 = 19 因子层 + 3 评估层；
  P6 = 9；P7 = 5 前身（核心 NET-NEW）；P8 = 0（NET-NEW）；P9 = 0 因子层（5 邻层锚，
  NET-NEW）；P10 = 3（仅 crossed 正向对照；none 侧 NET-NEW）；P11 = 12 因子层 + 6 邻层。
- 清单外：4 类 15 条（secret 守卫 7 / 边界互补 2 / alpha 层 3 / 数据层 3）。
- NET-NEW：4 项，归属 D2 / D3 / D4 / D1+D3+D4，均已钉格（§3.4）。
