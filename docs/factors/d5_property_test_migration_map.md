# D5 C5 腿 1 —— 性质测试搬迁映射表（旧 runner 测试 → 新套件）

> **状态**：D5 C5 准备交付物（设计 v3.2 §五腿 1「性质测试搬迁」的映射底账；C6 的 R16
> 「覆盖数非降」审计直接消费本表）。
> **口径**：旧套件 = 11 个旧 runner 的测试文件 `tests/test_eval_*_runner.py`（**实有 10 份、
> 59 条测试函数**——`peak_ridge_amount_ratio`（PR-M）**没有** runner 测试文件，编目
> §六 BUG 3 已登记，见 §五）。新套件 = 统一 runner 落地后的测试面（`test_factor_eval_runner.py`
> / `test_factor_eval_disclosures.py` / `test_factor_eval_providers.py` /
> `test_factor_eval_reconcile.py` / `test_exec_basis_eval.py` / `test_factor_service.py` /
> `test_factor_store_universe.py` / `test_minute_binding_vpq.py` /
> `test_minute_diagnostics_channel.py` / 各 `test_<factor>_factor.py` /
> `test_factor_eval_contract.py` / `test_factor_eval_standard.py` /
> `test_factor_requires_spec_v1.py` 等）。
> **基线**：`feat/factor-refactor-d5-c5-prep` @ origin/main（含 PR #105/#107）。全部 node-ID
> 在该基线上以 `pytest --collect-only -q` 核对存在（抽查记录见 §六）。
> **枚举方式**：`pytest --collect-only -q tests/test_eval_*_runner.py` 得每文件计数，再逐条
> `grep -n "def test"` 人工归类。计数：4+4+4+4+4+10+14+4+7+4 = **59**。

## 状态定义

| 状态 | 含义 |
|---|---|
| **已搬迁** | 测试体逐字（或近逐字）移入新文件；node-ID 改变，断言不变 |
| **已覆盖** | 旧测试主张的**性质**在新套件由一条或多条测试承担（载体可不同） |
| **载体退役** | 旧测试的**字面断言形态**不再存在，但其性质在新层有钉；理由逐条写明 |
| **NET-NEW** | 新套件超出旧套件的性质（旧套件从未测过） |

---

## 一、六个同构 4 测试文件（24 条）

适用文件（每文件 4 条，结构逐字同构，下表以 `test_eval_jump_amount_corr_runner.py`
的 node-ID 为代表，其余五个文件的同名测试映射相同）：

- `tests/test_eval_jump_amount_corr_runner.py`（PR-C，4）
- `tests/test_eval_minute_ideal_amplitude_runner.py`（PR-D，4）
- `tests/test_eval_amp_marginal_anomaly_vol_runner.py`（PR-E，4）
- `tests/test_eval_volume_peak_count_runner.py`（PR-F，4）
- `tests/test_eval_peak_interval_kurtosis_runner.py`（PR-H，4）
- `tests/test_eval_valley_relative_vwap_runner.py`（PR-I，4）

| 旧 node-ID（jump 代表） | 性质 | 新套件映射 | 状态 |
|---|---|---|---|
| `test_minute_loader_is_cache_only_and_discloses_empty` | 分钟取数 cache-only（`live_calls=0`）；未覆盖 symbol 披露为 empty、绝不暖跑；工程化 fixture 上的手算值逐格对上 | `tests/test_factor_eval_providers.py::test_minute_bars_reads_normalized_bars_with_zero_live_calls`、`::test_minute_bars_empty_and_missing_are_empty_never_live`（cache-only + 空披露）；`tests/test_factor_store_universe.py::test_second_symbol_batch_is_filled_not_silently_dropped[fid]`（缺口按 (date,symbol) 判定、不静默丢名）；手算值 → 各因子自己的 `tests/test_<factor>_factor.py`（D2 迁移，11 因子一文件，如 `test_jump_amount_corr_factor.py` 16 条含手算锚） | 已覆盖 |
| `test_minute_loader_blocks_when_nothing_cached` | 全空缓存 → loud `ValueError("no requested symbol produced")`，绝不静默出空结果 | `tests/test_factor_service.py::test_zero_output_symbol_is_not_silently_dropped`、`::test_zero_output_symbol_veto_mutation`（零输出否决 + mutation，**更强**：逐 symbol 级别）；`tests/test_factor_eval_providers.py::test_minute_bars_empty_and_missing_are_empty_never_live` | 已覆盖 |
| `test_evaluate_two_runs_produces_full_reports_and_incremental_axis` | 双跑（no_book/with_book）产出 8 个 mandatory sections + verdict；no_book purity=Skipped → Incremental 轴 NOT_ASSESSED；with_book purity=Section 含 `incremental_ic_ir` → 轴被评估；md/json/dashboard 全落盘；metrics 提取出门禁字段 | `tests/test_exec_basis_eval.py::test_exec_basis_eval_writes_its_own_reports_and_leaves_the_control_alone`、`::test_the_two_exec_artifacts_state_their_own_information_sets`、`::test_exec_basis_eval_discloses_parameters_and_coverage_in_every_report`（落盘 + 披露）；`tests/test_factor_eval_standard.py::test_all_eight_sections_are_present_and_ordered`（8 节齐备）；Incremental 轴语义 → `tests/test_factor_eval_contract.py` 的 purity/Incremental 判决测试（契约层 v0.7 起未动）；metrics → `qt/exec_basis_eval.py::_extract_metrics` 由 `test_exec_basis_eval.py` 各测试间接钉住（metrics dict 喂 CLI 行，见 `test_exec_basis_cli_line_survives_an_absent_metric`） | 已覆盖 |
| `test_no_book_run_never_reaches_adopt` | 结构性保证：no-book + 无执行事实 → 封顶 Watch（exploratory cap + NOT_ASSESSED 轴） | `tests/test_factor_eval_contract.py::test_exploratory_run_is_capped_at_watch_instead_of_adopt`（exploratory 封顶）；同文件的 Adopt 路径测试（`..._ADOPT_FACTS` 族）钉住 Tradable NOT_ASSESSED 时不可达 Adopt。**契约层在重构中一字未动**，此性质不需要 runner 级重测 | 已覆盖 |

## 二、`tests/test_eval_ridge_minute_return_runner.py`（PR-K，10 条）

| 旧 node-ID | 性质 | 新套件映射 | 状态 |
|---|---|---|---|
| `test_minute_loader_is_cache_only_and_discloses_empty` | 同 §一 #1 + CASE A 手算 ridge-minute return 日和（+2.5，区分求和 vs 复利） | §一 #1 各条 + `tests/test_ridge_minute_return_factor.py`（手算值测试族） | 已覆盖 |
| `test_minute_loader_reports_the_ridge_scarcity_distribution` | ridge 分布 + 有效率必须**实测**于真实 bar；披露的 floor 必须是本 run 实际施加的门（否则披露在描述没人执行的门）；render 行含门值 | 诊断帧产生 → `tests/test_minute_diagnostics_channel.py::test_the_diagnostic_set_is_exactly_the_declared_three`、`::test_the_frames_carry_the_symbol_label_the_summarizers_read[factor_id]`、`::test_the_sink_is_scoped_to_the_emit_window[factor_id]`；汇总+门值 → `tests/test_factor_eval_disclosures.py::test_ridge_return_summarize_*`（3 条）+ `::test_summarizer_defaults_are_the_gates_the_binding_applies`（汇总器默认 ≡ 计算门 ≡ 模块常量，防手抄漂移） | 已覆盖 |
| `test_coverage_reports_the_return_guard_attrition` | ridge bar 被日内 lag 丢弃必须**可见**（return_guard_attrition），不得静默吸收 | `tests/test_factor_eval_disclosures.py::test_ridge_return_summarize_reports_the_return_guard_attrition` | 已搬迁 |
| `test_summarize_ridge_coverage_counterfactual_at_the_comparison_floor` | 反事实：把 ridge floor 提到比较基准（20）还剩几天 → 降门买到的天数是个数不是口号 | `tests/test_factor_eval_disclosures.py::test_ridge_return_summarize_counterfactual_at_the_comparison_floor` | 已搬迁 |
| `test_summarize_ridge_coverage_handles_no_frames` | 空输入不除零、NaN 渲染 | `tests/test_factor_eval_disclosures.py::test_ridge_return_summarize_handles_no_frames`（**改名**：编目 BUG 6 的重名冲突在此家闭合，加因子前缀） | 已搬迁 |
| `test_minute_loader_blocks_when_nothing_cached` | 同 §一 #2 | 同 §一 #2 | 已覆盖 |
| `test_evaluate_two_runs_produces_full_reports_and_incremental_axis` | 同 §一 #3 | 同 §一 #3 | 已覆盖 |
| `test_extract_metrics_surfaces_the_pr_k_comparison_quantities` | 头对头比较量（turnover / net_long_short_by_cost {1,2,4} / rank_autocorr / half_life / 截面规模 / monotonicity / gross spread）以数字形式可取 | **性质层**：这些量由评估核产出并钉在契约/标准层（`tests/test_factor_eval_standard.py::test_cost_scenarios_only_move_the_cost_line`、`::test_turnover_charges_the_first_period_and_a_full_rotation` 等；报告 JSON payload 含全量）。**载体**：旧 runner 私有 `extract_metrics` 的**键集形状**不搬迁——统一 runner 的 `_extract_metrics`（`qt/exec_basis_eval.py:242`）按编目 §四收敛为单一公共子集，比较量留在报告 JSON/section payload 里读 | 载体退役（理由 R1，见 §四） |
| `test_absent_stability_metrics_render_as_na_not_a_crash` | Skipped section → 指标 None → 汇总行渲染 `n/a` 而非在错误处理之外抛 TypeError | `tests/test_exec_basis_eval.py::test_exec_basis_cli_line_survives_an_absent_metric`（同一性质，钉在新 CLI 行 `format_exec_basis_line` 上）。注：旧测试里对 `qt.cli._fmt_metric` 的直接断言随旧 CLI 命令在 C6 一并删除，性质已由新钉承担 | 已覆盖 |
| `test_no_book_run_never_reaches_adopt` | 同 §一 #4 | 同 §一 #4 | 已覆盖 |

## 三、`tests/test_eval_valley_ridge_vwap_ratio_runner.py`（PR-J，7 条）

| 旧 node-ID | 性质 | 新套件映射 | 状态 |
|---|---|---|---|
| `test_minute_loader_is_cache_only_and_discloses_empty` | 同 §一 #1 + CASE A 手算 valley/ridge VWAP 比值 | §一 #1 各条 + `tests/test_valley_ridge_vwap_ratio_factor.py` 手算值族 | 已覆盖 |
| `test_minute_loader_reports_the_ridge_scarcity_distribution` | 同 PR-K 同名性质（实测分布 + 门值即所施之门 + render） | `tests/test_minute_diagnostics_channel.py` 各条 + `tests/test_factor_eval_disclosures.py::test_valley_ridge_summarize_*`（2 条）+ `::test_summarizer_defaults_are_the_gates_the_binding_applies` | 已覆盖 |
| `test_summarize_ridge_coverage_counterfactual_at_the_valley_floor` | 反事实：ridge 腿按 valley floor（20）还剩几天 | `tests/test_factor_eval_disclosures.py::test_valley_ridge_summarize_counterfactual_at_the_valley_floor` | 已搬迁 |
| `test_summarize_ridge_coverage_handles_no_frames` | 空输入安全 | `tests/test_factor_eval_disclosures.py::test_valley_ridge_summarize_handles_no_frames` | 已搬迁 |
| `test_minute_loader_blocks_when_nothing_cached` | 同 §一 #2 | 同 §一 #2 | 已覆盖 |
| `test_evaluate_two_runs_produces_full_reports_and_incremental_axis` | 同 §一 #3 | 同 §一 #3 | 已覆盖 |
| `test_no_book_run_never_reaches_adopt` | 同 §一 #4 | 同 §一 #4 | 已覆盖 |

## 四、`tests/test_eval_valley_price_quantile_runner.py`（PR-L，14 条）

| 旧 node-ID | 性质 | 新套件映射 | 状态 |
|---|---|---|---|
| `test_minute_loader_is_cache_only_and_discloses_empty` | 同 §一 #1 + 手算 raw 日分位（三票 0.25/0.50/0.75）+ 无效日（hi==lo）不产值 | §一 #1 各条 + `tests/test_valley_price_quantile_factor.py`（手算族：`test_hand_value_*`、`test_flat_day_with_hi_equal_lo_is_invalid`）+ `tests/test_minute_binding_vpq.py::test_service_read_through_reproduces_the_legacy_runner_cell_for_cell`（service 路径逐格复现旧 runner） | 已覆盖 |
| `test_loader_ships_the_residual_not_the_raw_quantile` |  shipped 因子是反转中性化**残差**不是 raw 分位（防"算了又丢"接线 bug）；带截距 OLS 残差和≈0 | `tests/test_valley_price_quantile_factor.py::test_factor_is_residualized_not_raw_qbar` + `tests/test_minute_binding_vpq.py::test_combine_preserves_the_intermediate_rows_and_is_a_per_date_reduction` | 已覆盖 |
| `test_loader_actually_consumes_the_daily_closes` | 换日频 close 面板必须改变因子值（否则上面全在 raw 分位上空过） | `tests/test_minute_binding_vpq.py::test_shifted_reversal_tracks_a_perturbed_close_bit_identically`（扰动 close → 残差逐位跟动）+ `::test_service_refuses_a_missing_daily_provider_for_vpq`（缺日频 provider loud 拒绝） | 已覆盖 |
| `test_loader_reports_the_neutralization_coverage` | 中性化覆盖率是**实测**的（raw/rev/residual 行数、截面 min/max、raw×rev Spearman） | `tests/test_factor_eval_runner.py::test_valley_price_quantile_runs_end_to_end_with_neutralization_section`（用同一 fake **独立重算**期望值再比对——比旧测试更强，非重言式）+ `tests/test_factor_eval_disclosures.py::test_neutralization_summarize_*`（2 条汇总器单测）+ `::test_neutralization_render_matches_the_former_cli_inline_line`（渲染行逐字钉） | 已覆盖 |
| `test_loader_blocks_when_nothing_cached` | 同 §一 #2 | 同 §一 #2 | 已覆盖 |
| `test_thin_cross_section_leaves_the_factor_nan` | 截面 < `min_cross_section` → 该日全 NaN，绝不退回 raw | `tests/test_valley_price_quantile_factor.py::test_residualization_below_min_cross_section_is_all_nan` + `tests/test_factor_store_universe.py::test_min_cross_section_gate_still_bites_on_the_read_path` | 已覆盖 |
| `test_summarize_neutralization_counts_missing_reversal_rows` | 汇总器对缺 reversal 行的计数 | `tests/test_factor_eval_disclosures.py::test_neutralization_summarize_counts_missing_reversal_rows` | 已搬迁 |
| `test_summarize_neutralization_handles_an_all_missing_reversal` | 全缺 reversal → 零覆盖率、Spearman NaN | `tests/test_factor_eval_disclosures.py::test_neutralization_summarize_handles_an_all_missing_reversal` | 已搬迁 |
| `test_reversal_from_a_daily_panel_is_the_t_minus_1_ratio` | 反转输入 = 日频面板 close 的 **T-1** 比值（不用当日 close） | `tests/test_valley_price_quantile_factor.py::test_reversal_20_is_minus_the_t_minus_1_twenty_day_return`、`::test_reversal_uses_t_minus_1_close_not_day_d` + `tests/test_minute_binding_vpq.py::test_shifted_panel_reversal_is_bit_identical_to_internal_t1`（平移面板第三条路逐比特等价钉） | 已覆盖 |
| `test_evaluate_two_runs_produces_full_reports_and_incremental_axis` | 同 §一 #3 | 同 §一 #3 | 已覆盖 |
| `test_extract_metrics_surfaces_the_pr_l_comparison_quantities` | 同 PR-K 同名性质 + `ic_pearson_mean` 与 rank IC 并列 + sign=+1 时 aligned spread 不误标 | 同 PR-K：性质层量在契约/标准层有钉；`_extract_metrics` 键集形状不搬迁 | 载体退役（理由 R1，见 §四） |
| `test_spec_sign_is_positive_so_aligned_spread_is_not_mis_signed` | PR-L 的预注册 sign=+1（v0.8 成本符号缺陷只咬 sign=−1） | `tests/test_factor_requires_spec_v1.py::test_shipped_factor_declarations_match_the_d0_table[...]`（18 因子声明含 `expected_ic_sign` 参数化钉死）+ `tests/test_valley_price_quantile_factor.py::test_spec_declares_the_pre_registered_sign_and_the_pinned_deviations`；v0.8 对齐价差语义由契约层 verdict 测试钉（`tests/test_factor_eval_contract.py`，该层重构零改动） | 已覆盖 |
| `test_no_book_run_never_reaches_adopt` | 同 §一 #4 | 同 §一 #4 | 已覆盖 |
| `test_min_valley_bars_default_matches_the_pr_i_floor` | PR-L 复用 PR-I 的 valley floor，两 run 覆盖率可比 | **双侧钉值**：`tests/test_valley_price_quantile_factor.py::test_definition_constants_are_the_pinned_values`（`VALLEY_QUANTILE_MIN_VALLEY_BARS == 20`）+ `tests/test_valley_relative_vwap_factor.py`（`VALLEY_VWAP_MIN_VALLEY_BARS == 20`，line 401 断言）。任一侧漂移都会红其一；跨模块相等性断言语句本身不再存在 | 已覆盖（注：相等性由"同值双钉"承载，非单条等式断言） |

## 五、`tests/test_eval_intraday_amp_cut_runner.py`（PR-G，4 条）

| 旧 node-ID | 性质 | 新套件映射 | 状态 |
|---|---|---|---|
| `test_amp_cut_loader_is_cache_only_and_combines_cross_section` | cache-only + per-symbol 统计量 → 截面 z-score combine（截面归零）；stats 面板行数；empty 披露 | §一 #1 各条 + `tests/test_intraday_amp_cut_factor.py::test_amp_cut_cross_section_zscore_hand_value`、`::test_amp_cut_cross_section_below_min_is_all_nan`、`::test_amp_cut_full_pipeline_cross_section` + **D4c 中间量架构**：`tests/test_factor_store_universe.py::test_only_the_cross_sectional_factor_stores_an_intermediate`、`::test_the_cross_sectional_combine_runs_once_per_request`、`::test_served_panel_matches_the_direct_engine_and_adds_only_nan_rows[fid]` | 已覆盖 |
| `test_amp_cut_loader_blocks_when_nothing_cached` | 同 §一 #2 | 同 §一 #2 | 已覆盖 |
| `test_amp_cut_evaluate_two_runs_produces_full_reports_and_incremental_axis` | 同 §一 #3 | 同 §一 #3 | 已覆盖 |
| `test_amp_cut_no_book_run_never_reaches_adopt` | 同 §一 #4 | 同 §一 #4 | 已覆盖 |

## 六、PR-M（`peak_ridge_amount_ratio`）——旧套件 **0 条**（编目 BUG 3）

`qt/eval_peak_ridge_amount_ratio.py` 从未有 runner 测试文件（`grep -rln
eval_peak_ridge_amount_ratio tests/` 无命中）。新套件在此是**净增**：

- `tests/test_factor_eval_disclosures.py::test_peak_summarize_counterfactual_at_the_ridge_floor`
  与 `::test_peak_summarize_handles_no_frames`（**NET-NEW**：三个兄弟因子都有的
  反事实 + 空帧对，PR-M 补平）。
- `tests/test_factor_eval_disclosures.py::test_disclosure_binding_covers_exactly_the_three_publishing_factors`（披露绑定恰好覆盖 PR-J/K/M 三因子，其余因子**陈述式**无披露）。
- 因子数学由 `tests/test_peak_ridge_amount_ratio_factor.py`（43 条）承担。

## 七、新套件超出旧套件的 NET-NEW（选列，非穷举）

统一 runner 落地新增、旧 59 条从未覆盖的性质：

- **runner 配置门**（编目 C1/C2 collapse）：`test_factor_eval_runner.py::test_preconditions_fail_readably`（4 参数化）、`::test_missing_oos_is_a_readable_error`、`::test_invalid_book_mode_is_rejected_before_any_work`。
- **BUG 5 闭合**：`::test_config_book_empty_or_exact_is_accepted`、`::test_config_book_mismatch_is_a_readable_error`（runner 不读 config `factors:` → 声明不符即 loud）。
- **exec 身份 + book 双模式**：`::test_build_eval_config_declares_the_exec_identity_and_shared_kwargs`、`::test_end_to_end_wiring_both_book_modes`（含 `_bookclose` 后缀隔离）、`tests/test_factor_eval_reconcile.py` 的 `check_new_pair_consistency`（no_book/with_book 的 eval_config 只差 `book_view`）。
- **add-Section 桥**：`test_factor_eval_disclosures.py::test_extra_section_never_moves_the_verdict_or_mandatory_sections`、`::test_an_extra_section_may_never_shadow_a_mandatory_name`、`::test_exec_basis_augmentation_seam_preserves_report_and_verdict`。
- **诊断通道**（`test_minute_diagnostics_channel.py` 全文件，10 条）。
- **对账 harness**（`test_factor_eval_reconcile.py`，80 条：硬闸门、panels 具名类正反向、reports 白名单、anchors）。
- **service/store universe 语义**（`test_factor_service.py` / `test_factor_store_universe.py`：单点填≡批量填、零输出否决、(date,symbol) 缺口、中间量 universe 无关、max 形 floor 红测试等）。
- **vpq 绑定等价**（`test_minute_binding_vpq.py`：平移面板第三条路双侧钉）。

## 八、计数对照（R16 非降证据）

| 桶 | 条数 |
|---|---|
| 旧套件总条数（10 文件） | **59** |
| 已搬迁（逐字移入新文件） | 7（PR-J×2、PR-K×3、PR-L×2 汇总器测试） |
| 已覆盖（性质在新套件有钉） | 50 |
| 载体退役（性质仍在新层有钉，字面断言形态不搬迁） | 2（PR-K/PR-L 的 `extract_metrics` 键集形状，理由 R1 见下） |
| **有映射合计** | **59 / 59（100%）** |
| 完全无映射、无理由退役 | **0** |
| 新套件 NET-NEW（超出旧套件的性质，选列见 §七） | ≥40（含 PR-M 补平 3 条） |

**非降结论**：旧 59 条逐条有映射（7 搬迁 + 50 覆盖 + 2 载体退役但性质有钉），无一条
静默丢失；新套件另有 NET-NEW。性质覆盖数严格上升。

### 退役理由

- **R1（PR-K `test_extract_metrics_surfaces_the_pr_k_comparison_quantities` / PR-L
  `test_extract_metrics_surfaces_the_pr_l_comparison_quantities` 的键集形状）**：旧 runner
  各自私有 `extract_metrics`，键集三档（编目 §四实测"与因子无关"）。统一 runner 把
  metrics 收敛为单一公共子集（`qt/exec_basis_eval.py::_extract_metrics`），比较量
  （turnover / net-by-cost 三档 / autocorr / half-life / 截面规模 / monotonicity /
  ic_pearson）仍由评估核产出、钉在契约/标准层测试，并全文落在报告 JSON 的 section
  payload 里——**信息无损失，损失的是"从 metrics dict 里读"这个便利载体**。该载体随
  11 个旧 runner 在 C6 删除；若后续需要同样的 CLI 头对头摘要，应作为一个显式需求
  重新提出，而不是靠保留旧 runner 来保住它。

## 九、node-ID 抽查记录

按完成定义抽查 10 条（`pytest --collect-only -q <file>` 后 grep node-ID），全命中：

| # | node-ID | 结果 |
|---|---|---|
| 1 | `tests/test_factor_eval_providers.py::test_minute_bars_reads_normalized_bars_with_zero_live_calls` | ✅ |
| 2 | `tests/test_factor_service.py::test_zero_output_symbol_veto_mutation` | ✅ |
| 3 | `tests/test_factor_eval_contract.py::test_exploratory_run_is_capped_at_watch_instead_of_adopt` | ✅ |
| 4 | `tests/test_factor_eval_standard.py::test_all_eight_sections_are_present_and_ordered` | ✅ |
| 5 | `tests/test_factor_eval_disclosures.py::test_ridge_return_summarize_reports_the_return_guard_attrition` | ✅ |
| 6 | `tests/test_factor_eval_runner.py::test_valley_price_quantile_runs_end_to_end_with_neutralization_section` | ✅ |
| 7 | `tests/test_minute_binding_vpq.py::test_service_read_through_reproduces_the_legacy_runner_cell_for_cell` | ✅ |
| 8 | `tests/test_factor_store_universe.py::test_min_cross_section_gate_still_bites_on_the_read_path` | ✅ |
| 9 | `tests/test_exec_basis_eval.py::test_exec_basis_cli_line_survives_an_absent_metric` | ✅ |
| 10 | `tests/test_factor_requires_spec_v1.py::test_shipped_factor_declarations_match_the_d0_table` | ✅ |
