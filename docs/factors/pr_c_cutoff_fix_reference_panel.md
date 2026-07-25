# PR-C `jump_amount_corr_20` 截断修正 —— 两份冻结面板的 provenance

> **状态**：correctness fix 交付物（**不属于任何重构步骤**，见下「为什么单独成 PR」）。
> 本文是入 git 的权威 provenance；两份 bulk 面板都在 `artifacts/refactor_baseline/`
> （gitignored）。

---

## 一、缺陷（一句话）

`compute_jump_amount_corr` 是十一个分钟因子里**唯一没有 14:50 日内截断**的一个：它建于
2026-07-19「全日数据因子一律截断到 14:50」这条常设授权**之前**，此后没有被补上。它在
日 `d` 的取值因此吃进当天 14:50–15:00 的 1min bar。

在 PR #79 把入场锚从 `close(d)` 移到 **14:51 VWAP** 之后，这不再只是「定义与声明不符」，
而是**严格前视**：因子的信息集越过了它自己的入场锚十分钟，且正是 A 股全天成交最集中的
窗口。已发布的 exec artifact 里 `spec.decision_cutoff="14:50:00"` —— 报告声明了一个它
没有执行的检查。

修法：改用兄弟因子（ideal-amplitude / amp-anomaly / amp-cut）**完全相同**的
`factors.compute.minute.primitives.visible_minute_frame`，不新造机制、不动相关系数数学。

## 二、爆炸半径：哪条取值路径受影响，哪条不受

| 取值路径 | 修正前 | 说明 |
|---|---|---|
| 旧 eval runner `qt/eval_jump_amount_corr.py` | **有缺陷** | `end = 当日 + 23:59:59` 读整日 bar，直接喂 `compute_*`，全程无补救。**已发布的 22 份 artifact 走的就是这条**。 |
| `qt/panel_freeze.py`（D1 冻结） | **有缺陷** | 它调的就是 runner 的 `_load_jump_factor_panel`，所以 `artifacts/refactor_baseline/panels/jump_amount_corr_20.parquet` 继承同一缺陷。 |
| D4 materializer **decision 视图** | **本来就对** | `materialize.py` 在调 binding 前先跑 `minute_decision_cutoff`，与 `visible_minute_frame` 是**同一个谓词**（`available_time <= trade_date + decision_time`），因此双重过滤幂等 —— 这条路径的取值**逐位不变**。 |
| D4 materializer **close 视图** | 有缺陷 | close 视图不做日内截断，靠因子自己截。修正后与 decision 视图对齐（十个兄弟本来就是这样）。 |

`factors/compute/minute/binding.py` 的模块 docstring 早就写着「compute 函数以其
**默认 decision_time（14:50）作为冗余但一致的内部截断**」—— 这句话对十一个因子里的十个
成立，对 jump **不成立**。修正后它对全部十一个成立。

## 三、两份面板：各自是什么、该用在哪

| | `panels/jump_amount_corr_20.parquet` | `pr_c_cutoff_fix/panels/jump_amount_corr_20.parquet` |
|---|---|---|
| 定义 | 因子 **v1.0**，无 14:50 截断 | 因子 **v1.1**，截断后 |
| 产出 | D1 冻结 run（`main` @ `3669c90`），见 `d1_panel_freeze_manifest.md` | 见下 §四 |
| 引擎 | pre-D2 runner loader | **同一条 runner loader**（`qt.panel_freeze` + `--only`） |
| 用途 | **已发布内容的忠实记录**；D2 逐位对账的历史参照 | **D5 面板腿对该因子的有效参照** |
| 是否可被覆盖 | **否**（本次改动只新增，一个字节没动） | 可再生（命令在 §四） |

**为什么「旧引擎 + 截断输入」是干净的 refactor-only 对照**：本次改动是**纯输入截断**——
被喂进相关系数计算的 bar 少了一批，`compute_jump_amount_corr` 里从 `amp` 到 Pearson
闭式解的每一行代码一字未动。所以在**同一条 pre-D5 runner 取值路径**上重跑，得到的面板与
原基线之间只差「截断」这一个变量；D5 拿它对账新路径时，任何差异都仍然只能归因于加载几何 /
饱和深度 / anchor 截断，而不会混进一个定义变更。

**不要**把两份面板放进同一次对账去「取平均」或「看哪个更接近」：它们是两个不同定义的因子值，
不是同一个量的两次测量。

## 四、新参照面板的产生方式（可复跑）

```
cd <repo root>            # 缓存根 artifacts/cache/tushare/v1 就位
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.panel_freeze \
    --config config/phase_c_jump_amount_corr.yaml \
    --output-root artifacts/refactor_baseline/pr_c_cutoff_fix \
    --only jump_amount_corr_20
```

- `--only` 是本次为此新增的 **additive** 参数（默认 `None` = 全部 14 个因子，行为逐字节
  不变）。它存在的理由：**不为一个因子重造第二套冻结实现**，也不为十三个没变的因子重读
  2.79 亿行分钟缓存。选择在昂贵的数据面之前就被校验，未知 id 是可读报错 —— 「froze 0
  panels」被当成功报出去，正是本仓记录在案的空对账形态。
- 共享数据面（universe / 日频面板 / 富化）**照旧用完整因子簿**构建，所以过滤 run 看到的
  日频面板与整跑完全相同；`--only` 只决定**冻什么**，不决定**加载什么**。
- 该 run 的 manifest 自报 `selected_factors`，与原基线 manifest（`selected_factors: all`）
  一眼可分。
- ⚠️ 与原基线的 provenance 规则**刻意不同**：D1 基线要求「只许从钉住的 pre-D2 SHA
  checkout 重跑」；本参照面板**必须**在含有截断修正的树上跑（这正是它要记录的东西）。
  它的 `producing_git_sha` 记在自己的 manifest header 里。

## 五、实测结果（本次 run 填入）

见 `artifacts/refactor_baseline/pr_c_cutoff_fix/manifest.md` / `manifest.json`。要点：

- `stk_mins_live_calls = 0`（cache-only，与 D1 冻结同约束）；
- 写盘**前**先过 `_process_factors` + 与新 eval artifact 的 `data_coverage` 对账
  （`panel_rows` / `evaluation_periods` / `symbols_evaluated` /
  `universe_symbols_declared` / `dropped_symbols_count` / `factor_nan_rate`），
  对不上就 raise、不落盘；
- 确定性双跑（`jump_amount_corr_20` 是 `DETERMINISM_FACTORS` 成员）跨进程重建一次，
  canonical hash 必须相等。

## 六、重述后的评估结果（CSI500 2021-07-01..2026-06-30，cache-only，`stk_mins_live_calls=0`）

**唯一变量是截断**：同一条旧 runner、同一配置、同一缓存、同一评估器；另外只有两处
**非数值**元数据随定义变更（`spec.version` 1.0→1.1、`spec.description` 增加更正段），
它们进报告标题/dashboard 文本，不进任何计算。

**operative basis = `exec_to_exec`（14:51 VWAP）**：

| 指标 | old v1.0（未截断，取自冻结副本） | new v1.1（截断后） |
|---|---|---|
| IC mean | −0.030840 | −0.030539 |
| ICIR | −0.425348 | −0.426002 |
| ICIR 95% CI（N_eff） | [−0.485384, −0.365312] | [−0.485645, −0.366360] |
| N_eff | 1162.21 | **1177.89** |
| NW-t | −15.191 | −15.260 |
| win rate | 0.66915 | 0.67246 |
| 按日单调 | −0.059222，CI [−0.093645, −0.024799] | −0.059636，CI [−0.093840, −0.025432] |
| 换手（多空腿） | 0.414632 | 0.417067 |
| 净多空 1× | −0.000834 | −0.000843 |
| 增量 ICIR（with_book） | −0.300601，CI [−0.363276, −0.237926] | −0.299010，CI [−0.361400, −0.236619] |
| **verdict no_book / with_book** | **Watch / Watch** | **Watch / Watch**（三轴逐一相同） |

`close_to_close` 并排对照（保留作控制，正在退出评估契约）：IC mean −0.029845 → −0.029548；
ICIR −0.400976 → −0.401519；win rate 0.66584 → 0.66088；增量 ICIR −0.290002 → −0.288083；
verdict 同样 **Watch / Watch** 不变。

⚠️ **不许把「聚合指标只动了一点」读成「缺陷无害」。** 这两件事互相独立：
- **每一个** emitted cell 都被污染过（实测：真实缓存样本 1,477/1,477 与 1,373/1,373 全动，
  max|diff| 1.28 / 1.05；只扰动 `bar_end >= 14:51` 时 1,363/1,363 动、max|diff| 1.30）。
- 聚合的秩 IC 变化小，是因为被污染的 bar 只占 20 日池化窗口的约 4.5%，且秩 IC 逐日封顶 ——
  **这是稀释，不是无害**。而且这个「小」是**跑出来的**，不是从截断前后值的 pearson r=0.9993
  推出来的：相关系数高与 verdict 是否存活没有推理关系，所以必须重跑。
- verdict 不变**不追认旧 artifact**：旧值在 exec 基准下是前视产物，无论它当时给出什么标签，
  都不构成一个诚实的结论。现在这一版才是。

## 七、落地清单（合并顺序敏感）

并行的 D5 分支新增了一条**有意的响亮阻断**：`analytics/eval` 的 `subject_view(factor_id)`
从事实派生信息集视图，事实源是 `factors/compute/minute/binding.py` 的 deny list
`NOT_DECISION_CUTOFF_SAFE`；`jump_amount_corr_20` 在表里 ⇒ 派生出 (close, exec_to_exec)
非法配对 ⇒ `EvalConfig` 拒绝 ⇒ 它的 exec 评估 loud raise。在截断修正之前这是对的（不存在
它的诚实 exec artifact，阻断优于假声明）。

**本修正正是解除条件**，因此：

1. **合并顺序上靠后的那个 PR 负责删除 `NOT_DECISION_CUTOFF_SAFE` 里的 `jump_amount_corr`
   一条。** 本 PR 基于 `main@d806e70`，那时 deny list、`subject_view` 与
   `tests/test_decision_cutoff_visibility.py` **都还不存在**（已核实：全仓零命中），所以本
   分支上没有可删的东西；若 D5 先合并，本分支 rebase 后即删。
2. **不要让它依赖谁记得**：deny list 应当**断言等于**可见性测量出来的不安全集合，而不是
   手工维护 —— 那样 `jump_amount_corr` 一旦变安全，测试会自己变红逼出删除。否则两边都合并
   而无人删除时，这个因子会**永远**评不出 exec 结果，而且从不跑该评估的人看不见 ——
   正是本仓记录在案的「行为变了而措辞没变」家族。
3. **两处测量应合一**：本 PR 的 `tests/test_minute_decision_cutoff_leakage.py` 对全部 11 个
   分钟因子 + `mmp_ew` 测的就是同一个性质。两边合并后会存在两处事实源，应由靠后的那个 PR
   收敛成一处（author-once），而不是并存。

## 八、为什么单独成 PR（不许夹带进 D5）

这条截断本身是**研究侧**常设授权的补用（用户 2026-07-19：「全日数据因子：一律截断到
14:50…授权直接截断，不必逐因子再问」），**不是新的研究决策**。但从**重构视角**它仍然是一次
**定义变更**（已发布因子的取值变了），而设计 §〇 明令重构不改定义。两者不矛盾 —— 授权来自
研究侧，不来自重构。所以它自己一个 PR、显式标注 correctness fix，D5 的任何 commit 都不许
夹带它。
