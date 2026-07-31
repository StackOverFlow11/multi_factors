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
| 是否可被覆盖 | **否**（本次改动只新增，一个字节没动） | **否**（C6 起同样 frozen-forever；产生方式已退役，见 §四） |

**为什么「旧引擎 + 截断输入」是干净的 refactor-only 对照**：本次改动是**纯输入截断**——
被喂进相关系数计算的 bar 少了一批，`compute_jump_amount_corr` 里从 `amp` 到 Pearson
闭式解的每一行代码一字未动。所以在**同一条 pre-D5 runner 取值路径**上重跑，得到的面板与
原基线之间只差「截断」这一个变量；D5 拿它对账新路径时，任何差异都仍然只能归因于加载几何 /
饱和深度 / anchor 截断，而不会混进一个定义变更。

**不要**把两份面板放进同一次对账去「取平均」或「看哪个更接近」：它们是两个不同定义的因子值，
不是同一个量的两次测量。

## 四、新参照面板的产生方式（**已于 D5 C6 退役**，此处存档）

⚠️ **下面这条命令不再可用**：owner 2026-07-28 裁定退役重生成能力，`--only` 连同整个冻结
路径一并退役（它依赖 11 个旧 runner 的私有 loader，C6 删除它们后必然失效）。本面板与 D1
基线一样 **frozen-forever**；验证走
`python -m qt.panel_freeze --verify`，它会把本文 §五 表格里的 `canonical_sha256` 当作
git 里的权威期望值，对盘上这个面板逐格重算核对。原命令留在这里是 provenance 的一部分——
它记录的是这份面板**当初怎么来的**，不是现在怎么再来一次。

```
cd <repo root>            # 缓存根 artifacts/cache/tushare/v1 就位   [RETIRED, C6]
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
- 该 run 的 manifest header 自报 `selected_factors: jump_amount_corr_20`。**原基线
  manifest 里根本没有这个键**——它产生于 `--only` 存在之前，所以两者的区别是「有该键 vs
  无该键」，不是「两个不同的值」。（初稿曾写成原基线写着 `selected_factors: all`，评审实测
  证否；如实更正。）
- ⚠️ 与原基线的 provenance 规则**刻意不同**：D1 基线要求「只许从钉住的 pre-D2 SHA
  checkout 重跑」；本参照面板**必须**在含有截断修正的树上跑（这正是它要记录的东西）。
  它的 `producing_git_sha` 记在自己的 manifest header 里。

## 五、实测结果（bulk artifact gitignored，故权威数字记在这里）

机器可读版在 `artifacts/refactor_baseline/pr_c_cutoff_fix/manifest.{json,md}`。

| 项 | 值 |
|---|---|
| producing_git_sha | `199a152c93f8f94248227d201d06d635705ae6d9`（含截断修正的本分支 commit） |
| selected_factors | `jump_amount_corr_20` |
| universe / 窗口 | `index:000905.SH` / 2021-07-01..2026-06-30（996 成分、1210 交易日、面板 1,158,912 行） |
| `stk_mins_live_calls` | **0**（cache-only，与 D1 冻结同约束） |
| determinism_double_run | `jump_amount_corr_20:ok` |
| artifact_reconciliation | 1/1（对新 eval artifact 的 `data_coverage`） |
| elapsed | 729.8 s |

| factor_id | rows | n_symbols | n_nan | mean | std | canonical_sha256 |
|---|---|---|---|---|---|---|
| jump_amount_corr_20 | 1159263 | 995 | **891** | **0.5929255105118916** | **0.1575466344089912** | `fafda48514d6e99055bd070a420250e388655fe11c519a9905961e71cffb2707` |

对照 D1 原基线同一因子（**未截断**）：`n_nan` 880 / mean 0.5912439770493371 /
std 0.15727054230823656 / canonical `b6359f12…`。行数、日期范围、symbol 数完全相同 ——
变的是值本身与 891−880 = **11 个新增 NaU**（截断后某些日的 jump-pair 数掉到 `min_pairs`
以下，honest missing）。

**免费拿到的确定性证据**：本参照面板被冻结过**两次**——一次在未提交的工作树上、一次在
提交后的 `199a152` 上，两次跑在不同进程、不同时刻，`canonical_sha256` 与 `file_sha256`
**逐字节相同**。第一次的 manifest 因此被第二次覆盖（它的 `producing_git_sha` 指向
`d806e70`，即**不含**修正的分支基点 —— 那是一句会误导人的 provenance，所以重跑而不是
留着解释）。

写盘**前**先过 `_process_factors` + 与新 eval artifact 的 `data_coverage` 对账
（`panel_rows` / `evaluation_periods` / `symbols_evaluated` /
`universe_symbols_declared` / `dropped_symbols_count` / `factor_nan_rate`），
对不上就 raise、不落盘。

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

## 六之二、更正声明放在哪里（本 PR 第二次自查纠错）

**第一版把更正写成 `spec.description` 的一段散文，报告称「四份 artifact 各含 1 处、同时进
JSON 与 dashboard」——三个说法里两个是错的**：

| 承载 | 第一版实况 |
|---|---|
| Markdown | ✅ 完整 |
| JSON（**机器可读的那一份**） | ❌ `spec.description` 被 `sanitize_payload` 封顶 **200 字符**并追加 `...[truncated]`；更正从第 353 字符起，**整段不在里面** |
| dashboard PNG | ❌ 加长的 description 使 FACTOR DEFINITION 带从 3 行涨到 5 行，**与下面的 metadata 行叠字**，被叠掉的正是更正那几行 |

即：**更正在两条路上都没到达，而恰恰是我报告说已经做到的地方**——与本 PR 正在修的缺陷同形
（一份文档没有说出关于它自己出处的该说的话）。

**根因是通用的，不是 jump 特有**：实测已发布的 **44/44** 份 eval JSON 的 `spec.description`
全部被截断（共 218 个 `[truncated]` 标记，其中三处还是每份 artifact 里的方法学说明）。

**改法（评估契约 v1.1）**：更正成为**结构化字段** `FactorSpec.corrections`
（`FactorCorrection` 元组：`from_version`/`to_version`/`date`/`defect`/`effect`/`superseded`，
每项非空校验、**超长 raise 而非裁剪**、`to_version` 必须等于 spec 自身 version 以便
`spec.version` 单独就能判别一份存档在更正的哪一侧）。JSON 顶层新增 `corrections` 键
（与 `eval_contract_version` 同级），经 `corrections_record` 走**封顶通道之外**导出；Markdown
provenance 行与 PNG 头行标记**从同一元组派生**（author once）。`description` 恢复为纯定义。

**守卫（`tests/test_factor_correction_carrier.py` 15 项）不做渲染文本子串匹配**（D5b 刚因此
吃过亏）：把**声明的对象**经真实 `report_to_dict` + `json.dumps/loads` 往返后与原对象**相等
比对**；并钉住「这些字段确实超过封顶长度」，所以「往返完好」只有在承载真的在封顶通道之外时
才成立。

## 六之三、dashboard 叠字（既有缺陷，本 PR 顺带修 + 加守卫）

FACTOR DEFINITION 带把描述锚在 `y=0.62`、metadata 锚在 `y=0.20/0.10`，**中间无约束** ⇒ 长
描述直接画在 metadata 上。真实 dashboard 几何实测：**11 个分钟因子里 7 个重叠**
（`valley_price_quantile_20` 超 **296 px**，25 行对 4 行槽位）。
`analytics/eval/figures.py` **在 git 里只有一个 commit** ⇒ **不是本 PR 的回归，而是每一份已
产出 dashboard 都有的既有缺陷**（含冻结的 22 张）。

修法**不动任何因子的描述文本**（那才是爆炸半径）：`definition_description_lines` 限行到
`DEFINITION_MAX_LINES=4`，超出部分以**显式省略标记**收尾并指向 Markdown（那一份 description
是完整的）。省略优于覆盖：覆盖是两段文字同时不可读**且不声明**。

守卫 `tests/test_definition_band_layout.py`（25 项）用**真实 dashboard 的 figsize/GridSpec**
渲染后取 Text 的 window extent 比 bbox —— **几何断言，不是看图，也不是子串匹配**。

## 六之四、D2 手算锚重跑：让冻结面板的陈旧性显形

`qt/hand_anchors_d2.py::hand_jump_amount_corr` 原本注释明写「引擎无 14:50 截断」并据此读
整日 bar —— 一个**忠实复现缺陷**的手算参照。改成 `pit=True` 后重跑（`python -m
qt.hand_anchors_d2`，~7min）：

```
frozen 14: 70 rows, 5 mismatches   (RC=1)
FAIL jump_amount_corr_20 warmup_end     2021-07-02 000537.SZ hand=0.4613901757 engine=0.4730559418 rel=2.47e-02
FAIL jump_amount_corr_20 ex_date_window 2021-07-21 000050.SZ hand=0.7755175868 engine=0.7776570622 rel=2.75e-03
FAIL jump_amount_corr_20 random         2024-05-30 002690.SZ hand=0.3831735019 engine=0.4011519443 rel=4.48e-02
FAIL jump_amount_corr_20 random         2023-05-18 600867.SH hand=0.5843712903 engine=0.5831966539 rel=2.01e-03
FAIL jump_amount_corr_20 random         2026-01-30 002773.SZ hand=0.6210838723 engine=0.6210838723 rel=1.06e-03
```

**这 5 个 FAIL 是正确结果，不是回归**：手算侧已截断，而 `engine` 读的是**冻结的**
`artifacts/refactor_baseline/panels_d2/jump_amount_corr_20.parquet`（未截断口径，按约定不动）。
其余 **63 行全部 OK、无一其他因子移动** —— 差异被精确隔离在这一个因子上。

重跑**前**盘上的状态是 5 行 hand==engine **逐位相同**：两侧都复现同一个缺陷，所以看起来干净。
这正是「手算参照必须独立于引擎」的意义 —— 它一旦跟着引擎一起错，就不再是参照。
旧文件已备份到 scratchpad（gitignored，非 git 记录）。

## 七、deny list 的解除（本 PR 内完成）

D5b（PR #99）新增了一条**有意的响亮阻断**：`qt.exec_basis_eval.subject_view(factor_id)`
从事实派生信息集视图，事实源是 `factors/compute/minute/binding.py` 的 deny list
`NOT_DECISION_CUTOFF_SAFE`；`jump_amount_corr_20` 在表里 ⇒ 派生出 (close, exec_to_exec)
非法配对 ⇒ `EvalConfig` 拒绝 ⇒ 它的 exec 评估 loud raise。在截断修正之前这是对的：不存在
它的诚实 exec artifact，阻断优于假声明（红线 #9）。

**本修正正是解除条件**，且解除与重述在**同一个 PR** 内完成，不留「因子已修好、评估仍被拦」
的中间态：本分支切自 `main@d806e70`（deny list 尚不存在，全仓零命中），合入 `main@435a798`
后删除该条，`NOT_DECISION_CUTOFF_SAFE` 变为空集。

**独立佐证（比本 PR 自己的断言更有说服力）**：D5b 的
`tests/test_decision_cutoff_visibility.py` 是**另一个作者、另一套 fixture**（12 名 × 40 日 ×
240 bar 随机游走）独立写的测量。合入后**先不删 deny list 直接跑**，它给出：

```
FAILED test_the_known_exception_is_still_exactly_one_factor_and_still_leaks
AssertionError: jump_amount_corr_20 no longer depends on post-14:50:00 bars.
  If that was deliberate, this factor's PUBLISHED values changed: drop it from
  factors.compute.minute.binding.NOT_DECISION_CUTOFF_SAFE, and re-state its
  verdict rather than letting the artifacts drift silently.
```

—— 它测量到截断确实生效，并逐字给出了本 PR 随后执行的两个动作。删除后该因子进入 clean
参数化（十个 bars-bound 因子全测、全 0 cell 移动）。

配套改动（都不是削弱，是把主张改成仍然为真的那一个）：

- `test_the_known_exception_is_still_exactly_one_factor_and_still_leaks` →
  `test_the_deny_list_is_empty_and_that_emptiness_is_a_measurement`：断言**两件事**——
  deny list 为空 **且** 被测量的集合恰等于 bars-bound 集合。「空」与「测过」是不同的主张，
  只断言前者会让「从没测过」也通过。新 offender 仍被 clean 参数化抓住。
- `tests/test_eval_contract_v1.py::test_a_factor_that_is_not_cutoff_safe_cannot_get_an_exec_identity`：
  该拒绝路径不再有真实 offender，于是**注入**一个（monkeypatch 把一个真因子类放进 deny
  list）而不是删掉测试 —— 删掉会在下一个 offender 出现的那天没有守卫；继续点名一个已经修好
  的因子则正是本仓反复吃亏的 stale-wording。
- `docs/factors/d5_runner_difference_catalogue.md` §七之二 的「现在会 loud raise」条目改为
  「已解除，解除于本 PR」。D5b 评审留的那条 NIT（阻断点在 runner 里偏晚）随之自动消失。

**仍待收敛（留给后续，本 PR 不做）**：现在有**两处**测量同一性质 ——
D5b 的 `tests/test_decision_cutoff_visibility.py`（10 个 bars-bound 因子，随机游走 fixture）
与本 PR 的 `tests/test_minute_decision_cutoff_leakage.py`（11 个 Factor 子类 + `mmp_ew`，
两把刀，带覆盖闭包）。二者互为独立复核，现在**同时为真**是好事；但长期应 author-once
收敛成一处（本 PR 的覆盖是严格超集）。此处如实记录，不假装已经合并。

## 八、为什么单独成 PR（不许夹带进 D5）

这条截断本身是**研究侧**常设授权的补用（用户 2026-07-19：「全日数据因子：一律截断到
14:50…授权直接截断，不必逐因子再问」），**不是新的研究决策**。但从**重构视角**它仍然是一次
**定义变更**（已发布因子的取值变了），而设计 §〇 明令重构不改定义。两者不矛盾 —— 授权来自
研究侧，不来自重构。所以它自己一个 PR、显式标注 correctness fix，D5 的任何 commit 都不许
夹带它。
