# 因子评估契约层 + 十一个分钟因子复现（PR #63 – #74）

> 本文件是 `CLAUDE.md` 进度日志的**归档**，内容逐字搬移，未做任何改写。
> 索引见 [`docs/progress/README.md`](README.md)；当前状态与操作约定仍在仓库根 `CLAUDE.md`。

`analytics/eval/` 三轴判决契约（v0.1→v0.9）、十一因子结果总表与三条循环级结论、以及 v0.8/v0.9 两处判定门缺陷的修复。

---

- ✅ **因子评估契约层 `analytics/eval/` + 十一个分钟因子复现**（**PR #63–#73 已 merge 到 `main`**，**EXPLORATORY，非业绩声明**）：研报 → 复现循环（10/10 收官）与支撑它的判定契约层。全部因子同一数据面：CSI500 `000905.SH` 2021-07-01→2026-06-30 日频、cache-only（`stk_mins_live_calls=0`）、每因子 no_book / with_book 两跑（因子簿 = 已确认的 value_ep + value_bp + volatility_20）。
  - **契约层（PR #63，v0.1→v0.7 迭代后 frozen）**：`FactorSpec`（Factor 必带 classattr、**预注册 `expected_ic_sign`**、PIT 契约、intraday 块校验）+ `FactorEvaluator` ABC（8 个必填 section，缺一即 raise）+ `FactorEvalReport`（确定性 md/json，secret-redacting renderer）+ **三轴判决 Predictive × Incremental × Tradable → Adopt / Watch / Reject / INSUFFICIENT-DATA**。三条判定纪律：**非对称门**（hard Reject 先于样本量门）、**unknown never convicts**（判不了绝不定罪）、**PASS 一律读 N_eff 置信区间下界**（`N_eff = N/(1+2Σρ)`，Geyer 扩展；点估计不得买 PASS）。v0.7：增量轴独立门 `min_incremental_abs_icir=0.15`（正交化收缩残差尺度，沿用原始 0.30 是类别错误）+ 单调性作**方向门**（默认 0.0，非强度门）。exploratory 因子**封顶 Watch**（Tradable 轴全程 NOT_ASSESSED）。一次向量化 IR 喂全部 section（与朴素逐期循环等价性测试锁定）+ 单 PNG dashboard，带**强制 FACTOR DEFINITION 带**——不说明怎么算就不许展示因子。
  - **十一因子（下表 verdict 为当前 v0.9 close-to-close 口径，非各 PR 当时口径；出处 `tmp/design/RESULTS_post_pr75_2026-07-21.md` 参照表 + `tmp/Quantitative_Research_Report/factors/ledger.md`）**：

    | PR | 因子 | 家族 | sign | IC | ICIR | 换手 | 净1× | no_book / with_book |
    |---|---|---|---|---|---|---|---|---|
    | #63 | `jump_amount_corr` | 跳跃×成交额相关 | −1 | −0.0298 | −0.401 | 0.415 | −0.000834 | Watch / Watch |
    | #64 | `minute_ideal_amplitude` | 振幅 | −1 | −0.0291 | −0.464 | 0.990 | −0.001981 | Watch / Watch |
    | #65 | `amp_marginal_anomaly_vol` | 振幅波动 | **+1** | −0.0412 | −0.425 | 0.285 | −0.000504 | Reject / Reject（documented negative：预注册 +1 未迁移，**实测负 IC ≠ sign=−1**） |
    | #66 | `volume_peak_count` | 计数 | +1 | +0.0177 | +0.210 | 0.304 | **+0.000071** | Reject / Reject（符号迁移成功但 ICIR 下界 0.151 < 0.30 预注册门） |
    | #67 | `intraday_amp_cut` | 振幅 | −1 | −0.0347 | −0.435 | 0.578 | −0.000260 | **INSUFFICIENT-DATA** / Watch |
    | #68 | `peak_interval_kurtosis` | 时序 | +1 | +0.0002 | +0.006 | 0.576 | −0.000463 | Reject / Reject（诚实空结果；研报称 RankIC +7.19%/ICIR 4.63 → 本循环最大复现落差） |
    | #69 | `valley_relative_vwap` | 价格比值 | +1 | +0.0327 | **+0.506** | 0.473 | −0.000085 | Watch / Watch |
    | #70 | `valley_ridge_vwap_ratio` | 价格比值 | +1 | +0.0339 | +0.425 | **2.101** | −0.001705 | Watch / Watch（对齐单调 CI 下界仅 **+0.0010**，全表最薄） |
    | #71 | `ridge_minute_return` | 收益 | −1 | −0.0312 | −0.381 | **2.108** | −0.001983 | **INSUFFICIENT-DATA** / Reject（增量轴与因子簿冗余） |
    | #72 | `valley_price_quantile` | 价格位置 | +1 | +0.0265 | +0.414 | 0.679 | **+0.000048** | Watch / Watch（按日单调 +0.1064 全表最高） |
    | #73 | `peak_ridge_amount_ratio` | 成交额 | +1 | **+0.0401** | +0.488 | **2.127** | −0.001402 | Watch / Watch（IC 全场最高，零价格含量） |

  - **三条循环级结论**：① **家族图谱**——计数弱正 / 时序归零 / 收益 Reject / **价格比值×2 + 价格位置 + 成交额 = 四个双轴 PASS**；峰岭谷行为学划分本身有效，且 **不止价格信息有效**（PR-M 零价格含量却 IC 最高）。② **IC 强 ≠ 可交易（贯穿证据）**——11 个因子里只有 PR-F（+0.71bp）与 PR-L（+0.48bp）在 1× 费率下净多空为正，且 PR-L 扛不住 2×；毛价差常相近，差距**几乎全部来自换手**（PR-I/L 的 0.47/0.68 vs PR-J/K/M 的 2.1+）。③ **假说证伪记录（不得作为已确立规律引用）**——"秩IC/Pearson 背离比预测单调性"被 PR-M 证伪（3.86×→0.60 与 PR-J 4.22×→0.90 交叉），仅两个极端处有不确定关联。
  - **方法论（本轮建立，已作常设规则）**：① **不可能失败的测试**——三个实例：PR-L 的旗舰反泄漏测试（全截面统一扰动是带截距 OLS 残差不可见的仿射变换）、`compare_postmerge.py`（拿新结果和自己比，见下条 #74 重跑对账）、I5b fixture 给每根 bar 定价使 **VWAP ≡ close**（对"闸门读哪个"结构性失明）。**任何不变性测试都必须有实际跑过的 mutation 证据，且 mutation 要打中所主张的那个具体性质**。② **行为变了而措辞没变**（#76/#78 实证）——正确的修法**不是**"断言不存在第 N+1 处副本"（#76 就是这么做的，用扫描式正则，被七种改写全部绕过），而是 **把主张只写一遍、其余每一处都去组合它**：**正则断言不了"没有别的句子这么说"，而"根本没有别的句子"可以**。
- ✅ **因子评估契约 v0.8 + v0.9**（**PR #74 已 merge 到 `main`**，`1ae555e`）：十一个因子在 `analytics/eval/` **严格 no-touch** 下跑完后暴露的两项 frozen 缺陷，本 PR 是**对该 frozen 契约的首次授权修改**。
  - **缺陷一：对齐净价差把成本加了回去（判定门缺陷，非仅展示）**。四处写成 `sign * net`（`net` 已扣成本），`sign=−1` 下展开为 **`−gross + cost`**——因子付的手续费被当利润返还。正确语义 `aligned = sign*gross − cost`（先按假设翻腿、再永远扣成本）。**其中两处在 `verdict.py`（`_base_spread` / `_all_spreads_negative`），直接喂 Tradable 轴 PASS 条件**，历来未触发仅因该轴全程 NOT_ASSESSED。手算实证：`gross 0.000125 / net 1× −0.001983 → cost 0.002108 → aligned −0.002233`（缺陷式给 **+0.001983**）。`sign=+1` 逐比特不变。年化展示值修正：jump_amount_corr **+23.14%→−0.01%**、minute_ideal_amplitude **+64.07%→−0.27%**、intraday_amp_cut **+6.30%→−20.52%**、ridge_minute_return **+64.28%→−43.25%**（**旧值全是虚高**）。
  - **缺陷二：用量级敏感的池化统计量给秩基轴把门**。`monotonicity_spearman` 是"桶序号 vs 跨日等权算术均值"的相关，无界，少数极端收益日集中于某桶即可翻号，而每日封顶的秩 IC 几乎不动。新增 `monotonicity_spearman_by_date`（逐日 Spearman，先封顶 [−1,1] 再平均，与秩 IC 同构）并把门移过去；池化版保留为报告字段 + 回退披露。
  - **v0.9：新统计量被日内噪声衰减，裸 0.0 门落在噪声带内**。实测分位阶梯完美的因子按日只有 **0.045–0.106**，门却仍在裸 0.0 且无离散度估计，两个因子相差 0.021 就分生死。改为读该按日序列的 **N_eff 置信区间**（复用既有 `mean_ci`），三值：对齐 CI 下界 > bar → **HOLDS**；对齐 CI 上界 < 0 → **CONTRADICTED**（FAIL）；跨零 → **UNKNOWN**（既不定罪也不放行）。实测 CI 半宽 **0.0291–0.0448** → 裸 0.0 实为一道约 0.035 的隐式门槛。**最要害的路由规则：UNKNOWN 绝不救本就不合格的因子**——单调性已从点信号布尔链中摘出（三值事实塞进 `and` 链必然坍缩），ICIR / NW-t / win / OOS 任一不清关仍走 FAIL，UNKNOWN 只在其余全清关时把预测轴落到 INSUFFICIENT_DATA。
  - **3 处 verdict 变化**（全部归因于门的变化，并如此陈述）：`intraday_amp_cut`(no_book) Reject→INSUFFICIENT-DATA（对齐 CI [−0.0353,+0.0343] 跨零）、`intraday_amp_cut`(with_book) **Reject→Watch**（同上 + 增量轴本就 PASS）、`ridge_minute_return`(no_book) Watch→INSUFFICIENT-DATA（对齐 CI [−0.0151,+0.0558] 跨零）。`ridge_minute_return` 跨三版本轨迹 **v0.7 Reject → v0.8 Watch → v0.9 INSUFFICIENT-DATA**：前两次都是在噪声上判决、只是方向不同，第三次才承认判不了。
  - **不变量**：IC / ICIR / NW-t / win / 增量 ICIR / 两个单调点值 **全部逐比特不变**（修复未泄漏到 IC 路径与增量路径）。**路由自检**：`amp_marginal_anomaly_vol`（OOS 符号翻转，第一步即 FAIL）与 `peak_interval_kurtosis`（OOS 不一致 + ICIR 0.006）单调方向均为 UNKNOWN 但仍 Reject，无渗漏。
  - **合并后十一因子干净全量重跑（`main` @ `1ae555e`，11/11 rc=0，~87min）**：`repro_reconcile.py` 11 因子 × 20 字段 = 220 项，**容差 0.0**（精确浮点相等，不用 `isclose`、不留 epsilon —— 1e-6 是"可能动"的下限，用恰好为零来测才能让任何 ≥ 该下限的变化浮出来）→ **0 处差异**；`repro_compare.py` 对 22 个 JSON artifact 做展平叶子深比 → **5,756 叶、0 处不同**，外加 **22/22 Markdown 逐字节相同**。防空过：22 个 JSON 确实被重写（mtime 落在各自 run 窗口 09:24:33→10:43:54）、前后 HEAD 都是 `1ae555e`、11 个 rc=0、全程 `stk_mins_live_calls=0`。
  - ⚠️ **记一次自己的失误：写了一个不可能失败的对账脚本**。第三个脚本 `scratchpad/compare_postmerge.py` 曾被跑并报出结论，**但它是空的，其输出什么也不证明**：它执行 `_baseline_table.py` 取新值，而那个脚本第 123 行**会写** `baseline_rows.json`，随后"旧值"读的正是刚被覆盖的同一个文件 → `old == new` 由构造成立，哪怕 artifact 错得离谱也会打印"11 个全部精确复现"（已在沙箱里用 baseline `ic=1.0` vs fresh `ic=999.0` 复现该假通过）。它还顺带毁掉了 `baseline_rows.json`。**这正是整轮重跑要消灭的失效模式，而我们亲手造了一个**（同 P2-3「曾误以为是粒度」、P3-4「过拟合签名」，照实记录）。上面的结论只站在不碰该文件的两个脚本上（其中一个跑在覆盖之前）。**权威基线是 `scratchpad/baseline_reports/`（22 文件，冻结于 run 前）；复核必须比它，不是 `baseline_rows.json`。**
