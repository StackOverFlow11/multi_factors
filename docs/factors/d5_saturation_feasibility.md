# D5 前置探测：D4 materializer 能否跑真实评估 universe

> **状态**：D5 第二个交付物，**先于任何长 run**（lead 指令：「宁可报『这条路走不通、原因如下』，
> 也不要跑一个 2 小时后失败或产出错值的 run」）。
> 复验：`python -m qt.saturation_probe`（cache-only，实测 `live_calls=0`）。
> ⚠️ **前置条件**：需要 gitignored 的 `artifacts/`（分钟缓存 + D1 冻结面板）。**在没有它的
> worktree 里开箱跑不起来**——先对主 checkout 的 `artifacts` 建符号链接，或用
> `--cache-root` / `--universe-panel` 指向绝对路径（先例：`qt/factor_hotpath_smoke.py`）。
> 建链接时注意：worktree 里若已存在真实 `artifacts/` 目录，`ln -s` 会嵌套成
> `artifacts/artifacts`（本步实际踩到过）。
> 全部数字为**实测**，标注为外推的才是外推。

---

## 结论先行

**D4 的 materializer 跑不了 D5 的评估 universe。** 不是慢，是 **2.19× 于本机可用内存**。
它在 40 票热路径 smoke 上验收，评估面是 **995 票**，而它把整个 universe 装进**一个** frame。

且这不止影响 pooled 因子：**bounded 路径同样整 universe 一次性加载**
（`factors/materialize.py:262`），评估窗口本身就要 **52.7 GB**，已经贴着 56 GB 可用内存。
所以「per-symbol 流式化」不是 pooled 因子的局部优化，**是 11 个因子全都需要的结构前提**。

---

## 一、饱和为什么必然走到 floor（不是偶发，是构造性的）

`_pooled_pool_saturated` 要求**每一个被请求的 symbol** 在锁定子窗口里、在 `emit_start`
当天或之前攒够 `lookback_days` 个有效日，否则否决终止（D4 review HIGH 的修法，正确且
故意保守）。

**实测**：评估 universe 995 票中，**84 票的第一根 1min bar 在 `emit_start` 2021-07-01
当天或之后**（最晚到 2026 年）。这些票**无论往回加载多深都攒不出 emit_start 之前的有效日**
——因为那时它们还没上市。于是判据永远为 False，循环一路扩到声明 floor `2015-01-05`。

| 首根 bar 年份 | 票数 |
|---|---|
| 2015 | 668 |
| 2016–2020 | 223 |
| **2021（含 ≥07-01 的）** | 39 |
| **2022–2026** | **65** |

这是 CSI500 的**正常构成**（指数持续纳入新上市公司），不是数据缺陷。**换言之：只要
universe 里有一只在窗口内上市的票，pooled 加载就一定打到 floor。** 对 CSI500 五年窗口，
这是必然而非可能。

## 二、代价（实测 + 线性外推，外推处已标注）

**磁盘**（精确，遍历 month 分区）：

| 范围 | 分区文件 | 字节 |
|---|---|---|
| 评估窗口 2021-07..2026-06 | 57,542 | **8.18 GB** |
| floor 深度 2015-01..2026-06 | 117,971 | **16.43 GB**（2.01×） |

**内存/耗时**（20 票**实测**，× 49.75 **外推**——单帧 concat 的线性外推在这里是保守的，
真实 concat 峰值更高）：

| 加载 | 实测（20 票） | 外推（995 票） |
|---|---|---|
| 评估窗口 | 5.69M 行 / **1.06 GB** frame / 峰值 RSS 2.50 GB / 5.3s | 283M 行 / **52.7 GB** / ~4.4 min |
| floor 深度 | 13.21M 行 / **2.46 GB** frame / 峰值 RSS 5.57 GB / 12.7s | 657M 行 / **122.2 GB** / ~10.5 min |

本机 `free`：总 62 GB，**可用 56 GB**。

- floor 深度 **122.2 GB / 56 GB = 2.19×** → **OOM，不可行**。
- 评估窗口 **52.7 GB / 56 GB = 0.94×** → 名义装得下，但那是**因子计算开始之前**的裸 bar
  帧，且 8 个 pooled 因子每个都要一次。**实际同样不可行。**

**为什么旧 runner 没这个问题**：11 个旧 runner 的 `_load_*_panel` **逐 symbol** 读、读完
立刻聚合到日频，**从不同时持有整个 universe 的分钟 bar**。D4 的 materializer 改成了整批
加载——这是 D4 引入的新形态，在 40 票上看不出来。

## 三、修法：per-symbol 流式化，且切口必须在 cross-section combine **之前**

**实测的决定性证据**（12 票真实数据，整 universe 算 vs 逐 symbol 算再拼）：

| 因子 | index 相同 | 可比 cell 数 | max\|diff\| | NaN 集合差异 | 判定 |
|---|---|---|---|---|---|
| `ridge_minute_return_20`（pooled） | True | **728** | **0.000e+00** | **0** | **SAFE** |
| `volume_peak_count_20`（pooled） | True | **1524** | **0.000e+00** | **0** | **SAFE** |
| `intraday_amp_cut_10` | True | **0** | **n/a（无可比 cell）** | **1692（全部行）** | **NOT SAFE** |

前两个在 728 / 1524 个**真实可比 cell** 上逐位相同——per-symbol 纯性在真实数据上得到确认
（不是靠既有 isolation 测试推断，也不是在空集合上真空通过）。

⚠️ 第三行的 `max|diff|` **必须显示 `n/a` 而不是 `0.000e+00`**（评审 NIT，已改）：per-symbol
一侧全是 NaN，两边**没有任何一个 cell 可比**，而"0.000e+00"紧挨着 "NOT SAFE" 会被读成
「值其实一致、只是 NaN 标记不同」——恰好相反。`compared=0` 这一列就是为了让"没得比"
无法被误读成"比过了且相等"。

`intraday_amp_cut` **全部 1692 行翻成 NaN**：它的第 4 步是**按日截面 z-score**，要求当日
finite 对数 ≥ `AMP_CUT_MIN_CROSS_SECTION=10`，而**单票截面 n=1 恒 < 10 → 定义上全 NaN**。
这正是 D4 docstring 警告的耦合，实测复现。

**因此切口在哪里是有对错的**：

- ❌ 把**整个因子**逐 symbol 算再拼 → `intraday_amp_cut` 全毁。
- ✅ 逐 symbol 算到**该因子的 per-symbol 中间量**，拼成全 universe 面板，**再做那一次
  截面 combine**。`intraday_amp_cut` 的模块结构本来就是这样组织的
  （`factors/compute/minute/intraday_amp_cut.py` docstring：先出两列 `(V_mean, V_std)`
  面板，**再 `combine_amp_cut_cross_section` 一次**），旧 runner 亦然。

### 与 `AMP_CUT_MIN_CROSS_SECTION` 耦合的显式论证（lead 要求单独论证，不许当纯优化）

D4 的警告原文是「per-symbol 加载深度会改 `intraday_amp_cut` 的截面 z-score 组成」。
**该警告针对的是 per-symbol *截断*，不是 per-symbol *饱和***，两者必须分开：

- **per-symbol 截断**（每票只加载「它大概需要」的深度）：某票在日 d 本应 finite 却因加载
  过浅变 NaN → 当日截面成员变少 → z-score 分母变、`min_cross_section` 门可能翻 →
  **确实改值**。这条 D4 警告成立，**不可做**。
- **per-symbol 饱和**（每票加载到**它自己**的饱和/floor，判据仍是 D4 那条结构性判据，
  只是逐票判而非全 universe 齐步走）：每票的 per-symbol 中间量**等于**它在全 universe
  floor 加载下的值（上表前两行逐位实测 + 既有 cross-symbol isolation 性质）→ 日 d 的
  finite 成员集合**不变** → 截面 combine 输入不变 → **值不变**。

**即：把「全 universe 一起扩到 floor」换成「每票各自扩到自己的 floor」，对每票的
saturation *终点* 没有放松**（84 票里任何一票该到 floor 的仍到 floor），只是不再强迫
另外 911 票陪着一起到 floor。**这不是把判据放宽，是把判据从「universe 齐步」改成
「逐票各自」，而逐票判据的终点集合与齐步判据完全相同。**

⚠️ **仍属定义相邻、必须验收**：上述论证的承重前提是「per-symbol 中间量与加载几何无关」。
该前提对 10 个 per-symbol 纯因子已实测（0.0 逐位）；对 `intraday_amp_cut` 需要**单独**
证明其 `(V_mean, V_std)` per-symbol 部分同样与几何无关，并用**全 universe floor 加载
vs per-symbol 饱和加载**在小 universe 上做端到端逐格对账。**该对账未完成前，不得把
per-symbol 饱和当作已验证的等价改写。**

## 四、对 D5 计划的影响

1. **C5 的全量 run 在修好之前不可能执行**——不是「慢」，是 OOM。任何「先跑起来看看」
   都会在 10 分钟的加载后死掉，或更糟：在评估窗口那档勉强不死而开始 swap。
2. 修法（per-symbol 流式 + 切口在 combine 之前）**在 D4 已声明为 D5 工作**
   （`materialize.py:397` "A per-symbol saturation start is the D5 optimization"），
   故属 D5 范围，但**它是修改因子取值路径的改动，必须按 §五四腿验收**，
   不能作为「让 run 跑起来」的顺手改动混进去。
3. `valley_price_quantile` 的 D5 绑定（C4 交付物）**继承同一路径**——lead 已明令
   不许用固定深度 trim 绕过，per-symbol 饱和是它唯一合法的加载形态。

## 五、未做 / 留给后续（原文保留；第 1 条已由 D4b 完成，见第六节）

- **per-symbol 饱和的实现与四腿验收**：本探测只证明了「必须这么做」与「这么做在两个
  pooled 因子上逐位等价」，**没有实现它**，也没有对 `intraday_amp_cut` 做端到端对账。
- 声明 floor 目前仍是硬编码常量 `qt/factor_hotpath_smoke.py::CACHE_MINUTE_DATA_START
  = "2015-01-05"`，对所有 symbol 同值。**探测显示按票派生 floor 是有意义的**（668 票
  确实从 2015 开始，但 327 票晚得多），但**改 floor 派生方式本身**同样是定义相邻变更，
  与 per-symbol 饱和是两件事，不要合并处理。
- 本探测未测 `concat` 峰值与 GC 行为，线性外推对峰值是**乐观**的。

---

# 六、D4b 结果：修完之后的实测（本节为「后」，一至四节是「前」）

D4b 把 materializer 改成 **per-symbol 流式**：读一票 → 立刻聚合到该票日频中间量 →
丢弃 bars；**截面 combine 在装配完整 universe 面板之后跑一次**。因子数学、参数、阈值
（含 `AMP_CUT_MIN_CROSS_SECTION=10`）一个字未动。复测命令：

```
python -m qt.saturation_probe --mode stream-scale --factors <id> \
    --cache-root <...> --universe-panel <...> --out <...>.jsonl
```

一次一个因子、一个进程（`ru_maxrss` 才是该因子自己的峰值），JSONL 可续跑。

## 6.1 逐格对账：流式 vs 整 universe 单帧（真实缓存，40 票 × 6 个月）

`2023-01-03..2023-06-30`，参照系 = **被替换掉的那个几何**（全部票进一个 frame、一次
cutoff、一次整因子调用），单帧 3,711,400 行 / 0.69 GB；`live_calls=0`。

| 因子 | index 相同 | 可比 cell | NaN 集合差异 | max\|abs\| | max\|rel\| |
|---|---|---|---|---|---|
| `jump_amount_corr_20` | True | 4,720 | **0** | 3.275e-14 | 2.387e-13 |
| `minute_ideal_amp_10` | True | 4,720 | **0** | **0.000e+00** | 0.000e+00 |
| `amp_marginal_anomaly_vol_20` | True | 4,720 | **0** | **0.000e+00** | 0.000e+00 |
| `volume_peak_count_20` | True | 4,720 | **0** | **0.000e+00** | 0.000e+00 |
| **`intraday_amp_cut_10`** | True | 4,720 | **0** | 2.531e-14 | 2.155e-12 |
| `peak_interval_kurtosis_20` | True | 4,720 | **0** | **0.000e+00** | 0.000e+00 |
| `valley_relative_vwap_20` | True | 4,719 | **0** | 2.220e-16 | 2.232e-16 |
| `valley_ridge_vwap_ratio_20` | True | 2,295 | **0** | 2.220e-16 | 2.246e-16 |
| `ridge_minute_return_20` | True | 2,283 | **0** | **0.000e+00** | 0.000e+00 |
| `peak_ridge_amount_ratio_20` | True | 2,247 | **0** | 3.331e-16 | 4.943e-16 |

**NaN 集合差异全为 0** 是这张表里最承重的一列：finite↔NaN 翻转是覆盖率变化，不是浮点
现象，本步绝不允许。5/10 逐位相同；其余 5 个残差 ≤3.3e-14，是 pandas 滚动累加在不同
长度前缀上的重排（D2/D4 已编目的可归因项），比任何真实缺行效应低若干个数量级。

`intraday_amp_cut_10` 在 4,720 个真实 cell 上对上，**这是第三节留下的那笔账**：它是
唯一 pooled × 截面双耦合的因子，切口放错（按整因子逐票切）会让它全 NaN——该负对照已
committed 为测试，不是一次性观察。

## 6.2 规模证明：真实评估面（CSI500 995 票 × 5 年）

**名义与峰值分开报**——一至四节实测 frame:峰值RSS = 2.2–2.45×，只报名义会把需求少说
一半。以最危险的因子 `intraday_amp_cut_10`（唯一 pooled × 截面双耦合）为例：

| | 名义最大单帧 | 峰值 RSS | 可用内存 | 判定 |
|---|---|---|---|---|
| 旧：整 universe 单帧（floor 深度） | **122.2 GB** | **~115 GB**（评估窗口档）| 56 GB | **装不下** |
| 新：per-symbol 流式 | **0.0656 GB** | **0.964 GB** | 56 GB | **可跑（1.7% 内存）** |

峰值 RSS 降到 **约 1/119**，且**不随 universe 规模增长**——引擎每次只向 provider 要
**一票**（`max_symbols_per_call=1`，测试锁定，mutation 实证）。

## 6.3 每因子实测（`--mode stream-scale`，cache-only，一因子一进程）

CSI500 995 票 / `2021-07-01..2026-06-30`：

**三种加载几何各取一个**（这是完整的几何覆盖，不是抽样）：

| 因子 | 几何 | 输出行 | 有值 | 出现的票 | 每次请求票数 | 名义最大帧 | **峰值 RSS** | wall | live_calls |
|---|---|---|---|---|---|---|---|---|---|
| `intraday_amp_cut_10` | pooled × **截面** | 1,158,842 | 1,158,842 | **995/995** | **1** | 0.0656 GB | **0.964 GB** | 1990.5 s | **0** |
| `ridge_minute_return_20` | pooled | 593,245 | 592,431 | **995/995** | **1** | 0.0713 GB | **0.854 GB** | 1133.7 s | **0** |
| `jump_amount_corr_20` | bounded | 1,159,263 | 1,158,552 | **995/995** | **1** | 0.0658 GB | **0.730 GB** | 417.7 s | **0** |

**995/995**：没有任何一只被请求的票在流式化后消失（I5a 红线）。峰值 RSS 落在
**0.73–0.96 GB**，与因子和输出规模基本无关——峰值由「一票的 bars + 日频面板」决定，
不由 universe 决定。`live_calls=0` 全程 cache-only。

wall 按几何排序符合预期：bounded 无扩张（418 s）＜ pooled 20 chunk（1134 s）＜ pooled
60 天 chunk（1990 s，见 6.4）。**剩余 7 个因子未跑**——那已经是 D5 的全量 run 本身，
不是 D4b 的验收；同一命令可续跑，JSONL 里已记录的因子会被跳过（实测 SKIP 生效）。

⚠️ **wall 是上界，不是净耗时**：这台机器同时在跑其它验收作业（逐格对账、手算锚、
pytest），20 核，争用主要在磁盘。**内存数字不受影响**（`ru_maxrss` 是各进程自己的
高水位），wall 应读作「在有争用的条件下不超过这个数」。

## 6.4 代价结构：钱花在哪 84 只票上（实测，非推断）

`intraday_amp_cut_10`，各取 10 票，同 5 年 emit 窗口：

| 票的类型 | provider 调用/票 | wall/票 | 单票最大 frame |
|---|---|---|---|
| dense（2015 上市） | **1.0** | **0.59 s** | 0.066 GB |
| late（首根 bar ≥ emit_start） | **40.0** | **14.95 s** | 0.063 GB |

dense 票在**第一个 chunk 就饱和**；late 票攒不出 emit_start 之前的 10 个有效日，于是
一路扩到声明 floor，**每个 chunk 重读一遍它自己的全部历史**。40 次正是
`_saturation_chunk_calendar_days = warmup*2+40`（amp_cut 为 60 天）除 2015→2021 的
2368 天所得（预测 39.5，实测 40.0）。

**这正是 D4b 买到的东西**：扩到 floor 的代价现在**只落在这 84 只票上**，而不是拖着
另外 911 只一起走——旧规则下 995 票全都要付 floor 深度的加载，而那个加载**根本装不下**
（一至四节）。

⚠️ **步长是可以改的，但本步刻意没改**：D4 的 docstring 明写 chunk "only a STEP SIZE,
never a correctness input"，所以把它调大或改成几何增长是**纯性能**改动，能把这 40 次
大幅压低。但它仍然是取值路径的加载几何改动，需要自己的逐格对账，**不能夹带进 D4b**。
留给 D5 作为显式的性能项。
