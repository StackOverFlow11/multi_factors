# 因子层重构 D1 末面板冻结 manifest：收官 14 因子 raw 因子值基线

> **状态**：D1 末交付物（设计 v3.2 §五第 4 腿 + §九 D1 行；背景 `tmp/design/review_round3_v32_revision_list.md` R11）。
> 本文是 **入 git 的权威 manifest**；bulk 面板与机器可读 manifest 在
> `artifacts/refactor_baseline/`（gitignored，不入 git）。
> **产物意义**：`main` @ producing SHA 的因子数学与重构前逐位一致（D1 registry 是 dispatch-only）。
> 此后 D2 改写数学，本基线是 **D5 逐格对账的唯一比较对象**。

---

## 一、Provenance（先读这段再动任何再生成）

- **producing SHA**：`3669c9068cd53dc684b75e0edb0f94346384808f`（= `main` @ D1 registry 合并后）。
  两段 run（见 §四）期间 worktree `HEAD` 均为该 SHA 且 tracked 树零修改——冻结工具
  `qt/panel_freeze.py` 当时以 **untracked 新文件**存在，因子/runner/pipeline 模块全部
  逐位等于 `main@3669c90`。
- **provenance 规则（v3.2 §五第 4 腿，原文）**：**基线再生只许从钉住的 pre-D2 SHA checkout，
  绝不从当前代码**——这是 `compare_postmerge.py` 空对账（拿新结果和自己比，构造性恒真）
  的结构性预防。具体操作：checkout `3669c90`，把本分支的 `qt/panel_freeze.py` 原样放入
  （它只 import、不改任何因子数学模块），再跑下方命令。**任何在 D2 之后的树上直接重跑
  本工具得到的"基线"都不是基线**，与冻结哈希不一致时以本文记录的哈希为准。
- **重跑命令**（cwd = 仓库根，缓存根就位）：

  ```
  /home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.panel_freeze
  ```

  输出根默认 `artifacts/refactor_baseline`（`--output-root` 可改；`--resume` 语义见
  模块 docstring——已存在的面板从冻结文件读回并**重新走 process + 对账**后才被接受，
  绝不盲信旧文件）。

## 二、数据面（与十一因子评估循环同面）

| 项 | 值 |
|---|---|
| universe | CSI500 `000905.SH`，PIT 成分（73 快照，996 distinct 成分） |
| 窗口 | 2021-07-01 .. 2026-06-30（1210 个交易日） |
| config | `config/phase_c_jump_amount_corr.yaml`（已程序化验证：11 个 eval config 除 `project.name`/`data.output_name` 外**逐字段一致**，任取其一同面） |
| 缓存根 | `artifacts/cache/tushare/v1`，cache-only |
| 日频端点 | 10 端点 gap-fetch **全 0**（run log `data cache:` 行：market_daily/adj_factor/index_weight/suspend_d/namechange/stk_limit/stock_basic/daily_basic/fina_indicator/index_member_all） |
| `stk_mins_live_calls` | **0**（分钟读走 `IntradayParquetStore.read_range`，无 fetch 闭包，构造性为零；每因子 loader 的 `live_calls` 计数逐一断言为 0） |
| PanelStore | 写为冻结专名 `d1_panel_freeze_daily`（内容与 runner 面板同源同参，不覆盖任何已验收 eval run 的 artifact） |

## 三、冻结口径（什么被冻结、什么不被冻结）

- 冻结 **raw 因子值**（处理前）。分钟因子 = 各 runner 私有 `_load_*_panel`（逐 symbol
  cache-only 1min 读 → `data/clean` `compute_*` → 日频聚合）后按 runner 原样
  `factor[dates.isin(panel_dates)]` 限制到日频交易网格；书因子 = runner 的
  `_build_book_factors()` + `factor.compute(enriched_panel)`。**零公式重抄**：loader、
  compute、常数全部从 runner 模块 import（同一对象，非转录）。
- `_process_factors`（zscore + 行业/市值中性化）**不在冻结值内**——它是共享机器不是因子
  数学；但在对账中被调用（见 §五）。
- **canonical content hash**（权威指纹）：sha256 over 版本标签 + 行数 + 排序后
  (date,symbol) 的 int64-ns 日期字节 + `\x1f` 连接的 symbol utf-8 + float64 原始字节
  （NaN 归一为单一位型；±0.0 保持可区分）。定义与实现在 `qt/panel_freeze.py`
  （`canonical_content_hash`）。文件 sha256 仅为便利指纹（受 parquet writer 元数据影响）。

## 四、运行记录（两段，诚实披露）

| 段 | 时间（本地） | 内容 |
|---|---|---|
| 首段 | 2026-07-23 18:05:57 → ~19:05:54（被会话 harness 在 ~60min 处杀死） | 面板加载 + 3 书因子 + 9 分钟因子（jump…ridge_minute_return）全部冻结并对账通过 |
| resume 段 | 2026-07-23 19:09:08 → 19:35:24（wall **1576.1s**） | 书因子重算（哈希与首段**逐位一致**）；9 个已冻结面板**从文件读回重新 process+对账**（哈希与首段日志逐一一致）；补齐 valley_price_quantile_20 / peak_ridge_amount_ratio_20；确定性双跑；写 manifest |

⚠️ run log 留存披露：`_make_logger` 以截断模式打开日志（`qt/pipeline.py`，`mode="w"`），
故 `artifacts/logs/panel_freeze.log` 现只含 resume 段——首段的逐因子日志行未留存于盘。
这**不削弱**验收：① 工具对每个新建分钟因子的 `live_calls` 是 **raise-on-nonzero** 断言
（非零即中止，面板不会落盘），首段 12 个面板全部落盘本身就是 live_calls=0 的证据；
② resume 段把首段全部 9 个分钟面板**从文件内容**重新 process + 对账 + 重算 canonical
hash（哈希与首段实时监控记录逐一一致），3 个双跑对象另行端到端重建。resume 语义使
首段面板的验收从「写文件前对账过」升级为「**文件内容本身**重新对账过」。

## 五、验证义务结果

- **确定性双跑 3/3 OK**（`jump_amount_corr_20` / `valley_price_quantile_20` / `value_ep`）：
  重建走完整 loader/compute 链，canonical hash 与冻结值逐字节相同。其中 jump 与
  valley_price_quantile 为**跨进程**对比（resume 进程全新构建 vs 首段进程写出的文件内容），
  强于同进程双跑；value_ep 另有跨 run 佐证（首段与 resume 段独立重算同哈希
  `1404a68f…`）。
- **与既有 eval artifact 对账 11/11 全过**：对每个分钟因子，把冻结 raw 面板经 runner 自己的
  `_process_factors` 推回 evaluator 边界，与 `artifacts/reports/eval_*_no_book.json` 的
  `data_coverage` payload 逐字段比对——`panel_rows` / `evaluation_periods` /
  `symbols_evaluated` / `universe_symbols_declared` / `dropped_symbols_count`（整数精确相等）
  + `factor_nan_rate`（JSON writer 的 6 位舍入口径）。**6 字段 × 11 因子全部一致**；不一致
  即 raise、不入库（机制见 `qt/panel_freeze.py::reconcile_with_eval_artifact`）。
- **书因子无 JSON 覆盖字段可对账**（eval JSON 的 purity 段只有 anchor IC，无 book 行数/覆盖
  字段）——照实披露，不假装对过；其正确性证据 = 双跑 + 跨 run 同哈希 + 与面板网格的行数
  恒等（3 × 1,158,912 = 面板全网格）。
- **secret scan 0 命中**：17 文件（14 面板 parquet + manifest.json/md + run log）扫真实
  token 值与 secret 配置文件路径/键名标记，全无。

## 六、14 因子 manifest 表

（mean/std 为 float64 全精度 `Series.mean()` / `std(ddof=1)`，NaN 跳过；canonical hash 为权威。）

| factor_id | kind | rows | date_min | date_max | n_symbols | n_nan | mean | std | canonical_sha256 | file_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| value_ep | book | 1158912 | 2021-07-01 | 2026-06-30 | 996 | 141362 | 0.04807913635037822 | 0.050405247296850246 | 1404a68fc88778e78d47da1ed6375c39abec36244a29961c3316a74f0c042e76 | 3b3a8970fcfe51d9b221da79a8f001a03d622d37592fdd76d19275a5159164d4 |
| value_bp | book | 1158912 | 2021-07-01 | 2026-06-30 | 996 | 4028 | 0.5761035855649241 | 0.4768837399414735 | c2f4d0536f6adcee5475562a6aa4f2036073a3b07a1dae690aa22e7032e01c75 | ce582623bf7846e849dc12605cbdb0b1dc9a39ef1f229522309c57e720e4d9ff |
| volatility_20 | book | 1158912 | 2021-07-01 | 2026-06-30 | 996 | 19920 | 0.02480289343756926 | 0.012617262687296474 | 8d46a34c69852352f6aa063016f19b7a6f12b226c43dd880d58431f409f86d6b | 5f81b0a47a3672778a828160d5db4ab8de6e98d5aaea5f6d25ac3f030af5733c |
| jump_amount_corr_20 | minute | 1159263 | 2021-07-01 | 2026-06-30 | 995 | 880 | 0.5912439770493371 | 0.15727054230823656 | b6359f128c3f645672a7bb62e9f1903760fbf5c70e40842f02f097f9f43ccfb2 | ffea5d028ac2d8c28d3fd2befd5e7cf5d918bf6f68169030502437f4be8c2e5c |
| minute_ideal_amp_10 | minute | 1159263 | 2021-07-01 | 2026-06-30 | 995 | 3980 | 0.00015591462929114136 | 0.0006014530933985754 | b03c721eafe728fe9e68c4db55e1bbabbca386397312045b969b76078e092233 | 44c62ec64629bf3099dda6ad87eb45409db560b3f29a4ee602c6d0046c228bee |
| amp_marginal_anomaly_vol_20 | minute | 1159263 | 2021-07-01 | 2026-06-30 | 995 | 10931 | 0.008288440013596557 | 0.004227652811127039 | b189ecf6471e30e973a5cae4ba9c2f5410ca08caa74169cab1552aedb9c1fe1b | b293a9de4bf5063825e3aef55ebfc4bea6dc8addcbf1abcffe9d426bbc8fa198 |
| volume_peak_count_20 | minute | 1149313 | 2021-07-15 | 2026-06-30 | 995 | 8955 | 263.7242804452637 | 72.10585825271954 | b5e94aedc7c62332b3df399c4caa66ef7fc6e8cc2841c21903066e40ec11f2f0 | 6b5b9b7941b6f3456ae83bac3e707249617c7f382a1cef3228a4089f2186ee04 |
| intraday_amp_cut_10 | minute | 1154288 | 2021-07-08 | 2026-06-30 | 995 | 0 | 2.806990001685937e-18 | 0.8527302575863168 | 33d69c9a0cd3080b6489685211a8ebb3b4bbd2c1d27b487d4cd2627d06632ba0 | a48152e0e9ab79b294b2932183d7b373bf44c8c2116209f6f7e17aee0fbfe44d |
| peak_interval_kurtosis_20 | minute | 1149313 | 2021-07-15 | 2026-06-30 | 995 | 9567 | 8.522552350694497 | 5.613139688378547 | e8e337f21c56a727e2747f31e7f09a85e4c8c725e1d7f9134d90cb262232b624 | 71ad513f9a1af9cddc7fb75f428449963ae1946bf0e4a91e859517f2bf3d3daa |
| valley_relative_vwap_20 | minute | 1146878 | 2021-07-15 | 2026-06-30 | 995 | 8955 | 0.9993996386025606 | 0.0010562130843499646 | 0e3599ce151f099fb9278fb0c1ba5527f1383a906642e2cca01d1b32e8ea8e38 | 4780ad06b8053c1315256e0b44c43141a7877f19d80ce34d613a78c47867d9ea |
| valley_ridge_vwap_ratio_20 | minute | 591524 | 2021-07-15 | 2026-06-30 | 995 | 8955 | 0.9976201213193444 | 0.0028711378709016205 | e56b70f1bc95b02ffcbceabe827a47ab6c1e28a32535f07f6333c9378eb5a496 | c1194bdb0f5606d1bcf5b6da52cb31d940524015e88c79602cb2cc6608a738be |
| ridge_minute_return_20 | minute | 588536 | 2021-07-15 | 2026-06-30 | 995 | 8955 | 0.1622597430688057 | 0.14784924675685046 | b1f476f3e37b6bbf4ec0b5e563bb61cf520690e28d8cbcbe24deb61625cd32f6 | ea082f6e8a00e8c8a0e0ae46da433cd55d73431e83d9ac0d2a7eee3307cffc4d |
| valley_price_quantile_20 | minute | 1146878 | 2021-07-15 | 2026-06-30 | 995 | 10952 | 6.358786574311415e-19 | 0.027529790230469833 | 79f1219768ef95daf48484545658a4e307b0125a11e5fee28eb6c886ed5e422f | cee7ed6686ac4a44c72b7d3020beb418eb99e4194b2de75a58387cd3c1997481 |
| peak_ridge_amount_ratio_20 | minute | 579030 | 2021-07-15 | 2026-06-30 | 995 | 8955 | 0.3818472950505535 | 0.18604246679633485 | 8368be5d720184d4fa606cdfa5cf68bfa078179c4e27a196a933e42c26d0725c | 08ee6e3dc705bea03c66e87b566372f2966c3026d2fdb464e5e2d49bdded179b |

读数注（不是异常）：分钟因子共享同一分钟覆盖网格时 rows 相同（1,159,263）；带自身
classifiable/valley/ridge 门的因子 rows 少（peak/ridge 家族 ~57–59 万行是因子定义的
稀疏性，与 eval JSON 逐字段对账一致）；书因子 rows = 面板全网格 1,158,912。

## 七、D5 使用方式

D2 重写因子数学后，D5 在**同一数据面**重算 14 因子 raw 面板，与
`artifacts/refactor_baseline/panels/{factor_id}.parquet` 逐格对账；canonical hash 相同 ⇔
逐位一致，不同则用面板逐格 diff 定位。比较工具必须**独立读取两份产物**（不经共享中间
产物），并先核对本文 §一 的 producing SHA——防的是同一类空对账。
