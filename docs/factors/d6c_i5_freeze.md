# 因子层重构 D6c —— I5 intraday runner legacy 基线捕获（d6c_i5）

> **状态**：D6c PR-1 的交付物，D6c PR-2（`qt/intraday_tail_framework` +
> `qt/intraday_group_backtest` 切 FactorService）对账的**唯一基线**。
> 机器可读权威 manifest 是入 git 的 `docs/factors/d6c_i5_manifest.json`；
> 冻结字节在 `artifacts/refactor_baseline/d6c_i5/`（gitignored，不入 git）。
> 既有五个冻结目录（`panels/panels_d2/exec_baseline/pr_c_cutoff_fix/d6b_phase3`）全程未被触碰。

⚠️ **前置条件**：需要 gitignored 的 `artifacts/` 树（worktree 里建符号链接指主 checkout）。

---

## 一、捕获了什么

D6c 要把两条 intraday runner 切到 FactorService：tail runner（I5a/I5b/I5c/I5f，
`run_phase_i5a_intraday`）与 group runner（I5d/I5e，`run_phase_i5d_intraday_groups`）。
切换对账的基线必须是 **legacy 引擎 + 确定性数据面** 的输出。捕获分两条腿：

| 目录 | 内容 | 角色 |
|---|---|---|
| `L1/` | 6 配置 runner 腿全精度结果 JSON（NAV 全序列、event/feasibility/holdings 日志、分组 NAV/metrics/spread/monotonicity、limit 计数、liquidity 明细、score_coverage、factor_diagnostics）+ runner 报告 `.md` 快照 + I5d/I5e 图 | **切换对账基线** |
| `L1p/` | 同码同数据第二遍 + `*.vs_L1.compare.json` | legacy-vs-legacy 确定性对照（**全 6 配置 0 叶子差异**） |
| `L1/score_leg/<config>/` | score 腿：同一批已载 bars 上 legacy `_score_panel` 输出 vs 服务路径 serve 的逐格对账（`legacy_score.parquet` / `served_score.parquet` / `score_reconcile.json`） | 新注册因子 ≡ legacy hook 的**真数据值级证据** |

runner 腿的排除字段（与 D6b 同纪律）：`config`（携带 secret 路径）/ `elapsed`（墙钟）/
`report_path` / `log_path` / `figure_paths`（位置不是值）。其余全叶子捕获；序列化复用
`qt.phase3_capture.to_jsonable`（本次为它补了 bare `NaT`/`datetime` 叶子——I5 event log
的 exit 锚有 NaT，首轮捕获即踩中）。

score 腿：service 路径 = `factors.service.panel`（**逐 decision date 调用**——一次 range
fill 会把最外两个 anchor 之间的每个交易日都物化，正是 group runner anchor-sliced 设计
要避免的多年分钟读）+ `CacheMinuteProvider`（结构性零 live call）+ 共享
`open_factor_value_store`，`view=DECISION, basis=EXEC_TO_EXEC`， cutoff/session_open 取自
配置（与因子 spec 声明一致性有显式断言）。判定：`max_abs_diff = 0.0`、NaN-mask 失配 = 0、
legacy-only 索引 = 0、served-only 行全 NaN（store 的显式 footprint 行，有限值即发现）。

## 二、数据面证据（cache-only，逐 run 实证）

- **分钟读结构性只读**：两条 runner 的分钟读都无 fetch 闭包（tail 腿的 fetch 闭包是
  raise；group 腿直读 parquet store），每 run 结果 JSON 的 `minute_live_calls` 恒 0。
- **日频端点 P4 read-through**：harness 在进程内包装 runner 模块的 `_build_cache`
  （纯 passthrough 记录器，无行为变化），逐 run 取 `cache.stats()` 全端点差值；
  窗口尾（2026-05-31 / 2026-06-12）距今 ≥7 周，全部在 refresh 窗口之外。
- **实证结果（全部 12 次 runner 跑 + 6 次 score 腿）**：daily gap-fetches 全端点 0、
  `stk_mins_live_calls=0`、`stk_limit_gap_fetches=0`、score 腿 `provider_live_calls=0`。
  任一非零即作废并停下（driver 内置断言，未触发）。

## 三、判定

- **score 腿 6/6 verdict_pass**：`max_abs_diff = 0.0`、NaN-mask = 0、legacy-only = 0。
  逐配置对账格数：I5a `intraday_ret_0930_1450` 4,060；I5b 同 ret 4,060；I5f 同 ret
  4,060；I5c `intraday_mmp20_ew_0930_1450` 4,054；I5e 26,907（60 anchor 日 × CSI300
  覆盖名）；I5d 53,767（60 anchor 日 × CSI500 覆盖名）。
- **L1 ≡ L1p**：6/6 配置全叶子 0 差异。
- **锚复核（L1）**：I5b / I5f final NAV = 1.0193180689471217（锚 1.019318）；
  I5e annual Spearman = −0.5（锚 −0.5）；I5d annual Spearman = +0.9（锚 +0.9）。

## 四、捕获过程中的一次实弹

首轮 runner 腿在 JSON 序列化即失败：I5a event log 的 `exit_execution_ts` 含 bare
`pd.NaT`（DataFrame object dump 叶子），`to_jsonable` 只处理 `pd.Timestamp` 分支。
修复（bare NaT/datetime 叶子同样 ISO/NaT 序列化）落在共享的 `qt.phase3_capture`，
附网络无关单测；phase3 既有捕获语义不变（默认排除清单未动）。

## 五、验证

```bash
R=<有 artifacts 树的 checkout>   # worktree 需 artifacts 符号链接指主 checkout
PY=/home/shaofl/Development/env_tools/envs/quant_mf/bin/python
# manifest 每条是 {"sha256", "bytes"} 结构，逐条核对
$PY - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("<R>")
man = json.load(open(root / "docs/factors/d6c_i5_manifest.json"))
base = root / "artifacts/refactor_baseline/d6c_i5"
bad = [f for f, h in man["files"].items()
       if hashlib.sha256((base / f).read_bytes()).hexdigest() != h["sha256"]]
print(f"{len(man['files']) - len(bad)}/{len(man['files'])} OK" if not bad else f"MISMATCH: {bad}")
EOF
```

该脚本结构与 d6b_phase3_freeze.md 的验证脚本一致，已按同结构对 d6b manifest
实测（123/123 OK），非纸上脚本。

**对账纪律**：D6c PR-2 只允许读 `L1/`（经 manifest 验哈希）；L1p 是确定性对照，
不参与切换对账。
