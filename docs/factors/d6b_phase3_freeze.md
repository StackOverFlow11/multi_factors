# 因子层重构 D6b —— phase3 legacy 基线捕获（d6b_phase3）

> **状态**：D6b PR-1 的交付物，D6b PR-2（oos/subset 切 FactorService）对账的**唯一基线**。
> 机器可读权威 manifest 是入 git 的 `docs/factors/d6b_phase3_manifest.json`（123 条 sha256）；
> 冻结字节在 `artifacts/refactor_baseline/d6b_phase3/`（gitignored，不入 git）。
> 既有四个冻结目录（`panels/panels_d2/exec_baseline/pr_c_cutoff_fix`）全程未被触碰。

⚠️ **前置条件**：需要 gitignored 的 `artifacts/` 树（worktree 里建符号链接指主 checkout）。

---

## 一、捕获了什么

D6b 要把 oos/subset/robustness 三条 phase3 runner 切到 FactorService。切换对账的基线必须
是 **legacy 引擎 + 确定性数据面** 的输出。捕获分三层：

| 目录 | 内容 | 角色 |
|---|---|---|
| `L0/` | live legacy（原配置、无 cache 块、直抓）三条 runner 的全精度结果 JSON + 报告 | **vintage 诊断**（不做对账基线） |
| `L1/` | cached legacy（twin 配置指向冻结快照、`fina_tail_days: 0` 全隔离）三条 runner 结果 + **panels 腿产物** | **切换对账基线** |
| `L1p/` | 同快照同代码第二遍 | legacy-vs-legacy 确定性对照 |
| `L1_prefix_tiebreak/` | fina tie-break 修复（PR #122）**之前**的 L1 | 留档（修复的历史痕迹） |
| `L1_tail400/`、`L1p_tail400/` | 修复后但 fina tail 未封（仍打 live）的 L1/L1p | 留档（被全隔离版取代） |
| `L1/panels/` | 面板腿：逐 cell 的 legacy/served 面板 parquet + 逐格对账 JSON | 服务路径 ≡ legacy 的值级证据 |

panels 腿：同一 enriched panel 上 legacy `factor.compute(panel)` vs 服务路径
`factor_values(...)` 逐格比对，判定 `max_abs_diff = 0.0`、NaN-mask 失配 = 0、索引集合差 = 0
（允许差异类为空；close view 下改的是路径不是值）。

## 二、数据面（冻结快照与隔离口径）

- 快照：`tmp/context/d6b/cache_snapshot_daily/`（**不入 git、不进 refactor_baseline**）——
  共享缓存的**日频 11 端点**（分钟数据不含，phase3 不需要）在某次 warm 后的 `cp -a`。
- 隔离口径（两个旋钮都在 twin 配置里，属**运行期本地改动、不入 commit**）：
  ① `cache.root_dir` 指快照；② **`fina_tail_days: 0`**——fina 的 400 天 trailing-tail 策略
  默认每次跑都重抓上游，会把快照外的 vintage 混进来；置 0 后接受既有覆盖、零 live 调用。
- **隔离实证**：最终 L1/L1p 的 runner 日志**全部 10 个端点 gap-fetch 计数为 0**；
  `L1 ≡ L1p` 全叶子 0 差异（确定性对照）。

## 三、捕获过程中的两次实弹（为什么基线重做过）

1. **fina tie-break 缺陷（已单独成 PR #122）**：首轮 panels 腿 CSI300 格发现
   roe/netprofit_yoy 1,364+1,286 格失配（max 793.6）。根因 = `pit_financials.py:49` 的
   quicksort 不稳定去重：同 (symbol, ann_date) 多报告的去重结果随 frame 构成漂移
   （SSE50 frame 选年报、CSI300 frame 选 Q1），被共享 factor store 的 first-writer-wins
   放大成跨格矛盾。修复（最新 end_date 恒胜）后重捕获：**panels 腿 7/7 cell 全 0.0**。
2. **fina tail 泄漏（见上 §二②）**：首轮 L1/L1p 的 fina gap-fetch 非零，虽 L1≡L1p 当时
   成立，但 L1 与切换后重跑跨天时上游修订会成假警报源 → twins 加 `fina_tail_days: 0`
   后再次重捕获（本次为最终基线）。

## 四、L0 vs L1 的归因（L0 只作诊断）

L0（live、修复前 tie-break）vs L1（快照、修复后）的叶子差异
（oos 91 / robustness 310 / subset 852）= **vintage 漂移 + tie-break 重述** 的混合，
属预期内（D6a-5 先例：归档数字随上游修订漂移）；L1 才是切换基线。tie-break 重述幅度
已在 PR #122 量化（SSE50 5.12% / CSI300 5.11% 格）。

## 五、验证

```bash
R=<有 artifacts 树的 checkout>
PY=/home/shaofl/Development/env_tools/envs/quant_mf/bin/python
# 123 条 sha256 逐条核对（脚本与 exec_baseline 同款逻辑）
$PY - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("<R>")
man = json.load(open(root / "docs/factors/d6b_phase3_manifest.json"))
base = root / "artifacts/refactor_baseline/d6b_phase3"
bad = [f for f, h in man["files"].items()
       if hashlib.sha256((base / f).read_bytes()).hexdigest() != h]
print(f"{len(man['files']) - len(bad)}/{len(man['files'])} OK" if not bad else f"MISMATCH: {bad}")
EOF
```

**对账纪律**：D6b PR-2 的 S1/S2 只允许读 `L1/`（经 manifest 验哈希），绝不读 live 路径；
L0/L1p/留档目录不参与切换对账。
