# 因子层重构 D5 —— exec 基准评估 artifact 冻结

> **状态**：D5 第一个交付物（C1），**先于任何 D5 代码改动执行**。
> 设计出处：v3.2 §五第 3/4 腿（语义对账的比较对象）、§九 D5 行。
> 本文是 provenance 叙述；**机器可读权威 manifest 是入 git 的
> `docs/factors/d5_exec_baseline_manifest.json`**（77 条 sha256）。
> 冻结的字节在 `artifacts/refactor_baseline/exec_baseline/`（gitignored，不入 git）。
>
> ⚠️ **前置条件**：本文所有命令都需要 gitignored 的 `artifacts/` 树。**在没有它的 worktree 里
> 开箱跑不起来**——需先对主 checkout 的 `artifacts` 建符号链接，或用 `--repo-root` 指向
> 一个有该树的 checkout（先例：`qt/factor_hotpath_smoke.py` docstring 的同类说明）。
> 建符号链接时注意：若 worktree 里已存在真实 `artifacts/` 目录，`ln -s` 会嵌套成
> `artifacts/artifacts`（本步实际踩到过）。

---

## 一、为什么必须先冻结（读这段再动任何东西）

D5 要在删掉 11 个旧 runner 之前证明新引擎没有悄悄改变任何东西。唯一能证明的办法是拿新
runner 的输出和**旧 runner 的输出**比。而旧输出：

1. **只有一份**，在 `artifacts/reports/`，**gitignored、只存在于本机**；
2. **不可从当前代码再生**——需要 pre-D5 的树 + 约 1.5 小时真实 run；
3. 设计 §五 提到的耐久副本 `scratchpad/baseline_reports/` **已不存在**（旧会话 scratchpad，已清）；
4. **新 runner 沿用同名输出路径**。

即：新 runner 跑一次就会覆盖唯一的基线，随后的"对账"会比对刚被自己覆盖的文件——正是
CLAUDE.md 记录在案的 `compare_postmerge.py` **空对账**失效模式（拿新结果和自己比，
构造性恒真，`ic=1.0` vs `ic=999.0` 也会打印"精确复现"）。

**所以顺序不可交换：先冻结，再改代码。** 本 commit 不含任何 D5 引擎代码。

## 二、冻结了什么（77 个文件，按因子列表派生而非按目录扫描）

| 类别 | 数量 | 说明 |
|---|---|---|
| `eval_{factor}_exec_{book}.json` | 22 | **对账的权威机器可读源**（IC / ICIR / 分位价差 / verdict 全在此） |
| `eval_{factor}_exec_{book}.md` | 22 | 渲染报告 |
| `eval_{factor}_exec_basis_sanity.md` | 11 | #79 的覆盖率/偏差披露（含 `coverage_bias_bad_vwap` 原话） |
| `eval_{factor}_exec_{book}_dashboard.png` | 22 | 单图 dashboard（16MB，同样不可再生） |

11 因子 × 2 book。清单**由因子列表派生**（`expected_artifact_names()`），不是目录扫描——
少一个文件是**可读报错**，不是一个静静变小的基线。多一个未知成员同样报错。

> 注：`_exec_basis_sanity.md` 与 dashboard 不在 lead 交办的「22 json + 22 md」里，是本步
> 清点目录时发现的同族产物。它们同样单份、同样不可再生，故一并冻结——**冻结的成本是一次
> 文件拷贝，漏冻的成本是永久丢失**。

**未冻结**：close 基准的 **66** 个 artifact（22 json + 22 md + **22 png**）。它们不是 D5 的对账
参照（统一 runner 是 exec-only，决策 4），且未被新 runner 的输出路径覆盖，原地不动。
（本数曾误写 44——**漏的正是 dashboard**，而上一段刚说明 exec 侧发现 dashboard 才一并冻结，
close 侧没回头改数。同一次清点里，同一类东西数了两遍、两次结果不同。）

## 三、Provenance（观察到的事实 vs 归因）

**观察到的**（可复验）：

- 77 个 exec 文件的 mtime **全部落在同一秒 `2026-07-21 15:33:20`**（实测跨度 **9.46 ms**）。
- 对照：66 个 close 基准 artifact 里，**22 个 JSON 的 mtime 呈 11 个递进秒值**
  `09:24:33 → 09:32:45 → … → 10:43:54`——每因子一个（同因子的 no_book/with_book 同秒），
  正是一次真实 11 因子 run 的写入签名，且与 CLAUDE.md 记录的 #74 合并后全量重跑窗口
  （09:24:33→10:43:54）**逐值吻合**。
  ⚠️ **「11」是按 JSON 计的**；全部 66 个文件（含 md/png）共 **27 个不同秒值**，因为每因子
  的多个产物写在相邻但不同的秒里。引用本条时必须带口径，否则 11 与 27 会看起来矛盾。

**推论**：exec 家族**不是**在 `artifacts/reports/` 里跑出来的，而是**被整体拷贝进来的**
（一次 1.5h 的 run 不可能让 77 个文件落在同一个 9.46ms 窗口里；对照组恰恰不是这样）。拷贝时刻在 PR #79 合并
（`b04c966`，15:25）之后 8 分钟——与「#79 在独立 worktree 里跑完、合并后把产物拷进主 checkout」
的形态一致。

**归因（非观察）**：内容归属于 PR #79 的 exec 基准 run。这是**从 PR 记录推断的**，不是从
文件系统观察到的。本文如此区分，是因为 #79 的 code review 未返回、靠自查合并（CLAUDE.md
已记录），其 provenance 本就该按证据强度分级陈述，而不是写成"由 SHA X 产生"。

**冻结时刻**：`frozen_at_git_head = 45c14aa`（D0–D4 已合并的 `main`）。这是**冻结动作**发生
的 SHA，**不是产生 artifact 的 SHA**——manifest 字段名即为 `frozen_at_git_head`，不叫
`producing_sha`，避免 D1 manifest 那种「producing SHA」措辞被误读。

## 四、结构性隔离：对账工具为什么读不到 live 路径

lead 的要求是「用路径断言或结构性隔离保证，不靠纪律」。实现为三层，**第 2 层承重**：

| # | 机制 | 拦得住什么 |
|---|---|---|
| 1 | 读取器要求一份 git-tracked manifest 描述该目录 | 随手指向 `artifacts/reports/`（那里没有 manifest） |
| 2 | **每次读取都对该文件 sha256 校验 manifest** | **把 live 产物拷进冻结目录**——它问的不是"路径看着对不对"，而是"字节是不是冻结的那些字节" |
| 3 | 目录若解析为 `artifacts/reports` 直接拒绝 | 显式误用，早失败早报错 |

**为什么 (2) 有牙**：字节在 gitignored 的 `artifacts/` 下，哈希在 git 里。覆盖冻结树的人
无法同时悄悄移动哈希——哈希只能经一次会被 review 的 `git diff` 变动。

**mutation 证据（实跑，非声称）**：

| mutation | 结果 |
|---|---|
| 读取器指向 `artifacts/reports/` | **RAISED** `refusing to read the LIVE artifacts directory` |
| 冻结文件末字节 XOR 0x01 | **RAISED** `on-disk sha256 … != manifest …`；`verify_all` **76 ok / 1 problem**；同目录未改的兄弟文件仍正常读出（守卫是定向的，不是一刀切） |
| 冻结后改 **live** 源文件再 freeze | **RAISED** `differs from the live artifact`，且冻结副本**逐字节未动** |
| 少一个 / 多一个清单成员 | **RAISED**（分别 `absent` / `unexpected`） |
| **清空 manifest 的 `source_note` + `frozen_at_git_head`** | **rc=1 FAILED** `test_committed_manifest_describes_the_real_frozen_baseline`（新增的恒跑 provenance 断言；**修复前 16 个测试对此全绿**） |
| **把一个冻结文件塞进 git 索引**（`update-index --cacheinfo`，绕开 symlink） | **rc=1 FAILED** `test_frozen_bytes_are_untracked_…`；索引内 artifacts 路径 **0→1→0** |
| 在未改动的树上跑 `freeze`（真实执行） | `copied: 0 / already ok: 77 / manifest: **unchanged**`，manifest porcelain **0 行** |

⚠️ **本步第二次空 mutation（照实记录）**：上表倒数第二行的**第一版**用 `git add -f`，
但 git 对符号链接下的路径报 `beyond a symbolic link` 而**拒绝执行**——mutation 什么也没改，
测试于是"通过"，且是**为错误的理由通过**。改用 `git update-index --cacheinfo` 直接写索引
（不走文件系统、不受 symlink 影响）才真正把它变红。**同一份工作里两次写出不会失败的
mutation**，都靠"预期与观察不符"而非靠测试本身抓住——见 §四末与本行。

⚠️ **记一次本步自己的失误（照实记录，与项目既有先例同格式）**：tamper 测试的第一版把
`b'"ic"'` 替换成 `b'"IC"'`，而该字节串**在文件里根本不存在** → `.replace()` 是 no-op →
"mutation" 什么也没改，测试对着一个从未被触发的守卫打印通过。它之所以被抓住，只因为同一次
输出里 `verify_all` 报了 `ok=77 problems=0`，与"我刚破坏了一个文件"的预期矛盾。
**这正是 §六.10「不变性测试没有 mutation 证据」要防的形态，而本步亲手造了一个。**
现版本对 mutation 本身加了断言（`assert after != before` 且 sha256 必须不同），
使"mutation 是空的"变成测试失败而不是测试通过。

## 五、复验命令

```
# 校验冻结树与 manifest 一致（不拷贝任何东西）
python -m qt.exec_baseline_freeze --verify

# 再次冻结：**字节幂等**——树没变则一个字节都不写（含 manifest）；内容不同则拒绝并保留原副本
python -m qt.exec_baseline_freeze
```

⚠️ **「字节幂等」这个限定词是修出来的，不是一开始就对**（评审 HIGH，照实记录）：初版
`freeze()` 在末尾**无条件重写 manifest**，与本轮有没有拷贝文件无关，且 `source_note` 来自
CLI 参数、默认空、**不从既有 manifest 继承**。于是在一棵完全没变的树上跑上面这条本文档
**主动教人跑**的命令，会打印 `copied: 0 / already ok: 77`（纯成功），而 git diff 里
`frozen_at_utc` / `frozen_at_git_head` 已改指向重跑者、**§三整段 provenance 叙述被清空**——
**坏掉的恰是本 PR 最下功夫保护的东西**，而且 16 个测试在它被清空后全部通过。

重写后的 `frozen_at_git_head` 还是一个**主动的假声明**（它会宣称基线冻结于重跑者的 HEAD），
正是本文 §三末尾专门解释过要避免的那种误读——同 #76/#78/#82 形态。

**修法**：provenance 三字段从既有 manifest **继承**（显式 `--source-note` 才覆盖）；
内容不变则**完全不写**。现在 `--verify` 与 `freeze` 都不会改动 provenance。

冻结/校验工具：`qt/exec_baseline_freeze.py`；测试：`tests/test_exec_baseline_freeze.py`。
读取器 `FrozenExecBaseline` 是 D5 对账**唯一**支持的入口——对账代码不得自己 `open()`
`artifacts/reports/` 下的任何文件。

## 六、清单摘要

完整 77 条 `(name, sha256, size, source_mtime_utc)` 见
`docs/factors/d5_exec_baseline_manifest.json`。抽样：

| 文件 | sha256（前 12） | bytes |
|---|---|---|
| `eval_amp_marginal_anomaly_vol_exec_no_book.json` | `6755304f05c7` | 19487 |
| `eval_amp_marginal_anomaly_vol_exec_no_book.md` | `2300300c7d26` | 16286 |
| `eval_amp_marginal_anomaly_vol_exec_basis_sanity.md` | `4856188e2674` | 3892 |
