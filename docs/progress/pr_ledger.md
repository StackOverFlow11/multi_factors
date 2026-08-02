# PR 台账（PR #1 – #107）

> 本文件是 `CLAUDE.md` 进度日志的**归档**，内容逐字搬移，未做任何改写。
> 索引见 [`docs/progress/README.md`](README.md)；当前状态与操作约定仍在仓库根 `CLAUDE.md`。

下面一段是 `CLAUDE.md` 开发约定里 **Git** 那一条的原文，逐字搬移。
**OPEN 未合并：#38**（见 P-I5d 处置，未获授权关闭，永远保持 OPEN）。

---

- **Git**：feature 分支 + PR。**PR #1（P0+P1）、#2（P2-1）、#3（P2-2）、#4（进度文档）、#5（P2-3）、#6（进度文档）、#7（P2-4）、#8（进度文档）、#9（P3-1）、#10（进度文档）、#11（P3-2）、#12（P3-3）、#13（进度文档）、#14（P3-4）、#15（进度文档）、#16（P3-5）、#17（进度文档）、#18（P3-6）、#19（P3-7）、#20（进度文档）、#21（P3-8）、#22（进度文档）、#23（P4-1）、#24（进度文档）、#25（P4-2）、#28（tushare 权限/限频探测 + capability registry）、#29（I1–I4 分钟级 intraday pipeline，4 commit 一 PR）、#30（进度文档）、#31（P4-3 因子支撑端点缓存 + 21:00 data updater，2 commit 一 PR）、#33（P-I5a 事件驱动回测架构重构 + opt-in 分钟尾盘 event model，4 commit 一 PR）、#35（P-I5b 分钟尾盘执行期 raw stk_limit 涨跌停可行性，4 commit 一 PR）、#37（P-I5c MMP 分钟因子端到端 opt-in alpha）、#39（P-I5d MMP 五分位分组回测 standalone，含 I5c plumbing）、#41（数据层 D1 契约文档 + token 解析去重）、#43（数据层 D2 TushareCache specs/parsers 拆分）、#45（数据层 D3 report-only 数据质量层）、#47（数据层 D3b default-off `data-update` 质量报告钩子）、#49（数据层 D4 coverage ledger 批量写 + 进程内查找缓存）、#51（数据层 D5 opt-in 有界并发 + 全局限频器）、#53（P-I5e MMP 五分位 CSI300 独立泛化检验，负结果）、#55（P-I5f 分钟尾盘执行流动性/容量诊断，report-only）、#57（数据层 schema 注册表 + default-off drift 守卫）、#59（数据层 全A 增量 auto-warm + systemd 定时，default-off）、#61（数据层 全A 历史回填 `data-backfill`，分块/续跑/失败隔离）、#62（进度文档）、#63（因子评估契约层 `analytics/eval/` + StandardFactorEvaluator + dashboard + PR-C 因子，4 commit 一 PR）、#64–#73（PR-D…PR-M 十个分钟因子复现，**一因子一分支一 PR**）、#74（因子评估契约 v0.8 + v0.9 判定门修复）、#75（尾盘执行价 VWAP + 分钟持有期收益复权）、#76（报告披露修正 + I5f 名义按真实规模重测）、#77（`adj_factor` 下降质量检查）、#78（#76 的 follow-up：扫描式正则守卫被七种改写实测证否 → 闸门描述改单一来源）、#79（十一因子改 14:51 VWAP exec-to-exec 基准，**评审未返回、靠自查合并**）、#82（闸门/成交价描述全仓清扫，12 处；#78 守卫射程不足的修正）、#80（进度文档）、#84（因子层重构 D0 纸面契约：availability policy 常量 + 预指派表 + 性质编目）、#85（进度文档）、#86（因子层重构 D1：registry + PanelField requires + FactorSpec 契约 v1.0）、#87（因子层重构 D1 面板冻结：14 因子 raw 基线 + manifest）、#88（进度文档）、#89（因子层重构 D2：minute primitives + ops + 11 因子迁移，与冻结基线逐位一致）、#90（进度文档）、#91（因子层重构 D3：store 层——稀疏值存储 + key/指纹 + tail-recompute + R5 fina 视界统一）、#92（进度文档）、#93（因子层重构 D4：FactorService + decision-view materializer + pooled 饱和填充）、#94（进度文档）、#95（因子层重构 D5a：exec 基线冻结 + 11 runner 差异编目 + 可行性探测）、#96（进度文档）、#97（因子层重构 D4b：per-symbol 流式物化）、#98（进度文档）、#99（因子层重构 D5b：评估契约 v1.0 + 诊断通道 + 14:50 可见性守卫；**C4 中止上报**）、#101（因子层重构 D4c：store 去 universe 依赖——(date,symbol) 缺口判定 + store 只存 per-symbol 中间量、截面 combine 移到读出装配 + 调用方 symbol 列表单点归一化）、#103（jump_amount_corr_20 补 14:50 截断 correctness fix——唯一未做截断的分钟因子，exec_to_exec 下信息集越过自己入场锚的真实前视；含 dashboard 布局 5/4 分支修复 + census 守卫前缀匹配修复；单独成 PR 非重构一部分）、#105（因子层重构 D5 C4：统一 exec-only FactorEvalRunner——因子取数全走 factors.service、异构披露唯一家、C5 对账 harness 三模式 + 77/77 硬闸门、warmup 左延具名类登记）、#107（因子层重构 D5 C4b：valley_price_quantile 绑定——combine_daily 声明式引擎扩展 + 平移面板第三条路逐比特等价双侧钉 + binding.py 折进 code_hash + 截面因子 panels 类边界三裁定）均已 merge 到 `main`**；**OPEN 未合并：#38**（见 P-I5d 处置，未获授权关闭）。commit 用 conventional 格式，**无 attribution**（不加 Co-Authored-By）。

## 归档之后新增的 PR（本节按时间追加，原文那一大段不再改动）

| PR | 内容 |
|---|---|
| #108 | 进度文档（D5 C4b） |
| #109 | D5 C5 prep：性质测试迁移映射表 + 审计报告骨架 |
| #110 | 文档瘦身：进度日志从 `CLAUDE.md` 逐字搬进 `docs/progress/`（398 → 151 行） |
| #111 | 台账补记 #108–#110 + 状态表标明 gate 实测于哪个 commit |
| #112 | `amp_marginal_anomaly_vol` 残桶 correctness fix（单独成 PR，非重构一部分） |
| #113 | D5 C5 四腿全量对账收口（F2/F3/F4 扩类 + F5 修复 + NIT-1 + 审计报告） |
| #114 | 进度文档（F1 + D5 C5） |
| #115 | D5 C6 改造：三个重生成工具退役为 verify-only + CLI 收敛 + R16 核销 |
| #116 | D5 C6 删除：删 11 个旧 runner + 10 测试，校验接进 C5 harness 三个读入点 |
| #117 | 进度文档（D5 C6） |
| #118 | D6a：phase0/phase2 切 FactorService（含 B1 demo-store 隔离、日频 universe 无关守卫、caller 派生普查） |
| #119 | 修 retired-invocation 守卫的射程：文件清单从 git 派生而非目录遍历（`main` 曾红而 worktree 绿） |
| #120 | 进度文档（D6a） |
| #121 | retired-invocation 守卫射程声明补全（docstring 写明文件派生盲区） |
| #122 | fina 并列披露稳定去重 correctness fix + `pit_financials` 折 code_hash（重述量化：SSE50 5.12% / CSI300 5.11% 格） |
| #123 | D6b PR-1 phase3 基线捕获工程：capture harness + cached twins + `d6b_phase3` 冻结基线 |
| #124 | D6b：oos/subset/robustness 切 FactorService——S1 冷/S2 暖 vs 冻结基线全叶子 0 |
| #126 | D6c PR-1：`mmp_ew`/`ret` 注册为正式因子 + I5 捕获 harness + `d6c_i5` 冻结基线（score 腿 6/6 max_abs_diff=0.0） |
| #127 | D6c PR-2：intraday 两 runner 切 FactorService——S1 冷/S2 暖 × 6 配置 vs L1 全 12/12 n_diffs=0，执行面零改动 |
| #128 | 进度文档（D6c） |
| #129 | D6d PR-1：shim import 全 repoint 到 `factors.compute.minute.*` + 删 legacy 取数路径（`_serve_factor_panel` 成独一入口）+ 继任守卫 AST census |
| #130 | D6d PR-2：删 10 shim + `intraday_derived` + aggregate 因子数学（R14 通用核纯态；store schema-version 一次失效实测 46 key 恰 6 变） |
| #131 | D6d PR-3：新锚转正——`test_phase0_anchor.py` FINAL 全精度三级比对，删 4dp 旧锚断言（1-ulp mutation 验证不削弱门禁） |
| #132 | 进度文档（D6d） |
| #133 | D6a-2：panel 富化从 isinstance 分派改声明式路由（`requirements_of` 首个生产消费者；行为等价 fixture + 真数据 oos/subset 对账 n_diffs=0 ×2） |
| #134 | D7-PR0：run registry 首次接线（启动门 `sync_book_registry` + 评估后 append RunRecord + status 映射） |
| #135 | 进度文档（D6a-2 + D7-PR0 + D7 收官） |
