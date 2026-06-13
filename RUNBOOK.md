# RUNBOOK — Phase 1 (A-share cross-sectional multi-factor, bias-boundary)

Phase 0 = a runnable demo MVP (offline `DemoFeed`). Phase 1 adds the real-data
bias boundaries: front-adjustment, PIT index membership, tradability filters,
ann_date financial alignment, and industry+size neutralization. The demo path is
unchanged; the real path is opt-in via `config/example_tushare.yaml`.


## Environment

- Python (always use the absolute path; do NOT rely on `activate`):
  ```bash
  /home/shaofl/Development/env_tools/envs/quant_mf/bin/python
  ```
- Editable install is already done (`pip install -e . --no-deps`), so `import qt`,
  `import data`, `import factors`, ... work from the repo root.
- No tushare token is needed for Phase 0: `run-phase0` uses the offline,
  deterministic `DemoFeed` (no network, no credentials). The token lives ONLY in
  the external `/home/shaofl/Projects/financial_projects/.config.json` and is
  never read, printed, or logged by the demo pipeline.

## Commands

Run from the repo root
`/home/shaofl/Projects/financial_projects/stocks_market/Quantitative_Trading`.

### 1. Validate a config (CLI-001)

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli \
  validate-config --config config/example.yaml
```

Expected: prints `OK`, exit code `0`. A bad/missing config prints a readable
`ERROR: ...` (never a raw traceback) and exits `1`.

### 2. Run the end-to-end pipeline (CLI-002)

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli \
  run-phase0 --config config/example.yaml
```

Expected (numbers depend on demo data):

```
OK run-phase0: ic_mean=0.8681, annual_return=...
report: artifacts/reports/phase0_summary.md
```

This chains: demo feed -> `normalize_panel` -> `PanelStore.write/read` ->
`StaticUniverse` -> `MomentumFactor.compute` -> `ProcessingPipeline.transform` ->
`EqualWeightAlpha.fit(None).predict` (per rebalance date) -> `TopNEqualWeight.build`
-> `BacktestDriver.run` (`SimExecution`) -> analytics (IC, performance) -> report.

### 3. Stage helper sub-commands (CLI-005, optional)

Phase 0 keeps a single reproducible spine; these run that spine and report the
stage of interest (they do not persist partial cross-process state):

```bash
... -m qt.cli fetch-data       --config config/example.yaml
... -m qt.cli compute-factors  --config config/example.yaml
... -m qt.cli run-backtest     --config config/example.yaml
```

## Expected artifacts

All writes are confined to the configured `output.*` directories (SEC-003). With
the example config:

```text
artifacts/data/daily.parquet        # canonical (date, symbol) market panel
artifacts/factors/factors.parquet   # momentum_20 factor panel
artifacts/reports/phase0_summary.md # ANA-005 run summary (+ DOWNGRADES section)
artifacts/logs/run_phase0.log       # run log (no secrets)
```

`artifacts/` is git-ignored (SEC-002); the reports are fully regenerable by
re-running `run-phase0` (INV-006). Re-running over existing files is safe
(re-entrant).

## Phase 1 — real-data (tushare) path

The real path is documented by `config/example_tushare.yaml`. It needs the token
in `/home/shaofl/Projects/financial_projects/.config.json` (read from there, never
printed/committed) and hits tushare, so it is NOT run in CI. An index run loads
~300 names × daily plus flags/covariates (rate-limited), so it is heavy.

```bash
# validate the real-path config (no network)
... -m qt.cli validate-config --config config/example_tushare.yaml
# run the real path (network + token; heavy for an index universe)
... -m qt.cli run-phase0      --config config/example_tushare.yaml
```

What the real path turns on (each is a correctness red-line, disclosed in the
report's DATA PATH / DOWNGRADES section and `BIAS_AUDIT.md`):

| Toggle | Effect | Module |
|---|---|---|
| `data.source: tushare` | real **front-adjusted (qfq)** prices | `data/clean/adjust.py` |
| `universe.type: index` | **PIT** index membership (as-of, survivorship-safe) | `universe/index_universe.py` |
| `filters.{suspended,st,limit_up_down}` | **tradability** filtering | `universe/filters.py` + `data/clean/tradability.py` |
| `factors: [{name: roe}]` | **ann_date** PIT financial factor | `data/clean/pit_financials.py` |
| `processing.neutralize.enabled` | **industry + size** neutralization | `factors/process/neutralize.py` |

Financial factors / index universe / neutralization require the tushare path; on
the demo source they raise a readable error instead of fabricating data.

Correctness details (locked by tests):

- **Price-limit flags use RAW close** vs the raw `stk_limit` (limits are quoted in
  unadjusted price); flags are enriched BEFORE front-adjust, so the qfq close is
  used only for factors/returns, never for the limit comparison.
- **Financials look back ~16 months** before `start`, so the most recent report
  disclosed before the backtest starts is fetched and as-of carried forward onto
  the early trade dates (no NaN gap), still gated by `ann_date <= trade_date`.
- **Neutralization returns NaN** on a saturated cross-section (names ≤ 1 + #industries,
  i.e. no residual degrees of freedom) rather than fabricated ~0 residuals.

## Phase 2-1 — small-scale real-data reproducibility baseline

A REAL (tushare) end-to-end run of the EXISTING P0/P1 spine over a small universe
(SSE50, `000016.SH`) and a ~1-year window, designed to finish in ~10-30 min. It
adds NO new factor and does NO parameter search — it validates the real-data
plumbing and emits a richer diagnostic report. Documented by
`config/phase2_real_baseline.yaml`.

```bash
# validate (no network)
... -m qt.cli validate-config    --config config/phase2_real_baseline.yaml
# run the real baseline (network + token; ~10-30 min for ~50-70 names)
... -m qt.cli run-phase2-baseline --config config/phase2_real_baseline.yaml
```

Output: `artifacts/reports/phase2_real_baseline.md` (git-ignored, regenerable). The
report contains: data window, PIT membership summary (snapshots / distinct names /
churn), ann_date as-of financial coverage (a DATA-QUALITY diagnostic on `roe`, not
the alpha factor), tradability filter-hit funnel, rebalance dates, per-period
holdings, per-period turnover/cost, IC / quantile returns, performance summary, and
all P2 downgrades.

Guards (correctness / honesty):

- **Demo source is refused.** `run-phase2-baseline` raises a readable error on
  `data.source != 'tushare'` — a "baseline" on offline demo data carries no PIT /
  ann_date / tradability meaning and must not masquerade as a real validation.
- **Holdings are the driver's ACHIEVED book** (`BacktestDriver.holdings_log()`,
  post execution-feasibility) — the actual positions held each period, NOT the
  constructor's desired target (a blocked sell shows the carried name, a blocked
  buy is absent). The reporting never sees forward returns at the factor stage.
- **No secret leak.** The report echoes only non-sensitive config (window, universe,
  factor); the token / secret file path is never written into it.

## Phase 2-2 — execution realism (direction-aware fills + min_listing_days)

P2-2 closes execution/tradability gaps in the backtest. No new factor, no parameter
tuning — it changes how the driver *fills* a target and how selection eligibility is
computed, and it applies to every real-path run (`run-phase0` and
`run-phase2-baseline`).

**Selection vs execution feasibility (split):**

- *Selection* (`universe.tradable` / `apply_tradable_filters`): missing_close /
  suspended / ST / limit toggles, plus **`min_listing_days`** (UNI-008) — a
  buy/selection filter that drops names younger than `min_listing_days` as of each
  date. Real path enriches `list_date` from `stock_basic`; a missing list_date is a
  disclosed data gap (kept, never silently dropped); demo has no listing dates → a
  disclosed no-op.
- *Execution feasibility* (`runtime.fills.simulate_fills`): read off the panel flags,
  independent of the selection toggles — `at_up_limit` blocks **buys**, `at_down_limit`
  blocks **sells**, `suspended`/missing-close blocks **both**.

**Cash-coherent fill model:** sells execute first (freeing cash), buys are funded from
available cash and scaled down proportionally if blocked sells starved them (the book
never sums to > 1 — no leverage). Blocked trades carry the current position forward;
turnover/cost count only executed trades; idle cash earns the driver's `cash_return`
(BT-007). The demo panel carries no flags, so every trade is feasible and P0/P1
numbers are unchanged.

The phase2 baseline report (`artifacts/reports/phase2_real_baseline.md`) gains an
**Execution feasibility** section: per-rebalance blocked buys / blocked sells / carried
positions / executed turnover / invested fraction, from `BacktestDriver.feasibility_log()`.

## Phase 3-1 — first REAL multi-factor baseline (price + PIT financials)

P3-1 makes the pipeline consume **every enabled factor** (it previously used only
the first) and adds the first real multi-factor baseline: `momentum_20` + `roe` +
`netprofit_yoy`, combined by the SAME equal-weight alpha (row-wise mean of the
processed z-scored / neutralized columns). **No parameter search, no learned
weights, not a performance claim** — it validates the multi-factor plumbing.
Documented by `config/phase3_real_multifactor.yaml` (same SSE50 universe + window
as the phase2 baseline, so the two are comparable).

```bash
# validate (no network)
... -m qt.cli validate-config     --config config/phase3_real_multifactor.yaml
# run the multi-factor real baseline (network + token; heavy, ~10-30 min)
... -m qt.cli run-phase2-baseline --config config/phase3_real_multifactor.yaml
```

The real-baseline RUNNER is shared with phase2 (`run-phase2-baseline` — same
diagnostics machine); `output.baseline_report_name` keeps the report separate:
`artifacts/reports/phase3_real_multifactor.md` (git-ignored, regenerable).

What P3-1 changes (locked by tests):

- **Multiple enabled factors** each become their own factor-panel column
  (config order; duplicate names are a config error). Single-factor configs are
  unchanged (primary factor = the only factor).
- **Financial fields are fetched ONCE** for all financial factors and as-of
  aligned in a single `asof_financials` pass (`ann_date <= trade_date` per
  field) — no per-factor refetch. Demo + financial factor still raises a
  readable error (no fabricated financials).
- **Combination** is the equal-weight mean of the per-date processed columns
  (`EqualWeightAlpha`); `drop_missing` requires ALL enabled factors for a name
  on a date (disclosed — a name missing any factor is dropped from that
  cross-section, never scored on partial data).
- **Report**: active factor list; per-factor coverage / IC / quantile
  diagnostics plus the COMBO score's; financial ann_date coverage **per field**,
  labelled TRADED factor vs diagnostic-only. Standard analytics stays
  report-only (alphalens cross-checks the primary factor).

## Phase 3-2 — walk-forward IC-weighted alpha

P3-2 adds the first LEARNED factor combination while keeping the lookahead
boundary auditable: `alpha.model: ic_weighted` weights each factor by its mean
cross-sectional rank IC over a trailing window of **realized** observations
only. `EqualWeightAlpha` stays the default and the regression baseline.
Documented by `config/phase3_real_ic_weighted.yaml` (identical universe /
window / factors / neutralization to `phase3_real_multifactor.yaml` — the ONLY
change is the alpha model, so the two runs are directly comparable).

```bash
# validate (no network)
... -m qt.cli validate-config     --config config/phase3_real_ic_weighted.yaml
# run the ic-weighted real baseline (network + token; heavy, ~10-30 min)
... -m qt.cli run-phase2-baseline --config config/phase3_real_ic_weighted.yaml
```

Lookahead boundary (locked by tests):

- The factor layer NEVER sees forward returns (invariant #1 unchanged); the
  pipeline computes them at the alpha boundary and hands them ONLY to
  ``alpha.fit``.
- Training is **walk-forward**: at each date d, a (factor[t], fwd_h[t]) pair is
  admissible only once REALIZED — ``t + h <= d`` in trading-day positions (the
  h-day forward return of factor date t realizes at t+h). Perturbing any
  not-yet-realized forward return cannot change d's weights (perturbation test).
- Window modes: ``rolling`` (trailing `window` trading days, the conservative
  default) or ``expanding`` (all realized history).
- **Fallback**: if any factor has fewer than ``min_periods`` valid realized ICs
  in the window (or the ICs are degenerate), that date falls back to the EQUAL
  WEIGHT mean — identical to `EqualWeightAlpha` — and is counted + disclosed in
  the report.
- Weights are L1-normalized and sign-preserving (a negative-IC factor gets a
  negative weight). Fixed recipe, no parameter search — NOT a tuned-performance
  claim.

The baseline report gains an **Alpha model** section: active model,
hyper-params, training coverage (fallback count), and the effective weights at
every settled rebalance date (fallback rows flagged).

## Phase 3-3 — OOS stability validation (equal_weight vs ic_weighted)

A REPORT-ONLY validation layer (no portfolio / execution / factor-math change):
one shared data load, the SAME processed factor panel, two backtests — one per
alpha — and every diagnostic split at `oos.split_date`. Documented by
`config/phase3_real_oos_stability.yaml` (SSE50, two years 2022-07 ~ 2024-06,
split 2023-07-01 → train 1y / test 1y; the test year equals the phase3
baselines' window, so numbers line up).

```bash
# validate (no network)
... -m qt.cli validate-config  --config config/phase3_real_oos_stability.yaml
# run the OOS validation (network + token; heavy, ~15-30 min)
... -m qt.cli run-phase3-oos   --config config/phase3_real_oos_stability.yaml
```

Split semantics (locked by tests):

- train = `[data.start, split)`, test = `[split, data.end]`; the exact realized
  dates and day counts are written into the report.
- Evaluation is **walk-forward (rolling subperiod)**: weights at any date d use
  only observations REALIZED by d (`t + horizon <= d`) — perturbing every
  post-split forward return cannot change any train-period date's weights
  (split-boundary no-leakage test). Freezing weights at the split is NOT used
  (that would be a new alpha mode; P3-3 adds no alpha complexity).
- **Performance slicing is HOLDING-WINDOW aware**: a nav row is indexed by its
  rebalance (signal) date but its return covers [that rebalance, the next one] —
  so train rows must have their holding END on/before the split, test rows their
  holding START on/after it, and a straddling rebalance is EXCLUDED from both
  subperiods and disclosed in the report. IC stats are sliced by the realization
  date (`t + horizon`) the same way. The test subperiod therefore starts with
  the first post-split holding period — the same place the 1-year phase3
  baselines start.
- The config's `alpha` section carries the ic_weighted params and MUST set
  `alpha.model: ic_weighted` (guarded: any other model is refused — running
  equal_weight twice and labelling one leg ic_weighted would be a fake
  comparison); the `equal_weight` control leg is built internally.

Report (`artifacts/reports/phase3_oos_stability.md`): split boundaries;
per-subperiod performance for both models (annual / vol / Sharpe / maxDD /
turnover / rebalances); per-series IC stability (mean / IR / hit rate / n +
train-vs-test sign consistency) for every raw factor and both combo scores;
ic_weighted weight stability (per-rebalance weights with train/test labels,
sign-flip counts on trained rows, fallback count + reasons); and the explicit
caveat that this is a small-sample stability check, NOT a return claim.

## Phase 3-4 — robustness matrix (longer history + wider universe)

Re-runs the P3-3 OOS stability check on every (universe × window) cell to test
whether the single-sample conclusions generalize. No new factor, no new alpha,
no portfolio / execution / factor-math change. Documented by
`config/phase3_real_robustness_matrix.yaml` (SSE50 + CSI300 × two 2-year
folds 2020-2022 / 2022-2024; the CSI300×2020-2022 cell is explicitly
`skip_cells`-listed for runtime budget and DISCLOSED in the report).

```bash
# validate (no network)
... -m qt.cli validate-config        --config config/phase3_real_robustness_matrix.yaml
# run the matrix (network + token; HEAVY — a CSI300 2y cell alone is ~60-90 min)
... -m qt.cli run-phase3-robustness  --config config/phase3_real_robustness_matrix.yaml
```

Mechanics (locked by tests):

- Every cell reuses the P3-3 cell core VERBATIM (`qt.oos_stability._run_oos_cell`):
  holding-window performance slicing, realization-date IC slicing, walk-forward
  weights, equal-weight fallback disclosure, and the shared guards (real source,
  `oos` section, the ic_weighted fake-comparison guard). The single-run
  `run-phase3-oos` behaviour is unchanged.
- Per-cell configs are derived from the base with ONLY the cell identity
  swapped (universe.index_code, data window, split, and a per-cell
  `output_name` so panels never overwrite each other); the derived config
  re-runs full pydantic validation.
- `robustness.skip_cells` reduces coverage EXPLICITLY (validated against the
  declared universes/windows; at least one cell must remain) and every skip is
  disclosed in the report — coverage is never silently reduced.

Report (`artifacts/reports/phase3_robustness_matrix.md`): the cell table
(window / split / per-cell runtime) + skipped cells; a CROSS-CELL stability
summary (per series: #cells with positive test IC, #cells train→test
sign-consistent, test IC by cell; #cells where ic_weighted beats equal_weight
on test annual); full per-cell diagnostics (subperiod performance, IC
stability, weight sign flips, fallback counts, boundary rebalances); and the
explicit caveat that this is a stability check — NOT a return claim, not a
tuned result.

## Phase 3-5 — factor candidate pack (EXPLORATORY)

Adds a conservative, daily, PIT-safe candidate factor pack and re-runs the
P3-4 robustness matrix to test whether the legacy trio's weak signal was just
a too-narrow factor set. **Exploratory — not tuned, not a return claim.** No
alpha / portfolio / execution / OOS / robustness model change. Documented by
`config/phase3_real_factor_candidates.yaml` (legacy trio + 8 candidates,
same matrix shape as P3-4 incl. the disclosed CSI300×2020-2022 skip).

```bash
# validate (no network)
... -m qt.cli validate-config        --config config/phase3_real_factor_candidates.yaml
# run (network + token; HEAVY ~2-2.5h — CSI300 adds ~400 names × daily_basic pe/pb)
... -m qt.cli run-phase3-robustness  --config config/phase3_real_factor_candidates.yaml
```

Candidates (all locked by unit tests for math, leading-window NaN,
no-future-bars, per-symbol isolation, immutability):

| Factor | Definition | PIT argument |
|---|---|---|
| `reversal_5/20` | −(close[t]/close[t−w] − 1) | exact negative of momentum (≤t bars) |
| `volatility_20` | std of trailing 20 daily returns (ddof=1, full window) | ≤t bars |
| `liquidity_20` | log(mean(amount, 20)) (non-positive → NaN) | same-day bar amount |
| `overnight_mom_20` | Σ log(open[t]/close[t−1]) over 20d (non-positive → NaN) | open known at the t open; computed at the t close |
| `value_ep` / `value_bp` | 1/pe, 1/pb from `daily_basic` (one fetch; ≤0 → NaN) | ratios published same-day |
| `grossprofit_margin` | financial quality field | ann_date as-of (existing machinery) |

Registry / dispatch: `_build_factors` resolves `reversal*` / `volatility*` /
`liquidity*` / `overnight_mom*` / value fields / financial fields; window-named factors must agree
with `params.window` (a mismatch is a readable config error, never a silent
mislabel). Legacy configs are untouched and reproduce their numbers.

Old-vs-new comparison reads off ONE run: raw-factor ICs are per-column (each
factor's IC depends only on its own column + forward returns), so the legacy
trio's per-cell ICs in the candidates run double as a no-drift cross-check
against the P3-4 report, and the candidate ICs are directly comparable in the
same report. Note `drop_missing` requires ALL enabled factors per name/date
(e.g. a loss-maker's NaN `value_ep` drops it that day) — the combo legs are
therefore NOT the P3-4 combos; disclosed.

## Phase 3-6 — value+lowvol subset re-check + cost sensitivity (EXPLORATORY)

Compares configured FACTOR GROUPS head-to-head on the unchanged P3-4 matrix
(one shared data load + raw factor panel per cell) and repeats every backtest
under scaled trading-cost scenarios. **Report-only; no new alpha model, no
tuning, no portfolio / execution / OOS-slicing / aggregation change;
`_run_oos_cell` untouched.** Documented by
`config/phase3_real_subset_costs.yaml` (4 groups × 3 cost scenarios, same
matrix shape as P3-4/P3-5 incl. the disclosed CSI300×2020-2022 skip).

```bash
# validate (no network)
... -m qt.cli validate-config    --config config/phase3_real_subset_costs.yaml
# run (network + token; HEAVY ~2-2.5h — same data pulls as the P3-5 matrix)
... -m qt.cli run-phase3-subset  --config config/phase3_real_subset_costs.yaml
```

Factor groups (config-driven; labels and lists are disclosed in the report):

| Group | Factors |
|---|---|
| `legacy_trio` | momentum_20, roe, netprofit_yoy |
| `full_pack` | all 11 P3-5 factors |
| `value_lowvol` | value_ep, value_bp, volatility_20 |
| `value_lowvol_liq` | value_lowvol + liquidity_20 (the /goal's optional variant, measured) |

Cost scenarios scale `cost.fee_rate` only (`base`×1, `2x`×2, `high_cost`×4; a
multiplier-1.0 base scenario is REQUIRED — the cost-drag anchor). Scores and
fills never see the fee, so trades/turnover are IDENTICAL across scenarios —
only the cost line (and net return) changes (locked by tests).

Semantics worth knowing before reading the report:

- **Per-group reprocessing**: each group is re-processed independently from the
  shared raw factor panel — `drop_missing` requires completeness only across
  the GROUP's columns (a group with every column reproduces the P3-4/P3-5
  processing bitwise, locked by tests). The combo legs of different groups are
  therefore NOT each other's subsets.
- **No-drift hook**: raw-factor ICs are per-column and group-independent; the
  per-cell raw IC table must reproduce the P3-5 report's numbers.
- **POST-HOC selection (disclosed)**: the value+lowvol subset was chosen after
  seeing P3-5 results on these same windows — the run quantifies RELATIVE
  robustness + cost sensitivity, not independent confirmation; NOT a return
  claim.

## Phase 3-7 — genuinely independent sample validation (EXPLORATORY)

Moves the P3-5/P3-6 value/low-vol finding from "post-hoc comparison on the
screening windows" to a genuinely independent holdout: the same
`run-phase3-subset` mode, with cells explicitly labeled **independent holdout**
vs **screened/post-hoc** and a pre-declared hypothesis sign check. No new
factor / alpha / tuning; the P3-6 group/cost logic is unchanged. Documented by
`config/phase3_real_independent_validation.yaml`.

```bash
# validate (no network)
... -m qt.cli validate-config    --config config/phase3_real_independent_validation.yaml
# run (network + token; HEAVY ~2-2.5h — CSI300|2024-2026 dominates)
... -m qt.cli run-phase3-subset  --config config/phase3_real_independent_validation.yaml
```

Semantics worth knowing before reading the report:

- **Independence is a human declaration** (`subset_validation.independent_cells`):
  the machine cannot know which data took part in screening, so undeclared
  cells default to screened (conservative). A declared cell must exist in the
  matrix and must NOT be skip-listed (an independent validation that never
  runs is a config error).
- **Hypotheses are fixed BEFORE the run** (`subset_validation.hypotheses`,
  e.g. `value_ep: positive` / `volatility_20: negative`). The verdict is a
  factual IC SIGN check: a hypothesis HOLDS iff the expected sign appears in
  BOTH subperiods of the holdout cell. Statuses: SUPPORTED / PARTIAL /
  NOT SUPPORTED / INSUFFICIENT-DATA (fewer settled rebalances than
  `min_rebalances` — size always disclosed). Never a return claim.
- **Conclusions never mix**: the report's cross-cell summaries are computed
  per sample class, and the "Independent holdout verdict" section reads
  independent cells only. The screened anchor cell (SSE50|2022-2024) must
  reproduce the P3-6 numbers exactly — the in-run no-drift check.
- NOTE: every `run-phase3-subset` config writes
  `artifacts/reports/phase3_subset_validation.md` — a P3-7 run overwrites a
  P3-6 report (regenerable; numbers live in CLAUDE.md/TEST_REPORT).

## Phase 3-8 — CSI500 independent generalization check (EXPLORATORY)

Asks whether the P3-7 sign-level conclusion generalizes OUTSIDE the screened
universes: the same `run-phase3-subset` machinery (unchanged) with a NEW
independent cell `000905.SH|2024-2026` (CSI500 — independent in BOTH universe
and time), plus the two SSE50 anchors (screened 2022-2024 must reproduce
P3-6/P3-7; independent 2024-2026 must reproduce the P3-7 verdict numbers).
Documented by `config/phase3_real_csi500_generalization.yaml`.

```bash
# validate (no network)
... -m qt.cli validate-config    --config config/phase3_real_csi500_generalization.yaml
# run (network + token; HEAVY ~3.5h — the 735-name CSI500 cell dominates)
... -m qt.cli run-phase3-subset  --config config/phase3_real_csi500_generalization.yaml
```

- `output.subset_report_name` (the `baseline_report_name` precedent) gives this
  study its own report file (`phase3_csi500_generalization.md`), so the run no
  longer clobbers the accepted P3-7 artifact; configs without the key keep the
  historical filename bitwise (locked by tests).
- `output.subset_report_title` sets the report's H1 so it names THIS study
  ("Phase 3-8 — CSI500 Independent Generalization Check") instead of the
  machinery's default phase label; configs without it keep the renderer's
  sample-aware default (P3-7 independent / P3-6 post-hoc), locked by tests.
  CSI500|2022-2024 is skip_cells-listed (runtime budget) and disclosed.

## Phase 4-1 — persistent Tushare market-data cache (daily + adj_factor)

A persistent endpoint-level RAW cache below the feeds so real runs stop
refetching full daily bars + adj_factor every time. **Disabled by default**
(backward compatible — an existing config runs exactly as before). The cache
stores RAW rows only (unadjusted OHLCV/amount, raw adj_factor); `front_adjust`
still runs in memory downstream, unchanged. `PanelStore` stays a per-run
artifact, NOT the cache source of truth.

Opt in via `data.cache`:

```yaml
data:
  cache:
    enabled: true
    root_dir: artifacts/cache/tushare/v1   # gitignored
    refresh_recent_days: 14                 # refetch a recent tail near today
    force_refresh: []                       # e.g. ["market_daily","adj_factor"]
```

Read-through behaviour (`TushareFeed.get_bars`, cache enabled):

- the coverage ledger (`<root>/manifest/coverage.parquet`) records which
  `(endpoint, symbol, [start,end])` ranges were fetched (status ok/empty/failed)
  — so the planner fetches ONLY uncovered date gaps;
- a second identical run over a historical window makes **zero** `daily` /
  `adj_factor` API calls;
- a partial window extension fetches only the new tail;
- raw rows are upserted per `(symbol, date)` (latest wins; no duplicates);
- the cache + ledger never store a token or secret-file content;
- the run log shows the hit rate directly — `_load_panel` emits
  `data cache: market_daily_gap_fetches=<N> adj_factor_gap_fetches=<M>`
  (cold run nonzero; a warm historical rerun shows 0/0).

Cache layout (per-symbol parquet under a symbol_prefix shard):

```text
artifacts/cache/tushare/v1/
  manifest/coverage.parquet
  market_daily/symbol_prefix=600/600519.SH.parquet
  adj_factor/symbol_prefix=600/600519.SH.parquet
```

Real smoke (small real config; metrics must equal the non-cached run):

```bash
# populate, then re-run: the second run hits cache for market bars
... -m qt.cli run-phase2-baseline --config config/phase2_real_baseline_cached.yaml
... -m qt.cli run-phase2-baseline --config config/phase2_real_baseline_cached.yaml
```

Scope note: P4-1 caches market bars ONLY. `index_weight`, `daily_basic`,
`fina_indicator`, `stk_limit`, `suspend_d`, `namechange`, `stock_basic`,
`index_member_all` are still fetched live (cached in P4-2/P4-3); the warm run
therefore saves the market-bar calls, not those.

## Quality gate

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
```

## Downgrades / status

Each run's active path + downgrades are disclosed in its summary's DATA PATH /
DOWNGRADES section (INV-007) and explained in `BIAS_AUDIT.md`.

Implemented in P1 (real path):

- **Front-adjust (qfq)** — `adj_factor`-based; store stays raw, adjust in memory
  (batch≡incremental safe).
- **PIT index membership** — resolves the static-universe survivorship downgrade.
- **Tradability filters** — suspended / ST / limit-up-down (+ always missing_close).
- **ann_date financial alignment** — figures used only after disclosure date.
- **Industry + size neutralization** — per-date OLS residual.

Resolved in P2-2 / P2-3 (was deferred):

- **Direction-aware limits/suspension** (P2-2) — now in the execution layer (up-limit
  blocks buys, down-limit blocks sells, suspended/missing blocks both; blocked
  trades carry forward, executed-only turnover). No longer a crude both-direction
  selection drop.
- **min_listing_days** (P2-2) — enforced on the real path (`stock_basic.list_date`) as a
  buy/selection filter; demo stays a disclosed no-op.
- **PIT industry** (P2-3) — the neutralization industry covariate is now point-in-time
  SW (as-of trade date via `index_member_all` in/out dates, `data/clean/pit_industry.py`),
  not the current `stock_basic.industry` tag. The SW level is configurable
  (`processing.neutralize.industry_level`, L1/L2/L3, **default L1** = 31 broad sectors, the
  standard for neutralization and DOF-safe on small cross-sections). Real runs: old tag
  annual −17.6%, SW-L1 −10.2%, SW-L2 −9.3% — L1 ≈ L2, so the −17.6→−10 jump is the
  tushare→SW taxonomy switch (inherent to going PIT: only SW carries in/out-date history),
  NOT granularity. Names with no SW history get NaN (a disclosed coverage gap the
  neutralizer drops) — never a silent current-tag fallback; the actual level + PIT coverage
  are reported in `phase2_real_baseline.md`.

Resolved in P2-4 (was deferred):

- **Standard analytics** — alphalens-reloaded (IC / quantiles) and quantstats (CAGR /
  Sharpe / maxDD / vol) are now computed and shown in the report as a **report-only
  cross-check** (`analytics/alphalens_adapter.py`, `analytics/quantstats_adapter.py`).
  The simple numpy/pandas metrics remain the AUTHORITATIVE backtest result and drive
  the run; the standard tools never alter selection / portfolio / execution. When a
  library is unavailable or errors, the report discloses the `backend` (no silent fake).

Still downgraded / deferred (disclosed):

- **Demo path** uses offline `DemoFeed` — NOT real data (no PIT/financial meaning).
- **Static universe** option remains a PIT downgrade (use `type: index` for real).
- **Daily bars only** (minute-level link deferred).
