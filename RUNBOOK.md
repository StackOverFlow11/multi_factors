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
