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
- **Holdings are reconstructed read-only** (universe → scores → `constructor.build`,
  the same chain the driver runs); the reporting never sees forward returns at the
  factor stage and never re-derives returns.
- **No secret leak.** The report echoes only non-sensitive config (window, universe,
  factor); the token / secret file path is never written into it.

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

Still downgraded / deferred (disclosed):

- **Demo path** uses offline `DemoFeed` — NOT real data (no PIT/financial meaning).
- **Static universe** option remains a PIT downgrade (use `type: index` for real).
- **Industry tag is current** (`stock_basic`), not point-in-time — mild downgrade.
- **min_listing_days** configured but not enforced (no-op).
- **Daily bars only**; **simple IC/perf** (not alphalens/quantstats);
  limit filter is not yet trade-direction-aware (P2).
