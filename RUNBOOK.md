# RUNBOOK — Phase 0 (A-share cross-sectional multi-factor MVP)

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

## Quality gate

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
```

## Downgrades (Phase 0)

This MVP intentionally ships simplified components; each is disclosed in
`artifacts/reports/phase0_summary.md` (INV-007) and explained in `BIAS_AUDIT.md`:

- **Static universe** (date-independent membership) — a PIT downgrade (UNI-003);
  survivorship / look-ahead membership bias is present and intentional for P0.
- **Daily bars only** — minute-level link deferred to P1.
- **Simple IC / quantile analytics** — numpy/pandas, not `alphalens-reloaded`.
- **Simple performance metrics** — numpy/pandas, not `quantstats`.
- **adj_factor retained, no full forward-adjust** — DemoFeed `adj_factor == 1.0`;
  real tushare ingestion must forward-adjust using `adj_factor` (DATA-003).
