# TEST_REPORT — Phase 0 + Phase 1 + Phase 2 (bias-boundary → execution realism → PIT industry)

## Commands

Run from the repo root with the project python (env `quant_mf`):

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example_tushare.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase2_real_baseline.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli run-phase0 --config config/example.yaml
```

## Results

| Gate | Command | Result |
|---|---|---|
| Unit + integration | `pytest -q` | **223 passed, 0 failed** |
| Lint | `ruff check .` | **All checks passed** |
| Config validation | `validate-config` (demo + `example_tushare.yaml` + `phase2_real_baseline.yaml`) | exit `0`, prints `OK` |
| End-to-end run | `run-phase0` (demo) | exit `0`, writes `artifacts/reports/phase0_summary.md` |

Counts below are the actual per-file `pytest` numbers (sum = 223).

## Per-file breakdown — Phase 0 core (95)

| Test file | Tests | Area |
|---|---|---|
| `test_project_bootstrap.py` | 10 | bootstrap / imports / config |
| `test_config.py` | 5 | config schema |
| `test_data_schema.py` | 6 | panel schema |
| `test_panel_store.py` | 6 | parquet store |
| `test_data_feed.py` | 5 | demo + tushare feed boundary |
| `test_universe.py` | 4 | static universe |
| `test_factors_momentum.py` | 6 | momentum (no-lookahead) |
| `test_factor_processing.py` | 4 | drop_missing + z-score |
| `test_alpha_equal_weight.py` | 4 | equal-weight alpha |
| `test_portfolio_topn.py` | 8 | TopN construction |
| `test_sim_execution.py` | 8 | sim execution + cost |
| `test_backtest_driver.py` | 7 | backtest driver |
| `test_analytics_factor.py` | 5 | IC / quantile |
| `test_analytics_performance.py` | 3 | performance metrics |
| `test_phase0_pipeline.py` | 8 | end-to-end pipeline |
| `test_bias_audit_report.py` | 6 | bias audit doc (+2: P2-2 + P2-3 disclosures) |

## Per-file breakdown — Phase 1 bias-boundary (79)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_adjust.py` | 7 | front-adjust (qfq); ex-dividend gap removed |
| `test_tushare_throttle.py` | 5 | shared rate-limit + retry |
| `test_index_universe.py` | 7 | PIT index membership (as-of, survivorship) |
| `test_index_feed.py` | 4 | index_weight feed + 90-day paging |
| `test_index_pipeline.py` | 1 | pre-start snapshot lookback (as-of edge) |
| `test_config_index.py` | 3 | `universe.type=index` config |
| `test_tradability_filters.py` | 7 | shared suspended/ST/limit filter |
| `test_tradability_enrich.py` | 6 | flag enrichment onto panel |
| `test_tushare_flags.py` | 3 | suspend_d/namechange/stk_limit feed |
| `test_pit_financials.py` | 7 | **ann_date as-of** (no disclosure leak) |
| `test_tushare_fina.py` | 2 | fina_indicator feed |
| `test_factors_financial.py` | 3 | financial factor (roe/netprofit_yoy) |
| `test_financial_pipeline.py` | 5 | factor dispatch + demo-source guard |
| `test_neutralize.py` | 5 | industry+size residual orthogonality |
| `test_processing_neutralize.py` | 2 | neutralize wiring (covariates required) |
| `test_covariates_enrich.py` | 3 | industry + market_cap enrichment |
| `test_tushare_covariates.py` | 5 | stock_basic + daily_basic + index_member_all SW feed (level L1/L2/L3 select, bad level) |
| `test_real_path_config.py` | 4 | demo vs real-path downgrade disclosure (+ PIT industry) |

## Per-file breakdown — Phase 2-1 real-data baseline (15)

| Test file | Tests | Feature |
|---|---|---|
| `test_phase2_baseline.py` | 15 | collectors, demo/real guard, report-field contract, no-secret-leak, settled-vs-candidate dates, loaded-vs-in-window membership, list_date + PIT-industry coverage |

## Per-file breakdown — Phase 2-2 execution realism (22)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_fills.py` | 10 | direction-aware fill sim (`simulate_fills`) + panel→feasibility adapter; cash-coherent sell-then-buy, no leverage, executed-only turnover |
| `test_driver_feasibility.py` | 6 | end-to-end: down-limit carries, up-limit blocks buy, suspended no-trade, **holdings == achieved (not desired)**, feasibility log == nav index |
| `test_min_listing_days.py` | 6 | `min_listing_days` buy-eligibility boundaries (age <, ==, >; missing list_date kept; no-op cases) |

## Per-file breakdown — Phase 2-3 PIT industry (12)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_pit_industry.py` | 12 | SW **as-of** industry (switch at reclassification, carry-forward pre-start, missing → NaN, latest-in_date on overlap) + `enrich_pit_industry` + pipeline wiring (per-date industry, configured SW level passed, no current-tag fallback) + `industry_level` config (default L1, accepts L1/L2/L3, rejects invalid, phase2 config = L1) |
| **Total (P0 + P1 + P2-1 + P2-2 + P2-3)** | **223** | |

## Real-data validation (manual, not in CI — TEST-002 keeps the suite network-free)

- **P1** (front-adjust / PIT / ann_date / neutralization): see `BIAS_AUDIT.md` and
  `artifacts/reports/phase1_summary.md`.
- **P2-1** baseline (SSE50, ~11 min): `artifacts/reports/phase2_real_baseline.md` —
  settled-date diagnostics, ann_date coverage, tradability funnel.

## Notes

- No test hits the network or reads the tushare token (TEST-002, INV-004): the whole
  suite runs on `DemoFeed` / fixtures / monkeypatched SDKs.
- **P2-2 execution realism (locked by tests):** selection eligibility and execution
  feasibility are split. `runtime.fills.simulate_fills` is the cash-coherent
  sell-then-buy model — at-up-limit blocks buys, at-down-limit blocks sells,
  suspended/missing blocks both; blocked trades carry forward, turnover/cost count
  only executed trades, and idle cash earns the driver's `cash_return` (BT-007). The
  demo path has no flags, so every trade is feasible and P0/P1 numbers are unchanged.
- `universe.min_listing_days` is enforced on the real path (list_date from
  `stock_basic`) as a buy/selection filter; a missing list_date is kept and disclosed;
  the demo path stays a disclosed no-op.
- **P2-3 PIT industry (locked by tests):** the neutralization industry covariate is
  point-in-time SW (as-of trade date via `index_member_all` in/out dates), not the
  current `stock_basic.industry` tag. The SW level is configurable
  (`processing.neutralize.industry_level`, **default L1** = 31 broad sectors, the standard
  for neutralization and DOF-safe on small cross-sections). Real runs: old tag annual
  −17.6%, SW-L1 −10.2%, SW-L2 −9.3% — **L1 ≈ L2**, so the −17.6→−10 jump is the
  **tushare→SW taxonomy switch** (inherent to going PIT — only SW carries in/out-date
  history; the old tag cannot be PIT-aligned), NOT a granularity choice. Names with no SW
  history get NaN (the neutralizer drops them) — never a silent current-tag fallback; the
  actual level + PIT coverage are disclosed in the phase2 report. The neutralize math is
  unchanged.
- A duplicate test-function name across two files was found and renamed during P2-2
  (it had been silently shadowing one test in the full-suite run). A second, harmless
  duplicate (`test_enrich_does_not_mutate_input` in two files) was verified NOT to drop
  a test (per-file sum == full-run total = 217).
