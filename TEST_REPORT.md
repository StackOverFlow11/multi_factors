# TEST_REPORT — Phase 1 (bias-boundary)

## Commands

Run from the repo root with the project python (env `quant_mf`):

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli run-phase0 --config config/example.yaml
```

## Results

| Gate | Command | Result |
|---|---|---|
| Unit + integration | `pytest -q` | **168 passed, 0 failed** |
| Lint | `ruff check .` | **All checks passed** |
| Config validation | `validate-config` (demo + `example_tushare.yaml`) | exit `0`, prints `OK` |
| End-to-end run | `run-phase0` (demo) | exit `0`, writes `artifacts/reports/phase0_summary.md` |

Counts below are the actual `pytest --collect-only` numbers (sum = 168).

## Per-file breakdown — Phase 0 core (93)

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
| `test_bias_audit_report.py` | 4 | bias audit doc |

## Per-file breakdown — Phase 1 bias-boundary (75)

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
| `test_tushare_covariates.py` | 2 | stock_basic + daily_basic feed |
| `test_real_path_config.py` | 3 | demo vs real-path downgrade disclosure |
| **Total (P0 + P1)** | **168** | |

## Real-data validation (manual, not in CI — TEST-002 keeps the suite network-free)

Verified directly against tushare (results recorded in `BIAS_AUDIT.md` and
`artifacts/reports/phase1_summary.md`):

- **front-adjust**: 平安银行 2024-06-14 ex-dividend — raw −5.74% vs qfq +0.99%;
  momentum_20 shifts up to 6.77pp.
- **PIT index**: CSI300 2024 — 24 snapshots, 328 distinct names (28 in / 28 out);
  `members(2024-06-15)` uses the 2024-06-03 snapshot; a dropped name stays in its era.
- **ann_date**: 平安银行 Q1 (end 2024-03-31, ann 2024-04-20) — as-of roe stays the
  prior annual (10.24) until 04-19, switches to Q1 (3.12) on 04-22; no leak.
- **neutralization**: 12 names / 4 industries — corr(momentum, log_mcap)
  −0.617 → −0.000; per-industry residual means ≈ 0.

## Notes

- No test hits the network or reads the tushare token (TEST-002, INV-004): the whole
  suite runs on `DemoFeed` / fixtures / monkeypatched SDKs.
- Financial factors, the PIT index universe, tradability filters and neutralization
  all require the real tushare path; on demo they raise a readable error rather than
  fabricate (verified by `test_financial_pipeline.py`, `_build_universe`, the
  neutralize guard).
- The demo portfolio's annualized return is intentionally extreme (demo price paths
  include a 3x jump); P0/P1 do not optimize for realistic returns — pipeline
  correctness (event order, costs, no-lookahead) is what is under test.
- P1 acceptance hardening (locked by tests): price-limit flags compare the RAW
  (unadjusted) close to the raw `stk_limit`, enriched BEFORE front-adjust
  (`test_tradability_enrich::test_limit_flag_uses_raw_close_and_survives_front_adjust`);
  financials are fetched ~16 months before `start` so the prior disclosed report
  carries forward (`test_pit_financials::test_asof_carries_forward_report_disclosed_before_window`,
  `test_financial_pipeline::test_financial_fetch_uses_lookback_before_start`);
  neutralization returns NaN on a saturated cross-section instead of fabricated ~0
  residuals (`test_neutralize::test_saturated_cross_section_returns_nan_not_zeros`).
