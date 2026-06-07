# TEST_REPORT — Phase 0

## Commands

Run from the repo root with the project python:

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli run-phase0 --config config/example.yaml
```

## Results

| Gate | Command | Result |
|---|---|---|
| Unit + integration | `pytest -q` | **105 passed, 0 failed** |
| Lint | `ruff check .` | **All checks passed** |
| Config validation | `validate-config` | exit `0`, prints `OK` |
| End-to-end run | `run-phase0` | exit `0`, writes `artifacts/reports/phase0_summary.md` |

## Per-file breakdown (full suite)

| Test file | Tests | Slice |
|---|---|---|
| `test_project_bootstrap.py` | 10 | 0 — bootstrap |
| `test_config.py` | 5 | 1 — config |
| `test_data_schema.py` | 6 | 2 — schema |
| `test_panel_store.py` | 6 | 2 — store |
| `test_data_feed.py` | 5 | 3 — feeds |
| `test_universe.py` | 4 | 4 — universe |
| `test_factors_momentum.py` | 6 | 5 — momentum |
| `test_factor_processing.py` | 4 | 6 — processing |
| `test_alpha_equal_weight.py` | 4 | 7 — alpha |
| `test_portfolio_topn.py` | 8 | 8 — portfolio |
| `test_sim_execution.py` | 8 | 9 — sim execution |
| `test_backtest_driver.py` | 7 | 10 — backtest |
| `test_analytics_factor.py` | 5 | 11 — analytics (factor) |
| `test_analytics_performance.py` | 3 | 11 — analytics (perf) |
| `test_phase0_pipeline.py` | 8 | 12 — phase0 pipeline (NEW) |
| `test_bias_audit_report.py` | 4 | 13 — bias audit (NEW) |
| `test_adjust.py` | 7 | P1 — front-adjust / qfq (NEW) |
| `test_tushare_throttle.py` | 5 | P1 — tushare rate-limit + retry (NEW) |
| **Total** | **105** | |

## New integration tests (this slice)

`tests/test_phase0_pipeline.py` (Slice 12) — runs the REAL spine
(`qt.pipeline.run_phase0`) on the offline DemoFeed, all writes redirected under
`tmp_path` (no repo `artifacts/` pollution, SEC-003):

- `test_phase0_pipeline_runs_with_demo_data` — end-to-end run returns a populated
  result (NAV table with the contract columns; performance dict with the P0
  metrics).
- `test_phase0_pipeline_writes_expected_artifacts` — `daily.parquet`,
  `factors.parquet`, `phase0_summary.md`, `run_phase0.log` all exist and live
  inside the temp output dir.
- `test_phase0_summary_mentions_static_universe_downgrade` — the summary's
  DOWNGRADES section discloses the static-universe PIT downgrade, daily-data, and
  the simple-vs-alphalens/quantstats fallback (INV-007).
- `test_phase0_pipeline_is_reentrant` — re-running over already-written files
  succeeds (INV-006).
- `test_phase0_headline_metrics_are_finite_and_sane` — monthly NAV is annualized
  at 12 periods/year, not daily 252 (HIGH-1 regression).
- `test_phase0_report_has_no_hidden_na_quantile` — quantile means stay finite;
  no hidden `inf` rendered as `n/a` (MEDIUM-1 regression).
- `test_phase0_tushare_source_requires_secret_file` — `source='tushare'` without
  a secret path raises a readable error, not silent demo fallback (HIGH-3).
- `test_phase0_tushare_source_routes_to_tushare_feed` — a tushare config
  dispatches to `TushareFeed` without network/token access in the test (HIGH-3).

`tests/test_bias_audit_report.py` (Slice 13):

- `test_bias_audit_contains_required_sections` — every required section is present
  (未来函数/lookahead, PIT 成分股, 可交易过滤, ann_date 财务对齐, 复权, 交易成本).
- `test_bias_audit_records_known_phase0_limitations` — the audit records the P0
  downgrades (static universe PIT downgrade, missing_close-only tradable filter,
  adj_factor retained, forward returns confined to analytics).
- `test_bias_audit_discloses_min_listing_days_noop` — configured-but-unenforced
  `min_listing_days` is disclosed as a no-op downgrade.
- `test_bias_audit_discloses_missing_settlement_price_convention` — missing
  settlement close is disclosed as flat 0.0 return in P0.

## Notes

- No test hits the network or reads the tushare token (TEST-002, INV-004): the
  whole suite runs on `DemoFeed` / fixtures.
- The demo portfolio's annualized return is intentionally extreme — the demo
  price paths include a 3x jump (`000005.SZ`) and P0 does NOT optimize for
  realistic returns (spec §19). The pipeline correctness (event order, costs,
  no-lookahead) is what is under test, not strategy performance.
