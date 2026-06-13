# TEST_REPORT — Phase 0 + Phase 1 + Phase 2 + Phase 3 (bias-boundary → execution realism → PIT industry → standard analytics → multi-factor → walk-forward IC alpha → OOS stability → robustness matrix → factor candidates → subset + cost sensitivity → independent validation → CSI500 generalization → persistent market cache)

## Commands

Run from the repo root with the project python (env `quant_mf`):

```bash
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m pytest -q
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m ruff check .
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/example_tushare.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase2_real_baseline.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_multifactor.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_ic_weighted.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_oos_stability.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_robustness_matrix.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_factor_candidates.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_subset_costs.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_independent_validation.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase3_real_csi500_generalization.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli validate-config --config config/phase2_real_baseline_cached.yaml
/home/shaofl/Development/env_tools/envs/quant_mf/bin/python -m qt.cli run-phase0 --config config/example.yaml
```

## Results

| Gate | Command | Result |
|---|---|---|
| Unit + integration | `pytest -q` | **428 passed, 0 failed** |
| Lint | `ruff check .` | **All checks passed** |
| Config validation | `validate-config` (demo + `example_tushare.yaml` + `phase2_real_baseline.yaml` + `phase3_real_multifactor.yaml` + `phase3_real_ic_weighted.yaml` + `phase3_real_oos_stability.yaml` + `phase3_real_robustness_matrix.yaml` + `phase3_real_factor_candidates.yaml` + `phase3_real_subset_costs.yaml` + `phase3_real_independent_validation.yaml` + `phase3_real_csi500_generalization.yaml` + `phase2_real_baseline_cached.yaml`) | exit `0`, prints `OK` |
| End-to-end run | `run-phase0` (demo) | exit `0`, writes `artifacts/reports/phase0_summary.md` |

Counts below are the actual per-file `pytest` numbers (sum = 428).

## Per-file breakdown — Phase 0 core (97)

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
| `test_phase0_pipeline.py` | 10 | end-to-end pipeline (+ std-analytics additive / no-leak) |
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
| `test_pit_financials.py` | 8 | **ann_date as-of** (no disclosure leak; +1 P3-1 multi-field single-pass) |
| `test_tushare_fina.py` | 2 | fina_indicator feed |
| `test_factors_financial.py` | 3 | financial factor (roe/netprofit_yoy) |
| `test_financial_pipeline.py` | 5 | factor dispatch + demo-source guard |
| `test_neutralize.py` | 5 | industry+size residual orthogonality |
| `test_processing_neutralize.py` | 2 | neutralize wiring (covariates required) |
| `test_covariates_enrich.py` | 3 | industry + market_cap enrichment |
| `test_tushare_covariates.py` | 5 | stock_basic + daily_basic + index_member_all SW feed (level L1/L2/L3 select, bad level) |
| `test_real_path_config.py` | 4 | demo vs real-path downgrade disclosure (+ PIT industry) |

## Per-file breakdown — Phase 2-1 real-data baseline (16)

| Test file | Tests | Feature |
|---|---|---|
| `test_phase2_baseline.py` | 22 | collectors, demo/real guard, report-field contract, no-secret-leak, settled-vs-candidate dates, loaded-vs-in-window membership, list_date + PIT-industry coverage, standard-analytics cross-check, P3-1 factor-list/per-field-role/per-factor-table/report-name, P3-2 alpha-model disclosure (equal-weight line + ic-weighted weights/fallback/no-claim) |

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

## Per-file breakdown — Phase 2-4 standard analytics (8)

| Test file | Tests | Feature |
|---|---|---|
| `test_quantstats_adapter.py` | 4 | quantstats perf metrics + unavailable / error fallback disclosure (no silent fake) |
| `test_alphalens_adapter.py` | 4 | alphalens IC / quantile metrics + unavailable / error fallback + stdout suppression |

## Per-file breakdown — Phase 3-1 multi-factor (10)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_multifactor_pipeline.py` | 10 | all enabled factors built (order / disabled / duplicate / none), financials fetched ONCE for all fields + as-of both columns + input immutability, demo+financial readable error, e2e demo multi-factor panel + per-factor/combo analytics + primary==first, report factor list + combo + no secret, single-factor legacy shape |

## Per-file breakdown — Phase 3-2 walk-forward IC alpha (18)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_ic_weight_alpha.py` | 12 | **lookahead red-line**: perturbing unrealized forward returns cannot change weights; exact `t + h <= d` realization cutoff (min_periods boundary); insufficient-history equal-weight fallback (== EqualWeightAlpha row mean); single-factor degeneration to ±1; L1 normalization + sign preservation; degenerate-IC fallback; rolling-vs-expanding window; fit requires forward_returns; dated-cross-section contract; input immutability; weights/fallback log |
| `test_ic_alpha_pipeline.py` | 6 | alpha dispatch by config (equal_weight / ic_weighted + params / unknown = ConfigError), equal-weight default keeps exact demo numbers (ic 0.96 / annual 0.84) + report line, ic_weighted demo e2e (summary, weights log, early-fallback→late-trained, L1 rows, report disclosure, no secret), ic-weights differ from equal weight on diverging-IC synthetic data |

## Per-file breakdown — Phase 3-3 OOS stability (16)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_oos_stability.py` | 16 | **split-boundary no-leakage**: perturbing every post-split forward return leaves all train-period weights bit-identical; **holding-window slicing** (`split_nav_by_holding`: train = holding end ≤ split, test = start ≥ split, straddlers + unknown-end rows excluded and disclosed); `subperiod_perf` rebased nav + empty-slice NaN; `ic_period_stats` sliced by realization date (t+h) with mean/IR/hit-rate/n; `sign_consistent` nonzero-same-sign; `weight_sign_flips` on trained rows only (fallback rows excluded); fallback-reason aggregation; OOS config validates + split-inside-window ConfigError; runner rejects demo source / missing `oos` section / **non-ic_weighted alpha (fake-comparison guard)**; report renders boundaries / straddler disclosure / OOS metrics / weight stability / caveat / no secret |

## Per-file breakdown — Phase 3-4 robustness matrix (14)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_robustness_matrix.py` | 14 | matrix config validation (universes/windows/unique labels, split-inside-window, unknown-skip + all-skipped ConfigErrors); cell enumeration excludes skips in config order; per-cell config derivation swaps ONLY the cell identity (incl. unique per-cell output_name; derived cell passes the shared OOS preconditions); cross-cell aggregation attributes strictly per cell (no mixing, no dilution); runner guards (demo source / missing robustness section / non-ic_weighted alpha); report renders cell labels + skipped-cell disclosure + boundary/fallback/sign-consistency + cross-cell summary + caveat + no secret |

## Per-file breakdown — Phase 3-5 factor candidates (22 + 1 reworked)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_candidate_factors.py` | 22 | per-factor math vs manual references (reversal == −momentum, volatility rolling std, liquidity log-mean-amount, overnight Σlog(open/prev-close) incl. prev-close-never-crosses-symbols); leading-window NaN; future-bar perturbation invariance; per-symbol isolation; non-positive amount → NaN never −inf; missing-column readable errors; ValueFactor surfaces enriched column + rejects unknown fields; grossprofit_margin in SUPPORTED_FIELDS; dispatch builds every candidate + rejects name/params mismatch; value enrichment ONE fetch for both fields + 1/pe 1/pb math + non-positive guards + demo readable error + input immutability; demo e2e with price candidates (per-factor analytics, report, no secret); candidates config validates |
| `test_multifactor_pipeline.py` (reworked 1) | — | duplicate-name test now uses a true duplicate spec (the new name/params-mismatch guard catches the old mislabel construction earlier) |

## Per-file breakdown — Phase 3-6 subset validation + cost sensitivity (27)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_subset_validation.py` | 27 | subset config validation (groups non-empty / unique labels / non-empty unique factors / factors must reference ENABLED config factors incl. the disabled-factor case; scenario labels unique / positive multipliers / **mandatory multiplier-1.0 base scenario**; default = single base scenario); runner guards (demo source / missing `subset_validation` / missing `robustness` / non-ic_weighted alpha); `subperiod_cost` totals + arithmetic annualization + empty-slice NaN; **`_run_backtest_for(fee_rate=None)` default-preserving (old call shape bitwise identical)**; **cost scenarios change the cost line ONLY** (2× fee ⇒ identical trades/turnover/gross, exactly doubled cost, net = gross − cost); **per-group reprocessing** (drop_missing applies to the group's columns — a row killed by an excluded column's NaN survives; a group with every column reproduces `_process_factors` bitwise; missing column → readable error); cross-cell aggregation strictly per cell AND per group (incl. per-scenario cost ladder keyed by cell); report renders group/scenario disclosure + skipped cells + boundary + no-drift hook + POST-HOC & not-a-return-claim caveats + MATRIX SCOPE + union downgrades + no secret; CLI `run-phase3-subset` readable guard |
| **Total through P3-6** | **349** | |

## Per-file breakdown — Phase 3-7 independent validation (25 + 1 throttle hardening)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_independent_validation.py` | 25 | independent-cells config validation (must reference declared robustness universes/windows; a skip-listed cell cannot be declared independent — a holdout that never runs is a contradiction; hypotheses must reference ENABLED factors with a positive/negative literal; min_rebalances > 0; **the P3-6 config still validates with inert defaults**); explicit sample-class labeling (undeclared → screened, the conservative default); **verdict logic** (HOLDS iff expected IC sign in BOTH subperiods; SUPPORTED / PARTIAL / NOT SUPPORTED; **INSUFFICIENT-DATA overrides the sign check** with n_settled vs threshold disclosed; NaN or missing IC never holds); **per-class summaries never mix** (screened cell attributions never appear under independent and vice versa); report renders sample column + per-class cross-cell sections + a verdict section containing ONLY independent cells + INSUFFICIENT-DATA disclosure; a P3-6-era result (defaults) renders the old report shape; no secret; **sample-aware title/framing/caveats** (a run with independent cells must not carry the P3-6 'same windows / not independent confirmation' framing — review HIGH x2; P3-6-era rendering and downgrades text unchanged, locked by regression tests) |
| `test_tushare_throttle.py` (+1) | 1 | the DEFAULT retry budget survives a multi-failure transient outage (6 attempts ≈ 23s of capped exponential backoff; two real ~2h runs died on ConnectionError under the old 3-attempt ≈ 3s budget) |
| **Total through P3-7** | **375** | |

## Per-file breakdown — Phase 3-8 CSI500 generalization (8)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_csi500_generalization.py` | 8 | CSI500 config validates with the expected cell roles (screened anchor SSE50|2022-2024; independent SSE50|2024-2026 + 000905.SH|2024-2026; CSI500|2022-2024 skipped+disclosed; same groups/scenarios/hypotheses as P3-7 — no tuning); sample classes labeled correctly; **`output.subset_report_name`** lets each subset-validation study own its report file (default None keeps the historical `phase3_subset_validation.md` bitwise — the P3-6/P3-7 configs are locked unchanged), so a P3-8 run never clobbers the accepted P3-7 artifact; **`output.subset_report_title`** config-drives the report H1 so the CSI500 study names itself ("Phase 3-8 — CSI500 Independent Generalization Check", asserted NOT to start with Phase 3-7) while the P3-6/P3-7 configs leave it unset and keep the renderer's sample-aware default title (regression-locked) |
| **Total through P3-8** | **383** | |

## Per-file breakdown — Phase 4-1 persistent market cache (28)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_cache_config.py` | 17 | `data.cache` defaults disabled (backward compatible); fields + defaults; rejects negative refresh window / empty root / unknown key; **every existing config still validates** (parametrized over all `config/*.yaml`) |
| `test_tushare_cache_market.py` | 11 | interval subtraction (full/partial/none); **full miss populates daily+adj cache, full hit = ZERO endpoint calls**; partial gap fetches only the missing tail; empty endpoint return still records coverage (no refetch); duplicate upsert keeps one row per (symbol,date); **cached panel == direct-fetch panel byte-for-byte after `front_adjust()`**; no token / secret-path in any cache file or ledger column; **`TushareFeed.cache_stats()` exposes cold {2,2}→warm {0,0} gap-fetch counts** and `pipeline._log_cache_stats` logs `data cache: market_daily_gap_fetches=N adj_factor_gap_fetches=M` through the run-scoped logger (review follow-up: warm-hit evidence is now directly visible in the run log) |

## Per-file breakdown — Phase 4-2 universe + tradability cache (17)

| Test file | Tests | Red-line / feature |
|---|---|---|
| `test_tushare_cache_universe.py` | 17 | for each new endpoint (`index_weight`/`suspend_d`/`namechange`/`stk_limit`/`stock_basic`): **cold miss populates, warm rerun on a fresh cache over the same root = ZERO endpoint calls**; **cached feed output == direct feed output** (index_weight `assert_frame_equal` + PIT cross-section, suspend set, ST-interval sets incl. open `None` end, stk_limit frame in RAW price terms, listing-date dict); empty endpoint/snapshot return records coverage (no refetch); **FAILED fetch records NO coverage** (a later good fetch retries) — dense + snapshot; duplicate upsert keeps one row per natural key (index_weight `(date,symbol)`, suspend_d `(date,symbol,suspend_type)`); `force_refresh` re-pulls a fresh dimension snapshot; **cache-disabled path never touches the cache tree**; `pipeline._format_cache_stats` / `_log_run_cache_stats` name all five new endpoints in one line (None → no line); no token / secret-path in any cache file or ledger column across all five endpoints |
| **Total (P0 + P1 + P2-1..P2-4 + P3-1..P3-8 + P4-1 + P4-2)** | **428** | |

## Real-data validation (manual, not in CI — TEST-002 keeps the suite network-free)

- **P1** (front-adjust / PIT / ann_date / neutralization): see `BIAS_AUDIT.md` and
  `artifacts/reports/phase1_summary.md`.
- **P2-1** baseline (SSE50, ~11 min): `artifacts/reports/phase2_real_baseline.md` —
  settled-date diagnostics, ann_date coverage, tradability funnel.
- **P3-6** subset validation + cost sensitivity (3 cells × 4 groups × 3
  scenarios, ~2.1 h): `artifacts/reports/phase3_subset_validation.md` —
  THREE-layer no-drift reconciliation passed (raw-factor ICs 66/66 rows
  identical to the P3-5 matrix report; `legacy_trio`@base reproduces the
  P3-3/P3-4 numbers exactly; `full_pack`@base reproduces the P3-5 numbers
  exactly); turnover identical across cost scenarios as designed; secret scan
  0 occurrences (token value / "token" / ".config.json") in report and log.
- **P3-7** independent-sample validation (screened anchor + 2 independent
  holdout cells on 2024-07-01..2026-05-31, ~2.2 h; the P3-7 report overwrites
  the P3-6 one — same filename, regenerable): screened anchor SSE50|2022-2024
  reproduced the P3-6 numbers exactly (raw ICs 22/22 vs the P3-5 matrix
  report; all group base annuals identical); **independent verdict SUPPORTED
  in 2/2 holdout cells** (21 settled rebalances each vs minimum 8;
  value_ep/value_bp positive and volatility_20 negative in BOTH subperiods of
  BOTH cells) with visible magnitude attenuation in the later subperiod;
  secret scan 0 occurrences in report and log.
- **P4-1** persistent market cache (real smoke = phase2 baseline, non-cached
  reference vs cached cold vs cached warm): report metrics IDENTICAL across all
  three (the qfq-equivalence unit test proves cached==direct at the data layer);
  the cached WARM run makes zero market_daily / adj_factor API calls (coverage
  ledger unchanged between cold and warm); cache files + ledger carry no token /
  secret. Market bars only — index/financial endpoints still fetch live (P4-2/3). Two earlier run attempts died
  on transient ConnectionError → default retry budget hardened 3→6 attempts.
- **P4-2** universe + tradability cache (real smoke = phase2 baseline on a fresh
  temp cache root, cold → warm; the merged P4-1 `v1` cache left untouched):
  **cold** run-log line `data cache: market_daily=68 adj_factor=68
  index_weight=9 suspend_d=68 namechange=68 stk_limit=68 stock_basic=1` (all
  non-zero); coverage = 68 ok each for market/adj/namechange/stk_limit, 1 ok for
  index_weight (one gap, paged in 9 windows) and stock_basic, and **68 empty for
  suspend_d** (SSE50 large-caps had no suspensions in the window — empty-as-
  coverage). **warm** line shows **all seven endpoints = 0**, coverage ledger
  unchanged (zero new rows), and report metrics byte-identical to cold and to the
  P4-1 cached baseline (IC 0.0083 / annual −10.19% / maxDD −16.52% / vol 16.59% /
  sharpe −0.5703 / turnover 1.0818 / cost 1.19%). Wall: cold 960s → warm 366s
  (the ~594 s saved is the universe + tradability + market fetches now served from
  cache; `daily_basic` / `index_member_all` still fetch live — P4-3). Secret scan:
  0 token-value / `.config.json` occurrences across cache parquet, ledger, run
  log, and report.

## Notes

- No test hits the network or reads the tushare token (TEST-002, INV-004): the whole
  suite runs on `DemoFeed` / fixtures / monkeypatched SDKs.
- **Optional analytics extras:** `alphalens-reloaded` / `quantstats` are the
  `analytics` optional extra in `pyproject.toml` (installed in `quant_mf`). The
  adapter success-path tests `pytest.importorskip` them — a clean `.[dev]`-only
  environment SKIPS those 3 tests (disclosed) instead of failing; the
  fallback/disclosure tests monkeypatch the import and run everywhere. For the
  full 234-passed run, install `.[analytics]` too (or use `quant_mf`).
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
- **P2-4 standard analytics (locked by tests):** alphalens-reloaded and quantstats are
  thin, report-only adapters (`analytics/alphalens_adapter.py`,
  `analytics/quantstats_adapter.py`). The simple numpy/pandas metrics remain the
  authoritative backtest result and drive the run; the standard tools are an additive
  cross-check that never touches selection / portfolio / execution
  (`test_phase0_standard_analytics_is_additive_not_replacing`). Unavailable / erroring
  backends are disclosed (`backend` + exception TYPE only, no message) and keep the
  simple fallback. Empirically the alphalens IC matched the simple IC exactly on the
  demo (0.96), and the demo trading numbers (ic 0.96, annual 0.84) are unchanged.
- **P3-1 multi-factor (locked by tests):** the pipeline consumes EVERY enabled
  factor (one factor-panel column each, config order; duplicate names are a
  config error). Financial fields are fetched in ONE `fina_indicator` pass and
  as-of aligned in ONE `asof_financials` call (each field independently honours
  `ann_date <= trade_date`); demo + financial factor still raises a readable
  error. The combination is the EQUAL-WEIGHT mean of the processed columns —
  no learned weights, no forward-return fitting; `drop_missing` requires ALL
  enabled factors (a name missing any factor is dropped from that
  cross-section, disclosed). Single-factor configs keep their legacy shape and
  numbers (demo ic 0.96 / annual 0.84 unchanged); the baseline report gains the
  active factor list, per-factor coverage/IC/quantile tables, the combo-score
  diagnostics, and per-field ann_date coverage labelled TRADED vs
  diagnostic-only. `output.baseline_report_name` keeps the phase3 report file
  separate from the phase2 one.
- **P3-2 walk-forward IC alpha (locked by tests):** `alpha.model: ic_weighted`
  weights factors by mean realized rank IC over a trailing window — a
  (factor[t], fwd_h[t]) pair enters date d's weights only once REALIZED
  (`t + h <= d`, trading days); a perturbation test proves unrealized forward
  returns cannot change any date's weights. Forward returns are computed at the
  alpha boundary and handed ONLY to `alpha.fit` (the factor layer never sees
  them, invariant #1). Insufficient realized history (< min_periods valid ICs
  for any factor) falls back per-date to the EQUAL-WEIGHT mean — bitwise the
  EqualWeightAlpha combination — and is counted in the report. Weights are
  L1-normalized, sign-preserving. `EqualWeightAlpha` remains the default; its
  demo numbers (ic 0.96 / annual 0.84) are locked unchanged.
- **P3-3 OOS stability (locked by tests):** `run-phase3-oos` is a REPORT-ONLY
  validation layer — one shared data load, two backtests (equal_weight vs
  ic_weighted), every diagnostic split at `oos.split_date`. Performance slicing
  is HOLDING-WINDOW aware (a nav row's return covers [rebalance, next
  rebalance]: train rows END on/before the split, test rows START on/after it,
  straddlers are excluded from both and disclosed); IC stats slice by the
  realization date (t+h); subperiod navs are rebased so nothing bleeds across.
  Evaluation is walk-forward (rolling subperiod): the split-boundary test
  proves post-split forward returns cannot move any train-period weight. The
  runner refuses a non-ic_weighted `alpha.model` (a fake comparison guard).
  Portfolio / execution / factor math are untouched.
- **P3-4 robustness matrix (locked by tests):** `run-phase3-robustness` runs the
  UNCHANGED P3-3 cell core on every (universe × window) cell; per-cell configs
  derive from the base with only the cell identity swapped; skipped cells are
  config-validated and disclosed (never silent); cross-cell aggregation is
  strictly per cell. Single-run `run-phase3-oos` behaviour is unchanged.
- **P3-5 factor candidates (locked by tests):** conservative daily PIT-safe
  candidates (reversal_5/20, volatility_20, liquidity_20, overnight_mom_20, value_ep/bp via one
  daily_basic fetch with non-positive guards, grossprofit_margin via the
  existing ann_date machinery). The dispatch rejects a window-name/params
  mismatch; legacy configs reproduce their numbers (demo ic 0.96 / annual
  0.84 unchanged). EXPLORATORY — validated through the unchanged P3-4
  robustness matrix; not tuned, not a return claim.
- **P3-6 subset validation + cost sensitivity (locked by tests):**
  `run-phase3-subset` is a REPORT-ONLY comparison layer — per cell, ONE shared
  data load + raw factor panel (the same call sequence as the P3-3/P3-4 cell
  core, which is untouched), then each configured factor GROUP is re-processed
  independently (drop_missing per group; an all-columns group reproduces the
  old processing bitwise) and run through the same equal_weight-vs-ic_weighted
  comparison under every COST SCENARIO. Scenarios scale `cost.fee_rate` only:
  trades/turnover/gross are identical across scenarios, the cost line scales
  linearly, and `_run_backtest_for`'s new `fee_rate` parameter is
  default-preserving. POST-HOC subset selection is disclosed in the report;
  not tuned, not a return claim.
- **P3-7 independent validation (locked by tests):** independence is a HUMAN
  DECLARATION (`subset_validation.independent_cells`) — undeclared cells
  default to screened; a declared cell must exist in the matrix and must not
  be skip-listed. Hypotheses (expected IC signs) are fixed in config BEFORE
  the run; the verdict is a factual sign check (HOLDS iff the expected sign
  appears in BOTH subperiods; INSUFFICIENT-DATA when settled rebalances <
  min_rebalances, size disclosed; NaN/missing never holds). Cross-cell
  summaries are computed PER SAMPLE CLASS and the verdict section reads
  independent cells only — screened and independent numbers never mix. The
  P3-6 group/cost logic and config are unchanged.
- A duplicate test-function name across two files was found and renamed during P2-2
  (it had been silently shadowing one test in the full-suite run). A second, harmless
  duplicate (`test_enrich_does_not_mutate_input` in two files) was verified NOT to drop
  a test (per-file sum == full-run total).
