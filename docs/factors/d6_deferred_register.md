# D6 deferred register

Things D6a **found and deliberately did not do**, with the reason and the step
that owns them. The point of writing them down is that a deferral nobody
recorded is indistinguishable from a defect nobody noticed.

D6a's value is that it is a **path switch provable to be a no-op**: the factor
values stop coming from `factor.compute(panel)` in the runner and start coming
from `factors.service`, and the reconciliation shows the values did not move.
Every item below would have changed something else at the same time, which is
exactly what would have destroyed that property — if a number had moved there
would no longer be a single candidate explanation.

---

## D6a-1 — provenance dimension on the store key

**Status:** deferred, structural follow-up. **Owner:** not yet assigned.

The store key is `(factor_id, params, code, view)` plus a data fingerprint over
the PIT/schema modules. Nothing in it says *which data* produced the value.

D6a closed the half of this that is a **correctness** problem — a synthetic
source's values are not a function of `(symbol, date)` at all (see
`qt/factor_source.py`), so they are refused a durable store outright.

What a provenance dimension would *additionally* buy is telling two **real**
vintages apart: a cache backfill or a tail refresh can change the bytes a
factor was computed from while the `data.clean` module hashes stay put, and the
stored value is then reused as-is. That is a **pre-existing D3 property**
(`factors/service.py::_record_fill_footprint` already records it in its closing
note), not something D6a introduced, and it applies to every consumer of the
store — the evaluation plane included.

**Why not now:** it is a change to the D3 store's identity model, it invalidates
every stored artifact when it lands, and it is orthogonal to routing the daily
runners through the service.

---

## D6a-2 — the enrichment dispatch still uses `isinstance`

**Status:** deferred. **Owner:** D6d (or its own PR).

`qt/pipeline.py::_maybe_enrich_financials` and `::_maybe_enrich_value` decide
which columns to fetch by testing `isinstance(f, FinancialFactor)` /
`isinstance(f, ValueFactor)`. That is the pattern red line #5 forbids
("依赖显式声明，编排通用消费"), and the declaration it should be reading
already exists and is already correct:

* `FinancialFactor.spec.requires == (PanelField(field, source=FINA_INDICATOR),)`
* `ValueFactor.spec.requires == (PanelField(pe|pb, source=DAILY_BASIC),)`

and `factors.registry.requirements(names, params)` — whose own docstring calls
itself "the D4 materializer's one-stop shopping list" — has **zero production
callers**.

**Note this is a pre-existing violation, not one D6a introduced.** Converting the
dispatch needs a source-endpoint → enricher routing (the `requires` name the
*source* field `pe`, while the enricher writes the *derived* column `value_ep`),
so it is a real change with its own behaviour surface, not a rename.

---

## D6a-3 — three runners do not pass the shared cache to three enrichments

**Status:** deferred, behaviour-preserving-on-purpose. **Owner:** D6b.

`qt/phase2_baseline.py`, `qt/oos_stability.py` and `qt/subset_validation.py`
call

```
_maybe_enrich_financials(cfg, panel, symbols, factors, logger)
_maybe_enrich_value(cfg, panel, symbols, factors, logger)
_maybe_enrich_covariates(cfg, panel, symbols, logger)
```

with five positional arguments, so the trailing `cache=None` default applies and
those three endpoints (`fina_indicator`, `daily_basic`, `index_member_all`) are
fetched **live on every run**, bypassing the P4 read-through cache.
`qt.pipeline.run_phase0` passes the cache to all four.

**Why not now:** passing it *is* a behaviour change (gap-fetch counts and wall
time move, and a cached read can serve a different vintage than a live one), and
D6a's whole claim is that nothing but the factor-sourcing path changed. Fixing it
inside a path switch would mean a moved number with two candidate causes.

D6b touches these three runners anyway.

---

## D6a-4 — `config/phase2_real_baseline.yaml` has no cache block

**Status:** noted. **Owner:** D6b or ops.

The un-cached phase2 config cannot run cache-only; `config/phase2_real_baseline_cached.yaml`
is its cached twin (same universe, same window) and is what D6a's real-run
reconciliation used.

---

## D6a-5 — the phase2 baseline's published headline numbers no longer reproduce

**Status:** observed, **not** caused by the refactor. **Owner:** whoever next
cites those numbers.

D6a captured a phase2 run on **unmodified `main`** before switching anything
(that capture is the reconciliation's reference). Its headline metrics do not
match the ones recorded in the progress archive for the same config:

| metric | archived | measured on `main`, 2026-07-31 |
|---|---|---|
| `ic_mean` | 0.0083 | 0.008275 |
| `annual_return` | −10.19% | −9.41% |
| `max_drawdown` | −16.52% | −16.33% |
| `volatility` | 16.59% | 16.64% |
| `sharpe` | −0.5703 | −0.5168 |
| `avg_turnover` | 1.0818 | 1.0727 |

The factor IC is unchanged, so the market panel and the factor are not what
moved; the selection is, which points at the neutralization covariates
(PIT SW industry via `index_member_all`, market cap via `daily_basic`) — both
fetched **live** on this path (see D6a-3) and both subject to upstream revision.

**Consequence for anyone reading a reconciliation:** the archived numbers are not
a valid baseline for a code change made today. Compare a capture you took
yourself, on the tree you are changing, immediately before you change it.
