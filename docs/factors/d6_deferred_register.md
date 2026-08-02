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

## D6a-0 — INVARIANT (not a deferral): every daily factor is per-symbol

**Status:** now guarded by
`tests/test_daily_factor_universe_independence.py`. **Owner:** anyone adding a
daily factor.

The store key is `(factor_id, params, code, view)` and has **no universe
dimension** — deliberately, and it cannot have one: a PIT universe over a date
range is a time-varying *set*, not a value a key dimension could hold (design
revision A2). That is sound only while a stored value is universe-independent.

On the minute plane this failed once already and was measured: `intraday_amp_cut`
ran its cross-sectional z-score during materialization, so its stored value *was*
a function of the loaded universe while the key said otherwise — fill a store
with 12 names, read it back asking for 24, and 24 wrong cells were served with
zero recompute, zero error, zero disclosure. **D4c** fixed it by storing the
per-symbol intermediate and moving the combine to read-assembly.

**The daily plane never inherited that fix, because until D6a no daily factor
value was ever stored.** D6a is the step that puts them in the shared store, so
the property that keeps the daily plane sound becomes load-bearing here. It held
before by accident of what happened to be written; from here it holds on purpose.

The guard is **behavioural**, not a source scan: it computes every registered
daily factor over a 5-name universe and a 3-name subset and requires the shared
symbols to be bit-identical. A scan for `groupby(level="date")` / `.rank(` is
both too strict (`groupby(level="symbol").rank()` is a per-symbol time-series
rank) and too loose — mutation evidence: a cross-sectional demean written as
`x.unstack().sub(x.unstack().mean(axis=1), axis=0).stack()` turns the census red
while containing **zero** of those tokens.

**A red here is STOP-AND-REPORT.** The fix is D4c's — store the per-symbol
intermediate, combine at read-assembly — **not** an exemption list, which would
rebuild exactly the silent-wrong-value failure D4c closed.

Note this is a different question from [D6a-1](#d6a-1--provenance-dimension-on-the-store-key)
below: that one is about telling two *real vintages* apart, this one is about
whether a value is a function of *who else was loaded*.

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

**Status:** DISPOSED in D6b PR-1. The shared cache is now threaded through all
three enrichments at every call site (`qt/oos_stability.py`,
`qt/subset_validation.py`, `qt/phase2_baseline.py` — including the
ann_date-coverage diagnostic path). Configs without a `data.cache` block still
build `cache=None`, so for them the call-shape change is a byte-for-byte no-op
(the P4 `cache=None` semantics are unchanged); configs with caching enabled no
longer fetch `fina_indicator` / `daily_basic` / `index_member_all` live.

`qt/phase2_baseline.py`, `qt/oos_stability.py` and `qt/subset_validation.py`
used to call

```
_maybe_enrich_financials(cfg, panel, symbols, factors, logger)
_maybe_enrich_value(cfg, panel, symbols, factors, logger)
_maybe_enrich_covariates(cfg, panel, symbols, logger)
```

with five positional arguments, so the trailing `cache=None` default applied and
those three endpoints (`fina_indicator`, `daily_basic`, `index_member_all`) were
fetched **live on every run**, bypassing the P4 read-through cache.
`qt.pipeline.run_phase0` passed the cache to all four.

**Why it was not fixed inside D6a:** passing it *is* a behaviour change
(gap-fetch counts and wall time move, and a cached read can serve a different
vintage than a live one), and D6a's whole claim was that nothing but the
factor-sourcing path changed. Fixing it inside a path switch would have meant a
moved number with two candidate causes. D6b PR-1 owns the fix instead, paired
with the L0/L1 capture that measures exactly this vintage question.

---

## D6a-4 — `config/phase2_real_baseline.yaml` has no cache block

**Status:** DISPOSED in D6b PR-1 (for the phase-3 class the item stood for).

The un-cached phase2 config cannot run cache-only; `config/phase2_real_baseline_cached.yaml`
is its cached twin (same universe, same window) and is what D6a's real-run
reconciliation used. The same gap existed on all five `config/phase3_real_*`
validation configs (no `data.cache` block → every endpoint fetched live);
D6b PR-1 adds their cached twins (`config/phase3_real_*_cached.yaml` — byte-
identical except the added `data.cache` block), which is what the D6b L1/L1'
cache-only captures run on. The ORIGINALS stay un-cached on purpose: they are
the L0 (live legacy) capture configs, so the vintage question (D6a-5) is
measured rather than assumed away.

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

### The premise D6a's own reconciliation rests on, stated

The same live-covariate mechanism puts a condition on D6a's before/after
comparison, and it should be written down rather than left implicit:

> **The reconciliation is valid only if the covariates were not revised upstream
> between the two captures.**

They were taken 18 minutes apart (legacy 11:23, served 11:41), and because those
three endpoints are fetched live on this path (D6a-3), an upstream revision
landing in that window would have moved the served run for a reason that has
nothing to do with the change under test — in either direction, and equally able
to *mask* a real difference as to invent one.

Nothing of the sort happened here: the factor panel matched cell-for-cell and
every headline metric matched at full precision, which is itself strong evidence
that the covariates were stable across the window (a revision would have had to
leave all of them exactly unchanged). But that is evidence collected *after* the
fact, not a control designed in. A future reconciliation on this path that shows
a small movement must rule this out **before** attributing it to the code —
re-capture the legacy side and check legacy-vs-legacy first.
