"""FINAL phase0 anchor: full-precision demo metrics (R10 timeline, closed at D6d).

Anchor timeline (design v3.2 §五 / R10): the OLD coarse anchor
``ic 0.9600 / annual 0.8408`` stayed REQUIRED through D1 (dispatch-only, math
untouched). D2 rewrote the factor implementations; after the hand anchors (R12)
and the frozen-baseline cell-by-cell reconciliation passed, the FULL-PRECISION
metrics of the same demo run were frozen as the PROVISIONAL new anchor
(``tests/test_provisional_anchor_d2.py``). **D6d promotes it to FINAL** — the
zero-cost CI sentinel from here on; the retired coarse anchor survives only as
the 4dp corollary of these values and is no longer asserted separately.

FINAL means: the values below are the D2-accepted engine's output and may be
re-frozen ONLY as part of an explicitly authorized engine change (with the same
reconciliation discipline), never casually edited to green a build. They are
bit-identical to what the pre-D2 engine produced (D2 reconciliation: max
relative drift 0.0 vs the D1 frozen baseline on all 14 closing factors), and
they survived every later engine step bit-for-bit: D3 store, D4/D4b/D4c
materializer, D5/C4 runner migration, D6a/D6b/D6c service switch, D6d shim
deletion + the store-wide schema-version invalidation.

The demo run is offline and deterministic (DemoFeed), so exact float equality
is the right assertion: any drift >= 1 ulp is a real engine change that must
be explained, not absorbed.
"""

from __future__ import annotations

import pytest

from qt.pipeline import run_phase0

# Full-precision values captured from the D2-accepted engine (see module
# docstring). The coarse 4dp reading of ic_mean / annual_return matches the
# retired historical anchor 0.9600 / 0.8408 exactly; that corollary is what the
# gates print.
PHASE0_ANCHOR = {
    "ic_mean": 0.9600438863169586,
    "ic_ir": 7.617904227971471,
    "annual_return": 0.8407986461861146,
    "sharpe": 10.304260918159038,
    "volatility": 0.060904296602865324,
    "max_drawdown": 0.0,
    "avg_turnover": 0.2121212121212121,
    "cost_drag": 0.002333333333333333,
}


@pytest.fixture(scope="module")
def phase0_result():
    return run_phase0("config/example.yaml")


def test_phase0_anchor_full_precision(phase0_result):
    r = phase0_result
    actual = {
        "ic_mean": r.ic_mean,
        "ic_ir": r.ic_ir,
        "annual_return": r.performance["annual_return"],
        "sharpe": r.performance["sharpe"],
        "volatility": r.performance["volatility"],
        "max_drawdown": r.performance["max_drawdown"],
        "avg_turnover": r.avg_turnover,
        "cost_drag": r.cost_drag,
    }
    mismatches = {
        k: (actual[k], expected)
        for k, expected in PHASE0_ANCHOR.items()
        if actual[k] != expected
    }
    assert not mismatches, (
        "phase0 anchor drift (full precision, exact-equality contract): "
        f"{mismatches}. An intentional engine change must re-freeze the anchor "
        "explicitly with reconciliation evidence — never edit these numbers to "
        "green a build."
    )
