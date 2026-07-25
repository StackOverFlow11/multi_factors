"""D5 C4 groundwork: the per-day gate-attrition channel (network-free).

WHY THIS EXISTS. Four of the eleven minute factors publish a scarcity /
neutralization disclosure as an ADDED report section, and three of them build it
from a per-symbol DAY-LEVEL frame that the factor's compute function emits through
a ``diagnostics_out`` sink. The old eval runners had that frame because they ran
the compute themselves. The unified runner does not: values come from the D3 value
store through the D4/D4b materializer, and the store persists VALUES ONLY — the
day counts are not in it and cannot be recovered from it.

So the disclosure had exactly two possible fates: recompute it, or drop it. Dropping
a mandatory-by-construction disclosure because the plumbing changed is the silent
degradation this project forbids, so the channel exists and its cost is stated at
every gate it passes.

WHAT IS PINNED HERE
* asking for diagnostics NEVER changes a value (the D4b-verified path is untouched);
* the sink is emit-window-scoped, so the denominators mean the same thing they meant
  in the frozen artifacts (the days the run scored), not "the days that were loaded";
* a factor with no diagnostics binding yields an EMPTY frame and answers False to
  ``has_minute_diagnostics`` — an empty frame must never be readable as "no attrition";
* a daily factor + a sink is a readable error, not an empty sink;
* the service refuses a shared sink for several factors (the frames carry no factor
  label) and, on a WARM store, still materializes when a sink is asked for.

Every invariance claim below carries recorded mutation evidence, and each mutation
was first asserted to change its target — a sink that silently stayed empty would
make "values unchanged" pass for the wrong reason, which is the exact failure shape
this repo keeps catching in itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.availability_policy import View
from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors.compute.minute.binding import (
    _MINUTE_STREAM_BINDINGS,
    has_minute_diagnostics,
    minute_diagnostics_from_bars,
)
from factors.materialize import MaterializeSources, materialize_range
from factors.service import DecisionPoint, panel
from factors.store.values import FactorValueStore

#: Factors whose disclosure needs the day-level frame. Asserted as a SET against
#: the binding table below, so adding a fourth without a decision goes red.
DIAGNOSTIC_FACTOR_IDS = (
    "valley_ridge_vwap_ratio_20",
    "ridge_minute_return_20",
    "peak_ridge_amount_ratio_20",
)

N_SYMBOLS = 12
SYMBOLS = [f"6000{i:02d}.SH" for i in range(N_SYMBOLS)]
DATES = pd.bdate_range("2021-01-04", periods=48)
EMIT_START, EMIT_END = DATES[44], DATES[47]
BARS_PER_DAY = 238


def _bars() -> pd.DataFrame:
    rng = np.random.RandomState(31)
    rows: list[tuple] = []
    for s in SYMBOLS:
        for d in DATES:
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + rng.normal(0, 2)
            for i in range(BARS_PER_DAY):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


BARS = _bars()


class Prov:
    """Cache-free minute provider that honours ``symbols`` and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def minute_bars(self, symbols, start, end):
        self.calls += 1
        names = [str(s) for s in symbols]
        if not names:
            return BARS.iloc[0:0]
        t = BARS.index.get_level_values("time")
        sym = pd.Index(BARS.index.get_level_values("symbol"))
        return BARS[
            (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & sym.isin(names)
        ]

    def earliest_available(self, symbols):
        return DATES[0]


def _materialize(factor_id, *, diagnostics=None, provider=None):
    return materialize_range(
        factor_registry.build(factor_id),
        view=View.CLOSE,
        symbols=list(SYMBOLS),
        emit_start=EMIT_START,
        emit_end=EMIT_END,
        sources=MaterializeSources(minute=provider or Prov()),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# which factors publish a frame at all
# --------------------------------------------------------------------------- #
def test_the_diagnostic_set_is_exactly_the_declared_three():
    """Derived from the binding table, not from this list — a fourth goes red."""
    declared = {
        cls().name
        for cls, binding in _MINUTE_STREAM_BINDINGS.items()
        if binding.per_symbol_diagnostics is not None
    }
    assert declared == set(DIAGNOSTIC_FACTOR_IDS)


def test_a_factor_without_a_binding_answers_false_and_yields_an_empty_frame():
    """An empty frame is NOT "no attrition" — the caller must ask the other question."""
    factor = factor_registry.build("volume_peak_count_20")
    assert has_minute_diagnostics(factor) is False
    assert minute_diagnostics_from_bars(factor, BARS).empty


@pytest.mark.parametrize("factor_id", DIAGNOSTIC_FACTOR_IDS)
def test_a_declared_factor_answers_true(factor_id):
    assert has_minute_diagnostics(factor_registry.build(factor_id)) is True


# --------------------------------------------------------------------------- #
# the values do not move
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor_id", DIAGNOSTIC_FACTOR_IDS)
def test_asking_for_diagnostics_does_not_move_a_single_value(factor_id):
    """MUTATION EVIDENCE, and the pre-assertion that makes it non-vacuous.

    "Values unchanged with a sink" is trivially true if the sink is never filled —
    that is the shape of an unfailable test. So the sink is asserted NON-EMPTY
    first: the comparison only means something once the diagnostics path has
    actually run. Recorded mutation: returning ``pd.DataFrame()`` from
    ``_sink_diagnostics`` makes the pre-assertion FAIL (rc=1) rather than letting
    the equality pass for the wrong reason.
    """
    without = _materialize(factor_id)
    sink: list[pd.DataFrame] = []
    with_sink = _materialize(factor_id, diagnostics=sink)

    assert sink, "the diagnostics sink stayed empty — the comparison below is vacuous"
    assert sum(len(f) for f in sink) > 0
    pd.testing.assert_series_equal(without, with_sink, check_exact=True)


@pytest.mark.parametrize("factor_id", DIAGNOSTIC_FACTOR_IDS)
def test_the_sink_is_scoped_to_the_emit_window(factor_id):
    """The denominator must mean "the days this run scored", not "the days loaded".

    The engine loads warm-up / saturation history far behind ``emit_start``; letting
    those days into the frame would inflate ``symbol_days`` against the frozen
    artifacts' figure for no reason a reader could see.
    """
    sink: list[pd.DataFrame] = []
    _materialize(factor_id, diagnostics=sink)
    index = pd.DatetimeIndex(pd.concat(sink).index)
    assert index.min() >= EMIT_START and index.max() <= EMIT_END


@pytest.mark.parametrize("factor_id", DIAGNOSTIC_FACTOR_IDS)
def test_the_frames_carry_the_symbol_label_the_summarizers_read(factor_id):
    sink: list[pd.DataFrame] = []
    _materialize(factor_id, diagnostics=sink)
    frame = pd.concat(sink)
    assert "symbol" in frame.columns
    assert set(frame["symbol"]).issubset(set(SYMBOLS))
    assert "classifiable_bars" in frame.columns


def test_a_factor_without_diagnostics_leaves_the_sink_empty_rather_than_erroring():
    """Uniform call shape; whether a factor HAS a disclosure stays a separate question."""
    sink: list[pd.DataFrame] = []
    _materialize("volume_peak_count_20", diagnostics=sink)
    assert sink == []


def test_a_daily_factor_with_a_sink_is_a_readable_error():
    """Refusing beats returning an empty sink a caller would read as "no attrition"."""

    class DailyProv:
        def daily_panel(self, symbols, start, end):
            idx = pd.MultiIndex.from_product(
                [pd.bdate_range("2021-01-04", periods=60), SYMBOLS[:2]],
                names=["date", "symbol"],
            )
            return pd.DataFrame({"close": np.linspace(10.0, 20.0, len(idx))}, index=idx)

    with pytest.raises(ValueError, match="daily factor"):
        materialize_range(
            factor_registry.build("volatility_20"),
            view=View.CLOSE,
            symbols=SYMBOLS[:2],
            emit_start=EMIT_START,
            emit_end=EMIT_END,
            sources=MaterializeSources(daily=DailyProv()),
            diagnostics=[],
        )


# --------------------------------------------------------------------------- #
# the service-level contract
# --------------------------------------------------------------------------- #
def _decisions():
    return [DecisionPoint(date=d) for d in [EMIT_START, EMIT_END]]


def test_a_shared_sink_for_several_factors_is_refused(tmp_path):
    store = FactorValueStore(tmp_path / "store")
    with pytest.raises(ValueError, match="exactly ONE factor_id"):
        panel(
            ["ridge_minute_return_20", "valley_ridge_vwap_ratio_20"],
            SYMBOLS,
            _decisions(),
            store=store,
            sources=MaterializeSources(minute=Prov()),
            diagnostics=[],
        )


def test_a_warm_store_serves_values_but_a_sink_forces_the_recompute(tmp_path):
    """The disclosed cost, pinned as behaviour rather than left in a docstring.

    Warm read with NO sink: zero provider calls (the store answers). Warm read WITH
    a sink: the engine runs again, because the day counts are not in the store.
    """
    store = FactorValueStore(tmp_path / "store")
    src_cold = MaterializeSources(minute=Prov())
    cold = panel(
        ["ridge_minute_return_20"], SYMBOLS, _decisions(),
        store=store, sources=src_cold,
    )
    assert src_cold.minute.calls > 0

    warm_prov = Prov()
    warm = panel(
        ["ridge_minute_return_20"], SYMBOLS, _decisions(),
        store=store, sources=MaterializeSources(minute=warm_prov),
    )
    assert warm_prov.calls == 0, "a warm store must not re-read minute bars"
    pd.testing.assert_frame_equal(cold, warm, check_exact=True)

    diag_prov = Prov()
    sink: list[pd.DataFrame] = []
    with_diag = panel(
        ["ridge_minute_return_20"], SYMBOLS, _decisions(),
        store=store, sources=MaterializeSources(minute=diag_prov),
        diagnostics=sink,
    )
    assert diag_prov.calls > 0, "asking for diagnostics must re-run the engine"
    assert sink, "the sink must be filled by that re-run"
    # and the recompute reproduces the stored values exactly
    pd.testing.assert_frame_equal(cold, with_diag, check_exact=True)
