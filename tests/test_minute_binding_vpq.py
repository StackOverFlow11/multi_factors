"""D5 C4b: valley_price_quantile binding + the shifted-panel third path.

Network-free. Pins BOTH sides of the lead-authorized third path for binding
valley_price_quantile (vpq) into the streaming engine:

* BINDING SIDE — the shifted-panel reversal is algebraically identical to the
  legacy construction: ``reversal_20_shifted(daily_decision_lag(closes))`` ==
  ``reversal_20(closes)`` BIT-FOR-BIT (the factor's internal T-1 shift composed
  with the decision lag would double-lag to close_{d-2}). A sabotage control
  (the shifted variant fed the UN-lagged panel) must differ materially, so the
  equivalence can never pass vacuously; a perturbation control shows both
  constructions track a changed close identically.
* RUNNER SIDE — the full service read-through (store + materializer + combine)
  reproduces the LEGACY runner semantics cell for cell: per-symbol
  ``compute_valley_price_quantile_stats`` over the full history + ONE
  cross-sectional ``residualize_on_reversal`` against ``reversal_20`` on the
  un-lagged close panel. Including the hard assertion that the left-edge
  22-trading-day window shows ZERO NaN-set divergence (the shifted panel's
  leading NaN row must land BEFORE the needed window — §六.18).

The real-cache vpq reconciliation against the frozen exec baseline is the NEXT
commit (the unified runner enablement); this file pins the semantics on fixtures.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors.compute.minute.binding import (
    RAW_QBAR_COL,
    combine_minute_stats,
    minute_combine_daily_spec,
    minute_stats_from_bars,
)
from factors.compute.minute.valley_price_quantile import (
    VALLEY_QUANTILE_REVERSAL_DAYS,
    compute_valley_price_quantile_stats,
    residualize_on_reversal,
    reversal_20,
    reversal_20_shifted,
)
from factors.materialize import MaterializeSources
from factors.service import DecisionPoint, cross_section, panel
from factors.store.values import FactorValueStore
from factors.view_lag import daily_decision_lag

FID = "valley_price_quantile_20"
#: >= VALLEY_QUANTILE_MIN_CROSS_SECTION (=10) so the per-date OLS can run.
N_SYMBOLS = 12
SYMBOLS = [f"6000{i:02d}.SH" for i in range(N_SYMBOLS)]
DATES = pd.bdate_range("2021-01-04", periods=75)
EMIT_START, EMIT_END = DATES[40], DATES[59]
BARS_PER_DAY = 238
#: The declared daily-combine warmup (rev20's 21 closes + the lag's leading row).
DAILY_WARMUP = VALLEY_QUANTILE_REVERSAL_DAYS + 2


def _minute() -> pd.DataFrame:
    rng = np.random.RandomState(5)
    rows: list[tuple] = []
    for si, s in enumerate(SYMBOLS):
        for d in DATES:
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 3 + rng.normal(0, 2)
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


def _daily() -> pd.DataFrame:
    rng = np.random.RandomState(3)
    rows = []
    for si, s in enumerate(SYMBOLS):
        px = 100.0 + si * 2 + np.cumsum(rng.normal(0, 1.0, len(DATES)))
        for d, p in zip(DATES, px):
            rows.append((d, s, p - 0.3, p + 0.5, p - 0.5, p, 1e5, p * 1e5))
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return pd.DataFrame(rows, columns=cols).set_index(["date", "symbol"]).sort_index()


MINUTE = _minute()
DAILY = _daily()
CLOSES = DAILY[["close"]]


class MinuteProv:
    """Honours ``symbols`` (like the cache reader); declares the fixture start."""

    def minute_bars(self, symbols, start, end):
        if not symbols:
            return MINUTE.iloc[0:0]
        t = MINUTE.index.get_level_values("time")
        sym = MINUTE.index.get_level_values("symbol")
        keep = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)) & sym.isin(list(symbols))
        return MINUTE[keep]

    def earliest_available(self, symbols):
        return DATES[0]


class DailyProv:
    def daily_panel(self, symbols, start, end):
        d = DAILY.index.get_level_values("date")
        sym = DAILY.index.get_level_values("symbol")
        return DAILY[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)) & sym.isin(list(symbols))]


def _sources() -> MaterializeSources:
    return MaterializeSources(daily=DailyProv(), minute=MinuteProv())


def _factor():
    return factor_registry.build(FID)


def _assert_bit_identical(got: pd.Series, want: pd.Series, label: str) -> int:
    assert got.index.equals(want.index), f"{label}: index differs"
    a, b = got.to_numpy(), want.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{label}: NaN mask differs"
    finite = ~np.isnan(a)
    assert np.array_equal(a[finite], b[finite]), f"{label}: finite values differ"
    return int(finite.sum())


# --------------------------------------------------------------------------- #
# BINDING SIDE: shifted panel + T-1 OFF  ==  un-lagged panel + internal T-1
# --------------------------------------------------------------------------- #
def test_shifted_panel_reversal_is_bit_identical_to_internal_t1():
    """The third path's core identity, exact (the D5b review measured 0.0)."""
    want = reversal_20(CLOSES).sort_index()
    got = reversal_20_shifted(daily_decision_lag(CLOSES)).sort_index()
    assert want.equals(got), "shifted+no-T1 must equal un-lagged+T1 bit-for-bit"
    assert int(want.notna().sum()) > 0, "vacuous: no finite reversal values"


def test_shifted_reversal_on_the_UNLAGGED_panel_differs_materially():
    """Sabotage control: the equivalence above cannot pass vacuously.

    Feeding the shifted variant the UN-lagged panel computes a DIFFERENT
    quantity (one day too early); the measured max|diff| on this fixture is
    ~4.6e-2 (the D5b review's sabotage scale, 7.3e-2, on its own data). If a
    future edit made the two constructions quietly identical, THIS assertion —
    not the equivalence — turns red first.
    """
    want = reversal_20(CLOSES).sort_index()
    wrong = reversal_20_shifted(CLOSES).sort_index()
    diff = (want - wrong).abs()
    assert diff.max() > 1e-3, (
        f"the shifted variant on the un-lagged panel should differ materially "
        f"(measured max|diff|={diff.max():.3e}); if it does not, the equivalence "
        f"test above is vacuous"
    )


def test_shifted_reversal_tracks_a_perturbed_close_bit_identically():
    """Perturbation control: both constructions move TOGETHER under a bad print.

    Picks a close deep inside the reversal window of the emit range, bumps it,
    and asserts the two constructions stay bit-identical AND actually moved —
    the two failure modes this guards are 'the shifted path ignores part of the
    panel' and 'the test compares two constants'.
    """
    target = CLOSES.index[int(len(CLOSES) * 0.6)]
    perturbed = CLOSES.copy()
    perturbed.loc[target, "close"] = float(perturbed.loc[target, "close"]) * 1.5
    want = reversal_20(perturbed).sort_index()
    got = reversal_20_shifted(daily_decision_lag(perturbed)).sort_index()
    assert want.equals(got), "a perturbed close must move both constructions identically"
    assert not want.equals(reversal_20(CLOSES).sort_index()), (
        "vacuous: the perturbation moved nothing"
    )


def test_combine_preserves_the_intermediate_rows_and_is_a_per_date_reduction():
    """The two combine CONTRACTS the engine relies on (DailyCombineInput).

    (1) The residualized value has EXACTLY the intermediate's rows — the pooled
    saturation loop reads the value's output dates off the intermediate without
    loading the daily panel. (2) Restricting the intermediate to the emit
    window before combining cannot change a kept date's value — the streaming
    materializer and the store slice rely on it.
    """
    factor = _factor()
    stats = minute_stats_from_bars(factor, MINUTE)
    daily = daily_decision_lag(CLOSES)
    full = combine_minute_stats(factor, stats, daily=daily)
    assert full.index.equals(stats.index), (
        "the combine must return exactly the intermediate's rows"
    )
    d = stats.index.get_level_values("date")
    windowed = stats[(d >= EMIT_START) & (d <= EMIT_END)]
    got = combine_minute_stats(factor, windowed, daily=daily)
    want = full[(full.index.get_level_values("date") >= EMIT_START)]
    want = want[want.index.get_level_values("date") <= EMIT_END]
    _assert_bit_identical(got.sort_index(), want.sort_index(), "per-date reduction")


# --------------------------------------------------------------------------- #
# RUNNER SIDE: the service read-through == the legacy runner semantics
# --------------------------------------------------------------------------- #
def _legacy_reference() -> pd.Series:
    """The LEGACY runner semantics on the same fixtures (the truth to match).

    Per-symbol ``compute_valley_price_quantile_stats`` over the FULL history
    (the legacy runner reads each symbol's whole cached minute window; the
    compute applies its own 14:50 cutoff), then ONE cross-sectional
    residualization against ``reversal_20`` on the UN-lagged close panel.
    """
    series = []
    for s in SYMBOLS:
        bars = MINUTE[MINUTE.index.get_level_values("symbol") == s]
        series.append(compute_valley_price_quantile_stats(bars, lookback_days=20))
    raw = pd.concat(series).sort_index()
    rev = reversal_20(CLOSES)
    factor = residualize_on_reversal(raw, rev, name=FID)
    d = factor.index.get_level_values("date")
    return factor[(d >= EMIT_START) & (d <= EMIT_END)].sort_index()


def _service_values(dates) -> pd.Series:
    with tempfile.TemporaryDirectory() as td:
        got = panel(
            [FID], SYMBOLS, [DecisionPoint(date=d) for d in dates],
            store=FactorValueStore(td), sources=_sources(),
        )
    return got[FID].sort_index()


EMIT_DATES = DATES[(DATES >= EMIT_START) & (DATES <= EMIT_END)]


def test_service_read_through_reproduces_the_legacy_runner_cell_for_cell():
    """service.panel(vpq) vs the legacy construction: values + NaN masks exact.

    The served panel may carry EXTRA all-NaN rows: the D4c fill footprint writes
    an explicit NaN row for every covered-but-valueless (date, symbol) cell, and
    the combine passes them through as NaN. Those rows are a recorded-shape
    disclosure, not a divergence — asserted all-NaN, then stripped for the
    cell-for-cell comparison.
    """
    got = _service_values(EMIT_DATES)
    ref = _legacy_reference()
    extra = got[~got.index.isin(ref.index)]
    assert int(extra.notna().sum()) == 0, "footprint rows must be NaN, never values"
    aligned = got[got.index.isin(ref.index)]
    n = _assert_bit_identical(aligned, ref, "service vs legacy")
    assert n > 0, "vacuous: no finite values to compare"


def test_left_edge_window_has_zero_nan_set_divergence():
    """HARD assertion on the §六.18 trap: within the first ``DAILY_WARMUP``
    trading days of the emit window the served NaN set equals the legacy one
    EXACTLY — the shifted panel's leading NaN row must not eat into it."""
    got = _service_values(EMIT_DATES)
    ref = _legacy_reference()
    edge_end = DATES[DATES.get_loc(EMIT_START) + DAILY_WARMUP - 1]
    gd = got.index.get_level_values("date")
    rd = ref.index.get_level_values("date")
    got_edge = got[gd <= edge_end]
    ref_edge = ref[rd <= edge_end]
    aligned = got_edge[got_edge.index.isin(ref_edge.index)]
    assert aligned.index.equals(ref_edge.index), "left-edge rows differ"
    assert np.array_equal(
        np.isnan(aligned.to_numpy()), np.isnan(ref_edge.to_numpy())
    ), "left-edge NaN sets diverge — the shifted panel is under-warmed"
    assert int((~np.isnan(ref_edge.to_numpy())).sum()) > 0, (
        "vacuous: the legacy reference has no finite value in the left-edge window"
    )


def test_single_date_fill_equals_batch_fill_for_vpq():
    """§3.5 P8 through the store for the daily-bound cross-sectional factor."""
    dates = pd.DatetimeIndex([EMIT_START, DATES[45], EMIT_END])
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a = FactorValueStore(ta)
        for d in dates:
            cross_section([FID], SYMBOLS, DecisionPoint(date=d), store=a, sources=_sources())
        b = FactorValueStore(tb)
        panel([FID], SYMBOLS, [DecisionPoint(date=d) for d in dates], store=b, sources=_sources())
        fa = panel([FID], SYMBOLS, [DecisionPoint(date=d) for d in dates], store=a, sources=_sources())
        fb = panel([FID], SYMBOLS, [DecisionPoint(date=d) for d in dates], store=b, sources=_sources())
    x, y = fa[FID].sort_index(), fb[FID].sort_index()
    _assert_bit_identical(x, y, "single-fill vs batch-fill (served)")
    assert int(x.notna().sum()) > 0, "vacuous: no finite served values"


# --------------------------------------------------------------------------- #
# Engine wiring: declared daily input, loud on misuse
# --------------------------------------------------------------------------- #
def test_service_refuses_a_missing_daily_provider_for_vpq():
    """No DailyPanelProvider -> a readable error at assembly, never a silent NaN."""
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="DailyPanelProvider"):
            panel(
                [FID], SYMBOLS, [DecisionPoint(date=EMIT_START)],
                store=FactorValueStore(td),
                sources=MaterializeSources(minute=MinuteProv()),
            )


def test_combine_refuses_a_daily_panel_for_a_factor_that_declares_none():
    """Passing a daily panel to a bars-only factor's combine is refused, not
    silently ignored."""
    other = factor_registry.build("volume_peak_count_20")
    stats = minute_stats_from_bars(other, MINUTE)
    with pytest.raises(ValueError, match="declares no daily input"):
        combine_minute_stats(other, stats, daily=daily_decision_lag(CLOSES))


def test_declared_daily_input_shape_is_the_pinned_third_path():
    """The declaration itself: close column only, rev20+2 lagged warmup days."""
    spec = minute_combine_daily_spec(_factor())
    assert spec is not None
    assert spec.columns == ("close",)
    assert spec.warmup_days == DAILY_WARMUP
    assert RAW_QBAR_COL == "raw_qbar"
