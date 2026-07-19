"""PR-B: the vectorized eval-IR + ``StandardFactorEvaluator``.

The load-bearing tests here are the EQUIVALENCE ones. ``analytics/eval/ir.py``
replaces a per-rebalance Python loop with grouped whole-panel reductions, and the
only way that is a speed-up rather than a rewrite of the semantics is if it
produces the SAME numbers. So this file carries a deliberately naive per-period
loop (:func:`naive_ic`, :func:`naive_quantile_returns`, written in the shape
``analytics/factor.py`` actually uses today) and asserts the vectorized objects
match it — on ragged cross-sections, NaNs, ties, single-symbol dates and all-NaN
dates, not just on a tidy rectangle.

Everything is synthetic: no network, no cache, no tushare.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from analytics.eval import (
    ADOPT,
    AXIS_FAIL,
    AXIS_INSUFFICIENT_DATA,
    AXIS_NOT_ASSESSED,
    AXIS_PASS,
    INSUFFICIENT_DATA,
    MANDATORY_SECTIONS,
    REJECT,
    WATCH,
    EvalConfig,
    EvalContext,
    FactorEvalReport,
    Section,
    Skipped,
    StandardFactorEvaluator,
    VerdictInputs,
    VerdictThresholds,
    build_eval_ir,
    decide_verdict,
)
from analytics.eval.ir import assign_quantile_buckets, quantile_turnover
from analytics.eval.stats import (
    effective_sample_size,
    half_life,
    hypothesis_win_rate,
    information_ratio_ci,
    mean_ci,
    newey_west_lag,
    newey_west_t,
    sortino,
)
from analytics.factor import compute_ic
from factors.spec import FactorSpec
from qt.intraday_groups import assign_quantile_buckets as canonical_buckets

DATE = "date"
SYMBOL = "symbol"


# ==========================================================================
# the naive per-period reference (what the vectorized IR must reproduce)
# ==========================================================================


def naive_ic(factor: pd.Series, fwd: pd.Series, dates: pd.Index) -> pd.Series:
    """Rank IC, ONE PYTHON ITERATION PER DATE. Deliberately naive.

    Mirrors ``analytics/factor.py::compute_ic`` (drop non-finite pairs, NaN when
    fewer than 2 survive or either side is constant, Spearman otherwise).
    """
    aligned = pd.DataFrame({"f": factor, "r": fwd})
    date_values = aligned.index.get_level_values(DATE)
    out: dict[object, float] = {}
    for date, block in aligned.groupby(date_values, sort=True):
        pair = block.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2 or pair["f"].nunique() < 2 or pair["r"].nunique() < 2:
            out[date] = float("nan")
        else:
            out[date] = float(pair["f"].corr(pair["r"], method="spearman"))
    return pd.Series(out, dtype=float).reindex(dates)


def naive_quantile_returns(
    factor: pd.Series, fwd: pd.Series, n_quantiles: int, dates: pd.Index
) -> pd.DataFrame:
    """Mean forward return per (date, bucket), ONE PYTHON ITERATION PER DATE.

    Same semantics the IR documents: buckets come from the FACTOR cross-section
    alone, ordered by ``(factor, symbol)`` and split by position exactly as
    ``qt.intraday_groups.assign_quantile_buckets`` does.
    """
    rows: dict[object, dict[int, float]] = {}
    factor_by_date = factor.groupby(factor.index.get_level_values(DATE), sort=True)
    for date, block in factor_by_date:
        values = (
            block.droplevel(DATE).replace([np.inf, -np.inf], np.nan).dropna()
        )
        returns = fwd.xs(date, level=DATE) if len(values) else pd.Series(dtype=float)
        order = sorted(values.index, key=lambda s: (float(values.loc[s]), str(s)))
        chunks = np.array_split(np.array(order, dtype=object), n_quantiles)
        row: dict[int, float] = {}
        for i, chunk in enumerate(chunks, start=1):
            got = pd.Series(
                [returns.get(sym, float("nan")) for sym in chunk], dtype=float
            )
            got = got.replace([np.inf, -np.inf], np.nan).dropna()
            row[i] = float(got.mean()) if len(got) else float("nan")
        rows[date] = row
    frame = pd.DataFrame(rows).T if rows else pd.DataFrame()
    return frame.reindex(index=dates, columns=list(range(1, n_quantiles + 1)))


# ==========================================================================
# synthetic fixtures
# ==========================================================================


def make_price_panel(dates: pd.Index, symbols: list[str], rng, signal=None) -> pd.DataFrame:
    """A CANONICAL market panel whose returns optionally follow a planted signal."""
    n_d, n_s = len(dates), len(symbols)
    noise = rng.normal(0.0, 0.02, size=(n_d, n_s))
    returns = noise if signal is None else noise + signal
    log_price = np.log(100.0) + np.cumsum(returns, axis=0)
    close = pd.DataFrame(np.exp(log_price), index=dates, columns=symbols)
    stacked = close.stack()
    stacked.index.names = [DATE, SYMBOL]
    stacked = stacked.sort_index()
    return pd.DataFrame(
        {
            "open": stacked,
            "high": stacked * 1.01,
            "low": stacked * 0.99,
            "close": stacked,
            "volume": 1_000_000.0,
            "amount": 100_000_000.0,
            "adj_factor": 1.0,
        }
    )


def make_spec(**overrides) -> FactorSpec:
    kwargs = dict(
        factor_id="synth_factor",
        version="1.0",
        description="a synthetic factor for tests",
        expected_ic_sign=1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
    )
    kwargs.update(overrides)
    return FactorSpec(**kwargs)


def make_cfg(**overrides) -> EvalConfig:
    kwargs = dict(
        universe="TEST500",
        universe_is_pit=True,
        start="2024-01-01",
        end="2025-01-01",
        is_exploratory=True,
        post_hoc_selected=False,
        rebalance="daily",
        n_quantiles=5,
    )
    kwargs.update(overrides)
    return EvalConfig(**kwargs)


def panel_dates(obj) -> pd.DatetimeIndex:
    """The sorted unique dates of a panel, as real Timestamps."""
    return pd.DatetimeIndex(
        pd.unique(obj.index.get_level_values(DATE)), name=DATE
    ).sort_values()


@pytest.fixture
def rng():
    return np.random.default_rng(20260716)


def ragged_panel(rng, n_dates=40, n_symbols=23):
    """A panel with ragged cross-sections, NaNs, ties and degenerate dates.

    The point is that the vectorized path must survive exactly what real A-share
    panels do: names appearing/leaving, warm-up NaNs, tied factor values, a date
    with one name and a date with nothing at all.
    """
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    panel = make_price_panel(dates, symbols, rng)
    factor = pd.Series(rng.normal(size=len(panel)), index=panel.index, name="synth_factor")

    # ties: a whole block of identical values (rank tie-break must decide)
    factor.loc[(dates[3], symbols[0]):(dates[3], symbols[9])] = 0.5
    # ragged: drop a chunk of names on some dates
    for i, date in enumerate(dates[5:15]):
        for sym in symbols[: (i % 7) + 3]:
            factor.loc[(date, sym)] = np.nan
    # a single-symbol cross-section
    for sym in symbols[1:]:
        factor.loc[(dates[20], sym)] = np.nan
    # an all-NaN cross-section
    factor.loc[dates[21]] = np.nan
    # a constant (zero-variance) cross-section
    factor.loc[dates[22]] = 7.0
    # an infinity, which must be treated as missing rather than as an extreme
    factor.loc[(dates[25], symbols[2])] = np.inf
    return panel, factor


# ==========================================================================
# 1. EQUIVALENCE: vectorized IR vs the naive per-period loop
# ==========================================================================


def test_ic_matches_naive_loop_on_ragged_panel(rng):
    panel, factor = ragged_panel(rng)
    ir = build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    expected = naive_ic(ir.factor, ir.forward_returns, ir.dates)
    assert_series_equal(
        ir.ic.rename(None), expected.rename(None), check_names=False, rtol=1e-12, atol=1e-12
    )
    # the fixture must actually exercise the degenerate paths, or this proves nothing
    assert ir.ic.isna().sum() >= 3
    assert ir.ic.notna().sum() >= 20


def test_quantile_returns_match_naive_loop_on_ragged_panel(rng):
    panel, factor = ragged_panel(rng)
    cfg = make_cfg(n_quantiles=5)
    ir = build_eval_ir(factor, make_spec(), cfg, EvalContext(price_panel=panel))
    expected = naive_quantile_returns(ir.factor, ir.forward_returns, 5, ir.dates)
    got = ir.quantile_returns.copy()
    got.columns = list(got.columns)
    expected.columns = list(expected.columns)
    assert_frame_equal(
        got, expected, check_names=False, check_column_type=False, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("n_quantiles", [2, 3, 5, 7])
def test_quantile_returns_match_naive_loop_across_bucket_counts(rng, n_quantiles):
    panel, factor = ragged_panel(rng)
    ir = build_eval_ir(
        factor, make_spec(), make_cfg(n_quantiles=n_quantiles), EvalContext(price_panel=panel)
    )
    expected = naive_quantile_returns(ir.factor, ir.forward_returns, n_quantiles, ir.dates)
    got = ir.quantile_returns.copy()
    got.columns = list(got.columns)
    expected.columns = list(expected.columns)
    assert_frame_equal(
        got, expected, check_names=False, check_column_type=False, rtol=1e-12, atol=1e-12
    )


def test_ic_matches_the_projects_authoritative_compute_ic(rng):
    """The vectorized rank IC must equal ``analytics.factor.compute_ic`` exactly.

    Not merely 'equal to my own reference': the eval layer must not silently
    disagree with the IC the rest of the framework (P3-x runners, ic_weight alpha)
    has been reporting.
    """
    panel, factor = ragged_panel(rng)
    ir = build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    expected = compute_ic(ir.factor, ir.forward_returns, method="spearman")
    assert_series_equal(
        ir.ic.rename(None), expected.rename(None), check_names=False, rtol=1e-12, atol=1e-12
    )


def test_pearson_ic_matches_compute_ic(rng):
    panel, factor = ragged_panel(rng)
    ir = build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    expected = compute_ic(ir.factor, ir.forward_returns, method="pearson")
    assert_series_equal(
        ir.ic_pearson.rename(None), expected.rename(None), check_names=False,
        rtol=1e-10, atol=1e-10,
    )


def test_bucketing_matches_the_projects_canonical_equal_count_rule(rng):
    """Vectorized buckets == ``qt.intraday_groups.assign_quantile_buckets`` (I5d/I5e).

    That function is the project's canonical equal-count rank rule and it works
    per cross-section with ``sorted() + np.array_split``. The IR reproduces it with
    integer arithmetic on grouped ranks; if the two ever diverge, an eval report's
    quintiles would stop meaning what the grouped backtest's quintiles mean.
    """
    _, factor = ragged_panel(rng)
    for n_groups in (2, 3, 5, 7):
        labels = assign_quantile_buckets(factor, n_groups)
        for date in pd.unique(factor.index.get_level_values(DATE)):
            cross_section = factor.xs(date, level=DATE).replace(
                [np.inf, -np.inf], np.nan
            )
            expected = canonical_buckets(cross_section, n_groups)
            got = (
                labels.xs(date, level=DATE).to_dict()
                if date in labels.index.get_level_values(DATE)
                else {}
            )
            assert got == expected, f"bucket mismatch on {date} with {n_groups} groups"


def test_bucket_sizes_are_equal_count_with_extras_in_the_low_buckets():
    """13 names into 5 buckets -> 3,3,3,2,2 (the np.array_split convention)."""
    dates = pd.bdate_range("2024-01-02", periods=1, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(13)]
    index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
    factor = pd.Series(np.arange(13, dtype=float), index=index)
    labels = assign_quantile_buckets(factor, 5)
    assert labels.value_counts().sort_index().tolist() == [3, 3, 3, 2, 2]
    # bucket 1 must hold the LOWEST factor values
    assert labels.xs(dates[0], level=DATE).loc[symbols[0]] == 1
    assert labels.xs(dates[0], level=DATE).loc[symbols[12]] == 5


def test_bucket_arithmetic_matches_array_split_for_every_size_and_count():
    """EXHAUSTIVE: the integer formula vs ``np.array_split``, every (n, q).

    ``assign_quantile_buckets`` inverts ``np.array_split`` with integer arithmetic
    on grouped ranks instead of materializing the chunks. That inversion is the
    single fiddliest expression in the PR (the ``base == 0`` guard, the ``n % q``
    remainder head), and an off-by-one there would silently misplace names at a
    bucket boundary on every date. So check it everywhere, not on a lucky example.
    """
    dates = pd.bdate_range("2024-01-02", periods=1, name=DATE)
    mismatches = []
    for n in range(1, 41):
        symbols = [f"{i:06d}.SZ" for i in range(n)]
        index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
        factor = pd.Series(np.arange(n, dtype=float), index=index)
        for n_quantiles in range(2, 9):
            got = dict(assign_quantile_buckets(factor, n_quantiles).xs(dates[0], level=DATE))
            expected = {
                sym: i
                for i, chunk in enumerate(
                    np.array_split(np.array(symbols, dtype=object), n_quantiles), start=1
                )
                for sym in chunk
            }
            if got != expected:
                mismatches.append((n, n_quantiles))
    assert mismatches == []


def test_fewer_names_than_buckets_leaves_high_buckets_empty():
    dates = pd.bdate_range("2024-01-02", periods=1, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(3)]
    index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
    factor = pd.Series([1.0, 2.0, 3.0], index=index)
    labels = assign_quantile_buckets(factor, 5)
    assert sorted(labels.tolist()) == [1, 2, 3]


def test_ties_are_broken_by_symbol_ascending():
    """Every factor value identical -> buckets follow symbol order, deterministically."""
    dates = pd.bdate_range("2024-01-02", periods=1, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(10)]
    index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
    factor = pd.Series(np.full(10, 3.3), index=index)
    labels = assign_quantile_buckets(factor, 5).xs(dates[0], level=DATE)
    assert [labels.loc[s] for s in symbols] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_all_nan_and_empty_cross_sections_are_nan_not_zero(rng):
    panel, factor = ragged_panel(rng)
    dates = panel_dates(factor)
    ir = build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    all_nan_date = dates[21]
    assert math.isnan(ir.ic.loc[all_nan_date])
    assert ir.quantile_returns.loc[all_nan_date].isna().all()
    assert ir.cross_section_size.loc[all_nan_date] == 0
    # a constant cross-section has no rank variance -> undefined, not 0.0
    assert math.isnan(ir.ic.loc[dates[22]])
    # a single-symbol cross-section cannot be correlated
    assert math.isnan(ir.ic.loc[dates[20]])


def test_infinity_is_treated_as_missing_not_as_an_extreme(rng):
    panel, factor = ragged_panel(rng)
    dates = panel_dates(factor)
    ir = build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    labels = ir.quantile_labels
    assert (dates[25], "000002.SZ") not in labels.index


def test_ir_build_is_deterministic(rng):
    panel, factor = ragged_panel(rng)
    ctx = EvalContext(price_panel=panel)
    first = build_eval_ir(factor, make_spec(), make_cfg(), ctx)
    second = build_eval_ir(factor, make_spec(), make_cfg(), ctx)
    assert_series_equal(first.ic, second.ic)
    assert_frame_equal(first.quantile_returns, second.quantile_returns)
    assert_frame_equal(first.quantile_turnover, second.quantile_turnover)


def test_build_does_not_mutate_the_caller_s_panels(rng):
    panel, factor = ragged_panel(rng)
    factor_before, panel_before = factor.copy(), panel.copy()
    build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))
    assert_series_equal(factor, factor_before)
    assert_frame_equal(panel, panel_before)


# ==========================================================================
# 2. the IR's PIT contract
# ==========================================================================


def test_forward_returns_are_exactly_close_t_plus_h_over_close_t(rng):
    dates = pd.bdate_range("2024-01-02", periods=12, name=DATE)
    symbols = ["000001.SZ", "000002.SZ"]
    panel = make_price_panel(dates, symbols, rng)
    factor = pd.Series(1.0, index=panel.index)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=3), make_cfg(), EvalContext(price_panel=panel)
    )
    close = panel["close"]
    for i in range(len(dates) - 3):
        expected = (
            close.loc[(dates[i + 3], "000001.SZ")] / close.loc[(dates[i], "000001.SZ")] - 1.0
        )
        assert ir.forward_returns.loc[(dates[i], "000001.SZ")] == pytest.approx(expected)
    # the last h dates cannot have realized anything
    for date in dates[-3:]:
        assert ir.forward_returns.loc[(date, "000001.SZ")] != ir.forward_returns.loc[
            (date, "000001.SZ")
        ]
        assert pd.isna(ir.realized_date.loc[date])


def test_realized_date_is_t_plus_h_on_the_return_grid(rng):
    dates = pd.bdate_range("2024-01-02", periods=10, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    factor = pd.Series(1.0, index=panel.index)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=2), make_cfg(), EvalContext(price_panel=panel)
    )
    assert ir.realized_date.loc[dates[0]] == dates[2]
    assert ir.realized_date.loc[dates[7]] == dates[9]
    assert pd.isna(ir.realized_date.loc[dates[8]])


def test_settled_rebalances_excludes_the_unrealized_tail(rng):
    panel, factor = ragged_panel(rng, n_dates=30, n_symbols=12)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=1), make_cfg(), EvalContext(price_panel=panel)
    )
    assert ir.settled_rebalances == int(ir.ic.notna().sum())
    assert ir.settled_rebalances < ir.n_rebalances  # the tail never settles


def test_poisoning_prices_beyond_t_plus_h_cannot_move_an_earlier_ic(rng):
    """Lookahead lock by perturbation — the project's standard technique.

    Scramble every close from grid position 20 onward. No period whose forward
    return realizes at or before position 19 may notice.

    Two traps this test had to avoid, both of which would make it vacuous:
      * a UNIFORM scale factor leaves the cross-sectional RANKING untouched, so
        rank IC is invariant to it and nothing would move even with a real leak.
        Hence a per-symbol multiplier.
      * for t >= 20 BOTH close[t] and close[t+h] are poisoned and the multiplier
        cancels in the ratio — so only the boundary periods (t = 18, 19) can
        legitimately move. The test asserts they DO, or the poison proved nothing.
    """
    dates = pd.bdate_range("2024-01-02", periods=30, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(15)]
    panel = make_price_panel(dates, symbols, rng)
    factor = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    spec = make_spec(forward_return_horizon=2)
    base = build_eval_ir(factor, spec, make_cfg(), EvalContext(price_panel=panel))

    poisoned = panel.copy()
    mask = poisoned.index.get_level_values(DATE) >= dates[20]
    multiplier = pd.Series(rng.uniform(0.2, 5.0, len(symbols)), index=symbols)
    poisoned.loc[mask, "close"] = poisoned.loc[mask, "close"] * poisoned.loc[
        mask
    ].index.get_level_values(SYMBOL).map(multiplier)
    after = build_eval_ir(factor, spec, make_cfg(), EvalContext(price_panel=poisoned))

    moved_early = [
        d for d in dates[:18]
        if not np.allclose(base.ic[d], after.ic[d], equal_nan=True)
    ]
    assert moved_early == [], "LOOKAHEAD: an IC changed from data strictly after t+h"
    moved_late = sum(
        1 for d in dates[18:28]
        if not np.allclose(base.ic[d], after.ic[d], equal_nan=True)
    )
    assert moved_late > 0, "the poison was impotent, so the assertion above proves nothing"


def test_exec_to_exec_without_supplied_returns_is_a_readable_error(rng):
    dates = pd.bdate_range("2024-01-02", periods=5, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    factor = pd.Series(1.0, index=panel.index)
    spec = make_spec(
        is_intraday=True,
        return_basis="exec_to_exec",
        decision_cutoff="14:50:00",
        data_lag="1min",
        session_open="09:30:00",
        execution_model="next_minute_close",
        execution_window="[14:51,14:56:59]",
    )
    with pytest.raises(ValueError, match="execution-anchored"):
        build_eval_ir(factor, spec, make_cfg(), EvalContext(price_panel=panel))


def test_no_returns_at_all_is_a_readable_error(rng):
    dates = pd.bdate_range("2024-01-02", periods=5, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    factor = pd.Series(1.0, index=panel.index)
    with pytest.raises(ValueError, match="no forward returns available"):
        build_eval_ir(factor, make_spec(), make_cfg(), EvalContext())


def test_factor_dates_absent_from_the_price_panel_are_rejected(rng):
    dates = pd.bdate_range("2024-01-02", periods=8, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    stray = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2030-01-01"), "000001.SZ")], names=[DATE, SYMBOL]
    )
    factor = pd.concat([pd.Series(1.0, index=panel.index), pd.Series(1.0, index=stray)])
    with pytest.raises(ValueError, match="absent from the price panel"):
        build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))


def monthly_grid_setup(rng, n_days=200):
    """A factor on the FIRST trading day of each month, over a DAILY price panel."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, name=DATE)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    panel = make_price_panel(dates, symbols, rng)
    first_of_month = pd.DatetimeIndex(
        sorted(pd.Series(dates).groupby([dates.year, dates.month]).min()), name=DATE
    )
    index = pd.MultiIndex.from_product([first_of_month, symbols], names=[DATE, SYMBOL])
    factor = pd.Series(rng.normal(size=len(index)), index=index)
    return panel, factor, dates, first_of_month, symbols


def test_horizon_counts_evaluation_periods_not_price_panel_rows(rng):
    """``FactorSpec.forward_return_horizon`` is "in evaluation periods" (frozen contract).

    A monthly factor over a daily price panel with h=1 must be scored on its
    next-PERIOD return. Delegating ``periods=(1,)`` to the whole price panel would
    silently score it on the next TRADING DAY instead — a completely different
    (and much easier) claim, on a horizon nobody asked about.
    """
    panel, factor, dates, months, symbols = monthly_grid_setup(rng)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=1), make_cfg(rebalance="monthly"),
        EvalContext(price_panel=panel),
    )
    assert ir.median_period_gap_days > 25  # the grid really is sparse
    close = panel["close"]
    first, second = ir.dates[0], ir.dates[1]

    next_period = close.loc[(second, symbols[0])] / close.loc[(first, symbols[0])] - 1.0
    next_day = (
        close.loc[(dates[dates.get_loc(first) + 1], symbols[0])]
        / close.loc[(first, symbols[0])]
        - 1.0
    )
    assert ir.forward_returns.loc[(first, symbols[0])] == pytest.approx(next_period)
    assert ir.forward_returns.loc[(first, symbols[0])] != pytest.approx(next_day)
    # the realized date is the next EVALUATION PERIOD, not the next trading day
    assert ir.realized_date.loc[first] == second
    assert "evaluation periods" in ir.forward_return_source
    assert "restricted" in ir.forward_return_source


def test_dense_grid_is_unaffected_by_the_evaluation_period_resolution(rng):
    """When F's grid IS the price grid, h=trading days=periods — nothing changed."""
    dates = pd.bdate_range("2024-01-02", periods=20, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    factor = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=3), make_cfg(),
        EvalContext(price_panel=panel),
    )
    close = panel["close"]
    expected = close.loc[(dates[3], "000001.SZ")] / close.loc[(dates[0], "000001.SZ")] - 1.0
    assert ir.forward_returns.loc[(dates[0], "000001.SZ")] == pytest.approx(expected)
    assert "restricted" not in ir.forward_return_source


def test_duplicate_panel_rows_are_rejected(rng):
    dates = pd.bdate_range("2024-01-02", periods=4, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ"], rng)
    factor = pd.concat([pd.Series(1.0, index=panel.index)] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        build_eval_ir(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))


def test_ambiguous_multi_column_frame_is_rejected(rng):
    dates = pd.bdate_range("2024-01-02", periods=4, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    frame = pd.DataFrame({"a": 1.0, "b": 2.0}, index=panel.index)
    with pytest.raises(ValueError, match="none is named"):
        build_eval_ir(frame, make_spec(), make_cfg(), EvalContext(price_panel=panel))


# ==========================================================================
# 3. turnover / stats primitives
# ==========================================================================


def test_turnover_charges_the_first_period_and_a_full_rotation():
    dates = pd.bdate_range("2024-01-02", periods=2, name=DATE)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
    # day 1 ranks 1,2,3,4 -> day 2 exactly reversed: bucket 1 and 2 fully swap
    factor = pd.Series([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], index=index)
    labels = assign_quantile_buckets(factor, 2)
    turnover = quantile_turnover(labels, 2, pd.Index(dates, name=DATE))
    # establishing an equal-weight book of 2 names is sum|w| = 1.0
    assert turnover.loc[dates[0], 1] == pytest.approx(1.0)
    # a complete rotation sells 1.0 and buys 1.0
    assert turnover.loc[dates[1], 1] == pytest.approx(2.0)


def test_turnover_is_zero_when_membership_does_not_change():
    dates = pd.bdate_range("2024-01-02", periods=3, name=DATE)
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    index = pd.MultiIndex.from_product([dates, symbols], names=[DATE, SYMBOL])
    factor = pd.Series([1.0, 2.0, 3.0, 4.0] * 3, index=index)
    turnover = quantile_turnover(
        assign_quantile_buckets(factor, 2), 2, pd.Index(dates, name=DATE)
    )
    assert turnover.loc[dates[1], 1] == pytest.approx(0.0)
    assert turnover.loc[dates[2], 2] == pytest.approx(0.0)


def test_newey_west_t_is_smaller_than_the_naive_t_on_an_autocorrelated_series():
    """The whole reason design §A demands it: the IID t overstates significance."""
    rng = np.random.default_rng(7)
    n = 400
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.02)
    series = pd.Series(x + 0.05, index=pd.RangeIndex(n))
    result = newey_west_t(series)
    assert result["lags"] >= 1
    assert abs(result["t"]) < abs(result["t_iid"])
    # not a rounding difference: the naive t is materially inflated here
    assert abs(result["t"]) < 0.6 * abs(result["t_iid"])


def test_newey_west_t_on_white_noise_is_close_to_the_naive_t():
    rng = np.random.default_rng(11)
    series = pd.Series(rng.normal(0.01, 0.1, 500))
    result = newey_west_t(series)
    assert result["t"] == pytest.approx(result["t_iid"], rel=0.35)


def reference_newey_west_t(series: pd.Series, lags: int) -> float:
    """Deliberately dumb NW-t: an explicit double loop over (t, j) pairs.

    A pair contributes ONLY when both endpoints are observed — the definition of
    "the gap breaks the lag pair". Obviously correct, obviously slow.
    """
    values = pd.Series(series, dtype=float).to_numpy(dtype=float)
    total = len(values)
    valid = np.isfinite(values)
    n = int(valid.sum())
    mean = float(values[valid].mean())
    gamma = []
    for j in range(lags + 1):
        acc = 0.0
        for t in range(j, total):
            if valid[t] and valid[t - j]:
                acc += (values[t] - mean) * (values[t - j] - mean)
        gamma.append(acc / n)
    variance = gamma[0] + 2.0 * sum(
        (1.0 - j / (lags + 1.0)) * gamma[j] for j in range(1, lags + 1)
    )
    return mean / math.sqrt(variance / n)


def test_newey_west_t_breaks_lag_pairs_across_holes_instead_of_bridging_them():
    """An IC series HAS holes (degenerate cross-sections yield NaN).

    Dropping them and then lagging pairs observations that are j SURVIVING ROWS
    apart while calling them j PERIODS apart — a bridged gap, i.e. an
    autocovariance that does not exist. The statistic whose entire job is to stop
    the IID t from overstating significance must not itself be quietly wrong.
    """
    rng = np.random.default_rng(23)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 0.02)
    series = pd.Series(x + 0.04)
    # punch interior holes (not just a tail): exactly the degenerate-cross-section
    # pattern a real IC series shows.
    for i in (17, 18, 60, 61, 62, 140, 201, 202, 250):
        series.iloc[i] = np.nan
    assert series.iloc[1:-1].isna().sum() == 9  # holes really are INTERIOR

    got = newey_west_t(series, lags=4)
    assert got["t"] == pytest.approx(reference_newey_west_t(series, 4), rel=1e-12)
    assert got["n"] == 300 - 9
    assert got["n_dropped"] == 9

    # ... and the drop-then-lag version really does differ, or this proves nothing
    bridged = newey_west_t(series.dropna().reset_index(drop=True), lags=4)
    assert not np.isclose(got["t"], bridged["t"], rtol=1e-6)


def test_newey_west_t_without_holes_is_unchanged_by_the_gap_handling():
    """A hole-free series must give exactly what it always gave."""
    rng = np.random.default_rng(29)
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0, 0.03)
    series = pd.Series(x + 0.05)
    got = newey_west_t(series, lags=3)
    assert got["t"] == pytest.approx(reference_newey_west_t(series, 3), rel=1e-12)
    assert got["n_dropped"] == 0
    # with no holes, zero-filling is a no-op: identical to the dropna() path
    assert got["t"] == pytest.approx(newey_west_t(series.dropna(), lags=3)["t"], rel=1e-12)


def test_newey_west_t_ignores_a_trailing_nan_tail_like_a_settled_ic_series():
    """The last h periods of a real IC series never settle -> a NaN TAIL.

    A tail has no interior pair to break, so tail-only NaN must not change the
    estimate at all relative to simply trimming it.
    """
    rng = np.random.default_rng(31)
    values = pd.Series(rng.normal(0.02, 0.1, 120))
    with_tail = pd.concat([values, pd.Series([np.nan] * 5)], ignore_index=True)
    assert newey_west_t(with_tail, lags=3)["t"] == pytest.approx(
        newey_west_t(values, lags=3)["t"], rel=1e-12
    )


def test_newey_west_t_degenerate_inputs_are_nan_not_zero():
    assert math.isnan(newey_west_t(pd.Series([], dtype=float))["t"])
    assert math.isnan(newey_west_t(pd.Series([1.0]))["t"])
    constant = newey_west_t(pd.Series([2.0] * 30))
    assert math.isnan(constant["t"])  # zero variance -> undefined, never +inf


# --------------------------------------------------------------------------
# effective_sample_size — gate part A (design §6, v0.3)
# --------------------------------------------------------------------------


def _ar1(rho: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return pd.Series(x)


def test_a_long_but_heavily_autocorrelated_daily_ic_series_has_few_effective_samples():
    """⭐ THE test the three-part gate exists for.

    ~500 daily IC points with rho_1 ~ 0.95 are NOT 500 pieces of evidence: the
    AR(1) truth is N_eff = N*(1-rho)/(1+rho) ~ 12.8. The raw count sails past any
    raw floor; the effective count does not.

    ⚠️ ASSERTED OVER MANY DRAWS ON PURPOSE — the estimator is NOISY at this rho,
    and a single seed would be cherry-picking. Measured over 200 draws:
    median 17.9 (biased UP from the 12.8 truth — rho-hat is downward-biased and
    the positive-sequence truncation stops early), range 5.9..35.6, and
    P(N_eff < 24) = 80%. So the DEFAULT gate catches the typical such series but
    NOT every draw; what is invariant is the order of magnitude (P(N_eff<100) =
    100%, i.e. always a >5x cut from 500). Both facts are asserted; neither is
    tuned away. At rho=0.99 it is decisive (see the next test).
    """
    values = np.array(
        [float(effective_sample_size(_ar1(0.95, 500, seed=s))["n_eff"]) for s in range(40)]
    )
    # invariant: never anywhere near the raw count.
    assert values.max() < 100.0
    # the typical draw is gated by the default min_effective_samples=24 ...
    assert np.median(values) < 24.0
    # ... and most draws are, though NOT all (~80% at this rho).
    assert (values < 24.0).mean() >= 0.6

    got = effective_sample_size(_ar1(0.95, 500, seed=0))
    assert got["n"] == 500
    # it looked PAST the Newey-West floor, which alone truncates at ~5 lags and
    # reports a ~4x too permissive N_eff (~53 vs a ~13 truth).
    assert got["lags"] > got["lags_nw"]
    assert got["sum_rho"] > 1.0

    # end to end: the TYPICAL such series gates the Predictive axis on part A
    # (no OOS/book/exec facts either -> the deployment label is INSUFFICIENT-DATA).
    result = decide_verdict(
        VerdictInputs(
            expected_ic_sign=+1,
            settled_rebalances=500,
            effective_samples=float(np.median(values)),
            span_days=700.0,
        )
    )
    assert result.verdict == INSUFFICIENT_DATA
    assert result.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any(r.startswith("effective samples (A)") for r in result.predictive.reasons)


def test_a_near_unit_root_ic_series_is_gated_on_every_draw():
    """rho=0.99: the truth is N_eff ~ 2.5 and the gate catches it 99.5% of the
    time (200 draws). The noise that makes rho=0.95 a ~80% catch does not rescue
    a series this persistent."""
    values = np.array(
        [float(effective_sample_size(_ar1(0.99, 500, seed=s))["n_eff"]) for s in range(40)]
    )
    assert values.max() < 24.0


def test_an_iid_series_keeps_essentially_all_of_its_samples():
    """The other end: independent observations must NOT be penalized."""
    for seed in range(4):
        series = pd.Series(np.random.default_rng(seed).normal(size=500))
        got = effective_sample_size(series)
        assert got["n_eff"] > 0.6 * 500  # ~N, up to rho-hat noise
        assert got["n_eff"] <= 500       # never MORE than N


def test_a_perfectly_correlated_series_carries_one_sample():
    """A constant series holds exactly one distinct value: N_eff = 1, not 0/0."""
    got = effective_sample_size(pd.Series(np.ones(100)))
    assert got["n_eff"] == 1.0
    assert "constant" in str(got["status"])
    assert math.isfinite(got["n_eff"])


def test_effective_sample_size_is_monotone_in_the_autocorrelation():
    """More persistence must never buy MORE effective samples."""
    values = [effective_sample_size(_ar1(r, 400, seed=5))["n_eff"] for r in (0.0, 0.5, 0.9, 0.99)]
    assert values == sorted(values, reverse=True)


def test_effective_sample_size_never_emits_nan_inf_or_a_negative():
    """Every guarded path still yields a finite, non-negative, <= N number."""
    cases = {
        "empty": pd.Series([], dtype=float),
        "one": pd.Series([0.4]),
        "all_nan": pd.Series([np.nan, np.nan, np.nan]),
        "constant": pd.Series(np.ones(50)),
        "two_points": pd.Series([0.1, -0.1]),
        "alternating": pd.Series([1.0, -1.0] * 50),       # extreme anti-correlation
        "with_inf": pd.Series([1.0, np.inf, -1.0, 2.0, 0.5] * 10),
        "ar1_099": _ar1(0.99, 300, seed=3),
    }
    for name, series in cases.items():
        got = effective_sample_size(series)
        value = got["n_eff"]
        assert isinstance(value, float), name
        assert math.isfinite(value), name
        assert value >= 0.0, name
        assert value <= got["n"] or got["n"] == 0, name


def test_an_anti_correlated_series_is_clamped_to_n_never_credited_beyond_it():
    """A negative 1+2*sum(rho) must not become a negative or an inflated N_eff.

    An anti-correlated series is 'at least as informative as i.i.d.'; we credit
    exactly i.i.d. and no more, because a noisy rho-hat is precisely what would
    otherwise manufacture the evidence a short run does not have.
    """
    got = effective_sample_size(pd.Series([1.0, -1.0] * 50))
    assert got["denominator"] <= 0 or got["n_eff"] == got["n"]
    assert got["n_eff"] == 100.0  # == N, clamped
    assert math.isfinite(got["n_eff"])


def test_effective_sample_size_breaks_lag_pairs_across_holes_like_the_nw_t():
    """Same gap rule as newey_west_t: a hole DROPS the pair, never bridges it."""
    series = _ar1(0.9, 300, seed=7)
    holed = series.copy()
    for i in (17, 18, 60, 61, 140, 201, 250):
        holed.iloc[i] = np.nan
    got = effective_sample_size(holed)
    assert got["n"] == 300 - 7  # holes are not counted as observations
    # dropping-then-lagging (the bridge) genuinely differs -> the rule has teeth
    bridged = effective_sample_size(holed.dropna().reset_index(drop=True))
    assert not np.isclose(got["n_eff"], bridged["n_eff"], rtol=1e-6)


def test_effective_sample_size_shares_the_newey_west_lag_floor():
    """The two statistics must not disagree about the autocorrelation structure:
    the ESS window is anchored at the NW-t's bandwidth and only ever EXTENDS it."""
    for n in (30, 120, 500):
        got = effective_sample_size(pd.Series(np.random.default_rng(1).normal(size=n)))
        assert got["lags_nw"] == newey_west_lag(n)
        assert got["lags"] >= got["lags_nw"]
    # an explicit lags= disables the extension: the caller chose the truncation.
    fixed = effective_sample_size(_ar1(0.95, 500, seed=11), lags=5)
    assert fixed["lags"] == 5.0 and fixed["lags_nw"] == 5.0


def test_hypothesis_win_rate_is_relative_to_the_expected_sign():
    series = pd.Series([0.1, 0.2, -0.3, float("nan"), 0.0])
    assert hypothesis_win_rate(series, 1) == pytest.approx(0.5)  # 2 of 4 finite
    assert hypothesis_win_rate(series, -1) == pytest.approx(0.25)
    assert math.isnan(hypothesis_win_rate(pd.Series([], dtype=float), 1))


def test_half_life_is_defined_only_for_a_decaying_signal():
    assert half_life(0.5) == pytest.approx(1.0)
    assert math.isnan(half_life(-0.4))  # flips rather than decays
    assert math.isnan(half_life(1.0))  # never decays
    assert math.isnan(half_life(float("nan")))


def test_sortino_is_nan_without_downside():
    # no negative period -> the downside deviation is 0 and the ratio is undefined.
    # +inf would render as a spectacular fake, so NaN it is.
    assert math.isnan(sortino(pd.Series([0.01, 0.02, 0.03]), 252))
    assert math.isnan(sortino(pd.Series([0.01]), 252))  # too short to measure


def test_sortino_penalizes_only_the_downside():
    """Same mean, same vol, but one series has its dispersion on the upside."""
    downside_heavy = pd.Series([0.02, -0.04, 0.02, 0.04, -0.04, 0.04] * 4)
    upside_heavy = pd.Series([0.01, 0.01, 0.01, 0.01, -0.005, 0.06] * 4)
    assert sortino(downside_heavy, 252) < sortino(upside_heavy, 252)
    assert sortino(pd.Series([-0.01, -0.02, 0.005, -0.03] * 4), 252) < 0


# ==========================================================================
# 4. the sections
# ==========================================================================


def planted_signal_setup(rng, n_dates=180, n_symbols=40, strength=0.02, sign=1):
    """A panel where tomorrow's return genuinely follows today's factor."""
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    raw = rng.normal(size=(n_dates, n_symbols))
    # the planted edge: return[t] is driven by factor[t-1]
    signal = np.zeros_like(raw)
    signal[1:] = sign * strength * raw[:-1]
    panel = make_price_panel(dates, symbols, rng, signal=signal)
    factor = pd.DataFrame(raw, index=dates, columns=symbols).stack()
    factor.index.names = [DATE, SYMBOL]
    return panel, factor.sort_index().rename("synth_factor")


def incremental_book(rng, factor):
    """A pure-NOISE known-factor book the factor is genuinely incremental to.

    Independent of the factor, so residualizing the factor on it leaves the signal
    essentially intact -> the Incremental axis PASSes. (Contrast ``redundant_setup``,
    which reconstructs the factor -> the residual is ~ 0 -> the axis FAILs.)
    """
    return pd.DataFrame(
        {"noise_anchor": pd.Series(rng.normal(size=len(factor)), index=factor.index)}
    )


def redundant_setup(rng, n_dates=300, n_symbols=40, strength=0.05, noise_scale=0.5):
    """A factor that DUPLICATES a supplied known factor: strong RAW IC, ~0 residual.

    Returns follow the ANCHOR, and the factor is ``anchor + noise`` — so the factor
    predicts returns (strong raw IC, sign-consistent out-of-sample) yet residualizing
    it on the anchor book leaves ONLY the noise, which is independent of returns. Its
    incremental IC is therefore ~ 0: the Incremental axis FAILs and the factor is
    rejected DESPITE its perfect predictive/tradable story. This is the value_ep-twin
    scenario the three-axis verdict exists to catch."""
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    anchor_raw = rng.normal(size=(n_dates, n_symbols))
    signal = np.zeros_like(anchor_raw)
    signal[1:] = strength * anchor_raw[:-1]  # returns follow the ANCHOR, not the noise
    panel = make_price_panel(dates, symbols, rng, signal=signal)
    noise = rng.normal(size=(n_dates, n_symbols)) * noise_scale
    factor = pd.DataFrame(anchor_raw + noise, index=dates, columns=symbols).stack()
    factor.index.names = [DATE, SYMBOL]
    anchor = pd.DataFrame(anchor_raw, index=dates, columns=symbols).stack()
    anchor.index.names = [DATE, SYMBOL]
    book = pd.DataFrame({"anchor": anchor.sort_index()})
    return panel, factor.sort_index().rename("synth_factor"), book


def test_all_eight_sections_are_present_and_ordered(rng):
    panel, factor = planted_signal_setup(rng)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    assert [s.name for s in report.sections] == list(MANDATORY_SECTIONS)
    report.validate_all_mandatory_present()


def test_planted_signal_is_detected_with_the_expected_sign(rng):
    panel, factor = planted_signal_setup(rng)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    predictive = report.by_name()["predictive_power"]
    assert isinstance(predictive, Section)
    assert predictive.payload["ic_mean"] > 0.10
    assert predictive.payload["ic_win_rate"] > 0.70
    assert predictive.payload["ic_ir"] > 0.30
    returns = report.by_name()["return_risk"]
    assert returns.payload["monotonicity_spearman"] == pytest.approx(1.0)
    assert returns.payload["net_long_short_by_cost"][1.0] > 0


def test_negative_hypothesis_factor_reads_in_its_own_direction(rng):
    """A -1 factor (low-vol style) must not be punished for a negative Spearman."""
    panel, factor = planted_signal_setup(rng, sign=-1)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(expected_ic_sign=-1), make_cfg(), EvalContext(price_panel=panel)
    )
    predictive = report.by_name()["predictive_power"]
    returns = report.by_name()["return_risk"]
    assert predictive.payload["ic_mean"] < 0  # raw IC is negative...
    assert predictive.payload["ic_win_rate"] > 0.70  # ...but that IS the hypothesis
    assert returns.payload["monotonicity_spearman"] == pytest.approx(-1.0)  # RAW
    assert returns.payload["net_long_short_by_cost"][1.0] < 0  # RAW


def test_cost_scenarios_only_move_the_cost_line(rng):
    panel, factor = planted_signal_setup(rng)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    returns = report.by_name()["return_risk"]
    by_cost = returns.payload["net_long_short_by_cost"]
    # the SAME trades at k x the fee: strictly monotone degradation, gross fixed
    assert by_cost[1.0] > by_cost[2.0] > by_cost[4.0]
    stability = report.by_name()["stability_cost"]
    drag = stability.payload["mean_cost_per_period_by_scenario"]
    assert drag[2.0] == pytest.approx(2 * drag[1.0])
    assert drag[4.0] == pytest.approx(4 * drag[1.0])
    assert stability.payload["extra_cost_vs_base_by_scenario"][1.0] == pytest.approx(0.0)


def test_return_risk_note_labels_the_synthetic_leg_difference(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    note = report.by_name()["return_risk"].note
    assert "SYNTHETIC LONG-ONLY LEG DIFFERENCE" in note
    assert "NOT a dollar-neutral executed portfolio" in note


def test_purity_is_skipped_without_anchors_and_says_why(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    purity = report.by_name()["purity"]
    assert isinstance(purity, Skipped)
    assert "known_factors was not supplied" in purity.reason


def test_purity_reports_correlation_and_orthogonalized_ic_when_anchored(rng):
    panel, factor = planted_signal_setup(rng, n_dates=90, n_symbols=30)
    # "near_twin" explains almost all of the factor -> the orthogonalized IC must
    # collapse. "noise" explains none of it -> the signal must survive.
    anchors = pd.DataFrame(
        {
            "near_twin": factor + pd.Series(
                rng.normal(scale=0.05, size=len(factor)), index=factor.index
            ),
            "noise": pd.Series(rng.normal(size=len(factor)), index=factor.index),
        }
    )
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(),
        EvalContext(price_panel=panel, known_factors=anchors),
    )
    purity = report.by_name()["purity"]
    assert isinstance(purity, Section)
    raw_ic = purity.payload["ic_mean_raw"]
    assert purity.payload["mean_rank_corr_with_anchor"]["near_twin"] > 0.95
    assert abs(purity.payload["mean_rank_corr_with_anchor"]["noise"]) < 0.1
    # removing the near-twin removes the signal; removing noise does not
    assert abs(purity.payload["ic_mean_orthogonalized_vs_anchor"]["near_twin"]) < 0.3 * raw_ic
    assert purity.payload["ic_mean_orthogonalized_vs_anchor"]["noise"] > 0.7 * raw_ic
    assert purity.payload["vif"] is None
    assert "NOT COMPUTED" in purity.payload["vif_status"]


def test_orthogonalizing_against_a_perfect_twin_is_nan_not_zero(rng):
    """A residual of exactly zero has no rank variance: the IC is UNDEFINED.

    NaN is the honest answer — a 0.0 would read as "the signal survived
    orthogonalization and turned out to be worthless", which is a different claim.
    """
    panel, factor = planted_signal_setup(rng, n_dates=60, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(),
        EvalContext(price_panel=panel, known_factors=pd.DataFrame({"twin": factor})),
    )
    purity = report.by_name()["purity"]
    assert purity.payload["mean_rank_corr_with_anchor"]["twin"] == pytest.approx(1.0)
    assert math.isnan(purity.payload["ic_mean_orthogonalized_vs_anchor"]["twin"])


def test_execution_capacity_is_skipped_with_a_precise_reason(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    section = report.by_name()["execution_capacity"]
    assert isinstance(section, Skipped)
    assert "NOT WIRED" in section.reason
    assert "cannot reach the Adopt verdict" in section.reason


def test_execution_capacity_passes_through_measured_facts(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(),
        EvalContext(
            price_panel=panel,
            execution_capacity={"tradable": True, "capacity_sufficient": False,
                                "capacity_ratio_median": 1.104},
        ),
    )
    section = report.by_name()["execution_capacity"]
    assert isinstance(section, Section)
    assert section.payload["tradable"] is True
    assert section.payload["capacity_sufficient"] is False
    assert "MEASURED OUTSIDE this evaluator" in section.payload["source"]


def test_execution_capacity_marks_a_missing_flag_as_unknown(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(),
        EvalContext(price_panel=panel, execution_capacity={"tradable": True}),
    )
    section = report.by_name()["execution_capacity"]
    assert "UNKNOWN" in section.payload["capacity_sufficient_status"]


def test_data_coverage_reports_the_sample_and_the_dropped_names(rng):
    panel, factor = ragged_panel(rng)
    declared = [f"{i:06d}.SZ" for i in range(30)]  # 7 names never reach the panel
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(min_history_bars=2), make_cfg(),
        EvalContext(price_panel=panel, universe_symbols=tuple(declared)),
    )
    coverage = report.by_name()["data_coverage"]
    assert coverage.payload["settled_rebalances"] == int(
        report.by_name()["predictive_power"].payload["ic_periods_finite"]
    )
    assert coverage.payload["symbols_evaluated"] == 23
    assert coverage.payload["dropped_symbols_count"] == 7
    assert coverage.payload["universe_symbols_declared"] == 30
    assert coverage.payload["factor_nan_rate"] > 0
    assert coverage.payload["warmup_rows_excluded"] == 2 * 23


def test_data_coverage_says_when_the_universe_was_not_declared(rng):
    panel, factor = ragged_panel(rng)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    coverage = report.by_name()["data_coverage"]
    assert coverage.payload["dropped_symbols_count"] is None
    assert "NOT ASSESSED" in coverage.payload["dropped_symbols_status"]


def test_daily_panel_declared_monthly_is_rejected_not_silently_counted(rng):
    """The sample gate must not be satisfiable by a sample that does not exist.

    A daily panel declared 'monthly' would otherwise report ~250 settled
    rebalances and walk straight through min_rebalances=24 — the gate whose entire
    job is deciding INSUFFICIENT-DATA. Disclosure is not enough for a GATE.
    """
    panel, factor = planted_signal_setup(rng, n_dates=250, n_symbols=20)
    with pytest.raises(ValueError, match="declares evaluation periods") as excinfo:
        StandardFactorEvaluator().evaluate(
            factor, make_spec(), make_cfg(rebalance="monthly"),
            EvalContext(price_panel=panel),
        )
    message = str(excinfo.value)
    assert "'monthly'" in message                 # names the declared frequency
    assert "median 1.00 days apart" in message    # names the DETECTED one
    assert "min_rebalances" in message            # names the gate it would defeat
    assert "does not resample" in message


def test_min_rebalances_gates_on_the_real_count_of_a_matching_grid(rng):
    """A truly monthly grid yields ~monthly periods -> INSUFFICIENT-DATA, honestly."""
    panel, factor, _, months, _ = monthly_grid_setup(rng, n_days=250)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(rebalance="monthly"), EvalContext(price_panel=panel)
    )
    coverage = report.by_name()["data_coverage"]
    # ~12 months of data, NOT the ~250 rows the daily price panel carries
    assert coverage.payload["evaluation_periods"] == len(months)
    assert coverage.payload["settled_rebalances"] < 24
    assert coverage.payload["rebalance_grid_check"].startswith("OK")
    # and the gate now bites on the REAL sample size
    assert report.require_verdict().verdict == INSUFFICIENT_DATA


def test_a_matching_daily_grid_passes_the_check(rng):
    panel, factor = planted_signal_setup(rng, n_dates=60, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(rebalance="daily"), EvalContext(price_panel=panel)
    )
    coverage = report.by_name()["data_coverage"]
    assert coverage.payload["rebalance_grid_check"].startswith("OK")
    assert coverage.payload["median_period_gap_days"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("label", "should_raise"),
    [("daily", True), ("weekly", True), ("monthly", False), ("quarterly", True)],
)
def test_rebalance_grid_check_accepts_only_the_matching_declaration(rng, label, should_raise):
    """A ~monthly grid is accepted as 'monthly' and rejected as anything else."""
    panel, factor, _, _, _ = monthly_grid_setup(rng, n_days=250)
    ctx = EvalContext(price_panel=panel)
    if should_raise:
        with pytest.raises(ValueError, match="declares evaluation periods"):
            build_eval_ir(factor, make_spec(), make_cfg(rebalance=label), ctx)
    else:
        ir = build_eval_ir(factor, make_spec(), make_cfg(rebalance=label), ctx)
        assert ir.rebalance_grid_check.startswith("OK")


def test_an_unrecognized_rebalance_label_is_disclosed_as_unverified(rng):
    """A minute-frequency study has no expected spacing here — say so, don't guess.

    The declaration cannot be checked, so the report must not imply it was.
    """
    panel, factor = planted_signal_setup(rng, n_dates=60, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(rebalance="5min"), EvalContext(price_panel=panel)
    )
    coverage = report.by_name()["data_coverage"]
    check = coverage.payload["rebalance_grid_check"]
    assert check.startswith("NOT CHECKED")
    assert "UNVERIFIED" in check
    # and the annualization fallback is disclosed too, not silently assumed
    returns = report.by_name()["return_risk"]
    assert "DEFAULT 252" in returns.payload["periods_per_year_basis"]


def test_grid_check_is_not_checked_for_a_single_period(rng):
    dates = pd.bdate_range("2024-01-02", periods=1, name=DATE)
    panel = make_price_panel(dates, ["000001.SZ", "000002.SZ"], rng)
    factor = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    ir = build_eval_ir(
        factor, make_spec(), make_cfg(rebalance="monthly"), EvalContext(price_panel=panel)
    )
    assert ir.rebalance_grid_check.startswith("NOT CHECKED")


def test_h_on_the_factor_grid_still_holds_under_the_rebalance_check(rng):
    """The two fixes must coexist: a monthly grid passes the check AND h=1 period."""
    panel, factor, dates, months, symbols = monthly_grid_setup(rng)
    ir = build_eval_ir(
        factor, make_spec(forward_return_horizon=1), make_cfg(rebalance="monthly"),
        EvalContext(price_panel=panel),
    )
    assert ir.rebalance_grid_check.startswith("OK")
    close = panel["close"]
    first, second = ir.dates[0], ir.dates[1]
    expected = close.loc[(second, symbols[0])] / close.loc[(first, symbols[0])] - 1.0
    assert ir.forward_returns.loc[(first, symbols[0])] == pytest.approx(expected)
    assert ir.realized_date.loc[first] == second


def test_caveats_carry_the_honesty_flags(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(),
        make_cfg(
            is_exploratory=True, post_hoc_selected=True, tuned=True,
            universe_is_pit=False, n_factors_screened=11,
        ),
        EvalContext(price_panel=panel),
    )
    caveats = report.by_name()["caveats"].payload
    joined = " ".join(caveats["caveats"])
    assert "EXPLORATORY" in joined
    assert "POST-HOC SELECTED" in joined
    assert "TUNED" in joined
    assert "NON-PIT UNIVERSE" in joined
    assert "NO OOS SPLIT" in joined
    assert caveats["bonferroni_alpha_for_5pct_family"] == pytest.approx(0.05 / 11)


def test_caveats_flag_an_undeclared_multiple_testing_background(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    note = report.by_name()["caveats"].payload["multiple_testing_note"]
    assert "UNDECLARED" in note


# ==========================================================================
# 5. OOS & the verdict (end to end)
# ==========================================================================


def test_oos_is_skipped_without_a_split_and_says_so(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    section = report.by_name()["oos_generalization"]
    assert isinstance(section, Skipped)
    assert "cfg.oos_split not set" in section.reason
    assert "no OOS evidence" in section.reason


def test_oos_splits_by_the_realized_date_not_the_signal_date(rng):
    panel, factor = planted_signal_setup(rng, n_dates=120, n_symbols=25)
    dates = panel_dates(factor)
    split = dates[60]
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(forward_return_horizon=5), make_cfg(oos_split=str(split.date())),
        EvalContext(price_panel=panel),
    )
    section = report.by_name()["oos_generalization"]
    assert section.payload["split_basis"].startswith("realized date")
    # h=5 and the split sits at grid position 60. A period is train only if its
    # return has ALREADY been realized by the split, i.e. t+5 < split, i.e. t < 55.
    assert section.payload["train_periods_settled"] == 55
    # ... and NOT 60, which is what slicing by the SIGNAL date would give. That
    # off-by-h is precisely the P3-3 bug: it let 5 periods of post-split return
    # leak into the train segment.
    assert section.payload["train_periods_settled"] != 60
    assert "realized" in section.note.lower()


def test_oos_sign_consistency_on_a_genuine_signal(rng):
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40)
    dates = panel_dates(factor)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(oos_split=str(dates[130].date())),
        EvalContext(price_panel=panel),
    )
    section = report.by_name()["oos_generalization"]
    assert section.payload["oos_available"] is True
    assert section.payload["sign_consistent"] is True
    assert section.payload["sign_flipped"] is False
    assert section.payload["monotonicity_reversed"] is False
    assert section.payload["independent_cells_evaluated"] == 0


def test_oos_sign_flip_is_detected_and_hard_rejects(rng):
    """The project's signature failure: in-sample positive, out-of-sample negative."""
    n_dates, n_symbols = 300, 40
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    raw = rng.normal(size=(n_dates, n_symbols))
    signal = np.zeros_like(raw)
    half = n_dates // 2
    signal[1:half] = 0.03 * raw[: half - 1]     # train: the hypothesis holds
    signal[half:] = -0.03 * raw[half - 1 : -1]  # test: it reverses
    panel = make_price_panel(dates, symbols, rng, signal=signal)
    factor = pd.DataFrame(raw, index=dates, columns=symbols).stack()
    factor.index.names = [DATE, SYMBOL]
    factor = factor.sort_index()

    # DEFAULT thresholds — no gate-opening fixture. Since v0.4 the hard Rejects
    # are decided BEFORE the sample gate, so the flip this run MEASURED is
    # reported as such even though a regime-flip IC series carries only ~3
    # effective samples (pinned in its own test below). The sample gate guards
    # POSITIVE claims; a visible failure is not one.
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(oos_split=str(dates[half].date())),
        EvalContext(price_panel=panel),
    )
    section = report.by_name()["oos_generalization"]
    assert section.payload["train_ic_mean"] > 0
    assert section.payload["test_ic_mean"] < 0
    assert section.payload["sign_flipped"] is True
    assert section.payload["sign_consistent"] is False
    assert report.require_verdict().verdict == REJECT
    assert any("flipped" in r for r in report.require_verdict().reasons)


def test_a_regime_flip_ic_series_carries_almost_no_independent_evidence(rng):
    """The N_eff measurement itself, pinned deliberately (it is CORRECT).

    An IC series that is positive for one regime and negative for the next is a
    STEP FUNCTION: rho-hat stays positive for ~100 lags, so N_eff collapses to ~3
    out of ~300 raw periods. That is not a defect — two regimes ARE about two
    independent observations, and "the factor flipped" is then indistinguishable
    from "we saw two draws". The ESTIMATOR is doing its job and this test keeps
    pinning it.

    What the v0.4 ruling changed is what the VERDICT does with it. Under v0.3 the
    gate ran first, so this run drew no conclusion at all — the tool refused to
    say "bad" about the project's signature failure mode (I5e / P3-3 / P3-4).
    The gate exists to stop OVERCLAIMING, so it now blocks only the POSITIVE
    claims: a measured flip is a NEGATIVE finding and rejects on any sample size.
    The thin N_eff below is exactly why this run would never have been allowed an
    Adopt or a Watch.
    """
    n_dates, n_symbols = 300, 40
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    raw = rng.normal(size=(n_dates, n_symbols))
    signal = np.zeros_like(raw)
    half = n_dates // 2
    signal[1:half] = 0.03 * raw[: half - 1]
    signal[half:] = -0.03 * raw[half - 1 : -1]
    panel = make_price_panel(dates, symbols, rng, signal=signal)
    factor = pd.DataFrame(raw, index=dates, columns=symbols).stack()
    factor.index.names = [DATE, SYMBOL]
    factor = factor.sort_index()

    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(oos_split=str(dates[half].date())),
        EvalContext(price_panel=panel),
    )
    coverage = report.by_name()["data_coverage"].payload
    assert coverage["settled_rebalances"] > 250      # plenty of RAW periods
    assert coverage["effective_samples"] < 10        # ... and almost no independent ones
    assert coverage["span_days"] > 365               # not a span problem
    assert report.by_name()["oos_generalization"].payload["sign_flipped"] is True

    # v0.4: the measured flip is REPORTED, not swallowed by the gate.
    verdict = report.require_verdict()
    assert verdict.verdict == REJECT
    assert any("flipped" in r for r in verdict.reasons)
    # ... and it rejects ON THE FLIP, not on a mumbled word about the sample.
    assert not any(r.startswith("effective samples (A)") for r in verdict.reasons)

    # The gate is not gone, only NARROWED to the positive claims: this very sample
    # would still refuse a Predictive PASS. Same coverage facts, flip flag cleared.
    # (The Tradable axis is non-statistical, so it can still PASS -> the deployment
    # label is WATCH, but crucially NOT Adopt: the predictive claim stays gated.)
    gated = decide_verdict(
        VerdictInputs(
            expected_ic_sign=make_spec().expected_ic_sign,
            settled_rebalances=coverage["settled_rebalances"],
            effective_samples=coverage["effective_samples"],
            span_days=coverage["span_days"],
            ic_ir=0.8, ic_win_rate=0.7, ic_nw_t=3.5, monotonicity_spearman=0.9,
            net_long_short_by_cost=((1.0, 0.05),),
            oos_available=True, oos_sign_consistent=True,
            tradable=True, capacity_sufficient=True,
        )
    )
    assert gated.verdict != ADOPT
    assert gated.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any(r.startswith("effective samples (A)") for r in gated.predictive.reasons)


def test_same_cell_reversal_never_claims_independent_cell_evidence(rng):
    """A holdout reversal must NOT be reported as an independent-cell reversal.

    ``verdict.py`` documents ``oos_monotonicity_reversed`` as "# independent-cell
    reversal (I5e)" and its Reject reason reads "quantile monotonicity reversed on
    an independent cell". This evaluator scores ONE cell and reports
    independent_cells_evaluated=0 in the very same payload — so writing a same-cell
    number into that key would make the report claim an independent check ran and
    failed when none ran at all. The same-cell figure belongs under its own name.
    """
    n_dates, n_symbols = 300, 40
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name=DATE)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    raw = rng.normal(size=(n_dates, n_symbols))
    signal = np.zeros_like(raw)
    half = n_dates // 2
    signal[1:half] = 0.03 * raw[: half - 1]     # train: hypothesis holds
    signal[half:] = -0.03 * raw[half - 1 : -1]  # test: quantiles reverse
    panel = make_price_panel(dates, symbols, rng, signal=signal)
    factor = pd.DataFrame(raw, index=dates, columns=symbols).stack()
    factor.index.names = [DATE, SYMBOL]
    # DEFAULT thresholds: since v0.4 the hard Rejects precede the sample gate, so
    # this run reaches the rule under test (the independent-cell CLAIM) on the
    # strength of its MEASURED sign flip — no gate-opening fixture needed.
    report = StandardFactorEvaluator().evaluate(
        factor.sort_index(), make_spec(), make_cfg(oos_split=str(dates[half].date())),
        EvalContext(price_panel=panel),
    )
    section = report.by_name()["oos_generalization"]
    # the same-cell holdout genuinely DID reverse ...
    assert section.payload["test_monotonicity_spearman"] == pytest.approx(-1.0)
    assert section.payload["test_monotonicity_aligned"] < 0
    # ... but no independent cell was ever evaluated, so the contract's key stays
    # False and the report must never utter the independent-cell claim.
    assert section.payload["independent_cells_evaluated"] == 0
    assert section.payload["monotonicity_reversed"] is False
    assert "NOT ASSESSED" in section.payload["monotonicity_reversed_status"]

    verdict = report.require_verdict()
    assert not any("independent cell" in reason for reason in verdict.reasons)
    # the factor is still rejected — on the evidence that WAS gathered (the sign flip)
    assert verdict.verdict == REJECT
    assert any("sign flipped" in reason for reason in verdict.reasons)


def test_unparseable_oos_split_raises_instead_of_degrading_silently(rng):
    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    with pytest.raises(ValueError, match="not a parseable date"):
        StandardFactorEvaluator().evaluate(
            factor, make_spec(), make_cfg(oos_split="not-a-date"),
            EvalContext(price_panel=panel),
        )


def test_no_oos_evidence_can_never_reach_adopt(rng):
    """The contract's central promise, verified END TO END on a strong factor.

    The planted signal is deliberately huge — in-sample everything passes. Without
    a holdout it must still stop at Watch.
    """
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(),  # no oos_split
        EvalContext(
            price_panel=panel,
            # even with tradability fully established:
            execution_capacity={"tradable": True, "capacity_sufficient": True},
        ),
    )
    verdict = report.require_verdict()
    assert verdict.verdict != ADOPT
    assert verdict.verdict == WATCH
    assert any("no out-of-sample split" in r.lower() for r in verdict.reasons)


def test_insufficient_data_short_circuits_before_any_conclusion(rng):
    panel, factor = planted_signal_setup(rng, n_dates=12, n_symbols=20, strength=0.05)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    verdict = report.require_verdict()
    assert verdict.verdict == INSUFFICIENT_DATA
    assert verdict.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("settled rebalances" in r for r in verdict.reasons)


def test_adopt_requires_oos_plus_tradability_plus_a_positive_base_spread(rng):
    """The only path to Adopt: every leg of the §6 rule satisfied by real facts.

    ``is_exploratory=False`` is one of those legs since v0.2 — make_cfg() declares
    an EXPLORATORY run by default, which the §6 cap holds at Watch. This is the
    end-to-end guarantee that Adopt stays REACHABLE for a run that claims it.
    """
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    dates = panel_dates(factor)
    # ADOPT now needs all three axes to PASS -> a known-factor book the factor is
    # INCREMENTAL to (a default run, with no book, tops out at Watch by design).
    ctx = EvalContext(
        price_panel=panel,
        known_factors=incremental_book(rng, factor),
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    cfg = make_cfg(oos_split=str(dates[130].date()), is_exploratory=False)
    report = StandardFactorEvaluator().evaluate(factor, make_spec(), cfg, ctx)
    verdict = report.require_verdict()
    assert verdict.verdict == ADOPT
    assert verdict.incremental.verdict == AXIS_PASS

    # ... and remove ONLY the execution evidence: Adopt must fall back to Watch.
    without_execution = StandardFactorEvaluator().evaluate(
        factor,
        make_spec(),
        cfg,
        EvalContext(price_panel=panel, known_factors=incremental_book(rng, factor)),
    )
    assert without_execution.require_verdict().verdict == WATCH
    assert without_execution.require_verdict().tradable.verdict == AXIS_NOT_ASSESSED


def test_an_exploratory_declaration_caps_an_adopt_grade_run_at_watch(rng):
    """Design §6 (v0.2) END TO END: the identical facts that earn Adopt above stop
    at Watch when the run declares itself exploratory — and stop at WATCH, not
    Reject, because the declaration caps a claim rather than failing the factor.
    """
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    dates = panel_dates(factor)
    ctx = EvalContext(
        price_panel=panel,
        known_factors=incremental_book(rng, factor),
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    cfg = make_cfg(oos_split=str(dates[130].date()), is_exploratory=True)
    verdict = (
        StandardFactorEvaluator()
        .evaluate(factor, make_spec(), cfg, ctx)
        .require_verdict()
    )
    # all three axes PASS, but the exploratory flag caps the LABEL at Watch.
    assert verdict.verdict == WATCH
    assert all(a.verdict == AXIS_PASS for a in verdict.axes().values())
    assert any("is_exploratory=True" in r for r in verdict.reasons)
    assert any("CAPPED AT WATCH" in r for r in verdict.reasons)
    # the qualifying OOS evidence is still reported, not buried by the cap.
    assert any("out-of-sample subperiods" in r for r in verdict.reasons)


def test_post_hoc_selection_cannot_reach_adopt_end_to_end(rng):
    """§4 forces post_hoc_selected=True => is_exploratory=True, and the §6 cap then
    denies Adopt: a factor picked after seeing these results cannot be reported as
    a confirmation, no matter how good the numbers are."""
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    dates = panel_dates(factor)
    ctx = EvalContext(
        price_panel=panel,
        known_factors=incremental_book(rng, factor),
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    cfg = make_cfg(
        oos_split=str(dates[130].date()), is_exploratory=True, post_hoc_selected=True
    )
    verdict = (
        StandardFactorEvaluator()
        .evaluate(factor, make_spec(), cfg, ctx)
        .require_verdict()
    )
    assert verdict.verdict == WATCH
    assert any("is_exploratory=True" in r for r in verdict.reasons)


def test_a_factor_redundant_with_the_book_is_rejected_end_to_end(rng):
    """⭐ THE headline new capability (design §6, v0.5): a factor that DUPLICATES a
    supplied known factor is REJECTED — even though its raw predictive signal is
    strong, its out-of-sample sign is consistent, and it is fully tradable. Judging
    a factor in isolation would have Adopted it; the Incremental axis catches that a
    cross-sectional multi-factor book already has this signal (a value_ep twin gets
    rejected)."""
    panel, factor, book = redundant_setup(rng, n_dates=300, n_symbols=40)
    dates = panel_dates(factor)
    ctx = EvalContext(
        price_panel=panel,
        known_factors=book,
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    cfg = make_cfg(oos_split=str(dates[130].date()), is_exploratory=False)
    report = StandardFactorEvaluator().evaluate(factor, make_spec(), cfg, ctx)
    verdict = report.require_verdict()

    # the RAW predictive story is genuinely strong AND tradable ...
    assert verdict.predictive.verdict == AXIS_PASS
    assert verdict.tradable.verdict == AXIS_PASS
    purity = report.by_name()["purity"]
    assert purity.payload["ic_mean_raw"] > 0.10  # a real raw signal
    # ... but residualizing on the book collapses it ~ to zero.
    assert abs(purity.payload["ic_mean_orthogonalized_vs_book"]) < 0.3 * purity.payload[
        "ic_mean_raw"
    ]
    assert verdict.incremental.verdict == AXIS_FAIL
    assert verdict.verdict == REJECT
    assert any("incremental" in r for r in verdict.reasons)  # names the axis
    # the FAIL explains itself: it collapsed after residualizing on the book.
    assert any(
        "known-factor book" in r for r in verdict.incremental.reasons
    )


def test_a_factor_incremental_to_a_noise_book_passes_the_incremental_axis(rng):
    """The mirror of the headline: a book the factor does NOT duplicate leaves the
    orthogonalized IC essentially intact -> the Incremental axis PASSes."""
    # n_dates=300 clears the sample gate (span >= 365 calendar days) so the axis
    # can be PASS rather than INSUFFICIENT_DATA.
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    report = StandardFactorEvaluator().evaluate(
        factor,
        make_spec(),
        make_cfg(),
        EvalContext(price_panel=panel, known_factors=incremental_book(rng, factor)),
    )
    purity = report.by_name()["purity"]
    assert purity.payload["known_factors_supplied"] is True
    raw = purity.payload["ic_mean_raw"]
    # the book explains almost none of the signal, so the residual IC survives.
    assert purity.payload["ic_mean_orthogonalized_vs_book"] > 0.7 * raw
    assert report.require_verdict().incremental.verdict == AXIS_PASS


def test_verdict_thresholds_are_honoured(rng):
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    dates = panel_dates(factor)
    cfg = make_cfg(oos_split=str(dates[130].date()))
    strict = VerdictThresholds(min_rebalances=10_000)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), cfg, EvalContext(price_panel=panel), thresholds=strict
    )
    assert report.require_verdict().verdict == INSUFFICIENT_DATA


# ==========================================================================
# 6. report rendering / export
# ==========================================================================


def test_report_renders_and_exports_deterministically(rng):
    panel, factor = planted_signal_setup(rng, n_dates=60, n_symbols=20)
    ctx = EvalContext(price_panel=panel)
    evaluator = StandardFactorEvaluator()
    first = evaluator.evaluate(factor, make_spec(), make_cfg(), ctx)
    second = evaluator.evaluate(factor, make_spec(), make_cfg(), ctx)
    assert first.render() == second.render()
    assert first.to_json() == second.to_json()

    markdown = first.render()
    for heading in (
        "## 2. Predictive Power", "## 3. Return & Risk", "## 4. Stability & Cost",
        "## 5. Purity", "## 6. OOS & Generalization", "## 7. Execution & Capacity",
        "## 8. Data & Coverage", "## 9. Caveats & Provenance",
    ):
        assert heading in markdown
    assert "_MISSING" not in markdown

    exported = first.to_dict()
    assert [s["name"] for s in exported["sections"]] == list(MANDATORY_SECTIONS)
    assert all(s["status"] in ("ok", "skipped") for s in exported["sections"])
    assert exported["verdict"]["verdict"] in {ADOPT, WATCH, REJECT, INSUFFICIENT_DATA}


def test_report_export_has_no_missing_sections_for_the_standard_evaluator(rng):
    panel, factor = planted_signal_setup(rng, n_dates=60, n_symbols=20)
    report = StandardFactorEvaluator().evaluate(
        factor, make_spec(), make_cfg(), EvalContext(price_panel=panel)
    )
    statuses = {s["name"]: s["status"] for s in report.to_dict()["sections"]}
    assert set(statuses) == set(MANDATORY_SECTIONS)
    assert "missing" not in statuses.values()


def test_a_subclass_cannot_drop_a_mandatory_section(rng):
    """Enforcement layer #2 still bites a StandardFactorEvaluator subclass."""

    class Sneaky(StandardFactorEvaluator):
        def evaluate(self, factor_panel, spec, cfg, ctx=None, thresholds=None):
            ir = self.build_ir(factor_panel, spec, cfg, ctx)
            partial = FactorEvalReport.assemble(
                spec, cfg, [self.predictive_power(ir)]
            )
            return partial.with_verdict()

    panel, factor = planted_signal_setup(rng, n_dates=40, n_symbols=20)
    with pytest.raises(ValueError, match="missing mandatory section"):
        Sneaky().evaluate(factor, make_spec(), make_cfg(), EvalContext(price_panel=panel))


def test_evaluator_rejects_a_non_evalcontext_ctx(rng):
    panel, factor = planted_signal_setup(rng, n_dates=20, n_symbols=10)
    with pytest.raises(TypeError, match="EvalContext"):
        StandardFactorEvaluator().evaluate(
            factor, make_spec(), make_cfg(), {"price_panel": panel}
        )


# ==========================================================================
# #3: N_eff confidence intervals + pre-registered criteria (design §6, v0.6)
# ==========================================================================


def _normalize(values, mean, std):
    """Rescale a series to a target (mean, sample-std), preserving autocorrelation."""
    s = pd.Series(np.asarray(values), dtype=float)
    return (s - s.mean()) / s.std(ddof=1) * std + mean


def test_information_ratio_ci_uses_n_eff_and_gates_correlated_out_where_iid_passes():
    """⭐ THE whole point of the N_eff CI: two IC series of the SAME raw length and
    the SAME ICIR point, one i.i.d. and one heavily autocorrelated. The correlated
    one carries far fewer EFFECTIVE observations, so its CI is wider and its LOWER
    bound falls below the bar that the i.i.d. one's lower bound clears."""
    rng = np.random.default_rng(7)
    iid = _normalize(rng.normal(size=500), mean=0.5, std=1.0)
    ar = _normalize(_ar1(0.9, 500, seed=7), mean=0.5, std=1.0)
    ci_iid = information_ratio_ci(iid)
    ci_ar = information_ratio_ci(ar)

    assert ci_iid["point"] == pytest.approx(0.5)  # identical ICIR points ...
    assert ci_ar["point"] == pytest.approx(0.5)
    assert ci_ar["n_eff"] < ci_iid["n_eff"]       # ... but autocorrelation -> fewer
    assert ci_ar["se"] > ci_iid["se"]             # -> a WIDER interval
    assert ci_ar["ci_low"] < ci_iid["ci_low"]     # -> a lower lower-bound

    # the SE is the Lo Sharpe formula with N_eff, NOT the raw count:
    ir, n_eff = ci_ar["point"], ci_ar["n_eff"]
    assert ci_ar["se"] == pytest.approx(math.sqrt((1.0 + 0.5 * ir * ir) / n_eff))
    assert ci_ar["se"] > math.sqrt((1.0 + 0.5 * ir * ir) / 500)  # raw N understates

    # the gate consequence: at the default bar the i.i.d. lower CI clears, the
    # autocorrelated one does not -- the correlated series fails where iid passes.
    bar = VerdictThresholds().min_abs_icir
    assert ci_iid["ci_low"] > bar
    assert ci_ar["ci_low"] < bar


def test_mean_ci_uses_n_eff_not_raw_count():
    ar = _ar1(0.9, 500, seed=3)
    ci = mean_ci(ar)
    assert ci["n_eff"] < 500
    std = float(pd.Series(ar, dtype=float).std(ddof=1))
    assert ci["se"] == pytest.approx(std / math.sqrt(ci["n_eff"]))
    # the interval brackets the point symmetrically at the stated confidence.
    assert ci["confidence"] == 0.95
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]


def test_degenerate_series_give_nan_cis_never_a_fabricated_number():
    assert math.isnan(information_ratio_ci(pd.Series([1.0]))["ci_low"])
    assert math.isnan(information_ratio_ci(pd.Series([2.0, 2.0, 2.0]))["ci_low"])  # zero var
    assert math.isnan(mean_ci(pd.Series([], dtype=float))["ci_low"])


def test_predictive_and_purity_sections_report_n_eff_based_cis(rng):
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    report = StandardFactorEvaluator().evaluate(
        factor,
        make_spec(),
        make_cfg(),
        EvalContext(price_panel=panel, known_factors=incremental_book(rng, factor)),
    )
    pred = report.by_name()["predictive_power"].payload
    assert pred["ic_ir_ci_low"] < pred["ic_ir"] < pred["ic_ir_ci_high"]
    assert pred["ic_ir_ci_confidence"] == 0.95
    assert pred["ic_ir_ci_n_eff"] <= pred["ic_periods_finite"]  # N_eff never > N
    purity = report.by_name()["purity"].payload
    assert purity["incremental_ic_ir_ci_low"] < purity["incremental_ic_ir"]


def test_reported_cis_are_deterministic(rng):
    panel, factor = planted_signal_setup(rng, n_dates=120, n_symbols=25)
    ctx = EvalContext(price_panel=panel, known_factors=incremental_book(rng, factor))
    first = StandardFactorEvaluator().evaluate(factor, make_spec(), make_cfg(), ctx)
    second = StandardFactorEvaluator().evaluate(factor, make_spec(), make_cfg(), ctx)
    assert first.to_json() == second.to_json()


def test_declared_success_criteria_flow_through_the_standard_evaluator(rng):
    """End to end: a pre-registered bar on EvalConfig changes the deployment label
    AND is stamped. The strong signal Adopts under the default bar; a stricter
    declared bar (min ICIR 8.0, unreachable by any lower CI) blocks the Predictive
    PASS, so the same run no longer Adopts — and the report says which bar it used."""
    panel, factor = planted_signal_setup(rng, n_dates=300, n_symbols=40, strength=0.05)
    dates = panel_dates(factor)
    ctx = EvalContext(
        price_panel=panel,
        known_factors=incremental_book(rng, factor),
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    base_cfg = make_cfg(oos_split=str(dates[130].date()), is_exploratory=False)
    adopted = StandardFactorEvaluator().evaluate(factor, make_spec(), base_cfg, ctx)
    assert adopted.require_verdict().verdict == ADOPT
    assert adopted.criteria_source == "default"

    # A pre-registered bar set BETWEEN the ICIR lower CI bound and the point: the
    # POINT clears it, but the LOWER CI does not -> the CI gate (not the point)
    # blocks the Predictive PASS, so the same run reads INSUFFICIENT and no longer
    # Adopts. Picked from the run's own reported CI so it is robust to the draw.
    pred = adopted.by_name()["predictive_power"].payload
    bar = (pred["ic_ir_ci_low"] + pred["ic_ir"]) / 2.0
    assert pred["ic_ir_ci_low"] < bar < pred["ic_ir"]
    strict_cfg = make_cfg(
        oos_split=str(dates[130].date()),
        is_exploratory=False,
        success_criteria=VerdictThresholds(min_abs_icir=bar),
    )
    strict = StandardFactorEvaluator().evaluate(factor, make_spec(), strict_cfg, ctx)
    assert strict.criteria_source == "declared"
    assert strict.require_verdict().predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert strict.require_verdict().verdict != ADOPT
    assert "PRE-REGISTERED" in strict.render()


# ==========================================================================
# 7. layering
# ==========================================================================


def test_factors_layer_never_imports_analytics():
    """Invariant #1/#3: forward returns live in analytics; factors must not reach up."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "factors"
    offenders = [
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if "analytics" in path.read_text(encoding="utf-8")
        and any(
            line.strip().startswith(("import analytics", "from analytics"))
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert offenders == []
