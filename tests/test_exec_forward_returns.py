"""exec-to-exec forward returns: pricing, alignment, adjustment and missingness.

Every test here is written to FAIL under a specific plausible mistake, and the four
mutations named in the task card (§4) are checked against these tests by actually
patching the implementation and observing the failure — not by assertion in prose.

Naming note: test function names must be unique across the whole suite (the test
package has no ``__init__.py``, so two identically named functions in different
files silently collapse into one). Everything below is prefixed accordingly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval.config import EvalConfig
from analytics.eval.ir import EvalContext, build_eval_ir
from data.cache.intraday_cache import ENDPOINT as MINUTE_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from factors.spec import FactorSpec, PanelField
from qt.config import (
    AlphaCfg,
    BacktestCfg,
    CacheCfg,
    CostCfg,
    DataCfg,
    FactorCfg,
    IntradayCfg,
    OutputCfg,
    PortfolioCfg,
    RootConfig,
    UniverseCfg,
)
from qt.exec_basis_sanity import check_exec_basis
from qt.exec_forward_returns import (
    MISS_BAD_ADJ_FACTOR,
    MISS_BAD_VWAP,
    MISS_NO_BAR,
    STATUS_OK,
    ExecBasisParams,
    _restrict_to_execution_window,
    artifact_key,
    build_exec_price_panel,
    coverage_loss,
    exec_forward_returns,
    intraday_spec_variant,
)
from runtime.intraday_execution import build_execution_prices

LOGGER = logging.getLogger("test.exec_basis")

DAYS = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")


# --------------------------------------------------------------------------- #
# Fixtures: a tiny minute cache + a tiny daily panel
# --------------------------------------------------------------------------- #
def _bar(symbol: str, day: str, hhmmss: str, close: float, volume: float, amount: float) -> dict:
    end = pd.Timestamp(day) + pd.Timedelta(hhmmss)
    return {
        "symbol": symbol,
        "bar_end": end,
        "source_trade_time": end,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "amount": amount,
        "freq": RAW_INTRADAY_FREQ,
    }


def _session(symbol: str, day: str, *, close: float, volume: float, amount: float) -> list[dict]:
    """One session whose 14:51 bar has a VWAP DIFFERENT from its close.

    14:49 / 14:50 bars sit before the execution window and 14:52 after the earliest
    in-window bar, so a run that picked the wrong bar shows up as a wrong number
    rather than a wrong-looking shape.
    """
    return [
        _bar(symbol, day, "14:49:00", close * 3.0, 10.0, close * 30.0),
        _bar(symbol, day, "14:50:00", close * 2.0, 10.0, close * 20.0),
        _bar(symbol, day, "14:51:00", close, volume, amount),
        _bar(symbol, day, "14:52:00", close * 5.0, 10.0, close * 50.0),
    ]


def _write_cache(root, rows: list[dict]) -> IntradayParquetStore:
    store = IntradayParquetStore(str(root))
    frame = pd.DataFrame(rows)
    for symbol, part in frame.groupby("symbol"):
        store.upsert(MINUTE_ENDPOINT, str(symbol), RAW_INTRADAY_FREQ, part, KEY_COLS)
    return store


def _panel(symbols: tuple[str, ...], days: tuple[str, ...], adj: dict | None = None) -> pd.DataFrame:
    """Daily panel: only ``adj_factor`` and the (date, symbol) grid matter here."""
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([pd.Timestamp(d) for d in days]), list(symbols)],
        names=["date", "symbol"],
    )
    factors = [
        (adj or {}).get((pd.Timestamp(d).strftime("%Y-%m-%d"), s), 1.0)
        for d, s in index
    ]
    return pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1.0,
            "amount": 10.0,
            "adj_factor": factors,
        },
        index=index,
    )


def _config(root, start: str, end: str, symbols: tuple[str, ...], **intraday) -> RootConfig:
    return RootConfig(
        data=DataCfg(
            source="tushare",
            start=start,
            end=end,
            external_secret_file="/nonexistent.json",
            cache=CacheCfg(enabled=True, root_dir=str(root)),
        ),
        universe=UniverseCfg(type="static", symbols=list(symbols)),
        factors=[FactorCfg(name="momentum_20")],
        alpha=AlphaCfg(),
        portfolio=PortfolioCfg(top_n=1),
        backtest=BacktestCfg(),
        cost=CostCfg(),
        output=OutputCfg(data_dir=str(root.parent / "data")),
        intraday=IntradayCfg(enabled=True, **intraday) if intraday else None,
    )


def _build(tmp_path, rows, symbols, days, adj=None, cfg=None):
    root = tmp_path / "cache"
    _write_cache(root, rows)
    panel = _panel(symbols, days, adj)
    conf = cfg or _config(root, days[0], days[-1], symbols)
    params = ExecBasisParams.from_config(conf)
    prices = build_exec_price_panel(conf, panel, list(symbols), params, LOGGER)
    return conf, panel, params, prices


# --------------------------------------------------------------------------- #
# 1. Pricing: the VWAP of the selected bar, never its close
# --------------------------------------------------------------------------- #
def test_exec_basis_price_is_bar_vwap_never_bar_close(tmp_path):
    # 14:51 bar: close 10.0, volume 100, amount 1200 -> VWAP 12.0 != close.
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],))

    row = prices.frame.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert row["status"] == STATUS_OK
    assert row["raw_exec_price"] == pytest.approx(12.0)      # amount / volume
    assert row["raw_exec_price"] != pytest.approx(10.0)      # NOT the bar close
    assert row["exec_time"] == pd.Timestamp(DAYS[0]) + pd.Timedelta("14:51:00")


def test_exec_basis_selects_the_earliest_in_window_bar(tmp_path):
    """A 14:52 bar must never win over an existing 14:51 bar."""
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],))
    # the 14:52 bar's VWAP would be 50.0/10 = 5.0 * ... -> a very different number
    assert prices.frame.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ"), "raw_exec_price"] == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# 2. Adjustment: the ratio identity, and its invariances
# --------------------------------------------------------------------------- #
def test_exec_basis_return_is_the_adjusted_price_ratio(tmp_path):
    rows = (
        _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1000.0)
        + _session("AAA.SZ", DAYS[1], close=11.0, volume=100.0, amount=1100.0)
    )
    adj = {(DAYS[0], "AAA.SZ"): 2.0, (DAYS[1], "AAA.SZ"): 2.0}
    _, panel, _, prices = _build(tmp_path, rows, ("AAA.SZ",), DAYS[:2], adj)

    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:2]], name="date")
    returns = exec_forward_returns(prices.adjusted_price, dates, 1)
    got = returns.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert got == pytest.approx((11.0 * 2.0) / (10.0 * 2.0) - 1.0)


def test_exec_basis_adjustment_is_invariant_to_a_common_rescale(tmp_path):
    """Scaling every adj_factor by a constant cannot move a single return.

    This is the property that makes the identity exact and window-invariant (the
    per-symbol anchor cancels). A return that moved would mean the adjustment was
    applied as something other than a ratio.
    """
    rows = (
        _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1000.0)
        + _session("AAA.SZ", DAYS[1], close=11.0, volume=100.0, amount=1210.0)
    )
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:2]], name="date")
    base = {(DAYS[0], "AAA.SZ"): 3.0, (DAYS[1], "AAA.SZ"): 1.5}
    scaled = {k: v * 7.0 for k, v in base.items()}

    _, _, _, p1 = _build(tmp_path / "a", rows, ("AAA.SZ",), DAYS[:2], base)
    _, _, _, p2 = _build(tmp_path / "b", rows, ("AAA.SZ",), DAYS[:2], scaled)
    r1 = exec_forward_returns(p1.adjusted_price, dates, 1)
    r2 = exec_forward_returns(p2.adjusted_price, dates, 1)
    pd.testing.assert_series_equal(r1, r2)


def test_exec_basis_ex_date_period_is_corporate_action_free(tmp_path):
    """A 50% ex-date drop in the RAW price must not read as a -50% return."""
    rows = (
        _session("AAA.SZ", DAYS[0], close=20.0, volume=100.0, amount=2000.0)
        + _session("AAA.SZ", DAYS[1], close=10.0, volume=100.0, amount=1010.0)
    )
    # adj_factor doubles across the ex-date, exactly offsetting the price halving.
    adj = {(DAYS[0], "AAA.SZ"): 1.0, (DAYS[1], "AAA.SZ"): 2.0}
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), DAYS[:2], adj)
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:2]], name="date")
    got = exec_forward_returns(prices.adjusted_price, dates, 1).loc[
        (pd.Timestamp(DAYS[0]), "AAA.SZ")
    ]
    unadjusted = 10.1 / 20.0 - 1.0
    assert unadjusted == pytest.approx(-0.495)   # what a raw ratio would report
    assert got == pytest.approx((10.1 * 2.0) / (20.0 * 1.0) - 1.0)
    assert got == pytest.approx(0.01)            # the real +1% move survives


# --------------------------------------------------------------------------- #
# 3. Alignment: h steps on the EVALUATION grid
# --------------------------------------------------------------------------- #
def test_exec_basis_steps_the_evaluation_grid_not_the_data_grid(tmp_path):
    """The exit anchor is the next EVALUATION period, not the next cached day.

    The minute cache and the daily panel both hold three sessions; the factor is
    only evaluated on the first and third. The forward return at D1 must therefore
    be measured against D3 — a series that reached for D2 (the next day the data
    happens to have) is measuring a different holding period entirely.
    """
    rows = (
        _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1000.0)
        + _session("AAA.SZ", DAYS[1], close=10.0, volume=100.0, amount=5000.0)
        + _session("AAA.SZ", DAYS[2], close=10.0, volume=100.0, amount=2000.0)
    )
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), DAYS[:3])

    evaluation_grid = pd.DatetimeIndex(
        [pd.Timestamp(DAYS[0]), pd.Timestamp(DAYS[2])], name="date"
    )
    returns = exec_forward_returns(prices.adjusted_price, evaluation_grid, 1)
    got = returns.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert got == pytest.approx(20.0 / 10.0 - 1.0)      # D1 -> D3 (evaluation grid)
    assert got != pytest.approx(50.0 / 10.0 - 1.0)      # NOT D1 -> D2 (data grid)
    # D2 is not on the evaluation grid at all, so it carries no return.
    assert (pd.Timestamp(DAYS[1]), "AAA.SZ") not in returns.index


def test_exec_basis_horizon_counts_periods_not_days(tmp_path):
    rows = [
        bar
        for i, day in enumerate(DAYS[:3])
        for bar in _session("AAA.SZ", day, close=10.0, volume=100.0, amount=1000.0 * (i + 1))
    ]
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), DAYS[:3])
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:3]], name="date")
    two = exec_forward_returns(prices.adjusted_price, dates, 2)
    assert two.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")] == pytest.approx(30.0 / 10.0 - 1.0)
    assert np.isnan(two.loc[(pd.Timestamp(DAYS[1]), "AAA.SZ")])


# --------------------------------------------------------------------------- #
# 4. Missingness: three causes, counted apart, never softened
# --------------------------------------------------------------------------- #
def test_exec_basis_missing_bar_is_no_bar_and_never_a_close_fallback(tmp_path):
    """A session with bars only OUTSIDE the execution window prices at nothing."""
    rows = [
        _bar("AAA.SZ", DAYS[0], "14:49:00", 10.0, 100.0, 1000.0),
        _bar("AAA.SZ", DAYS[0], "14:50:00", 10.0, 100.0, 1000.0),
    ]
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],))
    row = prices.frame.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert row["status"] == MISS_NO_BAR
    assert np.isnan(row["raw_exec_price"])
    assert np.isnan(row["adj_exec_price"])
    assert prices.status_counts()[MISS_NO_BAR] == 1


@pytest.mark.parametrize(
    "volume, amount",
    [(0.0, 1000.0), (100.0, 0.0), (float("nan"), 1000.0), (100.0, float("nan"))],
)
def test_exec_basis_undefined_vwap_is_bad_vwap_never_the_bar_close(tmp_path, volume, amount):
    """No traded shares (or value) means no traded average price — not the close."""
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=volume, amount=amount)
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],))
    row = prices.frame.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert row["status"] == MISS_BAD_VWAP
    assert np.isnan(row["raw_exec_price"])
    assert not np.isclose(np.nan_to_num(row["raw_exec_price"], nan=-1.0), 10.0)
    assert prices.status_counts()[MISS_BAD_VWAP] == 1
    assert prices.status_counts()[MISS_NO_BAR] == 0


@pytest.mark.parametrize("factor", [float("nan"), 0.0, -1.0])
def test_exec_basis_unusable_adj_factor_is_counted_never_assumed_one(tmp_path, factor):
    """A missing/non-positive adj_factor blocks the pair; 1.0 is NOT substituted."""
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
    adj = {(DAYS[0], "AAA.SZ"): factor}
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],), adj)
    row = prices.frame.loc[(pd.Timestamp(DAYS[0]), "AAA.SZ")]
    assert row["status"] == MISS_BAD_ADJ_FACTOR
    assert row["raw_exec_price"] == pytest.approx(12.0)   # the bar priced fine ...
    assert np.isnan(row["adj_exec_price"])                # ... the pair still blocks
    assert prices.status_counts()[MISS_BAD_ADJ_FACTOR] == 1


def test_exec_basis_uncached_symbol_is_no_bar_without_any_live_call(tmp_path):
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
    _, _, _, prices = _build(tmp_path, rows, ("AAA.SZ", "ZZZ.SZ"), (DAYS[0],))
    assert prices.minute_live_calls == 0
    assert prices.symbols_with_bars == 1
    assert prices.frame.loc[(pd.Timestamp(DAYS[0]), "ZZZ.SZ"), "status"] == MISS_NO_BAR


def test_exec_basis_coverage_loss_is_partitioned_by_cause():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp(DAYS[0]), "A.SZ"),
            (pd.Timestamp(DAYS[0]), "B.SZ"),
            (pd.Timestamp(DAYS[0]), "C.SZ"),
            (pd.Timestamp(DAYS[0]), "D.SZ"),
        ],
        names=["date", "symbol"],
    )
    exec_returns = pd.Series([0.01, np.nan, np.nan, np.nan], index=index)
    close_returns = pd.Series([0.01, 0.02, 0.03, np.nan], index=index)
    status = pd.Series([STATUS_OK, MISS_NO_BAR, MISS_BAD_VWAP, MISS_BAD_ADJ_FACTOR], index=index)

    out = coverage_loss(exec_returns, close_returns, status, 1)
    assert out["lost_pairs"] == 2          # D has no close return -> not a loss
    assert out["lost_pairs_by_cause"] == {
        MISS_NO_BAR: 1,
        MISS_BAD_VWAP: 1,
        MISS_BAD_ADJ_FACTOR: 0,
    }
    assert out["distinct_symbols_affected"] == 2
    assert out["close_to_close_measurable_pairs"] == 3
    assert out["exec_to_exec_measurable_pairs"] == 1


# --------------------------------------------------------------------------- #
# 5. Reuse of the canonical execution layer
# --------------------------------------------------------------------------- #
def test_exec_basis_window_prefilter_matches_a_full_day_build(tmp_path):
    """The performance pre-filter must be semantically invisible.

    Building from the whole session and building from only the in-window bars must
    produce the same prices and the same fills — otherwise the pre-filter, not
    ``resolve_fill``, would be deciding which bar executes.
    """
    rows = (
        _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
        + _session("AAA.SZ", DAYS[1], close=11.0, volume=100.0, amount=1300.0)
    )
    stored = pd.DataFrame(rows)
    params = ExecBasisParams.from_config(
        _config(tmp_path / "cache", DAYS[0], DAYS[1], ("AAA.SZ",))
    )
    cfg = params.exec_config()
    dates = [pd.Timestamp(DAYS[0]), pd.Timestamp(DAYS[1])]

    def _norm(frame):
        return normalize_intraday_bars(
            frame.rename(columns={"bar_end": "time"})[
                ["time", "symbol", "open", "high", "low", "close", "volume",
                 "amount", "source_trade_time"]
            ],
            freq=RAW_INTRADAY_FREQ,
            data_lag=params.data_lag,
        )

    full_prices, full_fills = build_execution_prices(_norm(stored), dates, ["AAA.SZ"], cfg)
    win_prices, win_fills = build_execution_prices(
        _norm(_restrict_to_execution_window(stored, params)), dates, ["AAA.SZ"], cfg
    )
    pd.testing.assert_frame_equal(full_prices, win_prices)
    assert full_fills == win_fills


# --------------------------------------------------------------------------- #
# 6. The spec variant and the evaluator's guard
# --------------------------------------------------------------------------- #
def _daily_spec() -> FactorSpec:
    return FactorSpec(
        factor_id="demo_minute_factor",
        version="1.0",
        description="synthetic minute-derived factor",
        expected_ic_sign=1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="stk_mins_1min"),),
        adjustment="returns_invariant",
        overnight_boundary="none",
        family="microstructure",
    )


def test_exec_basis_spec_variant_carries_the_execution_parameters(tmp_path):
    cfg = _config(
        tmp_path / "cache", DAYS[0], DAYS[1], ("AAA.SZ",),
        decision_time="14:45:00", execution_window=("14:46:00", "14:50:00"),
    )
    params = ExecBasisParams.from_config(cfg)
    variant = intraday_spec_variant(_daily_spec(), params)

    assert variant.is_intraday is True
    assert variant.return_basis == "exec_to_exec"
    assert variant.decision_cutoff == "14:45:00"
    assert variant.execution_window == "[14:46:00,14:50:00]"
    assert variant.execution_model == params.execution_model
    assert variant.session_open == params.session_open
    assert variant.data_lag == params.data_lag
    # The FACTOR is untouched — only what it is scored against changed.
    daily = _daily_spec()
    assert (variant.factor_id, variant.version, variant.expected_ic_sign) == (
        daily.factor_id, daily.version, daily.expected_ic_sign
    )
    # And the declared parameters are the ones the returns were computed with.
    exec_cfg = params.exec_config()
    assert variant.decision_cutoff == exec_cfg.decision_time
    assert variant.execution_window == f"[{exec_cfg.execution_window[0]},{exec_cfg.execution_window[1]}]"


@pytest.mark.parametrize(
    "field", ["decision_cutoff", "data_lag", "session_open", "execution_model", "execution_window"]
)
def test_exec_basis_spec_variant_needs_the_whole_minute_block(field):
    """Dropping any one of the five must fail at FactorSpec construction."""
    import dataclasses

    params = ExecBasisParams.from_config(
        RootConfig(
            data=DataCfg(source="demo", start="2024-01-02", end="2024-01-05"),
            universe=UniverseCfg(type="static", symbols=["A.SZ"]),
            factors=[FactorCfg(name="momentum_20")],
            alpha=AlphaCfg(), portfolio=PortfolioCfg(top_n=1),
            backtest=BacktestCfg(), cost=CostCfg(), output=OutputCfg(),
        )
    )
    block = params.spec_fields()
    block[field] = None
    with pytest.raises(ValueError, match="minute block"):
        dataclasses.replace(
            _daily_spec(), is_intraday=True, return_basis="exec_to_exec", **block
        )


def test_exec_basis_evaluator_still_refuses_exec_basis_without_forward_returns(tmp_path):
    """The frozen guard must NOT be weakened: no returns supplied -> raise.

    Without this the evaluator would silently score an ``exec_to_exec`` factor on
    close-to-close returns and label the report ``exec_to_exec``.
    """
    params = ExecBasisParams.from_config(
        _config(tmp_path / "cache", DAYS[0], DAYS[1], ("AAA.SZ",))
    )
    spec = intraday_spec_variant(_daily_spec(), params)
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS]), ["A.SZ", "B.SZ"]],
        names=["date", "symbol"],
    )
    factor = pd.Series(np.arange(len(index), dtype=float), index=index)
    panel = _panel(("A.SZ", "B.SZ"), DAYS)
    cfg = EvalConfig(
        universe="static", universe_is_pit=False, start=DAYS[0], end=DAYS[-1],
        is_exploratory=True, post_hoc_selected=False, rebalance="daily",
    )
    with pytest.raises(ValueError, match="cannot be derived from a close panel"):
        build_eval_ir(factor, spec, cfg, EvalContext(price_panel=panel))


# --------------------------------------------------------------------------- #
# 7. The shared artifact
# --------------------------------------------------------------------------- #
def test_exec_basis_artifact_is_reused_and_rekeyed_on_a_parameter_change(tmp_path):
    rows = _session("AAA.SZ", DAYS[0], close=10.0, volume=100.0, amount=1200.0)
    conf, panel, params, first = _build(tmp_path, rows, ("AAA.SZ",), (DAYS[0],))
    assert first.reused is False
    assert first.path.exists()

    second = build_exec_price_panel(conf, panel, ["AAA.SZ"], params, LOGGER)
    assert second.reused is True
    assert second.path == first.path
    pd.testing.assert_frame_equal(first.frame, second.frame)

    # A different execution window is a different return series -> different key.
    other = ExecBasisParams.from_config(
        _config(tmp_path / "cache", DAYS[0], DAYS[0], ("AAA.SZ",),
                execution_window=("14:52:00", "14:56:59"))
    )
    dates = pd.DatetimeIndex([pd.Timestamp(DAYS[0])], name="date")
    assert artifact_key(conf, ["AAA.SZ"], dates, other)[0] != first.key


def test_exec_basis_params_default_to_the_project_minute_conventions(tmp_path):
    """No intraday block -> the framework's declared defaults, and it says so."""
    conf = _config(tmp_path / "cache", DAYS[0], DAYS[1], ("AAA.SZ",))
    assert conf.intraday is None
    params = ExecBasisParams.from_config(conf)
    assert params.decision_cutoff == "14:50:00"
    assert params.execution_window == ("14:51:00", "14:56:59")
    assert params.execution_price_basis == "bar_vwap"
    assert params.session_open == "09:30:00"
    assert "defaults" in params.source


# --------------------------------------------------------------------------- #
# 8. The implementation-independent sanity checks
# --------------------------------------------------------------------------- #
def _sanity_fixture(tmp_path):
    symbols = ("A.SZ", "B.SZ", "C.SZ")
    amounts = {"A.SZ": (1000.0, 1010.0, 1030.0, 1020.0),
               "B.SZ": (2000.0, 2040.0, 2010.0, 2050.0),
               "C.SZ": (3000.0, 2970.0, 3030.0, 3060.0)}
    rows = [
        bar
        for sym in symbols
        for i, day in enumerate(DAYS)
        for bar in _session(sym, day, close=10.0, volume=100.0, amount=amounts[sym][i])
    ]
    conf, panel, params, prices = _build(tmp_path, rows, symbols, DAYS)
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS], name="date")
    exec_returns = exec_forward_returns(prices.adjusted_price, dates, 1)
    # A close panel that MOVES with the exec prices, so the agreement check is a
    # real check rather than a comparison against a constant.
    close = panel.copy()
    close["close"] = prices.frame["raw_exec_price"].to_numpy()
    from analytics.factor import forward_returns as _fwd

    close_returns = _fwd(close, periods=(1,))["forward_return_1d"]
    return conf, params, prices, exec_returns, close_returns


def test_exec_basis_sanity_hand_check_agrees_with_the_generated_returns(tmp_path):
    conf, params, prices, exec_returns, close_returns = _sanity_fixture(tmp_path)
    sanity = check_exec_basis(
        prices.frame, exec_returns, close_returns, params,
        conf.data.cache.root_dir, 1, n_hand_checks=3,
    )
    assert sanity.corr_ok
    assert len(sanity.hand_checks) == 3
    assert sanity.hand_check_max_abs_diff < 1e-12
    for row in sanity.hand_checks:
        assert row["entry_vwap"] == pytest.approx(
            row["entry_amount"] / row["entry_volume"]
        )


def test_exec_basis_sanity_raises_when_agreement_collapses(tmp_path):
    conf, params, prices, exec_returns, close_returns = _sanity_fixture(tmp_path)
    with pytest.raises(ValueError, match="sanity check FAILED"):
        check_exec_basis(
            prices.frame, -exec_returns, close_returns, params,
            conf.data.cache.root_dir, 1, n_hand_checks=1,
        )
