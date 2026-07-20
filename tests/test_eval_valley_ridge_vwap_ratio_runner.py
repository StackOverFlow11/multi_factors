"""PR-J runner: cache-only minute loader, ridge-scarcity coverage, two-run eval (no net)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, MANDATORY_SECTIONS, Section, Skipped
from analytics.eval.verdict import AXIS_NOT_ASSESSED, AXIS_VERDICTS
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from data.clean.intraday_valley_ridge_vwap import (
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
)
from factors.compute.intraday_derived import ValleyRidgeVwapRatioFactor
from qt.config import (
    AlphaCfg,
    BacktestCfg,
    CacheCfg,
    CostCfg,
    DataCfg,
    FactorCfg,
    OutputCfg,
    PortfolioCfg,
    RootConfig,
    UniverseCfg,
)
from qt.eval_valley_ridge_vwap_ratio import (
    _load_valley_ridge_vwap_ratio_panel,
    evaluate_two_runs,
    extract_metrics,
    summarize_ridge_coverage,
)


# --------------------------------------------------------------------------- #
# Minute cache-only loader
# --------------------------------------------------------------------------- #
def _stored_rows(sym, day, vols, amts):
    """Build STORED_COLUMNS-shaped 1min rows for one session of CONSECUTIVE minutes.

    Bar i sits at ``09:31 + i`` minutes (all inside the 14:50 PIT window). OHLC are dummy
    constants; ``volume`` drives the classification and ``amount`` -- set INDEPENDENTLY --
    drives the VWAPs.
    """
    base = pd.Timestamp(day) + pd.Timedelta("09:31:00")
    rows = []
    for i, (v, a) in enumerate(zip(vols, amts)):
        be = base + pd.Timedelta(minutes=i)
        rows.append(
            {
                "symbol": sym,
                "bar_end": be,
                "source_trade_time": be,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": float(v),
                "amount": float(a),
                "freq": "1min",
            }
        )
    return pd.DataFrame(rows)


def _min_config(root, start, end):
    return RootConfig(
        data=DataCfg(
            source="tushare",
            start=start,
            end=end,
            external_secret_file="/nonexistent.json",
            cache=CacheCfg(enabled=True, root_dir=str(root)),
        ),
        universe=UniverseCfg(type="static", symbols=["A.SZ"]),
        factors=[FactorCfg(name="momentum_20")],
        alpha=AlphaCfg(),
        portfolio=PortfolioCfg(top_n=1),
        backtest=BacktestCfg(),
        cost=CostCfg(),
        output=OutputCfg(),
    )


# Three constant background days (baseline mu=100, sigma=0 -> eruptive threshold 100),
# then the hand-computed CASE A test day: 16 slots, ridge runs at (3,4) and (8,9) with
# volume 200 / amount 4000 (price 20), an ISOLATED peak at slot 12 (volume 300 / amount
# 9000 -> price 30, excluded from BOTH legs), and 11 valley bars at volume 100 carrying
# amount 1000 (x6) and 1200 (x5) -> valley VWAP 12000/1100, ridge VWAP 20, ratio 6/11.
_DAYS = ("2021-07-01", "2021-07-02", "2021-07-03")
_TEST_DAY = "2021-07-04"
_N_SLOTS = 16
_RIDGES = (3, 4, 8, 9)
_PEAK = 12
_CASE_A_RATIO = 6.0 / 11.0
_BG_VOLS = [100.0] * _N_SLOTS
_BG_AMTS = [1000.0] * _N_SLOTS


def _case_a():
    vols = [100.0] * _N_SLOTS
    amts = [0.0] * _N_SLOTS
    for s in _RIDGES:
        vols[s] = 200.0
        amts[s] = 4000.0
    vols[_PEAK] = 300.0
    amts[_PEAK] = 9000.0
    valley_slots = [s for s in range(_N_SLOTS) if s not in _RIDGES and s != _PEAK]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 6 else 1200.0
    return vols, amts


# Small gates so a 4-day cache produces a value (baseline needs >= 2 prior obs; the test
# day carries 4 tradable ridge bars and 11 tradable valley bars). lookback_days=1 so the
# test day's value IS its own ratio -- the background days have no eruption at all, hence
# no ridge, hence no valid day, so they cannot average in.
_LOAD_KW = dict(
    lookback_days=1, baseline_days=20, baseline_min_obs=2,
    sigma_k=1.0, min_valid_days=1, min_classifiable=1,
    min_valley_bars=1, min_ridge_bars=1,
)


def _seed_symbol(store, sym):
    for day in _DAYS:
        store.upsert(
            INTRADAY_ENDPOINT, sym, "1min",
            _stored_rows(sym, day, _BG_VOLS, _BG_AMTS), KEY_COLS,
        )
    store.upsert(
        INTRADAY_ENDPOINT, sym, "1min",
        _stored_rows(sym, _TEST_DAY, *_case_a()), KEY_COLS,
    )


def test_minute_loader_is_cache_only_and_discloses_empty(tmp_path):
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    # Two symbols with cached minute bars; a third has NO cached minute (must be
    # disclosed as empty, never fetched).
    for sym in ("AAA.SZ", "BBB.SZ"):
        _seed_symbol(store, sym)

    cfg = _min_config(root, "2021-07-01", "2021-07-04")
    spec = ValleyRidgeVwapRatioFactor().spec
    logger = logging.getLogger("test.vrr.loader")
    load = _load_valley_ridge_vwap_ratio_panel(
        cfg, ["AAA.SZ", "BBB.SZ", "CCC.SZ"], spec, logger, **_LOAD_KW
    )
    assert load.live_calls == 0                       # store read has no fetch closure
    assert set(load.covered) == {"AAA.SZ", "BBB.SZ"}  # both produced a value
    assert load.empty_symbols == ("CCC.SZ",)          # uncovered disclosed, not fetched
    assert load.factor.name == spec.factor_id
    assert load.factor.notna().any()
    # the test day carries the hand-computed valley/ridge VWAP ratio
    d = pd.Timestamp(_TEST_DAY)
    assert load.factor.loc[(d, "AAA.SZ")] == pytest.approx(_CASE_A_RATIO)
    assert load.factor.loc[(d, "BBB.SZ")] == pytest.approx(_CASE_A_RATIO)
    assert set(load.factor.index.get_level_values("symbol")) == {"AAA.SZ", "BBB.SZ"}
    # a day where NOTHING erupts has no ridge leg at all -> invalid -> no value emitted
    bg = pd.Timestamp(_DAYS[-1])
    assert (bg, "AAA.SZ") not in load.factor.index


def test_minute_loader_reports_the_ridge_scarcity_distribution(tmp_path):
    """The ridge-bar distribution + validity rate must be MEASURED on the real days."""
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    _seed_symbol(store, "AAA.SZ")
    cfg = _min_config(root, "2021-07-01", "2021-07-04")
    load = _load_valley_ridge_vwap_ratio_panel(
        cfg, ["AAA.SZ"], ValleyRidgeVwapRatioFactor().spec,
        logging.getLogger("test.vrr.cov"), **_LOAD_KW,
    )
    cov = load.ridge_coverage
    # four symbol-days seen, but the first two have no same-slot baseline yet (it needs
    # 2 prior observations) -> only two are CLASSIFIABLE, and of those only the test day
    # has a ridge at all -> validity rate 1/2 over the days that had a fair chance.
    assert cov.symbol_days == 4
    assert cov.classifiable_days == 2
    assert cov.days_below_classifiable_gate == 2  # the warm-up days, counted separately
    assert cov.valid_days == 1
    assert cov.validity_rate == pytest.approx(0.5)
    # the flat background day has zero ridge bars, the test day has four; warm-up days
    # are excluded from the distribution so they cannot drag it towards zero
    assert dict(cov.ridge_percentiles)[100] == pytest.approx(4.0)
    assert dict(cov.ridge_percentiles)[0] == pytest.approx(0.0)
    assert cov.ridge_mean == pytest.approx(2.0)
    # the floors reported are the ones this RUN applied (_LOAD_KW uses 1, not the module
    # defaults) -- otherwise the disclosure would describe gates nobody enforced
    assert cov.min_ridge_bars == 1
    assert cov.min_valley_bars == 1
    assert cov.days_below_ridge_gate == 1  # only the flat day (0 ridges) is below 1
    line = cov.render()
    assert "ridge scarcity" in line and "valid_days=1" in line
    assert "below_ridge_gate(1)" in line and "below_valley_gate(1)" in line


def test_summarize_ridge_coverage_counterfactual_at_the_valley_floor():
    """The disclosure quantifies exactly what the LOWERED ridge floor buys."""
    diag = pd.DataFrame(
        {
            "classifiable_bars": [240, 240, 240, 240],
            "valley_bars": [200, 200, 200, 200],
            "ridge_bars": [4, 12, 25, 30],
            # as the factor would mark them under the default floor of 10
            "valid": [False, True, True, True],
        },
        index=pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=4), name="trade_date"),
    )
    cov = summarize_ridge_coverage([diag])
    assert cov.symbol_days == 4
    assert cov.classifiable_days == 4  # every day clears PR-F's classifiable floor
    assert cov.valid_days == 3
    # holding the ridge leg to the VALLEY floor (20) would keep only the 25 / 30 days
    assert cov.valid_days_at_valley_floor == 2
    assert cov.days_below_ridge_gate == 1
    assert cov.days_below_valley_gate == 0
    assert cov.ridge_mean == pytest.approx((4 + 12 + 25 + 30) / 4)
    # defaults are the PINNED production floors when the caller does not override
    assert cov.min_ridge_bars == VALLEY_RIDGE_MIN_RIDGE_BARS == 10
    assert cov.min_valley_bars == VALLEY_RIDGE_MIN_VALLEY_BARS == 20
    assert f"below_ridge_gate({VALLEY_RIDGE_MIN_RIDGE_BARS})" in cov.render()


def test_summarize_ridge_coverage_handles_no_frames():
    cov = summarize_ridge_coverage([])
    assert cov.symbol_days == 0
    assert cov.classifiable_days == 0
    assert cov.valid_days == 0
    assert np.isnan(cov.validity_rate)
    assert cov.render()  # renders without dividing by zero


def test_minute_loader_blocks_when_nothing_cached(tmp_path):
    cfg = _min_config(tmp_path / "empty", "2021-07-01", "2021-07-04")
    spec = ValleyRidgeVwapRatioFactor().spec
    with pytest.raises(ValueError, match="no requested symbol produced"):
        _load_valley_ridge_vwap_ratio_panel(
            cfg, ["ZZZ.SZ"], spec, logging.getLogger("test.vrr.block"), **_LOAD_KW
        )


# --------------------------------------------------------------------------- #
# Evaluation core (two runs) — synthetic processed panels, no network
# --------------------------------------------------------------------------- #
def _synthetic_panels(n_days=90, n_symbols=15, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # processed (z-scored-ish) subject factor + a 3-column book on the same grid
    factor = pd.Series(
        rng.standard_normal(len(idx)), index=idx, name="valley_ridge_vwap_ratio_20"
    )
    book = pd.DataFrame(
        {
            "value_ep": rng.standard_normal(len(idx)),
            "value_bp": rng.standard_normal(len(idx)),
            "volatility_20": rng.standard_normal(len(idx)),
        },
        index=idx,
    )
    # a qfq-style price panel with the CORE_COLUMNS the forward-return boundary needs
    close = pd.Series(
        100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(len(idx)))), index=idx
    )
    price = pd.DataFrame(
        {
            "open": close.to_numpy(),
            "high": close.to_numpy() * 1.01,
            "low": close.to_numpy() * 0.99,
            "close": close.to_numpy(),
            "volume": 1_000.0,
            "amount": 100_000.0,
            "adj_factor": 1.0,
        },
        index=idx,
    )
    return factor, book, price


def _eval_cfg():
    return EvalConfig(
        universe="000905.SH",
        universe_is_pit=True,
        start="2022-01-03",
        end="2022-05-10",
        is_exploratory=True,
        post_hoc_selected=False,
        rebalance="daily",
        n_quantiles=5,
        oos_split="2022-03-15",
        winsorize=None,
        standardize="zscore",
        neutralization=("industry", "size"),
        industry_level="L1",
    )


def test_evaluate_two_runs_produces_full_reports_and_incremental_axis(tmp_path):
    factor, book, price = _synthetic_panels()
    spec = ValleyRidgeVwapRatioFactor().spec
    reports = evaluate_two_runs(
        factor, spec, _eval_cfg(), price, book,
        universe_symbols=tuple(sorted(set(book.index.get_level_values("symbol")))),
        fee_rate=0.001,
        report_dir=tmp_path,
    )

    for report in (reports.no_book, reports.with_book):
        by = report.by_name()
        assert set(by) == set(MANDATORY_SECTIONS)                 # all 8 present
        assert all(isinstance(s, (Section, Skipped)) for s in by.values())
        assert report.verdict is not None                         # a verdict was produced

    # no-book: purity Skipped (no book) -> Incremental NOT_ASSESSED
    assert isinstance(reports.no_book.by_name()["purity"], Skipped)
    assert reports.no_book.verdict.incremental.verdict == AXIS_NOT_ASSESSED

    # with-book: purity is a real Section that populated the Incremental facts
    purity = reports.with_book.by_name()["purity"]
    assert isinstance(purity, Section)
    assert purity.payload["known_factors_supplied"] is True
    assert "incremental_ic_ir" in purity.payload
    incr = reports.with_book.verdict.incremental.verdict
    assert incr in AXIS_VERDICTS and incr != AXIS_NOT_ASSESSED

    # reports were written to disk (md + json), both runs
    for p in (reports.no_book_md, reports.no_book_json,
              reports.with_book_md, reports.with_book_json):
        assert p.exists() and p.stat().st_size > 0

    # the research-style dashboard PNG is emitted for both runs (mandatory report
    # artifact alongside md/json)
    for p in (reports.no_book_dashboard, reports.with_book_dashboard):
        assert p.exists() and p.stat().st_size > 20_000
        with open(p, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

    # metrics extraction surfaces the gated fields
    m = extract_metrics(reports.no_book)
    assert m["deployment"] in {"Adopt", "Watch", "Reject", "INSUFFICIENT-DATA"}
    assert "effective_samples" in m and "ic_ir" in m


def test_no_book_run_never_reaches_adopt(tmp_path):
    # Structural guarantee (design §6): with no book + no execution facts, at most
    # Watch — regardless of the signal (exploratory cap + NOT_ASSESSED axes).
    factor, book, price = _synthetic_panels(seed=3)
    reports = evaluate_two_runs(
        factor, ValleyRidgeVwapRatioFactor().spec, _eval_cfg(), price, book,
        universe_symbols=(),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    assert reports.no_book.verdict.verdict != "Adopt"
    assert reports.no_book.verdict.tradable.verdict == AXIS_NOT_ASSESSED
