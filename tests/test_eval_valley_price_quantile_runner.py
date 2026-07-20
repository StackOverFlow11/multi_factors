"""PR-L runner: cache-only minute loader + reversal neutralization + the two evaluations.

Network-free throughout: the minute "cache" is a real ``IntradayParquetStore`` seeded in a
tmp_path, and the two-run evaluation seam is exercised on synthetic processed panels.

The structural novelty this file has to pin down, beyond PR-I..PR-K's loader contract, is
that the loader RESIDUALIZES: it streams per-symbol raw quantiles, then applies ONE
cross-sectional reversal neutralization against the T-1 reversal built from the daily
panel's FRONT-ADJUSTED closes. Tests cover that the shipped factor is the residual (not
the raw quantile), that the daily closes actually reach it, and that the neutralization
coverage is measured rather than assumed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval import MANDATORY_SECTIONS, EvalConfig, Section, Skipped
from analytics.eval.verdict import AXIS_NOT_ASSESSED, AXIS_VERDICTS
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from data.clean.intraday_valley_quantile import (
    VALLEY_QUANTILE_MIN_VALLEY_BARS,
    reversal_20,
)
from factors.compute.intraday_derived import ValleyPriceQuantileFactor
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
from qt.eval_valley_price_quantile import (
    _load_valley_price_quantile_panel,
    evaluate_two_runs,
    extract_metrics,
    summarize_neutralization,
)


# --------------------------------------------------------------------------- #
# Minute cache fixtures
# --------------------------------------------------------------------------- #
_N_SLOTS = 16
_ERUPT = (3, 7)
_HIGH_SLOT, _LOW_SLOT = 5, 9
_BG_DAYS = ("2021-07-01", "2021-07-02", "2021-07-05")
_TEST_DAY = "2021-07-06"
# Range on the test day: intraday [80, 120], prev_close 100 (inside) -> hi=120, lo=80.
_HI, _LO = 120.0, 80.0
_PREV_CLOSE = 100.0
# Per-symbol valley VWAP -> q = (vwap - 80) / 40.
_SYM_VWAP = {"AAA.SZ": 90.0, "BBB.SZ": 100.0, "CCC.SZ": 110.0}
_SYM_Q = {s: (v - _LO) / (_HI - _LO) for s, v in _SYM_VWAP.items()}


def _stored_rows(sym, day, *, vols, amts, highs, lows, closes):
    """STORED_COLUMNS-shaped 1min rows for one session of CONSECUTIVE minutes.

    Bar i sits at ``09:31 + i`` minutes (all inside the 14:50 PIT window). ``amount`` is
    set INDEPENDENTLY of ``volume`` (the per-bar price is amount/volume) and high / low /
    close are explicit, because this factor reads the day's range and the previous day's
    last visible close alongside the valley VWAP.
    """
    base = pd.Timestamp(day) + pd.Timedelta("09:31:00")
    rows = []
    for i in range(len(vols)):
        be = base + pd.Timedelta(minutes=i)
        rows.append(
            {
                "symbol": sym,
                "bar_end": be,
                "source_trade_time": be,
                "open": float(closes[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(vols[i]),
                "amount": float(amts[i]),
                "freq": "1min",
            }
        )
    return pd.DataFrame(rows)


def _background_day(sym, day):
    """Flat volume-100 session at price 100 -> baseline mu=100, sigma=0; hi == lo."""
    n = _N_SLOTS
    return _stored_rows(
        sym, day,
        vols=[100.0] * n, amts=[100.0 * _PREV_CLOSE] * n,
        highs=[_PREV_CLOSE] * n, lows=[_PREV_CLOSE] * n, closes=[_PREV_CLOSE] * n,
    )


def _test_day(sym):
    """Engineered day whose valley VWAP is ``_SYM_VWAP[sym]`` inside the range [80, 120]."""
    n = _N_SLOTS
    vwap = _SYM_VWAP[sym]
    vols = [100.0] * n
    amts = [100.0 * vwap] * n
    for s in _ERUPT:
        vols[s] = 200.0
        amts[s] = 200.0 * 500.0  # eruptive bars traded elsewhere; excluded from the VWAP
    highs = [110.0] * n
    lows = [95.0] * n
    highs[_HIGH_SLOT] = _HI
    lows[_LOW_SLOT] = _LO
    return _stored_rows(
        sym, _TEST_DAY,
        vols=vols, amts=amts, highs=highs, lows=lows, closes=[100.0] * n,
    )


def _seed_symbol(store, sym):
    for day in _BG_DAYS:
        store.upsert(INTRADAY_ENDPOINT, sym, "1min", _background_day(sym, day), KEY_COLS)
    store.upsert(INTRADAY_ENDPOINT, sym, "1min", _test_day(sym), KEY_COLS)


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


def _daily_panel(symbols, *, n_days=25, end=_TEST_DAY, seed=5):
    """Daily FRONT-ADJUSTED close panel spanning enough history for a T-1 20-day reversal.

    The panel is INDEPENDENT of the minute cache (it is the runner's own daily panel) and
    must reach back >= 22 business days before the signal day so ``rev20`` is finite there.
    """
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=n_days)
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0, 0.02, size=(len(dates), len(symbols))), axis=0)
    )
    return pd.DataFrame({"close": close.reshape(-1)}, index=idx)


# Small gates so a 4-day cache produces a value. lookback_days=1 so the test day's value IS
# its own daily quantile; the flat background days have hi == lo, hence no valid day, so
# they cannot accumulate in.
_LOAD_KW = dict(
    lookback_days=1, baseline_days=20, baseline_min_obs=2,
    sigma_k=1.0, min_valid_days=1, min_classifiable=1, min_valley_bars=1,
    min_cross_section=3, reversal_days=20,
)


# --------------------------------------------------------------------------- #
# Cache-only loader
# --------------------------------------------------------------------------- #
def test_minute_loader_is_cache_only_and_discloses_empty(tmp_path):
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in _SYM_VWAP:
        _seed_symbol(store, sym)

    syms = sorted(_SYM_VWAP)
    cfg = _min_config(root, "2021-07-01", _TEST_DAY)
    spec = ValleyPriceQuantileFactor().spec
    load = _load_valley_price_quantile_panel(
        cfg, syms + ["ZZZ.SZ"], spec, _daily_panel(syms + ["ZZZ.SZ"]),
        logging.getLogger("test.vpq.loader"), **_LOAD_KW,
    )
    assert load.live_calls == 0                    # store read has no fetch closure
    assert set(load.covered) == set(syms)          # all three produced a raw value
    assert load.empty_symbols == ("ZZZ.SZ",)       # uncovered disclosed, never fetched
    assert load.factor.name == spec.factor_id

    # The RAW panel carries the hand-computed daily quantiles.
    d = pd.Timestamp(_TEST_DAY)
    for sym, q in _SYM_Q.items():
        assert load.raw.loc[(d, sym)] == pytest.approx(q)

    # A flat background day (hi == lo) is invalid -> no value emitted.
    assert (pd.Timestamp(_BG_DAYS[-1]), "AAA.SZ") not in load.raw.index


def test_loader_ships_the_residual_not_the_raw_quantile(tmp_path):
    """The factor the loader returns is the REVERSAL-NEUTRALIZED panel.

    Guards the wiring bug where the neutralization is computed and then discarded. The
    residuals of a 3-name cross-section also sum to ~0 by construction of an intercept
    OLS fit, which the raw quantiles (0.25 / 0.50 / 0.75) do not.
    """
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in _SYM_VWAP:
        _seed_symbol(store, sym)
    syms = sorted(_SYM_VWAP)

    load = _load_valley_price_quantile_panel(
        _min_config(root, "2021-07-01", _TEST_DAY), syms,
        ValleyPriceQuantileFactor().spec, _daily_panel(syms),
        logging.getLogger("test.vpq.resid"), **_LOAD_KW,
    )
    d = pd.Timestamp(_TEST_DAY)
    shipped = np.array([float(load.factor.loc[(d, s)]) for s in syms])
    raw = np.array([float(load.raw.loc[(d, s)]) for s in syms])
    assert np.isfinite(shipped).all()
    assert not np.allclose(shipped, raw)
    assert shipped.sum() == pytest.approx(0.0, abs=1e-12)  # intercept OLS residual


def test_loader_actually_consumes_the_daily_closes(tmp_path):
    """A different daily close panel must produce a different factor.

    Without this, the reversal could be silently absent and every test above would still
    pass on the raw quantiles.
    """
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in _SYM_VWAP:
        _seed_symbol(store, sym)
    syms = sorted(_SYM_VWAP)
    cfg = _min_config(root, "2021-07-01", _TEST_DAY)
    spec = ValleyPriceQuantileFactor().spec

    a = _load_valley_price_quantile_panel(
        cfg, syms, spec, _daily_panel(syms, seed=5),
        logging.getLogger("test.vpq.px.a"), **_LOAD_KW,
    )
    b = _load_valley_price_quantile_panel(
        cfg, syms, spec, _daily_panel(syms, seed=99),
        logging.getLogger("test.vpq.px.b"), **_LOAD_KW,
    )
    d = pd.Timestamp(_TEST_DAY)
    va = [float(a.factor.loc[(d, s)]) for s in syms]
    vb = [float(b.factor.loc[(d, s)]) for s in syms]
    assert not np.allclose(va, vb)
    # ... while the RAW quantiles, which never see a daily close, are identical.
    assert [float(a.raw.loc[(d, s)]) for s in syms] == [
        float(b.raw.loc[(d, s)]) for s in syms
    ]


def test_loader_reports_the_neutralization_coverage(tmp_path):
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in _SYM_VWAP:
        _seed_symbol(store, sym)
    syms = sorted(_SYM_VWAP)

    load = _load_valley_price_quantile_panel(
        _min_config(root, "2021-07-01", _TEST_DAY), syms,
        ValleyPriceQuantileFactor().spec, _daily_panel(syms),
        logging.getLogger("test.vpq.cov"), **_LOAD_KW,
    )
    cov = load.coverage
    assert cov.raw_rows == 3          # one valid day x three symbols
    assert cov.rev_rows == 3          # all three had a finite T-1 reversal
    assert cov.residual_rows == 3     # ... and all three were residualized
    assert cov.dates_total == 1
    assert cov.dates_residualized == 1
    assert cov.cross_section_min == 3 and cov.cross_section_max == 3
    assert np.isfinite(cov.raw_rev_spearman_mean)


def test_loader_blocks_when_nothing_cached(tmp_path):
    cfg = _min_config(tmp_path / "empty", "2021-07-01", _TEST_DAY)
    with pytest.raises(ValueError, match="no requested symbol produced"):
        _load_valley_price_quantile_panel(
            cfg, ["ZZZ.SZ"], ValleyPriceQuantileFactor().spec,
            _daily_panel(["ZZZ.SZ"]), logging.getLogger("test.vpq.block"), **_LOAD_KW,
        )


def test_thin_cross_section_leaves_the_factor_nan(tmp_path):
    """Below ``min_cross_section`` the date cannot be residualized -> NaN, never raw."""
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in _SYM_VWAP:
        _seed_symbol(store, sym)
    syms = sorted(_SYM_VWAP)

    load = _load_valley_price_quantile_panel(
        _min_config(root, "2021-07-01", _TEST_DAY), syms,
        ValleyPriceQuantileFactor().spec, _daily_panel(syms),
        logging.getLogger("test.vpq.thin"),
        **dict(_LOAD_KW, min_cross_section=4),
    )
    assert not np.isfinite(load.factor.to_numpy(dtype=float)).any()
    assert load.raw.notna().any()  # the raw panel is unaffected
    assert load.coverage.residual_rows == 0


# --------------------------------------------------------------------------- #
# summarize_neutralization
# --------------------------------------------------------------------------- #
def _panel(dates, syms, values, name):
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=idx, name=name)


def test_summarize_neutralization_counts_missing_reversal_rows():
    dates = pd.bdate_range("2023-01-02", periods=2)
    syms = ["A", "B", "C"]
    raw = _panel(dates, syms, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "q")
    rev = _panel(dates, syms, [[1.0, 2.0, 3.0], [1.0, np.nan, np.nan]], "r")
    resid = _panel(dates, syms, [[0.0, 0.0, 0.0], [np.nan] * 3], "f")
    cov = summarize_neutralization(raw, rev, resid, min_cross_section=3)
    assert cov.raw_rows == 6
    assert cov.rev_rows == 4        # only four rows had BOTH
    assert cov.residual_rows == 3
    assert cov.dates_total == 2
    assert cov.dates_residualized == 1
    assert cov.cross_section_min == 1 and cov.cross_section_max == 3


def test_summarize_neutralization_handles_an_all_missing_reversal():
    dates = pd.bdate_range("2023-01-02", periods=1)
    syms = ["A", "B"]
    raw = _panel(dates, syms, [[0.1, 0.2]], "q")
    rev = _panel(dates, syms, [[np.nan, np.nan]], "r")
    resid = _panel(dates, syms, [[np.nan, np.nan]], "f")
    cov = summarize_neutralization(raw, rev, resid, min_cross_section=3)
    assert cov.rev_rows == 0
    assert cov.residual_rows == 0
    assert cov.dates_residualized == 0
    assert not np.isfinite(cov.raw_rev_spearman_mean)  # no exposure is measurable


def test_reversal_from_a_daily_panel_is_the_t_minus_1_ratio():
    """The runner's reversal input is the panel's close column, read at T-1."""
    dates = pd.bdate_range("2023-01-02", periods=25)
    panel = pd.DataFrame(
        {"close": np.tile(np.arange(100.0, 125.0), 1)},
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"]),
    )
    rev = reversal_20(panel[["close"]], days=20)
    d = dates[22]
    assert float(rev.loc[(d, "A")]) == pytest.approx(-((121.0 / 101.0) - 1.0))


# --------------------------------------------------------------------------- #
# Evaluation core (two runs) — synthetic processed panels, no network
# --------------------------------------------------------------------------- #
def _synthetic_panels(n_days=90, n_symbols=15, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    factor = pd.Series(
        rng.standard_normal(len(idx)), index=idx, name="valley_price_quantile_20"
    )
    book = pd.DataFrame(
        {
            "value_ep": rng.standard_normal(len(idx)),
            "value_bp": rng.standard_normal(len(idx)),
            "volatility_20": rng.standard_normal(len(idx)),
        },
        index=idx,
    )
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
    spec = ValleyPriceQuantileFactor().spec
    reports = evaluate_two_runs(
        factor, spec, _eval_cfg(), price, book,
        universe_symbols=tuple(sorted(set(book.index.get_level_values("symbol")))),
        fee_rate=0.001,
        report_dir=tmp_path,
    )

    for report in (reports.no_book, reports.with_book):
        by = report.by_name()
        assert set(by) == set(MANDATORY_SECTIONS)
        assert all(isinstance(s, (Section, Skipped)) for s in by.values())
        assert report.verdict is not None

    assert isinstance(reports.no_book.by_name()["purity"], Skipped)
    assert reports.no_book.verdict.incremental.verdict == AXIS_NOT_ASSESSED

    purity = reports.with_book.by_name()["purity"]
    assert isinstance(purity, Section)
    assert purity.payload["known_factors_supplied"] is True
    assert "incremental_ic_ir" in purity.payload
    incr = reports.with_book.verdict.incremental.verdict
    assert incr in AXIS_VERDICTS and incr != AXIS_NOT_ASSESSED

    for p in (reports.no_book_md, reports.no_book_json,
              reports.with_book_md, reports.with_book_json):
        assert p.exists() and p.stat().st_size > 0

    for p in (reports.no_book_dashboard, reports.with_book_dashboard):
        assert p.exists() and p.stat().st_size > 20_000
        with open(p, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

    m = extract_metrics(reports.no_book)
    assert m["deployment"] in {"Adopt", "Watch", "Reject", "INSUFFICIENT-DATA"}
    assert "effective_samples" in m and "ic_ir" in m


def test_extract_metrics_surfaces_the_pr_l_comparison_quantities(tmp_path):
    """PR-L is compared head-on with PR-I / PR-J / PR-K on the same cell.

    Beyond the shared tradability quantities, this run needs ``ic_pearson_mean`` next to
    the rank ``ic_mean``: the PR-K review found their divergence predicted the
    monotonicity gate failing, and PR-L is the first test of that on a sign=+1 factor.
    """
    factor, book, price = _synthetic_panels()
    reports = evaluate_two_runs(
        factor, ValleyPriceQuantileFactor().spec, _eval_cfg(), price, book,
        universe_symbols=tuple(sorted(set(book.index.get_level_values("symbol")))),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    m = extract_metrics(reports.no_book)
    for key in (
        "long_short_turnover", "rank_autocorr_lag1", "half_life_periods",
        "cross_section_size_mean", "cross_section_size_median",
        "monotonicity_spearman", "gross_long_short_mean",
        "ic_mean", "ic_pearson_mean",
    ):
        assert key in m, key
        assert m[key] is not None, key
    net = m["net_long_short_by_cost"]
    assert isinstance(net, dict) and net
    assert {1.0, 2.0, 4.0} <= {float(k) for k in net}
    assert len({round(float(v), 12) for v in net.values()}) > 1
    # sign=+1 -> the aligned spreads are NOT mis-signed for this factor and are surfaced.
    assert isinstance(m["aligned_spread_by_cost"], dict)


def test_spec_sign_is_positive_so_aligned_spread_is_not_mis_signed():
    """The frozen layer's cost-sign defect only bites at sign=-1; PR-L is +1."""
    assert ValleyPriceQuantileFactor().spec.expected_ic_sign == 1


def test_no_book_run_never_reaches_adopt(tmp_path):
    factor, book, price = _synthetic_panels(seed=3)
    reports = evaluate_two_runs(
        factor, ValleyPriceQuantileFactor().spec, _eval_cfg(), price, book,
        universe_symbols=(),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    assert reports.no_book.verdict.verdict != "Adopt"
    assert reports.no_book.verdict.tradable.verdict == AXIS_NOT_ASSESSED


def test_min_valley_bars_default_matches_the_pr_i_floor():
    """PR-L reuses PR-I's valley floor so the two runs' coverage is comparable."""
    from data.clean.intraday_valley_vwap import VALLEY_VWAP_MIN_VALLEY_BARS

    assert VALLEY_QUANTILE_MIN_VALLEY_BARS == VALLEY_VWAP_MIN_VALLEY_BARS
