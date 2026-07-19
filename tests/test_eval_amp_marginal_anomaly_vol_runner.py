"""PR-E runner: cache-only minute loader + the two-run evaluation core (no network)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, MANDATORY_SECTIONS, Section, Skipped
from analytics.eval.verdict import AXIS_NOT_ASSESSED, AXIS_VERDICTS
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from factors.compute.intraday_derived import AmpMarginalAnomalyVolFactor
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
from qt.eval_amp_marginal_anomaly_vol import (
    _load_amp_anomaly_vol_panel,
    evaluate_two_runs,
    extract_metrics,
)


# --------------------------------------------------------------------------- #
# Minute cache-only loader
# --------------------------------------------------------------------------- #
def _stored_rows(sym, day, closes, amps):
    """Build STORED_COLUMNS-shaped 1min rows for one session, ON the 5min grid.

    Bar i sits at ``09:35 + 5*i`` minutes (an on-grid time), so the runner's internal
    1min -> 5min resample gives one clean 5min bar per input bar with
    ``high=100*(1+amp_i)``, ``low=100`` (amplitude ``amp_i`` exactly) and
    ``close=close_i`` (the within-day return). All bars sit inside the 14:50 PIT window.
    """
    base = pd.Timestamp(day) + pd.Timedelta("09:35:00")
    rows = []
    for i, (c, a) in enumerate(zip(closes, amps)):
        be = base + pd.Timedelta(minutes=5 * i)
        rows.append(
            {
                "symbol": sym,
                "bar_end": be,
                "source_trade_time": be,
                "open": c,
                "high": 100.0 * (1.0 + a),
                "low": 100.0,
                "close": c,
                "volume": 1.0,
                "amount": 1.0,
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


# closes/amps designed so each day has 5 within-day pairs and exactly 2 anomalies
# (the two 0.20 |Δamp| bars) -> a finite value with small gates.
_CLOSES = [100.0, 100.0, 100.0, 110.0, 100.0, 105.0]
_AMPS = [0.02, 0.03, 0.04, 0.24, 0.04, 0.05]


def test_minute_loader_is_cache_only_and_discloses_empty(tmp_path):
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    # Two symbols with cached minute bars across two trading days; a third symbol has
    # NO cached minute at all (must be disclosed as empty, never fetched).
    for sym in ("AAA.SZ", "BBB.SZ"):
        for day in ("2021-07-01", "2021-07-02"):
            store.upsert(INTRADAY_ENDPOINT, sym, "1min",
                         _stored_rows(sym, day, _CLOSES, _AMPS), KEY_COLS)

    cfg = _min_config(root, "2021-07-01", "2021-07-02")
    spec = AmpMarginalAnomalyVolFactor().spec
    logger = logging.getLogger("test.amav.loader")
    load = _load_amp_anomaly_vol_panel(
        cfg, ["AAA.SZ", "BBB.SZ", "CCC.SZ"], spec, logger,
        lookback_days=20, min_pool=4, min_selected=2, sigma_k=1.0,
    )
    assert load.live_calls == 0                       # store read has no fetch closure
    assert set(load.covered) == {"AAA.SZ", "BBB.SZ"}  # both produced a value
    assert load.empty_symbols == ("CCC.SZ",)          # uncovered disclosed, not fetched
    assert load.factor.name == spec.factor_id
    assert load.factor.notna().any()
    # values live only on the covered symbols
    assert set(load.factor.index.get_level_values("symbol")) == {"AAA.SZ", "BBB.SZ"}


def test_minute_loader_blocks_when_nothing_cached(tmp_path):
    cfg = _min_config(tmp_path / "empty", "2021-07-01", "2021-07-02")
    spec = AmpMarginalAnomalyVolFactor().spec
    with pytest.raises(ValueError, match="no requested symbol produced"):
        _load_amp_anomaly_vol_panel(
            cfg, ["ZZZ.SZ"], spec, logging.getLogger("test.amav.block"),
            lookback_days=20, min_pool=4, min_selected=2, sigma_k=1.0,
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
        rng.standard_normal(len(idx)), index=idx, name="amp_marginal_anomaly_vol_20"
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
    spec = AmpMarginalAnomalyVolFactor().spec
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
    # the axis is now assessed (not NOT_ASSESSED) and is a valid axis state
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
        factor, AmpMarginalAnomalyVolFactor().spec, _eval_cfg(), price, book,
        universe_symbols=(),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    assert reports.no_book.verdict.verdict != "Adopt"
    assert reports.no_book.verdict.tradable.verdict == AXIS_NOT_ASSESSED
