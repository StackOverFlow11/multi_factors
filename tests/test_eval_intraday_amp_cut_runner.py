"""PR-G runner: cache-only per-symbol stats loop + cross-sectional combine + two-run eval.

Network-free: the minute loader is seeded through a temp ``IntradayParquetStore`` (no
fetch closure, so ``stk_mins`` live calls are provably 0), and the two-run evaluation core
uses synthetic processed panels. The distinguishing PR-G behaviour is that the loader
assembles the FULL-UNIVERSE ``(V_mean, V_std)`` panel across the per-symbol loop and then
applies the cross-sectional z-score combine ONCE.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, MANDATORY_SECTIONS, Section, Skipped
from analytics.eval.verdict import AXIS_NOT_ASSESSED, AXIS_VERDICTS
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from factors.compute.intraday_derived import IntradayAmpCutFactor
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
from qt.eval_intraday_amp_cut import (
    _load_amp_cut_panel,
    evaluate_two_runs,
    extract_metrics,
)

# Within-day close chain for a 6-bar day: bar 0 has no return, bars 1..5 have returns
# [+0.05, -0.02, +0.03, -0.04, +0.01]; TOP-return bar is index 1, BOTTOM is index 4. With
# lam=0.20 and 5 valid bars (k=1), V_day = amp[1] - amp[4].
_CLOSE_CHAIN = [100.0, 105.0, 102.9, 105.987, 101.74752, 102.7649952]


def _stored_amp_rows(sym, day, amps):
    """STORED_COLUMNS-shaped 1min rows: low=100, high=100*(1+amp), close on _CLOSE_CHAIN."""
    base = pd.Timestamp(day) + pd.Timedelta("09:31:00")
    rows = []
    for i, (a, c) in enumerate(zip(amps, _CLOSE_CHAIN)):
        be = base + pd.Timedelta(minutes=i)
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


_DAY1 = "2021-07-01"
_DAY2 = "2021-07-02"

# Each symbol has two valid days with DISTINCT V_day (bottom-bar amp differs day1->day2),
# so V_std > 0, and DISTINCT (V_mean, V_std) across symbols so the day-2 cross-section is
# non-degenerate. V_day = 0.02 - amps[4].
_SYM_AMPS = {
    "AAA.SZ": ([0.10, 0.02, 0.03, 0.04, 0.05, 0.06], [0.10, 0.02, 0.03, 0.04, 0.07, 0.06]),
    "BBB.SZ": ([0.10, 0.02, 0.03, 0.04, 0.06, 0.06], [0.10, 0.02, 0.03, 0.04, 0.10, 0.06]),
    "CCC.SZ": ([0.10, 0.02, 0.03, 0.04, 0.04, 0.06], [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]),
}

# Small gates so a 2-day cache produces a finite (V_mean, V_std) pair on day 2 and a
# 3-symbol cross-section is enough to z-score.
_LOAD_KW = dict(
    lookback_days=10, lam=0.20, min_day_minutes=5, min_valid_days=2, min_cross_section=3
)


def _seed_symbol(store, sym):
    a1, a2 = _SYM_AMPS[sym]
    store.upsert(INTRADAY_ENDPOINT, sym, "1min", _stored_amp_rows(sym, _DAY1, a1), KEY_COLS)
    store.upsert(INTRADAY_ENDPOINT, sym, "1min", _stored_amp_rows(sym, _DAY2, a2), KEY_COLS)


# --------------------------------------------------------------------------- #
# Minute cache-only loader (per-symbol stats -> cross-sectional combine)
# --------------------------------------------------------------------------- #
def test_amp_cut_loader_is_cache_only_and_combines_cross_section(tmp_path):
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))
    for sym in ("AAA.SZ", "BBB.SZ", "CCC.SZ"):
        _seed_symbol(store, sym)

    cfg = _min_config(root, _DAY1, _DAY2)
    spec = IntradayAmpCutFactor().spec
    logger = logging.getLogger("test.amp_cut.loader")
    load = _load_amp_cut_panel(
        cfg, ["AAA.SZ", "BBB.SZ", "CCC.SZ", "DDD.SZ"], spec, logger, **_LOAD_KW
    )
    assert load.live_calls == 0                              # store read has no fetch closure
    assert set(load.covered) == {"AAA.SZ", "BBB.SZ", "CCC.SZ"}
    assert load.empty_symbols == ("DDD.SZ",)                 # uncovered disclosed, not fetched
    assert load.factor.name == spec.factor_id
    # only day 2 carries a finite (V_mean, V_std) pair (min_valid_days=2)
    d2 = pd.Timestamp(_DAY2)
    assert set(load.factor.index.get_level_values("date")) == {d2}
    assert load.factor.notna().all()
    # the cross-sectional z-scores centre at 0 on the date
    assert load.factor.loc[d2].sum() == pytest.approx(0.0, abs=1e-9)
    # the assembled stats panel had day1 (NaN) + day2 rows for all three symbols
    assert load.stats_rows == 6


def test_amp_cut_loader_blocks_when_nothing_cached(tmp_path):
    cfg = _min_config(tmp_path / "empty", _DAY1, _DAY2)
    spec = IntradayAmpCutFactor().spec
    with pytest.raises(ValueError, match="no requested symbol produced"):
        _load_amp_cut_panel(
            cfg, ["ZZZ.SZ"], spec, logging.getLogger("test.amp_cut.block"), **_LOAD_KW
        )


# --------------------------------------------------------------------------- #
# Evaluation core (two runs) — synthetic processed panels, no network
# --------------------------------------------------------------------------- #
def _synthetic_panels(n_days=90, n_symbols=15, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"{i:06d}.SZ" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    factor = pd.Series(
        rng.standard_normal(len(idx)), index=idx, name="intraday_amp_cut_10"
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


def test_amp_cut_evaluate_two_runs_produces_full_reports_and_incremental_axis(tmp_path):
    factor, book, price = _synthetic_panels()
    spec = IntradayAmpCutFactor().spec
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


def test_amp_cut_no_book_run_never_reaches_adopt(tmp_path):
    # Structural guarantee: with no book + no execution facts, at most Watch.
    factor, book, price = _synthetic_panels(seed=3)
    reports = evaluate_two_runs(
        factor, IntradayAmpCutFactor().spec, _eval_cfg(), price, book,
        universe_symbols=(),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    assert reports.no_book.verdict.verdict != "Adopt"
    assert reports.no_book.verdict.tradable.verdict == AXIS_NOT_ASSESSED
