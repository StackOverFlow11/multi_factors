"""PR-I runner: cache-only minute loader + the two-run evaluation core (no network)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, MANDATORY_SECTIONS, Section, Skipped
from analytics.eval.verdict import AXIS_NOT_ASSESSED, AXIS_VERDICTS
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from factors.compute.intraday_derived import ValleyRelativeVwapFactor
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
from qt.eval_valley_relative_vwap import (
    _load_valley_relative_vwap_panel,
    evaluate_two_runs,
    extract_metrics,
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
# then the hand-computed CASE A test day: 12 slots, eruptions at 3 and 7 (volume 200 /
# amount 4000 -> price 20); the ten valley bars carry volume 100 with amount 1000 (x5)
# and 1200 (x5) -> valley VWAP 11.0, day VWAP 19000/1400, ratio 77/95.
_DAYS = ("2021-07-01", "2021-07-02", "2021-07-03")
_TEST_DAY = "2021-07-04"
_N_SLOTS = 12
_ERUPT = (3, 7)
_CASE_A_RATIO = 77.0 / 95.0
_BG_VOLS = [100.0] * _N_SLOTS
_BG_AMTS = [1000.0] * _N_SLOTS


def _case_a():
    vols = [100.0] * _N_SLOTS
    amts = [0.0] * _N_SLOTS
    valley_slots = [s for s in range(_N_SLOTS) if s not in _ERUPT]
    for i, s in enumerate(valley_slots):
        amts[s] = 1000.0 if i < 5 else 1200.0
    for s in _ERUPT:
        vols[s] = 200.0
        amts[s] = 4000.0
    return vols, amts


# Small gates so a 4-day cache produces a value (baseline needs >= 2 prior obs; the test
# day carries 10 tradable valley bars). lookback_days=1 so the test day's value IS its own
# ratio: with these tiny gates the last background day also becomes valid, and a day where
# NOTHING erupts has ratio exactly 1.0 (every bar is a valley, so the two VWAPs coincide),
# which would otherwise average into the assertion.
_LOAD_KW = dict(
    lookback_days=1, baseline_days=20, baseline_min_obs=2,
    sigma_k=1.0, min_valid_days=1, min_classifiable=1, min_valley_bars=1,
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
    spec = ValleyRelativeVwapFactor().spec
    logger = logging.getLogger("test.vrv.loader")
    load = _load_valley_relative_vwap_panel(
        cfg, ["AAA.SZ", "BBB.SZ", "CCC.SZ"], spec, logger, **_LOAD_KW
    )
    assert load.live_calls == 0                       # store read has no fetch closure
    assert set(load.covered) == {"AAA.SZ", "BBB.SZ"}  # both produced a value
    assert load.empty_symbols == ("CCC.SZ",)          # uncovered disclosed, not fetched
    assert load.factor.name == spec.factor_id
    assert load.factor.notna().any()
    # the test day carries the hand-computed valley/day VWAP ratio
    d = pd.Timestamp(_TEST_DAY)
    assert load.factor.loc[(d, "AAA.SZ")] == pytest.approx(_CASE_A_RATIO)
    assert load.factor.loc[(d, "BBB.SZ")] == pytest.approx(_CASE_A_RATIO)
    assert set(load.factor.index.get_level_values("symbol")) == {"AAA.SZ", "BBB.SZ"}
    # the last background day (nothing erupts -> every bar is a valley) has ratio
    # exactly 1.0: the two VWAPs are sums over the SAME bar set
    bg = pd.Timestamp(_DAYS[-1])
    assert load.factor.loc[(bg, "AAA.SZ")] == pytest.approx(1.0)


def test_minute_loader_blocks_when_nothing_cached(tmp_path):
    cfg = _min_config(tmp_path / "empty", "2021-07-01", "2021-07-04")
    spec = ValleyRelativeVwapFactor().spec
    with pytest.raises(ValueError, match="no requested symbol produced"):
        _load_valley_relative_vwap_panel(
            cfg, ["ZZZ.SZ"], spec, logging.getLogger("test.vrv.block"), **_LOAD_KW
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
        rng.standard_normal(len(idx)), index=idx, name="valley_relative_vwap_20"
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
    spec = ValleyRelativeVwapFactor().spec
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
        factor, ValleyRelativeVwapFactor().spec, _eval_cfg(), price, book,
        universe_symbols=(),
        fee_rate=0.001,
        report_dir=tmp_path,
    )
    assert reports.no_book.verdict.verdict != "Adopt"
    assert reports.no_book.verdict.tradable.verdict == AXIS_NOT_ASSESSED
