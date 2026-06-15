"""I5a: shared event-driven backtest engine + intraday tail event model.

Covers the goal's four test groups:

1. Daily golden — the engine with ``DailyCloseEventModel`` reproduces the legacy
   ``BacktestDriver`` ledger (NAV / cost / turnover / feasibility / holdings) and
   the wrapper stays import-compatible.
2. Event-model unit — daily schedule == the monthly rebalance schedule; intraday
   anchors carry the configured decision/execution times and block missing bars.
3. Intraday PIT — exec-to-exec differs from close-to-close; perturbing post-cutoff
   bars leaves the decision feature unchanged; perturbing the execution bar moves
   returns; perturbing daily close does NOT move intraday returns; a missing
   execution bar blocks (never daily-fallbacks); holdings/turnover are achieved.
4. Config — existing configs validate; ``intraday_tail_rebalance`` needs
   ``intraday.enabled``; invalid window/model fail readably.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from data.cache.intraday_cache import TushareIntradayCache
from data.cache.intraday_coverage import IntradayCoverageLedger
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import normalize_intraday_bars
from data.clean.schema import normalize_panel
from portfolio.construct import TopNEqualWeight
from qt.config import RootConfig, load_config
from qt.pipeline import _FrameScores
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import DailyCloseEventModel, IntradayTailEventModel
from runtime.backtest.events import (
    HoldingPeriod,
    monthly_anchor_pairs,
    monthly_rebalance_dates,
    trading_calendar,
)
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import (
    REASON_NO_BAR,
    IntradayExecutionConfig,
)
from universe.static import StaticUniverse

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_I5A_CONFIG = _CONFIG_DIR / "phase_i5a_intraday_tail_framework.yaml"

# Month-end calendar with consecutive month-ends so the monthly schedule yields
# >=2 settled holding periods.
_DATES = [
    "2024-01-30", "2024-01-31",
    "2024-02-28", "2024-02-29",
    "2024-03-28", "2024-03-29",
]
_JAN, _FEB, _MAR = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #
def _daily_panel(closes: dict) -> pd.DataFrame:
    """Canonical (date, symbol) panel from ``{(date_str, symbol): close}``."""
    rows = []
    for (d, s), c in closes.items():
        rows.append(
            {
                "date": pd.Timestamp(d), "symbol": s,
                "open": c, "high": c, "low": c, "close": c,
                "volume": 1.0, "amount": 1.0, "adj_factor": 1.0,
            }
        )
    return normalize_panel(pd.DataFrame(rows))


def _grid_panel(symbols: list[str], price_fn) -> pd.DataFrame:
    closes = {}
    for d in _DATES:
        for s in symbols:
            closes[(d, s)] = price_fn(d, s)
    return _daily_panel(closes)


def _scores(panel_map: dict) -> _FrameScores:
    """`_FrameScores` over a ``{(date, symbol): score}`` map."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for (d, s) in panel_map], names=["date", "symbol"]
    )
    return _FrameScores(pd.Series(list(panel_map.values()), index=idx, name="score"))


def _minute_bars(specs: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Canonical 1min bars from ``(symbol, 'YYYY-MM-DD HH:MM:SS', close)`` specs."""
    rows = [
        {
            "time": pd.Timestamp(t), "symbol": s,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 1.0, "amount": float(c), "source_trade_time": t,
        }
        for (s, t, c) in specs
    ]
    return normalize_intraday_bars(pd.DataFrame(rows), freq="1min", data_lag="1min")


# --------------------------------------------------------------------------- #
# 1. Daily golden — engine(DailyCloseEventModel) == legacy BacktestDriver
# --------------------------------------------------------------------------- #
def _daily_inputs():
    panel = _grid_panel(["A", "B", "C"], lambda d, s: 100.0 + hash((d, s)) % 7)
    universe = StaticUniverse(["A", "B", "C"])
    scores = _scores({(d, s): {"A": 3.0, "B": 2.0, "C": 1.0}[s] for d in _DATES for s in "ABC"})
    return panel, universe, scores


def test_engine_daily_equals_driver_wrapper():
    panel, universe, scores = _daily_inputs()

    driver = BacktestDriver(
        universe=universe, scores=scores,
        constructor=TopNEqualWeight(2, long_only=True),
        execution=SimExecution(fee_rate=0.001), prices=panel,
        rebalance="monthly", fee_rate=0.001, initial_nav=1.0, cash_return=0.0,
    )
    driver_nav = driver.run()

    engine = BacktestEngine(
        model=DailyCloseEventModel(panel), universe=universe, scores=scores,
        constructor=TopNEqualWeight(2, long_only=True),
        execution=SimExecution(fee_rate=0.001), selection_panel=panel,
        initial_nav=1.0, cash_return=0.0,
    )
    engine_nav = engine.run()

    assert_frame_equal(driver_nav, engine_nav)
    assert_frame_equal(driver.feasibility_log(), engine.feasibility_log())
    assert_frame_equal(driver.holdings_log(), engine.holdings_log())


def test_backtest_driver_import_compatible_surface():
    panel, universe, scores = _daily_inputs()
    driver = BacktestDriver(
        universe=universe, scores=scores,
        constructor=TopNEqualWeight(2), execution=SimExecution(), prices=panel,
    )
    # Legacy public surface still present.
    assert callable(driver.run)
    assert callable(driver.rebalance_dates)
    assert callable(driver.feasibility_log)
    assert callable(driver.holdings_log)
    assert driver.rebalance_dates() == monthly_rebalance_dates(trading_calendar(panel))


# --------------------------------------------------------------------------- #
# 2. Event-model unit
# --------------------------------------------------------------------------- #
def test_daily_event_model_schedule_matches_monthly_pairs():
    panel, _, _ = _daily_inputs()
    model = DailyCloseEventModel(panel)
    periods = model.holding_periods()
    pairs = monthly_anchor_pairs(trading_calendar(panel))
    assert [(p.date, p.exit_date) for p in periods] == pairs
    # daily model: decision == execution == entry == the rebalance date.
    for p in periods:
        assert p.decision_ts == p.date == p.entry_date == p.execution_ts


def test_intraday_event_model_anchors_and_blocks_missing():
    panel = _grid_panel(["A", "B"], lambda d, s: 100.0)
    bars = _minute_bars(
        # A has an execution bar on both month-ends; B is missing Feb's.
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 11.0),
         ("A", f"{_MAR.date()} 14:51:00", 12.0),
         ("B", f"{_JAN.date()} 14:51:00", 20.0), ("B", f"{_MAR.date()} 14:51:00", 22.0)]
    )
    cfg = IntradayExecutionConfig()
    model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=cfg)

    periods = model.holding_periods()
    p0 = periods[0]
    assert p0.decision_ts == pd.Timestamp(f"{p0.date.date()} 14:50:00")
    assert p0.execution_ts == pd.Timestamp(f"{p0.date.date()} 14:51:00")

    # Feasibility at Feb (entry) — A has a bar (tradable), B does not (blocked both).
    feb_period = next(p for p in periods if p.date == _FEB)
    can_buy, can_sell = model.feasibility(feb_period, ["A", "B"])
    assert can_buy["A"] is True and can_sell["A"] is True
    assert can_buy["B"] is False and can_sell["B"] is False
    assert any(f.blocked and f.reason == REASON_NO_BAR for f in model.blocked_fills())


# --------------------------------------------------------------------------- #
# 3. Intraday PIT
# --------------------------------------------------------------------------- #
def test_intraday_exec_to_exec_differs_from_close_to_close():
    # Daily close falls; minute execution price rises -> opposite-signed returns.
    panel = _daily_panel(
        {(str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 50.0,
         (str(_JAN.date()), "B"): 100.0, (str(_FEB.date()), "B"): 50.0}
    )
    bars = _minute_bars(
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 11.0)]
    )
    cfg = IntradayExecutionConfig()
    intra = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=cfg)
    daily = DailyCloseEventModel(panel)
    period = HoldingPeriod(_JAN, _JAN, _FEB, _JAN, _JAN)

    intra_r = intra.holding_returns(period, ["A"])
    daily_r = daily.holding_returns(period, ["A"])
    assert intra_r["A"] == pytest.approx(11.0 / 10.0 - 1.0)   # +0.10 exec-to-exec
    assert daily_r["A"] == pytest.approx(50.0 / 100.0 - 1.0)  # -0.50 close-to-close
    assert intra_r["A"] != pytest.approx(daily_r["A"])


def test_perturb_execution_bar_changes_intraday_returns():
    panel = _daily_panel(
        {(str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 100.0}
    )
    period = HoldingPeriod(_JAN, _JAN, _FEB, _JAN, _JAN)
    base = _minute_bars(
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 11.0)]
    )
    bumped = _minute_bars(
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 20.0)]
    )
    r0 = IntradayTailEventModel(calendar_panel=panel, bars=base, cfg=IntradayExecutionConfig())
    r1 = IntradayTailEventModel(calendar_panel=panel, bars=bumped, cfg=IntradayExecutionConfig())
    assert r0.holding_returns(period, ["A"])["A"] != pytest.approx(
        r1.holding_returns(period, ["A"])["A"]
    )


def test_perturb_daily_close_does_not_change_intraday_returns():
    bars = _minute_bars(
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 11.0)]
    )
    period = HoldingPeriod(_JAN, _JAN, _FEB, _JAN, _JAN)
    panel_a = _daily_panel(
        {(str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 50.0}
    )
    panel_b = _daily_panel(
        {(str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 999.0}
    )
    m_a = IntradayTailEventModel(calendar_panel=panel_a, bars=bars, cfg=IntradayExecutionConfig())
    m_b = IntradayTailEventModel(calendar_panel=panel_b, bars=bars, cfg=IntradayExecutionConfig())
    assert m_a.holding_returns(period, ["A"])["A"] == pytest.approx(
        m_b.holding_returns(period, ["A"])["A"]
    )


def test_missing_execution_bar_blocks_never_daily_fallback():
    # A held into Feb but Feb has NO execution bar -> omitted from returns (flat),
    # never the daily close (which would be a -0.5 fallback).
    panel = _daily_panel(
        {(str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 50.0}
    )
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])  # no Feb bar
    period = HoldingPeriod(_JAN, _JAN, _FEB, _JAN, _JAN)
    model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig())
    r = model.holding_returns(period, ["A"])
    assert "A" not in r.index                       # omitted (flat), NOT -0.5
    assert any(f.blocked for f in model.blocked_fills())


def test_perturb_post_decision_bars_leave_decision_feature_unchanged():
    # The score = the I3 intraday_ret feature; perturbing a post-14:50 bar must not
    # change it (only available_time <= 14:50 bars enter).
    from data.clean.intraday_aggregate import asof_daily_features

    base = _minute_bars(
        [("A", f"{_JAN.date()} 09:30:00", 10.0),
         ("A", f"{_JAN.date()} 14:49:00", 12.0),   # available 14:50 -> included
         ("A", f"{_JAN.date()} 14:55:00", 99.0)]   # available 14:56 -> excluded
    )
    perturbed = _minute_bars(
        [("A", f"{_JAN.date()} 09:30:00", 10.0),
         ("A", f"{_JAN.date()} 14:49:00", 12.0),
         ("A", f"{_JAN.date()} 14:55:00", 777.0)]  # perturb the excluded bar
    )
    f0 = asof_daily_features(base, decision_time="14:50:00", session_open="09:30:00")
    f1 = asof_daily_features(perturbed, decision_time="14:50:00", session_open="09:30:00")
    assert_frame_equal(f0, f1)


def test_intraday_engine_turnover_and_holdings_are_achieved():
    # Full engine run with the intraday model: turnover/holdings reflect the
    # ACHIEVED book after a blocked entry, not the desired target.
    panel = _grid_panel(["A", "B"], lambda d, s: 100.0)
    bars = _minute_bars(
        # A executes on every month-end; B can never be entered (no exec bars).
        [("A", f"{_JAN.date()} 14:51:00", 10.0), ("A", f"{_FEB.date()} 14:51:00", 11.0),
         ("A", f"{_MAR.date()} 14:51:00", 12.0)]
    )
    universe = StaticUniverse(["A", "B"])
    # B scores highest, but it has no execution bar -> blocked -> A is held.
    scores = _scores({(d, s): {"A": 1.0, "B": 9.0}[s] for d in _DATES for s in "AB"})
    engine = BacktestEngine(
        model=IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig()),
        universe=universe, scores=scores, constructor=TopNEqualWeight(2),
        execution=SimExecution(fee_rate=0.0), selection_panel=panel,
    )
    nav = engine.run()
    holdings = engine.holdings_log()
    assert not nav.empty
    # B never appears in the achieved book (its buy was always blocked).
    assert "B" not in set(holdings["symbol"])
    assert set(holdings["symbol"]) <= {"A"}


# --------------------------------------------------------------------------- #
# 4. Config
# --------------------------------------------------------------------------- #
def test_i5a_config_validates():
    cfg = load_config(str(_I5A_CONFIG))
    assert cfg.backtest.event_order == "intraday_tail_rebalance"
    assert cfg.intraday is not None and cfg.intraday.enabled is True


def _i5a_dict() -> dict:
    return yaml.safe_load(_I5A_CONFIG.read_text())


def test_intraday_event_order_requires_enabled():
    d = _i5a_dict()
    d["intraday"]["enabled"] = False
    with pytest.raises(ValidationError, match="requires an 'intraday' section"):
        RootConfig(**d)

    d2 = _i5a_dict()
    d2.pop("intraday")
    with pytest.raises(ValidationError, match="requires an 'intraday' section"):
        RootConfig(**d2)


def test_default_daily_config_needs_no_intraday_section():
    d = _i5a_dict()
    d["backtest"]["event_order"] = "close_to_next_period"
    d.pop("intraday")
    cfg = RootConfig(**d)  # validates fine without an intraday section
    assert cfg.intraday is None


def test_invalid_execution_model_fails_readably():
    d = _i5a_dict()
    d["intraday"]["execution_model"] = "tail_vwap"
    with pytest.raises(ValidationError, match="execution_model"):
        RootConfig(**d)


def test_invalid_execution_window_fails_readably():
    d = _i5a_dict()
    d["intraday"]["execution_window"] = ["14:49:00", "14:56:59"]  # before decision
    with pytest.raises(ValidationError, match="execution_window"):
        RootConfig(**d)


# --------------------------------------------------------------------------- #
# 5. Partial cache coverage (require_cache_coverage semantics) + event audit
# --------------------------------------------------------------------------- #
def _synthetic_stk_mins(symbol, start_dt, end_dt):
    """Raw stk_mins-shaped 1min bars for [start_dt, end_dt] (one bar at 14:51/day)."""
    rows = []
    day = pd.Timestamp(start_dt).normalize()
    end = pd.Timestamp(end_dt).normalize()
    while day <= end:
        for clock in ("09:31:00", "14:51:00"):
            ts = day + pd.Timedelta(clock)
            rows.append(
                {"ts_code": symbol, "trade_time": str(ts), "open": 10.0,
                 "high": 10.0, "low": 10.0, "close": 10.0, "vol": 1.0, "amount": 10.0}
            )
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _cfg_with_cache(tmp_path, start, end, *, require: bool) -> RootConfig:
    d = _i5a_dict()
    d["data"]["cache"]["root_dir"] = str(tmp_path)
    d["data"]["start"] = start
    d["data"]["end"] = end
    d["intraday"]["require_cache_coverage"] = require
    return RootConfig(**d)


def _warm_one_symbol(tmp_path, symbol, start, end):
    cache = TushareIntradayCache(IntradayParquetStore(str(tmp_path)), IntradayCoverageLedger(str(tmp_path)))
    cache.stk_mins_1min(
        [symbol], f"{start} 00:00:00", f"{end} 23:59:59", _synthetic_stk_mins, freq="1min"
    )


def test_partial_coverage_required_fails(tmp_path):
    from qt.intraday_tail_framework import _load_minute_bars_cache_only

    start, end = "2024-01-02", "2024-01-03"
    _warm_one_symbol(tmp_path, "AAA.SH", start, end)  # only AAA is covered
    cfg = _cfg_with_cache(tmp_path, start, end, require=True)
    log = logging.getLogger("test.i5a")
    # require_cache_coverage=true + one uncovered symbol -> hard blocker (no silent drop).
    with pytest.raises(ValueError, match="require_cache_coverage=true"):
        _load_minute_bars_cache_only(cfg, ["AAA.SH", "BBB.SH"], log)


def test_partial_coverage_lenient_drops_uncovered(tmp_path):
    from qt.intraday_tail_framework import _load_minute_bars_cache_only

    start, end = "2024-01-02", "2024-01-03"
    _warm_one_symbol(tmp_path, "AAA.SH", start, end)
    cfg = _cfg_with_cache(tmp_path, start, end, require=False)
    log = logging.getLogger("test.i5a")
    bars, covered, uncovered, live = _load_minute_bars_cache_only(cfg, ["AAA.SH", "BBB.SH"], log)
    assert covered == ["AAA.SH"]
    assert uncovered == ["BBB.SH"]
    assert live == 0                 # read-only: zero live stk_mins calls
    assert not bars.empty


def test_event_log_exposes_exit_and_next_execution_anchors():
    panel, universe, scores = _daily_inputs()
    engine = BacktestEngine(
        model=DailyCloseEventModel(panel), universe=universe, scores=scores,
        constructor=TopNEqualWeight(2), execution=SimExecution(), selection_panel=panel,
    )
    engine.run()
    ev = engine.event_log()
    for col in ("exit_execution_ts", "next_decision_ts", "next_execution_ts"):
        assert col in ev.columns
    # daily: the exit execution anchor is the exit date's close.
    assert (ev["exit_execution_ts"] == ev["exit_date"]).all()
    # consecutive periods chain: next_execution_ts == the following row's execution_ts.
    execs = list(ev["execution_ts"])
    nexts = list(ev["next_execution_ts"])
    for i in range(len(execs) - 1):
        assert nexts[i] == execs[i + 1]
    assert pd.isna(nexts[-1])  # last settled period has no successor
