"""I5f: report-only intraday execution liquidity diagnostics.

Covers the formula primitives, the config schema (default-off + readable enabled
validation), and the report-only invariants (enabling the diagnostic does not move
NAV/turnover/cost/holdings; capacity uses the execution-minute amount ONLY; existing
execution blocks keep their original reasons and are never reclassified). All
network-free: synthetic bars / fills / panels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data.clean.intraday_schema import normalize_intraday_bars
from data.clean.schema import normalize_panel
from qt.config import LiquidityDiagnosticsCfg, load_config
from qt.pipeline import _FrameScores
from portfolio.construct import TopNEqualWeight
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import (
    REASON_MISSING_PRICE,
    REASON_NO_BAR,
    ExecutionFill,
    IntradayExecutionConfig,
)
from runtime.intraday_liquidity import (
    bar_capacity_notional,
    build_liquidity_diagnostics,
    capacity_ratio,
    desired_trade_notional,
    trade_direction,
)
from universe.static import StaticUniverse

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_DATES = [
    "2024-01-30", "2024-01-31",
    "2024-02-28", "2024-02-29",
    "2024-03-28", "2024-03-29",
]
_JAN, _FEB, _MAR = (
    pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29"),
)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _daily_panel(closes: dict) -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(d), "symbol": s,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 1.0, "amount": 1.0, "adj_factor": 1.0,
        }
        for (d, s), c in closes.items()
    ]
    return normalize_panel(pd.DataFrame(rows))


def _grid_panel(symbols: list[str]) -> pd.DataFrame:
    return _daily_panel({(d, s): 100.0 for d in _DATES for s in symbols})


def _minute_bars(specs: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """1min bars from ``(symbol, 'YYYY-MM-DD HH:MM:SS', close, amount)`` specs."""
    rows = [
        {
            "time": pd.Timestamp(t), "symbol": s,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 1.0, "amount": float(a), "source_trade_time": t,
        }
        for (s, t, c, a) in specs
    ]
    return normalize_intraday_bars(pd.DataFrame(rows), freq="1min", data_lag="1min")


def _scores(map_: dict) -> _FrameScores:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for (d, s) in map_], names=["date", "symbol"]
    )
    return _FrameScores(pd.Series(list(map_.values()), index=idx, name="score"))


def _plan(rows: list[tuple]) -> pd.DataFrame:
    """plan_log frame from ``(date, symbol, target_weight, current_weight)`` tuples."""
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s,
             "target_weight": tw, "current_weight": cw}
            for (d, s, tw, cw) in rows
        ]
    )


# --------------------------------------------------------------------------- #
# 1. formula primitives
# --------------------------------------------------------------------------- #
def test_capacity_ratio_formula():
    assert desired_trade_notional(0.3, 0.1, 1_000_000) == pytest.approx(200_000.0)
    assert bar_capacity_notional(1_000_000, 0.05) == pytest.approx(50_000.0)
    assert capacity_ratio(200_000.0, 50_000.0) == pytest.approx(0.25)
    # ratio >= 1 means the capped bar covers the trade.
    assert capacity_ratio(40_000.0, 50_000.0) == pytest.approx(1.25)


def test_trade_direction_labeling():
    assert trade_direction(0.3, 0.1) == "buy"
    assert trade_direction(0.1, 0.3) == "sell"
    assert trade_direction(0.2, 0.2) is None  # flat -> not a trade


def test_zero_desired_trade_avoids_division_by_zero():
    # capacity_ratio refuses a zero desired notional (would divide by zero).
    with pytest.raises(ValueError):
        capacity_ratio(0.0, 50_000.0)
    # The builder simply SKIPS flat rows: only the real trade is inspected.
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0, 1_000_000.0)])
    plan = _plan([
        (_JAN, "A", 1.0, 0.0),   # real buy
        (_JAN, "B", 0.5, 0.5),   # flat -> skipped
    ])
    fills = [ExecutionFill("A", _JAN, pd.Timestamp(f"{_JAN.date()} 14:51:00"), 10.0, False, None)]
    diag = build_liquidity_diagnostics(
        plan_log=plan, fills=fills, up_blocked_buy_keys=set(),
        down_blocked_sell_keys=set(), bars=bars,
        portfolio_notional=1_000_000, max_participation_rate=0.05,
    )
    assert diag.total_desired_trades == 1  # the flat row never becomes a trade
    assert diag.inspected == 1


@pytest.mark.parametrize("amount", [None, float("nan"), 0.0, -5.0])
def test_missing_nan_zero_negative_amount_is_missing_capacity(amount):
    # primitive: bad amount -> None capacity (never inferred as zero).
    assert bar_capacity_notional(amount, 0.05) is None
    # builder: an executable trade whose exec-minute amount is bad is reported as
    # MISSING capacity data, not below-capacity and not a feasibility block. None
    # models "no bar row found at the exec time"; nan/0/-5 model a present-but-bad
    # amount on the selected exec bar.
    if amount is None:
        bars = _minute_bars([("Z", f"{_JAN.date()} 14:51:00", 10.0, 1.0)])  # unrelated bar only
    else:
        bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0, amount)])
    plan = _plan([(_JAN, "A", 1.0, 0.0)])
    fills = [ExecutionFill("A", _JAN, pd.Timestamp(f"{_JAN.date()} 14:51:00"), 10.0, False, None)]
    diag = build_liquidity_diagnostics(
        plan_log=plan, fills=fills, up_blocked_buy_keys=set(),
        down_blocked_sell_keys=set(), bars=bars,
        portfolio_notional=1_000_000, max_participation_rate=0.05,
    )
    assert diag.missing_capacity_rows == 1
    assert diag.inspected == 0
    assert diag.below_capacity == 0
    assert diag.feasibility_blocked == 0
    assert diag.trades[0].missing_capacity_data is True
    assert diag.trades[0].capacity_ratio is None


# --------------------------------------------------------------------------- #
# 2. config schema
# --------------------------------------------------------------------------- #
def test_default_off_and_old_configs_validate_unchanged():
    # default: the nested block is present but disabled -> behaviour unchanged.
    from qt.config import IntradayCfg

    ic = IntradayCfg()
    assert ic.liquidity_diagnostics.enabled is False
    # every shipped config still validates.
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        load_config(str(path))  # raises on failure


def test_i5f_config_loads_and_enables():
    cfg = load_config(str(_CONFIG_DIR / "phase_i5f_intraday_liquidity_diagnostics.yaml"))
    ld = cfg.intraday.liquidity_diagnostics
    assert ld.enabled is True
    assert ld.portfolio_notional == 10_000_000
    assert ld.max_participation_rate == 0.05
    assert ld.mode == "report_only"


def test_enabled_requires_positive_notional():
    with pytest.raises(ValueError, match="portfolio_notional"):
        LiquidityDiagnosticsCfg(enabled=True, portfolio_notional=None)
    with pytest.raises(ValueError, match="portfolio_notional"):
        LiquidityDiagnosticsCfg(enabled=True, portfolio_notional=-1.0)
    # positive notional is accepted.
    ok = LiquidityDiagnosticsCfg(enabled=True, portfolio_notional=1_000_000)
    assert ok.portfolio_notional == 1_000_000


def test_invalid_participation_rate_fails_readably():
    with pytest.raises(ValueError, match="max_participation_rate"):
        LiquidityDiagnosticsCfg(
            enabled=True, portfolio_notional=1_000_000, max_participation_rate=0.0
        )
    with pytest.raises(ValueError, match="max_participation_rate"):
        LiquidityDiagnosticsCfg(
            enabled=True, portfolio_notional=1_000_000, max_participation_rate=1.5
        )


def test_unsupported_mode_fails_readably():
    with pytest.raises(ValueError, match="report_only"):
        LiquidityDiagnosticsCfg(
            enabled=True, portfolio_notional=1_000_000, mode="enforce"
        )


def test_disabled_ignores_other_fields():
    # disabled -> the strict checks do not fire (defaults preserve all configs).
    cfg = LiquidityDiagnosticsCfg(enabled=False, portfolio_notional=None, mode="enforce")
    assert cfg.enabled is False


# --------------------------------------------------------------------------- #
# 3. report-only invariants (engine + helper)
# --------------------------------------------------------------------------- #
def _intraday_inputs():
    panel = _grid_panel(["A", "B"])
    universe = StaticUniverse(["A", "B"])
    scores = _scores({(d, s): {"A": 2.0, "B": 1.0}[s] for d in _DATES for s in "AB"})
    bars = _minute_bars(
        [(s, f"{day.date()} 14:51:00", 10.0 + i, 1_000_000.0)
         for i, day in enumerate((_JAN, _FEB, _MAR)) for s in ("A", "B")]
    )
    return panel, universe, scores, bars


def _engine(panel, universe, scores, bars, *, record_plan):
    return BacktestEngine(
        model=IntradayTailEventModel(
            calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig()
        ),
        universe=universe,
        scores=scores,
        constructor=TopNEqualWeight(2, long_only=True),
        execution=SimExecution(fee_rate=0.001),
        selection_panel=panel,
        initial_nav=1.0,
        cash_return=0.0,
        record_rebalance_plan=record_plan,
    )


def test_enabling_plan_does_not_change_backtest():
    panel, universe, scores, bars = _intraday_inputs()

    base = _engine(panel, universe, scores, bars, record_plan=False)
    base_nav = base.run()

    withplan = _engine(panel, universe, scores, bars, record_plan=True)
    plan_nav = withplan.run()

    # NAV / turnover / cost AND holdings/feasibility are byte-identical: the plan
    # log is pure bookkeeping; the report-only diagnostic never moves the book.
    assert_frame_equal(base_nav, plan_nav)
    assert_frame_equal(base.feasibility_log(), withplan.feasibility_log())
    assert_frame_equal(base.holdings_log(), withplan.holdings_log())
    # disabled (record_plan=False) -> NO plan rows recorded -> no diagnostic input.
    assert base.rebalance_plan_log().empty
    assert not withplan.rebalance_plan_log().empty


def test_diagnostics_use_execution_minute_amount_only():
    # The exec bar (14:51) carries amount 50e6; a LATER same-day minute carries a
    # huge 999e6 that must NEVER be used. Capacity must reflect the 14:51 amount.
    exec_time = pd.Timestamp(f"{_JAN.date()} 14:51:00")
    bars = _minute_bars([
        ("A", f"{_JAN.date()} 14:51:00", 10.0, 50_000_000.0),   # selected exec bar
        ("A", f"{_JAN.date()} 14:55:00", 10.0, 999_000_000.0),  # later minute: ignored
    ])
    plan = _plan([(_JAN, "A", 1.0, 0.0)])  # desired = 1.0 * 1e6 = 1e6
    fills = [ExecutionFill("A", _JAN, exec_time, 10.0, False, None)]
    diag = build_liquidity_diagnostics(
        plan_log=plan, fills=fills, up_blocked_buy_keys=set(),
        down_blocked_sell_keys=set(), bars=bars,
        portfolio_notional=1_000_000, max_participation_rate=0.05,
    )
    t = diag.trades[0]
    # 50e6 * 0.05 = 2.5e6 (NOT 999e6 * 0.05); ratio = 2.5e6 / 1e6 = 2.5
    assert t.bar_capacity_notional == pytest.approx(2_500_000.0)
    assert t.capacity_ratio == pytest.approx(2.5)
    assert diag.below_capacity == 0


def test_existing_blocks_keep_reason_not_reclassified():
    # U: buy blocked by raw up-limit; Dn: sell blocked by raw down-limit;
    # N: missing execution bar; M: missing price. None should be reclassified as a
    # liquidity (missing-capacity / below-capacity) row.
    bars = _minute_bars([
        ("U", f"{_JAN.date()} 14:51:00", 10.0, 1_000_000.0),
        ("Dn", f"{_JAN.date()} 14:51:00", 10.0, 1_000_000.0),
    ])
    plan = _plan([
        (_JAN, "U", 1.0, 0.0),    # buy
        (_JAN, "Dn", 0.0, 1.0),   # sell
        (_JAN, "N", 1.0, 0.0),    # buy, no bar
        (_JAN, "M", 1.0, 0.0),    # buy, missing price
    ])
    fills = [
        ExecutionFill("U", _JAN, pd.Timestamp(f"{_JAN.date()} 14:51:00"), 10.0, False, None),
        ExecutionFill("Dn", _JAN, pd.Timestamp(f"{_JAN.date()} 14:51:00"), 10.0, False, None),
        ExecutionFill("N", _JAN, None, None, True, REASON_NO_BAR),
        ExecutionFill("M", _JAN, pd.Timestamp(f"{_JAN.date()} 14:51:00"), None, True, REASON_MISSING_PRICE),
    ]
    diag = build_liquidity_diagnostics(
        plan_log=plan, fills=fills,
        up_blocked_buy_keys={(_JAN.normalize(), "U")},
        down_blocked_sell_keys={(_JAN.normalize(), "Dn")},
        bars=bars, portfolio_notional=1_000_000, max_participation_rate=0.05,
    )
    by_sym = {t.symbol: t for t in diag.trades}
    assert by_sym["U"].feasibility_blocked and by_sym["U"].block_reason == "up_limit"
    assert by_sym["Dn"].feasibility_blocked and by_sym["Dn"].block_reason == "down_limit"
    assert by_sym["N"].feasibility_blocked and by_sym["N"].block_reason == REASON_NO_BAR
    assert by_sym["M"].feasibility_blocked and by_sym["M"].block_reason == REASON_MISSING_PRICE
    # none reclassified as liquidity issues, none given a capacity ratio.
    assert diag.feasibility_blocked == 4
    assert diag.missing_capacity_rows == 0
    assert diag.below_capacity == 0
    assert diag.inspected == 0
    assert all(t.capacity_ratio is None for t in diag.trades)


def test_below_capacity_flagged_and_top_constrained_sorted():
    exec_time = pd.Timestamp(f"{_JAN.date()} 14:51:00")
    # Thin bar (amount 5e6 -> capacity 250k) vs a large desired trade (1e6) -> ratio 0.25.
    bars = _minute_bars([
        ("A", f"{_JAN.date()} 14:51:00", 10.0, 5_000_000.0),    # ratio 0.25 (constrained)
        ("B", f"{_JAN.date()} 14:51:00", 10.0, 100_000_000.0),  # ratio 5.0 (ample)
    ])
    plan = _plan([(_JAN, "A", 1.0, 0.0), (_JAN, "B", 1.0, 0.0)])
    fills = [
        ExecutionFill("A", _JAN, exec_time, 10.0, False, None),
        ExecutionFill("B", _JAN, exec_time, 10.0, False, None),
    ]
    diag = build_liquidity_diagnostics(
        plan_log=plan, fills=fills, up_blocked_buy_keys=set(),
        down_blocked_sell_keys=set(), bars=bars,
        portfolio_notional=1_000_000, max_participation_rate=0.05,
    )
    assert diag.inspected == 2
    assert diag.below_capacity == 1  # only A < 1.0
    assert diag.ratio_stats["min"] == pytest.approx(0.25)
    # top constrained lists the lowest ratio first.
    assert diag.top_constrained[0].symbol == "A"
    assert diag.top_constrained[0].capacity_ratio == pytest.approx(0.25)
