"""Execution price basis: the tail fill is the execution bar's VWAP by default.

The intraday tail model picks WHICH 1min bar fills (``next_minute_close``: the
earliest bar in ``[14:51, 14:56:59]``). WHAT price that bar fills at is the
separate ``execution_price_basis``, and the default is now ``bar_vwap`` =
``amount / volume``: the volume-weighted mean of every trade printed in that
minute, rather than the single closing tick.

What is locked here:

1. defaults / config — ``bar_vwap`` is the default at both the runtime config and
   the pydantic ``IntradayCfg``; an unknown basis fails readably; the pydantic
   Literal and the runtime supported set cannot drift apart; the runner actually
   forwards the setting (otherwise the knob would be dead);
2. arithmetic — hand-computed VWAP cases where ``amount/volume`` differs from the
   bar's close AND from its open/high/low, so a basis mix-up cannot pass;
3. undefined VWAP — non-positive / non-finite / missing ``volume`` or ``amount``
   is an EXPLICIT block on the existing missing-price path, never a fallback to
   the bar close and never to the daily close (the daily panel here carries a
   deliberately contradictory close to prove that);
4. I5b ordering + raw basis — a missing bar still blocks both directions BEFORE
   any limit logic; an undefined VWAP blocks before it too; the limit gate
   compares the RAW price that actually executes (the VWAP), and the daily close
   never enters the comparison;
5. ``bar_close`` reproduces the pre-VWAP behaviour (the I4 locks in
   ``test_intraday_execution.py`` assert the original values under that basis).

All network-free: synthetic bars / panels / limit rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from data.clean.intraday_schema import normalize_intraday_bars
from data.clean.schema import normalize_panel
from portfolio.construct import TopNEqualWeight
from qt.config import IntradayCfg, load_config
from qt.intraday_tail_framework import _exec_cfg_from
from qt.pipeline import _FrameScores
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import (
    PRICE_BASIS_CLOSE,
    PRICE_BASIS_VWAP,
    REASON_MISSING_PRICE,
    REASON_NO_BAR,
    SUPPORTED_PRICE_BASES,
    IntradayExecutionConfig,
    bar_execution_price,
    build_execution_prices,
    resolve_fill,
)
from universe.static import StaticUniverse

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_I5A_CONFIG = _CONFIG_DIR / "phase_i5a_intraday_tail_framework.yaml"

_DATES = [
    "2024-01-30", "2024-01-31",
    "2024-02-28", "2024-02-29",
    "2024-03-28", "2024-03-29",
]
_JAN, _FEB, _MAR = (
    pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29"),
)

_VWAP = IntradayExecutionConfig()                                        # default
_CLOSE = IntradayExecutionConfig(execution_price_basis=PRICE_BASIS_CLOSE)


# --------------------------------------------------------------------------- #
# builders — bar specs are (symbol, timestamp, close, volume, amount)
# --------------------------------------------------------------------------- #
def _bars(specs: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    rows = [
        {
            "time": pd.Timestamp(t), "symbol": s,
            # open/high/low are deliberately OFF the close and off the VWAP, so a
            # test that accidentally reads any of them is visible.
            "open": c - 0.25, "high": c + 0.75, "low": c - 0.75, "close": c,
            "volume": v, "amount": a, "source_trade_time": t,
        }
        for (s, t, c, v, a) in specs
    ]
    return normalize_intraday_bars(pd.DataFrame(rows), freq="1min", data_lag="1min")


def _day_bars(bars: pd.DataFrame, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
    work = bars.reset_index()
    work["date"] = work["bar_end"].dt.normalize()
    return work[(work["symbol"] == symbol) & (work["date"] == pd.Timestamp(date))]


def _daily_panel(symbols: list[str], close: float) -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(d), "symbol": s,
            "open": close, "high": close, "low": close, "close": close,
            "volume": 1.0, "amount": 1.0, "adj_factor": 1.0,
        }
        for d in _DATES for s in symbols
    ]
    return normalize_panel(pd.DataFrame(rows))


def _scores(values: dict[str, float]) -> _FrameScores:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for d in _DATES for s in values],
        names=["date", "symbol"],
    )
    return _FrameScores(
        pd.Series([values[s] for _ in _DATES for s in values], index=idx, name="score")
    )


def _limits(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s, "up_limit": up, "down_limit": dn}
            for (d, s, up, dn) in rows
        ]
    )


def _period(model: IntradayTailEventModel, date: pd.Timestamp):
    return next(p for p in model.holding_periods() if p.date == date)


def _fill(bars: pd.DataFrame, symbol: str, date: pd.Timestamp, cfg):
    return resolve_fill(symbol, date, _day_bars(bars, symbol, date), cfg)


# --------------------------------------------------------------------------- #
# 1. defaults, config validation, wiring
# --------------------------------------------------------------------------- #
def test_default_basis_is_bar_vwap_at_runtime_and_in_config():
    assert IntradayExecutionConfig().execution_price_basis == PRICE_BASIS_VWAP
    assert IntradayCfg().execution_price_basis == PRICE_BASIS_VWAP


def test_unknown_basis_fails_readably():
    with pytest.raises(ValueError, match="execution_price_basis"):
        IntradayExecutionConfig(execution_price_basis="daily_close")
    with pytest.raises(ValidationError, match="execution_price_basis"):
        IntradayCfg(execution_price_basis="daily_close")
    # the pure pricing function refuses an unknown basis rather than guessing.
    bar = pd.Series({"close": 10.0, "volume": 100.0, "amount": 1200.0})
    with pytest.raises(ValueError, match="execution_price_basis"):
        bar_execution_price(bar, "daily_close")


def test_config_literal_cannot_drift_from_runtime_supported_set():
    literal = get_args(IntradayCfg.model_fields["execution_price_basis"].annotation)
    assert set(literal) == set(SUPPORTED_PRICE_BASES)


def test_all_shipped_configs_still_validate():
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        load_config(str(path))  # raises on failure


def test_runner_forwards_the_price_basis():
    """The knob must reach ``IntradayExecutionConfig`` — otherwise it is dead."""
    cfg = load_config(str(_I5A_CONFIG))
    assert _exec_cfg_from(cfg).execution_price_basis == PRICE_BASIS_VWAP
    pinned = cfg.model_copy(
        update={
            "intraday": cfg.intraday.model_copy(
                update={"execution_price_basis": PRICE_BASIS_CLOSE}
            )
        }
    )
    assert _exec_cfg_from(pinned).execution_price_basis == PRICE_BASIS_CLOSE


# --------------------------------------------------------------------------- #
# 2. VWAP arithmetic — hand-computed, and distinguishable from every other price
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "close, volume, amount, expected_vwap",
    [
        # 1_200 RMB traded over 100 shares -> 12.0, while the last tick was 11.0.
        (11.0, 100.0, 1_200.0, 12.0),
        # 3e6 RMB over 200k shares -> 15.0, while the last tick was 20.0.
        (20.0, 200_000.0, 3_000_000.0, 15.0),
    ],
)
def test_fill_price_is_bar_amount_over_volume(close, volume, amount, expected_vwap):
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", close, volume, amount)])
    fill = _fill(bars, "A", _JAN, _VWAP)

    assert not fill.blocked
    assert fill.exec_time == pd.Timestamp(f"{_JAN.date()} 14:51:00")
    assert fill.exec_price == pytest.approx(expected_vwap)
    # ... and it is NOT the close, nor any other price printed on that bar, so the
    # assertion has real discriminating power.
    bar = _day_bars(bars, "A", _JAN).iloc[0]
    for column in ("close", "open", "high", "low"):
        assert fill.exec_price != pytest.approx(float(bar[column]))


def test_bar_close_basis_prices_the_same_bar_at_its_close():
    """Same bar, other basis: proves the switch actually changes the fill."""
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 11.0, 100.0, 1_200.0)])
    assert _fill(bars, "A", _JAN, _CLOSE).exec_price == pytest.approx(11.0)
    assert _fill(bars, "A", _JAN, _VWAP).exec_price == pytest.approx(12.0)


def test_vwap_equals_close_only_when_amount_is_close_times_volume():
    """Why the I5a/I5b/I5d fixtures are basis-neutral — by construction, not luck."""
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 13.5, 400.0, 13.5 * 400.0)])
    assert _fill(bars, "A", _JAN, _VWAP).exec_price == pytest.approx(13.5)
    assert _fill(bars, "A", _JAN, _CLOSE).exec_price == pytest.approx(13.5)


def test_window_selection_is_unchanged_by_the_basis():
    """The basis prices the bar; it never re-picks which bar executes."""
    bars = _bars([
        ("A", f"{_JAN.date()} 14:53:00", 11.0, 100.0, 1_200.0),  # earliest in window
        ("A", f"{_JAN.date()} 14:55:00", 50.0, 100.0, 9_900.0),  # later: ignored
    ])
    for cfg in (_VWAP, _CLOSE):
        assert _fill(bars, "A", _JAN, cfg).exec_time == pd.Timestamp(
            f"{_JAN.date()} 14:53:00"
        )
    assert _fill(bars, "A", _JAN, _VWAP).exec_price == pytest.approx(12.0)
    # a bar outside the window is still no bar at all.
    outside = _bars([("A", f"{_JAN.date()} 14:00:00", 11.0, 100.0, 1_200.0)])
    blocked = _fill(outside, "A", _JAN, _VWAP)
    assert blocked.blocked and blocked.reason == REASON_NO_BAR


# --------------------------------------------------------------------------- #
# 3. undefined VWAP -> explicit block, never a fallback price
# --------------------------------------------------------------------------- #
_UNDEFINED = [
    pytest.param(0.0, 1_200.0, id="zero-volume"),
    pytest.param(-100.0, 1_200.0, id="negative-volume"),
    pytest.param(float("nan"), 1_200.0, id="nan-volume"),
    pytest.param(float("inf"), 1_200.0, id="inf-volume"),
    pytest.param(100.0, 0.0, id="zero-amount"),
    pytest.param(100.0, -5.0, id="negative-amount"),
    pytest.param(100.0, float("nan"), id="nan-amount"),
    pytest.param(100.0, float("inf"), id="inf-amount"),
]


@pytest.mark.parametrize("volume, amount", _UNDEFINED)
def test_undefined_vwap_is_blocked_and_never_falls_back_to_the_bar_close(
    volume, amount
):
    close = 11.0
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", close, volume, amount)])

    fill = _fill(bars, "A", _JAN, _VWAP)
    assert fill.blocked
    # joins the EXISTING missing-price block path: same reason, and the bar's
    # timestamp is kept because the bar did exist.
    assert fill.reason == REASON_MISSING_PRICE
    assert fill.exec_time == pd.Timestamp(f"{_JAN.date()} 14:51:00")
    assert fill.exec_price is None
    # The close is a perfectly usable number on this very bar — a fallback would
    # have produced exactly it. It must not appear.
    assert _fill(bars, "A", _JAN, _CLOSE).exec_price == pytest.approx(close)
    assert fill.exec_price != close
    # the pricing primitive says "undefined", not "zero".
    bar = _day_bars(bars, "A", _JAN).iloc[0]
    assert bar_execution_price(bar, PRICE_BASIS_VWAP) is None


@pytest.mark.parametrize("volume, amount", _UNDEFINED)
def test_undefined_vwap_leaves_nan_in_the_price_matrix(volume, amount):
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 11.0, volume, amount)])
    prices, fills = build_execution_prices(bars, [_JAN], ["A"], _VWAP)
    assert pd.isna(prices.loc[_JAN, "A"])
    assert not (prices == 11.0).any().any()  # the close never leaks into the matrix
    assert [f.reason for f in fills] == [REASON_MISSING_PRICE]


def test_undefined_vwap_blocks_both_directions_and_earns_nothing():
    # A's execution minute has no traded shares; B is normal. The daily panel
    # carries a contradictory close (999.0) that must never be substituted.
    panel = _daily_panel(["A", "B"], close=999.0)
    bars = _bars(
        [(s, f"{d.date()} 14:51:00", 11.0, 0.0 if s == "A" else 100.0, 1_200.0)
         for d in (_JAN, _FEB, _MAR) for s in ("A", "B")]
    )
    model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=_VWAP)

    period = _period(model, _JAN)
    can_buy, can_sell = model.feasibility(period, ["A", "B"])
    assert can_buy["A"] is False and can_sell["A"] is False
    assert can_buy["B"] is True and can_sell["B"] is True
    # A is omitted from holding returns (settle treats it as flat) rather than
    # earning a close-priced return; B prices at its VWAP (12.0, flat over time).
    returns = model.holding_returns(period, ["A", "B"])
    assert "A" not in returns.index
    assert returns["B"] == pytest.approx(0.0)
    # neither the minute close (11.0) nor the daily close (999.0) is anywhere in
    # the execution price matrix.
    prices = model.execution_prices()
    assert pd.isna(prices.loc[_JAN, "A"])
    assert not prices.isin([11.0, 999.0]).any().any()
    assert prices["B"].tolist() == pytest.approx([12.0, 12.0, 12.0])
    assert {f.reason for f in model.blocked_fills()} == {REASON_MISSING_PRICE}


# --------------------------------------------------------------------------- #
# 4. I5b price-limit gate — ordering unchanged, RAW-vs-RAW on the executed price
# --------------------------------------------------------------------------- #
def _gate(bars, lim, cfg=_VWAP, **kwargs):
    """Build the gate on the DEFAULT limit_tolerance unless a test overrides it.

    Tests here deliberately do NOT pin the tolerance: widening the default must
    show up as a failure here (see the mutation evidence in the report).
    """
    panel = _daily_panel(["A"], close=50.0)
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=cfg, price_limits=lim,
        price_limit_check=True, require_price_limit_coverage=True, **kwargs,
    )
    return model, model.feasibility(_period(model, _JAN), ["A"])


# A limit-up execution minute has exactly two shapes and the gate must tell them
# apart. BOTH are tested: a one-sided test cannot distinguish a correct gate from
# one that simply blocks everything near a limit.
def test_locked_limit_up_minute_blocks_the_buy():
    """封死涨停: every print at the limit -> VWAP == limit -> no fill was available."""
    volume = 1_000.0
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 11.0, volume, 11.0 * volume)])
    lim = _limits([(_JAN, "A", 11.0, 5.0)])

    model, (can_buy, can_sell) = _gate(bars, lim)
    assert model.execution_prices().loc[_JAN, "A"] == pytest.approx(11.0)
    assert can_buy["A"] is False                 # blocked ...
    assert can_sell["A"] is True                 # ... and only the buy
    assert model.up_limit_blocked_buys() == 1    # existing I5b diagnostic
    assert model.opened_limit_up_minutes() == 0


def test_opened_limit_up_minute_allows_the_buy():
    """盘中打开: real prints below the limit -> a fill WAS achievable -> no block.

    Reproduced from real cached data — 601858.SH 2023-05-15 14:51: low 45.57,
    close == up_limit 46.13, amount/volume = 46.048583. The minute ended at the
    limit but traded through it on the way, so the buy must go through. This is a
    DELIBERATE divergence from a close-based gate, which would over-block it.
    """
    volume = 1_000_000.0
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 46.13, volume, 46.048583 * volume)])
    lim = _limits([(_JAN, "A", 46.13, 37.75)])

    model, (can_buy, can_sell) = _gate(bars, lim)
    assert model.execution_prices().loc[_JAN, "A"] == pytest.approx(46.048583)
    assert can_buy["A"] is True                  # NOT blocked — volume traded below
    assert can_sell["A"] is True
    assert model.up_limit_blocked_buys() == 0
    # the divergence from a close-based gate is counted, never silent
    assert model.opened_limit_up_minutes() == 1
    # and it hinges on a gap far above rounding scale (0.0814 RMB = 0.18%)
    assert 46.13 - 46.048583 > 0.08


def test_locked_limit_down_minute_blocks_the_sell():
    """Mirror: every print at the lower limit -> VWAP == limit -> no bid to hit."""
    volume = 1_000.0
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 5.0, volume, 5.0 * volume)])
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    model, (can_buy, can_sell) = _gate(bars, lim)
    assert can_sell["A"] is False and can_buy["A"] is True
    assert model.down_limit_blocked_sells() == 1
    assert model.opened_limit_down_minutes() == 0


def test_opened_limit_down_minute_allows_the_sell():
    """Mirror: prints above the lower limit -> a sell was achievable -> no block."""
    volume = 1_000.0
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 5.0, volume, 5.02 * volume)])
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    model, (can_buy, can_sell) = _gate(bars, lim)
    assert model.execution_prices().loc[_JAN, "A"] == pytest.approx(5.02)
    assert can_sell["A"] is True and can_buy["A"] is True
    assert model.down_limit_blocked_sells() == 0
    assert model.opened_limit_down_minutes() == 1


def test_tolerance_stays_at_rounding_scale_not_a_near_limit_band():
    """The calibration rule, as an executable fact.

    Real cached 14:51 bars closing at the up limit: the LOCKED ones sit at most
    0.0021% below the limit (`amount` rounding), the OPENED ones 0.013%-0.017%
    below — an order of magnitude apart. A rounding-scale band separates them; a
    "near-limit" band does not, it just re-blocks achievable fills.
    """
    volume = 3_000.0
    # locked, but `amount` rounded to the yuan -> vwap = 10.999666... (0.003% off)
    locked = _bars([("A", f"{_JAN.date()} 14:51:00", 11.0, volume, 32_999.0)])
    # opened: genuine prints below the limit -> vwap 10.94 (0.5% off)
    opened = _bars([("A", f"{_JAN.date()} 14:51:00", 11.0, volume, 10.94 * volume)])
    lim = _limits([(_JAN, "A", 11.0, 5.0)])

    # A rounding-scale band blocks the locked minute ...
    assert _gate(locked, lim, limit_tolerance=0.001)[1][0]["A"] is False
    # ... and still lets the opened one through.
    assert _gate(opened, lim, limit_tolerance=0.001)[1][0]["A"] is True
    # Widening to a "near limit" band destroys the distinction: the opened minute,
    # where volume demonstrably traded below the limit, gets blocked too.
    assert _gate(opened, lim, limit_tolerance=0.1)[1][0]["A"] is False

def test_missing_bar_still_blocks_before_any_limit_logic():
    panel = _daily_panel(["A"], close=50.0)
    bars = _bars([("A", f"{_FEB.date()} 14:51:00", 9.0, 100.0, 1_000.0)])  # no Jan bar
    lim = _limits([(_FEB, "A", 100.0, 1.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=_VWAP, price_limits=lim,
        price_limit_check=True, require_price_limit_coverage=True,
    )
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    assert can_buy["A"] is False and can_sell["A"] is False
    assert model.up_limit_blocked_buys() == 0    # the gate never ran
    assert model.down_limit_blocked_sells() == 0


def test_undefined_vwap_blocks_before_the_limit_gate_and_needs_no_limit_row():
    """An unpriceable bar cannot reach the gate, so it requires no coverage.

    Strict coverage therefore must NOT demand a limit row for it (that would be a
    spurious hard failure), and its block must stay a price block, not a limit one.
    """
    panel = _daily_panel(["A"], close=50.0)
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 9.0, 0.0, 1_000.0)])  # vwap undefined
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=_VWAP, price_limits=None,
        price_limit_check=True, require_price_limit_coverage=True,  # no raise
    )
    assert model.limit_coverage() == {"required": 0, "present": 0, "missing": 0}
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    assert can_buy["A"] is False and can_sell["A"] is False
    assert model.up_limit_blocked_buys() == 0
    assert model.missing_limit_rows() == 0  # never even consulted


def test_vwap_basis_diverges_from_close_basis_only_where_intended():
    """Enumerate BOTH directions of the behaviour change — nothing silent.

    Versus a close-based gate the VWAP gate is MORE permissive on opened limit
    minutes (by design: volume traded below the limit, so the fill was
    achievable) and STRICTER on unpriceable bars (undefined VWAP blocks both
    ways). Every other shape agrees. There is deliberately no one-sided
    "never more permissive" invariant here — that would be false, and asserting
    it would hide the very change the user asked for.
    """
    syms = ["LOCKED", "OPENED", "NORMAL", "NOVOL"]
    panel = _daily_panel(syms, close=50.0)
    specs = [
        ("LOCKED", 11.0, 1_000.0, 11_000.0),   # every print at the limit
        ("OPENED", 11.0, 1_000.0, 10_940.0),   # closed at the limit, traded below
        ("NORMAL", 9.0, 1_000.0, 9_400.0),     # nowhere near a limit
        ("NOVOL", 9.0, 0.0, 0.0),              # unpriceable on the VWAP basis
    ]
    bars = _bars([(s, f"{_JAN.date()} 14:51:00", c, v, a) for (s, c, v, a) in specs])
    lim = _limits([(_JAN, s, 11.0, 5.0) for s in syms])

    def _buys(cfg):
        model = IntradayTailEventModel(
            calendar_panel=panel, bars=bars, cfg=cfg, price_limits=lim,
            price_limit_check=True, require_price_limit_coverage=False,
        )
        return model.feasibility(_period(model, _JAN), syms)[0]

    vwap, close = _buys(_VWAP), _buys(_CLOSE)
    assert (close["OPENED"], vwap["OPENED"]) == (False, True)   # more permissive
    assert (close["NOVOL"], vwap["NOVOL"]) == (True, False)     # stricter
    assert vwap["LOCKED"] is False and close["LOCKED"] is False  # agree: blocked
    assert vwap["NORMAL"] is True and close["NORMAL"] is True    # agree: allowed


def test_limit_gate_ignores_the_daily_close_under_vwap():
    """Perturbing the daily close cannot move an intraday RAW-vs-RAW decision."""
    bars = _bars([("A", f"{_JAN.date()} 14:51:00", 9.0, 100.0, 1_000.0)])  # vwap 10.0
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    outcomes = []
    for daily_close in (50.0, 5_000.0):
        model = IntradayTailEventModel(
            calendar_panel=_daily_panel(["A"], close=daily_close), bars=bars,
            cfg=_VWAP, price_limits=lim, price_limit_check=True,
            require_price_limit_coverage=True,
        )
        outcomes.append(model.feasibility(_period(model, _JAN), ["A"]))
        assert model.execution_prices().loc[_JAN, "A"] == pytest.approx(10.0)
    assert outcomes[0] == outcomes[1]


# --------------------------------------------------------------------------- #
# 5. engine level — the basis moves the book, and only the basis
# --------------------------------------------------------------------------- #
def _engine(panel, bars, cfg):
    return BacktestEngine(
        model=IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=cfg),
        universe=StaticUniverse(["A", "B"]),
        scores=_scores({"A": 2.0, "B": 1.0}),
        constructor=TopNEqualWeight(1, long_only=True),
        execution=SimExecution(fee_rate=0.001),
        selection_panel=panel,
        initial_nav=1.0,
        cash_return=0.0,
    )


def test_engine_book_follows_the_vwap_not_the_close():
    panel = _daily_panel(["A", "B"], close=100.0)
    # A's closes are flat at 10.0 while its VWAP doubles from 8.0 to 16.0: the two
    # bases cannot be confused in the settled book.
    bars = _bars([
        ("A", f"{_JAN.date()} 14:51:00", 10.0, 100.0, 800.0),    # vwap 8.0
        ("A", f"{_FEB.date()} 14:51:00", 10.0, 100.0, 1_600.0),  # vwap 16.0
        ("A", f"{_MAR.date()} 14:51:00", 10.0, 100.0, 1_600.0),  # vwap 16.0
        ("B", f"{_JAN.date()} 14:51:00", 20.0, 100.0, 2_000.0),
        ("B", f"{_FEB.date()} 14:51:00", 20.0, 100.0, 2_000.0),
        ("B", f"{_MAR.date()} 14:51:00", 20.0, 100.0, 2_000.0),
    ])
    vwap_model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=_VWAP)
    close_model = IntradayTailEventModel(calendar_panel=panel, bars=bars, cfg=_CLOSE)
    jan = _period(vwap_model, _JAN)
    assert vwap_model.holding_returns(jan, ["A"])["A"] == pytest.approx(1.0)   # 16/8-1
    assert close_model.holding_returns(jan, ["A"])["A"] == pytest.approx(0.0)  # 10/10-1

    vwap_nav = _engine(panel, bars, _VWAP).run()
    close_nav = _engine(panel, bars, _CLOSE).run()
    assert not vwap_nav["nav"].equals(close_nav["nav"])
    assert float(vwap_nav["nav"].iloc[-1]) > float(close_nav["nav"].iloc[-1])


def test_bases_agree_exactly_when_amount_equals_close_times_volume():
    """The equivalence that keeps the I5a/I5b/I5d fixtures basis-neutral."""
    panel = _daily_panel(["A", "B"], close=100.0)
    specs = [
        ("A", _JAN, 10.0), ("A", _FEB, 12.0), ("A", _MAR, 9.0),
        ("B", _JAN, 20.0), ("B", _FEB, 21.0), ("B", _MAR, 19.0),
    ]
    bars = _bars(
        [(s, f"{d.date()} 14:51:00", c, 250.0, c * 250.0) for (s, d, c) in specs]
    )
    vwap_engine, close_engine = _engine(panel, bars, _VWAP), _engine(panel, bars, _CLOSE)
    assert_frame_equal(vwap_engine.run(), close_engine.run())
    assert_frame_equal(vwap_engine.holdings_log(), close_engine.holdings_log())
    assert_frame_equal(vwap_engine.feasibility_log(), close_engine.feasibility_log())


def test_undefined_vwap_does_not_silently_trade_at_the_close_in_the_engine():
    """A whole rebalance's names unpriceable -> no position, not a close-priced one."""
    panel = _daily_panel(["A", "B"], close=100.0)
    bars = _bars([
        # A's January execution minute printed no shares; February/March are fine.
        ("A", f"{_JAN.date()} 14:51:00", 10.0, 0.0, 0.0),
        ("A", f"{_FEB.date()} 14:51:00", 10.0, 100.0, 1_600.0),
        ("A", f"{_MAR.date()} 14:51:00", 10.0, 100.0, 1_600.0),
        ("B", f"{_JAN.date()} 14:51:00", 20.0, 100.0, 2_000.0),
        ("B", f"{_FEB.date()} 14:51:00", 20.0, 100.0, 2_000.0),
        ("B", f"{_MAR.date()} 14:51:00", 20.0, 100.0, 2_000.0),
    ])
    def _jan_weight_of_a(cfg) -> float:
        engine = _engine(panel, bars, cfg)
        nav = engine.run()
        assert np.isfinite(nav["nav"]).all()
        holdings = engine.holdings_log()
        jan = holdings[holdings["date"] == _JAN].set_index("symbol")["weight"]
        return float(jan.get("A", 0.0))

    # A is the top-scored name on both bases. On bar_close its January bar prices
    # fine (10.0), so it IS bought — which is exactly the fill a fallback would
    # have manufactured. On bar_vwap the same bar is unpriceable, so A is simply
    # not bought: the block is real, not a re-priced trade.
    assert _jan_weight_of_a(_CLOSE) == pytest.approx(1.0)
    assert _jan_weight_of_a(_VWAP) == pytest.approx(0.0)
