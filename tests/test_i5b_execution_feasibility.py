"""I5b: execution-time price-limit feasibility for the intraday tail model.

Covers the goal's four test groups:

1. Unit — direction-aware limit feasibility: an execution-minute raw close at the
   raw upper limit blocks a BUY but allows a SELL; at the raw lower limit blocks a
   SELL but allows a BUY; a missing execution bar blocks BOTH before limit logic;
   strict missing-coverage fails at construction; lenient missing-coverage is
   counted (never a silent passed check); the check disabled == I5a feasibility.
2. Integration (through BacktestEngine) — a blocked buy at the up-limit leaves cash
   / does not add the name; a blocked sell at the down-limit carries the old
   achieved position; turnover/holdings are the ACHIEVED book; perturbing the daily
   close does not change intraday limit feasibility when the raw minute close and
   raw stk_limit are unchanged; non-binding limits reproduce the I5a NAV.
3. Config — existing configs validate unchanged; negative tolerance fails; the I5b
   config enables the check; price_limit_check=false reproduces I5a feasibility.

The comparison is RAW-vs-RAW only — the price that EXECUTES (the bar VWAP under
the default basis, raw amount over raw volume) vs raw stk_limit, never qfq / the
daily close. The daily panel here carries deliberately CONTRADICTORY closes to
prove the limit gate ignores them.

Note for future edits: these fixtures price each execution bar so its VWAP equals
its close, which keeps the pre-PR#75 assertions valid on both bases but makes
them BLIND to which of the two the gate reads. The tests at the bottom of this
module cover that distinction directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from data.clean.intraday_schema import normalize_intraday_bars
from data.clean.schema import normalize_panel
from qt.config import RootConfig, load_config
from qt.pipeline import _FrameScores
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.backtest.sim_execution import SimExecution
from runtime.intraday_execution import IntradayExecutionConfig
from portfolio.construct import TopNEqualWeight
from universe.static import StaticUniverse

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_I5A_CONFIG = _CONFIG_DIR / "phase_i5a_intraday_tail_framework.yaml"
_I5B_CONFIG = _CONFIG_DIR / "phase_i5b_intraday_execution_feasibility.yaml"
_EXAMPLE_CONFIG = _CONFIG_DIR / "example.yaml"

_DATES = [
    "2024-01-30", "2024-01-31",
    "2024-02-28", "2024-02-29",
    "2024-03-28", "2024-03-29",
]
_JAN, _FEB, _MAR = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _daily_panel(closes: dict) -> pd.DataFrame:
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
    closes = {(d, s): price_fn(d, s) for d in _DATES for s in symbols}
    return _daily_panel(closes)


def _scores(panel_map: dict) -> _FrameScores:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for (d, s) in panel_map], names=["date", "symbol"]
    )
    return _FrameScores(pd.Series(list(panel_map.values()), index=idx, name="score"))


def _minute_bars(specs: list[tuple[str, str, float]]) -> pd.DataFrame:
    rows = [
        {
            "time": pd.Timestamp(t), "symbol": s,
            "open": c, "high": c, "low": c, "close": c,
            "volume": 1.0, "amount": float(c), "source_trade_time": t,
        }
        for (s, t, c) in specs
    ]
    return normalize_intraday_bars(pd.DataFrame(rows), freq="1min", data_lag="1min")


def _limits(rows: list[tuple]) -> pd.DataFrame:
    """Raw stk_limit frame from ``(date, symbol, up_limit, down_limit)`` rows."""
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "symbol": s, "up_limit": up, "down_limit": dn}
            for (d, s, up, dn) in rows
        ]
    )


def _period(model: IntradayTailEventModel, date: pd.Timestamp):
    return next(p for p in model.holding_periods() if p.date == date)


# --------------------------------------------------------------------------- #
# 1. Unit — direction-aware limit feasibility
# --------------------------------------------------------------------------- #
def test_i5b_up_limit_blocks_buy_allows_sell():
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])  # exec = 10.0
    # exec 10.0 sits AT the up-limit (10.0); down-limit 5.0 is far below.
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    assert can_buy["A"] is False   # limit-up blocks the BUY
    assert can_sell["A"] is True   # ... but the SELL still executes (bar exists)
    assert model.up_limit_blocked_buys() == 1
    assert model.down_limit_blocked_sells() == 0


def test_i5b_down_limit_blocks_sell_allows_buy():
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 5.0)])  # exec = 5.0
    # exec 5.0 sits AT the down-limit (5.0); up-limit 10.0 is far above.
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    assert can_sell["A"] is False  # limit-down blocks the SELL
    assert can_buy["A"] is True    # ... but the BUY still executes (bar exists)
    assert model.down_limit_blocked_sells() == 1
    assert model.up_limit_blocked_buys() == 0


def test_i5b_missing_exec_bar_blocks_both_before_limit():
    # A has NO execution bar at _JAN; a limit row exists but must never be consulted.
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_FEB.date()} 14:51:00", 10.0)])  # only Feb has a bar
    lim = _limits([(_FEB, "A", 100.0, 1.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    assert can_buy["A"] is False and can_sell["A"] is False  # missing bar -> both
    # the limit gate never ran for the missing-bar name (no reason-attributed block)
    assert model.up_limit_blocked_buys() == 0
    assert model.down_limit_blocked_sells() == 0


def test_i5b_strict_missing_limit_coverage_raises_at_construction():
    # A has a bar at the _JAN rebalance anchor -> a limit row is REQUIRED, but none
    # is supplied. Strict mode must fail BEFORE any result is emitted.
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])
    with pytest.raises(ValueError, match="require_price_limit_coverage"):
        IntradayTailEventModel(
            calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
            price_limits=None, price_limit_check=True,
            require_price_limit_coverage=True,
        )


def test_i5b_lenient_missing_limit_counts_not_silent_pass():
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=None, price_limit_check=True,
        require_price_limit_coverage=False,  # lenient: no raise
    )
    can_buy, can_sell = model.feasibility(_period(model, _JAN), ["A"])
    # falls back to the bar-exists rule (bar present) ...
    assert can_buy["A"] is True and can_sell["A"] is True
    # ... but the unchecked limit is disclosed, NOT silently claimed as passed.
    assert model.missing_limit_rows() == 1
    assert model.limit_coverage()["missing"] >= 1


def test_i5b_check_disabled_reproduces_bar_exists_rule():
    # A limit that WOULD block the buy is supplied, but price_limit_check=false.
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])
    lim = _limits([(_JAN, "A", 10.0, 5.0)])  # exec == up-limit
    off = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=False,
    )
    bare = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
    )  # exactly the I5a construction
    p = _period(off, _JAN)
    assert off.feasibility(p, ["A"]) == bare.feasibility(p, ["A"])
    assert off.feasibility(p, ["A"])[0]["A"] is True  # not blocked when check off
    assert off.up_limit_blocked_buys() == 0


def test_i5b_feasibility_idempotent_over_repeated_calls():
    # Reason-attributed counts must not double when feasibility() is called twice
    # (e.g. a second engine.run over the same model).
    panel = _grid_panel(["A"], lambda d, s: 50.0)
    bars = _minute_bars([("A", f"{_JAN.date()} 14:51:00", 10.0)])
    lim = _limits([(_JAN, "A", 10.0, 5.0)])
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    p = _period(model, _JAN)
    model.feasibility(p, ["A"])
    model.feasibility(p, ["A"])
    assert model.up_limit_blocked_buys() == 1  # not 2


# --------------------------------------------------------------------------- #
# 2. Integration through BacktestEngine
# --------------------------------------------------------------------------- #
def test_i5b_blocked_buy_at_up_limit_leaves_cash():
    panel = _grid_panel(["A", "B"], lambda d, s: 50.0)
    bars = _minute_bars(
        [(s, f"{d.date()} 14:51:00", {"A": 10.0, "B": 20.0}[s])
         for d in (_JAN, _FEB, _MAR) for s in ("A", "B")]
    )
    # A is pinned at its up-limit ONLY on _JAN (up == exec 10.0); elsewhere far away.
    lim = _limits(
        [(_JAN, "A", 10.0, 1.0), (_FEB, "A", 100.0, 1.0),
         (_JAN, "B", 200.0, 2.0), (_FEB, "B", 200.0, 2.0)]
    )
    scores = _scores({(d, s): {"A": 3.0, "B": 2.0}[s] for d in _DATES for s in "AB"})
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    engine = BacktestEngine(
        model=model, universe=StaticUniverse(["A", "B"]), scores=scores,
        constructor=TopNEqualWeight(2), execution=SimExecution(fee_rate=0.0),
        selection_panel=panel,
    )
    engine.run()
    holdings = engine.holdings_log()
    jan_names = set(holdings[holdings["date"] == _JAN]["symbol"])
    feb_names = set(holdings[holdings["date"] == _FEB]["symbol"])
    assert "A" not in jan_names and "B" in jan_names   # up-limit blocked A's buy
    assert "A" in feb_names                            # A enters once the limit clears
    assert model.up_limit_blocked_buys() >= 1
    # invested at _JAN is only B's half-weight; the rest is cash (no leverage).
    feas = engine.feasibility_log()
    assert feas.loc[_JAN, "invested"] == pytest.approx(0.5)


def test_i5b_blocked_sell_at_down_limit_carries_position():
    panel = _grid_panel(["A", "B"], lambda d, s: 50.0)
    bars = _minute_bars(
        [(s, f"{d.date()} 14:51:00", {"A": 10.0, "B": 20.0}[s])
         for d in (_JAN, _FEB, _MAR) for s in ("A", "B")]
    )
    # A pinned at its DOWN-limit on _FEB (down == exec 10.0) -> cannot be sold.
    lim = _limits(
        [(_JAN, "A", 100.0, 1.0), (_FEB, "A", 100.0, 10.0),
         (_JAN, "B", 200.0, 2.0), (_FEB, "B", 200.0, 2.0)]
    )
    # _JAN: A scores higher (held); _FEB: B scores higher (would exit A).
    scores = _scores(
        {(d, s): ({"A": 2.0, "B": 1.0} if d in ("2024-01-30", "2024-01-31")
                  else {"A": 1.0, "B": 2.0})[s]
         for d in _DATES for s in "AB"}
    )
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
    )
    engine = BacktestEngine(
        model=model, universe=StaticUniverse(["A", "B"]), scores=scores,
        constructor=TopNEqualWeight(1), execution=SimExecution(fee_rate=0.0),
        selection_panel=panel,
    )
    engine.run()
    holdings = engine.holdings_log()
    feb_names = set(holdings[holdings["date"] == _FEB]["symbol"])
    assert "A" in feb_names    # down-limit blocked the sell -> A is carried
    assert "B" not in feb_names  # no cash freed -> B's buy could not fund
    assert model.down_limit_blocked_sells() >= 1


def test_i5b_daily_close_perturbation_does_not_change_feasibility():
    # Two daily panels differing ONLY in close; identical minute bars + raw limits.
    bars = _minute_bars(
        [(s, f"{d.date()} 14:51:00", {"A": 10.0, "B": 20.0}[s])
         for d in (_JAN, _FEB, _MAR) for s in ("A", "B")]
    )
    lim = _limits(
        [(d, s, 999.0, 0.01) for d in (_JAN, _FEB) for s in ("A", "B")]
    )
    scores = _scores({(d, s): {"A": 3.0, "B": 2.0}[s] for d in _DATES for s in "AB"})

    def _run(close_fn):
        panel = _grid_panel(["A", "B"], close_fn)
        model = IntradayTailEventModel(
            calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
            price_limits=lim, price_limit_check=True, require_price_limit_coverage=True,
        )
        engine = BacktestEngine(
            model=model, universe=StaticUniverse(["A", "B"]), scores=scores,
            constructor=TopNEqualWeight(2), execution=SimExecution(fee_rate=0.0),
            selection_panel=panel,
        )
        nav = engine.run()
        return nav, engine.holdings_log()

    nav_a, hold_a = _run(lambda d, s: 50.0)
    nav_b, hold_b = _run(lambda d, s: 5.0 if s == "A" else 500.0)  # wildly different closes
    assert_frame_equal(nav_a, nav_b)
    assert_frame_equal(hold_a, hold_b)


def test_i5b_non_binding_limits_match_check_off_nav():
    # With limits that never bind, enabling the check must reproduce the I5a NAV.
    panel = _grid_panel(["A", "B"], lambda d, s: 50.0)
    bars = _minute_bars(
        [(s, f"{d.date()} 14:51:00", {"A": 10.0 + i, "B": 20.0 + i}[s])
         for i, d in enumerate((_JAN, _FEB, _MAR)) for s in ("A", "B")]
    )
    lim = _limits(
        [(d, s, 1.0e9, 0.0) for d in (_JAN, _FEB) for s in ("A", "B")]  # never binds
    )
    scores = _scores({(d, s): {"A": 3.0, "B": 2.0}[s] for d in _DATES for s in "AB"})

    def _run(check, limits):
        model = IntradayTailEventModel(
            calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
            price_limits=limits, price_limit_check=check,
            require_price_limit_coverage=True,
        )
        engine = BacktestEngine(
            model=model, universe=StaticUniverse(["A", "B"]), scores=scores,
            constructor=TopNEqualWeight(2), execution=SimExecution(fee_rate=0.001),
            selection_panel=panel,
        )
        return engine.run(), model

    nav_off, off = _run(False, None)
    nav_on, on = _run(True, lim)
    assert_frame_equal(nav_off, nav_on)
    assert on.up_limit_blocked_buys() == 0
    assert on.down_limit_blocked_sells() == 0


# --------------------------------------------------------------------------- #
# 3. Config
# --------------------------------------------------------------------------- #
def test_i5b_existing_configs_validate_unchanged():
    i5a = load_config(str(_I5A_CONFIG))
    assert i5a.intraday is not None
    # the I5a config never set the I5b knobs -> safe defaults (check OFF).
    assert i5a.intraday.price_limit_check is False
    assert i5a.intraday.require_price_limit_coverage is True
    assert i5a.intraday.limit_tolerance == pytest.approx(1e-6)
    example = load_config(str(_EXAMPLE_CONFIG))  # a daily config still validates
    assert example.intraday is None


def test_i5b_config_validates_and_enables_check():
    cfg = load_config(str(_I5B_CONFIG))
    assert cfg.backtest.event_order == "intraday_tail_rebalance"
    assert cfg.intraday is not None
    assert cfg.intraday.price_limit_check is True
    assert cfg.intraday.require_price_limit_coverage is True
    assert cfg.intraday.limit_tolerance == pytest.approx(1e-6)
    assert cfg.output.intraday_report_name == "phase_i5b_intraday_execution_feasibility"


def test_i5b_negative_tolerance_fails_readably():
    import yaml
    d = yaml.safe_load(_I5B_CONFIG.read_text())
    d["intraday"]["limit_tolerance"] = -1.0
    with pytest.raises(ValidationError, match="limit_tolerance"):
        RootConfig(**d)


def test_i5b_report_name_default_keeps_i5a_basename():
    # An intraday config without the knob keeps the historical I5a report basename.
    from qt.intraday_tail_framework import _report_basename
    assert _report_basename(load_config(str(_I5A_CONFIG))) == "phase_i5a_intraday_tail_framework"
    assert _report_basename(load_config(str(_I5B_CONFIG))) == "phase_i5b_intraday_execution_feasibility"


def test_i5b_report_heading_names_the_actual_study():
    # The report H1 must not stay frozen at the I5a label when the limit check is on.
    from qt.intraday_tail_framework import _report_heading
    i5a_title, _ = _report_heading("ret", False)
    i5b_title, i5b_intro = _report_heading("ret", True)
    assert i5a_title.startswith("# Phase I5a")
    assert i5b_title.startswith("# Phase I5b")
    assert "execution-feasibility hardening" in i5b_intro.lower()


# --- the report must state the gate input it ACTUALLY used (PR #75 follow-up) --- #
#
# PR #75 moved the limit gate from the bar close to the executed price (the bar
# VWAP by default) but left both report writers still claiming "the selected
# execution-minute raw 1min close". The run therefore SHIPPED a report that
# misdescribed the check it had performed. These pin the corrected wording; the
# mutation evidence for them is in the PR body.


def test_limit_basis_prose_names_the_active_execution_basis():
    from qt.intraday_tail_framework import limit_basis_lines

    for basis in ("bar_vwap", "bar_close"):
        text = " ".join(limit_basis_lines(basis))
        assert f"`{basis}`" in text, f"the prose must name the active basis {basis!r}"
        # The defect being locked out: asserting a close comparison regardless of basis.
        assert "1min close to the raw" not in text
        assert "1min close vs the" not in text


def test_limit_basis_prose_does_not_claim_a_close_comparison_under_vwap():
    from qt.intraday_tail_framework import limit_basis_lines

    text = " ".join(limit_basis_lines("bar_vwap")).lower()
    assert "executes" in text
    # It must explain the LOCKED/OPENED distinction, which is the whole reason the
    # executed price rather than the close is the faithful input.
    assert "locked" in text and "opened" in text
    # And it must still assert the raw-vs-raw property, which the move to a VWAP
    # does not break (a bar VWAP is raw amount over raw volume).
    assert "raw-vs-raw" in text


def test_group_report_feasibility_prose_names_the_active_execution_basis():
    from types import SimpleNamespace

    from qt.intraday_group_report import _feasibility_lines

    cfg = load_config(str(_I5B_CONFIG))
    groups = (
        SimpleNamespace(
            group=1,
            up_limit_blocked_buys=2,
            down_limit_blocked_sells=0,
            missing_limit_rows=0,
            opened_limit_up_minutes=3,
            opened_limit_down_minutes=1,
            missing_adj_factor_pairs=4,
        ),
    )
    result = SimpleNamespace(
        config=cfg,
        price_limit_check=True,
        limit_coverage={"required": 10, "present": 10, "missing": 0},
        stk_limit_gap_fetches=0,
        groups=groups,
    )
    text = " ".join(_feasibility_lines(result))
    assert f"`{cfg.intraday.execution_price_basis}`" in text
    assert "1min close to the raw" not in text
    # The divergence counters and the dropped-return count must reach the reader,
    # not merely exist on the result object.
    assert "opened limit-up" in text
    assert "| 3 | 1 | 4 |" in text
    assert "adj_factor" in text


def test_no_file_anywhere_claims_the_gate_compares_a_close():
    """Repo-wide: NO file may state that the I5b gate compares a close.

    The first version of this scanned only the two report modules. Review then
    found SIX more live instances it could not reach even in principle -- one in
    `qt/config.py` and five in the I5b/c/d/e/f config YAMLs, which are exactly what
    an operator reads before enabling the feature, a higher-stakes readership than
    a docstring. Scoping a guard to the files you already fixed guarantees it only
    ever confirms what you already know.

    Two restatements survive by necessity and are covered by this scan rather than
    by composition: `runtime/backtest/event_models.py` and `qt/config.py` both sit
    UPSTREAM of `qt.intraday_tail_framework` in the import graph (it imports them),
    so neither can call `limit_basis_phrase` without a cycle.

    Deliberately NOT matched: `bar_close` the basis name, "that bar's single
    closing tick" the basis description, and the DAILY universe tradability filter
    (`universe/filters.py`, `data/clean/tradability.py`), which is a separate and
    older mechanism that genuinely does compare a raw daily close -- see
    RUNBOOK.md. Those are correct and must not be swept up.

    Scope, stated honestly: this catches a claim written with the word "close"
    near "execution minute". It does NOT catch a reworded synonym ("its last
    print", "the final tick") -- review demonstrated seven such escapes. No lexical
    guard can assert "no sentence anywhere makes this claim". Composition is the
    real answer where the import graph allows it; this scan is the net under the
    places where it does not.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    forbidden = re.compile(
        r"execution[- ]minute[^.\n]{0,40}\bclose\b"
        r"|\b1min close\b"
        r"|raw\s+close\s+to\s+(the\s+)?raw\s+`?`?stk_limit",
        re.IGNORECASE,
    )
    targets = [root / "qt" / "config.py", *sorted((root / "config").glob("*.yaml"))]
    targets += sorted((root / "qt").glob("intraday*.py"))
    targets += [root / "runtime" / "backtest" / "event_models.py",
                root / "runtime" / "intraday_execution.py"]

    hits = []
    for f in targets:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not hits, (
        "these files claim the I5b price-limit gate compares a CLOSE. It compares "
        "the price that EXECUTES (the bar VWAP under the default basis). Where the "
        "import graph allows it, compose limit_basis_phrase; where it does not, "
        "point at it:\n  " + "\n  ".join(hits)
    )


def test_every_gate_description_composes_the_single_phrase():
    """All three descriptions of the gate compose one phrase; none re-authors it.

    This is the guard the regex above cannot be: a lexical scan enumerates
    wordings, whereas having exactly one authored sentence leaves nothing to
    reword. If a future edit writes its own sentence instead of composing
    `limit_basis_phrase`, that sentence stops containing the phrase and this
    fails.
    """
    from types import SimpleNamespace

    from qt.intraday_group_report import _feasibility_lines
    from qt.intraday_tail_framework import limit_basis_lines, limit_basis_phrase

    cfg = load_config(str(_I5B_CONFIG))
    basis = cfg.intraday.execution_price_basis
    phrase = limit_basis_phrase(basis)
    assert f"`{basis}`" in phrase  # the phrase is basis-derived, not a constant

    # 1. the tail report's feasibility section
    assert phrase in " ".join(limit_basis_lines(basis))

    # 2. the group report's feasibility section
    result = SimpleNamespace(
        config=cfg,
        price_limit_check=True,
        limit_coverage={"required": 1, "present": 1, "missing": 0},
        stk_limit_gap_fetches=0,
        groups=(
            SimpleNamespace(
                group=1, up_limit_blocked_buys=0, down_limit_blocked_sells=0,
                missing_limit_rows=0, opened_limit_up_minutes=0,
                opened_limit_down_minutes=0, missing_adj_factor_pairs=0,
            ),
        ),
    )
    assert phrase in " ".join(_feasibility_lines(result))

    # 3. the tail report's Limitations bullet -- the one the first pass missed.
    # Read from source rather than rendering, since building a full I5aResult
    # here would test the dataclass, not the sentence.
    import inspect

    import qt.intraday_tail_framework as tf

    src = inspect.getsource(tf._write_report)
    assert "limit_basis_phrase(ec.execution_price_basis)" in src, (
        "the Limitations bullet must COMPOSE limit_basis_phrase, not restate the "
        "comparison in its own words"
    )


def test_composed_gate_sentences_are_grammatical_in_every_context():
    """The phrase is an appositive: every host sentence must close its em-dash.

    Composing one fragment into several sentences trades three ways to state a
    fact wrong for one way to punctuate it wrong. The first version of this
    refactor did exactly that -- it dropped the CLOSING em-dash in two of the
    three hosts, leaving "...on the `bar_vwap` basis to the raw stk_limit band"
    with an unclosed dash. Rendered output was NOT byte-identical to the previous
    release, contrary to what the change claimed, and a rendering probe caught it.

    So: wherever the phrase is embedded mid-sentence, the em-dash that opens the
    appositive inside it must be matched by one after it.
    """
    from types import SimpleNamespace

    from qt.intraday_group_report import _feasibility_lines
    from qt.intraday_tail_framework import limit_basis_phrase

    cfg = load_config(str(_I5B_CONFIG))
    phrase = limit_basis_phrase(cfg.intraday.execution_price_basis)
    assert "—" in phrase, "the phrase itself opens an appositive with an em-dash"

    result = SimpleNamespace(
        config=cfg, price_limit_check=True,
        limit_coverage={"required": 1, "present": 1, "missing": 0},
        stk_limit_gap_fetches=0,
        groups=(SimpleNamespace(
            group=1, up_limit_blocked_buys=0, down_limit_blocked_sells=0,
            missing_limit_rows=0, opened_limit_up_minutes=0,
            opened_limit_down_minutes=0, missing_adj_factor_pairs=0),),
    )
    for line in _feasibility_lines(result):
        if phrase in line:
            tail = line.split(phrase, 1)[1]
            assert tail.lstrip().startswith("—"), (
                f"the phrase is embedded mid-sentence but its appositive is never "
                f"closed; the text reads '...{phrase[-30:]}{tail[:40]}'"
            )
            break
    else:
        raise AssertionError("the group report no longer composes the phrase at all")
