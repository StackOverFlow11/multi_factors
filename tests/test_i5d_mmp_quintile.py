"""I5d: MMP quintile grouped intraday-tail backtest.

Covers the goal's five test groups:

1. Bucket assignment — deterministic 5 equal-count rank buckets, Q1 lowest / Q5
   highest, every finite name in exactly one group, NaN/non-finite excluded, ties
   deterministic, too-few-names leaves high groups empty (no crash).
2. Group constructor / backtest integration — ``EqualWeightAll`` ignores ``top_n``
   and equal-weights the whole bucket; ``GroupScores`` exposes one group; a fresh
   execution/model per group (no cross-group state); ``fee_rate`` reaches
   ``SimExecution`` (a nonzero-turnover period has positive cost).
3. PIT / MMP invariants — grouping uses only the PIT MMP score (available_time <=
   14:50); perturbing a post-cutoff bar cannot change the assignment; future
   (exit) bars never enter the assignment.
4. Execution feasibility — raw ``stk_limit`` blocking stays active in grouped runs;
   a missing execution bar blocks (never daily-close fallback); per-group blocked
   counts do not double-count across groups.
5. Report / figures — H1 names I5d (not a stale I5c/I5b label), mentions
   ``fee_rate=0.001`` and ``analytics.quantiles=5``; the three required PNGs are
   written and non-empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from data.clean.intraday_aggregate import asof_daily_features
from data.clean.schema import normalize_panel
from qt.config import load_config
from qt.intraday_group_backtest import (
    GroupRunResult,
    I5dResult,
    _build_group_assignment,
    _group_metrics,
    _max_drawdown,
)
from qt.intraday_group_report import _write_figures, _write_report
from qt.intraday_groups import EqualWeightAll, GroupScores, assign_quantile_buckets
from runtime.backtest.engine import BacktestEngine
from runtime.backtest.event_models import IntradayTailEventModel
from runtime.intraday_execution import REASON_NO_BAR, IntradayExecutionConfig
from runtime.backtest.sim_execution import SimExecution
from universe.static import StaticUniverse

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "phase_i5d_mmp_quintile_5y.yaml"

# Consecutive month-ends -> >=2 settled monthly periods.
_DATES = ["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"]
_JAN, _FEB = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _daily_panel(closes: dict) -> pd.DataFrame:
    rows = [
        {"date": pd.Timestamp(d), "symbol": s, "open": c, "high": c, "low": c,
         "close": c, "volume": 1.0, "amount": 1.0, "adj_factor": 1.0}
        for (d, s), c in closes.items()
    ]
    return normalize_panel(pd.DataFrame(rows))


def _grid_panel(symbols: list[str]) -> pd.DataFrame:
    return _daily_panel({(d, s): 100.0 for d in _DATES for s in symbols})


def _bars(rows: list[dict]) -> pd.DataFrame:
    return normalize_intraday_bars(pd.DataFrame(rows), freq="1min", data_lag="1min")


def _exec_rows(specs: list[tuple[str, str, float]]) -> list[dict]:
    """Single-bar specs ``(symbol, 'YYYY-MM-DD HH:MM:SS', close)`` -> bar rows."""
    return [
        {"time": pd.Timestamp(t), "symbol": s, "open": c, "high": c, "low": c,
         "close": c, "volume": 1.0, "amount": float(c), "source_trade_time": t}
        for (s, t, c) in specs
    ]


def _session_rows(symbol: str, date: str, *, n: int = 24, slope: float) -> list[dict]:
    """``n`` in-session 1min bars (from 09:30) with distinct OHLCV -> finite MMP.

    A non-zero ``slope`` makes each symbol's MMP distinct so the rank buckets
    differ. ``n=24`` > the 20-bar MMP baseline, so a few valid ``MMP_t`` exist.
    """
    day = pd.Timestamp(date)
    rows = []
    for i in range(n):
        t = day + pd.Timedelta("09:30:00") + pd.Timedelta(minutes=i)
        c = 10.0 + slope * i + 0.01 * ((i * 7) % 5)
        rows.append({
            "time": t, "symbol": symbol,
            "open": c - 0.02, "high": c + 0.05 + 0.01 * (i % 3),
            "low": c - 0.05 - 0.01 * (i % 2), "close": c,
            "volume": 1000.0 + 50.0 * ((i * 3) % 7),
            "amount": c * (1000.0 + 50.0 * ((i * 3) % 7)),
            "source_trade_time": str(t),
        })
    return rows


def _limits(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "up_limit": up, "down_limit": dn}
         for (d, s, up, dn) in rows]
    )


# --------------------------------------------------------------------------- #
# 1. Bucket assignment
# --------------------------------------------------------------------------- #
def test_buckets_q1_lowest_qn_highest():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})
    out = assign_quantile_buckets(scores, 5)
    assert out == {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}  # Q1 lowest, Q5 highest


def test_buckets_equal_count():
    scores = pd.Series({chr(ord("a") + i): float(i) for i in range(10)})
    out = assign_quantile_buckets(scores, 5)
    sizes = [sum(1 for v in out.values() if v == g) for g in range(1, 6)]
    assert sizes == [2, 2, 2, 2, 2]
    # lowest two scores are group 1, highest two are group 5.
    assert out["a"] == 1 and out["b"] == 1
    assert out["i"] == 5 and out["j"] == 5


def test_buckets_all_finite_assigned_exactly_once():
    scores = pd.Series({s: float(ord(s)) for s in "ABCDEFG"})
    out = assign_quantile_buckets(scores, 5)
    assert set(out) == set("ABCDEFG")  # every finite name assigned
    assert all(1 <= g <= 5 for g in out.values())


def test_buckets_drop_nan_and_inf():
    scores = pd.Series({"A": 1.0, "B": float("nan"), "C": float("inf"),
                        "D": -float("inf"), "E": 2.0})
    out = assign_quantile_buckets(scores, 5)
    assert set(out) == {"A", "E"}  # NaN / +-inf excluded


def test_buckets_ties_deterministic_by_symbol():
    scores = pd.Series({"B": 1.0, "A": 1.0, "D": 1.0, "C": 1.0})
    first = assign_quantile_buckets(scores, 2)
    second = assign_quantile_buckets(scores, 2)
    assert first == second  # deterministic
    # all tied -> symbol tie-break (A,B | C,D) splits the lower half to Q1.
    assert first["A"] == 1 and first["B"] == 1
    assert first["C"] == 2 and first["D"] == 2


def test_buckets_too_few_names_leaves_high_groups_empty():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    out = assign_quantile_buckets(scores, 5)
    sizes = [sum(1 for v in out.values() if v == g) for g in range(1, 6)]
    assert sizes == [1, 1, 1, 0, 0]  # no crash; Q4/Q5 empty


def test_buckets_empty_input():
    assert assign_quantile_buckets(pd.Series(dtype=float), 5) == {}


# --------------------------------------------------------------------------- #
# 2. Group constructor / scores
# --------------------------------------------------------------------------- #
def test_equal_weight_all_ignores_top_n_and_equal_weights():
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    w = EqualWeightAll().build(scores)
    assert len(w) == 4  # NOT capped at any top_n
    assert pytest.approx(w.sum()) == 1.0
    assert (w == 0.25).all()


def test_equal_weight_all_drops_nan_and_empty():
    w = EqualWeightAll().build(pd.Series({"A": 1.0, "B": float("nan")}))
    assert list(w.index) == ["A"] and pytest.approx(w.sum()) == 1.0
    assert EqualWeightAll().build(pd.Series(dtype=float)).empty


def test_group_scores_selects_only_group_members():
    assignments = {_JAN: {"A": 1, "B": 2, "C": 1}}
    s = GroupScores(assignments, 1).get(_JAN, ["A", "B", "C", "D"])
    assert s["A"] == 1.0 and s["C"] == 1.0  # group-1 members
    assert pd.isna(s["B"]) and pd.isna(s["D"])  # non-members -> NaN


def test_fee_rate_reaches_simexecution_positive_cost():
    panel = _grid_panel(["A", "B"])
    bars = _bars(_exec_rows([
        ("A", f"{_JAN.date()} 14:51:00", 10.0), ("B", f"{_JAN.date()} 14:51:00", 10.0),
        ("A", f"{_FEB.date()} 14:51:00", 11.0), ("B", f"{_FEB.date()} 14:51:00", 9.0),
    ]))
    assignments = {_JAN: {"A": 1}, _FEB: {"A": 1}}
    model = IntradayTailEventModel(calendar_panel=panel, bars=bars,
                                   cfg=IntradayExecutionConfig())
    engine = BacktestEngine(
        model=model, universe=StaticUniverse(["A", "B"]),
        scores=GroupScores(assignments, 1), constructor=EqualWeightAll(),
        execution=SimExecution(fee_rate=0.001), selection_panel=panel,
    )
    nav = engine.run()
    # first period buys A (turnover 1.0) -> cost = 1.0 * 0.001 > 0.
    assert nav["turnover"].iloc[0] > 0
    assert nav["cost"].iloc[0] == pytest.approx(nav["turnover"].iloc[0] * 0.001)
    assert nav["cost"].iloc[0] > 0


# --------------------------------------------------------------------------- #
# 3. PIT / MMP invariants (grouping is a function of the PIT score only)
# --------------------------------------------------------------------------- #
def _mmp_assignment(extra_rows: list[dict] | None = None) -> dict:
    """Build the MMP score on _JAN for A/B/C and group it into 3 buckets."""
    rows: list[dict] = []
    for sym, slope in (("A", 0.00), ("B", 0.02), ("C", -0.02)):
        rows += _session_rows(sym, str(_JAN.date()), slope=slope)
    if extra_rows:
        rows += extra_rows
    bars = _bars(rows)
    score = asof_daily_features(
        bars, decision_time="14:50:00", session_open="09:30:00", features=["mmp_ew"]
    ).iloc[:, 0].rename("score")
    panel = _daily_panel({(str(_JAN.date()), s): 10.0 for s in "ABC"})
    assignment = _build_group_assignment(
        score, StaticUniverse(["A", "B", "C"]), panel, [_JAN], {"A", "B", "C"}, 3
    )
    return assignment.by_date[_JAN]


def test_grouping_uses_pit_score_only():
    base = _mmp_assignment()
    # every scored name assigned to exactly one of 3 groups.
    assert set(base) == {"A", "B", "C"}
    assert sorted(base.values()) == [1, 2, 3]


def test_post_cutoff_bar_cannot_change_grouping():
    base = _mmp_assignment()
    # a post-14:50 bar with an extreme price must NOT change the assignment.
    poison = _exec_rows([("A", f"{_JAN.date()} 14:55:00", 9999.0)])
    perturbed = _mmp_assignment(poison)
    assert perturbed == base


def test_future_exit_bar_cannot_change_grouping():
    base = _mmp_assignment()
    # an exit-date (future) execution bar is irrelevant to the decision-date score.
    future = _exec_rows([("A", f"{_FEB.date()} 14:51:00", 0.01)])
    perturbed = _mmp_assignment(future)
    assert perturbed == base


# --------------------------------------------------------------------------- #
# 4. Execution feasibility in grouped runs
# --------------------------------------------------------------------------- #
def _grouped_engine(group, assignments, bars, panel, limits=None):
    model = IntradayTailEventModel(
        calendar_panel=panel, bars=bars, cfg=IntradayExecutionConfig(),
        price_limits=limits, price_limit_check=limits is not None,
        require_price_limit_coverage=False,
    )
    engine = BacktestEngine(
        model=model, universe=StaticUniverse(sorted({s for a in assignments.values() for s in a})),
        scores=GroupScores(assignments, group), constructor=EqualWeightAll(),
        execution=SimExecution(fee_rate=0.001), selection_panel=panel,
    )
    nav = engine.run()
    return model, engine, nav


def test_grouped_raw_limit_blocks_and_no_double_count():
    panel = _grid_panel(["A", "B"])
    bars = _bars(_exec_rows([
        ("A", f"{_JAN.date()} 14:51:00", 10.0), ("B", f"{_JAN.date()} 14:51:00", 5.0),
        ("A", f"{_FEB.date()} 14:51:00", 10.0), ("B", f"{_FEB.date()} 14:51:00", 5.0),
    ]))
    # A is at its raw upper limit on _JAN -> a buy must be blocked.
    limits = _limits([
        (f"{_JAN.date()}", "A", 10.0, 1.0), (f"{_JAN.date()}", "B", 9.0, 1.0),
        (f"{_FEB.date()}", "A", 99.0, 1.0), (f"{_FEB.date()}", "B", 9.0, 1.0),
    ])
    assignments = {_JAN: {"A": 1, "B": 2}, _FEB: {"A": 1, "B": 2}}
    m1, _, _ = _grouped_engine(1, assignments, bars, panel, limits)  # holds A
    m2, _, _ = _grouped_engine(2, assignments, bars, panel, limits)  # holds B
    # the up-limit block is attributed ONLY to the group that wanted to buy A.
    assert m1.up_limit_blocked_buys() >= 1
    assert m2.up_limit_blocked_buys() == 0  # no double count across groups


def test_grouped_missing_bar_blocks_no_daily_fallback():
    panel = _daily_panel({  # daily close MOVES A 100 -> 200; a fallback would show it
        (str(_JAN.date()), "A"): 100.0, (str(_FEB.date()), "A"): 200.0,
        (str(_JAN.date()), "B"): 100.0, (str(_FEB.date()), "B"): 100.0,
    })
    # A has NO execution bar on _JAN (entry) -> blocked; only B has bars.
    bars = _bars(_exec_rows([
        ("B", f"{_JAN.date()} 14:51:00", 10.0), ("B", f"{_FEB.date()} 14:51:00", 10.0),
        ("A", f"{_FEB.date()} 14:51:00", 20.0),
    ]))
    assignments = {_JAN: {"A": 1}, _FEB: {"A": 1}}
    model, _, nav = _grouped_engine(1, assignments, bars, panel)
    reasons = {f.reason for f in model.blocked_fills()}
    assert REASON_NO_BAR in reasons  # A blocked by missing bar
    # A earned nothing (no daily-close 100->200 fallback): first period stays ~cash.
    assert abs(nav["gross_return"].iloc[0]) < 1e-9


# --------------------------------------------------------------------------- #
# 5. Report / figures
# --------------------------------------------------------------------------- #
def _toy_nav(finals: list[float]) -> pd.DataFrame:
    dates = [_JAN, _FEB]
    prev = 1.0
    rows = []
    for i, d in enumerate(dates):
        v = finals[i]
        rows.append({"date": d, "nav": v, "gross_return": v / prev - 1.0,
                     "cost": 0.001, "turnover": 1.0, "net_return": v / prev - 1.0})
        prev = v
    return pd.DataFrame(rows).set_index("date")


def _toy_group(group: int, finals: list[float]) -> GroupRunResult:
    nav = _toy_nav(finals)
    return GroupRunResult(
        group=group, nav_table=nav,
        holdings_log=pd.DataFrame(columns=["date", "symbol", "weight", "rank"]),
        metrics=_group_metrics(nav, pd.DataFrame()),
        up_limit_blocked_buys=0, down_limit_blocked_sells=0,
        missing_limit_rows=0,
        opened_limit_up_minutes=0, opened_limit_down_minutes=0,
        missing_adj_factor_pairs=0,
        blocked_fill_reasons={},
    )


def test_figures_written_and_nonempty(tmp_path):
    groups = tuple(_toy_group(g, [1.0 + 0.01 * g, 1.0 + 0.02 * g]) for g in range(1, 6))
    per = pd.Series([0.01, 0.01], index=[_JAN, _FEB])
    cum = per.cumsum()
    paths = _write_figures(tmp_path, groups, per, cum, 5)
    assert set(paths) == {"nav", "spread", "metrics"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert paths["spread"].name == "mmp_q5_minus_q1_spread.png"


def test_report_h1_and_mentions(tmp_path):
    cfg = load_config(str(_CONFIG))
    groups = tuple(_toy_group(g, [1.0 + 0.01 * g, 1.0 + 0.02 * g]) for g in range(1, 6))
    per = pd.Series([0.01, 0.01], index=[_JAN, _FEB])
    cum = per.cumsum()
    fig_dir = tmp_path / "figs"
    figure_paths = _write_figures(fig_dir, groups, per, cum, 5)
    report_path = tmp_path / "phase_i5d_mmp_quintile_5y.md"
    result = I5dResult(
        config=cfg, n_groups=5, score_feature="intraday_mmp20_ew_0930_1450",
        score_feature_key="mmp_ew", requested_symbols=10, covered_symbols=10,
        uncovered_symbols=(), anchor_dates=2, raw_rows=100, normalized_rows=100,
        minute_live_calls=0, rebalance_count=2, groups=groups,
        spread_per_period=per, spread_cumulative=cum,
        spread_summary={"mean_per_period": 0.01, "total": 0.02, "n_periods": 2},
        monotonicity={"annual_spearman": 1.0, "final_nav_spearman": 1.0},
        per_date_rows=({"date": _JAN, "n_scored": 10, "sizes": (2, 2, 2, 2, 2),
                        "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0},),
        score_coverage={"rows": 20, "valid": 20, "nan": 0},
        price_limit_check=True, limit_coverage={"required": 4, "present": 4, "missing": 0},
        stk_limit_gap_fetches=0, figure_paths=figure_paths,
        report_path=report_path, log_path=tmp_path / "x.log", elapsed=1.0,
    )
    _write_report(result)
    text = report_path.read_text(encoding="utf-8")
    h1 = text.splitlines()[0]
    assert "I5d" in h1  # not a stale I5c/I5b label
    assert "I5c" not in h1 and "I5b" not in h1
    assert "fee_rate=0.001" in text
    assert "analytics.quantiles=5" in text
    assert "mmp_quintile_nav.png" in text  # figures embedded


# --------------------------------------------------------------------------- #
# metric helper sanity
# --------------------------------------------------------------------------- #
def test_max_drawdown_includes_initial_baseline():
    nav = _toy_nav([0.9, 1.2])  # drops to 0.9 from the 1.0 start
    assert _max_drawdown(nav) == pytest.approx(-0.1)
