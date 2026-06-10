"""Phase 2-1 real-data baseline: collectors, demo/real guard, report fields.

Every test here is NETWORK-FREE (TEST-002): the pure collectors run on synthetic
panels/universes, the demo-guard is checked WITHOUT a tushare call, and the report
renderer runs on a hand-built ``Phase2Result``. No test reads the tushare token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qt.config import load_config
from qt.phase2_baseline import (
    Phase2Result,
    financial_coverage_at_dates,
    summarize_universe,
    tradability_hit_stats,
)
from qt.pipeline import _FrameScores
from qt.reports import (
    phase2_baseline_required_sections,
    render_phase2_baseline,
)
from portfolio.construct import TopNEqualWeight
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution
from universe.index_universe import PITIndexUniverse
from universe.static import StaticUniverse

_CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "phase2_real_baseline.yaml")


# --------------------------------------------------------------------------- #
# Config + demo/real guard (no path confusion)
# --------------------------------------------------------------------------- #
def test_phase2_config_validates_as_real_path():
    cfg = load_config(_CONFIG)
    assert cfg.data.source == "tushare"
    assert cfg.universe.type == "index"
    assert cfg.universe.index_code == "000016.SH"
    # the baseline exercises every P1 red-line
    assert cfg.universe.filters.suspended and cfg.universe.filters.st
    assert cfg.universe.filters.limit_up_down
    assert cfg.processing.neutralize.enabled


def test_run_phase2_baseline_rejects_demo_source(example_config_path):
    # example_config_path is the DEMO config; the baseline must refuse it BEFORE
    # any network call (demo carries no PIT/ann_date/tradability meaning).
    from qt.phase2_baseline import run_phase2_baseline

    with pytest.raises(ValueError, match="REAL-data|tushare"):
        run_phase2_baseline(example_config_path)


# --------------------------------------------------------------------------- #
# summarize_universe
# --------------------------------------------------------------------------- #
def _pit_universe() -> PITIndexUniverse:
    # two snapshots, one name churns out and one churns in between them.
    rows = [
        {"date": "2024-01-31", "symbol": "000001.SZ"},
        {"date": "2024-01-31", "symbol": "000002.SZ"},
        {"date": "2024-01-31", "symbol": "000003.SZ"},
        {"date": "2024-02-29", "symbol": "000001.SZ"},
        {"date": "2024-02-29", "symbol": "000002.SZ"},
        {"date": "2024-02-29", "symbol": "000004.SZ"},  # 003 out, 004 in
    ]
    return PITIndexUniverse(pd.DataFrame(rows), filters={})


def test_summarize_universe_pit_counts_churn():
    summ = summarize_universe(_pit_universe(), "index", "2024-01-01", "2024-03-01")
    assert summ["pit"] is True
    assert summ["n_loaded_snapshots"] == 2
    assert summ["n_window_snapshots"] == 2
    assert summ["distinct_names_in_window"] == 4  # 001,002,003,004
    assert summ["min_size"] == 3 and summ["max_size"] == 3
    assert summ["avg_churn_in"] == 1.0 and summ["avg_churn_out"] == 1.0


def test_summarize_universe_excludes_prestart_lookback():
    # 2022-06-30 carries a name 'Y' that is superseded before the window; the
    # as-of anchor at start (2023-07-01) is 2022-12-31. 'Y' must NOT count toward
    # the in-window distinct names, and the pre-start snapshots must NOT count as
    # in-window snapshots (the MEDIUM finding).
    rows = [
        {"date": "2022-06-30", "symbol": "A"},
        {"date": "2022-06-30", "symbol": "B"},
        {"date": "2022-06-30", "symbol": "Y"},  # only here -> superseded pre-window
        {"date": "2022-12-31", "symbol": "A"},
        {"date": "2022-12-31", "symbol": "B"},
        {"date": "2022-12-31", "symbol": "X"},  # anchor membership at start
        {"date": "2024-01-31", "symbol": "A"},
        {"date": "2024-01-31", "symbol": "B"},
        {"date": "2024-01-31", "symbol": "C"},
        {"date": "2024-02-29", "symbol": "A"},
        {"date": "2024-02-29", "symbol": "B"},
        {"date": "2024-02-29", "symbol": "D"},
    ]
    uni = PITIndexUniverse(pd.DataFrame(rows), filters={})
    summ = summarize_universe(uni, "index", "2023-07-01", "2024-03-01")
    assert summ["n_loaded_snapshots"] == 4
    assert pd.Timestamp(summ["loaded_first"]) == pd.Timestamp("2022-06-30")
    assert summ["n_window_snapshots"] == 2  # only the two 2024 snapshots
    assert pd.Timestamp(summ["anchor_snapshot"]) == pd.Timestamp("2022-12-31")
    # A,B,X (anchor),C,D — but NOT Y (superseded before the window)
    assert summ["distinct_names_in_window"] == 5


def test_summarize_universe_static_marks_non_pit():
    uni = StaticUniverse(["000001.SZ", "000002.SZ"], {})
    summ = summarize_universe(uni, "static", "2024-01-01", "2024-12-31")
    assert summ["pit"] is False
    assert summ["distinct_names_in_window"] == 2


# --------------------------------------------------------------------------- #
# tradability_hit_stats
# --------------------------------------------------------------------------- #
def _flag_panel() -> pd.DataFrame:
    # one date, 4 symbols: one normal, one suspended, one ST, one at up-limit.
    date = pd.Timestamp("2024-01-31")
    syms = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    idx = pd.MultiIndex.from_product([[date], syms], names=["date", "symbol"])
    return pd.DataFrame(
        {
            "close": [10.0, 10.0, 10.0, 10.0],
            "suspended": [False, True, False, False],
            "is_st": [False, False, True, False],
            "at_up_limit": [False, False, False, True],
            "at_down_limit": [False, False, False, False],
        },
        index=idx,
    )


def test_tradability_hit_stats_counts_each_reason():
    panel = _flag_panel()
    date = pd.Timestamp("2024-01-31")
    uni = StaticUniverse(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"], {})
    filters = {"suspended": True, "st": True, "limit_up_down": True, "missing_close": True}
    hits = tradability_hit_stats(uni, panel, [date], filters)
    assert int(hits.loc["suspended", "hits"]) == 1
    assert int(hits.loc["is_st", "hits"]) == 1
    assert int(hits.loc["at_up_limit", "hits"]) == 1
    assert int(hits.loc["missing_close", "hits"]) == 0
    assert hits.attrs["candidates"] == 4
    assert hits.attrs["tradable"] == 1  # only 000001.SZ survives


def test_tradability_hit_stats_multiflag_counted_once():
    # A name flagged by MULTIPLE filters must be counted ONCE, in the first
    # matching bucket (mirrors apply_tradable_filters' first-match continue), so
    # the buckets stay exclusive and sum(hits) == candidates - tradable.
    date = pd.Timestamp("2024-01-31")
    idx = pd.MultiIndex.from_product([[date], ["000001.SZ"]], names=["date", "symbol"])
    panel = pd.DataFrame(
        {"close": [10.0], "suspended": [True], "is_st": [True],
         "at_up_limit": [False], "at_down_limit": [False]},
        index=idx,
    )
    uni = StaticUniverse(["000001.SZ"], {})
    hits = tradability_hit_stats(
        uni, panel, [date], {"suspended": True, "st": True, "limit_up_down": True}
    )
    assert int(hits.loc["suspended", "hits"]) == 1  # first match
    assert int(hits.loc["is_st", "hits"]) == 0  # NOT double-counted
    assert int(hits["hits"].sum()) == hits.attrs["candidates"] - hits.attrs["tradable"]


def test_tradability_hit_stats_missing_close_counted():
    panel = _flag_panel().copy()
    panel.loc[(pd.Timestamp("2024-01-31"), "000001.SZ"), "close"] = np.nan
    uni = StaticUniverse(["000001.SZ"], {})
    hits = tradability_hit_stats(uni, panel, [pd.Timestamp("2024-01-31")], {})
    assert int(hits.loc["missing_close", "hits"]) == 1
    assert hits.attrs["tradable"] == 0


# --------------------------------------------------------------------------- #
# driver achieved holdings (the source the report uses)
# --------------------------------------------------------------------------- #
def test_holdings_dates_equal_settled_nav_index_not_candidates():
    # Regression for the HIGH finding: the driver SKIPS a terminal rebalance with
    # no forward holding period, so the diagnostics must key off nav_table.index
    # (settled dates), NOT driver.rebalance_dates() (candidate dates). Holdings
    # listed for the skipped terminal date would be a phantom period.
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    cal = pd.bdate_range("2024-01-01", "2024-03-29")  # spans 3 month-ends
    idx = pd.MultiIndex.from_product([cal, symbols], names=["date", "symbol"])
    close = [100.0 + i + j for i in range(len(cal)) for j in range(len(symbols))]
    panel = pd.DataFrame({"close": close}, index=idx)

    uni = StaticUniverse(symbols, {})
    constructor = TopNEqualWeight(2)
    probe = BacktestDriver(
        universe=uni, scores=_FrameScores(pd.Series(dtype=float)),
        constructor=constructor, execution=SimExecution(), prices=panel,
    )
    candidate_dates = probe.rebalance_dates()  # 3 (Jan/Feb/Mar month-ends)
    # scores at every candidate date so each settled period holds names.
    score_idx = pd.MultiIndex.from_product([candidate_dates, symbols], names=["date", "symbol"])
    score_panel = pd.Series(
        [0.3 - j * 0.1 for _ in candidate_dates for j in range(len(symbols))],
        index=score_idx, name="score",
    )
    scores = _FrameScores(score_panel)
    driver = BacktestDriver(
        universe=uni, scores=scores, constructor=constructor,
        execution=SimExecution(), prices=panel,
    )
    nav = driver.run()

    assert len(nav) == len(candidate_dates) - 1  # terminal date skipped (BT-003)
    holdings = driver.holdings_log()  # the ACHIEVED book the report uses
    assert set(holdings["date"]) == set(nav.index)
    assert candidate_dates[-1] not in set(holdings["date"])  # no phantom terminal


# --------------------------------------------------------------------------- #
# financial_coverage_at_dates
# --------------------------------------------------------------------------- #
def test_financial_coverage_counts_non_nan_members():
    date = pd.Timestamp("2024-01-31")
    syms = ["000001.SZ", "000002.SZ", "000003.SZ"]
    idx = pd.MultiIndex.from_product([[date], syms], names=["date", "symbol"])
    aligned = pd.Series([1.0, np.nan, 3.0], index=idx, name="roe")
    uni = StaticUniverse(syms, {})
    cov = financial_coverage_at_dates(aligned, uni, _flag_panel(), [date])
    row = cov.iloc[0]
    assert int(row["n_members"]) == 3
    assert int(row["n_covered"]) == 2
    assert row["coverage"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# render_phase2_baseline — all required sections + no secret leak
# --------------------------------------------------------------------------- #
def _synthetic_result() -> Phase2Result:
    cfg = load_config(_CONFIG)  # real config (carries the secret FILE PATH)
    date = pd.Timestamp("2024-01-31")
    nav = pd.DataFrame(
        {
            "nav": [1.01],
            "gross_return": [0.011],
            "cost": [0.001],
            "turnover": [1.0],
            "net_return": [0.01],
        },
        index=pd.Index([date], name="date"),
    )
    holdings = pd.DataFrame(
        {"date": [date, date], "symbol": ["000001.SZ", "000002.SZ"],
         "weight": [0.5, 0.5], "rank": [1, 2]}
    )
    hits = tradability_hit_stats(
        StaticUniverse(["000001.SZ"], {}), _flag_panel(), [date],
        {"suspended": True, "st": True, "limit_up_down": True},
    )
    cov = pd.DataFrame(
        {"date": [date], "n_members": [2], "n_covered": [1], "coverage": [0.5]}
    )
    feas = pd.DataFrame(
        {"blocked_buys": [1], "blocked_sells": [0], "cash_constrained_buys": [0],
         "carried": [1], "executed_turnover": [1.0], "invested": [0.5]},
        index=pd.Index([date], name="date"),
    )
    qret = pd.DataFrame({1: [0.01], 2: [0.02]}, index=[date])
    return Phase2Result(
        config=cfg,
        elapsed_seconds=12.3,
        first_trade_date=date,
        last_trade_date=date,
        trade_days=1,
        panel_rows=2,
        panel_symbols=2,
        universe_summary=summarize_universe(_pit_universe(), "index", "2024-01-01", "2024-03-01"),
        list_date_known=67,
        list_date_total=68,
        industry_pit_coverage=0.985,
        financial_coverage={
            "roe": {"is_factor": False, "overall": 0.5, "by_rebalance": cov}
        },
        tradability_hits=hits,
        feasibility_log=feas,
        rebalance_dates=(date,),
        candidate_rebalance_dates=(date, pd.Timestamp("2024-02-29")),
        skipped_terminal_dates=(pd.Timestamp("2024-02-29"),),
        holdings=holdings,
        factor_name="momentum_20",
        factor_names=("momentum_20",),
        per_factor={
            "momentum_20": {"ic_mean": 0.05, "ic_ir": 0.5,
                            "quantile_returns": qret, "coverage": 0.9}
        },
        combo_analytics={"ic_mean": 0.04, "ic_ir": 0.45, "quantile_returns": qret},
        alpha_summary={"model": "equal_weight"},
        alpha_weights=None,
        nav_table=nav,
        avg_turnover=1.0,
        cost_drag=0.001,
        ic_mean=0.05,
        ic_ir=0.5,
        quantile_returns=qret,
        performance={"annual_return": 0.1, "max_drawdown": -0.05,
                     "volatility": 0.2, "sharpe": 0.5},
        std_performance={"backend": "quantstats", "cagr": 0.11, "sharpe": 0.52,
                         "max_drawdown": -0.06, "volatility": 0.21},
        std_factor={"backend": "alphalens", "ic_mean": 0.05, "ic_ir": 0.4,
                    "quantile_mean": {1: -0.01, 2: 0.0, 3: 0.01, 4: 0.02, 5: 0.03},
                    "n_dates": 11},
        downgrades=("DATA PATH = REAL tushare: ...", "small-scale baseline"),
        report_path=Path("artifacts/reports/phase2_real_baseline.md"),
        log_path=Path("artifacts/logs/run_phase2_baseline.log"),
    )


def test_render_has_all_required_sections():
    md = render_phase2_baseline(_synthetic_result())
    for section in phase2_baseline_required_sections():
        assert section in md, f"missing report section: {section}"


def test_render_does_not_leak_secret_file_or_token():
    result = _synthetic_result()
    md = render_phase2_baseline(result)
    # the secret FILE PATH and token KEY must never be echoed into the report.
    assert result.config.data.external_secret_file not in md
    assert result.config.data.tushare_token_key not in md
    assert "token" not in md.lower()


def test_render_shows_standard_analytics_cross_check():
    md = render_phase2_baseline(_synthetic_result())
    assert "## Standard analytics" in md
    assert "quantstats" in md and "alphalens" in md  # backends named
    assert "authoritative" in md.lower()  # simple metrics flagged authoritative


def test_render_reports_holdings_and_rebalance_dates():
    md = render_phase2_baseline(_synthetic_result())
    assert "000001.SZ" in md and "000002.SZ" in md  # holdings listed
    assert "2024-01-31" in md  # rebalance date listed


def test_render_discloses_list_date_coverage_this_run():
    md = render_phase2_baseline(_synthetic_result())
    assert "67/68" in md          # known/total for THIS run
    assert "missing" in md.lower()  # the data gap is disclosed, not just generic


def test_render_discloses_pit_industry_coverage():
    md = render_phase2_baseline(_synthetic_result())
    assert "point-in-time SW-L1" in md  # phase2 config sets industry_level: L1
    assert "98.50%" in md  # industry_pit_coverage=0.985 rendered as a pct
    assert "current-tag fallback" in md  # explicitly no silent fallback


# --------------------------------------------------------------------------- #
# P3-1 — multi-factor report surface
# --------------------------------------------------------------------------- #
def test_render_shows_active_factor_list_and_combo():
    md = render_phase2_baseline(_synthetic_result())
    assert "factors (active)" in md          # active factor list disclosed
    assert "combo score (equal-weight)" in md  # combo diagnostics rendered


def test_render_financial_coverage_labels_role_per_field():
    import dataclasses

    base = _synthetic_result()
    cov = base.financial_coverage["roe"]["by_rebalance"]
    multi = dataclasses.replace(
        base,
        financial_coverage={
            "roe": {"is_factor": True, "overall": 0.9, "by_rebalance": cov},
            "netprofit_yoy": {"is_factor": False, "overall": 0.4, "by_rebalance": cov},
        },
    )
    md = render_phase2_baseline(multi)
    assert "### `roe` — TRADED financial factor" in md
    assert "### `netprofit_yoy` — diagnostic only" in md
    assert "90.00%" in md and "40.00%" in md  # per-field overall coverage


def test_render_per_factor_table_includes_coverage_and_ic():
    md = render_phase2_baseline(_synthetic_result())
    # the per-factor table row carries the factor's coverage + IC.
    assert "| `momentum_20` | 90.00% | 0.0500 | 0.5000 |" in md


def test_baseline_report_name_is_configurable(tmp_path):
    """output.baseline_report_name overrides the default phase2 report filename."""
    import yaml as _yaml

    raw = _yaml.safe_load(Path(_CONFIG).read_text(encoding="utf-8"))
    raw["output"]["baseline_report_name"] = "phase3_real_multifactor.md"
    p = tmp_path / "cfg.yaml"
    p.write_text(_yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.output.baseline_report_name == "phase3_real_multifactor.md"
    # default stays None -> historical filename preserved.
    assert load_config(_CONFIG).output.baseline_report_name is None


# --------------------------------------------------------------------------- #
# P3-2 — alpha model disclosure in the report
# --------------------------------------------------------------------------- #
def test_render_equal_weight_alpha_disclosed_without_weights():
    md = render_phase2_baseline(_synthetic_result())
    assert "## Alpha model" in md
    assert "`equal_weight`" in md
    assert "no trained weights" in md


def test_render_ic_weighted_alpha_shows_weights_and_fallback():
    import dataclasses

    base = _synthetic_result()
    date = base.rebalance_dates[0]
    weights = pd.DataFrame(
        {"momentum_20": [0.7], "roe": [-0.3], "fallback": [False]},
        index=pd.Index([date], name="date"),
    )
    ic = dataclasses.replace(
        base,
        alpha_summary={"model": "ic_weighted", "window": 60, "min_periods": 20,
                       "horizon": 1, "mode": "rolling", "n_dates": 220,
                       "n_fallback": 25, "trained_coverage": 195 / 220},
        alpha_weights=weights,
    )
    md = render_phase2_baseline(ic)
    assert "`ic_weighted`" in md and "walk-forward" in md
    assert "195/220" in md and "25" in md          # fallback count disclosed
    assert "0.7000" in md and "-0.3000" in md      # per-rebalance weights table
    assert "t + horizon <= d" in md                # lookahead boundary stated
    assert "NOT a tuned-" in md                    # no tuned-performance claim
    assert "equal-weight baseline" in md.lower()   # comparison pointer present
