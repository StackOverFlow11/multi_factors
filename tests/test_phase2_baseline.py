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
    reconstruct_holdings,
    summarize_universe,
    tradability_hit_stats,
)
from qt.pipeline import _FrameScores
from qt.reports import (
    phase2_baseline_required_sections,
    render_phase2_baseline,
)
from portfolio.construct import TopNEqualWeight
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
    summ = summarize_universe(_pit_universe(), "index")
    assert summ["pit"] is True
    assert summ["n_snapshots"] == 2
    assert summ["distinct_names"] == 4  # 001,002,003,004
    assert summ["min_size"] == 3 and summ["max_size"] == 3
    assert summ["avg_churn_in"] == 1.0 and summ["avg_churn_out"] == 1.0


def test_summarize_universe_static_marks_non_pit():
    uni = StaticUniverse(["000001.SZ", "000002.SZ"], {})
    summ = summarize_universe(uni, "static")
    assert summ["pit"] is False
    assert summ["distinct_names"] == 2


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
# reconstruct_holdings
# --------------------------------------------------------------------------- #
def test_reconstruct_holdings_mirrors_topn_build():
    date = pd.Timestamp("2024-01-31")
    syms = ["000001.SZ", "000002.SZ", "000003.SZ"]
    panel = pd.DataFrame(
        {"close": [1.0, 1.0, 1.0]},
        index=pd.MultiIndex.from_product([[date], syms], names=["date", "symbol"]),
    )
    score_panel = pd.Series(
        [0.3, 0.1, 0.2],
        index=pd.MultiIndex.from_product([[date], syms], names=["date", "symbol"]),
        name="score",
    )
    uni = StaticUniverse(syms, {})
    holdings = reconstruct_holdings(
        _FrameScores(score_panel), uni, panel, TopNEqualWeight(2), [date]
    )
    assert len(holdings) == 2  # top_n=2
    held = set(holdings["symbol"])
    assert held == {"000001.SZ", "000003.SZ"}  # two highest scores
    assert holdings["weight"].tolist() == [0.5, 0.5]
    # rank 1 is the highest score (000001.SZ @ 0.3)
    top = holdings.sort_values("rank").iloc[0]
    assert top["symbol"] == "000001.SZ"


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
    qret = pd.DataFrame({1: [0.01], 2: [0.02]}, index=[date])
    return Phase2Result(
        config=cfg,
        elapsed_seconds=12.3,
        first_trade_date=date,
        last_trade_date=date,
        trade_days=1,
        panel_rows=2,
        panel_symbols=2,
        universe_summary=summarize_universe(_pit_universe(), "index"),
        financial_field="roe",
        financial_coverage_overall=0.5,
        financial_coverage_by_rebalance=cov,
        tradability_hits=hits,
        rebalance_dates=(date,),
        holdings=holdings,
        factor_name="momentum_20",
        nav_table=nav,
        avg_turnover=1.0,
        cost_drag=0.001,
        ic_mean=0.05,
        ic_ir=0.5,
        quantile_returns=qret,
        performance={"annual_return": 0.1, "max_drawdown": -0.05,
                     "volatility": 0.2, "sharpe": 0.5},
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


def test_render_reports_holdings_and_rebalance_dates():
    md = render_phase2_baseline(_synthetic_result())
    assert "000001.SZ" in md and "000002.SZ" in md  # holdings listed
    assert "2024-01-31" in md  # rebalance date listed
