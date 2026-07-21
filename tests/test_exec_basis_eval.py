"""The runner-facing exec-basis evaluation glue — network-free, end to end.

Checks the three things the eleven runners depend on and that a unit test of the
return builder alone would not catch:

  * the exec reports are written UNDER THEIR OWN names, so the accepted
    close_to_close artifacts cannot be overwritten;
  * the mandatory disclosure (execution parameters, coverage loss by cause,
    measured live-call count) actually reaches the rendered report;
  * adding that disclosure does NOT move a verdict axis — the Tradable axis stays
    NOT_ASSESSED, because a return basis is not a fill-feasibility measurement.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from analytics.eval.config import EvalConfig
from analytics.eval.verdict import AXIS_NOT_ASSESSED
from data.cache.intraday_cache import ENDPOINT as MINUTE_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ
from factors.spec import FactorSpec
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
from qt.exec_basis_eval import run_exec_basis_evaluation

LOGGER = logging.getLogger("test.exec_basis_eval")
SYMBOLS = ("A.SZ", "B.SZ", "C.SZ", "D.SZ")


def _days(n: int) -> list[pd.Timestamp]:
    """``n`` weekday sessions — a daily grid the evaluator accepts as daily."""
    out: list[pd.Timestamp] = []
    day = pd.Timestamp("2024-01-01")
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += pd.Timedelta(days=1)
    return out


def _minute_rows(symbol: str, day: pd.Timestamp, amount: float) -> list[dict]:
    rows = []
    for offset, (close, vol, amt) in enumerate(
        [(9.0, 10.0, 90.0), (9.5, 10.0, 95.0), (10.0, 100.0, amount), (11.0, 10.0, 110.0)]
    ):
        end = day + pd.Timedelta("14:49:00") + pd.Timedelta(minutes=offset)
        rows.append(
            {
                "symbol": symbol, "bar_end": end, "source_trade_time": end,
                "open": close, "high": close, "low": close, "close": close,
                "volume": vol, "amount": amt, "freq": RAW_INTRADAY_FREQ,
            }
        )
    return rows


def _fixture(tmp_path, n_days: int = 40):
    days = _days(n_days)
    rng = np.random.default_rng(7)
    root = tmp_path / "cache"
    store = IntradayParquetStore(str(root))

    price = {s: 1000.0 for s in SYMBOLS}
    rows: list[dict] = []
    closes: dict[tuple[pd.Timestamp, str], float] = {}
    for day in days:
        for sym in SYMBOLS:
            price[sym] *= 1.0 + float(rng.normal(0.0, 0.02))
            closes[(day, sym)] = price[sym]
            # exec bar VWAP == amount / volume == price (volume 100).
            rows.extend(_minute_rows(sym, day, price[sym] * 100.0))
    frame = pd.DataFrame(rows)
    for sym, part in frame.groupby("symbol"):
        store.upsert(MINUTE_ENDPOINT, str(sym), RAW_INTRADAY_FREQ, part, KEY_COLS)

    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(days), list(SYMBOLS)], names=["date", "symbol"]
    )
    close_values = [closes[(d, s)] for d, s in index]
    panel = pd.DataFrame(
        {
            "open": close_values, "high": close_values, "low": close_values,
            "close": close_values, "volume": 100.0, "amount": 100.0,
            "adj_factor": 1.0,
        },
        index=index,
    )
    factor = pd.Series(
        rng.normal(size=len(index)), index=index, name="demo_minute_factor"
    )
    book = pd.DataFrame({"book_a": rng.normal(size=len(index))}, index=index)
    cfg = RootConfig(
        data=DataCfg(
            source="tushare",
            start=str(days[0].date()),
            end=str(days[-1].date()),
            external_secret_file="/nonexistent.json",
            cache=CacheCfg(enabled=True, root_dir=str(root)),
        ),
        universe=UniverseCfg(type="static", symbols=list(SYMBOLS)),
        factors=[FactorCfg(name="momentum_20")],
        alpha=AlphaCfg(), portfolio=PortfolioCfg(top_n=2),
        backtest=BacktestCfg(), cost=CostCfg(),
        output=OutputCfg(
            data_dir=str(tmp_path / "data"), report_dir=str(tmp_path / "reports")
        ),
    )
    spec = FactorSpec(
        factor_id="demo_minute_factor", version="1.0",
        description="synthetic minute-derived factor", expected_ic_sign=1,
        is_intraday=False, forward_return_horizon=1,
        return_basis="close_to_close", input_fields=("close",),
        family="microstructure",
    )
    eval_cfg = EvalConfig(
        universe="static", universe_is_pit=False,
        start=str(days[0].date()), end=str(days[-1].date()),
        is_exploratory=True, post_hoc_selected=False, rebalance="daily",
    )
    return cfg, panel, factor, book, spec, eval_cfg, tmp_path / "reports"


def test_exec_basis_eval_writes_its_own_reports_and_leaves_the_control_alone(tmp_path):
    cfg, panel, factor, book, spec, eval_cfg, report_dir = _fixture(tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Stand-ins for the accepted close_to_close artifacts.
    control = {
        report_dir / "demo_no_book.md": "CLOSE-TO-CLOSE CONTROL",
        report_dir / "demo_with_book.md": "CLOSE-TO-CLOSE CONTROL",
    }
    for path, text in control.items():
        path.write_text(text, encoding="utf-8")

    out = run_exec_basis_evaluation(
        factor, spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo",
    )

    for path, text in control.items():
        assert path.read_text(encoding="utf-8") == text  # untouched
    assert out.no_book_md.name == "demo_exec_no_book.md"
    assert out.with_book_md.name == "demo_exec_with_book.md"
    for path in (out.no_book_md, out.with_book_md, out.no_book_json,
                 out.with_book_json, out.sanity_report_path,
                 out.no_book_dashboard, out.with_book_dashboard):
        assert path.exists()
    assert out.spec.is_intraday is True
    assert out.spec.return_basis == "exec_to_exec"
    assert out.minute_live_calls == 0


def test_exec_basis_eval_discloses_parameters_and_coverage_in_every_report(tmp_path):
    cfg, panel, factor, book, spec, eval_cfg, report_dir = _fixture(tmp_path)
    out = run_exec_basis_evaluation(
        factor, spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo",
    )
    for path in (out.no_book_md, out.with_book_md):
        text = path.read_text(encoding="utf-8")
        assert "exec_to_exec" in text
        assert "14:51:00" in text                      # execution window
        assert "bar_vwap" in text                      # price basis
        assert "adj_factor" in text                    # the adjustment identity
        assert "stk_mins_live_calls" in text
        for cause in ("no_bar", "bad_vwap", "bad_adj_factor"):
            assert f"lost_pairs_by_cause_{cause}" in text
    for path in (out.no_book_json, out.with_book_json):
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert "exec_to_exec" in blob
        assert "sanity_corr_vs_close_to_close_median" in blob
    # The forward-return provenance says the returns came from OUTSIDE the evaluator.
    assert "computed OUTSIDE this evaluator" in out.no_book_md.read_text(encoding="utf-8")


def test_exec_basis_disclosure_does_not_move_the_tradable_axis(tmp_path):
    """A return-basis disclosure must not read as measured fill feasibility."""
    cfg, panel, factor, book, spec, eval_cfg, report_dir = _fixture(tmp_path)
    out = run_exec_basis_evaluation(
        factor, spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo",
    )
    for report in (out.no_book, out.with_book):
        assert report.require_verdict().tradable.verdict == AXIS_NOT_ASSESSED
    assert out.no_book_metrics["tradable"] == AXIS_NOT_ASSESSED


def test_exec_basis_eval_shares_one_artifact_across_factors(tmp_path):
    """Two factors on the same universe/window/parameters get the SAME returns.

    This is what makes cross-factor comparison meaningful: eleven factors scored
    against eleven slightly different return series would not be comparable.
    """
    cfg, panel, factor, book, spec, eval_cfg, report_dir = _fixture(tmp_path)
    first = run_exec_basis_evaluation(
        factor, spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo_one",
    )
    other_spec = FactorSpec(
        factor_id="other_minute_factor", version="2.0",
        description="a different synthetic minute factor", expected_ic_sign=-1,
        is_intraday=False, forward_return_horizon=1,
        return_basis="close_to_close", input_fields=("close",),
    )
    other_factor = factor.rename("other_minute_factor") * -1.0
    second = run_exec_basis_evaluation(
        other_factor, other_spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo_two",
    )
    assert first.artifact_key == second.artifact_key
    assert first.artifact_reused is False
    assert second.artifact_reused is True
    assert first.coverage == second.coverage


def test_exec_basis_cli_line_survives_an_absent_metric(tmp_path):
    """A Skipped section leaves None metrics; the summary must not raise on them.

    The print runs after four reports have been written and outside the command's
    error handling, so a TypeError here would bury finished work in a traceback.
    """
    cfg, panel, factor, book, spec, eval_cfg, report_dir = _fixture(tmp_path, n_days=30)
    out = run_exec_basis_evaluation(
        factor, spec, eval_cfg, book,
        cfg=cfg, panel=panel, symbols=list(SYMBOLS), logger=LOGGER,
        report_dir=report_dir, stem="demo",
    )
    import dataclasses

    from qt.exec_basis_eval import format_exec_basis_line

    blanked = dataclasses.replace(
        out,
        no_book_metrics={**out.no_book_metrics, "ic_mean": None, "ic_ir": None},
        with_book_metrics={**out.with_book_metrics, "incremental_ic_ir": None},
    )
    line = format_exec_basis_line(blanked)
    assert "ic_mean=n/a" in line
    assert "incr_ic_ir=n/a" in line
    # the measured facts are still reported
    assert "stk_mins_live_calls=0" in line
    assert "no_bar=" in line
