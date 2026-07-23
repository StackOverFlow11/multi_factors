"""Tests for the factor-evaluation dashboard renderer (analytics/eval/figures.py).

Pure-rendering checks (a PNG is written, no display, no crash on degenerate data)
plus the two contract guarantees:
  * ``evaluate_with_ir`` returns a report byte-identical to ``evaluate``'s;
  * the MANDATORY factor-definition text carries how the factor is computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.eval import EvalConfig, EvalContext, StandardFactorEvaluator
from analytics.eval.figures import (
    DashboardData,
    _definition_block_line,
    _definition_meta_line,
    render_factor_dashboard,
)
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.spec import FactorSpec, PanelField


# --------------------------------------------------------------------------- #
# toy builders (small, self-contained)
# --------------------------------------------------------------------------- #
def _panel(dates, symbols, rng, signal=None):
    n_d, n_s = len(dates), len(symbols)
    noise = rng.normal(0.0, 0.02, size=(n_d, n_s))
    returns = noise if signal is None else noise + signal
    close = pd.DataFrame(np.exp(np.log(100.0) + np.cumsum(returns, axis=0)),
                         index=dates, columns=symbols)
    stacked = close.stack().sort_index()
    stacked.index.names = [DATE_LEVEL, SYMBOL_LEVEL]
    return pd.DataFrame({
        "open": stacked, "high": stacked * 1.01, "low": stacked * 0.99,
        "close": stacked, "volume": 1_000_000.0, "amount": 100_000_000.0,
        "adj_factor": 1.0,
    })


def _spec(**over):
    kw = dict(factor_id="synth_factor", version="1.0",
              description="a synthetic factor computed as the 20d mean of X",
              expected_ic_sign=1, is_intraday=False, forward_return_horizon=1,
              return_basis="close_to_close", input_fields=("close", "amount"),
              # Contract v1.0 mandatory declarations (D1).
              requires=(PanelField("close", source="market_daily"),
                        PanelField("amount", source="market_daily")),
              adjustment="returns_invariant", overnight_boundary="none",
              family="microstructure")
    kw.update(over)
    return FactorSpec(**kw)


def _cfg(**over):
    kw = dict(universe="TEST500", universe_is_pit=True, start="2024-01-01",
              end="2025-01-01", is_exploratory=True, post_hoc_selected=False,
              rebalance="daily", n_quantiles=5)
    kw.update(over)
    return EvalConfig(**kw)


@pytest.fixture
def rng():
    return np.random.default_rng(20260718)


def _report_and_ir(rng, *, oos=False):
    dates = pd.bdate_range("2024-01-02", periods=60, name=DATE_LEVEL)
    symbols = [f"{i:06d}.SZ" for i in range(15)]
    # a planted cross-sectional signal so IC / quantiles are non-degenerate
    strength = np.linspace(-0.004, 0.004, len(symbols))
    panel = _panel(dates, symbols, rng, signal=strength)
    factor = pd.Series(
        np.repeat(strength[None, :], len(dates), axis=0).reshape(-1),
        index=panel.index, name="synth_factor",
    ) + rng.normal(0, 1e-4, size=len(panel))
    cfg = _cfg(oos_split="2024-02-15") if oos else _cfg()
    ctx = EvalContext(price_panel=panel[["open", "high", "low", "close",
                                         "volume", "amount", "adj_factor"]],
                      universe_symbols=tuple(symbols))
    return StandardFactorEvaluator().evaluate_with_ir(factor, _spec(), cfg, ctx)


# --------------------------------------------------------------------------- #
# evaluate_with_ir mirrors evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_with_ir_report_is_identical_to_evaluate(rng):
    dates = pd.bdate_range("2024-01-02", periods=50, name=DATE_LEVEL)
    symbols = [f"{i:06d}.SZ" for i in range(12)]
    panel = _panel(dates, symbols, rng)
    factor = pd.Series(rng.normal(size=len(panel)), index=panel.index,
                       name="synth_factor")
    ctx = EvalContext(price_panel=panel)
    ev = StandardFactorEvaluator()
    report_only = ev.evaluate(factor, _spec(), _cfg(), ctx)
    report_wi, ir = ev.evaluate_with_ir(factor, _spec(), _cfg(), ctx)
    assert report_wi.to_json() == report_only.to_json()
    assert isinstance(ir.ic, pd.Series)
    assert isinstance(ir.quantile_returns, pd.DataFrame)


# --------------------------------------------------------------------------- #
# the dashboard renders to a PNG
# --------------------------------------------------------------------------- #
def test_render_writes_a_nontrivial_png(rng, tmp_path):
    report, ir = _report_and_ir(rng, oos=True)
    out = render_factor_dashboard(report, ir, tmp_path / "dash.png")
    assert out.exists()
    assert out.stat().st_size > 20_000  # a real multi-panel figure, not an empty axes
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_render_survives_empty_series_and_missing_payloads(tmp_path):
    """Degenerate input (no IC, no quantiles, no sections) must not crash."""
    from analytics.eval.verdict import REJECT, VerdictResult

    data = DashboardData(
        spec=_spec(), verdict=VerdictResult(REJECT, ("hand-built",)),
        payloads={}, ic=pd.Series(dtype=float), quantile_returns=pd.DataFrame(),
    )
    from analytics.eval.figures import _render

    out = _render(data, tmp_path / "empty.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_from_report_pulls_spec_verdict_payloads_and_series(rng):
    report, ir = _report_and_ir(rng)
    data = DashboardData.from_report(report, ir)
    assert data.spec.factor_id == "synth_factor"
    assert data.verdict is report.require_verdict()
    assert "predictive_power" in data.payloads
    assert data.ic.equals(ir.ic)


# --------------------------------------------------------------------------- #
# MANDATORY factor-definition content
# --------------------------------------------------------------------------- #
def test_definition_meta_line_states_how_the_factor_is_computed():
    spec = _spec()
    line = _definition_meta_line(spec)
    for field in spec.input_fields:
        assert field in line
    assert "expected sign: +1" in line
    assert f"horizon: {spec.forward_return_horizon}" in line
    assert spec.return_basis in line
    assert spec.family in line
    assert f"min-history: {spec.min_history_bars}" in line
    assert spec.price_adjust in line


def test_definition_block_is_none_for_a_daily_factor():
    assert _definition_block_line(_spec(is_intraday=False)) is None


def test_definition_block_states_the_minute_contract_for_an_intraday_factor():
    spec = _spec(
        is_intraday=True, return_basis="exec_to_exec",
        decision_cutoff="14:50:00", data_lag="1min", session_open="09:30:00",
        execution_model="next_minute_close", execution_window="[14:51,14:56:59]",
    )
    block = _definition_block_line(spec)
    assert block is not None
    for token in ("14:50:00", "1min", "09:30:00", "next_minute_close",
                  "[14:51,14:56:59]"):
        assert token in block
