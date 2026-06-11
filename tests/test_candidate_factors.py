"""P3-5: candidate factor pack (network-free).

Locks the candidate-factor contract:
  * every factor uses ONLY data known at the trade date (perturbing future bars
    cannot change today's value); leading windows are NaN; computation is
    strictly per-symbol (no cross-symbol leakage); inputs are never mutated;
  * value_ep / value_bp surface an enriched daily_basic column (1/pe, 1/pb;
    non-positive ratios -> NaN) and fail readably on the demo source;
  * grossprofit_margin joins the ann_date-aligned financial fields;
  * the factor dispatch (registry) builds every new name, keeps the legacy
    names byte-identical, and rejects a name/params mismatch;
  * a demo end-to-end run with the price-based candidates works (per-factor
    analytics populated) — legacy demo numbers unchanged elsewhere.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factors.compute.candidates import (
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import SUPPORTED_FIELDS, FinancialFactor
from factors.compute.momentum import MomentumFactor
from qt.config import load_config
from qt.pipeline import _build_factors, run_phase0

_CANDIDATES_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_factor_candidates.yaml"
)


def _panel(n_days=30, syms=("A", "B")):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, list(syms)], names=["date", "symbol"])
    close = []
    open_ = []
    amount = []
    for i in range(n_days):
        for j, _ in enumerate(syms):
            c = 100.0 + i + 50.0 * j               # deterministic per-symbol path
            close.append(c)
            open_.append(c - 0.4 - 0.1 * j)        # a stable overnight gap
            amount.append(1e6 * (1 + j) + 1e3 * i)
    return pd.DataFrame(
        {"open": open_, "close": close, "amount": amount}, index=idx
    )


# --------------------------------------------------------------------------- #
# reversal
# --------------------------------------------------------------------------- #
def test_reversal_is_negative_momentum_and_lagged_only():
    panel = _panel()
    rev = ReversalFactor(window=5).compute(panel)
    mom = MomentumFactor(window=5).compute(panel)
    pd.testing.assert_series_equal(
        rev, (-mom).rename("reversal_5"), check_names=True
    )
    assert rev.name == "reversal_5"
    # leading window is NaN
    first_sym_a = rev.xs("A", level="symbol")
    assert first_sym_a.iloc[:5].isna().all() and math.isfinite(first_sym_a.iloc[5])


def test_reversal_ignores_future_bars():
    panel = _panel()
    t = pd.Timestamp(panel.index.get_level_values("date").unique()[10])
    before = ReversalFactor(window=5).compute(panel).loc[(t, "A")]
    poisoned = panel.copy()
    future = poisoned.index.get_level_values("date") > t
    poisoned.loc[future, "close"] = 9_999.0
    after = ReversalFactor(window=5).compute(poisoned).loc[(t, "A")]
    assert before == after


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def test_volatility_matches_manual_rolling_std():
    panel = _panel()
    vol = VolatilityFactor(window=20).compute(panel)
    assert vol.name == "volatility_20"
    close_a = panel.xs("A", level="symbol")["close"]
    manual = close_a.pct_change().rolling(20, min_periods=20).std(ddof=1)
    got = vol.xs("A", level="symbol")
    pd.testing.assert_series_equal(
        got, manual.rename("volatility_20"), check_names=True
    )
    assert got.iloc[:20].isna().all()  # needs a full window of RETURNS


def test_volatility_is_per_symbol_no_cross_leakage():
    panel = _panel()
    # poison symbol B entirely; A's volatility must not move.
    before = VolatilityFactor(window=20).compute(panel).xs("A", level="symbol")
    poisoned = panel.copy()
    b_rows = poisoned.index.get_level_values("symbol") == "B"
    poisoned.loc[b_rows, "close"] = 1.0
    after = VolatilityFactor(window=20).compute(poisoned).xs("A", level="symbol")
    pd.testing.assert_series_equal(before, after)


# --------------------------------------------------------------------------- #
# liquidity
# --------------------------------------------------------------------------- #
def test_liquidity_is_log_mean_amount():
    panel = _panel()
    liq = LiquidityFactor(window=20).compute(panel)
    assert liq.name == "liquidity_20"
    amt_a = panel.xs("A", level="symbol")["amount"]
    manual = np.log(amt_a.rolling(20, min_periods=20).mean())
    pd.testing.assert_series_equal(
        liq.xs("A", level="symbol"), manual.rename("liquidity_20"),
        check_names=True,
    )


def test_liquidity_nonpositive_amount_is_nan_not_crash():
    panel = _panel()
    panel = panel.copy()
    panel.loc[panel.index[:8], "amount"] = 0.0  # degenerate turnover
    liq = LiquidityFactor(window=4).compute(panel)
    assert not np.isinf(liq.dropna()).any()  # log(0) never leaks as -inf


def test_liquidity_requires_amount_column():
    panel = _panel().drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        LiquidityFactor(window=20).compute(panel)


# --------------------------------------------------------------------------- #
# overnight momentum
# --------------------------------------------------------------------------- #
def test_overnight_mom_matches_manual_sum_of_log_gaps():
    panel = _panel()
    out = OvernightMomentumFactor(window=20).compute(panel)
    assert out.name == "overnight_mom_20"
    a = panel.xs("A", level="symbol")
    manual = np.log(a["open"] / a["close"].shift(1)).rolling(
        20, min_periods=20
    ).sum()
    pd.testing.assert_series_equal(
        out.xs("A", level="symbol"), manual.rename("overnight_mom_20"),
        check_names=True,
    )
    # needs w overnight returns (= w+1 bars): rows 0..19 are NaN, row 20 finite
    got_a = out.xs("A", level="symbol")
    assert got_a.iloc[:20].isna().all() and math.isfinite(got_a.iloc[20])


def test_overnight_mom_ignores_future_bars():
    panel = _panel()
    t = pd.Timestamp(panel.index.get_level_values("date").unique()[25])
    before = OvernightMomentumFactor(window=20).compute(panel).loc[(t, "A")]
    poisoned = panel.copy()
    future = poisoned.index.get_level_values("date") > t
    poisoned.loc[future, ["open", "close"]] = 9_999.0
    after = OvernightMomentumFactor(window=20).compute(poisoned).loc[(t, "A")]
    assert before == after


def test_overnight_mom_prev_close_never_crosses_symbols():
    panel = _panel()
    before = OvernightMomentumFactor(window=5).compute(panel).xs("A", level="symbol")
    poisoned = panel.copy()
    b_rows = poisoned.index.get_level_values("symbol") == "B"
    poisoned.loc[b_rows, ["open", "close"]] = 1.0
    after = OvernightMomentumFactor(window=5).compute(poisoned).xs("A", level="symbol")
    pd.testing.assert_series_equal(before, after)


def test_overnight_mom_nonpositive_prices_are_nan_not_inf():
    panel = _panel().copy()
    panel.loc[panel.index[:6], "open"] = 0.0
    out = OvernightMomentumFactor(window=3).compute(panel)
    assert not np.isinf(out.dropna()).any()


def test_overnight_mom_requires_open_column():
    panel = _panel().drop(columns=["open"])
    with pytest.raises(ValueError, match="open"):
        OvernightMomentumFactor(window=20).compute(panel)


# --------------------------------------------------------------------------- #
# value (daily_basic enrichment surface)
# --------------------------------------------------------------------------- #
def test_value_factor_surfaces_enriched_column():
    panel = _panel().copy()
    panel["value_ep"] = 0.05
    out = ValueFactor("value_ep").compute(panel)
    assert out.name == "value_ep"
    assert (out == 0.05).all()


def test_value_factor_missing_column_is_readable_error():
    with pytest.raises(ValueError, match="value_ep"):
        ValueFactor("value_ep").compute(_panel())


def test_value_factor_rejects_unknown_field():
    with pytest.raises(ValueError, match="value_ep|value_bp"):
        ValueFactor("value_cashflow")


# --------------------------------------------------------------------------- #
# quality financial field
# --------------------------------------------------------------------------- #
def test_grossprofit_margin_is_a_supported_financial_field():
    assert "grossprofit_margin" in SUPPORTED_FIELDS
    factor = FinancialFactor("grossprofit_margin")
    assert factor.name == "grossprofit_margin"


# --------------------------------------------------------------------------- #
# dispatch (factor registry)
# --------------------------------------------------------------------------- #
def _cfg_with(tmp_path, example_config_path, factors, source="demo"):
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    raw["data"]["source"] = source
    raw["factors"] = factors
    out = tmp_path / "artifacts"
    raw["output"] = {
        "root_dir": str(out), "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"), "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"), "overwrite": True,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(p))


def test_dispatch_builds_every_candidate(tmp_path, example_config_path):
    specs = [
        {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
        {"name": "reversal_5", "enabled": True, "params": {"window": 5}},
        {"name": "reversal_20", "enabled": True, "params": {"window": 20}},
        {"name": "volatility_20", "enabled": True, "params": {"window": 20}},
        {"name": "liquidity_20", "enabled": True, "params": {"window": 20}},
        {"name": "overnight_mom_20", "enabled": True, "params": {"window": 20}},
        {"name": "value_ep", "enabled": True, "params": {}},
        {"name": "value_bp", "enabled": True, "params": {}},
        {"name": "roe", "enabled": True, "params": {}},
        {"name": "grossprofit_margin", "enabled": True, "params": {}},
    ]
    cfg = _cfg_with(tmp_path, example_config_path, specs, source="tushare")
    factors = _build_factors(cfg)
    assert [f.name for f in factors] == [s["name"] for s in specs]
    assert isinstance(factors[1], ReversalFactor)
    assert isinstance(factors[3], VolatilityFactor)
    assert isinstance(factors[4], LiquidityFactor)
    assert isinstance(factors[5], OvernightMomentumFactor)
    assert isinstance(factors[6], ValueFactor)
    assert isinstance(factors[9], FinancialFactor)


def test_dispatch_rejects_name_params_mismatch(tmp_path, example_config_path):
    # spec says reversal_5 but params build reversal_10 -> a silent mislabel.
    specs = [{"name": "reversal_5", "enabled": True, "params": {"window": 10}}]
    cfg = _cfg_with(tmp_path, example_config_path, specs)
    with pytest.raises(ValueError, match="mismatch|resolve"):
        _build_factors(cfg)


# --------------------------------------------------------------------------- #
# value enrichment wiring
# --------------------------------------------------------------------------- #
def test_value_enrichment_demo_source_raises(tmp_path, example_config_path):
    from qt.pipeline import _maybe_enrich_value

    specs = [{"name": "value_ep", "enabled": True, "params": {}}]
    cfg = _cfg_with(tmp_path, example_config_path, specs, source="demo")
    with pytest.raises(ValueError, match="demo"):
        _maybe_enrich_value(
            cfg, _panel(), ["000001.SZ"], [ValueFactor("value_ep")],
            logging.getLogger("test"),
        )


def test_value_enrichment_inverts_ratios_and_guards_nonpositive(
    tmp_path, example_config_path, monkeypatch
):
    from qt.pipeline import _maybe_enrich_value

    specs = [{"name": "value_ep", "enabled": True, "params": {}},
             {"name": "value_bp", "enabled": True, "params": {}}]
    cfg = _cfg_with(tmp_path, example_config_path, specs, source="tushare")
    calls = []

    class _FakeFeed:
        def __init__(self, *a, **k):
            pass

        def value_ratios(self, symbols, start, end):
            calls.append(list(symbols))
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-02",
                                            "2024-01-03"]),
                    "symbol": ["A", "A", "A"],
                    "pe": [20.0, -5.0, None],   # negative / missing -> NaN
                    "pb": [2.0, 0.0, 4.0],      # zero -> NaN
                }
            )

    monkeypatch.setattr("qt.pipeline.TushareCovariatesFeed", _FakeFeed)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({"close": [1.0, 1.1, 1.2]}, index=idx)
    out = _maybe_enrich_value(
        cfg, panel, ["A"],
        [ValueFactor("value_ep"), ValueFactor("value_bp")],
        logging.getLogger("test"),
    )
    assert len(calls) == 1  # ONE fetch for both value fields
    assert out.loc[(pd.Timestamp("2024-01-01"), "A"), "value_ep"] == pytest.approx(1 / 20)
    assert math.isnan(out.loc[(pd.Timestamp("2024-01-02"), "A"), "value_ep"])  # pe<0
    assert math.isnan(out.loc[(pd.Timestamp("2024-01-03"), "A"), "value_ep"])  # pe NaN
    assert out.loc[(pd.Timestamp("2024-01-01"), "A"), "value_bp"] == pytest.approx(0.5)
    assert math.isnan(out.loc[(pd.Timestamp("2024-01-02"), "A"), "value_bp"])  # pb==0
    assert "value_ep" not in panel.columns  # input not mutated


# --------------------------------------------------------------------------- #
# demo e2e with the price-based candidates + config validation
# --------------------------------------------------------------------------- #
def test_demo_e2e_with_price_candidates(tmp_path, example_config_path):
    specs = [
        {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
        {"name": "reversal_5", "enabled": True, "params": {"window": 5}},
        {"name": "volatility_20", "enabled": True, "params": {"window": 20}},
        {"name": "liquidity_20", "enabled": True, "params": {"window": 20}},
    ]
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    raw["factors"] = specs
    out = tmp_path / "artifacts"
    raw["output"] = {
        "root_dir": str(out), "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"), "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"), "overwrite": True,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = run_phase0(str(p))
    assert result.factor_names == (
        "momentum_20", "reversal_5", "volatility_20", "liquidity_20"
    )
    for name in result.factor_names:
        assert name in result.per_factor
    text = result.report_path.read_text(encoding="utf-8")
    assert "reversal_5" in text and "liquidity_20" in text
    assert "token" not in text.lower()


def test_candidates_config_validates():
    cfg = load_config(_CANDIDATES_CONFIG)
    names = [f.name for f in cfg.factors if f.enabled]
    # legacy trio kept for the old-vs-new comparison + the candidate pack
    assert {"momentum_20", "roe", "netprofit_yoy"} <= set(names)
    assert {"reversal_5", "reversal_20", "volatility_20", "liquidity_20",
            "overnight_mom_20", "value_ep", "value_bp",
            "grossprofit_margin"} <= set(names)
    assert cfg.robustness is not None  # runs through the P3-4 matrix
    assert cfg.alpha.model == "ic_weighted"
