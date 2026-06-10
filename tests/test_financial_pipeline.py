"""Pipeline dispatch for financial factors + the demo-source guard (no fabrication)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml

from factors.compute.financial import FinancialFactor
from factors.compute.momentum import MomentumFactor
from qt.config import load_config
from qt.pipeline import _build_factors, _maybe_enrich_financials


def _cfg(tmp_path, example_config_path, factor_name, source):
    base = yaml.safe_load(open(example_config_path, encoding="utf-8"))
    base["factors"] = [{"name": factor_name, "enabled": True, "params": {}}]
    base["data"]["source"] = source
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return load_config(str(path))


def test_build_factor_dispatches_financial(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "roe", "tushare")
    assert isinstance(_build_factors(cfg)[0], FinancialFactor)


def test_build_factor_dispatches_momentum(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "momentum_20", "demo")
    assert isinstance(_build_factors(cfg)[0], MomentumFactor)


def test_financial_factor_on_demo_source_raises(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "roe", "demo")
    factors = _build_factors(cfg)
    with pytest.raises(ValueError, match="cannot run on demo"):
        _maybe_enrich_financials(
            cfg, pd.DataFrame(), ["000001.SZ"], factors, logging.getLogger("test")
        )


def test_unknown_factor_name_raises(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "totally_made_up", "demo")
    with pytest.raises(ValueError, match="Unknown factor"):
        _build_factors(cfg)


def test_financial_fetch_uses_lookback_before_start(tmp_path, example_config_path, monkeypatch):
    # the financial fetch must reach BEFORE data.start so the prior disclosed
    # report can be carried forward onto early trade dates.
    from qt.pipeline import _FINANCIAL_LOOKBACK_DAYS, _maybe_enrich_financials

    cfg = _cfg(tmp_path, example_config_path, "roe", "tushare")
    captured = {}

    class _FakeFeed:
        def __init__(self, *a, **k):
            pass

        def get_fina_indicator(self, symbols, start, end, fields=None):
            captured["start"] = start
            return pd.DataFrame(
                {"symbol": [], "ann_date": [], "end_date": [], "roe": []}
            )

    monkeypatch.setattr("qt.pipeline.TushareFinancialFeed", _FakeFeed)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime([cfg.data.start]), ["000001.SZ"]], names=["date", "symbol"]
    )
    panel = pd.DataFrame({"close": [1.0]}, index=idx)
    _maybe_enrich_financials(
        cfg, panel, ["000001.SZ"], [FinancialFactor("roe")], logging.getLogger("test")
    )
    fetched = pd.Timestamp(captured["start"])
    assert fetched <= pd.Timestamp(cfg.data.start) - pd.Timedelta(
        days=_FINANCIAL_LOOKBACK_DAYS - 1
    )
