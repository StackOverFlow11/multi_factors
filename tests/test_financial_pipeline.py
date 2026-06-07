"""Pipeline dispatch for financial factors + the demo-source guard (no fabrication)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml

from factors.compute.financial import FinancialFactor
from factors.compute.momentum import MomentumFactor
from qt.config import load_config
from qt.pipeline import _build_factor, _maybe_enrich_financials


def _cfg(tmp_path, example_config_path, factor_name, source):
    base = yaml.safe_load(open(example_config_path, encoding="utf-8"))
    base["factors"] = [{"name": factor_name, "enabled": True, "params": {}}]
    base["data"]["source"] = source
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return load_config(str(path))


def test_build_factor_dispatches_financial(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "roe", "tushare")
    assert isinstance(_build_factor(cfg), FinancialFactor)


def test_build_factor_dispatches_momentum(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "momentum_20", "demo")
    assert isinstance(_build_factor(cfg), MomentumFactor)


def test_financial_factor_on_demo_source_raises(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "roe", "demo")
    factor = _build_factor(cfg)
    with pytest.raises(ValueError, match="cannot run on demo"):
        _maybe_enrich_financials(
            cfg, pd.DataFrame(), ["000001.SZ"], factor, logging.getLogger("test")
        )


def test_unknown_factor_name_raises(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, "totally_made_up", "demo")
    with pytest.raises(ValueError, match="Unknown factor"):
        _build_factor(cfg)
