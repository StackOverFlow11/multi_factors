"""P3-1: multi-factor pipeline (multiple enabled factors, network-free).

Locks the P3-1 contract:
  * every ENABLED factor is instantiated (config order) and lands as its own
    factor-panel column;
  * financial fields are fetched ONCE for all financial factors and as-of
    aligned by ann_date in a single pass (no per-factor refetch);
  * demo + financial factor still fails readably (no fabricated financials);
  * single-factor configs keep their old behaviour (primary == only factor);
  * the result exposes per-factor + combo-score analytics and the report
    discloses the active factor list without leaking any secret.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from factors.compute.financial import FinancialFactor
from factors.compute.momentum import MomentumFactor
from qt.config import load_config
from qt.pipeline import _build_factors, _maybe_enrich_financials, run_phase0


def _write_cfg(tmp_path: Path, example_config_path: str, factors: list[dict],
               source: str = "demo", name: str = "cfg.yaml") -> Path:
    """Copy the example config, swap the factor list, redirect outputs to tmp."""
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    raw["data"]["source"] = source
    raw["data"]["start"] = "2024-01-01"
    raw["data"]["end"] = "2024-06-30"
    raw["factors"] = factors
    raw["output"] = {
        "root_dir": str(out),
        "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"),
        "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"),
        "overwrite": True,
    }
    cfg_path = tmp_path / name
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_path


_TWO_MOMENTUM = [
    {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
    {"name": "momentum_5", "enabled": True, "params": {"window": 5}},
]


# --------------------------------------------------------------------------- #
# _build_factors
# --------------------------------------------------------------------------- #
def test_build_factors_returns_all_enabled_in_config_order(tmp_path, example_config_path):
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, _TWO_MOMENTUM)))
    factors = _build_factors(cfg)
    assert [f.name for f in factors] == ["momentum_20", "momentum_5"]
    assert all(isinstance(f, MomentumFactor) for f in factors)


def test_build_factors_skips_disabled(tmp_path, example_config_path):
    specs = [
        {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
        {"name": "momentum_5", "enabled": False, "params": {"window": 5}},
    ]
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, specs)))
    assert [f.name for f in _build_factors(cfg)] == ["momentum_20"]


def test_build_factors_mixed_price_and_financial(tmp_path, example_config_path):
    specs = _TWO_MOMENTUM + [
        {"name": "roe", "enabled": True, "params": {}},
        {"name": "netprofit_yoy", "enabled": True, "params": {}},
    ]
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, specs, source="tushare")))
    factors = _build_factors(cfg)
    assert [f.name for f in factors] == ["momentum_20", "momentum_5", "roe", "netprofit_yoy"]
    assert isinstance(factors[2], FinancialFactor)
    assert isinstance(factors[3], FinancialFactor)


def test_build_factors_duplicate_names_raise(tmp_path, example_config_path):
    specs = [
        {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
        {"name": "momentum_x", "enabled": True, "params": {"window": 20}},  # same name
    ]
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, specs)))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _build_factors(cfg)


def test_build_factors_none_enabled_raises(tmp_path, example_config_path):
    specs = [{"name": "momentum_20", "enabled": False, "params": {}}]
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, specs)))
    with pytest.raises(ValueError, match="[Nn]o enabled factor"):
        _build_factors(cfg)


# --------------------------------------------------------------------------- #
# batched financial enrichment (single fetch + single as-of pass)
# --------------------------------------------------------------------------- #
def test_financials_fetched_once_for_all_fields(tmp_path, example_config_path, monkeypatch):
    specs = [
        {"name": "roe", "enabled": True, "params": {}},
        {"name": "netprofit_yoy", "enabled": True, "params": {}},
    ]
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, specs, source="tushare")))
    calls: list[list[str]] = []

    class _FakeFeed:
        def __init__(self, *a, **k):
            pass

        def get_fina_indicator(self, symbols, start, end, fields=None):
            calls.append(list(fields or []))
            return pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "ann_date": ["20240110"],
                    "end_date": ["20231231"],
                    "roe": [5.0],
                    "netprofit_yoy": [10.0],
                }
            )

    monkeypatch.setattr("qt.pipeline.TushareFinancialFeed", _FakeFeed)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-05", "2024-01-15"]), ["000001.SZ"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({"close": [1.0, 1.1]}, index=idx)
    factors = [FinancialFactor("roe"), FinancialFactor("netprofit_yoy")]
    enriched = _maybe_enrich_financials(
        cfg, panel, ["000001.SZ"], factors, logging.getLogger("test")
    )

    # ONE fetch for BOTH fields (no per-factor refetch).
    assert len(calls) == 1
    assert set(calls[0]) == {"roe", "netprofit_yoy"}
    # both columns as-of aligned: invisible before ann_date, visible after.
    assert math.isnan(enriched.loc[(pd.Timestamp("2024-01-05"), "000001.SZ"), "roe"])
    assert enriched.loc[(pd.Timestamp("2024-01-15"), "000001.SZ"), "roe"] == 5.0
    assert enriched.loc[(pd.Timestamp("2024-01-15"), "000001.SZ"), "netprofit_yoy"] == 10.0
    # the input panel is never mutated (immutability).
    assert "roe" not in panel.columns


def test_demo_with_financial_factor_raises_readable(tmp_path, example_config_path):
    cfg = load_config(str(_write_cfg(
        tmp_path, example_config_path,
        _TWO_MOMENTUM + [{"name": "roe", "enabled": True, "params": {}}],
        source="demo",
    )))
    factors = _build_factors(cfg)
    with pytest.raises(ValueError, match="cannot run on demo"):
        _maybe_enrich_financials(
            cfg, pd.DataFrame(), ["000001.SZ"], factors, logging.getLogger("test")
        )


# --------------------------------------------------------------------------- #
# end-to-end demo run with two price factors
# --------------------------------------------------------------------------- #
def test_phase0_multifactor_panel_and_analytics(tmp_path, example_config_path):
    cfg_path = _write_cfg(tmp_path, example_config_path, _TWO_MOMENTUM)
    result = run_phase0(str(cfg_path))

    # every enabled factor is an own column in the persisted factor panel.
    stored = pd.read_parquet(result.factor_path)
    assert {"momentum_20", "momentum_5"} <= set(stored.columns)

    # active factor list + per-factor + combo analytics on the result.
    assert result.factor_names == ("momentum_20", "momentum_5")
    assert result.factor_name == "momentum_20"  # primary = first enabled
    assert set(result.per_factor.keys()) == {"momentum_20", "momentum_5"}
    for metrics in result.per_factor.values():
        assert math.isfinite(metrics["ic_mean"])
        assert 0.0 <= metrics["coverage"] <= 1.0
    assert math.isfinite(result.combo_analytics["ic_mean"])

    # top-level (legacy) metrics == the PRIMARY factor's metrics.
    assert result.ic_mean == result.per_factor["momentum_20"]["ic_mean"]
    assert result.ic_ir == result.per_factor["momentum_20"]["ic_ir"]


def test_phase0_multifactor_report_lists_factors_and_combo(tmp_path, example_config_path):
    cfg_path = _write_cfg(tmp_path, example_config_path, _TWO_MOMENTUM)
    result = run_phase0(str(cfg_path))
    text = result.report_path.read_text(encoding="utf-8")
    assert "momentum_20" in text and "momentum_5" in text  # active factor list
    assert "combo" in text.lower()  # combo score diagnostics shown
    assert "token" not in text.lower()  # no secret leak


def test_phase0_single_factor_keeps_legacy_shape(tmp_path, example_config_path):
    """A single-factor config behaves exactly as before (primary == only factor)."""
    specs = [{"name": "momentum_20", "enabled": True,
              "params": {"window": 20, "price_col": "close"}}]
    cfg_path = _write_cfg(tmp_path, example_config_path, specs)
    result = run_phase0(str(cfg_path))
    assert result.factor_names == ("momentum_20",)
    assert result.factor_name == "momentum_20"
    assert result.per_factor["momentum_20"]["ic_mean"] == result.ic_mean
    stored = pd.read_parquet(result.factor_path)
    assert "momentum_20" in stored.columns
