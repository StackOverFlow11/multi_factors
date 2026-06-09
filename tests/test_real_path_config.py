"""The tushare real-data path config loads and is disclosed as REAL (not demo)."""

from __future__ import annotations

from pathlib import Path

from qt.config import load_config
from qt.pipeline import _collect_downgrades


def _real_path(example_config_path):
    return str(Path(example_config_path).parent / "example_tushare.yaml")


def test_real_path_config_loads(example_config_path):
    cfg = load_config(_real_path(example_config_path))
    assert cfg.data.source == "tushare"
    assert cfg.universe.type == "index"
    assert cfg.universe.index_code == "000300.SH"
    assert cfg.processing.neutralize.enabled is True


def test_downgrades_mark_real_path(example_config_path):
    items = _collect_downgrades(load_config(_real_path(example_config_path)))
    assert items[0].startswith("DATA PATH = REAL tushare")
    assert any("PIT index membership" in x for x in items)
    assert any("neutralized" in x for x in items)


def test_downgrades_industry_is_pit_not_current(example_config_path):
    # P2-3: the real-path neutralization downgrade must disclose POINT-IN-TIME SW
    # industry (UNI-010) and must NOT claim the industry is the CURRENT stock_basic tag.
    items = _collect_downgrades(load_config(_real_path(example_config_path)))
    text = " ".join(items)
    assert "point-in-time" in text and "UNI-010" in text
    assert "the industry tag is the CURRENT" not in text


def test_downgrades_mark_demo_path(example_config_path):
    items = _collect_downgrades(load_config(example_config_path))
    assert items[0].startswith("DATA PATH = DEMO")
    assert any("Static universe" in x for x in items)
    assert any("No neutralization" in x for x in items)
