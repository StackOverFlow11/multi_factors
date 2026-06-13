"""P4-1: cache config validation (network-free).

Locks the cache config contract:
  * ``data.cache`` defaults to disabled (backward compatible — an existing
    config runs exactly as before);
  * the four configured knobs exist with the documented defaults;
  * bad values (negative refresh window, empty root) fail readably;
  * every existing config still validates with the new (defaulted) section.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qt.config import ConfigError, DataCfg, load_config

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_cache_defaults_disabled_and_backward_compatible():
    # A DataCfg built without a cache section gets the disabled default.
    cfg = DataCfg(start="2024-01-01", end="2024-06-30")
    assert cfg.cache.enabled is False
    assert cfg.cache.root_dir == "artifacts/cache/tushare/v1"
    assert cfg.cache.refresh_recent_days == 14
    assert cfg.cache.force_refresh == []


def test_cache_enabled_with_overrides(tmp_path):
    raw = yaml.safe_load((_CONFIG_DIR / "example.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache"] = {
        "enabled": True,
        "root_dir": "artifacts/cache/tushare/v1",
        "refresh_recent_days": 7,
        "force_refresh": ["market_daily", "adj_factor"],
    }
    p = tmp_path / "cache.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.data.cache.enabled is True
    assert cfg.data.cache.refresh_recent_days == 7
    assert cfg.data.cache.force_refresh == ["market_daily", "adj_factor"]


def test_cache_rejects_negative_refresh_window(tmp_path):
    raw = yaml.safe_load((_CONFIG_DIR / "example.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache"] = {"enabled": True, "refresh_recent_days": -1}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="refresh_recent_days"):
        load_config(str(p))


def test_cache_rejects_empty_root_dir(tmp_path):
    raw = yaml.safe_load((_CONFIG_DIR / "example.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache"] = {"enabled": True, "root_dir": "  "}
    p = tmp_path / "bad2.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="root_dir"):
        load_config(str(p))


def test_cache_rejects_unknown_key(tmp_path):
    raw = yaml.safe_load((_CONFIG_DIR / "example.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache"] = {"enabled": True, "ttl_days": 3}  # not a field
    p = tmp_path / "bad3.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


@pytest.mark.parametrize("cfg_path", sorted(_CONFIG_DIR.glob("*.yaml")))
def test_all_existing_configs_still_validate(cfg_path):
    cfg = load_config(str(cfg_path))
    # the cache section is present (defaulted) and disabled unless opted in.
    assert cfg.data.cache.enabled in (True, False)
