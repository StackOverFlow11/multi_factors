"""Slice 1: config system tests (backlog section 4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from qt.config import ConfigError, RootConfig, load_config


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


_BASE = """\
project:
  name: test
data:
  source: {source}
  freq: D
{start_line}
{end_line}
universe:
  type: static
  symbols: ["000001.SZ", "000002.SZ"]
factors:
  - name: momentum_20
    enabled: true
    params: {{window: 20}}
alpha:
  model: equal_weight
  params: {{}}
portfolio:
  constructor: topn_equal_weight
  top_n: {top_n}
  long_only: true
backtest:
  rebalance: monthly
cost:
  fee_rate: 0.001
output:
  root_dir: artifacts
"""


def _cfg_text(
    source: str = "demo",
    start: str | None = "2024-01-01",
    end: str | None = "2024-12-31",
    top_n: int = 3,
) -> str:
    start_line = f'  start: "{start}"' if start is not None else ""
    end_line = f'  end: "{end}"' if end is not None else ""
    return _BASE.format(
        source=source, start_line=start_line, end_line=end_line, top_n=top_n
    )


def test_config_requires_start_and_end(tmp_path: Path) -> None:
    path = _write(tmp_path, _cfg_text(start=None, end=None))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    msg = str(exc.value).lower()
    assert "start" in msg or "end" in msg


def test_config_rejects_start_after_end(tmp_path: Path) -> None:
    path = _write(tmp_path, _cfg_text(start="2024-12-31", end="2024-01-01"))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "start" in str(exc.value).lower()


def test_config_rejects_invalid_top_n(tmp_path: Path) -> None:
    path = _write(tmp_path, _cfg_text(top_n=0))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "top_n" in str(exc.value).lower()


def test_config_accepts_demo_data_source(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _cfg_text(source="demo")))
    assert isinstance(cfg, RootConfig)
    assert cfg.data.source == "demo"


def test_config_accepts_tushare_data_source(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _cfg_text(source="tushare")))
    assert cfg.data.source == "tushare"
