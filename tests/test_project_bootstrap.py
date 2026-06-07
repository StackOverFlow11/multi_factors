"""Slice 0: project bootstrap tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CORE_PACKAGES = [
    "data",
    "universe",
    "factors",
    "alpha",
    "portfolio",
    "runtime",
    "analytics",
    "qt",
]


@pytest.mark.parametrize("pkg", CORE_PACKAGES)
def test_import_core_packages(pkg: str) -> None:
    """Every layer package imports cleanly after the editable install."""
    assert importlib.import_module(pkg) is not None


def test_cli_module_importable() -> None:
    """The CLI module imports and exposes a callable main()."""
    cli = importlib.import_module("qt.cli")
    assert callable(cli.main)
    assert callable(cli.build_parser)


def test_validate_example_config() -> None:
    """The shipped example config parses into a RootConfig."""
    from qt.config import RootConfig, load_config

    path = Path(__file__).resolve().parents[1] / "config" / "example.yaml"
    cfg = load_config(str(path))
    assert isinstance(cfg, RootConfig)
    assert cfg.data.source == "demo"
    assert cfg.portfolio.top_n == 3
    assert cfg.factors[0].name == "momentum_20"
