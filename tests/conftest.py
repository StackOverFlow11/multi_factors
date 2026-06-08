"""Shared pytest fixtures for the framework test-suite.

Downstream agents: import these by name in your test functions, e.g.
``def test_x(demo_panel): ...``. Do not re-implement the demo data — use these.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.panel_factory import (
    SYMBOLS,
    make_demo_panel,
    make_factor_panel,
    make_scores,
)


@pytest.fixture
def demo_panel() -> pd.DataFrame:
    """A normalized market panel (5 symbols, 45 business days)."""
    return make_demo_panel()


@pytest.fixture
def factor_panel() -> pd.DataFrame:
    """A small two-column factor panel aligned to the demo panel index."""
    return make_factor_panel()


@pytest.fixture
def demo_symbols() -> list[str]:
    """The 5 demo symbols, in order."""
    return list(SYMBOLS)


@pytest.fixture
def scores_factory():
    """Callable: ``scores_factory(date) -> pd.Series`` of symbol-indexed scores."""
    return make_scores


@pytest.fixture
def example_config_path() -> str:
    """Repo-relative path to the example config (resolved from this file)."""
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "config" / "example.yaml")
