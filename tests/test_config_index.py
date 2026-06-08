"""Config support for the PIT index universe (universe.type == 'index')."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qt.config import UniverseCfg


def test_index_type_requires_index_code():
    with pytest.raises(ValidationError, match="index_code"):
        UniverseCfg(type="index")


def test_index_type_with_code_is_valid():
    cfg = UniverseCfg(type="index", index_code="000300.SH")
    assert cfg.type == "index"
    assert cfg.index_code == "000300.SH"


def test_static_type_is_default_and_needs_no_index_code():
    cfg = UniverseCfg(symbols=["000001.SZ"])
    assert cfg.type == "static"
    assert cfg.index_code is None
