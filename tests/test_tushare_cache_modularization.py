"""D2 regression guard: the TushareCache endpoint specs/parsers split must keep
the public facade imports working and re-export the same objects.

These are narrow compatibility tests — the cache *behaviour* is locked by
tests/test_tushare_cache_{market,universe,p4_3}.py and tests/test_data_updater.py.
Here we only assert the module boundary (specs/parsers/planning) and the
backward-compatible re-exports from data.cache.tushare_cache.
"""

from __future__ import annotations

import pandas as pd


def test_backward_compatible_imports_from_tushare_cache():
    """Downstream code imports these names straight from tushare_cache; keep them."""
    from data.cache.tushare_cache import (  # noqa: F401
        ALL_ENDPOINTS,
        DAILY_BASIC,
        FINA_FIELDS,
        FINA_INDICATOR,
        INDEX_MEMBER_ALL,
        INDEX_WEIGHT,
        NAMECHANGE,
        STK_LIMIT,
        STOCK_BASIC,
        SUSPEND_D,
        TushareCache,
    )

    assert TushareCache.__name__ == "TushareCache"
    assert len(ALL_ENDPOINTS) == 10
    assert FINA_FIELDS == ("roe", "netprofit_yoy", "grossprofit_margin")


def test_facade_reexports_are_the_same_objects_as_specs():
    """The facade re-exports the spec constants, not copies."""
    from data.cache import tushare_cache as facade
    from data.cache import tushare_specs as specs

    for name in (
        "MARKET_DAILY", "ADJ_FACTOR", "INDEX_WEIGHT", "SUSPEND_D", "NAMECHANGE",
        "STK_LIMIT", "STOCK_BASIC", "DAILY_BASIC", "FINA_INDICATOR",
        "INDEX_MEMBER_ALL", "ALL_ENDPOINTS", "FINA_FIELDS",
    ):
        assert getattr(facade, name) is getattr(specs, name), name


def test_parsers_live_in_parser_module_and_facade_uses_them():
    """The facade imports the parser functions from the parser module (identity)."""
    from data.cache import tushare_cache as facade
    from data.cache import tushare_parsers as parsers

    for name in (
        "_parse_daily", "_parse_adj", "_parse_suspend", "_parse_stk_limit",
        "_parse_index_weight", "_parse_namechange", "_parse_stock_basic",
        "_parse_daily_basic", "_parse_fina", "_parse_index_member",
    ):
        assert getattr(facade, name) is getattr(parsers, name), name


def test_planning_helpers_split_out():
    from data.cache.tushare_planning import _compact, _fields_hash

    assert _compact(pd.Timestamp("2024-01-03")) == "20240103"
    # order-independent stable hash
    assert _fields_hash(["b", "a"]) == _fields_hash(["a", "b"])


def test_parse_fina_returns_superset_schema():
    """A representative parser still yields the FINA_FIELDS superset + empty schema."""
    from data.cache.tushare_parsers import _parse_fina
    from data.cache.tushare_specs import FINA_FIELDS

    empty = _parse_fina(None)
    assert list(empty.columns) == ["date", "symbol", "ann_date", "end_date", *FINA_FIELDS]
    assert len(empty) == 0

    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240420"],
            "end_date": ["20231231"],
            "roe": [12.3],
            # netprofit_yoy / grossprofit_margin missing -> filled with NaN
        }
    )
    out = _parse_fina(raw)
    assert list(out.columns) == ["date", "symbol", "ann_date", "end_date", *FINA_FIELDS]
    assert out.loc[0, "symbol"] == "000001.SZ"
    assert out.loc[0, "ann_date"] == "20240420"
    assert pd.isna(out.loc[0, "netprofit_yoy"])
