"""PR-1 all-A incremental warm: symbol resolution + config (network-free).

Locks the new default-off all-A capability:
  * ``TushareCovariatesFeed.all_a_symbols()`` returns EVERY listed symbol from the
    stock_basic snapshot (sorted / deduped / str) via the read-through cache path,
    with NO live client call on a warm snapshot;
  * ``_resolve_symbols`` takes the all-A branch only for ``universe_scope='all_a'``
    and is byte-identical to before for the default ``'config'`` scope (static /
    index), never touching the all-A path;
  * ``data_update.universe_scope`` defaults to ``'config'``, the shipped all-A
    config validates, and an unknown value is rejected.

All feeds/frames are fakes; nothing hits tushare.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache
from data.feed.tushare_covariates import TushareCovariatesFeed
from qt.config import ConfigError, DataUpdateCfg, load_config
from qt.data_updater import UpdateFeeds, _resolve_symbols, run_data_update

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_CLK = lambda: pd.Timestamp("2026-06-13 21:00:00")  # noqa: E731


class _FakeBasicPro:
    """Fake tushare client: stock_basic returns the WHOLE market (unfiltered)."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def stock_basic(self, fields=None):  # noqa: ARG002
        self.calls += 1
        return pd.DataFrame(
            {"ts_code": [r[0] for r in self.rows], "list_date": [r[1] for r in self.rows]}
        )


class _RaisingPro:
    """A client that must never be built/called (warm-path guard)."""

    def stock_basic(self, fields=None):  # noqa: ARG002
        raise AssertionError("live stock_basic call on a warm snapshot")


def _mk_cache(tmp_path):
    root = str(tmp_path / "cache")
    return TushareCache(
        CacheParquetStore(root), CoverageLedger(root), clock=_CLK, refresh_recent_days=0
    )


_ROWS = [
    ("600000.SH", "19991110"),
    ("000002.SZ", "19910129"),
    ("000001.SZ", "19910403"),
    ("688111.SH", "20190722"),
]


# --------------------------------------------------------------------------- #
# all_a_symbols
# --------------------------------------------------------------------------- #
def test_all_a_symbols_cache_path_sorted_deduped_str(tmp_path, monkeypatch):
    feed = TushareCovariatesFeed("x.json", cache=_mk_cache(tmp_path))
    pro = _FakeBasicPro(_ROWS)
    monkeypatch.setattr(feed, "_client", lambda: pro)

    out = feed.all_a_symbols()

    assert out == ["000001.SZ", "000002.SZ", "600000.SH", "688111.SH"]  # sorted, all
    assert all(isinstance(s, str) for s in out)
    assert pro.calls == 1  # one cold fetch of the global snapshot


def test_all_a_symbols_warm_makes_no_live_call(tmp_path, monkeypatch):
    # Cold feed populates the shared cache; a warm feed over the SAME cache must
    # read the cached snapshot without ever building/calling the client.
    cache = _mk_cache(tmp_path)
    cold = TushareCovariatesFeed("x.json", cache=cache)
    monkeypatch.setattr(cold, "_client", lambda: _FakeBasicPro(_ROWS))
    assert cold.all_a_symbols()  # populate

    warm = TushareCovariatesFeed("x.json", cache=cache)
    monkeypatch.setattr(warm, "_client", lambda: _RaisingPro())  # blows up if called
    out = warm.all_a_symbols()

    assert out == ["000001.SZ", "000002.SZ", "600000.SH", "688111.SH"]


def test_all_a_symbols_direct_path_dedupes(tmp_path, monkeypatch):
    # No cache => direct fetch via the same _stock_basic_fetch closure; the method
    # itself sorts/dedupes/strs even with duplicate raw rows.
    feed = TushareCovariatesFeed("x.json", cache=None)
    dup_rows = [*_ROWS, ("000002.SZ", "19910129")]
    monkeypatch.setattr(feed, "_client", lambda: _FakeBasicPro(dup_rows))

    out = feed.all_a_symbols()

    assert out == ["000001.SZ", "000002.SZ", "600000.SH", "688111.SH"]


def test_all_a_symbols_empty_snapshot(tmp_path, monkeypatch):
    feed = TushareCovariatesFeed("x.json", cache=_mk_cache(tmp_path))
    monkeypatch.setattr(feed, "_client", lambda: _FakeBasicPro([]))
    assert feed.all_a_symbols() == []


# --------------------------------------------------------------------------- #
# _resolve_symbols
# --------------------------------------------------------------------------- #
class _RecCovariates:
    def __init__(self, symbols):
        self.symbols = symbols
        self.calls = 0

    def all_a_symbols(self):
        self.calls += 1
        return list(self.symbols)


class _RecIndex:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list[str] = []

    def get_constituents(self, code, s, e):  # noqa: ARG002
        self.calls.append(code)
        return self.mapping.get(code, pd.DataFrame(columns=["symbol"]))


def _load_all_a_cfg():
    return load_config(str(_CONFIG_DIR / "data_update_all_a.yaml"))


def _load_config_scope_cfg():
    # config/data_update.yaml is universe.type=index, default scope 'config'.
    return load_config(str(_CONFIG_DIR / "data_update.yaml"))


def test_resolve_all_a_scope_uses_covariates(tmp_path):  # noqa: ARG001
    cfg = _load_all_a_cfg()
    assert cfg.data_update.universe_scope == "all_a"
    cov = _RecCovariates(["000001.SZ", "600000.SH"])
    idx = _RecIndex({})
    feeds = UpdateFeeds(covariates=cov, index=idx)

    out = _resolve_symbols(cfg, feeds, "2024-01-01", "2024-12-31")

    assert out == ["000001.SZ", "600000.SH"]
    assert cov.calls == 1
    assert idx.calls == []  # all-A never touches the config-universe index path


def test_resolve_all_a_requires_covariates_feed():
    cfg = _load_all_a_cfg()
    feeds = UpdateFeeds(covariates=None)
    with pytest.raises(ValueError, match="all_a.*covariates|covariates.*stock_basic"):
        _resolve_symbols(cfg, feeds, "2024-01-01", "2024-12-31")


def test_resolve_config_scope_index_unchanged():
    cfg = _load_config_scope_cfg()
    assert cfg.data_update.universe_scope == "config"  # default
    idx = _RecIndex(
        {"000300.SH": pd.DataFrame({"symbol": ["600000.SH", "000001.SZ", "600000.SH"]})}
    )
    # all_a_symbols would raise if the config-scope path ever called it.
    cov = _RecCovariates(None)
    cov.all_a_symbols = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("all_a path taken under config scope")
    )
    feeds = UpdateFeeds(index=idx, covariates=cov)

    out = _resolve_symbols(cfg, feeds, "2024-01-01", "2024-12-31")

    assert out == ["000001.SZ", "600000.SH"]  # sorted union, deduped (existing logic)
    assert idx.calls == ["000300.SH"]


def test_resolve_config_scope_static_unchanged(tmp_path):
    raw = yaml.safe_load(
        (_CONFIG_DIR / "data_update.yaml").read_text(encoding="utf-8")
    )
    raw["universe"] = {"type": "static", "symbols": ["600000.SH", "000001.SZ"]}
    p = tmp_path / "static.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(str(p))

    feeds = UpdateFeeds()  # static path touches no feed
    out = _resolve_symbols(cfg, feeds, "2024-01-01", "2024-12-31")

    assert out == ["600000.SH", "000001.SZ"]  # exact config order, unchanged


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_universe_scope_defaults_to_config():
    assert DataUpdateCfg().universe_scope == "config"


def test_all_a_config_validates():
    cfg = _load_all_a_cfg()
    assert cfg.data_update.universe_scope == "all_a"
    assert cfg.data_update.concurrency.max_workers == 4
    assert cfg.data_update.rate_limit_per_min == 450


def test_unknown_universe_scope_rejected(tmp_path):
    raw = yaml.safe_load(
        (_CONFIG_DIR / "data_update.yaml").read_text(encoding="utf-8")
    )
    raw["data_update"]["universe_scope"] = "the_whole_planet"
    p = tmp_path / "bad_scope.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


# --------------------------------------------------------------------------- #
# end-to-end composition: the all-A list flows into update_endpoints
# --------------------------------------------------------------------------- #
def test_run_data_update_all_a_symbols_flow_into_update_endpoints(tmp_path, monkeypatch):
    # Drive run_data_update over the SHIPPED all-A config (cache root redirected to
    # tmp so nothing touches the repo artifacts dir), with the feed layer and
    # update_endpoints faked — proving the all-A resolution composes into the warm
    # call. Network-free: no real feed is built, no client is constructed.
    raw = yaml.safe_load(
        (_CONFIG_DIR / "data_update_all_a.yaml").read_text(encoding="utf-8")
    )
    raw["data"]["cache"]["root_dir"] = str(tmp_path / "cache")
    cfg_path = tmp_path / "all_a.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    all_a = ["000001.SZ", "000002.SZ", "600000.SH", "688111.SH"]
    fake_feeds = UpdateFeeds(covariates=_RecCovariates(all_a))
    monkeypatch.setattr(
        "qt.data_updater._build_feeds", lambda *a, **k: fake_feeds
    )

    captured: dict[str, list[str]] = {}

    def _fake_update_endpoints(cache, feeds, symbols, **kwargs):  # noqa: ARG001
        captured["symbols"] = list(symbols)
        return {}

    monkeypatch.setattr(
        "qt.data_updater.update_endpoints", _fake_update_endpoints
    )

    result = run_data_update(str(cfg_path), today="2024-12-31")

    assert captured["symbols"] == all_a  # the resolved all-A list reached the warm
    assert result.symbols == all_a
    assert fake_feeds.covariates.calls == 1  # resolved via all_a_symbols(), once
