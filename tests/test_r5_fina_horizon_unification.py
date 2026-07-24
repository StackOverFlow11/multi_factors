"""R5: the fina revision horizon is a SINGLE source, threaded to both builders.

Factor-refactor step D3, commit 1 (design v3.2 §1.3 note / §3.3 R5). Before this
fix the backtest read-through cache (``qt.pipeline._build_cache``) passed NO
``recent_tail_overrides`` at all, so the fina revision horizon silently collapsed
to ``refresh_recent_days`` (14) — while the data-update warm used 400. The two
paths could disagree; the factor store's incremental overlap window (D3) is sized
from this horizon, so a wrong value is a correctness bug, not a perf nit.

The fix moves the knob to its correct home ``data.cache.fina_tail_days`` (a
property of the cache/endpoint, R5), makes BOTH cache builders read it from
there, and removes the drift-prone ``data_update.fina_tail_days`` knob. These
tests pin: the value reaches both builders, a config bump follows through
(mutation), the old knob is rejected, and the value flows into the D0
``revision_horizon`` the D3 tail-recompute engine consumes.

Network-free: only cache OBJECTS are constructed (no fetch, no I/O beyond the
temp root); ``_recent_tail_overrides`` is read off the constructed cache.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.availability_policy import FINA_INDICATOR, revision_horizon
from qt.config import CacheCfg, DataUpdateCfg, load_config
from qt.data_updater import _build_caches
from qt.pipeline import _build_cache

_DATA_UPDATE_CONFIG = "config/data_update.yaml"


def _cfg_with_cache_root(config_path: str, tmp_root: str):
    """Load a config and point its cache root at a temp dir (hermetic)."""
    cfg = load_config(config_path)
    new_data = cfg.data.model_copy(
        update={"cache": cfg.data.cache.model_copy(update={"root_dir": str(tmp_root)})}
    )
    return cfg.model_copy(update={"data": new_data})


def _cfg_with_fina_tail(cfg, value: int):
    new_data = cfg.data.model_copy(
        update={"cache": cfg.data.cache.model_copy(update={"fina_tail_days": value})}
    )
    return cfg.model_copy(update={"data": new_data})


# -- the single source of truth exists and defaults to 400 --------------------


def test_fina_tail_days_lives_on_data_cache_and_defaults_to_400():
    assert CacheCfg().fina_tail_days == 400


def test_data_update_config_no_longer_accepts_fina_tail_days():
    # The drift-prone data_update knob is removed; declaring it is a config error
    # (extra=forbid). This is what stops a second, independently-drifting source.
    with pytest.raises(Exception):  # pydantic.ValidationError
        DataUpdateCfg(fina_tail_days=400)


def test_negative_fina_tail_days_is_a_readable_error():
    with pytest.raises(Exception):
        CacheCfg(fina_tail_days=-1)


# -- the value reaches the BACKTEST read-through builder (the R5 defect) -------


def test_build_cache_passes_the_fina_override(tmp_path):
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    cache = _build_cache(cfg)
    assert cache is not None  # data.cache.enabled is true in the data_update config
    # The backtest read-through now carries the fina override (used to be absent).
    assert cache._recent_tail_overrides.get(FINA_INDICATOR) == 400


def test_build_cache_fina_override_follows_the_config_bump(tmp_path):
    # Mutation: raise data.cache.fina_tail_days -> the value threaded to the cache
    # MUST follow (proves it is read from config, not hardcoded).
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    cfg = _cfg_with_fina_tail(cfg, 512)
    cache = _build_cache(cfg)
    assert cache._recent_tail_overrides.get(FINA_INDICATOR) == 512


# -- BOTH builders resolve the SAME single source (no drift) ------------------


def test_data_update_builder_reads_the_same_single_source(tmp_path):
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    today = pd.Timestamp("2025-01-02")
    cache, _intraday = _build_caches(cfg, cfg.data_update, today)
    assert cache._recent_tail_overrides.get(FINA_INDICATOR) == 400


def test_data_update_builder_override_follows_the_config_bump(tmp_path):
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    cfg = _cfg_with_fina_tail(cfg, 333)
    today = pd.Timestamp("2025-01-02")
    cache, _intraday = _build_caches(cfg, cfg.data_update, today)
    assert cache._recent_tail_overrides.get(FINA_INDICATOR) == 333


def test_both_builders_agree_on_the_fina_horizon(tmp_path):
    # The whole point of R5: with one source, the two builders can never disagree.
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    cfg = _cfg_with_fina_tail(cfg, 400)
    today = pd.Timestamp("2025-01-02")
    backtest_cache = _build_cache(cfg)
    warm_cache, _ = _build_caches(cfg, cfg.data_update, today)
    assert (
        backtest_cache._recent_tail_overrides.get(FINA_INDICATOR)
        == warm_cache._recent_tail_overrides.get(FINA_INDICATOR)
        == 400
    )


# -- the value flows into the D0 revision_horizon (the D3 consumer) -----------


def test_config_value_feeds_the_d0_revision_horizon(tmp_path):
    # The D3 tail-recompute engine sizes its overlap window from
    # data.availability_policy.revision_horizon, fed the SAME cache-config values.
    cfg = _cfg_with_cache_root(_DATA_UPDATE_CONFIG, tmp_path)
    cfg = _cfg_with_fina_tail(cfg, 400)
    overrides = {FINA_INDICATOR: cfg.data.cache.fina_tail_days}
    horizon = revision_horizon(
        FINA_INDICATOR,
        refresh_recent_days=cfg.data.cache.refresh_recent_days,
        recent_tail_overrides=overrides,
    )
    assert horizon == 400
