"""Real-cache eval provider wiring (D5 C4, commit 1).

Network-free. Pins: the CacheMinuteProvider MOVE (single source — the hotpath
smoke and the probes import it from ``qt.factor_eval_providers`` now), the
provider's cache-only/zero-live-call behavior against a real on-disk intraday
store, the DailyEvalPanelProvider's close-view window/symbol slicing, and
``build_eval_service``'s call order + bundle contents (pipeline helpers
monkeypatched, so no tushare client is ever built).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

import qt.factor_eval_providers as fep
import qt.factor_hotpath_smoke as hotpath_smoke
import qt.panel_leg_probe as panel_leg_probe
import qt.saturation_probe as saturation_probe
from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_parquet_store import KEY_COLS, IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.materialize import MaterializeSources
from factors.store import FactorValueStore, StoreKey
from qt.factor_eval_providers import (
    CACHE_MINUTE_DATA_START,
    DEFAULT_STORE_ROOT,
    CacheMinuteProvider,
    DailyEvalPanelProvider,
    EvalServiceBundle,
    build_eval_service,
)


# --------------------------------------------------------------------------- #
# the MOVE: one source for the cache minute provider
# --------------------------------------------------------------------------- #
def test_cache_minute_provider_has_a_single_source():
    assert hotpath_smoke.CacheMinuteProvider is fep.CacheMinuteProvider
    assert saturation_probe.CacheMinuteProvider is fep.CacheMinuteProvider
    assert panel_leg_probe.CacheMinuteProvider is fep.CacheMinuteProvider
    assert saturation_probe.CACHE_MINUTE_DATA_START is fep.CACHE_MINUTE_DATA_START


def test_earliest_available_is_the_declared_floor():
    provider = CacheMinuteProvider("does/not/exist")
    assert provider.earliest_available(["AAA", "BBB"]) == pd.Timestamp("2015-01-05")
    assert CACHE_MINUTE_DATA_START == "2015-01-05"


# --------------------------------------------------------------------------- #
# CacheMinuteProvider against a REAL on-disk intraday store (cache-only)
# --------------------------------------------------------------------------- #
def _write_bars(root, symbol, rows):
    """rows: list of (bar_end, close). Upsert raw 1min bars the store's way."""
    store = IntradayParquetStore(str(root))
    frame = pd.DataFrame(
        {
            "symbol": [symbol] * len(rows),
            "bar_end": [pd.Timestamp(r[0]) for r in rows],
            "source_trade_time": [pd.Timestamp(r[0]) for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[1] for r in rows],
            "volume": [100.0] * len(rows),
            "amount": [1000.0] * len(rows),
            "freq": [RAW_INTRADAY_FREQ] * len(rows),
        }
    )
    store.upsert(INTRADAY_ENDPOINT, symbol, RAW_INTRADAY_FREQ, frame, list(KEY_COLS))


def test_minute_bars_reads_normalized_bars_with_zero_live_calls(tmp_path):
    _write_bars(
        tmp_path,
        "000001.SZ",
        [("2024-01-02 09:31:00", 11.0), ("2024-01-02 09:32:00", 12.0)],
    )
    _write_bars(tmp_path, "000002.SZ", [("2024-01-02 09:31:00", 21.0)])
    provider = CacheMinuteProvider(str(tmp_path))
    bars = provider.minute_bars(
        ["000001.SZ", "000002.SZ"], "2024-01-02 09:00:00", "2024-01-02 10:00:00"
    )
    assert provider.calls == 1
    assert provider.live_calls == 0  # read_range has no fetch closure
    assert len(bars) == 3
    assert bars.index.names == ["time", "symbol"]
    times = bars.index.get_level_values("time")
    assert (times == times.normalize()).sum() == 0  # minute precision kept
    closes = bars["close"].astype(float).tolist()
    assert closes == [11.0, 21.0, 12.0]  # sorted by (time, symbol)


def test_minute_bars_empty_and_missing_are_empty_never_live(tmp_path):
    provider = CacheMinuteProvider(str(tmp_path))
    empty = provider.minute_bars([], "2024-01-02", "2024-01-03")
    assert empty.empty
    missing = provider.minute_bars(["NOPE.SZ"], "2024-01-02", "2024-01-03")
    assert missing.empty  # an absent month partition is an empty read, not a fetch
    assert provider.live_calls == 0


# --------------------------------------------------------------------------- #
# DailyEvalPanelProvider: close-view, un-lagged, window/symbol slicing
# --------------------------------------------------------------------------- #
def _panel():
    dates = pd.bdate_range("2024-01-01", periods=6)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=[DATE_LEVEL, SYMBOL_LEVEL])
    return pd.DataFrame(
        {
            "close": [float(i) for i in range(len(idx))],
            "open": [float(i) + 100.0 for i in range(len(idx))],
        },
        index=idx,
    )


def test_daily_panel_slices_window_and_symbols_without_lagging():
    provider = DailyEvalPanelProvider(_panel())
    dates = pd.bdate_range("2024-01-01", periods=6)
    out = provider.daily_panel(["AAA"], dates[2], dates[4])
    assert out.index.get_level_values(SYMBOL_LEVEL).unique().tolist() == ["AAA"]
    got_dates = pd.DatetimeIndex(pd.unique(out.index.get_level_values(DATE_LEVEL)))
    assert got_dates.equals(dates[2:5])  # both bounds inclusive
    # close-view, NOT lagged: the value dated d is the panel's own close at d.
    full = _panel()
    for d in dates[2:5]:
        assert out.loc[(d, "AAA"), "close"] == full.loc[(d, "AAA"), "close"]


def test_daily_panel_unknown_symbols_and_empty_panel():
    provider = DailyEvalPanelProvider(_panel())
    assert provider.daily_panel(["NOPE"], "2024-01-01", "2024-01-31").empty
    empty_provider = DailyEvalPanelProvider(pd.DataFrame())
    assert empty_provider.daily_panel(["AAA"], "2024-01-01", "2024-01-31").empty


# --------------------------------------------------------------------------- #
# build_eval_service: legacy call order + bundle contents (helpers faked)
# --------------------------------------------------------------------------- #
def _fake_cfg():
    return SimpleNamespace(data=SimpleNamespace(cache=SimpleNamespace(root_dir="cache/root")))


def test_build_eval_service_calls_pipeline_helpers_in_legacy_order(tmp_path, monkeypatch):
    calls: list[str] = []
    sentinel_cache = object()
    panel = _panel()

    def rec(name, ret=None):
        def _fn(*args, **kwargs):
            calls.append(name)
            return ret

        return _fn

    monkeypatch.setattr(fep, "_build_cache", rec("_build_cache", sentinel_cache))
    monkeypatch.setattr(fep, "_build_universe", rec("_build_universe", (object(), ["AAA", "BBB"])))
    monkeypatch.setattr(fep, "_load_panel", rec("_load_panel", panel))
    monkeypatch.setattr(fep, "_maybe_enrich_value", rec("_maybe_enrich_value", panel))
    monkeypatch.setattr(fep, "_maybe_enrich_covariates", rec("_maybe_enrich_covariates", panel))
    monkeypatch.setattr(fep, "_log_run_cache_stats", rec("_log_run_cache_stats"))

    logger = logging.getLogger("test.build_eval_service")
    bundle = build_eval_service(
        _fake_cfg(), logger, value_factors=["vf1"], store_root=str(tmp_path / "store")
    )

    assert calls == [
        "_build_cache",
        "_build_universe",
        "_load_panel",
        "_maybe_enrich_value",
        "_maybe_enrich_covariates",
        "_log_run_cache_stats",
    ]
    assert isinstance(bundle, EvalServiceBundle)
    assert bundle.cache is sentinel_cache
    assert bundle.symbols == ["AAA", "BBB"]
    assert bundle.panel is panel
    assert isinstance(bundle.store, FactorValueStore)
    assert isinstance(bundle.sources, MaterializeSources)
    assert isinstance(bundle.sources.daily, DailyEvalPanelProvider)
    assert isinstance(bundle.sources.minute, CacheMinuteProvider)
    # the minute provider reads the configured cache root; the store root is the
    # caller's (design §3.4 R22 default is artifacts/factor_store).
    key = StoreKey(factor_id="f", params_hash="p", code_hash="c", view="decision")
    assert str(bundle.store.path(key)).startswith(str(tmp_path / "store"))


def test_build_eval_service_default_store_root_is_the_r22_artifacts_root():
    assert DEFAULT_STORE_ROOT == "artifacts/factor_store"


def test_build_eval_service_threads_value_factors_and_cache_root(monkeypatch):
    seen: dict = {}

    def _enrich(cfg, panel, symbols, factors, logger, cache):
        seen["value_factors"] = list(factors)
        return panel

    monkeypatch.setattr(fep, "_build_cache", lambda cfg: None)
    monkeypatch.setattr(fep, "_build_universe", lambda cfg, logger, cache: (None, ["AAA"]))
    monkeypatch.setattr(fep, "_load_panel", lambda cfg, symbols, logger, cache: _panel())
    monkeypatch.setattr(fep, "_maybe_enrich_value", _enrich)
    monkeypatch.setattr(fep, "_maybe_enrich_covariates", lambda cfg, panel, s, logger, cache: panel)
    monkeypatch.setattr(fep, "_log_run_cache_stats", lambda cache, logger: None)

    bundle = build_eval_service(
        _fake_cfg(), logging.getLogger("test.build_eval_service2"), value_factors=["v"]
    )
    assert seen["value_factors"] == ["v"]
    # cache disabled (None) still wires the cache-only minute provider.
    assert bundle.cache is None
    assert isinstance(bundle.sources.minute, CacheMinuteProvider)
