"""PR-2 historical backfill: chunking / failure-tolerance / minute-window /
resumability / all-A universe / config (network-free, FAKE feeds).

Everything here is a fake — no test hits tushare or reads the real token. The
recorder-style tests fake ``_build_feeds`` + ``update_endpoints`` to capture the
per-batch symbol partitions and windows; the resumability test drives a REAL
read-through cache with a fake tushare client so the coverage ledger genuinely
skips already-covered gaps on the second run.

The incremental ``data-update`` path is proven UNCHANGED by the existing
``test_data_updater`` / ``test_data_update_all_a`` / ``test_data_update_concurrency``
suites passing unaltered (they exercise ``run_data_update`` /
``update_endpoints`` / ``_build_feeds`` / ``_resolve_symbols``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from data.feed.tushare_feed import TushareFeed
from qt.config import BackfillCfg, ConfigError, DataUpdateCfg, load_config
from qt.data_backfill import BackfillResult, _chunk, format_summary, run_data_backfill
from qt.data_updater import UpdateFeeds

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_backfill_cfg_defaults():
    bf = BackfillCfg()
    assert bf.start == "2020-01-01"
    assert bf.chunk_size == 300
    assert bf.include_minute is True


def test_data_update_cfg_has_backfill_default():
    # A DataUpdateCfg built with no backfill block gets the all-defaults sub-config.
    du = DataUpdateCfg()
    assert du.backfill.chunk_size == 300
    assert du.backfill.start == "2020-01-01"
    assert du.backfill.include_minute is True


def test_existing_data_update_yaml_gets_default_backfill():
    # config/data_update.yaml has NO backfill block -> defaults, still validates.
    cfg = load_config(str(_CONFIG_DIR / "data_update.yaml"))
    assert cfg.data_update is not None
    assert cfg.data_update.backfill.chunk_size == 300


def test_all_a_config_backfill_validates():
    cfg = load_config(str(_CONFIG_DIR / "data_update_all_a.yaml"))
    bf = cfg.data_update.backfill
    assert bf.start == "2020-01-01"
    assert bf.chunk_size == 300
    assert bf.include_minute is True


@pytest.mark.parametrize("bad", [0, -1, -50])
def test_bad_chunk_size_rejected(bad):
    with pytest.raises(ValidationError):
        BackfillCfg(chunk_size=bad)


@pytest.mark.parametrize("bad", ["not-a-date", "2024-13-01", "20240101"])
def test_bad_start_rejected(bad):
    with pytest.raises(ValidationError):
        BackfillCfg(start=bad)


def test_bad_chunk_size_rejected_via_load_config(tmp_path):
    raw = yaml.safe_load((_CONFIG_DIR / "data_update_all_a.yaml").read_text(encoding="utf-8"))
    raw["data_update"]["backfill"] = {"start": "2020-01-01", "chunk_size": 0}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="chunk_size"):
        load_config(str(p))


def test_chunk_helper_partitions_in_order():
    assert _chunk(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert _chunk([], 3) == []
    assert _chunk(["a"], 5) == [["a"]]
    with pytest.raises(ValueError, match="chunk_size"):
        _chunk(["a"], 0)


# --------------------------------------------------------------------------- #
# fakes for the recorder-style run tests
# --------------------------------------------------------------------------- #
class _RecCovariates:
    """Fake covariates feed: all_a_symbols() returns the whole (fake) market."""

    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.calls = 0

    def all_a_symbols(self):
        self.calls += 1
        return list(self.symbols)


class _EndpointRecorder:
    """Stand-in for update_endpoints: records each per-batch warm; can raise."""

    def __init__(self, *, fail_if_contains=None, exc=ConnectionError):
        self.calls: list[dict] = []
        self.fail_if_contains = fail_if_contains
        self.exc = exc

    def __call__(
        self,
        cache,
        feeds,
        symbols,
        *,
        start,
        end,
        endpoints,
        index_codes,
        fina_fields,
        sw_level="L1",
        intraday_cache=None,
        intraday_window=None,
        capture=None,
    ):
        self.calls.append({
            "symbols": list(symbols),
            "start": start,
            "end": end,
            "endpoints": list(endpoints),
            "intraday_cache": intraday_cache,
            "intraday_window": intraday_window,
        })
        if self.fail_if_contains is not None and self.fail_if_contains in symbols:
            raise self.exc("transient boom")
        return {}


def _write_cfg(
    tmp_path,
    *,
    base="data_update_all_a.yaml",
    backfill=None,
    endpoints=None,
    universe=None,
    universe_scope=None,
    index_codes=None,
    tail_refresh_days=None,
    not_ready_days=None,
):
    raw = yaml.safe_load((_CONFIG_DIR / base).read_text(encoding="utf-8"))
    raw["data"]["cache"]["root_dir"] = str(tmp_path / "cache")
    if backfill is not None:
        raw["data_update"]["backfill"] = backfill
    if endpoints is not None:
        raw["data_update"]["endpoints"] = endpoints
    if universe_scope is not None:
        raw["data_update"]["universe_scope"] = universe_scope
    if index_codes is not None:
        raw["data_update"]["index_codes"] = index_codes
    if tail_refresh_days is not None:
        raw["data_update"]["tail_refresh_days"] = tail_refresh_days
    if not_ready_days is not None:
        raw["data_update"]["not_ready_days"] = not_ready_days
    if universe is not None:
        raw["universe"] = universe
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def _syms(n):
    return [f"{i:06d}.SZ" for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_chunking_partitions_symbols_over_wide_window(tmp_path, monkeypatch):
    all_a = _syms(7)
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2024-01-01", "chunk_size": 3, "include_minute": False},
        endpoints=["market_daily", "adj_factor"],
    )
    feeds = UpdateFeeds(covariates=_RecCovariates(all_a))
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder()
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    result = run_data_backfill(cfg_path, today="2024-06-30")

    assert isinstance(result, BackfillResult)
    assert result.n_batches == 3  # ceil(7 / 3)
    assert [len(c["symbols"]) for c in rec.calls] == [3, 3, 1]
    # in-order, exact partition of the resolved universe
    assert [s for c in rec.calls for s in c["symbols"]] == all_a
    # EVERY batch warmed over the WIDE window [start, today] (not a lookback tail)
    for c in rec.calls:
        assert c["start"] == "2024-01-01"
        assert c["end"] == "2024-06-30"
    assert result.window_start == pd.Timestamp("2024-01-01")
    assert result.window_end == pd.Timestamp("2024-06-30")
    assert result.universe_size == 7
    assert result.failed_batches == 0
    assert result.failed_symbols == []


# --------------------------------------------------------------------------- #
# per-batch failure tolerance (the key PR-2 requirement)
# --------------------------------------------------------------------------- #
def test_batch_failure_is_isolated_and_tallied(tmp_path, monkeypatch):
    all_a = _syms(6)  # chunk 2 -> 3 batches; batch 2 = 000003 + 000004
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2024-01-01", "chunk_size": 2, "include_minute": False},
        endpoints=["market_daily", "adj_factor"],
    )
    feeds = UpdateFeeds(covariates=_RecCovariates(all_a))
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder(fail_if_contains="000003.SZ")
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    # must NOT raise
    result = run_data_backfill(cfg_path, today="2024-06-30")

    assert result.n_batches == 3
    assert len(rec.calls) == 3  # every batch attempted, INCLUDING those after the fail
    assert result.failed_batches == 1
    assert result.failed_symbols == ["000003.SZ", "000004.SZ"]
    # the other two batches were still warmed (failure isolated to its batch)
    warmed = [c["symbols"] for c in rec.calls]
    assert ["000001.SZ", "000002.SZ"] in warmed
    assert ["000005.SZ", "000006.SZ"] in warmed


def test_failure_summary_is_secret_free_and_tallied(tmp_path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2024-01-01", "chunk_size": 1, "include_minute": False},
        endpoints=["market_daily", "adj_factor"],
    )
    feeds = UpdateFeeds(covariates=_RecCovariates(["000001.SZ", "000002.SZ"]))
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder(fail_if_contains="000002.SZ")
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    result = run_data_backfill(cfg_path, today="2024-06-30")
    text = format_summary(result)

    assert "FAILED batches: 1/2" in text
    assert "000002.SZ" in text  # the failed symbol is reported (benign)
    # secret-free: no token / config path leaks into the summary
    assert "token" not in text.lower()
    assert ".config.json" not in text


# --------------------------------------------------------------------------- #
# minute full-window control
# --------------------------------------------------------------------------- #
def test_include_minute_warms_full_window(tmp_path, monkeypatch):
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2022-01-01", "chunk_size": 10, "include_minute": True},
        endpoints=["market_daily", "adj_factor"],
    )
    feeds = UpdateFeeds(covariates=_RecCovariates(["000001.SZ", "000002.SZ"]))
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder()
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    result = run_data_backfill(cfg_path, today="2024-06-30")

    assert result.n_batches == 1
    c = rec.calls[0]
    assert "stk_mins_1min" in c["endpoints"]
    assert c["intraday_cache"] is not None
    # the FULL window, NOT a 7-day tail
    assert c["intraday_window"] == ("2022-01-01 00:00:00", "2024-06-30 23:59:59")
    assert "stk_mins_1min" in result.endpoints
    assert result.include_minute is True
    assert "stk_mins_1min" in result.summary


def test_include_minute_false_skips_minutes_even_if_configured(tmp_path, monkeypatch):
    # include_minute=False is the SOLE control: a configured stk_mins_1min endpoint
    # must NOT trigger minute warming when include_minute is off.
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2022-01-01", "chunk_size": 10, "include_minute": False},
        endpoints=["market_daily", "adj_factor", "stk_mins_1min"],
    )
    feeds = UpdateFeeds(covariates=_RecCovariates(["000001.SZ", "000002.SZ"]))
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder()
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    result = run_data_backfill(cfg_path, today="2024-06-30")

    c = rec.calls[0]
    assert "stk_mins_1min" not in c["endpoints"]
    assert c["intraday_cache"] is None
    assert c["intraday_window"] is None
    assert "stk_mins_1min" not in result.endpoints
    assert result.include_minute is False


# --------------------------------------------------------------------------- #
# all-A universe resolution
# --------------------------------------------------------------------------- #
def test_all_a_universe_resolved_and_chunked(tmp_path, monkeypatch):
    all_a = _syms(10)
    cfg_path = _write_cfg(
        tmp_path,
        universe_scope="all_a",
        backfill={"start": "2024-01-01", "chunk_size": 4, "include_minute": False},
        endpoints=["market_daily", "adj_factor"],
    )
    cov = _RecCovariates(all_a)
    feeds = UpdateFeeds(covariates=cov)
    monkeypatch.setattr("qt.data_backfill._build_feeds", lambda *a, **k: feeds)
    rec = _EndpointRecorder()
    monkeypatch.setattr("qt.data_backfill.update_endpoints", rec)

    result = run_data_backfill(cfg_path, today="2024-06-30")

    assert cov.calls == 1  # resolved once via all_a_symbols()
    assert result.universe_size == 10
    assert result.n_batches == 3  # ceil(10 / 4)
    assert [s for c in rec.calls for s in c["symbols"]] == all_a


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_backfill_requires_cache_enabled(tmp_path, monkeypatch):
    raw = yaml.safe_load((_CONFIG_DIR / "data_update_all_a.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache"]["enabled"] = False
    raw["data"]["cache"]["root_dir"] = str(tmp_path / "cache")
    p = tmp_path / "nocache.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cache.enabled"):
        run_data_backfill(str(p), today="2024-06-30")


def test_future_start_raises_not_silent_noop(tmp_path):
    # A future backfill.start (config typo) would subtract to ZERO gaps everywhere
    # -> silent no-op exit-0. The runtime guard must raise readably instead.
    cfg_path = _write_cfg(
        tmp_path,
        backfill={"start": "2099-01-01", "chunk_size": 10, "include_minute": False},
    )
    with pytest.raises(ValueError, match="after today"):
        run_data_backfill(cfg_path, today="2024-06-30")


# --------------------------------------------------------------------------- #
# resumability (REAL read-through cache + fake tushare client)
# --------------------------------------------------------------------------- #
class _FakeMarketPro:
    """Fake tushare client: deterministic daily/adj rows + counters.

    ``fail_symbols`` (mutable) makes ``daily``/``adj_factor`` RAISE for those
    ts_codes (a transient error); ``daily_fetched`` records every ts_code a fetch
    was attempted for, so a test can assert WHICH symbols hit the API.
    """

    def __init__(self, fail_symbols=()):
        self.daily_calls = 0
        self.adj_calls = 0
        self.fail_symbols = set(fail_symbols)
        self.daily_fetched: list[str] = []
        self.adj_fetched: list[str] = []

    def _days(self, start_date, end_date):
        return pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))

    def daily(self, ts_code, start_date, end_date, **_):
        self.daily_calls += 1
        self.daily_fetched.append(ts_code)
        if ts_code in self.fail_symbols:
            raise ConnectionError("transient boom")
        days = self._days(start_date, end_date)
        if len(days) == 0:
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "open", "high", "low",
                         "close", "vol", "amount"]
            )
        rows = []
        for i, d in enumerate(days):
            px = 10.0 + i
            rows.append({
                "ts_code": ts_code, "trade_date": d.strftime("%Y%m%d"),
                "open": px - 0.2, "high": px + 0.3, "low": px - 0.4,
                "close": px, "vol": 1000.0 + i, "amount": 1.0e6 + i,
            })
        return pd.DataFrame(rows)

    def adj_factor(self, ts_code, start_date, end_date, **_):
        self.adj_calls += 1
        self.adj_fetched.append(ts_code)
        if ts_code in self.fail_symbols:
            raise ConnectionError("transient boom")
        days = self._days(start_date, end_date)
        if len(days) == 0:
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        return pd.DataFrame({
            "ts_code": [ts_code] * len(days),
            "trade_date": [d.strftime("%Y%m%d") for d in days],
            "adj_factor": [1.0] * len(days),
        })


def _static_market_cfg(tmp_path, **overrides):
    """A config/data_update.yaml-based, config-scope, static-universe, market-only
    backfill config (tail/not_ready = 0 so an identical re-run is fully warm)."""
    base = dict(
        base="data_update.yaml",
        universe={
            "type": "static",
            "symbols": ["000001.SZ", "000002.SZ"],
            "min_listing_days": 60,
            "filters": {"missing_close": True, "suspended": False,
                        "st": False, "limit_up_down": False},
        },
        universe_scope="config",
        endpoints=["market_daily", "adj_factor"],
        index_codes=[],
        tail_refresh_days=0,
        not_ready_days=0,
        backfill={"start": "2024-01-02", "chunk_size": 1, "include_minute": False},
    )
    base.update(overrides)
    return _write_cfg(tmp_path, **base)


def test_backfill_second_run_is_warm_zero_fetches(tmp_path, monkeypatch):
    # Static universe + config scope so _resolve_symbols needs no feed; endpoints
    # only market -> a single fake market feed suffices. tail/not_ready = 0 so the
    # identical 2nd run over the SAME fixed [start, today] window is fully warm
    # (no moving-tail / pending-day refetch) — isolating the coverage-skip.
    import tushare as ts

    fake_pro = _FakeMarketPro()
    monkeypatch.setattr(ts, "pro_api", lambda token=None: fake_pro)

    cfg_path = _static_market_cfg(tmp_path)

    fake_secret = tmp_path / "secret.json"
    fake_secret.write_text(json.dumps({"tushare": {"token": "FAKE_TOKEN"}}), encoding="utf-8")

    def _fake_build_feeds(cfg, cache, intraday_cache, rate_limit, scheduler=None):
        market = TushareFeed(secret_file=str(fake_secret), cache=cache)
        return UpdateFeeds(market=market)

    monkeypatch.setattr("qt.data_backfill._build_feeds", _fake_build_feeds)

    r1 = run_data_backfill(cfg_path, today="2024-01-31")
    n1 = fake_pro.daily_calls
    assert n1 > 0  # cold run: real gap fetches happened
    assert r1.failed_batches == 0
    assert r1.summary["market_daily"]["requests"] > 0

    r2 = run_data_backfill(cfg_path, today="2024-01-31")
    assert fake_pro.daily_calls == n1  # warm run: ZERO new daily fetches (all covered)
    assert fake_pro.adj_calls  # (sanity: it was called at least once overall)
    assert r2.failed_batches == 0
    assert r2.summary["market_daily"]["requests"] == 0  # coverage ledger skipped it
    assert r2.summary["adj_factor"]["requests"] == 0


def test_failed_batch_reretried_covered_skipped_real_cache(tmp_path, monkeypatch):
    # The operator workflow, end to end, against a REAL TushareCache/CoverageLedger:
    # run 1 fails for ONE symbol (its batch stays uncovered) while the other symbol
    # is durable; run 2 (healthy client) re-fetches ONLY the previously-failed
    # symbol and skips the already-covered one. chunk_size=1 isolates each symbol
    # to its own batch. The feed's real retry runs, sped up by no-op'ing its sleep.
    import tushare as ts

    import data.feed.throttle as throttle

    monkeypatch.setattr(throttle.time, "sleep", lambda *a, **k: None)  # instant retries

    fake_pro = _FakeMarketPro(fail_symbols={"000002.SZ"})
    monkeypatch.setattr(ts, "pro_api", lambda token=None: fake_pro)

    cfg_path = _static_market_cfg(tmp_path)
    fake_secret = tmp_path / "secret.json"
    fake_secret.write_text(json.dumps({"tushare": {"token": "FAKE_TOKEN"}}), encoding="utf-8")

    def _fake_build_feeds(cfg, cache, intraday_cache, rate_limit, scheduler=None):
        return UpdateFeeds(market=TushareFeed(secret_file=str(fake_secret), cache=cache))

    monkeypatch.setattr("qt.data_backfill._build_feeds", _fake_build_feeds)

    # RUN 1: 000002 fails (uncovered, retryable); 000001 succeeds (durable).
    r1 = run_data_backfill(cfg_path, today="2024-01-31")
    assert r1.failed_batches == 1
    assert r1.failed_symbols == ["000002.SZ"]
    assert "000001.SZ" in fake_pro.daily_fetched  # the good one was fetched+stored

    # RUN 2: healthy client. Only the previously-failed symbol's gap is refetched.
    fake_pro.fail_symbols = set()
    fake_pro.daily_fetched.clear()
    fake_pro.adj_fetched.clear()
    r2 = run_data_backfill(cfg_path, today="2024-01-31")

    assert r2.failed_batches == 0
    assert "000002.SZ" in fake_pro.daily_fetched  # previously-failed -> refetched
    assert "000001.SZ" not in fake_pro.daily_fetched  # already covered -> skipped
    assert "000001.SZ" not in fake_pro.adj_fetched
    # after run 2 the whole universe is durable: a 3rd run is fully warm.
    fake_pro.daily_fetched.clear()
    r3 = run_data_backfill(cfg_path, today="2024-01-31")
    assert fake_pro.daily_fetched == []
    assert r3.failed_batches == 0
