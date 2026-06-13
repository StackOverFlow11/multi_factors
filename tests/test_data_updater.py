"""P4-3 feed cached==direct + data-updater orchestration tests (network-free)."""

from __future__ import annotations

import pandas as pd

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import DAILY_BASIC, TushareCache
from data.clean.pit_financials import asof_financials
from data.feed.tushare_covariates import TushareCovariatesFeed
from data.feed.tushare_fina import TushareFinancialFeed
from qt.data_updater import UpdateFeeds, update_endpoints

_CLK = lambda: pd.Timestamp("2026-06-13 21:00:00")  # noqa: E731


class _FakePro:
    """Window-aware fake tushare client for daily_basic / fina / index_member_all."""

    def daily_basic(self, ts_code, start_date, end_date, fields=None):  # noqa: ARG002
        cat = {
            "000001.SZ": [("20240102", 10.0, 1.0, 1000.0), ("20240103", 11.0, 1.1, 1100.0)],
            "000002.SZ": [("20240102", 20.0, 2.0, 2000.0)],
        }
        S, E = pd.Timestamp(start_date), pd.Timestamp(end_date)
        rows = [r for r in cat.get(ts_code, []) if S <= pd.Timestamp(r[0]) <= E]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": [ts_code] * len(rows),
            "trade_date": [r[0] for r in rows],
            "pe": [r[1] for r in rows], "pb": [r[2] for r in rows],
            "total_mv": [r[3] for r in rows],
        })

    def fina_indicator(self, ts_code, start_date, end_date, fields=None):  # noqa: ARG002
        # returns the full superset (roe / netprofit_yoy / grossprofit_margin)
        cat = {"000001.SZ": [("20230331", "20230420", 5.0, 10.0, 30.0),
                             ("20230630", "20230815", 6.0, 12.0, 31.0)]}
        S, E = pd.Timestamp(start_date), pd.Timestamp(end_date)
        rows = [r for r in cat.get(ts_code, []) if S <= pd.Timestamp(r[0]) <= E]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": [ts_code] * len(rows),
            "end_date": [r[0] for r in rows], "ann_date": [r[1] for r in rows],
            "roe": [r[2] for r in rows], "netprofit_yoy": [r[3] for r in rows],
            "grossprofit_margin": [r[4] for r in rows],
        })

    def index_member_all(self, ts_code):
        cat = {"000001.SZ": [("Banks", "B2", "B3", "20200101", None)],
               "000002.SZ": [("RealEstate", "R2", "R3", "20200101", "20221231")]}
        rows = cat.get(ts_code, [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": [ts_code] * len(rows),
            "l1_name": [r[0] for r in rows], "l2_name": [r[1] for r in rows],
            "l3_name": [r[2] for r in rows], "in_date": [r[3] for r in rows],
            "out_date": [r[4] for r in rows],
        })


def _covariates(tmp_path, monkeypatch, with_cache):
    feed = TushareCovariatesFeed("x.json", cache=_mk_cache(tmp_path) if with_cache else None)
    monkeypatch.setattr(feed, "_client", lambda: _FakePro())
    return feed


def _mk_cache(tmp_path):
    root = str(tmp_path / "cache")
    return TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK,
                        refresh_recent_days=0)


_SYMS = ["000001.SZ", "000002.SZ"]


def _sorted(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_market_cap_cached_equals_direct(tmp_path, monkeypatch):
    direct = _covariates(tmp_path / "d", monkeypatch, with_cache=False)
    cached = _covariates(tmp_path / "c", monkeypatch, with_cache=True)
    a = direct.market_cap(_SYMS, "2024-01-01", "2024-01-31")
    b = cached.market_cap(_SYMS, "2024-01-01", "2024-01-31")
    pd.testing.assert_frame_equal(_sorted(a), _sorted(b))


def test_value_ratios_cached_equals_direct(tmp_path, monkeypatch):
    direct = _covariates(tmp_path / "d", monkeypatch, with_cache=False)
    cached = _covariates(tmp_path / "c", monkeypatch, with_cache=True)
    a = direct.value_ratios(_SYMS, "2024-01-01", "2024-01-31")
    b = cached.value_ratios(_SYMS, "2024-01-01", "2024-01-31")
    pd.testing.assert_frame_equal(_sorted(a), _sorted(b))


def test_pit_sw_intervals_cached_equals_direct(tmp_path, monkeypatch):
    direct = _covariates(tmp_path / "d", monkeypatch, with_cache=False)
    cached = _covariates(tmp_path / "c", monkeypatch, with_cache=True)
    a = direct.pit_sw_intervals(_SYMS, "L1")
    b = cached.pit_sw_intervals(_SYMS, "L1")
    assert set(a) == set(b)
    for sym in a:
        assert set(a[sym]) == set(b[sym])  # identical interval set per symbol


def test_fina_subset_warm_does_not_block_other_subset(tmp_path, monkeypatch):
    # REGRESSION (codex acceptance blocker): a data-update / backtest that warms
    # fina for ["roe"] must NOT poison a later config needing grossprofit_margin.
    cache = _mk_cache(tmp_path)
    f1 = TushareFinancialFeed("x.json", cache=cache)
    monkeypatch.setattr(f1, "_client", lambda: _FakePro())
    # warm requesting only roe (e.g. the default data_update fields)
    f1.get_fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", ["roe"])
    # a later feed over the SAME cache requests grossprofit_margin
    f2 = TushareFinancialFeed("x.json", cache=cache)
    monkeypatch.setattr(f2, "_client", lambda: _FakePro())
    out = f2.get_fina_indicator(
        ["000001.SZ"], "2023-01-01", "2023-12-31", ["grossprofit_margin"]
    )
    assert "grossprofit_margin" in out.columns  # no KeyError, real column
    assert set(out["grossprofit_margin"]) == {30.0, 31.0}  # real data, not NaN


def test_fina_asof_cached_equals_direct(tmp_path, monkeypatch):
    fields = ["roe", "netprofit_yoy"]
    direct = TushareFinancialFeed("x.json")
    monkeypatch.setattr(direct, "_client", lambda: _FakePro())
    cached = TushareFinancialFeed("x.json", cache=_mk_cache(tmp_path))
    monkeypatch.setattr(cached, "_client", lambda: _FakePro())
    a = direct.get_fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", fields)
    b = cached.get_fina_indicator(["000001.SZ"], "2023-01-01", "2023-12-31", fields)
    pd.testing.assert_frame_equal(_sorted(a), _sorted(b))
    # the as-of result is identical too
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-09-01"), "000001.SZ")], names=["date", "symbol"]
    )
    pd.testing.assert_frame_equal(
        asof_financials(idx, a, fields), asof_financials(idx, b, fields)
    )


# --------------------------------------------------------------------------- #
# updater orchestration: only warm methods are called, summary is produced
# --------------------------------------------------------------------------- #
class _RecFeed:
    def __init__(self):
        self.calls: list[str] = []

    def _rec(self, name, value):
        self.calls.append(name)
        return value

    def get_bars(self, symbols, s, e):  # noqa: ARG002
        return self._rec("get_bars", None)

    def get_constituents(self, code, s, e):  # noqa: ARG002
        return self._rec(f"constituents:{code}", pd.DataFrame())

    def suspended(self, symbols, s, e):  # noqa: ARG002
        return self._rec("suspended", set())

    def st_intervals(self, symbols):  # noqa: ARG002
        return self._rec("st_intervals", {})

    def limits(self, symbols, s, e):  # noqa: ARG002
        return self._rec("limits", pd.DataFrame())

    def listing_dates(self, symbols):  # noqa: ARG002
        return self._rec("listing_dates", {})

    def market_cap(self, symbols, s, e):  # noqa: ARG002
        return self._rec("market_cap", pd.DataFrame())

    def get_fina_indicator(self, symbols, s, e, fields=None):  # noqa: ARG002
        return self._rec("fina", pd.DataFrame())

    def pit_sw_intervals(self, symbols, level):  # noqa: ARG002
        return self._rec("pit_sw", {})

    def get_minutes(self, symbols, s, e):  # noqa: ARG002
        return self._rec("get_minutes", pd.DataFrame())


def test_update_endpoints_warms_only_configured(tmp_path):
    root = str(tmp_path / "cache")
    cache = TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK)
    rec = _RecFeed()
    feeds = UpdateFeeds(market=rec, index=rec, flags=rec, covariates=rec,
                        fina=rec, intraday=rec)

    class _IC:
        def stats(self):
            return {"stk_mins_1min": 0}

    summary = update_endpoints(
        cache, feeds, _SYMS,
        start="2024-01-01", end="2024-01-31",
        endpoints=["daily_basic", "fina_indicator", "index_member_all", "stk_mins_1min"],
        index_codes=["000300.SH"], fina_fields=["roe"],
        intraday_cache=_IC(), intraday_window=("2024-01-24 00:00:00", "2024-01-31 23:59:59"),
    )
    # exactly the warm methods for the requested endpoints fired
    assert rec.calls == ["market_cap", "fina", "pit_sw", "get_minutes"]
    # summary has every daily endpoint (seeded) + the intraday entry
    assert DAILY_BASIC in summary and "stk_mins_1min" in summary


def test_update_endpoints_skips_unconfigured(tmp_path):
    root = str(tmp_path / "cache")
    cache = TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK)
    rec = _RecFeed()
    feeds = UpdateFeeds(market=rec, index=rec, flags=rec, covariates=rec, fina=rec)
    update_endpoints(
        cache, feeds, _SYMS, start="2024-01-01", end="2024-01-31",
        endpoints=["market_daily", "adj_factor"], index_codes=[], fina_fields=["roe"],
    )
    assert rec.calls == ["get_bars"]  # one call warms both market endpoints
