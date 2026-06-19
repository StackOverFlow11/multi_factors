"""D3b report-only data-update quality hook — network-free, synthetic frames.

Covers: default-disabled config; clean report; bad daily+intraday frames -> hard
findings without raising / altering the updater summary; endpoint-selection is
honored (a not-warmed endpoint is never checked); report_name path validation;
and the generated report never carries a secret path / token key / token value.
The D3 check matrix itself is exercised in tests/test_data_quality_*.py — here we
only verify the hook surfaces those checks.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.tushare_cache import TushareCache
from data.clean.schema import normalize_panel
from data.quality import HARD, make_finding
from qt.config import DataUpdateCfg, DataUpdateQualityCfg, load_config
from qt.data_update_quality import (
    QUALITY_ENDPOINTS,
    collect_findings,
    write_quality_report,
)
from qt.data_updater import UpdateFeeds, update_endpoints

_CLK = lambda: pd.Timestamp("2026-06-13 21:00:00")  # noqa: E731

# Secret-looking literals a finding might (wrongly) carry; the report must redact
# them. These are SCAN TARGETS, not real secrets.
_SECRET_PATH = "/home/shaofl/Projects/financial_projects/.config.json"
_SECRET_KEY = "tushare.token"
_SECRET_NEEDLES = (_SECRET_PATH, ".config.json", _SECRET_KEY, "token=")


# --------------------------------------------------------------------------- #
# synthetic frames (shapes mirror get_bars / get_minutes outputs)
# --------------------------------------------------------------------------- #
def _market_panel(rows: list[dict]) -> pd.DataFrame:
    """A normalized MultiIndex(date, symbol) market panel (as get_bars returns)."""
    return normalize_panel(pd.DataFrame(rows))


def _clean_market() -> pd.DataFrame:
    rows = []
    for sym, base in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
        for i, d in enumerate(("2024-01-03", "2024-01-04")):
            px = base + i * 0.1
            rows.append({
                "date": pd.Timestamp(d), "symbol": sym,
                "open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                "volume": 1000.0 + i, "amount": 1_000_000.0 + i, "adj_factor": 1.0,
            })
    return _market_panel(rows)


def _bad_market() -> pd.DataFrame:
    """One row with high < low -> a hard structural finding."""
    rows = [
        {"date": pd.Timestamp("2024-01-03"), "symbol": "000001.SZ",
         "open": 10.0, "high": 9.0, "low": 10.5, "close": 9.5,
         "volume": 1000.0, "amount": 1_000_000.0, "adj_factor": 1.0},
        {"date": pd.Timestamp("2024-01-04"), "symbol": "000001.SZ",
         "open": 10.1, "high": 10.6, "low": 9.9, "close": 10.2,
         "volume": 1010.0, "amount": 1_000_100.0, "adj_factor": 1.0},
    ]
    return _market_panel(rows)


def _clean_intraday() -> pd.DataFrame:
    times = pd.to_datetime(["2024-01-03 09:31:00", "2024-01-03 09:32:00"])
    return pd.DataFrame({
        "bar_end": list(times),
        "symbol": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.1], "high": [10.3, 10.4],
        "low": [9.9, 10.0], "close": [10.1, 10.2],
        "volume": [100.0, 110.0], "amount": [1000.0, 1100.0],
    })


def _bad_intraday() -> pd.DataFrame:
    df = _clean_intraday()
    df.loc[0, "low"] = 0.0  # non-positive OHLC -> hard
    return df


def _mk_cache(tmp_path) -> TushareCache:
    root = str(tmp_path)
    return TushareCache(CacheParquetStore(root), CoverageLedger(root), clock=_CLK)


# --------------------------------------------------------------------------- #
# 1. default config keeps quality disabled
# --------------------------------------------------------------------------- #
def test_default_data_update_config_has_quality_disabled():
    cfg = load_config("config/data_update.yaml")
    assert cfg.data_update is not None
    assert cfg.data_update.quality.enabled is False
    assert cfg.data_update.quality.report_name == "data_update_quality_report.md"


def test_data_update_cfg_without_quality_block_defaults_disabled():
    cfg = DataUpdateCfg()
    assert cfg.quality.enabled is False
    assert set(cfg.quality.endpoints) <= set(QUALITY_ENDPOINTS)


# --------------------------------------------------------------------------- #
# 2. enabling quality writes a clean report for clean warmed frames
# --------------------------------------------------------------------------- #
def test_enabled_quality_writes_clean_report(tmp_path):
    findings, checked = collect_findings(
        selected=list(QUALITY_ENDPOINTS), warmed=set(QUALITY_ENDPOINTS),
        market_frame=_clean_market(), intraday_frame=_clean_intraday(),
    )
    assert findings == []
    outcome = write_quality_report(
        findings, report_dir=str(tmp_path), report_name="q.md",
        window_start=pd.Timestamp("2024-01-03"), window_end=pd.Timestamp("2024-01-04"),
        n_symbols=2, checked_endpoints=checked,
    )
    assert outcome.findings_count == 0 and outcome.hard_count == 0
    assert outcome.report_path == tmp_path / "q.md"
    text = outcome.report_path.read_text(encoding="utf-8")
    assert "No data-quality findings" in text
    assert "endpoints checked: market_daily, adj_factor, stk_mins_1min" in text
    assert "symbols checked: 2" in text


# --------------------------------------------------------------------------- #
# 3. bad daily + intraday frames -> hard findings, no raise
# --------------------------------------------------------------------------- #
def test_enabled_quality_bad_frames_produce_hard(tmp_path):
    findings, checked = collect_findings(
        selected=list(QUALITY_ENDPOINTS), warmed=set(QUALITY_ENDPOINTS),
        market_frame=_bad_market(), intraday_frame=_bad_intraday(),
    )
    outcome = write_quality_report(
        findings, report_dir=str(tmp_path), report_name="q.md",
        window_start=pd.Timestamp("2024-01-03"), window_end=pd.Timestamp("2024-01-04"),
        n_symbols=1, checked_endpoints=checked,
    )
    assert outcome.hard_count >= 2  # market high<low + intraday non-positive ohlc
    text = outcome.report_path.read_text(encoding="utf-8")
    assert "[hard]" in text
    assert "`market_daily`" in text and "`stk_mins_1min`" in text
    for needle in _SECRET_NEEDLES:
        assert needle not in text


def test_report_is_deterministic(tmp_path):
    findings, checked = collect_findings(
        selected=list(QUALITY_ENDPOINTS), warmed=set(QUALITY_ENDPOINTS),
        market_frame=_bad_market(), intraday_frame=_bad_intraday(),
    )
    kw = dict(
        report_name="q.md", window_start=pd.Timestamp("2024-01-03"),
        window_end=pd.Timestamp("2024-01-04"), n_symbols=1, checked_endpoints=checked,
    )
    a = write_quality_report(findings, report_dir=str(tmp_path / "a"), **kw)
    b = write_quality_report(findings, report_dir=str(tmp_path / "b"), **kw)
    assert a.report_path.read_text() == b.report_path.read_text()


# --------------------------------------------------------------------------- #
# 4. endpoint selection honored — not-warmed endpoints are never checked
# --------------------------------------------------------------------------- #
def test_collect_findings_skips_not_warmed_endpoint():
    # stk_mins_1min is selected AND a (bad) frame is present, but it was NOT
    # warmed by data_update.endpoints -> it must not be checked.
    findings, checked = collect_findings(
        selected=["market_daily", "adj_factor", "stk_mins_1min"],
        warmed={"market_daily", "adj_factor"},
        market_frame=_clean_market(), intraday_frame=_bad_intraday(),
    )
    assert "stk_mins_1min" not in checked
    assert "market_daily" in checked and "adj_factor" in checked
    assert not any(f.dataset == "stk_mins_1min" for f in findings)
    assert findings == []  # clean market, intraday skipped


def test_collect_findings_skips_unselected_endpoint():
    # market_daily/adj_factor warmed but NOT selected for quality -> not checked,
    # even though the (bad) frame would produce a hard finding.
    findings, checked = collect_findings(
        selected=["stk_mins_1min"],
        warmed={"market_daily", "adj_factor", "stk_mins_1min"},
        market_frame=_bad_market(), intraday_frame=_clean_intraday(),
    )
    assert checked == ["stk_mins_1min"]
    assert findings == []


# --------------------------------------------------------------------------- #
# 5. report_name path validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad", ["/abs/x.md", "../x.md", "sub/x.md", "a\\b.md", "..", ".", "  "]
)
def test_report_name_rejects_paths(bad):
    with pytest.raises(ValidationError):
        DataUpdateQualityCfg(report_name=bad)


def test_report_name_accepts_bare_filename():
    assert DataUpdateQualityCfg(report_name="ok.md").report_name == "ok.md"


def test_unknown_quality_endpoint_rejected():
    with pytest.raises(ValidationError):
        DataUpdateQualityCfg(endpoints=["market_daily", "index_weight"])


# --------------------------------------------------------------------------- #
# 6. report never carries secret-looking inputs (D3 redaction inherited)
# --------------------------------------------------------------------------- #
def test_report_redacts_secret_looking_finding(tmp_path):
    findings = [
        make_finding(
            "market_daily", "x", HARD, 1,
            examples=[{"symbol": "000001.SZ", "path": _SECRET_PATH}],
            note=f"leaked at {_SECRET_PATH} / {_SECRET_KEY}",
        )
    ]
    outcome = write_quality_report(
        findings, report_dir=str(tmp_path), report_name="q.md",
        window_start=pd.Timestamp("2024-01-01"), window_end=pd.Timestamp("2024-01-31"),
        n_symbols=2, checked_endpoints=["market_daily"],
    )
    text = outcome.report_path.read_text(encoding="utf-8")
    for needle in (_SECRET_PATH, ".config.json", _SECRET_KEY):
        assert needle not in text
    assert "[REDACTED]" in text
    assert "000001.SZ" in text  # benign content still renders


# --------------------------------------------------------------------------- #
# capture wiring: capturing the warmed frames never alters the updater summary
# --------------------------------------------------------------------------- #
class _CaptureFeed:
    """Fake market+intraday feed returning fixed frames (drives update_endpoints)."""

    def __init__(self, bars, minutes):
        self._bars = bars
        self._minutes = minutes

    def get_bars(self, symbols, s, e):  # noqa: ARG002
        return self._bars

    def get_minutes(self, symbols, s, e):  # noqa: ARG002
        return self._minutes


class _IntradayCacheStub:
    def stats(self):
        return {"stk_mins_1min": 0}


def test_capture_does_not_change_summary(tmp_path):
    feed = _CaptureFeed(_clean_market(), _clean_intraday())
    feeds = UpdateFeeds(market=feed, intraday=feed)
    kwargs = dict(
        start="2024-01-01", end="2024-01-31",
        endpoints=["market_daily", "adj_factor", "stk_mins_1min"],
        index_codes=[], fina_fields=["roe"],
        intraday_cache=_IntradayCacheStub(),
        intraday_window=("2024-01-24 00:00:00", "2024-01-31 23:59:59"),
    )
    s_plain = update_endpoints(_mk_cache(tmp_path / "p"), feeds, ["000001.SZ"], **kwargs)
    cap: dict = {}
    s_cap = update_endpoints(
        _mk_cache(tmp_path / "c"), feeds, ["000001.SZ"], capture=cap, **kwargs
    )
    assert s_plain == s_cap  # capturing never alters the summary
    assert cap["market"] is not None and cap["intraday"] is not None
    # and the captured frames flow cleanly through the hook (no hard findings)
    findings, checked = collect_findings(
        selected=list(QUALITY_ENDPOINTS), warmed={"market_daily", "adj_factor", "stk_mins_1min"},
        market_frame=cap["market"], intraday_frame=cap["intraday"],
    )
    assert findings == []
    assert checked == ["market_daily", "adj_factor", "stk_mins_1min"]
