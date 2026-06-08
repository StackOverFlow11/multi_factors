"""Slice 3 tests: DemoFeed + TushareFeed boundary (DATA-001/002/004, SEC-001/004).

No real network. The tushare SDK is monkeypatched; a FAKE tmp config json with a
FAKE token is used everywhere — the real /home/shaofl/.../.config.json is NEVER
read. We assert the token comes from the external config and never leaks to
stdout or logging.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout

import pandas as pd
import pytest

from data.clean.schema import CORE_COLUMNS, validate_panel
from data.feed.base import DataFeed
from data.feed.demo_feed import DemoFeed
from data.feed.tushare_feed import TushareFeed
from tests.fixtures.panel_factory import SYMBOLS

FAKE_TOKEN = "FAKE_TUSHARE_TOKEN_do_not_leak_0123456789abcdef"


# --------------------------------------------------------------------------- #
# DemoFeed
# --------------------------------------------------------------------------- #
def test_demo_feed_returns_standard_panel():
    feed = DemoFeed()
    assert isinstance(feed, DataFeed)

    symbols = SYMBOLS[:3]
    start, end = "2024-01-03", "2024-01-10"
    panel = feed.get_bars(symbols, start, end, freq="D")

    # Canonical shape: must pass the foundation validator unchanged.
    validate_panel(panel)
    for col in CORE_COLUMNS:
        assert col in panel.columns

    # Honors the requested symbol list (no extras, no missing).
    got_symbols = set(panel.index.get_level_values("symbol").unique())
    assert got_symbols == set(symbols)

    # Honors the [start, end] inclusive date range.
    dates = panel.index.get_level_values("date")
    assert dates.min() >= pd.Timestamp(start)
    assert dates.max() <= pd.Timestamp(end)

    # adj_factor fixed at 1.0 (DATA-003: at least preserve adj_factor).
    assert (panel["adj_factor"] == 1.0).all()

    # Deterministic: same call twice -> identical frame (no randomness, no now()).
    panel2 = feed.get_bars(symbols, start, end, freq="D")
    pd.testing.assert_frame_equal(panel, panel2)


# --------------------------------------------------------------------------- #
# TushareFeed — token sourcing
# --------------------------------------------------------------------------- #
def _write_fake_config(tmp_path, token: str = FAKE_TOKEN):
    """Write a fake nested config json mimicking the real .config.json layout."""
    cfg = {"tushare": {"token": token}, "other": {"unused": 1}}
    path = tmp_path / "fake_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _tushare_style_frame(symbol: str) -> pd.DataFrame:
    """A raw tushare `daily` style frame (English-but-non-canonical columns)."""
    return pd.DataFrame(
        {
            "ts_code": [symbol, symbol],
            "trade_date": ["20240103", "20240104"],
            "open": [10.0, 10.5],
            "high": [10.8, 11.0],
            "low": [9.9, 10.4],
            "close": [10.6, 10.9],
            "vol": [123456.0, 234567.0],   # tushare uses 'vol', not 'volume'
            "amount": [1_300_000.0, 2_550_000.0],
        }
    )


def test_tushare_feed_reads_token_from_external_config(tmp_path, monkeypatch):
    cfg_path = _write_fake_config(tmp_path)

    captured: dict[str, object] = {}

    class FakePro:
        def daily(self, **kwargs):
            return _tushare_style_frame(kwargs.get("ts_code", "000001.SZ"))

        def adj_factor(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": [kwargs.get("ts_code", "000001.SZ")] * 2,
                    "trade_date": ["20240103", "20240104"],
                    "adj_factor": [1.0, 1.0],
                }
            )

    def fake_pro_api(token=None):
        captured["token"] = token
        return FakePro()

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", fake_pro_api)

    feed = TushareFeed(secret_file=str(cfg_path), token_key="tushare.token")
    # Trigger lazy client construction.
    feed.get_bars(["000001.SZ"], "2024-01-03", "2024-01-04")

    assert captured["token"] == FAKE_TOKEN


def test_tushare_feed_does_not_log_token(tmp_path, monkeypatch, caplog):
    cfg_path = _write_fake_config(tmp_path)

    class FakePro:
        def daily(self, **kwargs):
            return _tushare_style_frame(kwargs.get("ts_code", "000001.SZ"))

        def adj_factor(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": [kwargs.get("ts_code", "000001.SZ")] * 2,
                    "trade_date": ["20240103", "20240104"],
                    "adj_factor": [1.0, 1.0],
                }
            )

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", lambda token=None: FakePro())

    stdout = io.StringIO()
    with caplog.at_level(logging.DEBUG), redirect_stdout(stdout):
        feed = TushareFeed(secret_file=str(cfg_path))
        feed.get_bars(["000001.SZ"], "2024-01-03", "2024-01-04")
        # Force the feed's own repr to be safe too.
        repr(feed)

    assert FAKE_TOKEN not in stdout.getvalue()
    assert FAKE_TOKEN not in caplog.text
    for record in caplog.records:
        assert FAKE_TOKEN not in record.getMessage()


def test_tushare_feed_normalizes_columns(tmp_path, monkeypatch):
    cfg_path = _write_fake_config(tmp_path)

    class FakePro:
        def daily(self, **kwargs):
            return _tushare_style_frame(kwargs.get("ts_code", "000001.SZ"))

        def adj_factor(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": [kwargs.get("ts_code", "000001.SZ")] * 2,
                    "trade_date": ["20240103", "20240104"],
                    "adj_factor": [1.0, 1.0],
                }
            )

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", lambda token=None: FakePro())

    feed = TushareFeed(secret_file=str(cfg_path))
    panel = feed.get_bars(["000001.SZ"], "2024-01-03", "2024-01-04")

    # Output is canonical: passes the foundation validator and has CORE_COLUMNS.
    validate_panel(panel)
    for col in CORE_COLUMNS:
        assert col in panel.columns

    # tushare 'ts_code'/'trade_date'/'vol' were mapped to symbol/date/volume.
    assert "volume" in panel.columns
    assert "vol" not in panel.columns
    assert "ts_code" not in panel.columns
    assert "trade_date" not in panel.columns
    assert list(panel.index.names) == ["date", "symbol"]
    assert set(panel.index.get_level_values("symbol").unique()) == {"000001.SZ"}

    # vol values landed in volume; first row close mapped correctly.
    first = panel.xs("000001.SZ", level="symbol").iloc[0]
    assert first["volume"] == 123456.0
    assert first["close"] == 10.6


def test_tushare_feed_missing_token_key_raises_readable_error(tmp_path, monkeypatch):
    # token_key points at a nested key that does not exist -> readable error.
    cfg_path = _write_fake_config(tmp_path)
    feed = TushareFeed(secret_file=str(cfg_path), token_key="tushare.nope")

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", lambda token=None: object())

    with pytest.raises(ValueError) as exc:
        feed.get_bars(["000001.SZ"], "2024-01-03", "2024-01-04")
    msg = str(exc.value)
    assert "tushare.nope" in msg
    # The error message must never leak the real token contents.
    assert FAKE_TOKEN not in msg
