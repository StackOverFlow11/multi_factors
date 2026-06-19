"""IndexConstituentsFeed mapping tests — no network, fake SDK.

The token path uses a FAKE tmp config json with a FAKE token; the real
/home/shaofl/.../.config.json is NEVER read and no network call is made.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout

import pandas as pd
import pytest

from data.feed.index_feed import IndexConstituentsFeed

FAKE_TOKEN = "FAKE_TUSHARE_TOKEN_do_not_leak_0123456789abcdef"


def _write_fake_config(tmp_path, token: str = FAKE_TOKEN):
    """Write a fake nested config json mimicking the real .config.json layout."""
    cfg = {"tushare": {"token": token}, "other": {"unused": 1}}
    path = tmp_path / "fake_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class _FakePro:
    """Stand-in for the tushare pro client returning an index_weight frame."""

    def index_weight(self, index_code, start_date, end_date):  # noqa: ARG002
        return pd.DataFrame(
            {
                "index_code": ["000300.SH"] * 3,
                "con_code": ["000002.SZ", "000001.SZ", "000001.SZ"],
                "trade_date": ["20240131", "20240131", "20240229"],
                "weight": [1.5, 2.5, 2.0],
            }
        )


def _feed(monkeypatch):
    feed = IndexConstituentsFeed("fake.json")
    monkeypatch.setattr(feed, "_client", lambda: _FakePro())
    return feed


def test_get_constituents_maps_to_canonical_columns(monkeypatch):
    feed = _feed(monkeypatch)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    assert list(out.columns) == ["date", "symbol", "weight"]
    assert out["symbol"].dtype == object
    assert str(out["date"].dtype).startswith("datetime64")


def test_get_constituents_sorted_by_date_symbol(monkeypatch):
    feed = _feed(monkeypatch)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    pairs = list(zip(out["date"], out["symbol"]))
    assert pairs == sorted(pairs)
    # 20240131 cross-section: both symbols present, ascending
    jan = out[out["date"] == pd.Timestamp("2024-01-31")]["symbol"].tolist()
    assert jan == ["000001.SZ", "000002.SZ"]


def test_get_constituents_pages_long_window(monkeypatch):
    # A >90-day window must be split into multiple index_weight calls and the
    # snapshots concatenated, so the tushare ~6000-row cap can't drop early dates.
    class _PagingPro:
        def __init__(self):
            self.calls = []

        def index_weight(self, index_code, start_date, end_date):  # noqa: ARG002
            self.calls.append((start_date, end_date))
            return pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": [start_date],  # one snapshot per window
                    "weight": [1.0],
                }
            )

    feed = IndexConstituentsFeed("fake.json")
    pro = _PagingPro()
    monkeypatch.setattr(feed, "_client", lambda: pro)
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-06-30")
    assert len(pro.calls) >= 2  # window was paged
    assert out["date"].nunique() >= 2  # snapshots from multiple windows kept


def test_get_constituents_empty_result_is_schema_shaped(monkeypatch):
    feed = IndexConstituentsFeed("fake.json")

    class _Empty:
        def index_weight(self, **_kw):
            return pd.DataFrame()

    monkeypatch.setattr(feed, "_client", lambda: _Empty())
    out = feed.get_constituents("000300.SH", "2024-01-01", "2024-02-29")
    assert list(out.columns) == ["date", "symbol", "weight"]
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# Token sourcing — real _client() path through the shared read_token (D1)
# --------------------------------------------------------------------------- #
def test_index_feed_sources_token_through_read_token(tmp_path, monkeypatch, caplog):
    """The REAL _client() reads the token via secret.read_token and hands it to
    tushare.pro_api — sourced from the external fake config, never leaked."""
    cfg_path = _write_fake_config(tmp_path)

    captured: dict[str, object] = {}

    class _FakeProToken:
        def index_weight(self, index_code, start_date, end_date):  # noqa: ARG002
            return pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "con_code": ["000001.SZ"],
                    "trade_date": [start_date],
                    "weight": [1.0],
                }
            )

    def fake_pro_api(token=None):
        captured["token"] = token
        return _FakeProToken()

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", fake_pro_api)

    feed = IndexConstituentsFeed(secret_file=str(cfg_path), token_key="tushare.token")
    stdout = io.StringIO()
    with caplog.at_level(logging.DEBUG), redirect_stdout(stdout):
        out = feed.get_constituents("000300.SH", "2024-01-01", "2024-01-31")

    # Token came from the external fake config via the shared reader.
    assert captured["token"] == FAKE_TOKEN
    assert list(out.columns) == ["date", "symbol", "weight"]
    # Never leaked to stdout or logging.
    assert FAKE_TOKEN not in stdout.getvalue()
    assert FAKE_TOKEN not in caplog.text


def test_index_feed_missing_token_key_raises_readable_error(tmp_path, monkeypatch):
    """A missing dotted token_key raises a readable error naming the key path,
    and never echoes the real token value."""
    cfg_path = _write_fake_config(tmp_path)
    feed = IndexConstituentsFeed(secret_file=str(cfg_path), token_key="tushare.nope")

    import tushare as ts

    monkeypatch.setattr(ts, "pro_api", lambda token=None: object())

    with pytest.raises(ValueError) as exc:
        feed.get_constituents("000300.SH", "2024-01-01", "2024-01-31")
    msg = str(exc.value)
    assert "tushare.nope" in msg
    assert FAKE_TOKEN not in msg
