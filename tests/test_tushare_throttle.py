"""Tests for TushareFeed rate-limit + retry (SEC-004). No network, no token read.

Constructing TushareFeed is lazy (no token / no client), so ``_call`` / ``_throttle``
can be exercised directly with a fake callable and a monkeypatched ``time.sleep``.
"""

from __future__ import annotations

import pytest

from data.feed import tushare_feed
from data.feed.tushare_feed import TushareFeed


@pytest.fixture
def no_sleep(monkeypatch):
    """Record sleeps instead of actually sleeping (keeps tests instant)."""
    recorded: list[float] = []
    monkeypatch.setattr(tushare_feed.time, "sleep", recorded.append)
    return recorded


def test_call_retries_then_succeeds(no_sleep):
    calls = {"n": 0}

    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    feed = TushareFeed("x.json", max_retries=5)
    assert feed._call(flaky) == "ok"
    assert calls["n"] == 3  # failed twice, third attempt succeeded


def test_call_raises_after_exhausting_retries(no_sleep):
    def always_fail(**_kw):
        raise RuntimeError("upstream boom")

    feed = TushareFeed("x.json", max_retries=2)
    with pytest.raises(RuntimeError, match="2 attempt"):
        feed._call(always_fail)


def test_throttle_sleeps_per_rate_limit(no_sleep):
    feed = TushareFeed("x.json", rate_limit=120)  # 120/min -> 0.5s spacing
    feed._throttle()
    assert no_sleep == [0.5]


def test_throttle_is_noop_when_unset(no_sleep):
    feed = TushareFeed("x.json")  # rate_limit None
    feed._throttle()
    assert no_sleep == []


def test_call_throttles_after_success(no_sleep):
    feed = TushareFeed("x.json", rate_limit=60)  # -> 1.0s spacing
    assert feed._call(lambda **_kw: "ok") == "ok"
    assert 1.0 in no_sleep
