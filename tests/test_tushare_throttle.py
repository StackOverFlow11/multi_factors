"""Tests for the shared rate-limit + retry helper (SEC-004) and TushareFeed wiring.

No network, no token read. ``request_with_retry`` is exercised with a fake
callable and a monkeypatched ``time.sleep`` (so tests are instant).
"""

from __future__ import annotations

import pytest

from data.feed import throttle
from data.feed.throttle import request_with_retry
from data.feed.tushare_feed import TushareFeed


@pytest.fixture
def no_sleep(monkeypatch):
    """Record sleeps instead of actually sleeping (keeps tests instant)."""
    recorded: list[float] = []
    monkeypatch.setattr(throttle.time, "sleep", recorded.append)
    return recorded


def test_retries_then_succeeds(no_sleep):
    calls = {"n": 0}

    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert request_with_retry(flaky, max_retries=5) == "ok"
    assert calls["n"] == 3  # failed twice, third attempt succeeded


def test_raises_after_exhausting_retries(no_sleep):
    def always_fail(**_kw):
        raise RuntimeError("upstream boom")

    with pytest.raises(RuntimeError, match="2 attempt"):
        request_with_retry(always_fail, max_retries=2)


def test_throttles_per_rate_limit(no_sleep):
    request_with_retry(lambda **_kw: "ok", rate_limit=120)  # 120/min -> 0.5s
    assert no_sleep == [0.5]


def test_noop_when_rate_limit_unset(no_sleep):
    request_with_retry(lambda **_kw: "ok")
    assert no_sleep == []


def test_tushare_feed_call_delegates(no_sleep):
    feed = TushareFeed("x.json", rate_limit=60, max_retries=3)  # -> 1.0s spacing
    assert feed._call(lambda **_kw: "ok") == "ok"
    assert 1.0 in no_sleep
