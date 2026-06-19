"""D5 global rate limiter + throttle scheduler wiring (network-free, fake clock).

Proves the limiter spaces calls GLOBALLY (not per-thread), that retries acquire a
slot per attempt, and that a scheduler-mode failure stays secret-safe. No
wall-clock timing — ``monotonic`` / ``sleep`` are injected.
"""

from __future__ import annotations

import threading

import pytest

from data.feed import throttle
from data.feed.scheduler import GlobalRateLimiter
from data.feed.throttle import request_with_retry


class _FrozenClock:
    """A monotonic frozen at 0.0 with a recording (non-advancing) sleep.

    Frozen time models all callers arriving 'at once' — so each reserved slot is
    one interval further out and the recorded waits escalate, which only happens
    if the budget is GLOBAL (a per-thread limiter would record a constant wait).
    """

    def __init__(self):
        self.sleeps: list[float] = []

    def monotonic(self):
        return 0.0

    def sleep(self, d):
        self.sleeps.append(d)


def test_global_spacing_escalates_under_contention():
    clk = _FrozenClock()
    rl = GlobalRateLimiter(120, monotonic=clk.monotonic, sleep=clk.sleep)  # 0.5s
    for _ in range(4):
        rl.acquire()
    # first goes immediately (no sleep); each later caller waits one more interval
    assert clk.sleeps == [0.5, 1.0, 1.5]


def test_disabled_when_rate_non_positive():
    clk = _FrozenClock()
    rl = GlobalRateLimiter(0, monotonic=clk.monotonic, sleep=clk.sleep)
    for _ in range(5):
        rl.acquire()
    assert clk.sleeps == []
    assert rl.rate_limit_per_min == 0


def test_advancing_clock_spaces_each_call():
    # a clock that advances by the slept amount -> sequential callers each wait one
    # interval (the steady-state global cadence).
    state = {"t": 0.0}
    sleeps: list[float] = []

    def mono():
        return state["t"]

    def slp(d):
        sleeps.append(d)
        state["t"] += d

    rl = GlobalRateLimiter(60, monotonic=mono, sleep=slp)  # 1.0s
    for _ in range(3):
        rl.acquire()
    assert sleeps == [1.0, 1.0]  # first immediate, then 1s spacing each


def test_global_not_per_thread_under_real_threads():
    # Frozen clock + no-op-recording sleep; many threads each acquire once. The
    # limiter must reserve exactly N global slots (next_allowed == N*interval),
    # regardless of interleaving — proving the budget is shared, not per-thread.
    clk = _FrozenClock()
    lock = threading.Lock()

    def safe_sleep(d):
        with lock:
            clk.sleeps.append(d)

    rl = GlobalRateLimiter(120, monotonic=clk.monotonic, sleep=safe_sleep)  # 0.5s
    n = 8
    barrier = threading.Barrier(n)

    def worker():
        barrier.wait()
        rl.acquire()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # N reservations consumed N global slots (not N per-thread)
    assert rl._next_allowed == pytest.approx(n * 0.5)
    # cumulative wait is order-independent: 0 + 0.5 + 1.0 + ... + (n-1)*0.5
    assert sum(clk.sleeps) == pytest.approx(0.5 * n * (n - 1) / 2)


class _SpyScheduler:
    def __init__(self):
        self.acquires = 0

    def acquire(self):
        self.acquires += 1


def test_retry_acquires_global_slot_per_attempt(monkeypatch):
    monkeypatch.setattr(throttle.time, "sleep", lambda *_a: None)  # no backoff wait
    sched = _SpyScheduler()
    calls = {"n": 0}

    def flaky(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert request_with_retry(flaky, max_retries=5, scheduler=sched) == "ok"
    assert calls["n"] == 3
    assert sched.acquires == 3  # one global slot per attempt, including retries


def test_scheduler_mode_skips_per_call_rate_sleep(monkeypatch):
    # With a scheduler, the historical per-call ``rate_limit`` sleep must NOT also
    # fire (that would double the spacing / multiply the quota).
    recorded: list[float] = []
    monkeypatch.setattr(throttle.time, "sleep", recorded.append)
    sched = _SpyScheduler()
    request_with_retry(lambda **_kw: "ok", rate_limit=120, scheduler=sched)
    assert recorded == []  # no 0.5s rate-limit sleep
    assert sched.acquires == 1


def test_scheduler_failure_is_secret_safe(monkeypatch):
    monkeypatch.setattr(throttle.time, "sleep", lambda *_a: None)
    sched = _SpyScheduler()

    def always_fail(**_kw):
        raise ValueError("super-secret-token-leak")

    with pytest.raises(RuntimeError) as ei:
        request_with_retry(always_fail, max_retries=2, scheduler=sched)
    msg = str(ei.value)
    assert "ValueError" in msg  # exception TYPE only
    assert "super-secret-token-leak" not in msg  # never the payload
    assert sched.acquires == 2


def test_feed_call_forwards_its_scheduler():
    # the feed plumbing actually hands its scheduler to request_with_retry: a
    # ``_call`` through a feed built with a spy scheduler acquires one slot.
    from data.feed.tushare_feed import TushareFeed

    sched = _SpyScheduler()
    feed = TushareFeed("x.json", rate_limit=120, scheduler=sched)
    assert feed._call(lambda **_kw: "ok") == "ok"
    assert sched.acquires == 1


def test_feed_without_scheduler_is_unchanged(monkeypatch):
    # default (no scheduler) keeps the per-call rate-limit sleep, byte-identical.
    from data.feed.tushare_feed import TushareFeed

    recorded: list[float] = []
    monkeypatch.setattr(throttle.time, "sleep", recorded.append)
    feed = TushareFeed("x.json", rate_limit=60)  # 1.0s spacing, no scheduler
    feed._call(lambda **_kw: "ok")
    assert recorded == [1.0]
