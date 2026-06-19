"""Global request scheduler / rate limiter for bounded-concurrency warms (D5).

When a cache warm runs with more than one worker, every worker must funnel its
API attempts through ONE shared budget — otherwise N threads each sleeping
``60 / rate_limit`` would multiply the global quota by N and blow the Tushare
ceiling. :class:`GlobalRateLimiter` reserves the next evenly-spaced time slot
under a lock, then sleeps OUTSIDE the lock until that slot, so concurrent callers
are serialized onto one global cadence.

It is transport-agnostic and secret-free: it never sees a token, kwargs, an
endpoint, or an exception payload — only a per-minute integer budget. ``monotonic``
and ``sleep`` are injectable so the spacing can be asserted with a fake clock (no
wall-clock timing in tests).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class GlobalRateLimiter:
    """A single shared per-minute request budget enforced ACROSS worker threads.

    ``rate_limit_per_min <= 0`` disables throttling (``acquire`` is a no-op). Each
    ``acquire`` reserves the next slot spaced ``60 / rate_limit_per_min`` seconds
    after the previous reservation, regardless of which thread calls it — so the
    global rate holds no matter how many workers contend.
    """

    def __init__(
        self,
        rate_limit_per_min: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate_limit_per_min = int(rate_limit_per_min)
        self._interval = (
            60.0 / self._rate_limit_per_min if self._rate_limit_per_min > 0 else 0.0
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed: float | None = None

    @property
    def rate_limit_per_min(self) -> int:
        """The configured global per-minute budget (0 == disabled)."""
        return self._rate_limit_per_min

    def acquire(self) -> None:
        """Block until this caller's globally-spaced slot; no-op if disabled.

        The slot is reserved under the lock (so two threads never claim the same
        instant), then the wait happens outside the lock so other threads can keep
        reserving their own later slots concurrently.
        """
        if self._interval <= 0:
            return
        with self._lock:
            now = self._monotonic()
            if self._next_allowed is None or now >= self._next_allowed:
                # the budget is idle: this caller goes now, next slot one interval out
                self._next_allowed = now + self._interval
                wait = 0.0
            else:
                # the budget is busy: wait for the reserved slot, push the next one out
                wait = self._next_allowed - now
                self._next_allowed = self._next_allowed + self._interval
        if wait > 0:
            self._sleep(wait)
