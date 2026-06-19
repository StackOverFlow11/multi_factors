"""Shared rate-limit + retry for tushare endpoints (SEC-004).

A tushare endpoint is a plain callable (e.g. ``pro.daily``, ``pro.index_weight``).
``request_with_retry`` retries transient failures with exponential backoff, then
sleeps to respect a per-minute call cap. It never echoes the token: the failure
message carries only the exception TYPE, not its payload.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def request_with_retry(
    fn: Callable[..., Any],
    *,
    max_retries: int = 6,
    rate_limit: int | None = None,
    scheduler: Any = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(**kwargs)`` with retry-on-error + a per-minute throttle.

    Retries transient exceptions with exponential backoff up to ``max_retries``
    attempts. Throttling has two mutually-exclusive modes:

    * ``scheduler`` set (D5 bounded concurrency): acquire one global slot BEFORE
      each attempt, so every worker's attempts (including retries) share one budget
      — no per-call ``rate_limit`` sleep is added (that would double the spacing).
    * ``scheduler`` unset (the historical serial path): on success sleep
      ``60 / rate_limit`` seconds (no-op if unset).

    Raises a readable ``RuntimeError`` (exception TYPE only, never the token /
    payload) if every attempt fails.
    """
    attempts = max(1, int(max_retries))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        if scheduler is not None:
            scheduler.acquire()  # global slot before EVERY attempt (incl. retries)
        try:
            result = fn(**kwargs)
        except Exception as exc:  # transient API / network error
            last_exc = exc
            time.sleep(min(2.0**attempt, 8.0))
            continue
        if scheduler is None and rate_limit:
            time.sleep(60.0 / rate_limit)
        return result
    raise RuntimeError(
        f"tushare call failed after {attempts} attempt(s): "
        f"{type(last_exc).__name__}. Check connectivity / rate limit."
    ) from last_exc
