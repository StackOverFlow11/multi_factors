"""Historical backfill of the Tushare read-through caches (PR-2).

A SEPARATE, manual entry point from :func:`qt.data_updater.run_data_update` (the
21:00 incremental warm). Backfill warms the SAME caches over a WIDE window
``[backfill.start, today]`` — the full history, not the incremental
``today - lookback_days`` tail — and (optionally) the full 1min-bar history rather
than the 7-day tail the nightly job tops up.

Three properties make it safe to run for hours on the whole A-share market:

* **Chunked**: symbols are processed in batches of ``backfill.chunk_size``, so
  progress is durable batch-by-batch and memory stays bounded.
* **Per-batch failure-tolerant**: a persistent per-batch fetch error (after the
  feeds' own retries) is LOGGED (secret-free: batch index + exception TYPE only)
  and the run CONTINUES to the next batch. The failed batch's gaps stay
  uncovered, so a re-run retries them — consistent with the cache's "a failed
  fetch records no coverage" semantic. This DIFFERS on purpose from the nightly
  ``run_data_update``, which stays fail-fast (unchanged).
* **Resumable**: resumability is INHERENT to the coverage ledgers — a re-run over
  the same window fetches only still-uncovered gaps. Nothing extra is
  implemented here; the fixed ``[start, today]`` window keeps re-runs stable.

Like the updater it NEVER computes factors / builds an alpha or portfolio / runs
a backtest / writes a ``PanelStore``, and the caches store RAW endpoint facts
only (no token, no qfq, no derived flag).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from qt.config import load_config
from qt.data_updater import (
    _build_caches,
    _build_feeds,
    _build_scheduler,
    _resolve_symbols,
    _resolve_today,
    update_endpoints,
)

_LOGGER = logging.getLogger("qt.data_backfill")

# Cap the failed-symbol list carried in the result / summary so a wide failure
# never produces an unbounded (or memory-heavy) DTO or log line.
_MAX_FAILED_SYMBOLS = 200


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of a historical backfill run (immutable)."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    universe_size: int
    chunk_size: int
    n_batches: int
    include_minute: bool
    endpoints: list[str]
    summary: dict[str, dict[str, int]]
    failed_batches: int
    failed_symbols: list[str]
    elapsed_seconds: float = 0.0
    # Concurrency mode of the warm (max_workers=1 == serial; rate_limit is the
    # global per-minute cap). Surfaced because a manual multi-hour backfill's
    # throughput profile is operationally relevant (mirrors UpdateResult).
    max_workers: int = 1
    rate_limit_per_min: int = 0
    notes: list[str] = field(default_factory=list)


def _chunk(symbols: list[str], size: int) -> list[list[str]]:
    """Split ``symbols`` into contiguous batches of at most ``size`` (order kept)."""
    if size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {size}.")
    return [list(symbols[i : i + size]) for i in range(0, len(symbols), size)]


def run_data_backfill(config_path: str, *, today=None) -> BackfillResult:
    """Warm the tushare caches over the WIDE historical window from ``config_path``.

    Mirrors ``run_data_update``'s guards (tushare source, external secret, cache
    enabled, a ``data_update`` section), reuses the SAME cache + feed construction,
    resolves the same universe (all-A / static / index), then warms in per-symbol
    batches over ``[backfill.start, today]`` with per-batch failure tolerance.
    Returns an immutable :class:`BackfillResult`.
    """
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    du = cfg.data_update
    if du is None:
        raise ValueError("data-backfill requires a 'data_update' config section.")
    if cfg.data.source != "tushare":
        raise ValueError(
            "data-backfill requires data.source='tushare' (it pulls real data)."
        )
    if not cfg.data.external_secret_file:
        raise ValueError("data-backfill requires data.external_secret_file (token).")
    if not cfg.data.cache.enabled:
        raise ValueError("data-backfill requires data.cache.enabled=true.")

    bf = du.backfill
    today_ts = _resolve_today(cfg, today)
    end = today_ts.strftime("%Y-%m-%d")
    start = bf.start  # WIDE window start, NOT today - lookback_days

    # Guard the inverted-window footgun: an interval [start, end] with start AFTER
    # end subtracts to ZERO gaps everywhere, so a future ``backfill.start`` (a
    # config typo) would iterate every batch, fetch nothing, record no coverage,
    # and exit 0 "OK" — silent no-op. Fail readably instead (mirrors
    # ``DataCfg._check_date_order``). ``start`` is already date-validated by
    # ``BackfillCfg``; ``today_ts`` is normalized, so compare against it directly.
    if datetime.strptime(start, "%Y-%m-%d") > today_ts.to_pydatetime():
        raise ValueError(
            f"data_update.backfill.start ({start}) is after today ({end}); the "
            "backfill window is empty (check the date — a future start would "
            "fetch nothing and exit OK)."
        )

    scheduler = _build_scheduler(du)
    cache, intraday_cache = _build_caches(cfg, du, today_ts)
    feeds = _build_feeds(
        cfg, cache, intraday_cache, du.rate_limit_per_min, scheduler=scheduler
    )

    symbols = _resolve_symbols(cfg, feeds, start, end)

    # include_minute is the SOLE control of minute warming for backfill: drop any
    # configured stk_mins_1min from the dense set, then re-add it (with the FULL
    # window + the intraday cache) only when include_minute is on. Minutes are
    # warmed over [start, today] — NOT the nightly 7-day tail.
    dense_endpoints = [e for e in du.endpoints if e != "stk_mins_1min"]
    warm_endpoints = list(dense_endpoints)
    if bf.include_minute:
        warm_endpoints.append("stk_mins_1min")
    intraday_window = (
        f"{start} 00:00:00",
        today_ts.strftime("%Y-%m-%d 23:59:59"),
    )

    batches = _chunk(symbols, bf.chunk_size)
    n_batches = len(batches)
    failed_batches = 0
    failed_symbols: list[str] = []

    for i, batch in enumerate(batches, start=1):
        try:
            update_endpoints(
                cache,
                feeds,
                batch,
                start=start,
                end=end,
                endpoints=warm_endpoints,
                index_codes=du.index_codes,
                fina_fields=du.fina_fields,
                sw_level=cfg.processing.neutralize.industry_level,
                intraday_cache=intraday_cache if bf.include_minute else None,
                intraday_window=intraday_window if bf.include_minute else None,
            )
        except Exception as exc:  # noqa: BLE001 - tolerate ANY per-batch failure
            failed_batches += 1
            if len(failed_symbols) < _MAX_FAILED_SYMBOLS:
                failed_symbols.extend(
                    batch[: _MAX_FAILED_SYMBOLS - len(failed_symbols)]
                )
            # Secret-free: batch index + size + exception TYPE only (never the
            # token, the fetch args, or the exception message).
            _LOGGER.warning(
                "backfill batch %d/%d FAILED (%d symbols): %s — skipped; gaps stay "
                "uncovered (retryable on re-run)",
                i,
                n_batches,
                len(batch),
                type(exc).__name__,
            )
        # Progress line, secret-free (cumulative gap-fetch counts across endpoints).
        cum_requests = sum(cache.stats().values())
        if bf.include_minute:
            cum_requests += sum(intraday_cache.stats().values())
        _LOGGER.info(
            "backfill batch %d/%d: %d symbols, cumulative requests=%d, elapsed=%.1fs",
            i,
            n_batches,
            len(batch),
            cum_requests,
            time.perf_counter() - t0,
        )

    # Build the per-endpoint summary once, from the shared caches, so it is robust
    # to a per-batch failure (a failed last batch never drops the tallies).
    summary = cache.update_summary()
    if bf.include_minute:
        st = intraday_cache.stats()
        summary["stk_mins_1min"] = {
            "requests": int(st.get("stk_mins_1min", 0)),
            "rows_written": 0,
            "not_ready": 0,
        }

    return BackfillResult(
        window_start=pd.Timestamp(start),
        window_end=pd.Timestamp(end),
        universe_size=len(symbols),
        chunk_size=bf.chunk_size,
        n_batches=n_batches,
        include_minute=bf.include_minute,
        endpoints=list(warm_endpoints),
        summary=summary,
        failed_batches=failed_batches,
        failed_symbols=failed_symbols,
        elapsed_seconds=time.perf_counter() - t0,
        max_workers=du.concurrency.max_workers,
        rate_limit_per_min=du.rate_limit_per_min,
    )


def format_summary(result: BackfillResult) -> str:
    """One human line per endpoint + a batch-failure tally (secret-free)."""
    mode = "serial" if result.max_workers <= 1 else f"{result.max_workers} workers"
    lines = [
        f"data-backfill window [{result.window_start.date()} .. "
        f"{result.window_end.date()}], {result.universe_size} symbols in "
        f"{result.n_batches} batch(es) of {result.chunk_size} "
        f"(include_minute={result.include_minute}; {mode}, global "
        f"rate_limit_per_min={result.rate_limit_per_min})"
    ]
    for ep in result.endpoints:
        s = result.summary.get(ep, {})
        lines.append(
            f"  {ep}: requests={s.get('requests', 0)} "
            f"rows_written={s.get('rows_written', 0)} "
            f"not_ready={s.get('not_ready', 0)}"
        )
    if result.failed_batches:
        shown = ", ".join(result.failed_symbols[:10])
        more = (
            ""
            if len(result.failed_symbols) <= 10
            else f" (+{len(result.failed_symbols) - 10} more)"
        )
        lines.append(
            f"  FAILED batches: {result.failed_batches}/{result.n_batches}; "
            f"failed symbols: {shown}{more} — gaps uncovered, retry by re-running"
        )
    else:
        lines.append(f"  all {result.n_batches} batch(es) OK")
    return "\n".join(lines)
