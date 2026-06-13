"""Standalone Tushare cache updater (P4-3) — the 21:00 incremental warm.

A SEPARATE entry point from the backtest pipeline: it only WARMS / UPDATES the
read-through caches (daily endpoint cache + the I2 intraday 1min cache). It never
computes factors, never builds an alpha/portfolio, never runs a backtest, and
never writes a ``PanelStore`` — the backtest still does its own read-through to
fill gaps. Scheduling is external (a systemd timer / cron fires the CLI at 21:00
Asia/Shanghai); this module is just the job body + an explainable summary.

Incremental semantics come from the cache layer: a fully-covered historical run
makes ~0 API calls; a new trading day fetches only the new dates or the recent
tail; a failed fetch records no coverage (retried later); an EMPTY return inside
the not-ready pending window (today, unpublished at 21:00) is recorded as
``not_ready`` — never frozen as permanent coverage. Daily and intraday coverage
use their own ledgers/stores (never mixed). No token / qfq / factor result is
ever stored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from qt.config import RootConfig, load_config

# the intraday window the 21:00 job warms (the bulk historical minute backfill is
# a separate manual run; the daily job only tops up the recent tail).
_INTRADAY_TAIL_DAYS = 7


@dataclass
class UpdateFeeds:
    """The feed objects the updater drives (all share the daily cache)."""

    market: object | None = None
    index: object | None = None
    flags: object | None = None
    covariates: object | None = None
    fina: object | None = None
    intraday: object | None = None


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of a data-update run (immutable)."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    symbols: list[str]
    endpoints: list[str]
    summary: dict[str, dict[str, int]]
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def update_endpoints(
    cache,
    feeds: UpdateFeeds,
    symbols: list[str],
    *,
    start: str,
    end: str,
    endpoints: list[str],
    index_codes: list[str],
    fina_fields: list[str],
    sw_level: str = "L1",
    intraday_cache=None,
    intraday_window: tuple[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Warm each requested endpoint through its feed; return the per-endpoint summary.

    Pure orchestration: every feed shares the read-through ``cache``, so only
    uncovered ranges hit the API. Calls NOTHING but the cache-warming feed
    methods — no factor / alpha / portfolio / backtest / PanelStore.
    """
    eps = set(endpoints)
    if ({"market_daily", "adj_factor"} & eps) and feeds.market is not None:
        feeds.market.get_bars(symbols, start, end)
    if "index_weight" in eps and feeds.index is not None:
        for code in index_codes:
            feeds.index.get_constituents(code, start, end)
    if "suspend_d" in eps and feeds.flags is not None:
        feeds.flags.suspended(symbols, start, end)
    if "namechange" in eps and feeds.flags is not None:
        feeds.flags.st_intervals(symbols)
    if "stk_limit" in eps and feeds.flags is not None:
        feeds.flags.limits(symbols, start, end)
    if "stock_basic" in eps and feeds.covariates is not None:
        feeds.covariates.listing_dates(symbols)
    if "daily_basic" in eps and feeds.covariates is not None:
        feeds.covariates.market_cap(symbols, start, end)
    if "fina_indicator" in eps and feeds.fina is not None:
        feeds.fina.get_fina_indicator(symbols, start, end, fields=fina_fields)
    if "index_member_all" in eps and feeds.covariates is not None:
        feeds.covariates.pit_sw_intervals(symbols, sw_level)

    summary = cache.update_summary()
    if "stk_mins_1min" in eps and feeds.intraday is not None and intraday_window:
        s, e = intraday_window
        feeds.intraday.get_minutes(symbols, s, e)
        st = intraday_cache.stats() if intraday_cache is not None else {}
        summary["stk_mins_1min"] = {
            "requests": int(st.get("stk_mins_1min", 0)),
            "rows_written": 0,
            "not_ready": 0,
        }
    return summary


def _resolve_today(cfg: RootConfig, today) -> pd.Timestamp:
    if today is not None:
        return pd.Timestamp(today).normalize()
    return pd.Timestamp.now(tz=cfg.data_update.timezone).tz_localize(None).normalize()


def _resolve_symbols(cfg: RootConfig, feeds: UpdateFeeds, start: str, end: str) -> list[str]:
    """Universe to warm: static symbols, or the union of index constituents."""
    if cfg.universe.type == "static":
        return list(cfg.universe.symbols)
    codes = cfg.data_update.index_codes or (
        [cfg.universe.index_code] if cfg.universe.index_code else []
    )
    syms: set[str] = set()
    if feeds.index is not None:
        for code in codes:
            cons = feeds.index.get_constituents(code, start, end)
            if not cons.empty:
                syms.update(cons["symbol"].astype(str).tolist())
    return sorted(syms)


def _build_feeds(cfg: RootConfig, cache, intraday_cache, rate_limit: int) -> UpdateFeeds:
    """Construct the real tushare feeds, all sharing the read-through caches."""
    from data.feed.index_feed import IndexConstituentsFeed
    from data.feed.tushare_covariates import TushareCovariatesFeed
    from data.feed.tushare_feed import TushareFeed
    from data.feed.tushare_fina import TushareFinancialFeed
    from data.feed.tushare_flags import TushareFlagsFeed
    from data.feed.tushare_intraday import TushareIntradayFeed

    secret = cfg.data.external_secret_file
    key = cfg.data.tushare_token_key
    return UpdateFeeds(
        market=TushareFeed(secret, token_key=key, rate_limit=rate_limit, cache=cache),
        index=IndexConstituentsFeed(secret, token_key=key, cache=cache),
        flags=TushareFlagsFeed(secret, token_key=key, rate_limit=rate_limit, cache=cache),
        covariates=TushareCovariatesFeed(
            secret, token_key=key, rate_limit=rate_limit, cache=cache
        ),
        fina=TushareFinancialFeed(
            secret, token_key=key, rate_limit=rate_limit, cache=cache
        ),
        intraday=TushareIntradayFeed(
            secret, token_key=key, rate_limit=rate_limit, cache=intraday_cache
        ),
    )


def run_data_update(config_path: str, *, today=None) -> UpdateResult:
    """Run the 21:00 incremental cache warm from ``config_path``.

    Requires ``data.source == 'tushare'``, an external secret file, and
    ``data.cache.enabled``. Builds the daily + intraday read-through caches with
    the not-ready pending window and the fina late-disclosure tail, warms each
    configured endpoint, and returns the per-endpoint summary. Never runs a
    backtest / writes a PanelStore.
    """
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    du = cfg.data_update
    if du is None:
        raise ValueError("data-update requires a 'data_update' config section.")
    if cfg.data.source != "tushare":
        raise ValueError("data-update requires data.source='tushare' (it pulls real data).")
    if not cfg.data.external_secret_file:
        raise ValueError("data-update requires data.external_secret_file (token).")
    if not cfg.data.cache.enabled:
        raise ValueError("data-update requires data.cache.enabled=true.")

    today_ts = _resolve_today(cfg, today)
    end = today_ts.strftime("%Y-%m-%d")
    start = (today_ts - pd.Timedelta(days=du.lookback_days)).strftime("%Y-%m-%d")

    from data.cache import (
        CacheParquetStore,
        CoverageLedger,
        IntradayCoverageLedger,
        IntradayParquetStore,
        TushareCache,
        TushareIntradayCache,
    )

    root = cfg.data.cache.root_dir
    cache = TushareCache(
        CacheParquetStore(root),
        CoverageLedger(root),
        refresh_recent_days=du.tail_refresh_days,
        refresh_dimension_days=cfg.data.cache.refresh_dimension_days,
        force_refresh=tuple(du.force_refresh),
        today=today_ts,
        not_ready_days=du.not_ready_days,
        recent_tail_overrides={"fina_indicator": du.fina_tail_days},
    )
    intraday_cache = TushareIntradayCache(
        IntradayParquetStore(root), IntradayCoverageLedger(root)
    )
    feeds = _build_feeds(cfg, cache, intraday_cache, du.rate_limit_per_min)

    symbols = _resolve_symbols(cfg, feeds, start, end)
    intraday_window = (
        (today_ts - pd.Timedelta(days=_INTRADAY_TAIL_DAYS)).strftime("%Y-%m-%d 00:00:00"),
        today_ts.strftime("%Y-%m-%d 23:59:59"),
    )
    summary = update_endpoints(
        cache, feeds, symbols,
        start=start, end=end,
        endpoints=du.endpoints, index_codes=du.index_codes,
        fina_fields=du.fina_fields,
        sw_level=cfg.processing.neutralize.industry_level,
        intraday_cache=intraday_cache, intraday_window=intraday_window,
    )
    return UpdateResult(
        window_start=pd.Timestamp(start),
        window_end=pd.Timestamp(end),
        symbols=symbols,
        endpoints=list(du.endpoints),
        summary=summary,
        elapsed_seconds=time.perf_counter() - t0,
    )


def format_summary(result: UpdateResult) -> str:
    """One human line per endpoint: requests / rows_written / not_ready."""
    lines = [
        f"data-update window [{result.window_start.date()} .. "
        f"{result.window_end.date()}], {len(result.symbols)} symbols"
    ]
    for ep in result.endpoints:
        s = result.summary.get(ep, {})
        lines.append(
            f"  {ep}: requests={s.get('requests', 0)} "
            f"rows_written={s.get('rows_written', 0)} "
            f"not_ready={s.get('not_ready', 0)}"
        )
    return "\n".join(lines)
