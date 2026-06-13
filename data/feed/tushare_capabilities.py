"""Tushare endpoint capability registry — stable metadata + usage intent.

This is a HUMAN/PLANNING reference, NOT a runtime permission gate. It records,
per endpoint, what the data means to THIS project (phase, date field, PIT
timing, cache relevance, batch viability) so we can reason about which sources
are safe/useful to wire in.

Hard rules (do not violate):
  * NO token, NO account points, NO secret of any kind lives here.
  * Production code MUST NOT branch on ``observed_status`` to auto-skip an
    endpoint — whether a call is authorized is decided by the live API response,
    and permissions/points change over time. ``observed_status`` is a dated,
    non-secret snapshot for auditing only (see docs/data/tushare_permissions.md
    and the gitignored artifacts/permissions/ probe JSON).

``observed_status`` enum: authorized | empty_authorized | rate_limited |
no_permission | unknown. ``observed_at`` is the probe date for that snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

OBSERVED_AT = "2026-06-13"

# pit_timing vocabulary (when a value becomes known, for look-ahead safety):
#   eod                 -> known after T's close (e.g. daily bar)
#   eod_postclose       -> published only in T's evening (use no earlier than T+1)
#   same_day_publish    -> published same day, PIT-safe to use on T (e.g. pe/pb)
#   ann_date_disclosure -> usable only on/after its ann_date (financial reports)
#   asof_snapshot       -> latest snapshot on-or-before the date (index_weight)
#   asof_interval       -> in/out interval membership (PIT SW industry)
#   dimension_snapshot  -> slow-moving descriptor (list_date / name history)
#   event_date          -> tied to an event date (survey date)
#   intraday            -> within-day data (minute / auction)

# batch_viability_note vocabulary:
#   proven_in_project_runs       -> pulled at scale in real P0..P4-2 runs
#   authorized_rate_unverified   -> single call OK; per-batch rate NOT tested
#   not_batch_viable_1_per_hour  -> authorized but throttled to 1 call/hour
#   no_access                    -> no permission (40203)


@dataclass(frozen=True)
class EndpointCapability:
    """Stable, non-secret metadata + this project's intent for one endpoint."""

    name: str
    category: str
    phase_or_usage: str
    date_field: str
    key_fields: tuple[str, ...]
    pit_timing: str
    cache_relevance: str
    batch_viability_note: str
    observed_status: str = "unknown"
    observed_at: str = OBSERVED_AT


_CAPS: tuple[EndpointCapability, ...] = (
    # -- market bars (P4-1 cached) ---------------------------------------- #
    EndpointCapability(
        "daily", "market", "P0 raw OHLCV (qfq downstream)", "trade_date",
        ("ts_code", "trade_date"), "eod", "p4_1_cached",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "adj_factor", "market", "P1 front-adjust factor", "trade_date",
        ("ts_code", "trade_date"), "eod", "p4_1_cached",
        "proven_in_project_runs", "authorized",
    ),
    # -- universe / tradability (P4-2 cached) ----------------------------- #
    EndpointCapability(
        "index_weight", "universe", "P1 PIT index membership", "trade_date",
        ("index_code", "trade_date", "con_code"), "asof_snapshot", "p4_2_cached",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "suspend_d", "tradability", "P2-2 suspension flag", "trade_date",
        ("ts_code", "trade_date", "suspend_type"), "eod", "p4_2_cached",
        "proven_in_project_runs", "empty_authorized",
    ),
    EndpointCapability(
        "namechange", "tradability", "P2-2 ST name intervals", "start_date",
        ("ts_code", "start_date", "end_date", "name"), "dimension_snapshot",
        "p4_2_cached", "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "stock_basic", "universe", "P2-2 list_date (min_listing_days)", "list_date",
        ("ts_code", "list_date"), "dimension_snapshot", "p4_2_cached",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "stk_limit", "tradability", "P2-2 raw price limits", "trade_date",
        ("ts_code", "trade_date", "up_limit", "down_limit"), "eod", "p4_2_cached",
        "proven_in_project_runs", "authorized",
    ),
    # -- covariates / financials (P4-3 planned cache) --------------------- #
    EndpointCapability(
        "daily_basic", "covariate", "P2-3/P3-5 pe/pb/total_mv", "trade_date",
        ("ts_code", "trade_date"), "same_day_publish", "p4_3_planned",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "fina_indicator", "financial", "P3-1 roe/np_yoy (ann_date as-of)", "end_date",
        ("ts_code", "end_date", "ann_date"), "ann_date_disclosure", "p4_3_planned",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "income", "financial", "financial source (ann_date)", "end_date",
        ("ts_code", "end_date", "ann_date"), "ann_date_disclosure", "p4_3_planned",
        "proven_in_project_runs", "authorized",
    ),
    EndpointCapability(
        "index_member_all", "covariate", "P2-3 PIT SW industry", "in_date",
        ("ts_code", "in_date", "out_date", "l1_name"), "asof_interval", "p4_3_planned",
        "proven_in_project_runs", "authorized",
    ),
    # -- EXPLORATORY candidate factor sources (authorized; rate UNVERIFIED) #
    EndpointCapability(
        "stk_factor", "special_factor", "EXPLORATORY technical factors", "trade_date",
        ("ts_code", "trade_date"), "eod", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "stk_factor_pro", "special_factor", "EXPLORATORY technical factors pro",
        "trade_date", ("ts_code", "trade_date"), "eod", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "cyq_perf", "special_factor", "EXPLORATORY chip cost / winner rate",
        "trade_date", ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "cyq_chips", "special_factor", "EXPLORATORY chip distribution", "trade_date",
        ("ts_code", "trade_date", "price", "percent"), "eod_postclose",
        "candidate_uncached", "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "moneyflow", "moneyflow", "EXPLORATORY money flow (lg/sm)", "trade_date",
        ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "moneyflow_dc", "moneyflow", "EXPLORATORY money flow (DC)", "trade_date",
        ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "moneyflow_ths", "moneyflow", "EXPLORATORY money flow (THS)", "trade_date",
        ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "empty_authorized",
    ),
    EndpointCapability(
        "moneyflow_hsgt", "moneyflow", "EXPLORATORY north/south bound flow",
        "trade_date", ("trade_date", "north_money"), "eod_postclose",
        "candidate_uncached", "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "limit_list_d", "sentiment", "EXPLORATORY limit-up/down stats", "trade_date",
        ("trade_date", "ts_code"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "top_list", "sentiment", "EXPLORATORY dragon-tiger board", "trade_date",
        ("trade_date", "ts_code"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "empty_authorized",
    ),
    EndpointCapability(
        "block_trade", "reference", "block trades", "trade_date",
        ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "empty_authorized",
    ),
    EndpointCapability(
        "margin_detail", "reference", "margin trading detail", "trade_date",
        ("ts_code", "trade_date"), "eod_postclose", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    EndpointCapability(
        "stk_surv", "alt_data", "institutional research records", "surv_date",
        ("ts_code", "surv_date"), "event_date", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    # -- minute bars: the 1-call/hour cap was LIFTED (re-probe 2026-06-13,
    #    3/3 calls in ~12s) -> now authorized + batch-plausible; the exact rate
    #    ceiling was NOT stress-tested (no brute force).
    EndpointCapability(
        "stk_mins", "market_intraday", "EXPLORATORY minute bars (roadmap)",
        "trade_time", ("ts_code", "trade_time"), "intraday", "candidate_uncached",
        "authorized_rate_unverified", "authorized",
    ),
    # -- throttled to 1 call/hour (NOT batch-viable) ---------------------- #
    EndpointCapability(
        "hm_detail", "sentiment", "EXPLORATORY hot-money detail", "trade_date",
        ("trade_date", "ts_code", "hm_name"), "eod_postclose", "candidate_uncached",
        "not_batch_viable_1_per_hour", "rate_limited",
    ),
    # -- no permission (40203) -------------------------------------------- #
    EndpointCapability(
        "stk_auction_o", "auction", "open call-auction", "trade_date",
        ("ts_code", "trade_date"), "intraday", "not_applicable",
        "no_access", "no_permission",
    ),
    EndpointCapability(
        "stk_auction_c", "auction", "close call-auction", "trade_date",
        ("ts_code", "trade_date"), "intraday", "not_applicable",
        "no_access", "no_permission",
    ),
)

CAPABILITIES: dict[str, EndpointCapability] = {c.name: c for c in _CAPS}


def get(name: str) -> EndpointCapability | None:
    """Return the capability record for ``name`` (or None if unknown)."""
    return CAPABILITIES.get(name)


def names() -> tuple[str, ...]:
    """All registered endpoint names, in declaration order."""
    return tuple(c.name for c in _CAPS)


def by_category(category: str) -> tuple[EndpointCapability, ...]:
    """All capabilities in ``category`` (declaration order)."""
    return tuple(c for c in _CAPS if c.category == category)
