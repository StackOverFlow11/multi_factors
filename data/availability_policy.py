"""Availability policy for the two-view contract (factor-refactor step D0).

SINGLE SOURCE OF TRUTH for the availability policy table of the factor-layer
refactor design v3.2 §1.3 (``tmp/design/factor_refactor_design_v3.md``).
Documents and reports QUOTE this module (constant names, enum values); they
never restate the table — author once (the #76/#78/#82 lesson applied to data
semantics). The taxonomy enums the future FactorSpec fields (``adjustment``,
``overnight_boundary``) will reference and the view x return-basis pairing
constants also live here: one import site for D1+.

Views are a CLOSED two-value enum, never a parameter space: ``decision(d)``
= the information set at the 14:50 decision on day ``d``; ``close(d)`` = the
full-day set (close + evening releases).

DOMAIN ASSUMPTIONS (design R2) — exactly three declared assumptions, carried
as first-class flags and never mixed silently with code-provable facts:
(1) ``daily_basic`` is published in the EVENING of ``d`` (decision view backs
off to <= d-1, so it is not load-bearing); (2) ``stk_limit`` for ``d`` is
published BEFORE the open; (3) suspensions / ST flags (``suspend_d`` /
``namechange``) are announced BEFORE the open — one assumption covering two
endpoints, so the flag is True on four rows for three assumptions.

Revision horizons are NEVER hardcoded here (design R5): callers pass the
live cache config values (``refresh_recent_days``, ``recent_tail_overrides``)
as plain values and this module only routes them per endpoint. The single
exception is the minute endpoint's constant 0, which encodes the SEMANTIC
fact that a written 1min bar never changes — not a configuration choice.

Leaf-module discipline: standard library only. Never import qt, feeds,
caches, or pandas — the D4 materializer and D1 registry sit ABOVE this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Mapping

# Views, return bases, and the legal pairing (design §1.4 mechanism 1).
@unique
class View(StrEnum):
    """The two legal information-set views. Closed: anything else is an error."""

    DECISION = "decision"
    CLOSE = "close"


@unique
class ReturnBasis(StrEnum):
    """The two return bases a view may legally pair with."""

    EXEC_TO_EXEC = "exec_to_exec"
    CLOSE_TO_CLOSE = "close_to_close"


#: The ONLY legal view x basis pairs: decision pairs with exec-to-exec
#: (information at 14:50 precedes the 14:51 execution anchor) and close pairs
#: with close-to-close (same instant, declared as the legacy convention).
LEGAL_VIEW_BASIS_PAIRS: frozenset[tuple[View, ReturnBasis]] = frozenset({
    (View.DECISION, ReturnBasis.EXEC_TO_EXEC),
    (View.CLOSE, ReturnBasis.CLOSE_TO_CLOSE),
})


def _coerce_enum(enum_cls: type[StrEnum], value: object, what: str) -> StrEnum:
    """Coerce ``value`` into ``enum_cls`` with a readable error."""
    try:
        return enum_cls(value)  # type: ignore[arg-type]
    except ValueError:
        allowed = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(
            f"unknown {what} {value!r}; allowed values: {allowed}."
        ) from None


def require_legal_pairing(view: object, basis: object) -> tuple[View, ReturnBasis]:
    """Validate a view x return-basis pairing at construction time: anything
    but the two legal pairs — most importantly the close view scored against
    the 14:51 execution — raises readably instead of being a doc convention.
    """
    v = _coerce_enum(View, view, "view")
    b = _coerce_enum(ReturnBasis, basis, "return basis")
    if (v, b) not in LEGAL_VIEW_BASIS_PAIRS:
        legal = " or ".join(f"({pv.value!r}, {pb.value!r})"
                            for pv, pb in sorted(LEGAL_VIEW_BASIS_PAIRS))
        raise ValueError(
            f"illegal view/basis pairing ({v.value!r}, {b.value!r}); the only "
            f"legal pairings are {legal}. A close-view factor scored on "
            f"exec-to-exec returns (or a decision-view factor scored "
            f"close-to-close) mixes information sets across the execution "
            f"anchor."
        )
    return (v, b)


# Taxonomy enums referenced by the future FactorSpec (design §3.4 / §1.3 note).
@unique
class Adjustment(StrEnum):
    """How a factor's stored values relate to the price-adjustment anchor.

    ``NONE``: never touches the price channel (volume/amount/published ratios
    only); anchor perturbation is trivially invisible. ``RETURNS_INVARIANT``:
    consumes prices only through ratios in which the anchor cancels; must pass
    the perturb-the-anchor -> value-unchanged test. ``PRICE_LEVEL``: depends
    on the adjusted price LEVEL; its store fingerprint must include the
    per-symbol adj_factor event table (mechanism stays fixture-tested even at
    zero current members).
    """

    NONE = "none"
    RETURNS_INVARIANT = "returns_invariant"
    PRICE_LEVEL = "price_level"


@unique
class OvernightBoundary(StrEnum):
    """How a factor's day-d value relates to the ex-date basis break.

    ``NONE``: no raw-price comparison crosses the overnight boundary; must
    pass the rescale-the-prior-basis -> day-d-value-unchanged test.
    ``CROSSED_DISCLOSED``: the pinned definition deliberately crosses, ex-date
    values are kept, and the deviation is measured and disclosed. ``MASKED``:
    the materializer NaNs ex-dates (with a distribution disclosure); zero
    current members — applying it to an existing factor is a definition
    change.
    """

    NONE = "none"
    CROSSED_DISCLOSED = "crossed_disclosed"
    MASKED = "masked"


# Endpoint identifiers — string literals mirroring the cache layer's ids
# (data/cache/tushare_specs.py; data/cache/intraday_cache.py::ENDPOINT).
# Repeated, not imported (leaf module); a drift test keeps the sets identical.

MARKET_DAILY = "market_daily"
ADJ_FACTOR = "adj_factor"
STK_MINS_1MIN = "stk_mins_1min"
DAILY_BASIC = "daily_basic"
FINA_INDICATOR = "fina_indicator"
STK_LIMIT = "stk_limit"
SUSPEND_D = "suspend_d"
NAMECHANGE = "namechange"
INDEX_WEIGHT = "index_weight"
INDEX_MEMBER_ALL = "index_member_all"
STOCK_BASIC = "stock_basic"

#: market_daily availability is FIELD-level (design R7): ``open`` is the only
#: field final by the 14:50 decision; high/low/volume/amount (and close) keep
#: moving until 15:00, so they are decision-visible only at <= d-1.
MARKET_DAILY_FIELDS: tuple[str, ...] = ("open", "close", "high", "low", "volume", "amount")


@unique
class CloseVisibility(StrEnum):
    """When day-d data becomes part of the close(d) information set."""

    SAME_DAY = "same_day"                  # rows dated <= d
    SESSION_CLOSE = "session_close"        # minute bars up to d 15:00
    ANN_DATE_SAME_DAY = "ann_date_same_day"  # ann_date <= d
    AS_OF_DAY = "as_of_day"                # PIT dimension, as-of d
    STATIC = "static"                      # effectively constant metadata


@unique
class DecisionVisibility(StrEnum):
    """When data becomes part of the decision(d) 14:50 information set."""

    SAME_DAY = "same_day"                  # day d itself is legal at 14:50
    PREV_DAY = "prev_day"                  # only rows dated <= d-1
    INTRADAY_CUTOFF = "intraday_cutoff"    # available_time <= d 14:50
    ANN_DATE_PREV_DAY = "ann_date_prev_day"  # ann_date <= d-1
    AS_OF_DAY = "as_of_day"                # PIT dimension, as-of d
    STATIC = "static"


@unique
class RevisionHorizonSource(StrEnum):
    """Where an endpoint's revision horizon (overlap-window input) comes from."""

    REFRESH_RECENT_DAYS = "refresh_recent_days"   # cache recent-tail refetch
    FINA_TAIL_OVERRIDE = "fina_tail_override"     # late-disclosure backfill tail
    IMMUTABLE = "immutable"                       # written-once data: horizon 0
    NOT_FACTOR_INPUT = "not_factor_input"         # never feeds the factor store


@dataclass(frozen=True, slots=True)
class AvailabilityRule:
    """One frozen row of the §1.3 availability policy table."""

    source: str
    field: str | None
    close_visibility: CloseVisibility
    decision_visibility: DecisionVisibility
    domain_assumption: bool
    horizon_source: RevisionHorizonSource
    rationale: str


_CV, _DV, _HS = CloseVisibility, DecisionVisibility, RevisionHorizonSource


def _rule(
    source: str, close_v: CloseVisibility, decision_v: DecisionVisibility,
    horizon: RevisionHorizonSource, why: str, *,
    field: str | None = None, assumed: bool = False,
) -> AvailabilityRule:
    return AvailabilityRule(
        source=source, field=field, close_visibility=close_v,
        decision_visibility=decision_v, domain_assumption=assumed,
        horizon_source=horizon, rationale=why,
    )


_OPEN_WHY = (
    "The open is fixed at 09:30 and known by 14:50; blocking it would make "
    "overnight-style factors uncomputable in the decision view, allowing the "
    "full row would be lookahead — field-level is the only split that "
    "neither leaks nor over-blocks (R7)."
)
_FULL_DAY_WHY = (
    "Full-day value only final at the 15:00 close; high/low/volume/amount "
    "keep moving between 14:50 and 15:00 — only the open is final intraday."
)

_RULES: tuple[AvailabilityRule, ...] = (
    _rule(MARKET_DAILY, _CV.SAME_DAY, _DV.SAME_DAY, _HS.REFRESH_RECENT_DAYS,
          _OPEN_WHY, field="open"),
    *(
        _rule(MARKET_DAILY, _CV.SAME_DAY, _DV.PREV_DAY, _HS.REFRESH_RECENT_DAYS,
              _FULL_DAY_WHY, field=f)
        for f in ("close", "high", "low", "volume", "amount")
    ),
    _rule(ADJ_FACTOR, _CV.SAME_DAY, _DV.PREV_DAY, _HS.REFRESH_RECENT_DAYS,
          "Published with the same daily batch as the close. The ex-date basis "
          "break this implies for day d is governed by the OvernightBoundary "
          "taxonomy, not by relaxing this row."),
    _rule(STK_MINS_1MIN, _CV.SESSION_CLOSE, _DV.INTRADAY_CUTOFF, _HS.IMMUTABLE,
          "The I1/I3 machinery already enforces available_time <= cutoff; a "
          "written 1min bar never changes, so the horizon is the semantic "
          "constant 0 (not configuration)."),
    _rule(DAILY_BASIC, _CV.SAME_DAY, _DV.PREV_DAY, _HS.REFRESH_RECENT_DAYS,
          "DOMAIN ASSUMPTION: published in the evening (not provable from "
          "code). The decision view already backs off to <= d-1, so the "
          "assumption is not load-bearing.", assumed=True),
    _rule(FINA_INDICATOR, _CV.ANN_DATE_SAME_DAY, _DV.ANN_DATE_PREV_DAY,
          _HS.FINA_TAIL_OVERRIDE,
          "ann_date is day-granular (code-provable) and cannot prove the "
          "release happened before 14:50, hence the conservative d-1. The "
          "horizon is the late-disclosure backfill tail, supplied by the "
          "caller from the cache config."),
    _rule(STK_LIMIT, _CV.SAME_DAY, _DV.SAME_DAY, _HS.NOT_FACTOR_INPUT,
          "Band derived from the previous close (market fact) + DOMAIN "
          "ASSUMPTION: published before the open. 'd legal' has no "
          "conservative fallback, so becoming a factor input requires a "
          "spec-level declaration-is-testable assertion first.", assumed=True),
    _rule(SUSPEND_D, _CV.SAME_DAY, _DV.SAME_DAY, _HS.NOT_FACTOR_INPUT,
          "DOMAIN ASSUMPTION: suspensions are announced before the open; "
          "intraday ad-hoc suspensions are a known uncovered corner.",
          assumed=True),
    _rule(NAMECHANGE, _CV.SAME_DAY, _DV.SAME_DAY, _HS.NOT_FACTOR_INPUT,
          "ST flag intervals; DOMAIN ASSUMPTION shared with suspend_d: "
          "announced before the open.", assumed=True),
    _rule(INDEX_WEIGHT, _CV.AS_OF_DAY, _DV.AS_OF_DAY, _HS.NOT_FACTOR_INPUT,
          "Low-frequency PIT dimension; no intraday concern."),
    _rule(INDEX_MEMBER_ALL, _CV.AS_OF_DAY, _DV.AS_OF_DAY, _HS.NOT_FACTOR_INPUT,
          "SW industry in/out intervals; PIT as-of, no intraday concern."),
    _rule(STOCK_BASIC, _CV.STATIC, _DV.STATIC, _HS.NOT_FACTOR_INPUT,
          "list_date only; effectively static metadata."),
)


def _build_index() -> tuple[
    Mapping[tuple[str, str | None], AvailabilityRule],
    frozenset[str],
    Mapping[str, RevisionHorizonSource],
]:
    by_key: dict[tuple[str, str | None], AvailabilityRule] = {}
    horizon_by_source: dict[str, RevisionHorizonSource] = {}
    for rule in _RULES:
        key = (rule.source, rule.field)
        if key in by_key:
            raise RuntimeError(f"duplicate availability rule for {key!r}.")
        by_key[key] = rule
        seen = horizon_by_source.get(rule.source)
        if seen is not None and seen is not rule.horizon_source:
            raise RuntimeError(
                f"inconsistent horizon_source for endpoint {rule.source!r}: "
                f"{seen.value!r} vs {rule.horizon_source.value!r}."
            )
        horizon_by_source[rule.source] = rule.horizon_source
    market_fields = {f for s, f in by_key if s == MARKET_DAILY}
    if market_fields != set(MARKET_DAILY_FIELDS):
        raise RuntimeError(
            "market_daily rows must cover exactly MARKET_DAILY_FIELDS; got "
            f"{sorted(market_fields, key=str)!r}."
        )
    return (MappingProxyType(by_key), frozenset(horizon_by_source),
            MappingProxyType(horizon_by_source))


#: (source, field) -> rule; market_daily keyed per field, others field=None.
AVAILABILITY_POLICY, KNOWN_SOURCES, _HORIZON_SOURCE_BY_ENDPOINT = _build_index()


def require_known_source(source: str) -> str:
    """R8 endpoint closure: an undeclared endpoint has NO availability rule,
    and a close-like fallback for it would be a field-level lookahead entry
    point — so unknown sources are a readable error, never a dict.get default.
    """
    if source not in KNOWN_SOURCES:
        raise ValueError(
            f"unknown availability source {source!r}; the availability policy "
            f"table declares only: {sorted(KNOWN_SOURCES)}. Add the endpoint "
            f"to data/availability_policy.py (with its visibility rules and "
            f"revision-horizon source) before any factor may require it."
        )
    return source


def rules_for_source(source: str) -> tuple[AvailabilityRule, ...]:
    """All rules of one endpoint (six for market_daily, one otherwise)."""
    require_known_source(source)
    return tuple(r for r in _RULES if r.source == source)


def rule_for(source: str, field: str | None = None) -> AvailabilityRule:
    """Resolve the availability rule for ``(source, field)``.

    market_daily REQUIRES a field (availability is field-level there, R7).
    For every other endpoint availability is endpoint-level, so ``field`` is
    accepted and ignored (field-name validity is the feed's business).
    """
    require_known_source(source)
    if source == MARKET_DAILY:
        if field is None:
            raise ValueError(
                "market_daily availability is FIELD-level (R7): pass "
                f"field=... with one of {list(MARKET_DAILY_FIELDS)}. Only "
                "'open' is final by the 14:50 decision."
            )
        if field not in MARKET_DAILY_FIELDS:
            raise ValueError(
                f"unknown market_daily field {field!r}; known fields: "
                f"{list(MARKET_DAILY_FIELDS)}."
            )
        return AVAILABILITY_POLICY[(source, field)]
    return AVAILABILITY_POLICY[(source, None)]


def _require_day_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int (trading-day count); got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}.")
    return value


def revision_horizon(
    source: str,
    *,
    refresh_recent_days: int,
    recent_tail_overrides: Mapping[str, int],
) -> int | None:
    """Resolve an endpoint's revision horizon from the LIVE cache config.

    The horizon is the trailing trading-day window within which cached rows
    may still be rewritten upstream — the sole input of the incremental
    overlap window (design §3.3). Values are caller-supplied so the horizon
    always tracks the actual cache config (R5). Returns None for endpoints
    that never feed the factor store. Raises ValueError on an unknown source,
    a negative/non-int day count, or a fina_indicator lookup whose override
    is missing — silently falling back to ``refresh_recent_days`` is exactly
    the defect shape R5 found on main, so it is refused loudly.
    """
    require_known_source(source)
    horizon_source = _HORIZON_SOURCE_BY_ENDPOINT[source]
    if horizon_source is RevisionHorizonSource.NOT_FACTOR_INPUT:
        return None
    if horizon_source is RevisionHorizonSource.IMMUTABLE:
        return 0
    if horizon_source is RevisionHorizonSource.REFRESH_RECENT_DAYS:
        return _require_day_count("refresh_recent_days", refresh_recent_days)
    # RevisionHorizonSource.FINA_TAIL_OVERRIDE
    if FINA_INDICATOR not in recent_tail_overrides:
        raise ValueError(
            "revision horizon for 'fina_indicator' must come from "
            "recent_tail_overrides['fina_indicator'] (the late-disclosure "
            "refetch tail), and the override is missing. Refusing to fall "
            "back to refresh_recent_days: that silent fallback would "
            "understate the fina overlap window (R5). Pass the same override "
            "mapping the cache builder uses."
        )
    return _require_day_count(
        "recent_tail_overrides['fina_indicator']",
        recent_tail_overrides[FINA_INDICATOR],
    )
