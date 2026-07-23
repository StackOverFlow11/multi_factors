"""Tests for the D0 availability policy table (``data/availability_policy.py``).

Network-free. These tests are the independent witness of the paper contract:
the expected endpoint sets, field splits, and DOMAIN ASSUMPTION placements are
hard-coded HERE (not derived from the module) so a silent table edit fails a
test instead of shipping. Derivation tests are mutation-style: they assert the
horizon MOVES when the injected config value moves (R5), not just a value.
"""

from __future__ import annotations

import dataclasses

import pytest

from data.availability_policy import (
    ADJ_FACTOR,
    AVAILABILITY_POLICY,
    Adjustment,
    CloseVisibility,
    DAILY_BASIC,
    DecisionVisibility,
    FINA_INDICATOR,
    INDEX_MEMBER_ALL,
    INDEX_WEIGHT,
    KNOWN_SOURCES,
    LEGAL_VIEW_BASIS_PAIRS,
    MARKET_DAILY,
    MARKET_DAILY_FIELDS,
    NAMECHANGE,
    OvernightBoundary,
    ReturnBasis,
    RevisionHorizonSource,
    STK_LIMIT,
    STK_MINS_1MIN,
    STOCK_BASIC,
    SUSPEND_D,
    View,
    require_known_source,
    require_legal_pairing,
    revision_horizon,
    rule_for,
    rules_for_source,
)

# Endpoints that never feed the factor store (horizon None).
_NON_FACTOR_INPUTS = (
    STK_LIMIT, SUSPEND_D, NAMECHANGE, INDEX_WEIGHT, INDEX_MEMBER_ALL,
    STOCK_BASIC,
)


# ---------------------------------------------------------------------------
# Table coverage + endpoint closure (R8).
# ---------------------------------------------------------------------------


def test_known_sources_cover_current_factor_inputs():
    """Every endpoint the current 14-factor roster requires is declared."""
    for source in (
        MARKET_DAILY, ADJ_FACTOR, STK_MINS_1MIN, DAILY_BASIC, FINA_INDICATOR,
    ):
        assert require_known_source(source) == source
    # market_daily is declared per field (R7) — all six fields present.
    for field in ("open", "close", "high", "low", "volume", "amount"):
        assert rule_for(MARKET_DAILY, field=field).field == field


def test_endpoint_ids_match_the_cache_layer():
    """Drift guard: the leaf module repeats (not imports) the cache ids."""
    from data.cache import intraday_cache, tushare_specs

    assert KNOWN_SOURCES - {STK_MINS_1MIN} == set(tushare_specs.ALL_ENDPOINTS)
    assert STK_MINS_1MIN == intraday_cache.ENDPOINT


def test_unknown_source_raises_and_lists_known_endpoints():
    with pytest.raises(ValueError) as excinfo:
        require_known_source("moneyflow")
    message = str(excinfo.value)
    assert "moneyflow" in message
    # The error must teach: it lists the declared endpoints.
    assert "market_daily" in message
    assert "daily_basic" in message
    assert "fina_indicator" in message


def test_unknown_source_is_rejected_by_every_entry_point():
    with pytest.raises(ValueError, match="moneyflow"):
        rule_for("moneyflow")
    with pytest.raises(ValueError, match="moneyflow"):
        rules_for_source("moneyflow")
    with pytest.raises(ValueError, match="moneyflow"):
        revision_horizon(
            "moneyflow", refresh_recent_days=14, recent_tail_overrides={}
        )


# ---------------------------------------------------------------------------
# Field-level market_daily split (R7).
# ---------------------------------------------------------------------------


def test_market_daily_open_is_decision_same_day():
    rule = rule_for(MARKET_DAILY, field="open")
    assert rule.decision_visibility is DecisionVisibility.SAME_DAY
    assert rule.close_visibility is CloseVisibility.SAME_DAY


def test_market_daily_full_day_fields_are_decision_prev_day():
    """Only ``open`` is final by 14:50; every other field waits for d-1."""
    for field in ("close", "high", "low", "volume", "amount"):
        rule = rule_for(MARKET_DAILY, field=field)
        assert rule.decision_visibility is DecisionVisibility.PREV_DAY, field
        assert rule.close_visibility is CloseVisibility.SAME_DAY, field


def test_market_daily_requires_a_field():
    with pytest.raises(ValueError, match="FIELD-level"):
        rule_for(MARKET_DAILY)


def test_market_daily_unknown_field_raises_listing_fields():
    with pytest.raises(ValueError) as excinfo:
        rule_for(MARKET_DAILY, field="vwap")
    message = str(excinfo.value)
    assert "vwap" in message
    assert "open" in message and "amount" in message


def test_endpoint_level_lookup_ignores_field():
    """Availability is endpoint-level everywhere else; a field is harmless."""
    assert rule_for(DAILY_BASIC, field="pe") is rule_for(DAILY_BASIC)


def test_rules_for_source_returns_six_market_rows_one_otherwise():
    assert len(rules_for_source(MARKET_DAILY)) == len(MARKET_DAILY_FIELDS)
    assert len(rules_for_source(DAILY_BASIC)) == 1


# ---------------------------------------------------------------------------
# Minute bars and the remaining decision rules.
# ---------------------------------------------------------------------------


def test_minute_bars_use_intraday_cutoff_and_session_close():
    rule = rule_for(STK_MINS_1MIN)
    assert rule.decision_visibility is DecisionVisibility.INTRADAY_CUTOFF
    assert rule.close_visibility is CloseVisibility.SESSION_CLOSE


def test_fina_is_ann_date_based_and_conservative_in_the_decision_view():
    rule = rule_for(FINA_INDICATOR)
    assert rule.decision_visibility is DecisionVisibility.ANN_DATE_PREV_DAY
    assert rule.close_visibility is CloseVisibility.ANN_DATE_SAME_DAY


def test_adj_factor_and_daily_basic_are_decision_prev_day():
    for source in (ADJ_FACTOR, DAILY_BASIC):
        assert (
            rule_for(source).decision_visibility is DecisionVisibility.PREV_DAY
        ), source


# ---------------------------------------------------------------------------
# Revision horizon derivation (R5): mutation-style, config-injected.
# ---------------------------------------------------------------------------


def test_revision_horizon_market_family_follows_refresh_recent_days():
    """Widen the cache refresh tail -> the horizon widens in lockstep."""
    overrides = {FINA_INDICATOR: 400}
    for source in (MARKET_DAILY, ADJ_FACTOR, DAILY_BASIC):
        at_14 = revision_horizon(
            source, refresh_recent_days=14, recent_tail_overrides=overrides
        )
        at_21 = revision_horizon(
            source, refresh_recent_days=21, recent_tail_overrides=overrides
        )
        assert at_14 == 14, source
        assert at_21 == 21, source  # the mutation moved -> the horizon moved


def test_fina_horizon_equals_the_override_value():
    for tail in (400, 250):
        got = revision_horizon(
            FINA_INDICATOR,
            refresh_recent_days=14,
            recent_tail_overrides={FINA_INDICATOR: tail},
        )
        assert got == tail  # never refresh_recent_days


def test_fina_missing_override_raises_never_falls_back():
    """R5: the silent fallback to refresh_recent_days is the main defect."""
    with pytest.raises(ValueError) as excinfo:
        revision_horizon(
            FINA_INDICATOR, refresh_recent_days=14, recent_tail_overrides={}
        )
    message = str(excinfo.value)
    assert "fina_indicator" in message
    assert "recent_tail_overrides" in message
    assert "refresh_recent_days" in message  # names the refused fallback


def test_minute_horizon_is_always_zero():
    """Written-once bars: the 0 is semantic, not configuration."""
    for refresh in (0, 14, 21, 999):
        assert (
            revision_horizon(
                STK_MINS_1MIN,
                refresh_recent_days=refresh,
                recent_tail_overrides={},
            )
            == 0
        )


def test_non_factor_input_endpoints_have_no_horizon():
    for source in _NON_FACTOR_INPUTS:
        assert (
            revision_horizon(
                source, refresh_recent_days=14, recent_tail_overrides={}
            )
            is None
        ), source


def test_negative_refresh_days_rejected():
    with pytest.raises(ValueError, match="refresh_recent_days"):
        revision_horizon(
            MARKET_DAILY, refresh_recent_days=-1, recent_tail_overrides={}
        )


def test_negative_or_non_int_override_rejected():
    with pytest.raises(ValueError, match="fina_indicator"):
        revision_horizon(
            FINA_INDICATOR,
            refresh_recent_days=14,
            recent_tail_overrides={FINA_INDICATOR: -400},
        )
    with pytest.raises(ValueError, match="fina_indicator"):
        revision_horizon(
            FINA_INDICATOR,
            refresh_recent_days=14,
            recent_tail_overrides={FINA_INDICATOR: "400"},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# View x return-basis pairing legality (§1.4 mechanism 1).
# ---------------------------------------------------------------------------


def test_legal_pairings_pass():
    assert require_legal_pairing("decision", "exec_to_exec") == (
        View.DECISION,
        ReturnBasis.EXEC_TO_EXEC,
    )
    assert require_legal_pairing(View.CLOSE, ReturnBasis.CLOSE_TO_CLOSE) == (
        View.CLOSE,
        ReturnBasis.CLOSE_TO_CLOSE,
    )
    assert len(LEGAL_VIEW_BASIS_PAIRS) == 2


def test_illegal_pairings_raise_readably():
    for view, basis in (
        ("close", "exec_to_exec"),
        ("decision", "close_to_close"),
    ):
        with pytest.raises(ValueError) as excinfo:
            require_legal_pairing(view, basis)
        message = str(excinfo.value)
        assert "illegal view/basis pairing" in message
        # The error must teach both legal pairs.
        assert "exec_to_exec" in message
        assert "close_to_close" in message


def test_unknown_view_or_basis_raise():
    with pytest.raises(ValueError, match="view"):
        require_legal_pairing("intraday", "exec_to_exec")
    with pytest.raises(ValueError, match="return basis"):
        require_legal_pairing("decision", "open_to_open")


# ---------------------------------------------------------------------------
# DOMAIN ASSUMPTION flags (R2): exactly the three declared assumptions.
# ---------------------------------------------------------------------------


def test_domain_assumption_exactly_at_the_declared_places():
    """Three declared assumptions; suspend/ST share one -> four True rows."""
    expected = {DAILY_BASIC, STK_LIMIT, SUSPEND_D, NAMECHANGE}
    flagged = {r.source for r in AVAILABILITY_POLICY.values() if r.domain_assumption}
    assert flagged == expected
    # Reverse direction: every other row is code-provable (False).
    for rule in AVAILABILITY_POLICY.values():
        if rule.source not in expected:
            assert not rule.domain_assumption, rule.source


# ---------------------------------------------------------------------------
# Frozen / immutable structures.
# ---------------------------------------------------------------------------


def test_rules_are_frozen():
    rule = rule_for(DAILY_BASIC)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.domain_assumption = False  # type: ignore[misc]


def test_policy_mapping_is_immutable():
    with pytest.raises(TypeError):
        AVAILABILITY_POLICY[(DAILY_BASIC, None)] = rule_for(DAILY_BASIC)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Taxonomy enums (single source for the D1 FactorSpec fields).
# ---------------------------------------------------------------------------


def test_taxonomy_enums_are_three_by_three():
    assert {m.value for m in Adjustment} == {
        "none", "returns_invariant", "price_level",
    }
    assert {m.value for m in OvernightBoundary} == {
        "none", "crossed_disclosed", "masked",
    }


def test_views_are_a_closed_two_value_enum():
    assert {m.value for m in View} == {"decision", "close"}
    assert {m.value for m in ReturnBasis} == {"exec_to_exec", "close_to_close"}


def test_horizon_source_is_consistent_per_endpoint():
    """All six market_daily field rows share one horizon source."""
    sources = {r.horizon_source for r in rules_for_source(MARKET_DAILY)}
    assert sources == {RevisionHorizonSource.REFRESH_RECENT_DAYS}
