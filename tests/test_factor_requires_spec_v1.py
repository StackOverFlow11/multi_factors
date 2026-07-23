"""PanelField + FactorSpec contract v1.0 (factor-refactor D1).

Locks the two new declaration layers:

  * :class:`factors.requires.PanelField` — endpoint-only requires entries with
    R8 endpoint closure (unknown source -> readable error at DECLARATION time)
    and R7 field-level validation for ``market_daily``.
  * FactorSpec contract v1.0 — ``requires`` / ``adjustment`` /
    ``overnight_boundary`` are MANDATORY, value domains are the
    ``data.availability_policy`` enums (single source), and the D0 §1.1
    static check (adjustment='none' must not require a price-channel field)
    fires at construction.

Plus the D0 §2 pre-assignment table lock: every shipped factor's declarations
must match the reviewed paper contract row-for-row — a silent drift of any
declaration is exactly what these rows would catch.
"""

from __future__ import annotations

import pytest

from data.availability_policy import (
    Adjustment,
    OvernightBoundary,
)
from factors.compute.candidates import (
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import FinancialFactor
from factors.compute.intraday_derived import (
    AmpMarginalAnomalyVolFactor,
    IntradayAmpCutFactor,
    JumpAmountCorrFactor,
    MinuteIdealAmplitudeFactor,
    PeakIntervalKurtosisFactor,
    PeakRidgeAmountRatioFactor,
    RidgeMinuteReturnFactor,
    ValleyPriceQuantileFactor,
    ValleyRelativeVwapFactor,
    ValleyRidgeVwapRatioFactor,
    VolumePeakCountFactor,
)
from factors.compute.momentum import MomentumFactor
from factors.spec import FactorSpec, PanelField


def _spec(**overrides) -> FactorSpec:
    base = dict(
        factor_id="unit_test_factor",
        version="1.0",
        description="A test factor.",
        expected_ic_sign=+1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="market_daily"),),
        adjustment="returns_invariant",
        overnight_boundary="none",
    )
    base.update(overrides)
    return FactorSpec(**base)


# --------------------------------------------------------------------------- #
# PanelField
# --------------------------------------------------------------------------- #
def test_panel_field_is_frozen_and_hashable():
    pf = PanelField("close", source="market_daily")
    assert (pf.field, pf.source) == ("close", "market_daily")
    with pytest.raises(Exception):  # frozen dataclass forbids mutation
        pf.field = "open"
    assert PanelField("close", source="market_daily") == pf
    assert len({pf, PanelField("close", source="market_daily")}) == 1


def test_panel_field_unknown_source_is_a_readable_error():
    # R8 endpoint closure at DECLARATION time: an undeclared endpoint must
    # never be dict.get-defaulted into close-like availability downstream.
    with pytest.raises(ValueError, match="unknown availability source"):
        PanelField("net_mf_amount", source="moneyflow")


def test_panel_field_market_daily_field_level_validation():
    # R7: market_daily availability is field-level, so an unknown field has
    # no availability rule and must be rejected at declaration time.
    with pytest.raises(ValueError, match="unknown market_daily field"):
        PanelField("vwap", source="market_daily")
    # All six declared fields are constructible.
    for field in ("open", "close", "high", "low", "volume", "amount"):
        PanelField(field, source="market_daily")


def test_panel_field_non_market_endpoints_accept_feed_level_fields():
    # Field-name validity is the feed's business for endpoint-level sources
    # (documented in data.availability_policy.rule_for).
    PanelField("pe", source="daily_basic")
    PanelField("roe", source="fina_indicator")
    PanelField("volume", source="stk_mins_1min")


def test_panel_field_blank_parts_are_rejected():
    with pytest.raises(ValueError, match="field"):
        PanelField("", source="market_daily")
    with pytest.raises(ValueError, match="source"):
        PanelField("close", source="   ")


# --------------------------------------------------------------------------- #
# FactorSpec contract v1.0: the three declarations are MANDATORY
# --------------------------------------------------------------------------- #
def test_missing_requires_is_a_readable_error():
    with pytest.raises(ValueError, match="missing the 'requires' declaration"):
        _spec(requires=None)


def test_missing_adjustment_is_a_readable_error():
    with pytest.raises(ValueError, match="missing the 'adjustment' declaration"):
        _spec(adjustment=None)


def test_missing_overnight_boundary_is_a_readable_error():
    with pytest.raises(
        ValueError, match="missing the 'overnight_boundary' declaration"
    ):
        _spec(overnight_boundary=None)


def test_taxonomy_accepts_enum_members_and_strings_and_stores_enums():
    a = _spec(adjustment=Adjustment.RETURNS_INVARIANT,
              overnight_boundary=OvernightBoundary.NONE)
    b = _spec(adjustment="returns_invariant", overnight_boundary="none")
    assert a.adjustment is Adjustment.RETURNS_INVARIANT
    assert b.adjustment is Adjustment.RETURNS_INVARIANT
    assert a.overnight_boundary is OvernightBoundary.NONE
    assert b.overnight_boundary is OvernightBoundary.NONE


def test_unknown_taxonomy_values_are_readable_errors_listing_the_enum():
    # The allowed values in the message come from the policy enum itself —
    # there is no second value list to drift.
    with pytest.raises(ValueError, match="Adjustment"):
        _spec(adjustment="qfq_level")
    with pytest.raises(ValueError, match="OvernightBoundary"):
        _spec(overnight_boundary="masked_disclosed")
    with pytest.raises(ValueError, match="adjustment"):
        _spec(adjustment=True)  # bool is not a taxonomy value either


def test_requires_must_be_a_nonempty_panel_field_sequence():
    with pytest.raises(ValueError, match="sequence of PanelField"):
        _spec(requires="close")  # a bare string is the classic footgun
    with pytest.raises(ValueError, match="sequence of PanelField"):
        _spec(requires=PanelField("close", source="market_daily"))  # bare entry
    with pytest.raises(ValueError, match="non-empty"):
        _spec(requires=())
    with pytest.raises(ValueError, match="PanelField instances"):
        _spec(requires=("close",))


def test_requires_list_is_normalized_to_tuple_and_duplicates_rejected():
    spec = _spec(requires=[PanelField("close", source="market_daily")])
    assert isinstance(spec.requires, tuple)
    with pytest.raises(ValueError, match="duplicate"):
        _spec(
            requires=(
                PanelField("close", source="market_daily"),
                PanelField("close", source="market_daily"),
            )
        )


def test_same_field_from_two_sources_is_not_a_duplicate():
    # valley_price_quantile's real shape: minute close AND daily close.
    spec = _spec(
        requires=(
            PanelField("close", source="stk_mins_1min"),
            PanelField("close", source="market_daily"),
        )
    )
    assert len(spec.requires) == 2


def test_adjustment_none_with_a_price_channel_requirement_contradicts():
    # D0 §1.1 static check: 'none' IS the claim "never touches the price
    # channel", so requiring OHLC under it must fail loudly.
    with pytest.raises(ValueError, match="price-channel"):
        _spec(adjustment="none")  # base requires includes market_daily close
    # ... while a volume/amount-only requires under 'none' is legitimate:
    _spec(
        adjustment="none",
        input_fields=("amount",),
        requires=(PanelField("amount", source="market_daily"),),
    )
    # ... and returns_invariant WITHOUT an OHLC field is legitimate too (the
    # VWAP-ratio factors: price arrives via sum(amount)/sum(volume)) — the
    # converse direction is deliberately NOT checked.
    _spec(
        adjustment="returns_invariant",
        input_fields=("volume", "amount"),
        requires=(
            PanelField("volume", source="stk_mins_1min"),
            PanelField("amount", source="stk_mins_1min"),
        ),
    )


# --------------------------------------------------------------------------- #
# D0 §2 pre-assignment table lock (all shipped factors, row for row)
# --------------------------------------------------------------------------- #
_MD = "market_daily"
_MIN = "stk_mins_1min"

# (factory, adjustment, overnight_boundary, requires as (field, source) pairs)
_D0_TABLE = [
    # finishing-line 14 (docs/factors/refactor_d0_contract.md §2), rows 1-14
    (lambda: JumpAmountCorrFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE,
     (("high", _MIN), ("low", _MIN), ("open", _MIN), ("amount", _MIN))),
    (lambda: MinuteIdealAmplitudeFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.CROSSED_DISCLOSED,
     (("high", _MIN), ("low", _MIN), ("close", _MIN))),
    (lambda: AmpMarginalAnomalyVolFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE,
     (("high", _MIN), ("low", _MIN), ("close", _MIN))),
    (lambda: VolumePeakCountFactor(), Adjustment.NONE, OvernightBoundary.NONE,
     (("volume", _MIN),)),
    (lambda: IntradayAmpCutFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE,
     (("high", _MIN), ("low", _MIN), ("close", _MIN))),
    (lambda: PeakIntervalKurtosisFactor(), Adjustment.NONE,
     OvernightBoundary.NONE, (("volume", _MIN),)),
    (lambda: ValleyRelativeVwapFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("volume", _MIN), ("amount", _MIN))),
    (lambda: ValleyRidgeVwapRatioFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("volume", _MIN), ("amount", _MIN))),
    (lambda: RidgeMinuteReturnFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("volume", _MIN), ("close", _MIN))),
    (lambda: ValleyPriceQuantileFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.CROSSED_DISCLOSED,
     (("volume", _MIN), ("amount", _MIN), ("high", _MIN), ("low", _MIN),
      ("close", _MIN), ("close", _MD))),
    (lambda: PeakRidgeAmountRatioFactor(), Adjustment.NONE,
     OvernightBoundary.NONE, (("volume", _MIN), ("amount", _MIN))),
    (lambda: ValueFactor("value_ep"), Adjustment.NONE, OvernightBoundary.NONE,
     (("pe", "daily_basic"),)),
    (lambda: ValueFactor("value_bp"), Adjustment.NONE, OvernightBoundary.NONE,
     (("pb", "daily_basic"),)),
    (lambda: VolatilityFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("close", _MD),)),
    # remaining daily factors (declarations derived in D1, evidence in each
    # spec docstring)
    (lambda: MomentumFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("close", _MD),)),
    (lambda: ReversalFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("close", _MD),)),
    (lambda: LiquidityFactor(), Adjustment.NONE, OvernightBoundary.NONE,
     (("amount", _MD),)),
    (lambda: OvernightMomentumFactor(), Adjustment.RETURNS_INVARIANT,
     OvernightBoundary.NONE, (("open", _MD), ("close", _MD))),
    (lambda: FinancialFactor("roe"), Adjustment.NONE, OvernightBoundary.NONE,
     (("roe", "fina_indicator"),)),
    (lambda: FinancialFactor("netprofit_yoy"), Adjustment.NONE,
     OvernightBoundary.NONE, (("netprofit_yoy", "fina_indicator"),)),
    (lambda: FinancialFactor("grossprofit_margin"), Adjustment.NONE,
     OvernightBoundary.NONE, (("grossprofit_margin", "fina_indicator"),)),
]


@pytest.mark.parametrize(
    "factory, adjustment, overnight, requires",
    _D0_TABLE,
    ids=[f"{i:02d}" for i in range(len(_D0_TABLE))],
)
def test_shipped_factor_declarations_match_the_d0_table(
    factory, adjustment, overnight, requires
):
    spec = factory().spec
    assert spec.adjustment is adjustment, spec.factor_id
    assert spec.overnight_boundary is overnight, spec.factor_id
    assert tuple((r.field, r.source) for r in spec.requires) == requires, (
        spec.factor_id
    )


def test_zero_price_level_members_today():
    # D0 statistics: PRICE_LEVEL has zero members in the shipped set. The
    # fixture-tested store-fingerprint mechanism for it is a D3 obligation
    # (kill archive A5-F02) — this row just keeps the census honest.
    for factory, *_ in _D0_TABLE:
        assert factory().spec.adjustment is not Adjustment.PRICE_LEVEL


def test_crossed_disclosed_members_are_exactly_the_two_declared():
    crossed = sorted(
        factory().spec.factor_id
        for factory, *_ in _D0_TABLE
        if factory().spec.overnight_boundary is OvernightBoundary.CROSSED_DISCLOSED
    )
    assert crossed == ["minute_ideal_amp_10", "valley_price_quantile_20"]


def test_minute_ideal_amplitude_docstring_tracks_the_d2_measurement_debt():
    # D0 §2 note 1: the CROSSED_DISCLOSED pre-assignment is a CANDIDATE whose
    # third requirement (the measured deviation) is still missing; D1 pins
    # the obligation in the spec docstring so it cannot silently evaporate.
    doc = type(MinuteIdealAmplitudeFactor()).spec.fget.__doc__
    assert "MISSING" in doc and "D2" in doc


def test_spec_variant_replace_carries_the_new_declarations():
    # dataclasses.replace re-runs validation and must carry the v1.0 fields —
    # the exec-basis spec variant path depends on this.
    from dataclasses import replace

    spec = _spec()
    twin = replace(spec, version="2.0")
    assert twin.requires == spec.requires
    assert twin.adjustment is spec.adjustment
    assert twin.overnight_boundary is spec.overnight_boundary
