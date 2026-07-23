"""Factor-evaluation contract tests (PR-A): the CI enforcement layer (design §7).

Network-free. These lock the contract itself — the validators, the mandatory-spec
enforcement, the migration of every real factor, the mandatory sections, the
verdict rule table, and the determinism/secret-safety of the report.

``StandardFactorEvaluator`` and the vectorized eval-IR are PR-B; nothing here
computes a metric.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math

import pytest

from analytics.eval import (
    ADOPT,
    AXIS_FAIL,
    AXIS_INSUFFICIENT_DATA,
    AXIS_NOT_ASSESSED,
    AXIS_PASS,
    INSUFFICIENT_DATA,
    MANDATORY_SECTIONS,
    REJECT,
    VERDICT_KEYS,
    WATCH,
    AxisVerdict,
    EvalConfig,
    FactorEvalReport,
    FactorEvaluator,
    Section,
    Skipped,
    VerdictInputs,
    VerdictResult,
    VerdictThresholds,
    decide_verdict,
)
from analytics.eval.render import MAX_VALUE_CHARS

# The v0.8 spread fix lived in an exact arithmetic expression, so the lock has to
# be on that expression itself, not on a PASS/FAIL that happens to agree with it.
from analytics.eval.verdict import (
    MONO_CONTRADICTED,
    MONO_HOLDS,
    MONO_UNKNOWN,
    _aligned_net,
    _all_spreads_negative,
    _base_spread,
    _monotonicity_direction,
)
from factors.base import Factor
from factors.compute.candidates import (
    VALUE_FIELDS,
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import SUPPORTED_FIELDS, FinancialFactor
from factors.compute.momentum import MomentumFactor
from factors.spec import (
    INTRADAY_FIELDS,
    INTRADAY_RETURN_BASIS,
    RETURN_BASES,
    FactorSpec,
    PanelField,
)


def _spec(**overrides) -> FactorSpec:
    """A valid daily FactorSpec; override one field per test."""
    base = dict(
        factor_id="unit_test_factor",
        version="1.0",
        description="A test factor.",
        expected_ic_sign=+1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        # Contract v1.0 mandatory declarations (D1).
        requires=(PanelField("close", source="market_daily"),),
        adjustment="returns_invariant",
        overnight_boundary="none",
    )
    base.update(overrides)
    return FactorSpec(**base)


def _intraday_block() -> dict:
    """The project's minute conventions (I5a/I5b)."""
    return {
        "decision_cutoff": "14:50:00",
        "data_lag": "1min",
        "session_open": "09:30:00",
        "execution_model": "next_minute_close",
        "execution_window": "[14:51,14:56:59]",
    }


def _cfg(**overrides) -> EvalConfig:
    base = dict(
        universe="000016.SH",
        universe_is_pit=True,
        start="2023-07-01",
        end="2024-06-30",
        is_exploratory=True,
        post_hoc_selected=False,
    )
    base.update(overrides)
    return EvalConfig(**base)


# --------------------------------------------------------------------------
# FactorSpec validators (enforcement layer #1)
# --------------------------------------------------------------------------


def test_valid_spec_is_frozen_and_normalizes_input_fields():
    spec = _spec(input_fields=["close", "amount"])
    assert spec.input_fields == ("close", "amount")  # list -> tuple, hashable
    with pytest.raises(Exception):
        spec.factor_id = "mutated"  # frozen


@pytest.mark.parametrize("horizon", [0, -1, -20])
def test_non_positive_forward_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="forward_return_horizon"):
        _spec(forward_return_horizon=horizon)


@pytest.mark.parametrize("sign", [None, 0, 2, -2, "+1", 1.0, True])
def test_expected_ic_sign_must_be_plus_or_minus_one(sign):
    """None is explicitly forbidden: the direction is a pre-run commitment."""
    with pytest.raises(ValueError, match="expected_ic_sign"):
        _spec(expected_ic_sign=sign)


@pytest.mark.parametrize("sign", [+1, -1])
def test_expected_ic_sign_accepts_both_directions(sign):
    assert _spec(expected_ic_sign=sign).expected_ic_sign == sign


def test_intraday_true_requires_the_whole_minute_block():
    for missing in INTRADAY_FIELDS:
        block = _intraday_block()
        block.pop(missing)
        with pytest.raises(ValueError, match="minute block"):
            _spec(is_intraday=True, return_basis="exec_to_exec", **block)


def test_intraday_true_with_full_block_is_valid():
    spec = _spec(is_intraday=True, return_basis="exec_to_exec", **_intraday_block())
    assert spec.decision_cutoff == "14:50:00"
    assert spec.execution_model == "next_minute_close"


@pytest.mark.parametrize("field_name", INTRADAY_FIELDS)
def test_daily_factor_may_not_set_any_intraday_field(field_name):
    with pytest.raises(ValueError, match="is_intraday=False"):
        _spec(is_intraday=False, **{field_name: "14:50:00"})


def test_intraday_block_of_empty_strings_is_rejected():
    """Present-but-blank is the same half-declared contract as missing.

    An all-empty block passes an ``is not None`` check while declaring nothing —
    exactly what the guard claims to reject.
    """
    block = dict.fromkeys(INTRADAY_FIELDS, "")
    with pytest.raises(ValueError, match="non-empty strings"):
        _spec(is_intraday=True, return_basis="exec_to_exec", **block)


@pytest.mark.parametrize("blank", ["", "   ", 1450])
@pytest.mark.parametrize("field_name", INTRADAY_FIELDS)
def test_intraday_block_rejects_any_blank_or_non_string_entry(field_name, blank):
    block = _intraday_block()
    block[field_name] = blank
    with pytest.raises(ValueError, match="non-empty strings"):
        _spec(is_intraday=True, return_basis="exec_to_exec", **block)


def test_intraday_factor_may_not_claim_close_to_close_returns():
    """I5a: an intraday holding period is exec(T)->exec(T_next), never close-to-close."""
    with pytest.raises(ValueError, match="exec_to_exec"):
        _spec(is_intraday=True, return_basis="close_to_close", **_intraday_block())


def test_intraday_return_basis_constant_matches_the_supported_bases():
    assert INTRADAY_RETURN_BASIS in RETURN_BASES


@pytest.mark.parametrize("field_name", ["factor_id", "version", "description"])
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_identity_fields_must_be_non_empty(field_name, bad):
    with pytest.raises(ValueError, match=field_name):
        _spec(**{field_name: bad})


@pytest.mark.parametrize("bad", [(), [], None])
def test_input_fields_must_be_non_empty(bad):
    with pytest.raises(ValueError, match="input_fields"):
        _spec(input_fields=bad)


def test_input_fields_rejects_a_bare_string():
    """'close' must not silently become ('c','l','o','s','e')."""
    with pytest.raises(ValueError, match="not a bare string"):
        _spec(input_fields="close")


@pytest.mark.parametrize("basis", ["close_to_open", "", "exec"])
def test_return_basis_must_be_supported(basis):
    with pytest.raises(ValueError, match="return_basis"):
        _spec(return_basis=basis)


@pytest.mark.parametrize("adjust", ["hfq", "none", "raw"])
def test_price_adjust_only_supports_qfq_today(adjust):
    with pytest.raises(ValueError, match="price_adjust"):
        _spec(price_adjust=adjust)


def test_min_history_bars_must_be_non_negative():
    with pytest.raises(ValueError, match="min_history_bars"):
        _spec(min_history_bars=-1)


# --------------------------------------------------------------------------
# Factor.__init_subclass__ enforcement (spec is mandatory)
# --------------------------------------------------------------------------


def test_factor_subclass_without_spec_raises_at_class_definition():
    with pytest.raises(TypeError, match="must declare a FactorSpec"):

        class NoSpecFactor(Factor):
            name = "no_spec"

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


def test_factor_subclass_with_classattr_spec_is_accepted():
    class ClassAttrFactor(Factor):
        name = "class_attr_factor"
        spec = _spec(factor_id="class_attr_factor")

        def compute(self, panel):
            return panel

    assert ClassAttrFactor().spec.factor_id == "class_attr_factor"


def test_factor_subclass_with_property_spec_is_accepted():
    class PropertyFactor(Factor):
        name = "property_factor"

        @property
        def spec(self):
            return _spec(factor_id="property_factor")

        def compute(self, panel):
            return panel

    assert PropertyFactor().spec.factor_id == "property_factor"


def test_factor_subclass_with_cached_property_spec_is_accepted():
    """cached_property is an explicitly allowed deferred form (it needs an instance)."""

    class CachedPropertyFactor(Factor):
        name = "cached_property_factor"

        @functools.cached_property
        def spec(self):
            return _spec(factor_id="cached_property_factor")

        def compute(self, panel):
            return panel

    assert CachedPropertyFactor().spec.factor_id == "cached_property_factor"


def test_factor_subclass_with_non_spec_classattr_raises_at_class_definition():
    with pytest.raises(TypeError, match="neither a FactorSpec"):

        class StringSpecFactor(Factor):
            name = "string_spec"
            spec = "momentum_20"

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


def test_spec_as_a_bare_method_is_rejected_at_class_definition():
    """The classic slip: @property forgotten.

    A plain function IS a descriptor, so a ``hasattr(type(x), '__get__')`` check
    waves it through and the contract only fails much later, at instantiation,
    with a generic message. The message must name the actual mistake.
    """
    with pytest.raises(TypeError, match="did you forget @property"):

        class BareMethodFactor(Factor):
            name = "bare_method"

            def spec(self):  # pragma: no cover - never defined
                return _spec(factor_id="bare_method")

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


def test_spec_as_a_staticmethod_is_rejected_at_class_definition():
    with pytest.raises(TypeError, match="did you forget @property"):

        class StaticMethodFactor(Factor):
            name = "static_method"

            @staticmethod
            def spec():  # pragma: no cover - never defined
                return _spec(factor_id="static_method")

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


def test_spec_as_a_classmethod_is_rejected_at_class_definition():
    with pytest.raises(TypeError, match="did you forget @property"):

        class ClassMethodFactor(Factor):
            name = "class_method"

            @classmethod
            def spec(cls):  # pragma: no cover - never defined
                return _spec(factor_id="class_method")

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


# --------------------------------------------------------------------------
# The spec requirement fires on CONCRETE classes only (the factor layer will be
# refactored; a shared family base must not have to invent a bogus spec).
# --------------------------------------------------------------------------


def test_abstract_intermediate_base_needs_no_spec():
    """An intermediate that leaves ``compute`` abstract is not a factor yet."""

    class AbstractFamilyBase(Factor):
        """A shared base for a factor family: no compute, hence still abstract."""

        def _shared_helper(self) -> int:
            return 1

    assert AbstractFamilyBase.__abstractmethods__ == frozenset({"compute"})
    with pytest.raises(TypeError, match="abstract"):
        AbstractFamilyBase()  # still not instantiable — it is not a factor


def test_concrete_subclass_of_an_abstract_intermediate_still_needs_a_spec():
    class AbstractFamilyBase(Factor):
        def _shared_helper(self) -> int:
            return 1

    with pytest.raises(TypeError, match="must declare a FactorSpec"):

        class ConcreteChild(AbstractFamilyBase):
            name = "concrete_child"

            def compute(self, panel):  # pragma: no cover - never defined
                return panel


def test_concrete_subclass_of_an_abstract_intermediate_works_with_a_spec():
    class AbstractFamilyBase(Factor):
        def _shared_helper(self) -> int:
            return 1

    class ConcreteChild(AbstractFamilyBase):
        name = "concrete_child"
        spec = _spec(factor_id="concrete_child")

        def compute(self, panel):
            return panel

    assert ConcreteChild().spec.factor_id == "concrete_child"
    assert ConcreteChild()._shared_helper() == 1


def test_instance_spec_is_validated_at_construction():
    """A property returning a non-FactorSpec cannot even be instantiated."""

    class LyingFactor(Factor):
        name = "lying"

        @property
        def spec(self):
            return {"factor_id": "lying"}  # a dict is not a FactorSpec

        def compute(self, panel):  # pragma: no cover - never reached
            return panel

    with pytest.raises(TypeError, match="must be a FactorSpec instance"):
        LyingFactor()


# --------------------------------------------------------------------------
# Migration lock: every real factor declares a valid spec
# --------------------------------------------------------------------------

REAL_FACTORS = [
    MomentumFactor(),
    MomentumFactor(window=60),
    ReversalFactor(),
    ReversalFactor(window=5),
    VolatilityFactor(),
    LiquidityFactor(),
    OvernightMomentumFactor(),
    *[FinancialFactor(f) for f in SUPPORTED_FIELDS],
    *[ValueFactor(f) for f in VALUE_FIELDS],
]


@pytest.mark.parametrize("factor", REAL_FACTORS, ids=lambda f: f.name)
def test_every_real_factor_exposes_a_valid_spec(factor):
    spec = factor.spec
    assert isinstance(spec, FactorSpec)
    # factor_id IS the panel column: a mismatch would misjoin the whole report.
    assert spec.factor_id == factor.name
    assert spec.expected_ic_sign in (+1, -1)
    assert spec.forward_return_horizon > 0
    assert spec.input_fields
    assert not spec.is_intraday  # every factor today is daily
    assert all(getattr(spec, f) is None for f in INTRADAY_FIELDS)


def _all_subclasses(cls) -> set:
    found = set(cls.__subclasses__())
    for sub in list(found):
        found |= _all_subclasses(sub)
    return found


def test_every_shipped_factor_subclass_declares_a_spec():
    """Drift guard: a NEW factor class cannot be merged without a spec.

    Scoped to classes defined under ``factors.`` on purpose — this module and
    other tests define deliberately-broken subclasses, and a class whose
    ``__init_subclass__`` raised still lingers in ``__subclasses__()``.
    """
    shipped = {
        cls for cls in _all_subclasses(Factor) if cls.__module__.startswith("factors.")
    }
    assert len(shipped) >= 7  # never pass vacuously
    for cls in shipped:
        assert "spec" in cls.__dict__, f"{cls.__name__} declares no spec"


@pytest.mark.parametrize("window", [5, 20, 60])
def test_parameterized_factor_id_tracks_the_window(window):
    """A property (not a classattr) is why momentum_60 is not labelled momentum_20."""
    factor = MomentumFactor(window=window)
    assert factor.spec.factor_id == f"momentum_{window}"
    assert factor.spec.min_history_bars == window
    assert ReversalFactor(window=window).spec.factor_id == f"reversal_{window}"


def test_reversal_sign_is_the_exact_negation_of_momentum():
    """Reversal = -momentum by construction, so the hypotheses cannot drift."""
    assert ReversalFactor(window=5).spec.expected_ic_sign == -(
        MomentumFactor(window=5).spec.expected_ic_sign
    )


def test_independently_confirmed_signs_match_the_project_evidence():
    """P3-5/P3-7/P3-8: value_ep/value_bp +1, volatility_20 -1 (low-vol)."""
    assert ValueFactor("value_ep").spec.expected_ic_sign == +1
    assert ValueFactor("value_bp").spec.expected_ic_sign == +1
    assert VolatilityFactor().spec.expected_ic_sign == -1


# --------------------------------------------------------------------------
# EvalConfig validators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 0, -5])
def test_n_quantiles_below_two_rejected(n):
    with pytest.raises(ValueError, match="n_quantiles"):
        _cfg(n_quantiles=n)


@pytest.mark.parametrize("scenarios", [(2.0, 4.0), (0.5,), ()])
def test_cost_scenarios_must_contain_the_base_anchor(scenarios):
    with pytest.raises(ValueError, match="base anchor 1.0"):
        _cfg(cost_scenarios=scenarios)


def test_cost_scenarios_with_base_anchor_accepted():
    assert _cfg(cost_scenarios=[1.0, 2.0, 4.0]).cost_scenarios == (1.0, 2.0, 4.0)


def test_post_hoc_selection_requires_exploratory():
    with pytest.raises(ValueError, match="post_hoc_selected=True requires"):
        _cfg(is_exploratory=False, post_hoc_selected=True)


def test_post_hoc_with_exploratory_accepted():
    assert _cfg(is_exploratory=True, post_hoc_selected=True).post_hoc_selected


def test_long_short_defaults_to_top_and_bottom_bucket():
    assert _cfg(n_quantiles=5).long_short == (5, 1)
    assert _cfg(n_quantiles=10).long_short == (10, 1)


def test_long_short_must_stay_in_natural_orientation():
    """Flipping the legs here would double-flip a -1 factor in the verdict."""
    with pytest.raises(ValueError, match="top > bottom"):
        _cfg(n_quantiles=5, long_short=(1, 5))


@pytest.mark.parametrize("pair", [(6, 1), (5, 0), (5, 5)])
def test_long_short_out_of_range_rejected(pair):
    with pytest.raises(ValueError, match="long_short"):
        _cfg(n_quantiles=5, long_short=pair)


@pytest.mark.parametrize("field_name", ["universe", "start", "end"])
def test_universe_and_window_must_be_non_empty(field_name):
    with pytest.raises(ValueError, match=field_name):
        _cfg(**{field_name: ""})


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_max_participation_rate_must_be_a_fraction(rate):
    with pytest.raises(ValueError, match="max_participation_rate"):
        _cfg(max_participation_rate=rate)


@pytest.mark.parametrize("notional", [0, -1, -1e6, "10000000", []])
def test_capacity_notional_must_be_none_or_positive(notional):
    with pytest.raises(ValueError, match="capacity_notional"):
        _cfg(capacity_notional=notional)


def test_capacity_notional_accepts_none_or_a_positive_number():
    assert _cfg().capacity_notional is None  # None = capacity not assessed
    assert _cfg(capacity_notional=10_000_000).capacity_notional == 10_000_000


# --------------------------------------------------------------------------
# bool is an int subclass: True must never sneak into a numeric field (the same
# gotcha as expected_ic_sign=1.0). FactorSpec already guards every numeric
# validator; these lock the EvalConfig ones that did not.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_capacity_notional_rejects_bool(value):
    with pytest.raises(ValueError, match="capacity_notional"):
        _cfg(capacity_notional=value)


@pytest.mark.parametrize("value", [True, False])
def test_max_participation_rate_rejects_bool(value):
    with pytest.raises(ValueError, match="max_participation_rate"):
        _cfg(max_participation_rate=value)


@pytest.mark.parametrize("value", [True, False])
def test_cost_scenarios_reject_bool_entries(value):
    """float(True) is 1.0 — a stray True would masquerade as the base anchor."""
    with pytest.raises(ValueError, match="cost_scenarios"):
        _cfg(cost_scenarios=(value, 2.0, 4.0))


@pytest.mark.parametrize("value", [True, False])
def test_n_quantiles_rejects_bool(value):
    with pytest.raises(ValueError, match="n_quantiles"):
        _cfg(n_quantiles=value)


@pytest.mark.parametrize("value", [True, False])
def test_n_factors_screened_rejects_bool(value):
    with pytest.raises(ValueError, match="n_factors_screened"):
        _cfg(n_factors_screened=value)


@pytest.mark.parametrize("value", [True, False])
def test_forward_return_horizon_rejects_bool(value):
    with pytest.raises(ValueError, match="forward_return_horizon"):
        _spec(forward_return_horizon=value)


@pytest.mark.parametrize("value", [True, False])
def test_min_history_bars_rejects_bool(value):
    with pytest.raises(ValueError, match="min_history_bars"):
        _spec(min_history_bars=value)


@pytest.mark.parametrize("value", [True, False])
def test_long_short_rejects_bool_entries(value):
    with pytest.raises(ValueError, match="long_short"):
        _cfg(n_quantiles=5, long_short=(value, 1))


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "3"])
def test_n_factors_screened_must_be_none_or_a_positive_int(n):
    with pytest.raises(ValueError, match="n_factors_screened"):
        _cfg(n_factors_screened=n)


def test_n_factors_screened_accepts_none_or_a_positive_int():
    assert _cfg().n_factors_screened is None
    assert _cfg(n_factors_screened=11).n_factors_screened == 11


def test_declared_sequences_are_coerced_to_tuples_and_stay_immutable():
    """frozen=True must actually MEAN immutable + hashable.

    A caller handing in a list otherwise keeps a live reference into the config
    (mutate the list -> mutate the "frozen" provenance record) and hash() raises.
    """
    cells = [("SSE50", "2024-2026")]
    neutralization = ["industry", "size"]
    cfg = _cfg(independent_cells=cells, neutralization=neutralization)
    assert cfg.independent_cells == (("SSE50", "2024-2026"),)
    assert cfg.neutralization == ("industry", "size")
    assert isinstance(cfg.independent_cells, tuple)
    assert isinstance(cfg.neutralization, tuple)
    hash(cfg)  # would raise on a list field
    cells.append(("CSI300", "2024-2026"))
    neutralization.append("beta")
    assert cfg.independent_cells == (("SSE50", "2024-2026"),)
    assert cfg.neutralization == ("industry", "size")


def test_declared_sequence_defaults_are_already_tuples():
    cfg = _cfg()
    assert cfg.independent_cells == ()
    assert cfg.neutralization == ("industry", "size")
    hash(cfg)


@pytest.mark.parametrize("field_name", ["independent_cells", "neutralization"])
def test_declared_sequences_reject_a_bare_string(field_name):
    """'industry' must not silently become ('i','n','d','u','s','t','r','y')."""
    with pytest.raises(ValueError, match=field_name):
        _cfg(**{field_name: "industry"})


@pytest.mark.parametrize("field_name", ["independent_cells", "neutralization"])
def test_declared_sequences_reject_a_non_sequence(field_name):
    with pytest.raises(ValueError, match=field_name):
        _cfg(**{field_name: 3})


# --------------------------------------------------------------------------
# Section / Skipped / report assembly (enforcement layer #2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["", "   "])
def test_skipped_without_a_reason_rejected(reason):
    with pytest.raises(ValueError, match="reason"):
        Skipped("purity", reason)


def test_skipped_with_reason_accepted():
    assert Skipped("purity", "no known-factor anchor set configured").reason


#: A data_coverage payload that clears all three parts of the §6 v0.3 sample
#: gate (raw floor / effective samples / calendar span), so a test about
#: something else is not silently gated into INSUFFICIENT-DATA.
_COVERAGE_OK: dict = {
    "settled_rebalances": 36,
    "effective_samples": 36.0,
    "span_days": 400.0,
}


def _full_sections(**payloads) -> list:
    return [Section(name, payloads.get(name, {})) for name in MANDATORY_SECTIONS]


def test_missing_mandatory_section_raises():
    sections = _full_sections()[:-1]  # drop 'caveats'
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections)
    with pytest.raises(ValueError, match="caveats"):
        report.validate_all_mandatory_present()


def test_skipped_with_reason_counts_as_present():
    sections = _full_sections()[:-1] + [Skipped("caveats", "nothing to disclose")]
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections)
    report.validate_all_mandatory_present()  # must not raise


def test_duplicate_sections_rejected():
    sections = _full_sections() + [Section("purity", {})]
    with pytest.raises(ValueError, match="duplicate"):
        FactorEvalReport.assemble(_spec(), _cfg(), sections)


def test_non_section_object_rejected():
    with pytest.raises(TypeError, match="Section or an explicit"):
        FactorEvalReport.assemble(_spec(), _cfg(), [*_full_sections(), None])


def test_extra_custom_section_is_allowed():
    sections = [*_full_sections(), Section("crowding", {"note": 1})]
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections)
    report.validate_all_mandatory_present()
    assert "crowding" in report.by_name()


# --------------------------------------------------------------------------
# Verdict rules: three axes + a derived deployment label (design §6, v0.5)
# --------------------------------------------------------------------------

#: A strong signal whose ICIR point clears AND whose N_eff-based 95% CI LOWER
#: bound (design §6, v0.6) also clears min_abs_icir=0.30 — a PASS-grade estimate.
_STRONG = dict(
    ic_ir=0.8,
    ic_ir_ci_low=0.55,
    ic_ir_ci_high=1.05,
    ic_win_rate=0.7,
    ic_nw_t=3.5,
    monotonicity_spearman=0.9,
    net_long_short_by_cost=((1.0, 0.05), (2.0, 0.04), (4.0, 0.02)),
)
_TRADABLE = dict(tradable=True, capacity_sufficient=True)
_OOS_OK = dict(oos_available=True, oos_sign_consistent=True)
#: a known-factor book the factor is genuinely INCREMENTAL to (orthogonalized
#: ICIR point AND lower CI bound both clear the bar in the expected direction).
_INCREMENTAL_OK = dict(
    known_factors_supplied=True,
    incremental_ic_ir=0.6,
    incremental_ic_mean=0.04,
    incremental_ic_ir_ci_low=0.42,
    incremental_ic_ir_ci_high=0.78,
)
#: everything all three axes need to PASS — the ONLY route to Adopt (v0.5).
_ADOPT_FACTS = {**_STRONG, **_TRADABLE, **_OOS_OK, **_INCREMENTAL_OK}


#: A sample that CLEARS all three gate parts, so a rule test exercises the RULE
#: it is about instead of tripping over the gate.
_SAMPLE_OK = dict(settled_rebalances=36, effective_samples=36.0, span_days=400.0)

#: The project's signature failure SHAPE: plenty of raw periods over a comfortable
#: calendar span, but a regime-flipping IC series is a STEP FUNCTION whose rho-hat
#: stays positive for ~100 lags, so N_eff collapses to ~3 out of ~300.
_THIN_SAMPLE = dict(settled_rebalances=300, effective_samples=3.0, span_days=420.0)


def _inputs(**overrides) -> VerdictInputs:
    base = dict(expected_ic_sign=+1, **_SAMPLE_OK)
    base.update(overrides)
    return VerdictInputs(**base)


# -- the new data model ----------------------------------------------------


def test_axis_verdict_rejects_an_unknown_state():
    with pytest.raises(ValueError, match="axis verdict"):
        AxisVerdict("MAYBE")


def test_verdict_result_exposes_the_three_axes_and_the_derived_label():
    r = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=False))
    assert set(r.axes()) == {"predictive", "incremental", "tradable"}
    assert all(isinstance(a, AxisVerdict) for a in r.axes().values())
    assert r.verdict in (ADOPT, WATCH, REJECT, INSUFFICIENT_DATA)


def test_verdict_result_defaults_axes_to_not_assessed_when_hand_built():
    """A placeholder verdict (e.g. on an incomplete report) stays constructible."""
    hand = VerdictResult(REJECT, ("hand-built",))
    assert all(a.verdict == AXIS_NOT_ASSESSED for a in hand.axes().values())


# -- Axis A: Predictive ----------------------------------------------------


def test_predictive_axis_passes_with_oos_and_metrics():
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK))
    assert r.predictive.verdict == AXIS_PASS
    assert any("both out-of-sample subperiods" in x for x in r.predictive.reasons)


def test_predictive_axis_insufficient_on_a_thin_sample():
    r = decide_verdict(_inputs(effective_samples=5.0, **_STRONG, **_OOS_OK))
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("effective samples (A)" in x for x in r.predictive.reasons)


def test_predictive_axis_insufficient_without_oos():
    """No holdout -> no predictive claim; in-sample metrics are reported, not a PASS."""
    r = decide_verdict(_inputs(**_STRONG, oos_available=False))
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("no out-of-sample split" in x for x in r.predictive.reasons)


def test_predictive_axis_fails_on_a_known_oos_sign_flip():
    r = decide_verdict(
        _inputs(**_STRONG, oos_available=True, oos_sign_flipped=True)
    )
    assert r.predictive.verdict == AXIS_FAIL
    assert any("sign flipped" in x for x in r.predictive.reasons)


def test_predictive_axis_fails_on_an_independent_cell_reversal():
    r = decide_verdict(
        _inputs(**_STRONG, oos_available=True, oos_monotonicity_reversed=True)
    )
    assert r.predictive.verdict == AXIS_FAIL
    assert any("independent cell" in x for x in r.predictive.reasons)


def test_predictive_axis_fails_when_the_signal_is_absent_despite_oos():
    """Sufficient data + OOS available, but nothing convincing -> a NEGATIVE finding."""
    r = decide_verdict(
        _inputs(
            oos_available=True,
            oos_sign_consistent=False,
            ic_ir=0.05,
            ic_win_rate=0.5,
            ic_nw_t=0.3,
            monotonicity_spearman=0.1,
        )
    )
    assert r.predictive.verdict == AXIS_FAIL


def test_predictive_axis_passes_on_weak_but_aligned_monotonicity():
    """v0.7 direction gate: monotonicity is a DIRECTION check, not a strength bar.

    A tail-concentrated real factor (e.g. jump-amount-corr: strong ICIR/NW-t, yet a
    hump-shaped quantile profile whose aligned Spearman is only ~0.3) used to FAIL
    the old 0.80 STRENGTH bar despite a genuine out-of-sample signal. Under the
    default 0.0 direction gate a strictly-positive aligned monotonicity clears, so
    the axis PASSes on its ICIR/NW-t/win-rate/OOS evidence.
    """
    r = decide_verdict(
        _inputs(**{**_STRONG, "monotonicity_spearman": 0.3}, **_OOS_OK)
    )
    assert r.predictive.verdict == AXIS_PASS


def test_predictive_axis_fails_on_reversed_monotonicity_despite_strong_ic():
    """v0.7 direction gate still bites the WRONG way: aligned monotonicity <= 0.

    Strong ICIR/NW-t/win-rate/OOS, but the quantile buckets are ordered AGAINST the
    hypothesis (aligned Spearman -0.4). The direction gate is not cleared, so the
    point signal is absent and the axis is a NEGATIVE finding, not a PASS.
    """
    r = decide_verdict(
        _inputs(**{**_STRONG, "monotonicity_spearman": -0.4}, **_OOS_OK)
    )
    assert r.predictive.verdict == AXIS_FAIL
    assert any("monotonicity" in x for x in r.predictive.reasons)


# -- Axis B: Incremental ---------------------------------------------------


def test_incremental_axis_not_assessed_without_a_book():
    """The DEFAULT: no known_factors supplied -> the axis cannot be assessed."""
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK))
    assert r.incremental.verdict == AXIS_NOT_ASSESSED
    assert any("no known-factor book" in x for x in r.incremental.reasons)


def test_incremental_axis_passes_on_a_genuinely_additive_factor():
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK, **_INCREMENTAL_OK))
    assert r.incremental.verdict == AXIS_PASS
    assert any("adds a signal" in x for x in r.incremental.reasons)


def test_incremental_axis_fails_on_a_redundant_factor():
    """Book supplied, sufficient sample, orthogonalized IC ~ 0 -> redundant."""
    r = decide_verdict(
        _inputs(
            **_STRONG,
            **_OOS_OK,
            known_factors_supplied=True,
            incremental_ic_ir=0.02,
            incremental_ic_mean=0.001,
        )
    )
    assert r.incremental.verdict == AXIS_FAIL
    assert any("redundant" in x for x in r.incremental.reasons)


def test_incremental_axis_uses_its_own_lower_bar_not_the_raw_predictive_bar():
    """v0.7: the Incremental axis gates on min_incremental_abs_icir (0.15), a bar
    SEPARATE from — and lower than — the raw predictive min_abs_icir (0.30).

    Orthogonalization removes variance, so a genuinely-additive residual lives on a
    smaller scale. An orthogonalized ICIR of 0.20 (point AND lower CI bound above
    0.15) is a PASS under the incremental bar, though it would have missed the raw
    0.30 bar the axis used before v0.7.
    """
    r = decide_verdict(
        _inputs(
            **_STRONG,
            **_OOS_OK,
            known_factors_supplied=True,
            incremental_ic_ir=0.20,
            incremental_ic_mean=0.012,
            incremental_ic_ir_ci_low=0.17,
            incremental_ic_ir_ci_high=0.23,
        )
    )
    assert r.incremental.verdict == AXIS_PASS


def test_predictive_bar_is_unchanged_by_the_separate_incremental_bar():
    """The v0.7 split must NOT lower the raw predictive bar: an ICIR of 0.20 clears
    the incremental 0.15 bar but still fails the predictive 0.30 point bar.
    """
    r = decide_verdict(
        _inputs(
            **_OOS_OK,
            ic_ir=0.20,
            ic_ir_ci_low=0.17,
            ic_ir_ci_high=0.23,
            ic_win_rate=0.7,
            ic_nw_t=3.5,
            monotonicity_spearman=0.9,
        )
    )
    assert r.predictive.verdict == AXIS_FAIL


def test_incremental_axis_fails_on_the_wrong_direction():
    r = decide_verdict(
        _inputs(
            **_STRONG,
            **_OOS_OK,
            known_factors_supplied=True,
            incremental_ic_ir=-0.6,
            incremental_ic_mean=-0.04,
        )
    )
    assert r.incremental.verdict == AXIS_FAIL
    assert any("anti-incremental" in x for x in r.incremental.reasons)


def test_incremental_axis_insufficient_when_a_book_is_supplied_but_thin():
    r = decide_verdict(
        _inputs(effective_samples=5.0, **_STRONG, **_OOS_OK, **_INCREMENTAL_OK)
    )
    assert r.incremental.verdict == AXIS_INSUFFICIENT_DATA


def test_incremental_axis_insufficient_when_the_orthogonalized_ic_is_unknown():
    """An unmeasurable orthogonalized IC is UNKNOWN, never a FAIL (unknown never
    convicts) — distinct from a measured ~ 0, which IS a redundant FAIL."""
    r = decide_verdict(
        _inputs(
            **_STRONG,
            **_OOS_OK,
            known_factors_supplied=True,
            incremental_ic_ir=float("nan"),
        )
    )
    assert r.incremental.verdict == AXIS_INSUFFICIENT_DATA
    assert any("UNKNOWN" in x for x in r.incremental.reasons)


# -- Axis C: Tradable ------------------------------------------------------


def test_tradable_axis_not_assessed_by_default():
    """Execution facts (I5b/I5f) are measured elsewhere, so the default is unset."""
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK))
    assert r.tradable.verdict == AXIS_NOT_ASSESSED
    assert any("not assessed" in x for x in r.tradable.reasons)


def test_tradable_axis_passes_with_facts_and_a_positive_base_spread():
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK, **_TRADABLE))
    assert r.tradable.verdict == AXIS_PASS


def test_tradable_axis_fails_when_not_tradable():
    r = decide_verdict(
        _inputs(**_STRONG, **_OOS_OK, tradable=False, capacity_sufficient=True)
    )
    assert r.tradable.verdict == AXIS_FAIL
    assert any("not tradable" in x for x in r.tradable.reasons)


def test_tradable_axis_fails_on_insufficient_capacity():
    r = decide_verdict(
        _inputs(**_STRONG, **_OOS_OK, tradable=True, capacity_sufficient=False)
    )
    assert r.tradable.verdict == AXIS_FAIL
    assert any("capacity insufficient" in x for x in r.tradable.reasons)


def test_tradable_axis_fails_when_every_cost_scenario_is_negative():
    kwargs = {**_STRONG, **_OOS_OK, **_TRADABLE}
    kwargs["net_long_short_by_cost"] = ((1.0, -0.01), (2.0, -0.03), (4.0, -0.08))
    r = decide_verdict(_inputs(**kwargs))
    assert r.tradable.verdict == AXIS_FAIL
    assert any("EVERY cost scenario" in x for x in r.tradable.reasons)


def test_tradable_axis_insufficient_on_partial_facts():
    """tradable known but capacity not: established neither way — not PASS, not FAIL."""
    r = decide_verdict(
        _inputs(**_STRONG, **_OOS_OK, tradable=True, capacity_sufficient=None)
    )
    assert r.tradable.verdict == AXIS_INSUFFICIENT_DATA


# -- the deployment label derivation (design §6 table) ---------------------


def test_deployment_reject_on_any_axis_fail():
    r = decide_verdict(_inputs(**{**_ADOPT_FACTS, "tradable": False}))
    assert r.verdict == REJECT
    assert r.tradable.verdict == AXIS_FAIL
    assert any("tradable axis" in x for x in r.reasons)


def test_deployment_adopt_when_all_three_pass():
    r = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=False))
    assert r.verdict == ADOPT
    assert r.predictive.verdict == AXIS_PASS
    assert r.incremental.verdict == AXIS_PASS
    assert r.tradable.verdict == AXIS_PASS


def test_deployment_watch_when_one_pass_and_the_rest_unresolved():
    """>=1 PASS with the rest INSUFFICIENT/NOT_ASSESSED -> WATCH, naming the gaps."""
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK))  # only predictive can PASS
    assert r.verdict == WATCH
    assert r.predictive.verdict == AXIS_PASS
    assert r.incremental.verdict == AXIS_NOT_ASSESSED
    assert r.tradable.verdict == AXIS_NOT_ASSESSED
    assert any("unresolved" in x for x in r.reasons)


def test_deployment_insufficient_when_no_pass_and_no_fail():
    """Nothing demonstrated and nothing refuted -> INSUFFICIENT-DATA (NOT the old
    fallback Reject: with no OOS, no book and no execution facts we cannot tell)."""
    r = decide_verdict(_inputs())  # sufficient sample, but no OOS/book/exec facts
    assert r.verdict == INSUFFICIENT_DATA
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert r.incremental.verdict == AXIS_NOT_ASSESSED
    assert r.tradable.verdict == AXIS_NOT_ASSESSED


def test_default_run_tops_out_at_watch():
    """⚠️ INTENDED: with no known_factors and no execution facts the Incremental
    and Tradable axes are NOT_ASSESSED, so a default run can never have all three
    PASS — it maxes at WATCH, even with a flawless out-of-sample predictive signal.
    This is the multi-factor point: a factor judged in isolation has not earned an
    Adopt."""
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK, is_exploratory=False))
    assert r.verdict == WATCH
    assert r.incremental.verdict == AXIS_NOT_ASSESSED
    assert r.tradable.verdict == AXIS_NOT_ASSESSED


def test_full_facts_make_adopt_reachable():
    """The companion to the ceiling: a clean book + execution facts + OOS -> Adopt."""
    assert decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=False)).verdict == ADOPT


# -- unknown never convicts, under the new axis structure ------------------


@pytest.mark.parametrize(
    "unknown",
    [
        dict(oos_sign_flipped=False, oos_monotonicity_reversed=False),  # no reversal
        dict(tradable=None, capacity_sufficient=None),                  # tradability unknown
        dict(known_factors_supplied=True, incremental_ic_ir=float("nan")),  # ortho unknown
        dict(net_long_short_by_cost=()),                               # no cost measured
        dict(net_long_short_by_cost=((1.0, float("nan")), (2.0, float("nan")))),
    ],
)
def test_no_unknown_ever_produces_a_fail_on_any_axis(unknown):
    """⚠️ THE safety property: with the FAILs decided first (bypassing the gate), an
    unknown that COULD convict would convict immediately and at any sample size. It
    must not — every axis FAIL tests for an explicitly KNOWN failure, so each
    unknown below yields INSUFFICIENT_DATA / NOT_ASSESSED, never a FAIL."""
    r = decide_verdict(
        _inputs(**{**_THIN_SAMPLE, **_STRONG, **_TRADABLE, **_OOS_OK, **unknown})
    )
    for name, axis in r.axes().items():
        assert axis.verdict != AXIS_FAIL, name
    assert r.verdict != REJECT


# -- the asymmetric gate: a FAIL bypasses the sample gate ------------------


def test_a_regime_flip_on_a_thin_sample_rejects_rather_than_gating():
    """THE headline: a regime-flipping factor is the project's signature failure
    mode (I5e / P3-3 / P3-4). Its N_eff ~ 3 is CORRECT — two regimes really are
    about two observations — but the MEASURED flip is a NEGATIVE finding, so the
    Predictive axis FAILs it BEFORE the sample gate and the label is REJECT."""
    r = decide_verdict(
        _inputs(
            **_THIN_SAMPLE,
            **_STRONG,
            **_TRADABLE,
            oos_available=True,
            oos_sign_flipped=True,
        )
    )
    assert r.verdict == REJECT
    assert r.predictive.verdict == AXIS_FAIL
    assert any("sign flipped" in x for x in r.predictive.reasons)
    # ... and it rejects ON THE FLIP, not with a mumbled word about the sample.
    assert not any("effective samples" in x for x in r.predictive.reasons)


def test_not_tradable_on_a_thin_sample_still_rejects():
    """"It cannot be executed" is not a statistical claim: it needs no sample."""
    r = decide_verdict(
        _inputs(
            **_THIN_SAMPLE,
            **_STRONG,
            **_OOS_OK,
            tradable=False,
            capacity_sufficient=True,
        )
    )
    assert r.verdict == REJECT
    assert r.tradable.verdict == AXIS_FAIL


def test_all_cost_negative_on_a_thin_sample_still_rejects():
    kwargs = {**_STRONG, **_OOS_OK, **_TRADABLE}
    kwargs["net_long_short_by_cost"] = ((1.0, -0.01), (2.0, -0.03), (4.0, -0.08))
    r = decide_verdict(_inputs(**_THIN_SAMPLE, **kwargs))
    assert r.verdict == REJECT
    assert any("EVERY cost scenario" in x for x in r.tradable.reasons)


def test_a_measured_failure_rejects_even_when_the_sample_facts_are_unknown():
    """A Predictive FAIL reads none of the gate's facts, so an unmeasured N_eff /
    span cannot rescue a factor that visibly flipped."""
    r = decide_verdict(
        _inputs(
            settled_rebalances=2,
            effective_samples=float("nan"),
            span_days=float("nan"),
            **_STRONG,
            **_TRADABLE,
            oos_available=True,
            oos_sign_flipped=True,
        )
    )
    assert r.verdict == REJECT


def test_a_thin_sample_with_nothing_measured_reads_cannot_tell_not_bad():
    """The subtle half of the asymmetry: no measured failure on a thin sample must
    read INSUFFICIENT-DATA ("cannot tell"), never a manufactured Reject."""
    assert decide_verdict(_inputs(**_THIN_SAMPLE)).verdict == INSUFFICIENT_DATA


def test_a_thin_sample_still_blocks_both_positive_claims():
    """The gate did not get weaker, it got NARROWER: a thin sample buys neither a
    Predictive nor an Incremental PASS. Only a MEASURED failure is let past."""
    predictive = _inputs(**_THIN_SAMPLE, **_STRONG, **_OOS_OK)
    assert predictive.effective_samples < 24  # sanity
    assert decide_verdict(predictive).predictive.verdict == AXIS_INSUFFICIENT_DATA
    incremental = _inputs(**_THIN_SAMPLE, **_STRONG, **_OOS_OK, **_INCREMENTAL_OK)
    assert decide_verdict(incremental).incremental.verdict == AXIS_INSUFFICIENT_DATA


# -- the three-part sample gate governs the statistical axes (design §6) ----


def test_each_gate_part_fires_independently_on_the_predictive_axis():
    good = {**_STRONG, **_OOS_OK}
    only_floor = VerdictThresholds(
        min_rebalances=30, min_effective_samples=1.0, min_span_days=0
    )
    r = decide_verdict(
        _inputs(settled_rebalances=20, effective_samples=20.0, **good), only_floor
    )
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any(x.startswith("raw floor") for x in r.predictive.reasons)
    assert not any(
        "effective samples" in x or "calendar span" in x for x in r.predictive.reasons
    )

    r = decide_verdict(_inputs(effective_samples=5.0, **good))
    assert any(x.startswith("effective samples (A)") for x in r.predictive.reasons)
    assert not any(
        x.startswith("raw floor") or "calendar span" in x for x in r.predictive.reasons
    )

    r = decide_verdict(_inputs(span_days=30.0, **good))
    assert any(x.startswith("calendar span (B)") for x in r.predictive.reasons)
    assert not any(
        x.startswith("raw floor") or "effective samples" in x
        for x in r.predictive.reasons
    )


def test_the_gate_names_every_part_that_failed_not_just_the_first():
    r = decide_verdict(
        _inputs(
            settled_rebalances=4, effective_samples=2.0, span_days=10.0,
            **_STRONG, **_OOS_OK,
        )
    )
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    reasons = r.predictive.reasons
    assert any(x.startswith("raw floor") for x in reasons)
    assert any(x.startswith("effective samples (A)") for x in reasons)
    assert any(x.startswith("calendar span (B)") for x in reasons)
    joined = " ".join(reasons)
    for actual, required in (("4", "12"), ("2.00", "24.0"), ("10", "365")):
        assert actual in joined and required in joined


def test_a_dense_but_short_window_is_gated_on_span_despite_many_periods():
    r = decide_verdict(
        _inputs(
            settled_rebalances=500, effective_samples=480.0, span_days=35.0,
            **_STRONG, **_OOS_OK,
        )
    )
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any(x.startswith("calendar span (B)") for x in r.predictive.reasons)


@pytest.mark.parametrize("unknown", ["effective_samples", "span_days"])
def test_an_unknown_gate_fact_fails_the_gate_and_never_passes_it(unknown):
    r = decide_verdict(_inputs(**{unknown: float("nan")}, **_STRONG, **_OOS_OK))
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("UNKNOWN" in x for x in r.predictive.reasons)


def test_effective_samples_boundary_is_inclusive():
    """Exactly min_effective_samples lets the Predictive axis PASS (>=, not >)."""
    ok = decide_verdict(
        _inputs(settled_rebalances=24, effective_samples=24.0, **_STRONG, **_OOS_OK)
    )
    assert ok.predictive.verdict == AXIS_PASS
    below = decide_verdict(
        _inputs(settled_rebalances=23, effective_samples=23.0, **_STRONG, **_OOS_OK)
    )
    assert below.predictive.verdict == AXIS_INSUFFICIENT_DATA


def test_span_days_boundary_is_inclusive():
    ok = decide_verdict(_inputs(span_days=365.0, **_STRONG, **_OOS_OK))
    assert ok.predictive.verdict == AXIS_PASS
    short = decide_verdict(_inputs(span_days=364.0, **_STRONG, **_OOS_OK))
    assert short.predictive.verdict == AXIS_INSUFFICIENT_DATA


def test_project_holdout_sample_size_hits_the_default_gate():
    """⚠️ Calibration flag, and the headline consequence of the v0.3 gate.

    The real P3-7/P3-8 holdouts settled ~21 monthly rebalances spanning ~670
    calendar days. Under the defaults the STATISTICAL axes are gated on part A
    (N_eff <= N, so 21 raw periods can never reach 24), while the Tradable axis —
    which is not a statistical claim — still PASSes, so the label is WATCH. Locked
    as a REMINDER that these defaults are UNCALIBRATED, not an endorsement."""
    real = _inputs(
        settled_rebalances=21, effective_samples=21.0, span_days=670.0, **_ADOPT_FACTS
    )
    r = decide_verdict(real)
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("effective samples (A)" in x for x in r.predictive.reasons)
    assert not any("calendar span" in x for x in r.predictive.reasons)  # 670 >= 365
    assert r.incremental.verdict == AXIS_INSUFFICIENT_DATA
    assert r.tradable.verdict == AXIS_PASS
    assert r.verdict == WATCH
    # ... and it takes lowering part A (not just the raw floor) to let it through.
    assert decide_verdict(real, VerdictThresholds(min_rebalances=8)).verdict == WATCH
    lenient = VerdictThresholds(
        min_rebalances=8, min_effective_samples=8.0, min_span_days=365
    )
    non_exploratory = _inputs(
        settled_rebalances=21, effective_samples=21.0, span_days=670.0,
        **_ADOPT_FACTS, is_exploratory=False,
    )
    assert decide_verdict(non_exploratory, lenient).verdict == ADOPT


def test_negative_sign_factor_is_judged_in_its_own_direction():
    """A low-vol factor (sign -1): negative IC / spread / orthogonalized IC ARE the
    hypothesis, on EVERY axis."""
    kwargs = dict(
        expected_ic_sign=-1,
        **_SAMPLE_OK,
        ic_ir=-0.8,
        ic_ir_ci_low=-1.05,                  # CI brackets the negative point; the
        ic_ir_ci_high=-0.55,                 # aligned lower bound is +0.55 > 0.30
        ic_win_rate=0.7,
        ic_nw_t=-3.5,
        monotonicity_spearman=-0.9,          # high-vol bucket underperforms
        net_long_short_by_cost=((1.0, -0.05),),  # QN - Q1 negative: as predicted
        # v0.8: aligning a NET spread needs the GROSS leg difference, because the
        # cost (gross - net = 0.01) must stay SUBTRACTED after the legs are
        # flipped: aligned = -(-0.04) - 0.01 = +0.03. Supplying only the net (as
        # this fixture did pre-v0.8) now reads UNKNOWN rather than being scored
        # with the costs handed back as profit.
        gross_long_short_mean=-0.04,
        known_factors_supplied=True,
        incremental_ic_ir=-0.6,              # residual IC negative: as predicted
        incremental_ic_mean=-0.04,
        incremental_ic_ir_ci_low=-0.78,
        incremental_ic_ir_ci_high=-0.42,     # aligned lower bound +0.42 > 0.30
        tradable=True,
        capacity_sufficient=True,
        oos_available=True,
        oos_sign_consistent=True,
        is_exploratory=False,
    )
    r = decide_verdict(VerdictInputs(**kwargs))
    assert r.verdict == ADOPT
    assert r.incremental.verdict == AXIS_PASS


def test_thresholds_are_configurable_but_the_rule_structure_is_not():
    weak = dict(
        ic_ir=0.1, ic_ir_ci_low=0.06, ic_ir_ci_high=0.14,
        ic_win_rate=0.52, ic_nw_t=1.0, monotonicity_spearman=0.5,
        net_long_short_by_cost=((1.0, 0.01),),
    )
    # default thresholds: the weak POINT (0.1 < 0.30) does not clear -> Predictive
    # FAILs (point-based negative finding).
    assert decide_verdict(_inputs(**weak, **_OOS_OK)).predictive.verdict == AXIS_FAIL
    # lenient thresholds: the point clears (0.1 > 0.05) AND the lower CI bound
    # (0.06 > 0.05) clears too -> PASS.
    lenient = VerdictThresholds(
        min_abs_icir=0.05, min_ic_win_rate=0.5, min_abs_nw_t=0.5,
        min_monotonicity_spearman=0.4,
    )
    assert (
        decide_verdict(_inputs(**weak, **_OOS_OK), lenient).predictive.verdict
        == AXIS_PASS
    )


# --------------------------------------------------------------------------
# The exploratory cap (design §6, v0.2, preserved): is_exploratory=True caps the
# DEPLOYMENT LABEL (an all-PASS Adopt -> Watch). Adopt IS a claim; an exploratory
# run declares it is not making one.
# --------------------------------------------------------------------------


def test_exploratory_run_is_capped_at_watch_instead_of_adopt():
    r = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=True))
    assert r.verdict == WATCH
    assert any("is_exploratory=True" in x for x in r.reasons)
    assert any("CAPPED AT WATCH" in x for x in r.reasons)


def test_the_same_facts_without_the_exploratory_flag_still_adopt():
    assert decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=False)).verdict == ADOPT
    # ... and the VerdictInputs default (is_exploratory=False) is the Adopt path.
    assert decide_verdict(_inputs(**_ADOPT_FACTS)).verdict == ADOPT


def test_the_capped_watch_still_reports_the_qualifying_evidence():
    """Capping withholds the LABEL, not the facts: the axes are identical either
    way, and the capped Watch prepends exactly the cap reason to the Adopt evidence."""
    capped = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=True))
    adopted = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=False))
    assert capped.axes() == adopted.axes()  # only the label moved
    assert set(adopted.reasons).issubset(set(capped.reasons))
    assert len(capped.reasons) == len(adopted.reasons) + 1


def test_the_cap_lands_on_watch_never_reject_or_insufficient():
    """The cap only downgrades the LABEL — it never converts a PASS into a FAIL."""
    r = decide_verdict(_inputs(**_ADOPT_FACTS, is_exploratory=True))
    assert r.verdict == WATCH
    # a genuine failure under the same flag still Rejects: the cap does not rescue it.
    failed = decide_verdict(
        _inputs(**{**_ADOPT_FACTS, "is_exploratory": True, "tradable": False})
    )
    assert failed.verdict == REJECT


@pytest.mark.parametrize(
    "failure",
    [
        dict(oos_sign_flipped=True),
        dict(oos_monotonicity_reversed=True),
        dict(tradable=False),
        dict(net_long_short_by_cost=((1.0, -0.01), (2.0, -0.03), (4.0, -0.08))),
    ],
)
def test_an_exploratory_factor_that_fails_is_still_rejected(failure):
    """A negative finding is a legitimate exploratory outcome (I5e); the cap must
    not rescue a failure."""
    r = decide_verdict(_inputs(**{**_ADOPT_FACTS, "is_exploratory": True, **failure}))
    assert r.verdict == REJECT


def test_an_exploratory_run_with_no_evidence_reads_insufficient_not_reject():
    """No OOS, no book, no execution facts -> no PASS, no FAIL -> INSUFFICIENT-DATA."""
    assert decide_verdict(_inputs(is_exploratory=True)).verdict == INSUFFICIENT_DATA


def test_the_sample_gate_still_precedes_the_exploratory_cap():
    """A too-small sample gates the statistical axes regardless of the flag."""
    r = decide_verdict(
        _inputs(settled_rebalances=23, effective_samples=23.0, **_ADOPT_FACTS, is_exploratory=True)
    )
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert r.incremental.verdict == AXIS_INSUFFICIENT_DATA
    assert r.verdict == WATCH  # tradable PASS carries it; the cap never applied


# --------------------------------------------------------------------------
# The deployment derivation end-to-end THROUGH THE REPORT (extract -> decide).
# --------------------------------------------------------------------------


def _adopt_grade_sections(**payloads) -> list:
    """Section payloads that satisfy every axis of the §6 Adopt rule.

    A caller may OVERRIDE any section payload (e.g. purity= to test a redundant
    book), so the overrides are merged over the defaults rather than passed
    alongside them (which would be a duplicate keyword).
    """
    defaults = dict(
        data_coverage=_COVERAGE_OK,
        predictive_power={
            "ic_ir": 0.8,
            "ic_ir_ci_low": 0.55,
            "ic_ir_ci_high": 1.05,
            "ic_win_rate": 0.7,
            "ic_nw_t": 3.5,
        },
        return_risk={
            "monotonicity_spearman": 0.9,
            "net_long_short_by_cost": {1.0: 0.05, 2.0: 0.04},
        },
        purity={
            "known_factors_supplied": True,
            "incremental_ic_ir": 0.6,
            "incremental_ic_mean": 0.04,
            "incremental_ic_ir_ci_low": 0.42,
            "incremental_ic_ir_ci_high": 0.78,
        },
        oos_generalization={"oos_available": True, "sign_consistent": True},
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    defaults.update(payloads)
    return _full_sections(**defaults)


def test_post_hoc_selection_cannot_reach_adopt_through_the_report():
    """§4 forces post_hoc_selected=True => is_exploratory=True, so the §6 cap
    closes the post-hoc -> Adopt hole end to end."""
    cfg = _cfg(is_exploratory=True, post_hoc_selected=True)
    report = FactorEvalReport.assemble(_spec(), cfg, _adopt_grade_sections()).with_verdict()
    assert report.verdict.verdict == WATCH
    assert any("is_exploratory=True" in r for r in report.verdict.reasons)


def test_the_report_reads_the_cap_off_the_config_not_the_caveats_payload():
    cfg = _cfg(is_exploratory=True)
    lying = _adopt_grade_sections(caveats={"is_exploratory": False})
    assert (
        FactorEvalReport.assemble(_spec(), cfg, lying).with_verdict().verdict.verdict
        == WATCH
    )
    without = _adopt_grade_sections()[:-1] + [Skipped("caveats", "not produced")]
    assert (
        FactorEvalReport.assemble(_spec(), cfg, without).with_verdict().verdict.verdict
        == WATCH
    )


def test_a_non_exploratory_report_can_still_reach_adopt():
    report = FactorEvalReport.assemble(
        _spec(), _cfg(is_exploratory=False), _adopt_grade_sections()
    ).with_verdict()
    assert report.verdict.verdict == ADOPT
    assert report.verdict.incremental.verdict == AXIS_PASS


def test_a_report_without_a_book_tops_out_at_watch():
    """The multi-factor point through the report: a purity section with no book flag
    leaves the Incremental axis NOT_ASSESSED, so the label maxes at WATCH."""
    sections = _adopt_grade_sections(purity={})  # purity present, but no book
    report = FactorEvalReport.assemble(
        _spec(), _cfg(is_exploratory=False), sections
    ).with_verdict()
    assert report.verdict.verdict == WATCH
    assert report.verdict.incremental.verdict == AXIS_NOT_ASSESSED


def test_a_report_with_a_redundant_factor_is_rejected_through_the_report():
    """THE headline new capability: a factor redundant with the supplied book gets
    REJECTED even when its raw predictive signal and tradability are perfect."""
    redundant = _adopt_grade_sections(
        purity={
            "known_factors_supplied": True,
            "incremental_ic_ir": 0.01,
            "incremental_ic_mean": 0.0,
        }
    )
    report = FactorEvalReport.assemble(
        _spec(), _cfg(is_exploratory=False), redundant
    ).with_verdict()
    assert report.verdict.verdict == REJECT
    assert report.verdict.incremental.verdict == AXIS_FAIL


def test_verdict_thresholds_validated():
    with pytest.raises(ValueError, match="min_rebalances"):
        VerdictThresholds(min_rebalances=0)
    with pytest.raises(ValueError, match="min_ic_win_rate"):
        VerdictThresholds(min_ic_win_rate=1.5)


# --------------------------------------------------------------------------
# VerdictThresholds type + range validation.
# These 5 numbers decide Adopt/Watch/Reject, so a garbage threshold yields a
# garbage VERDICT. Structural only: passing says NOTHING about calibration.
# --------------------------------------------------------------------------

_THRESHOLD_FIELDS = (
    "min_rebalances",
    "min_effective_samples",
    "min_span_days",
    "min_abs_icir",
    "min_incremental_abs_icir",
    "min_ic_win_rate",
    "min_abs_nw_t",
    "min_monotonicity_spearman",
)


def test_every_threshold_field_is_validated():
    """Drift guard: a NEW threshold cannot be added without validation."""
    import dataclasses

    declared = {f.name for f in dataclasses.fields(VerdictThresholds)}
    assert declared == set(_THRESHOLD_FIELDS), (
        "VerdictThresholds gained/lost a field: add it to the validators in "
        "verdict.py AND to _THRESHOLD_FIELDS here."
    )


@pytest.mark.parametrize("field_name", _THRESHOLD_FIELDS)
@pytest.mark.parametrize("value", [True, False])
def test_verdict_thresholds_reject_bool(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        VerdictThresholds(**{field_name: value})


@pytest.mark.parametrize("field_name", _THRESHOLD_FIELDS)
def test_verdict_thresholds_reject_a_string_with_a_readable_error(field_name):
    """A bare TypeError leaking out of '"3" < 1' is not a readable error."""
    with pytest.raises(ValueError, match=field_name):
        VerdictThresholds(**{field_name: "3"})


@pytest.mark.parametrize("field_name", _THRESHOLD_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_verdict_thresholds_reject_non_finite(field_name, value):
    """NaN/inf silently disable a rule: every '>' against NaN is False."""
    with pytest.raises(ValueError, match=field_name):
        VerdictThresholds(**{field_name: value})


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("min_rebalances", 0),
        ("min_rebalances", -5),
        ("min_abs_icir", -0.1),
        ("min_abs_nw_t", -1.0),
        ("min_ic_win_rate", 1.5),
        ("min_ic_win_rate", -0.1),
        ("min_monotonicity_spearman", 1.5),
        ("min_monotonicity_spearman", -0.1),
    ],
)
def test_verdict_thresholds_reject_out_of_range(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        VerdictThresholds(**{field_name: value})


def test_verdict_thresholds_accept_their_inclusive_lower_bounds():
    """Lower bounds are inclusive; the upper bounds are EXCLUSIVE (see below)."""
    thr = VerdictThresholds(
        min_rebalances=1,
        min_abs_icir=0.0,
        min_ic_win_rate=0.0,
        min_abs_nw_t=0.0,
        min_monotonicity_spearman=0.0,
    )
    assert thr.min_rebalances == 1
    assert thr.min_monotonicity_spearman == 0.0


@pytest.mark.parametrize(
    "field_name", ["min_ic_win_rate", "min_monotonicity_spearman"]
)
def test_bounded_thresholds_reject_exactly_one_point_zero(field_name):
    """1.0 is unreachable: both metrics max out at 1.0 and the rules use '>'.

    Same class of silent breakage as NaN/inf — a threshold that quietly makes the
    rule it gates unsatisfiable — it just fails STRICT (the rule can never fire)
    instead of permissive.
    """
    with pytest.raises(ValueError, match=field_name) as excinfo:
        VerdictThresholds(**{field_name: 1.0})
    message = str(excinfo.value)
    assert "[0.0, 1.0)" in message  # the exclusive bound is stated
    assert "strict" in message  # and WHY 1.0 can never be exceeded


@pytest.mark.parametrize(
    "field_name", ["min_ic_win_rate", "min_monotonicity_spearman"]
)
def test_bounded_thresholds_accept_just_under_one(field_name):
    assert getattr(VerdictThresholds(**{field_name: 0.999}), field_name) == 0.999


def test_a_threshold_just_under_one_still_flows_through_decide_verdict():
    """0.999 is legal AND has teeth: it really gates the Predictive PASS rule."""
    near_perfect = VerdictThresholds(
        min_ic_win_rate=0.999, min_monotonicity_spearman=0.999
    )
    facts = _inputs(**_STRONG, **_TRADABLE, **_OOS_OK)
    # defaults: OOS + strong metrics -> Predictive PASS (deployment WATCH, no book).
    assert decide_verdict(facts).predictive.verdict == AXIS_PASS
    assert decide_verdict(facts).verdict == WATCH
    # demanding a >0.999 win rate: 0.7 no longer clears -> the Predictive axis fails.
    assert decide_verdict(facts, near_perfect).predictive.verdict == AXIS_FAIL


def test_valid_custom_thresholds_still_flow_through_decide_verdict():
    """Validation must not break the configurability the axis rules depend on."""
    lenient = VerdictThresholds(
        min_rebalances=8,
        min_effective_samples=8.0,
        min_span_days=180,
        min_abs_icir=0.05,
        min_ic_win_rate=0.5,
        min_abs_nw_t=0.5,
        min_monotonicity_spearman=0.4,
    )
    real = _inputs(
        settled_rebalances=21, effective_samples=21.0, span_days=200.0,
        **_ADOPT_FACTS, is_exploratory=False,
    )
    assert decide_verdict(real, lenient).verdict == ADOPT
    # defaults gate the statistical axes; the Tradable axis still PASSes -> WATCH.
    assert decide_verdict(real).verdict == WATCH


def test_int_is_accepted_for_a_float_threshold():
    """1 (int) is a perfectly good |ICIR| threshold; only bool is excluded."""
    assert VerdictThresholds(min_abs_icir=1).min_abs_icir == 1


# --------------------------------------------------------------------------
# #3: pre-registered criteria + lower-CI gating (design §6, v0.6)
# --------------------------------------------------------------------------


def test_predictive_pass_gates_on_the_lower_ci_bound_not_the_point():
    """THE CI-gating headline: a POINT that clears the bar but a LOWER CI that does
    NOT -> the axis is NOT a PASS. The sample gate passed (sufficient N_eff), so this
    is the 'promising, unconfirmed' INSUFFICIENT ABOVE the gate — not a gate failure
    below it, and not a FAIL."""
    wide = decide_verdict(
        _inputs(
            ic_ir=0.8, ic_ir_ci_low=0.10, ic_ir_ci_high=1.50,  # lower 0.10 < 0.30
            ic_win_rate=0.7, ic_nw_t=3.5, monotonicity_spearman=0.9, **_OOS_OK,
        )
    )
    assert wide.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert wide.predictive.verdict != AXIS_PASS
    assert any("LOWER CI" in r for r in wide.predictive.reasons)


def test_predictive_pass_when_the_lower_ci_clears():
    """Regression: a tight estimate whose lower CI clears -> PASS (Adopt reachable)."""
    assert decide_verdict(_inputs(**_STRONG, **_OOS_OK)).predictive.verdict == AXIS_PASS


def test_incremental_pass_gates_on_the_lower_ci_bound():
    wide = decide_verdict(
        _inputs(
            **_STRONG, **_OOS_OK,
            known_factors_supplied=True, incremental_ic_ir=0.6, incremental_ic_mean=0.04,
            incremental_ic_ir_ci_low=0.05, incremental_ic_ir_ci_high=1.15,  # lower < 0.30
        )
    )
    assert wide.incremental.verdict == AXIS_INSUFFICIENT_DATA
    assert wide.incremental.verdict != AXIS_PASS


def test_a_nan_ci_bound_never_produces_a_fail_and_never_passes():
    """Unknown never convicts, under CI gating: a point that clears in the expected
    direction with an UNKNOWN CI is NOT a PASS (can't confirm) and NOT a FAIL."""
    pred = decide_verdict(
        _inputs(ic_ir=0.8, ic_win_rate=0.7, ic_nw_t=3.5, monotonicity_spearman=0.9, **_OOS_OK)
    )  # no ic_ir_ci_* -> NaN
    assert pred.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert pred.predictive.verdict != AXIS_FAIL
    incr = decide_verdict(
        _inputs(
            **_STRONG, **_OOS_OK,
            known_factors_supplied=True, incremental_ic_ir=0.6, incremental_ic_mean=0.04,
        )
    )  # no incremental_ic_ir_ci_* -> NaN
    assert incr.incremental.verdict == AXIS_INSUFFICIENT_DATA
    assert incr.incremental.verdict != AXIS_FAIL


def test_a_redundant_point_still_fails_regardless_of_its_ci():
    """FAIL stays POINT-based: a measured ~ 0 orthogonalized IC is redundant even
    with a (wide) CI attached — the CI is only consulted for the PASS test."""
    r = decide_verdict(
        _inputs(
            **_STRONG, **_OOS_OK,
            known_factors_supplied=True, incremental_ic_ir=0.01, incremental_ic_mean=0.0,
            incremental_ic_ir_ci_low=-0.40, incremental_ic_ir_ci_high=0.42,
        )
    )
    assert r.incremental.verdict == AXIS_FAIL


def test_default_criteria_source_is_stamped_default():
    report = _verdicted_report()  # _cfg declares no success_criteria
    assert report.criteria_source == "default"
    assert json.loads(report.to_json())["criteria_source"] == "default"
    assert "DEFAULT global bar" in report.render()


def test_declared_success_criteria_are_stamped_and_used():
    """A pre-registered bar declared on the frozen EvalConfig is USED and DISCLOSED."""
    criteria = VerdictThresholds(min_abs_icir=0.05)
    cfg = _cfg(is_exploratory=False, success_criteria=criteria)
    report = FactorEvalReport.assemble(
        _spec(), cfg, _adopt_grade_sections()
    ).with_verdict()
    assert report.criteria_source == "declared"
    assert report.thresholds is criteria  # the declared object, not a default
    exported = json.loads(report.to_json())
    assert exported["criteria_source"] == "declared"
    assert exported["thresholds"]["min_abs_icir"] == 0.05
    assert "PRE-REGISTERED" in report.render()


def test_a_stricter_declared_bar_flips_a_borderline_pass_through_decide_verdict():
    """The load-bearing point of pre-registration: the threshold is a run-declared
    input now, so a stricter declared bar changes the verdict end to end. _STRONG's
    ICIR lower CI bound is 0.55: a 0.50 bar clears, a 0.70 bar does not."""
    lenient = _cfg(
        is_exploratory=False, success_criteria=VerdictThresholds(min_abs_icir=0.50)
    )
    strict = _cfg(
        is_exploratory=False, success_criteria=VerdictThresholds(min_abs_icir=0.70)
    )
    passed = FactorEvalReport.assemble(
        _spec(), lenient, _adopt_grade_sections()
    ).with_verdict()
    blocked = FactorEvalReport.assemble(
        _spec(), strict, _adopt_grade_sections()
    ).with_verdict()
    assert passed.verdict.predictive.verdict == AXIS_PASS
    assert blocked.verdict.predictive.verdict != AXIS_PASS


def test_success_criteria_must_be_a_verdict_thresholds_not_a_dict():
    """A dict would be silently ignored (fall back to default) — a pre-registration
    that never took effect. Rejected loudly instead."""
    with pytest.raises(ValueError, match="success_criteria"):
        _cfg(success_criteria={"min_abs_icir": 0.3})


def test_a_declared_bar_is_not_weakened_for_being_per_run():
    """The per-run object passes exactly the same VerdictThresholds validation."""
    with pytest.raises(ValueError, match="min_ic_win_rate"):
        _cfg(success_criteria=VerdictThresholds(min_ic_win_rate=1.0))  # 1.0 unreachable
    cfg = _cfg(success_criteria=VerdictThresholds(min_abs_icir=0.4))
    assert cfg.success_criteria.min_abs_icir == 0.4


# --------------------------------------------------------------------------
# Report render / JSON: deterministic + secret-free
# --------------------------------------------------------------------------


def _verdicted_report(**payloads) -> FactorEvalReport:
    defaults = dict(
        data_coverage=_COVERAGE_OK,
        predictive_power={
            "ic_ir": 0.8,
            "ic_ir_ci_low": 0.55,
            "ic_ir_ci_high": 1.05,
            "ic_win_rate": 0.7,
            "ic_nw_t": 3.5,
        },
        return_risk={
            "monotonicity_spearman": 0.9,
            "net_long_short_by_cost": {1.0: 0.05, 2.0: 0.04},
        },
        purity={
            "known_factors_supplied": True,
            "incremental_ic_ir": 0.6,
            "incremental_ic_mean": 0.04,
            "incremental_ic_ir_ci_low": 0.42,
            "incremental_ic_ir_ci_high": 0.78,
        },
        oos_generalization={"oos_available": True, "sign_consistent": True},
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    defaults.update(payloads)
    sections = _full_sections(**defaults)
    # is_exploratory=False on purpose: these render/export tests want a report
    # that actually reaches Adopt, and the §6 cap denies Adopt to an exploratory
    # run (which _cfg() declares by default).
    report = FactorEvalReport.assemble(_spec(), _cfg(is_exploratory=False), sections)
    report.validate_all_mandatory_present()
    return report.with_verdict()


def test_render_requires_a_verdict():
    report = FactorEvalReport.assemble(_spec(), _cfg(), _full_sections())
    with pytest.raises(ValueError, match="no verdict"):
        report.render()
    with pytest.raises(ValueError, match="no verdict"):
        report.to_dict()


def test_render_is_deterministic_and_contains_all_ten_sections():
    report = _verdicted_report()
    first, second = report.render(), report.render()
    assert first == second
    assert first.startswith("# Factor Evaluation — unit_test_factor (v1.0)")
    assert "## 0. Header & Provenance" in first
    assert "## 1. Verdict & Scorecard" in first
    for number, name in enumerate(MANDATORY_SECTIONS, start=2):
        assert f"## {number}." in first, name
    assert "Adopt" in first


def test_render_discloses_a_skipped_section_with_its_reason():
    sections = _full_sections()[:-1] + [Skipped("caveats", "no caveats configured")]
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections).with_verdict()
    rendered = report.render()
    assert "_Skipped: no caveats configured_" in rendered


def test_payload_key_order_does_not_change_the_render():
    """Insertion order must not leak into the artifact (sorted keys)."""
    a = Section("purity", {"vif": 1.2, "corr_value_ep": 0.3})
    b = Section("purity", {"corr_value_ep": 0.3, "vif": 1.2})
    spec, cfg = _spec(), _cfg()
    ra = FactorEvalReport.assemble(spec, cfg, _full_sections()[:3] + [a] + _full_sections()[4:])
    rb = FactorEvalReport.assemble(spec, cfg, _full_sections()[:3] + [b] + _full_sections()[4:])
    assert ra.with_verdict().render() == rb.with_verdict().render()


def test_json_export_is_deterministic_and_machine_readable():
    report = _verdicted_report()
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "0.1"
    assert payload["spec"]["factor_id"] == "unit_test_factor"
    assert payload["verdict"]["verdict"] == ADOPT
    assert {s["name"] for s in payload["sections"]} == set(MANDATORY_SECTIONS)
    assert report.to_json() == report.to_json()


def test_report_is_secret_free():
    """A token/path smuggled into a payload or reason never reaches the artifact."""
    leaky = "/home/u/.config.json token=abcdef123456"
    sections = _full_sections()[:-1] + [Skipped("caveats", f"loaded {leaky}")]
    sections[3] = Section("purity", {"src": leaky}, note=f"note {leaky}")
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections).with_verdict()
    for text in (report.render(), report.to_json()):
        assert "abcdef123456" not in text
        assert ".config.json" not in text
        assert "[REDACTED]" in text


def test_report_redacts_a_secret_looking_payload_key():
    """Keys reach the artifact too, not just values.

    D3's own _format_examples redacts example KEYS; leaving them raw here was
    both leaky and inconsistent with the layer this reuses.
    """
    report = _verdicted_report(
        purity={"token=abcdef123456": 1},
        stability_cost={"/home/u/.config.json": 2},
    )
    for text in (report.render(), report.to_json()):
        assert "abcdef123456" not in text
        assert ".config.json" not in text
        assert "[REDACTED]" in text


def test_a_nested_secret_looking_payload_key_is_redacted_too():
    """Redaction recurses: a key nested inside a payload value is redacted too.

    NOTE the assertion is scoped to what the D3 layer actually promises — it
    matches secret-SHAPED patterns (``tushare.token`` / ``token=...`` /
    ``*.config.json``), it is not a secrets detector. A bare 'abcdef123456' with
    no such marker is left alone by design, so asserting otherwise here would
    lock in a guarantee the layer does not make.
    """
    report = _verdicted_report(
        purity={"sources": {"tushare.token": "token=abcdef123456"}}
    )
    for text in (report.render(), report.to_json()):
        assert "tushare.token" not in text  # the nested KEY
        assert "abcdef123456" not in text  # the pattern-shaped VALUE
        assert "[REDACTED]" in text


def test_redacting_keys_keeps_the_render_deterministic():
    """Two secret-shaped keys can collapse onto one marker; it must not flap."""
    report = _verdicted_report(purity={"token=aaa": 1, "token=bbb": 2})
    assert report.render() == report.render()
    assert report.to_json() == report.to_json()


def test_section_payload_is_copied_not_aliased():
    payload = {"ic_ir": 0.5}
    section = Section("predictive_power", payload)
    payload["ic_ir"] = 999.0  # mutating the caller's dict must not touch the section
    assert section.payload["ic_ir"] == 0.5


def test_section_payload_nested_values_are_deep_copied():
    """A shallow dict() left the NESTED map aliased to the caller's object.

    ``net_long_short_by_cost`` is the exact key the verdict reads, so a post-hoc
    nested mutation could rewrite a finished report's facts.
    """
    payload = {"net_long_short_by_cost": {1.0: 0.05, 2.0: 0.04}, "tags": ["base"]}
    section = Section("return_risk", payload)
    payload["net_long_short_by_cost"][1.0] = -999.0
    payload["tags"].append("mutated")
    assert section.payload["net_long_short_by_cost"] == {1.0: 0.05, 2.0: 0.04}
    assert section.payload["tags"] == ["base"]


def test_nested_mutation_after_the_fact_cannot_rewrite_the_verdict():
    """End-to-end of the same hole: an Adopt must not be mutable into a Reject."""
    costs = {1.0: 0.05, 2.0: 0.04}
    sections = _full_sections(
        data_coverage=dict(_COVERAGE_OK),
        predictive_power={
            "ic_ir": 0.8,
            "ic_ir_ci_low": 0.55,
            "ic_ir_ci_high": 1.05,
            "ic_win_rate": 0.7,
            "ic_nw_t": 3.5,
        },
        return_risk={"monotonicity_spearman": 0.9, "net_long_short_by_cost": costs},
        purity={
            "known_factors_supplied": True,
            "incremental_ic_ir": 0.6,
            "incremental_ic_mean": 0.04,
            "incremental_ic_ir_ci_low": 0.42,
            "incremental_ic_ir_ci_high": 0.78,
        },
        oos_generalization={"oos_available": True, "sign_consistent": True},
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    report = FactorEvalReport.assemble(
        _spec(), _cfg(is_exploratory=False), sections
    ).with_verdict()
    assert report.verdict.verdict == ADOPT
    costs[1.0] = -1.0  # flip the base spread AFTER the report was built
    assert report.with_verdict().verdict.verdict == ADOPT


# --------------------------------------------------------------------------
# A verdicted report is ALWAYS a complete report (enforcement layer #2 made
# unbypassable), and no export may silently omit a mandatory section.
# --------------------------------------------------------------------------


def test_with_verdict_validates_even_when_the_caller_skips_the_explicit_check():
    """The contract's central promise must not depend on caller discipline.

    ``assemble().with_verdict()`` is public; without an internal validation a
    3-of-8 report gets a real verdict and renders as a corrupt record.
    """
    report = FactorEvalReport.assemble(_spec(), _cfg(), _full_sections()[:3])
    with pytest.raises(ValueError, match="missing mandatory section"):
        report.with_verdict()  # no explicit validate_all_mandatory_present() call


def test_with_verdict_names_every_missing_section():
    report = FactorEvalReport.assemble(_spec(), _cfg(), _full_sections()[:3])
    with pytest.raises(ValueError) as excinfo:
        report.with_verdict()
    for name in MANDATORY_SECTIONS[3:]:
        assert name in str(excinfo.value)


def test_with_verdict_still_accepts_a_complete_report_of_skipped_sections():
    """Disclosure, not results: Skipped-with-reason is a complete report."""
    sections = [Skipped(name, "not configured") for name in MANDATORY_SECTIONS]
    report = FactorEvalReport.assemble(_spec(), _cfg(), sections).with_verdict()
    assert report.verdict is not None


def _hand_built_incomplete_report() -> FactorEvalReport:
    """Bypass assemble+with_verdict entirely — the only route left to an export."""
    return FactorEvalReport(
        spec=_spec(),
        cfg=_cfg(),
        sections=tuple(_full_sections()[:3]),
        verdict=VerdictResult(REJECT, ("hand-built",)),
    )


def test_json_marks_a_missing_mandatory_section_instead_of_dropping_it():
    """The actual hole: render() printed _MISSING_ 5x while to_json() shipped 3/8.

    A short record is structurally valid and would silently poison the cross-run
    factor library, so the JSON states the hole explicitly.
    """
    payload = json.loads(_hand_built_incomplete_report().to_json())
    names = [s["name"] for s in payload["sections"]]
    assert names[: len(MANDATORY_SECTIONS)] == list(MANDATORY_SECTIONS)
    by_name = {s["name"]: s for s in payload["sections"]}
    for name in MANDATORY_SECTIONS[:3]:
        assert by_name[name]["status"] == "ok"
    for name in MANDATORY_SECTIONS[3:]:
        assert by_name[name]["status"] == "missing"
        assert "MISSING" in by_name[name]["reason"]


def test_markdown_and_json_agree_about_a_hole():
    report = _hand_built_incomplete_report()
    rendered = report.render()
    assert rendered.count("_MISSING — the contract requires this section._") == 5
    exported = report.to_dict()
    n_missing = sum(1 for s in exported["sections"] if s["status"] == "missing")
    assert n_missing == 5


def test_json_section_order_is_canonical_not_assembly_order():
    """Two equivalent reports assembled in different order must be byte-identical.

    sort_keys=True sorts dict KEYS, not list elements — the sections list has to
    be canonicalized itself, or a cross-run diff is pure noise.
    """
    spec, cfg = _spec(), _cfg()
    payloads = dict(
        data_coverage=_COVERAGE_OK,
        predictive_power={"ic_ir": 0.8, "ic_win_rate": 0.7, "ic_nw_t": 3.5},
        return_risk={
            "monotonicity_spearman": 0.9,
            "net_long_short_by_cost": {1.0: 0.05},
        },
        oos_generalization={"oos_available": True, "sign_consistent": True},
        execution_capacity={"tradable": True, "capacity_sufficient": True},
    )
    forward = _full_sections(**payloads)
    backward = list(reversed(forward))
    a = FactorEvalReport.assemble(spec, cfg, forward).with_verdict()
    b = FactorEvalReport.assemble(spec, cfg, backward).with_verdict()
    assert a.to_json() == b.to_json()
    assert a.render() == b.render()
    names = [s["name"] for s in json.loads(a.to_json())["sections"]]
    assert names == list(MANDATORY_SECTIONS)


def test_extra_sections_are_canonicalized_by_name_too():
    spec, cfg = _spec(), _cfg()
    extras = [Section("crowding", {"x": 1}), Section("alt_data", {"y": 2})]
    a = FactorEvalReport.assemble(spec, cfg, [*_full_sections(), *extras])
    b = FactorEvalReport.assemble(spec, cfg, [*_full_sections(), *reversed(extras)])
    assert a.with_verdict().to_json() == b.with_verdict().to_json()
    names = [s["name"] for s in json.loads(a.with_verdict().to_json())["sections"]]
    assert names == [*MANDATORY_SECTIONS, "alt_data", "crowding"]


def test_a_huge_payload_value_is_truncated_in_the_artifact():
    """Bounded like the D3 quality layer's examples: a report is a summary."""
    report = _verdicted_report(purity={"dump": "x" * 5000})
    rendered = report.render()
    assert "...[truncated]" in rendered
    assert "x" * 5000 not in rendered
    exported = report.to_dict()
    dump = next(s for s in exported["sections"] if s["name"] == "purity")["payload"]
    assert len(dump["dump"]) == MAX_VALUE_CHARS + len("...[truncated]")


# --------------------------------------------------------------------------
# FactorEvaluator ABC
# --------------------------------------------------------------------------


class _SpyEvaluator(FactorEvaluator):
    """Minimal evaluator: records call order, computes nothing (PR-B does that)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def build_ir(self, factor_panel, spec, cfg, ctx=None):
        self.calls.append("build_ir")
        return object()

    def _record(self, name: str) -> Section:
        self.calls.append(name)
        return Section(name, {})

    def predictive_power(self, ir):
        self.calls.append("predictive_power")
        return Section("predictive_power", {"ic_ir": 0.8, "ic_win_rate": 0.7, "ic_nw_t": 3.5})

    def return_risk(self, ir):
        self.calls.append("return_risk")
        return Section(
            "return_risk",
            {"monotonicity_spearman": 0.9, "net_long_short_by_cost": {1.0: 0.05}},
        )

    def stability_cost(self, ir):
        return self._record("stability_cost")

    def purity(self, ir):
        return self._record("purity")

    def oos_generalization(self, ir):
        self.calls.append("oos_generalization")
        return Skipped("oos_generalization", "no oos_split configured")

    def execution_capacity(self, ir):
        return self._record("execution_capacity")

    def data_coverage(self, ir):
        self.calls.append("data_coverage")
        # clears the three-part gate, so the ORDER/verdict assertions below are
        # about the template method rather than the sample gate.
        return Section(
            "data_coverage",
            {"settled_rebalances": 30, "effective_samples": 30.0, "span_days": 400.0},
        )

    def caveats(self, ir):
        return self._record("caveats")


def test_evaluator_missing_a_mandatory_section_cannot_be_instantiated():
    """The ABC is lock #1: you cannot drop a mandatory section, only Skip it."""

    class MissingCaveats(FactorEvaluator):
        def build_ir(self, factor_panel, spec, cfg, ctx=None):
            return object()

        def predictive_power(self, ir):
            return Section("predictive_power", {})

        def return_risk(self, ir):
            return Section("return_risk", {})

        def stability_cost(self, ir):
            return Section("stability_cost", {})

        def purity(self, ir):
            return Section("purity", {})

        def oos_generalization(self, ir):
            return Section("oos_generalization", {})

        def execution_capacity(self, ir):
            return Section("execution_capacity", {})

        def data_coverage(self, ir):
            return Section("data_coverage", {})

        # 'caveats' deliberately not implemented

    with pytest.raises(TypeError, match="abstract"):
        MissingCaveats()


def test_evaluator_missing_the_ir_builder_cannot_be_instantiated():
    class NoIR(_SpyEvaluator):
        build_ir = FactorEvaluator.build_ir

    with pytest.raises(TypeError, match="abstract"):
        NoIR()


def test_template_method_calls_sections_in_the_fixed_order():
    spy = _SpyEvaluator()
    report = spy.evaluate(factor_panel=None, spec=_spec(), cfg=_cfg())
    assert spy.calls == ["build_ir", *MANDATORY_SECTIONS]
    report.validate_all_mandatory_present()
    assert report.verdict is not None


def test_template_method_verdict_reflects_a_skipped_oos_section():
    """Skipped OOS -> Predictive INSUFFICIENT_DATA; no book / no execution facts
    -> Incremental & Tradable NOT_ASSESSED -> no PASS, no FAIL -> INSUFFICIENT-DATA.

    (Under the scalar verdict this was Watch; the three-axis structure reads it as
    "we cannot tell" — there is no positive claim on any axis.)
    """
    report = _SpyEvaluator().evaluate(factor_panel=None, spec=_spec(), cfg=_cfg())
    assert report.verdict.verdict == INSUFFICIENT_DATA
    assert report.verdict.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("no out-of-sample split" in r for r in report.verdict.reasons)
    assert "_Skipped: no oos_split configured_" in report.render()


# --------------------------------------------------------------------------
# v0.8 fix #1: the hypothesis-aligned NET spread must SUBTRACT cost, not add it
# back (design §6, v0.8). The aligned spread is `sign * gross - cost`, never
# `sign * net` — the latter expands at sign=-1 to `-gross + cost`.
# --------------------------------------------------------------------------


def test_aligned_net_at_positive_sign_returns_the_net_untouched():
    """sign=+1 is BIT-IDENTICAL to the pre-v0.8 expression (`1 * net == net`)."""
    for gross, net in ((0.05, 0.04), (0.000125, -0.001983), (-0.02, -0.03)):
        assert _aligned_net(net, gross, 1) == net
    # and it does not even need the gross: an unknown gross must not turn a
    # perfectly known sign=+1 spread into UNKNOWN.
    assert _aligned_net(0.04, float("nan"), 1) == 0.04


def test_aligned_net_at_negative_sign_flips_the_legs_then_subtracts_cost():
    """The worked example from the v0.8 task card, to the last digit.

    gross = 0.000125, net(1x) = -0.001983  =>  cost = 0.002108
    aligned = -gross - cost = -0.000125 - 0.002108 = -0.002233
    The pre-v0.8 `sign * net` gave +0.001983 — a POSITIVE spread conjured out of
    a factor that loses money gross AND pays 0.002108 to trade.
    """
    gross, net = 0.000125, -0.001983
    assert _aligned_net(net, gross, -1) == pytest.approx(-0.002233, abs=1e-12)
    # the defective reading, named so a regression cannot quietly restore it
    assert _aligned_net(net, gross, -1) != pytest.approx(-1 * net, abs=1e-9)


def test_aligned_net_is_always_worse_than_the_gross_flip_by_exactly_the_cost():
    """Cost is a DRAG in both directions: aligned = sign*gross - cost, always."""
    gross, net = 0.000125, -0.001983
    cost = gross - net
    assert cost > 0
    assert _aligned_net(net, gross, -1) == pytest.approx(-gross - cost, abs=1e-12)


def test_base_spread_reads_the_aligned_cost_subtracted_value_at_sign_minus_one():
    inputs = VerdictInputs(
        expected_ic_sign=-1,
        net_long_short_by_cost=((1.0, -0.001983), (2.0, -0.004091)),
        gross_long_short_mean=0.000125,
    )
    assert _base_spread(inputs) == pytest.approx(-0.002233, abs=1e-12)


def test_all_spreads_negative_is_true_when_every_aligned_scenario_is_negative():
    """sign=-1, gross known, cost subtracted -> every scenario negative -> the
    Tradable axis has a KNOWN failure to convict on."""
    inputs = VerdictInputs(
        expected_ic_sign=-1,
        net_long_short_by_cost=((1.0, -0.001983), (2.0, -0.004091), (4.0, -0.008307)),
        gross_long_short_mean=0.000125,
    )
    assert _all_spreads_negative(inputs) is True
    r = decide_verdict(
        VerdictInputs(
            expected_ic_sign=-1,
            **_SAMPLE_OK,
            **_TRADABLE,
            net_long_short_by_cost=(
                (1.0, -0.001983), (2.0, -0.004091), (4.0, -0.008307),
            ),
            gross_long_short_mean=0.000125,
        )
    )
    assert r.tradable.verdict == AXIS_FAIL
    assert any("EVERY cost scenario" in x for x in r.tradable.reasons)


def test_an_unknown_gross_makes_a_negative_sign_aligned_spread_unknown():
    """UNKNOWN NEVER CONVICTS, and never passes either: without the gross the cost
    cannot be recovered from the net, so the aligned spread is not a fact."""
    inputs = VerdictInputs(
        expected_ic_sign=-1,
        net_long_short_by_cost=((1.0, -0.001983), (2.0, -0.004091)),
        # gross_long_short_mean deliberately absent -> NaN
    )
    assert math.isnan(_base_spread(inputs))
    assert _all_spreads_negative(inputs) is False
    r = decide_verdict(
        VerdictInputs(
            expected_ic_sign=-1,
            **_SAMPLE_OK,
            **_TRADABLE,
            net_long_short_by_cost=((1.0, -0.001983), (2.0, -0.004091)),
        )
    )
    assert r.tradable.verdict == AXIS_INSUFFICIENT_DATA


def test_positive_sign_tradable_axis_is_unchanged_by_the_v08_fix():
    """The whole sign=+1 world must be bit-identical: same verdicts with the gross
    supplied, absent, or nonsense — it is never consulted at sign=+1."""
    for gross in (0.06, float("nan"), -99.0):
        r = decide_verdict(
            _inputs(**_STRONG, **_OOS_OK, **_TRADABLE, gross_long_short_mean=gross)
        )
        assert r.tradable.verdict == AXIS_PASS
        kwargs = {**_STRONG, **_OOS_OK, **_TRADABLE}
        kwargs["net_long_short_by_cost"] = ((1.0, -0.01), (2.0, -0.03))
        failed = decide_verdict(_inputs(**kwargs, gross_long_short_mean=gross))
        assert failed.tradable.verdict == AXIS_FAIL


# --------------------------------------------------------------------------
# v0.8 fix #2: the Predictive axis gates on the PER-DATE monotonicity, and
# discloses when it had to fall back to the pooled one.
# --------------------------------------------------------------------------


def test_predictive_gate_reads_the_per_date_monotonicity_over_the_pooled_one():
    """When both are present the per-date figure decides — and it decides BOTH
    ways, so this is not just 'the new field is accepted'."""
    # pooled says REVERSED, per-date says correctly ordered -> the axis is not
    # convicted by the magnitude-sensitive statistic.
    rescued = decide_verdict(
        _inputs(
            **{**_STRONG, "monotonicity_spearman": -0.5,
               "monotonicity_spearman_by_date": 0.7},
            **_OOS_OK,
        )
    )
    assert rescued.predictive.verdict == AXIS_PASS
    # pooled looks fine, per-date says REVERSED -> FAIL. The new gate has teeth.
    convicted = decide_verdict(
        _inputs(
            **{**_STRONG, "monotonicity_spearman": 0.9,
               "monotonicity_spearman_by_date": -0.3},
            **_OOS_OK,
        )
    )
    assert convicted.predictive.verdict == AXIS_FAIL
    assert any("monotonicity" in x for x in convicted.predictive.reasons)


def test_a_missing_per_date_monotonicity_falls_back_and_says_so():
    """A pre-v0.8 IR stays judgeable, but the report must not pretend the weaker
    statistic is the new one."""
    r = decide_verdict(_inputs(**_STRONG, **_OOS_OK))  # pooled only
    assert r.predictive.verdict == AXIS_PASS
    assert any("FELL BACK" in x for x in r.predictive.reasons)
    assert any("magnitude-sensitive" in x for x in r.predictive.reasons)


def test_no_fallback_note_when_the_per_date_monotonicity_is_supplied():
    r = decide_verdict(
        _inputs(**{**_STRONG, "monotonicity_spearman_by_date": 0.7}, **_OOS_OK)
    )
    assert r.predictive.verdict == AXIS_PASS
    assert not any("FELL BACK" in x for x in r.predictive.reasons)


def test_the_negative_sign_monotonicity_gate_is_still_direction_aligned():
    """v0.7's direction gate is unchanged: sign=-1 wants a NEGATIVE raw per-date
    monotonicity, and a positive one is the reversal that FAILs."""
    common = dict(
        expected_ic_sign=-1, **_SAMPLE_OK, **_OOS_OK,
        ic_ir=-0.8, ic_ir_ci_low=-1.05, ic_ir_ci_high=-0.55,
        ic_win_rate=0.7, ic_nw_t=-3.5,
    )
    ok = decide_verdict(
        VerdictInputs(**common, monotonicity_spearman_by_date=-0.7)
    )
    assert ok.predictive.verdict == AXIS_PASS
    reversed_ = decide_verdict(
        VerdictInputs(**common, monotonicity_spearman_by_date=0.7)
    )
    assert reversed_.predictive.verdict == AXIS_FAIL


# --------------------------------------------------------------------------
# v0.9: the monotonicity DIRECTION gate is decided by the per-date CI and is
# THREE-VALUED (holds / contradicted / UNKNOWN).
#
# Why: the v0.8 per-date statistic turned out to be heavily attenuated by daily
# noise — an empirically perfect ladder scores ~0.05-0.11 — while the gate sat at
# a bare 0.0 with no dispersion estimate at all. Two real factors landed 0.021
# apart across that boundary.
# --------------------------------------------------------------------------


#: CI bounds only; the point is supplied too, since a real payload always carries
#: both and the reasons quote it.
def _mono(point: float, low: float, high: float) -> dict:
    return {
        "monotonicity_spearman_by_date": point,
        "monotonicity_spearman_by_date_ci_low": low,
        "monotonicity_spearman_by_date_ci_high": high,
    }


def test_the_monotonicity_direction_gate_is_three_valued_on_the_ci():
    """The three states, each reached by moving ONLY the interval."""
    # (a) entirely above the bar -> the direction may be asserted -> PASS
    holds = decide_verdict(
        _inputs(**{**_STRONG, **_mono(0.08, 0.03, 0.13)}, **_OOS_OK)
    )
    assert holds.predictive.verdict == AXIS_PASS

    # (b) entirely below 0 -> a MEASURED reversal -> FAIL
    contradicted = decide_verdict(
        _inputs(**{**_STRONG, **_mono(-0.08, -0.13, -0.03)}, **_OOS_OK)
    )
    assert contradicted.predictive.verdict == AXIS_FAIL
    assert any("CONTRADICTED" in x for x in contradicted.predictive.reasons)

    # (c) straddling 0 -> UNKNOWN -> neither convicts nor acquits.
    #     THE POINT IS POSITIVE (0.08, which v0.8 would have waved through as a
    #     direction that "holds"): it is the INTERVAL that withholds the PASS.
    unknown = decide_verdict(
        _inputs(**{**_STRONG, **_mono(0.08, -0.02, 0.18)}, **_OOS_OK)
    )
    assert unknown.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert any("straddles" in x for x in unknown.predictive.reasons)
    assert any(
        "not a PASS and not a FAIL" in x for x in unknown.predictive.reasons
    )


def test_an_unknown_monotonicity_does_not_rescue_a_factor_that_fails_elsewhere():
    """THE routing rule. An unknown direction withholds a PASS; it must never
    withhold a FAIL that the other evidence already earned — otherwise 'we could not
    measure the ladder' would upgrade every genuinely bad factor from Reject to
    INSUFFICIENT-DATA, which is the exact opposite of quick-to-reject."""
    straddling = _mono(0.08, -0.02, 0.18)

    # each of the other predictive criteria, broken ONE AT A TIME
    weak_icir = {**_STRONG, "ic_ir": 0.05, "ic_ir_ci_low": -0.2, "ic_ir_ci_high": 0.3}
    weak_nw_t = {**_STRONG, "ic_nw_t": 0.4}
    weak_win = {**_STRONG, "ic_win_rate": 0.41}
    for broken in (weak_icir, weak_nw_t, weak_win):
        r = decide_verdict(_inputs(**{**broken, **straddling}, **_OOS_OK))
        assert r.predictive.verdict == AXIS_FAIL
        assert r.verdict == REJECT
        # and the unknown direction is DISCLOSED alongside, not swallowed
        assert any("INDISTINGUISHABLE FROM 0" in x for x in r.predictive.reasons)

    # out-of-sample inconsistency is "elsewhere" too
    oos_bad = decide_verdict(
        _inputs(
            **{**_STRONG, **straddling},
            oos_available=True,
            oos_sign_consistent=False,
        )
    )
    assert oos_bad.predictive.verdict == AXIS_FAIL

    # ... and with everything else intact the SAME unknown yields INSUFFICIENT_DATA
    alone = decide_verdict(_inputs(**{**_STRONG, **straddling}, **_OOS_OK))
    assert alone.predictive.verdict == AXIS_INSUFFICIENT_DATA


def test_an_unknown_monotonicity_axis_does_not_become_a_reject_label():
    """Predictive INSUFFICIENT_DATA + nothing failing = INSUFFICIENT-DATA, never
    Reject: the deployment derivation must not read a withheld PASS as a failure."""
    r = decide_verdict(
        _inputs(**{**_STRONG, **_mono(0.08, -0.02, 0.18)}, **_OOS_OK)
    )
    assert r.predictive.verdict == AXIS_INSUFFICIENT_DATA
    assert r.verdict == INSUFFICIENT_DATA


def test_the_aligned_monotonicity_ci_swaps_min_and_max_at_sign_minus_one():
    """Multiplying an interval by -1 REVERSES it. A low-vol factor (sign=-1) whose
    RAW per-date CI is entirely NEGATIVE is a factor whose ALIGNED CI is entirely
    positive — i.e. its hypothesis holds. Getting the swap wrong reads it backwards."""
    thr = VerdictThresholds()
    # raw [-0.13, -0.03] at sign=-1 -> aligned [+0.03, +0.13] -> HOLDS
    state, reasons = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=-1, **_mono(-0.08, -0.13, -0.03)), thr
    )
    assert state == MONO_HOLDS
    assert "[+0.0300, +0.1300]" in reasons[0]   # re-sorted, not left reversed

    # raw [+0.03, +0.13] at sign=-1 -> aligned [-0.13, -0.03] -> CONTRADICTED
    state, _ = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=-1, **_mono(0.08, 0.03, 0.13)), thr
    )
    assert state == MONO_CONTRADICTED

    # the same raw intervals at sign=+1 read the other way round
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(0.08, 0.03, 0.13)), thr
    )[0] == MONO_HOLDS
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(-0.08, -0.13, -0.03)), thr
    )[0] == MONO_CONTRADICTED

    # end to end: the sign=-1 PASS survives the new gate
    common = dict(
        expected_ic_sign=-1, **_SAMPLE_OK, **_OOS_OK,
        ic_ir=-0.8, ic_ir_ci_low=-1.05, ic_ir_ci_high=-0.55,
        ic_win_rate=0.7, ic_nw_t=-3.5,
    )
    assert decide_verdict(
        VerdictInputs(**common, **_mono(-0.08, -0.13, -0.03))
    ).predictive.verdict == AXIS_PASS
    assert decide_verdict(
        VerdictInputs(**common, **_mono(0.08, 0.03, 0.13))
    ).predictive.verdict == AXIS_FAIL


def test_the_direction_gate_is_strict_and_a_ci_touching_the_bar_is_unknown():
    """`> bar`, like every other level comparison in this module: an aligned lower
    bound sitting exactly ON 0.0 has not shown the direction holds."""
    thr = VerdictThresholds()
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(0.09, 0.0, 0.18)), thr
    )[0] == MONO_UNKNOWN
    # and an upper bound exactly at 0 is not a demonstrated reversal either
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(-0.09, -0.18, 0.0)), thr
    )[0] == MONO_UNKNOWN


def test_a_raised_bar_withholds_a_pass_without_calling_the_factor_reversed():
    """The PASS side moves with the (configurable) bar; the FAIL side stays pinned
    at 0. A correctly-ordered-but-weak ladder is UNKNOWN under a strength bar — it
    is NOT 'reversed', which is a claim about the sign."""
    strict = VerdictThresholds(min_monotonicity_spearman=0.5)
    state, reasons = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(0.08, 0.03, 0.13)), strict
    )
    assert state == MONO_UNKNOWN
    assert any("straddles the bar 0.5" in x for x in reasons)
    # genuinely reversed is still CONTRADICTED under the same raised bar
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(-0.08, -0.13, -0.03)), strict
    )[0] == MONO_CONTRADICTED


def test_the_monotonicity_fallback_chain_has_two_disclosed_levels():
    """CI absent -> the bare per-date point (v0.8); per-date point absent too ->
    the pooled magnitude statistic (v0.7). Each level is ANNOUNCED in the reasons,
    because judging on a weaker statistic while presenting a v0.9 verdict would be
    the silent degradation this whole layer exists to prevent."""
    thr = VerdictThresholds()

    # level 1: point present, CI absent -> two-valued, v0.8 behaviour
    holds, notes = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, monotonicity_spearman_by_date=0.7), thr
    )
    assert holds == MONO_HOLDS
    assert any("REVERTED to the BARE per-date POINT" in n for n in notes)
    assert not any("FELL BACK to the pooled" in n for n in notes)
    contradicted, _ = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, monotonicity_spearman_by_date=-0.3), thr
    )
    # NOT unknown: without a dispersion estimate there is nothing to be uncertain
    # with, and inventing an UNKNOWN would turn every pre-v0.9 FAIL into
    # INSUFFICIENT_DATA behind the reader's back.
    assert contradicted == MONO_CONTRADICTED

    # level 2: per-date figure absent too -> the pooled one, both notes present
    pooled, notes2 = _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, monotonicity_spearman=0.9), thr
    )
    assert pooled == MONO_HOLDS
    assert any("FELL BACK to the pooled" in n for n in notes2)
    assert any("REVERTED to the BARE per-date POINT" in n for n in notes2)

    # nothing at all: unmeasurable monotonicity still lands where v0.7/v0.8 put it
    assert _monotonicity_direction(VerdictInputs(expected_ic_sign=+1), thr)[0] \
        == MONO_CONTRADICTED

    # and the levels surface through the real axis, not just the helper
    axis = decide_verdict(
        _inputs(**{**_STRONG, "monotonicity_spearman_by_date": 0.7}, **_OOS_OK)
    ).predictive
    assert axis.verdict == AXIS_PASS
    assert any("REVERTED to the BARE per-date POINT" in x for x in axis.reasons)


def test_a_supplied_ci_is_used_even_when_it_disagrees_with_the_point():
    """The CI is the gate, not a decoration on the point. A point above the bar with
    an interval that straddles 0 must NOT pass — that is the v0.9 fix — and a point
    below 0 with an interval that clears must not be convicted on the point."""
    thr = VerdictThresholds()
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(0.9, -0.05, 1.85)), thr
    )[0] == MONO_UNKNOWN
    assert _monotonicity_direction(
        VerdictInputs(expected_ic_sign=+1, **_mono(-0.4, 0.05, 0.20)), thr
    )[0] == MONO_HOLDS
    # a HALF-supplied CI is not a CI: fall back rather than guess the other end
    half, notes = _monotonicity_direction(
        VerdictInputs(
            expected_ic_sign=+1,
            monotonicity_spearman_by_date=0.7,
            monotonicity_spearman_by_date_ci_low=0.03,
        ),
        thr,
    )
    assert half == MONO_HOLDS
    assert any("REVERTED to the BARE per-date POINT" in n for n in notes)


def test_the_return_risk_verdict_keys_document_the_monotonicity_ci():
    """VERDICT_KEYS is the documented contract of what the verdict reads; a gate
    input missing from it is an undocumented dependency."""
    assert "monotonicity_spearman_by_date_ci_low" in VERDICT_KEYS["return_risk"]
    assert "monotonicity_spearman_by_date_ci_high" in VERDICT_KEYS["return_risk"]
    # every documented return_risk key is a real VerdictInputs field
    fields = {f.name for f in dataclasses.fields(VerdictInputs)}
    for key in VERDICT_KEYS["return_risk"]:
        assert key in fields
