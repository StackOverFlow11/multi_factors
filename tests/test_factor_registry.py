"""Factor registry (D1): the ONE name -> class dispatch, three naming timepoints.

What is locked here:

  * timepoint 1 (registration / class definition): concrete-Factor-only,
    mandatory-spec revalidation reuse, static-spec/class-name agreement,
    duplicate keys and ambiguous prefixes as readable errors;
  * timepoint 2 (build): unknown names, config-name/instance-name mismatch
    (the retired ``_build_factors`` semantics, kept verbatim), instance-name/
    spec-id mismatch, property-spec (window-parameterized) compatibility —
    the spec check runs on the INSTANCE, never forcing a class attribute;
  * timepoint 3 (config parse): ``qt.config.FactorCfg`` resolves enabled
    names in the registry, so ``validate-config`` fails before any run;
  * dispatch parity with the retired chain (same class, same name, per
    family) and ``requirements`` aggregation/dedup;
  * the view x basis pairing forwarding call point (deep wiring is D4).

Collision tests run on FRESH ``FactorRegistry`` instances so the default
registry ``factors.registry.builtin`` populated is never polluted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from factors.base import Factor
from factors.compute.candidates import (
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import FinancialFactor
from factors.compute.intraday_derived import JumpAmountCorrFactor
from factors.compute.momentum import MomentumFactor
from factors.registry import (
    DEFAULT_REGISTRY,
    FactorRegistry,
    build,
    requirements,
    require_legal_pairing,
    resolve,
)
from factors.spec import FactorSpec, PanelField


def _spec_kwargs(factor_id: str) -> dict:
    return dict(
        factor_id=factor_id,
        version="1.0",
        description="A registry test factor.",
        expected_ic_sign=+1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="market_daily"),),
        adjustment="returns_invariant",
        overnight_boundary="none",
    )


class _WindowFactor(Factor):
    """Window-parameterized fixture: property spec (design §6 pit #2)."""

    name: str = "winfac_20"

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self.name = f"winfac_{window}"

    @property
    def spec(self) -> FactorSpec:
        return FactorSpec(**_spec_kwargs(self.name))

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return panel["close"].rename(self.name)


def _window_builder(name, params):
    return _WindowFactor(window=int(params.get("window", 20)))


# --------------------------------------------------------------------------- #
# timepoint 1: registration
# --------------------------------------------------------------------------- #
def test_register_rejects_a_non_factor_class():
    reg = FactorRegistry()
    with pytest.raises(TypeError, match="Factor subclass"):
        reg.register(object, prefix="obj", builder=lambda n, p: object())


def test_register_rejects_an_abstract_family_base():
    class _FamilyBase(Factor):  # no compute -> still abstract, not a factor
        pass

    reg = FactorRegistry()
    with pytest.raises(TypeError, match="abstract"):
        reg.register(_FamilyBase, prefix="family", builder=lambda n, p: None)


def test_register_checks_static_spec_against_the_class_name():
    # timepoint 1 of the five-name consistency: a CLASS-ATTRIBUTE spec must
    # agree with the class-level name at definition/registration time.
    class _Lying(Factor):
        name = "honest_name"
        spec = FactorSpec(**_spec_kwargs("some_other_id"))

        def compute(self, panel):  # pragma: no cover - never built
            return panel["close"].rename(self.name)

    reg = FactorRegistry()
    with pytest.raises(ValueError, match="factor_id"):
        reg.register(_Lying, exact=("honest_name",), builder=lambda n, p: _Lying())


def test_register_accepts_a_consistent_static_spec():
    class _Static(Factor):
        name = "static_fac"
        spec = FactorSpec(**_spec_kwargs("static_fac"))

        def compute(self, panel):
            return panel["close"].rename(self.name)

    reg = FactorRegistry()
    reg.register(_Static, exact=("static_fac",), builder=lambda n, p: _Static())
    assert isinstance(reg.build("static_fac", {}), _Static)


def test_register_needs_exactly_one_of_exact_or_prefix():
    reg = FactorRegistry()
    with pytest.raises(ValueError, match="EITHER exact names or a prefix"):
        reg.register(_WindowFactor, builder=_window_builder)
    with pytest.raises(ValueError, match="EITHER exact names or a prefix"):
        reg.register(
            _WindowFactor, exact=("winfac_20",), prefix="winfac",
            builder=_window_builder,
        )


def test_duplicate_exact_name_is_a_readable_error():
    reg = FactorRegistry()
    reg.register(_WindowFactor, exact=("winfac_20",), builder=_window_builder)
    with pytest.raises(ValueError, match="duplicate factor registration"):
        reg.register(_WindowFactor, exact=("winfac_20",), builder=_window_builder)


def test_duplicate_prefix_is_a_readable_error():
    reg = FactorRegistry()
    reg.register(_WindowFactor, prefix="winfac", builder=_window_builder)
    with pytest.raises(ValueError, match="duplicate factor registration"):
        reg.register(_WindowFactor, prefix="winfac", builder=_window_builder)


def test_mutually_prefixing_prefixes_are_rejected():
    # "win" vs "winfac": dispatch would depend on registration order — the
    # exact ambiguity the guard exists for, in both directions.
    reg = FactorRegistry()
    reg.register(_WindowFactor, prefix="winfac", builder=_window_builder)
    with pytest.raises(ValueError, match="ambiguous factor prefixes"):
        reg.register(_WindowFactor, prefix="win", builder=_window_builder)
    with pytest.raises(ValueError, match="ambiguous factor prefixes"):
        reg.register(_WindowFactor, prefix="winfac_extra", builder=_window_builder)


def test_blank_registration_keys_are_rejected():
    reg = FactorRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        reg.register(_WindowFactor, exact=("",), builder=_window_builder)


# --------------------------------------------------------------------------- #
# timepoint 2: build
# --------------------------------------------------------------------------- #
def test_unknown_factor_name_is_a_readable_error():
    with pytest.raises(ValueError, match="Unknown factor"):
        build("totally_made_up", {})


def test_name_params_mismatch_keeps_the_build_factors_semantics():
    # The retired chain's check, verbatim: config says reversal_5 but the
    # params build reversal_10 -> a silent column mislabel, refused.
    with pytest.raises(ValueError, match="mismatch|resolve"):
        build("reversal_5", {"window": 10})


def test_property_spec_window_factor_builds_and_checks_at_build_time():
    # Pit #2: the spec of a window-parameterized factor only exists on the
    # INSTANCE. The registry must stay compatible — never demand a class
    # attribute — and run the naming checks after construction.
    reg = FactorRegistry()
    reg.register(_WindowFactor, prefix="winfac", builder=_window_builder)
    fac = reg.build("winfac_5", {"window": 5})
    assert isinstance(fac, _WindowFactor) and fac.name == "winfac_5"
    with pytest.raises(ValueError, match="mismatch"):
        reg.build("winfac_5", {})  # default window 20 -> winfac_20 != winfac_5


def test_instance_spec_id_must_match_the_instance_name():
    class _DriftingSpec(Factor):
        name = "drift_1"

        def __init__(self) -> None:
            self.name = "drift_1"

        @property
        def spec(self) -> FactorSpec:
            return FactorSpec(**_spec_kwargs("drift_2"))  # lies about the id

        def compute(self, panel):  # pragma: no cover - never computed
            return panel["close"].rename(self.name)

    reg = FactorRegistry()
    reg.register(_DriftingSpec, exact=("drift_1",), builder=lambda n, p: _DriftingSpec())
    with pytest.raises(ValueError, match="name/spec mismatch"):
        reg.build("drift_1", {})


def test_builder_returning_the_wrong_class_is_refused():
    reg = FactorRegistry()
    reg.register(
        _WindowFactor, prefix="winfac",
        builder=lambda n, p: MomentumFactor(window=20),
    )
    with pytest.raises(TypeError, match="not the registered"):
        reg.build("winfac_20", {})


def test_exact_names_win_over_prefixes():
    # Same precedence as the retired chain (membership tests before
    # startswith): an exact registration shadows a would-match prefix.
    class _Exact(Factor):
        name = "winfac_20"

        @property
        def spec(self) -> FactorSpec:
            return FactorSpec(**_spec_kwargs(self.name))

        def compute(self, panel):
            return panel["close"].rename(self.name)

    reg = FactorRegistry()
    reg.register(_WindowFactor, prefix="winfac", builder=_window_builder)
    reg.register(_Exact, exact=("winfac_20",), builder=lambda n, p: _Exact())
    assert isinstance(reg.build("winfac_20", {}), _Exact)
    assert isinstance(reg.build("winfac_5", {"window": 5}), _WindowFactor)


# --------------------------------------------------------------------------- #
# dispatch parity with the retired if/elif chain (default registry)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, params, cls",
    [
        ("momentum_20", {"window": 20, "price_col": "close"}, MomentumFactor),
        ("reversal_5", {"window": 5}, ReversalFactor),
        ("volatility_20", {"window": 20}, VolatilityFactor),
        ("liquidity_20", {"window": 20}, LiquidityFactor),
        ("overnight_mom_20", {"window": 20}, OvernightMomentumFactor),
        ("value_ep", {}, ValueFactor),
        ("value_bp", {}, ValueFactor),
        ("roe", {}, FinancialFactor),
        ("netprofit_yoy", {}, FinancialFactor),
        ("grossprofit_margin", {}, FinancialFactor),
        # never in the chain, but part of the registry's full surface:
        ("jump_amount_corr_20", {}, JumpAmountCorrFactor),
    ],
)
def test_default_registry_builds_every_family(name, params, cls):
    factor = build(name, params)
    assert isinstance(factor, cls)
    assert factor.name == name
    assert factor.spec.factor_id == name


def test_resolve_is_lookup_only_and_finds_every_family():
    for name in ("momentum_20", "reversal_5", "value_ep", "roe",
                 "valley_price_quantile_20"):
        entry = resolve(name)
        assert name.startswith(entry.key)


# --------------------------------------------------------------------------- #
# requirements aggregation
# --------------------------------------------------------------------------- #
def test_requirements_aggregates_and_deduplicates_in_first_seen_order():
    reqs = requirements(["momentum_20", "volatility_20", "value_ep"])
    # momentum and volatility both need market_daily close -> ONE entry.
    assert [(r.field, r.source) for r in reqs] == [
        ("close", "market_daily"),
        ("pe", "daily_basic"),
    ]


def test_requirements_honours_per_name_params():
    reqs = requirements(
        ["momentum_5"], params_by_name={"momentum_5": {"window": 5}}
    )
    assert [(r.field, r.source) for r in reqs] == [("close", "market_daily")]
    # ... and without the params the naming check fires exactly like build():
    with pytest.raises(ValueError, match="mismatch"):
        requirements(["momentum_5"])


# --------------------------------------------------------------------------- #
# view x basis pairing forwarding (deep wiring is D4; D1 = callable + tested)
# --------------------------------------------------------------------------- #
def test_pairing_forwarding_accepts_the_two_legal_pairs():
    view, basis = require_legal_pairing("decision", "exec_to_exec")
    assert (view.value, basis.value) == ("decision", "exec_to_exec")
    view, basis = require_legal_pairing("close", "close_to_close")
    assert (view.value, basis.value) == ("close", "close_to_close")


def test_pairing_forwarding_rejects_the_cross_pairs_readably():
    with pytest.raises(ValueError, match="illegal view/basis pairing"):
        require_legal_pairing("close", "exec_to_exec")
    with pytest.raises(ValueError, match="illegal view/basis pairing"):
        require_legal_pairing("decision", "close_to_close")


# --------------------------------------------------------------------------- #
# timepoint 3: config parse (qt side calls the registry)
# --------------------------------------------------------------------------- #
def test_factor_cfg_rejects_an_unknown_enabled_name_at_parse_time():
    from pydantic import ValidationError

    from qt.config import FactorCfg

    with pytest.raises(ValidationError, match="Unknown factor"):
        FactorCfg(name="totally_made_up")


def test_factor_cfg_accepts_known_names_and_skips_disabled_entries():
    from qt.config import FactorCfg

    FactorCfg(name="momentum_20")
    FactorCfg(name="value_ep", params={})
    # Disabled entries are not dispatched by _build_factors, so they are not
    # name-checked either (mirrored behavior, locked here on purpose).
    FactorCfg(name="totally_made_up", enabled=False)


def test_default_registry_is_the_module_singleton_the_helpers_use():
    assert resolve("momentum_20") is DEFAULT_REGISTRY.resolve("momentum_20")
