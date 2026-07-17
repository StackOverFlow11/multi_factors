"""``EvalConfig``: the PER-RUN half of the factor-evaluation contract.

Where :class:`~factors.spec.FactorSpec` describes the factor (and never
changes), ``EvalConfig`` describes ONE evaluation of it: the window, the
universe, what was neutralized away — and the honesty flags the project has
learned to demand (``is_exploratory`` / ``post_hoc_selected`` / ``tuned`` /
``n_factors_screened``).

Together they are the complete provenance an evaluator requires (design doc
``tmp/design/factor_eval_contract_v0.1.md`` §3). The validators here are
enforcement layer #1: a run that cannot state its honesty flags coherently
cannot be configured at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from analytics.eval.verdict import VerdictThresholds

# Tolerance for spotting the base (multiplier 1.0) cost scenario among floats.
_BASE_COST = 1.0
_COST_TOL = 1e-9


@dataclass(frozen=True)
class EvalConfig:
    """Parameters + honesty declarations of ONE factor evaluation (immutable).

    Attributes
    ----------
    universe : the universe identifier (e.g. an index code) being evaluated.
    universe_is_pit : whether the constituents are point-in-time. Declaring
        False is allowed but must be EXPLICIT — it is a survivorship caveat.
    start, end : the evaluation window.
    is_exploratory : an exploratory study, not a return claim.
    post_hoc_selected : was this factor picked AFTER seeing results on (some of)
        this data? If so it cannot be a confirmation — see the validator.
    rebalance : rebalance frequency label. Defaults to "daily" (user ruling: the
        project's default rebalance frequency). NOTE this label is not
        decoration — an evaluator resolves the forward-return horizon and the
        annualization on the panel's OWN grid and checks the supplied spacing
        against this label, and the verdict's sample gate reads a CALENDAR span
        and an EFFECTIVE sample size precisely because a raw count means
        something different at each frequency (design §6, v0.3).
    n_quantiles : number of factor buckets (>= 2).
    long_short : (top, bottom) bucket labels. Defaults to (n_quantiles, 1). NOTE
        this is a SYNTHETIC long-only leg difference, not a dollar-neutral
        executed portfolio — the report must say so (P-I5d/P-I5e lesson).
    cost_scenarios : fee multipliers. MUST contain 1.0, the base anchor every
        other scenario is read against ("the same trades at k x the fee").
    limit_feasibility : apply the raw stk_limit up/down execution gate (I5b).
    capacity_notional, max_participation_rate : the I5f capacity diagnostic.
    oos_split : train/test split date; None means the report must state
        explicitly that NO out-of-sample split was done.
    independent_cells : cells declared genuinely independent of factor
        screening. Independence is a HUMAN declaration — the machine cannot know
        which data took part in screening. Overlap with the screening window
        cannot be checked here (this object does not know it), so an evaluator
        that can check it should; otherwise the report carries the caveat.
    winsorize, standardize, neutralization, industry_level : what style
        exposure was stripped before evaluating.
    tuned : whether any parameter was tuned. Default False = "not tuned".
    n_factors_screened : multiple-testing background (how many factors were
        looked at to find this one).
    data_snapshot_id : data/cache version, for reproducibility.
    success_criteria : the PRE-REGISTERED verdict bar (design §6, v0.6). A frozen
        :class:`~analytics.eval.verdict.VerdictThresholds` declared HERE, before
        ``evaluate`` runs, IS pre-registered by construction: you cannot tune the
        bar after seeing the result without producing a NEW report that visibly
        declares different criteria (the report stamps ``criteria_source``). None
        means the documented global default is used. A supplied object must pass
        exactly the same ``VerdictThresholds.__post_init__`` validation — it is not
        weakened for being per-run.
    """

    universe: str
    universe_is_pit: bool
    start: str
    end: str
    is_exploratory: bool
    post_hoc_selected: bool
    rebalance: str = "daily"
    n_quantiles: int = 5
    long_short: tuple[int, int] | None = None
    cost_scenarios: tuple[float, ...] = (1.0, 2.0, 4.0)
    limit_feasibility: bool = True
    capacity_notional: float | None = None
    max_participation_rate: float = 0.05
    oos_split: str | None = None
    independent_cells: tuple[object, ...] = ()
    winsorize: str | None = "mad"
    standardize: str | None = "zscore"
    neutralization: tuple[str, ...] = ("industry", "size")
    industry_level: str = "L1"
    tuned: bool = False
    n_factors_screened: int | None = None
    data_snapshot_id: str | None = None
    success_criteria: VerdictThresholds | None = None

    def __post_init__(self) -> None:
        self._check_window()
        self._check_honesty()
        self._check_quantiles()
        self._check_costs()
        self._check_capacity()
        self._check_declared_sequences()
        self._check_success_criteria()

    # -- validators (enforcement layer #1) --------------------------------

    def _check_window(self) -> None:
        for field_name in ("universe", "start", "end", "rebalance"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"EvalConfig.{field_name} must be a non-empty string; got "
                    f"{value!r}."
                )
        if not isinstance(self.universe_is_pit, bool):
            raise ValueError(
                f"EvalConfig.universe_is_pit must be a bool (a non-PIT universe "
                f"must be admitted explicitly — survivorship); got "
                f"{self.universe_is_pit!r}."
            )

    def _check_honesty(self) -> None:
        for field_name in ("is_exploratory", "post_hoc_selected", "tuned"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(
                    f"EvalConfig.{field_name} must be a bool; got {value!r}."
                )
        if self.post_hoc_selected and not self.is_exploratory:
            raise ValueError(
                "EvalConfig: post_hoc_selected=True requires is_exploratory=True. "
                "A factor chosen after seeing results on this data cannot be "
                "reported as a confirmation (P3-6 lesson)."
            )
        screened = self.n_factors_screened
        if screened is not None and (
            isinstance(screened, bool) or not isinstance(screened, int) or screened < 1
        ):
            raise ValueError(
                f"EvalConfig.n_factors_screened must be None or a positive int "
                f"(the multiple-testing background); got {screened!r}."
            )

    def _check_quantiles(self) -> None:
        n = self.n_quantiles
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError(
                f"EvalConfig.n_quantiles must be an int >= 2; got {n!r}."
            )
        if self.long_short is None:
            # default: top bucket vs bottom bucket
            object.__setattr__(self, "long_short", (n, 1))
            return
        pair = tuple(self.long_short)
        if len(pair) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) for x in pair
        ):
            raise ValueError(
                f"EvalConfig.long_short must be a (top, bottom) pair of ints; got "
                f"{self.long_short!r}."
            )
        top, bottom = pair
        if not (1 <= top <= n) or not (1 <= bottom <= n):
            raise ValueError(
                f"EvalConfig.long_short=({top}, {bottom}) must name buckets within "
                f"1..{n}."
            )
        if top <= bottom:
            # The legs stay in their natural (higher bucket, lower bucket)
            # orientation so the hypothesis is applied in exactly ONE place: the
            # verdict, via expected_ic_sign. A pre-flipped pair here would
            # double-flip a -1 factor (e.g. low-vol) into a false Reject.
            raise ValueError(
                f"EvalConfig.long_short=({top}, {bottom}) must have top > bottom: "
                f"the leg difference is always (higher bucket - lower bucket) and "
                f"the DIRECTION comes from FactorSpec.expected_ic_sign, never from "
                f"flipping the legs here."
            )
        object.__setattr__(self, "long_short", pair)

    def _check_costs(self) -> None:
        scenarios = self.cost_scenarios
        if isinstance(scenarios, str) or not isinstance(scenarios, Sequence):
            raise ValueError(
                f"EvalConfig.cost_scenarios must be a sequence of fee multipliers; "
                f"got {scenarios!r}."
            )
        # Guarded BEFORE the float() coercion, which is what makes this the worst
        # of the bool holes: float(True) is 1.0, so a stray True would not merely
        # sneak in — it would silently satisfy the mandatory 1.0 base anchor below
        # and the run would report "base cost" scenarios that nobody declared.
        bad_bools = [c for c in scenarios if isinstance(c, bool)]
        if bad_bools:
            raise ValueError(
                f"EvalConfig.cost_scenarios entries must be numeric fee "
                f"multipliers, never bool: float(True) is 1.0, so a stray True "
                f"would masquerade as the mandatory 1.0 base anchor. Got "
                f"{tuple(scenarios)!r}."
            )
        normalized = tuple(float(c) for c in scenarios)
        bad = [c for c in normalized if c <= 0]
        if bad:
            raise ValueError(
                f"EvalConfig.cost_scenarios multipliers must be positive; got {bad}."
            )
        if not any(abs(c - _BASE_COST) < _COST_TOL for c in normalized):
            raise ValueError(
                f"EvalConfig.cost_scenarios must contain the base anchor 1.0 — every "
                f"other scenario is only readable as 'the same trades at k x the "
                f"fee' (project rule); got {normalized}."
            )
        object.__setattr__(self, "cost_scenarios", normalized)

    def _check_capacity(self) -> None:
        # ``bool`` is an int subclass, so True sails through a bare numeric check
        # and lands in a declared-float field (and in the exported record) — the
        # same gotcha as ``expected_ic_sign=1.0``. Guarded explicitly, as in every
        # other numeric validator here and in FactorSpec.
        notional = self.capacity_notional
        if notional is not None and (
            isinstance(notional, bool)
            or not isinstance(notional, (int, float))
            or notional <= 0
        ):
            raise ValueError(
                f"EvalConfig.capacity_notional must be None or a positive number; "
                f"got {notional!r}."
            )
        rate = self.max_participation_rate
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not (0 < rate <= 1)
        ):
            raise ValueError(
                f"EvalConfig.max_participation_rate must be in (0, 1]; got {rate!r}."
            )
        if not isinstance(self.limit_feasibility, bool):
            raise ValueError(
                f"EvalConfig.limit_feasibility must be a bool; got "
                f"{self.limit_feasibility!r}."
            )

    def _check_success_criteria(self) -> None:
        """The pre-registered bar, if supplied, must be a real VerdictThresholds.

        Its own ``__post_init__`` has already validated type/range/NaN/inf/the
        1.0-exclusive bounds — a per-run object is NOT weakened for being per-run.
        This only rejects the wrong TYPE (e.g. a bare dict) rather than silently
        ignoring it and falling back to the default bar, which would let a caller
        THINK they pre-registered a bar that never took effect.
        """
        criteria = self.success_criteria
        if criteria is not None and not isinstance(criteria, VerdictThresholds):
            raise ValueError(
                f"EvalConfig.success_criteria must be None or a VerdictThresholds "
                f"(the pre-registered verdict bar, already self-validating); got "
                f"{type(criteria).__name__}. A dict/other would be silently ignored "
                f"and the run would fall back to the default bar — a pre-registration "
                f"that never took effect."
            )

    def _check_declared_sequences(self) -> None:
        """Coerce the remaining declared sequences to tuples, like the others.

        ``cost_scenarios`` / ``long_short`` are already normalized above. Without
        the same treatment here a caller handing in a LIST keeps a live reference
        into a ``frozen=True`` config (mutate the list -> mutate the config) and
        ``hash(cfg)`` raises — an immutable provenance record that is neither.
        """
        for field_name in ("independent_cells", "neutralization"):
            value = getattr(self, field_name)
            # A bare string would silently become a tuple of single characters.
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ValueError(
                    f"EvalConfig.{field_name} must be a sequence (not a bare "
                    f"string); got {value!r}."
                )
            object.__setattr__(self, field_name, tuple(value))


__all__ = ["EvalConfig"]
