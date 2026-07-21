"""Verdict rules: three independent axis-verdicts + a derived deployment label.

Design doc ``tmp/design/factor_eval_contract_v0.1.md`` §6 (v0.5). The RULE
STRUCTURE is fixed (a factual check — no eyeballing a factor into "promising");
the THRESHOLDS are config (:class:`VerdictThresholds`).

WHY THREE AXES (v0.5 — the multi-factor point). Judging a factor in ISOLATION is
the wrong unit for a cross-sectional MULTI-FACTOR book. A single scalar verdict
answered "is there a signal?" but never "does it add anything the book does not
already have?" — so a factor that merely re-expresses value_ep would sail through
on its own raw IC. The verdict is now split into three questions that are
answered INDEPENDENTLY and reported side by side:

    Predictive   is there a real, OUT-OF-SAMPLE predictive signal?
    Incremental  does it add alpha BEYOND the known factor set? (the new axis)
    Tradable     can the signal actually be harvested?

Each axis takes one of four states: ``PASS`` / ``FAIL`` / ``INSUFFICIENT_DATA`` /
``NOT_ASSESSED``. A single DEPLOYMENT LABEL (Adopt / Watch / Reject /
INSUFFICIENT-DATA) is DERIVED from the three (see :func:`_derive_deployment`),
preserving the old asymmetric gate + exploratory cap that the scalar verdict won.

THE DEPLOYMENT DERIVATION (design §6 table)

    any axis FAIL                         -> REJECT  (FAIL is evaluated FIRST, so
                                                      it BYPASSES the sample gate:
                                                      a thin-sample sign-flip still
                                                      Rejects)
    no FAIL, all three PASS               -> ADOPT   (capped to WATCH if the run
                                                      declares itself exploratory)
    no FAIL, >=1 PASS, rest unresolved    -> WATCH   (reason names the unresolved
                                                      axes)
    no FAIL, no PASS                      -> INSUFFICIENT-DATA

    ``default run -> at most WATCH`` is INTENDED: with no known_factors the
    Incremental axis is NOT_ASSESSED and with no execution facts the Tradable axis
    is NOT_ASSESSED, so a default run can never have all three PASS and tops out
    at WATCH. That is the honest reading, not a defect — a factor evaluated with
    no book and no execution evidence has not earned an Adopt.

THE ASYMMETRIC GATE — FAIL BYPASSES THE SAMPLE GATE (v0.4, kept)
    The sample gate exists to prevent OVERCLAIMING, so it guards the POSITIVE
    claims (a PASS on Predictive/Incremental) and nothing else. A FAIL is a
    NEGATIVE FINDING: thin data plus a VISIBLE failure — an out-of-sample sign
    flip, an untradable spread, a loss at every cost level — is already enough to
    say "do not trade this". Each axis therefore decides its FAIL BEFORE consulting
    the sample gate (Predictive and Tradable have KNOWN-fact FAILs that never touch
    the gate; only the Incremental FAIL, which is a statistical "it adds nothing"
    claim, waits for a sufficient sample). The costs are asymmetric — a false
    Reject skips one possibly-good factor; a false Adopt trades a bad one with real
    money — so the rule is too: quick to reject, slow to adopt.

    WHAT THIS FIXES (concretely): a regime-flipping factor — the project's
    signature failure mode (I5e, P3-3, P3-4) — has an IC series that is a STEP
    FUNCTION, so N_eff collapses to ~3 out of ~300 raw periods. That N_eff is
    CORRECT (two regimes really are about two observations), but a gate-first order
    would report INSUFFICIENT-DATA instead of Reject about precisely the failure
    the project most wants flagged. The flip is a fact the run MEASURED; the gate
    has no business suppressing it. Predictive FAIL runs first, so it Rejects.

UNKNOWN NEVER CONVICTS. Every FAIL requires a KNOWN fact — ``tradable is False``
(never a falsy ``None``), an explicit flip/reversal flag, a non-empty set of
FINITE all-non-positive spreads, or a FINITE orthogonalized ICIR that measurably
fails to clear on a sufficient sample. A ``None`` / ``NaN`` / absent fact yields
NO FAIL: it falls through to INSUFFICIENT_DATA (Predictive/Incremental) or
NOT_ASSESSED (Tradable). And unknown never passes the gate as a PASS either
(same conservatism, opposite direction).

THE SAMPLE GATE IS THREE PARTS (v0.3, kept). A RAW COUNT IS NOT A SAMPLE SIZE.
Consecutive daily IC observations are heavily autocorrelated: 500 raw points can
carry only a few dozen independent ones. The gate (raw floor + N_eff + calendar
span) governs whether Predictive/Incremental may be PASS vs INSUFFICIENT_DATA —
same thresholds, now applied PER AXIS.

      raw floor (min_rebalances)   a LOW precondition, NOT the sample gate: below
                                   a handful of points the IC autocorrelation —
                                   and hence N_eff itself — is not estimable.
      (A) min_effective_samples    N_eff = N / (1 + 2*sum_k rho_k) over the IC
                                   series. THE part that does the work.
      (B) min_span_days            calendar span of the IC series, which is
                                   frequency-independent.

THE EXPLORATORY CAP (v0.2, kept) caps the DEPLOYMENT LABEL, not an axis:
``EvalConfig.is_exploratory=True`` turns an all-PASS ADOPT into WATCH. Adopt IS a
performance claim, and an exploratory run declares it is not making one. Because
EvalConfig §4 forces ``post_hoc_selected=True`` => ``is_exploratory=True``, this
also closes the post-hoc -> Adopt hole. The cap only downgrades the LABEL; the
axes and their per-axis evidence are reported unchanged.

TWO CONVENTIONS THAT MAKE THE RULES SIGN-SAFE
    1. Section payloads carry RAW facts: ``monotonicity_spearman`` is the plain
       Spearman of bucket index vs bucket mean return, and
       ``net_long_short_by_cost`` is the plain (top - bottom) leg difference. The
       HYPOTHESIS (``expected_ic_sign``) is applied in exactly ONE place — here —
       so a low-vol factor (sign -1) reads in its own direction.

       APPLYING THE SIGN TO A NET SPREAD IS NOT A MULTIPLICATION (v0.8). The
       hypothesis decides WHICH LEG IS LONG; the trading cost is a drag in EITHER
       direction. So the aligned spread flips the legs and THEN subtracts cost —
       ``sign * gross - cost`` — never ``sign * net``, which at sign -1 expands to
       ``-gross + cost`` and hands the factor its own costs back as profit. This
       is why ``gross_long_short_mean`` is an input: the cost is recovered as
       ``gross - net``, and an unknown gross makes the ALIGNED spread unknown
       (which, per the rule above, then neither convicts nor passes).
    2. ``ic_win_rate`` and the orthogonalized ICs are hypothesis-relative by the
       same rule: direction is applied by multiplying with ``expected_ic_sign``.

PRE-REGISTERED CRITERIA + CONFIDENCE INTERVALS (v0.6, change #3). The verdict
thresholds are pre-registered per run: ``EvalConfig.success_criteria`` (a frozen
``VerdictThresholds`` constructed BEFORE ``evaluate`` runs) is the declared bar;
when absent the documented global default is used, and the report stamps which
(``criteria_source``). And the magnitude PASS test is now on the LOWER CONFIDENCE
BOUND, not the naked point: the standard layer attaches an N_eff-based CI to every
gated estimate (ICIR, incremental ICIR), and :func:`_clears_magnitude` requires
the aligned lower bound to clear the bar. A thin/noisy estimate has a wide CI and a
low lower bound, so it fails the positive claim automatically. This does NOT
duplicate the sample gate — the gate asks "can we estimate a CI at all?"
(INSUFFICIENT below it); the lower-CI test asks "is it convincingly above the bar?"
(PASS above it). FAIL stays POINT-based and known-fact-only; a NaN CI bound never
convicts.

THE MONOTONICITY DIRECTION IS THREE-VALUED (v0.9). v0.8 moved the monotonicity gate
onto a per-date rank statistic, which was the right KIND of statistic — and then the
eleven-factor re-run showed that statistic is heavily attenuated by daily noise: an
empirically perfect quantile ladder scores 0.045-0.106 on it, while the direction
gate still sat at a bare 0.0 with no dispersion estimate anywhere. Two factors
landed 0.021 apart across that boundary. So the direction is now decided by the
N_eff-based CI of the per-date series (:func:`_monotonicity_direction`), and the
answer can be UNKNOWN — CI straddles 0 — as well as holds/contradicted.

    UNKNOWN NEVER RESCUES. An unknown direction is checked AFTER every other
    predictive criterion: if ICIR / NW-t / win rate / OOS consistency fail, the axis
    still FAILs and the unknown is merely disclosed. Only when everything else has
    cleared and the direction ALONE cannot be asserted does the axis become
    INSUFFICIENT_DATA. "We could not tell" withholds a PASS; it never withholds a
    FAIL that was independently earned. This is the same asymmetry as UNKNOWN NEVER
    CONVICTS above, read from the other side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# -- deployment labels (the DERIVED, user-facing verdict) ----------------

INSUFFICIENT_DATA = "INSUFFICIENT-DATA"
ADOPT = "Adopt"
WATCH = "Watch"
REJECT = "Reject"

VERDICTS: tuple[str, ...] = (INSUFFICIENT_DATA, ADOPT, WATCH, REJECT)

# -- axis-verdict states (each of the three axes takes one) --------------
#
# Deliberately their OWN vocabulary, distinct from the deployment labels above:
# an axis answers PASS/FAIL/INSUFFICIENT_DATA/NOT_ASSESSED, and the deployment
# label is derived from the three. (The axis "INSUFFICIENT_DATA" uses an
# underscore; the deployment "INSUFFICIENT-DATA" a hyphen — they are related
# ideas at different scopes, not the same value.)

AXIS_PASS = "PASS"
AXIS_FAIL = "FAIL"
AXIS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
AXIS_NOT_ASSESSED = "NOT_ASSESSED"

AXIS_VERDICTS: tuple[str, ...] = (
    AXIS_PASS,
    AXIS_FAIL,
    AXIS_INSUFFICIENT_DATA,
    AXIS_NOT_ASSESSED,
)

#: the three axes, in report order.
AXIS_NAMES: tuple[str, ...] = ("predictive", "incremental", "tradable")


# -- threshold domains ---------------------------------------------------
#
# These thresholds decide the axis PASS/FAIL and so the VERDICT — the contract's
# central claim. They are validated for TYPE and RANGE like everything else here.
#
# ⚠️ Structural validation only: it checks a threshold is a sane number, NOT that
# it is the RIGHT number. The defaults remain UNVALIDATED-BY-DATA and still need
# one round of empirical calibration against real runs (design §11).

#: thresholds compared against a MAGNITUDE: finite and >= 0, no natural upper bound.
_MAGNITUDE_THRESHOLDS: tuple[str, ...] = (
    "min_abs_icir", "min_abs_nw_t", "min_incremental_abs_icir",
)

#: thresholds living inside a closed metric domain: lower bound INCLUSIVE, upper
#: bound EXCLUSIVE. Both metrics max out at 1.0 and both rules compare with a
#: strict '>', so a threshold AT 1.0 could never be exceeded and would silently
#: make the rule it gates unsatisfiable — the same class of silent breakage as
#: NaN/inf, it just fails strict instead of permissive.
_BOUNDED_THRESHOLDS: dict[str, tuple[float, float]] = {
    "min_ic_win_rate": (0.0, 1.0),
    "min_monotonicity_spearman": (0.0, 1.0),
}


def _check_real(name: str, value: object) -> None:
    """Reject non-numbers / bools / NaN / inf with a READABLE ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"VerdictThresholds.{name} must be a real number (never a bool: True "
            f"is an int subclass and would silently become 1); got {value!r}."
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"VerdictThresholds.{name} must be finite; got {value!r}. NaN/inf "
            f"would SILENTLY disable the rule it gates — every '>' comparison "
            f"against NaN is False, and nothing exceeds +inf."
        )


@dataclass(frozen=True)
class VerdictThresholds:
    """Thresholds behind the (fixed) axis rules.

    ⚠️ UNVALIDATED DEFAULTS — plausible starting points, NOT calibrated numbers.
    Design §11 lists their calibration as an open item. Pass a tuned instance to
    override; never silently re-interpret a threshold as "validated" because a
    report printed it.
    """

    # -- the three-part sample gate (design §6, v0.3) ---------------------
    #
    # RAW FLOOR — deliberately LOW, and deliberately NOT the sample gate any more.
    # Its only job is to refuse a sample too small to estimate the IC
    # autocorrelation from at all. Parts A and B do the real gating.
    min_rebalances: int = 12
    # (A) effective sample size N_eff = N / (1 + 2*sum_k rho_k) of the IC series.
    # Note N_eff <= N always (clamped), so this also implies "at least 24 settled
    # rebalances".
    min_effective_samples: float = 24.0
    # (B) calendar span of the IC series, in days — frequency-independent. 365 is
    # a FLOOR, not sufficiency (P3-2's single-year outperformance died in
    # P3-3/P3-4).
    min_span_days: int = 365
    # |mean(IC)/std(IC)| — magnitude only; direction is checked out-of-sample.
    min_abs_icir: float = 0.30
    # (v0.7) Incremental axis bar: |orthogonalized ICIR| of the residual AFTER the
    # book. SEPARATE from min_abs_icir because orthogonalization removes variance,
    # so the residual ICIR lives on a structurally SMALLER scale than the raw ICIR
    # — reusing the raw 0.30 bar was a category error (it silently failed genuinely
    # partial-incremental factors). Default ~= half the raw bar; UNVALIDATED (§11).
    min_incremental_abs_icir: float = 0.15
    # share of periods whose IC carries the EXPECTED sign (0.5 = a coin flip).
    min_ic_win_rate: float = 0.55
    # |Newey-West corrected t| of the mean IC (~5% two-sided).
    min_abs_nw_t: float = 2.0
    # hypothesis-aligned Spearman of bucket index vs bucket mean return.
    # (v0.7) DIRECTION gate, not strength: default 0.0 = "aligned monotonicity must
    # be strictly > 0" (buckets ordered the RIGHT way / NOT reversed), NOT "strongly
    # monotone". A 0.8 strength bar rejected tail-concentrated real factors whose
    # power sits in one extreme (e.g. jump-amount-corr: ICIR -0.40, NW-t -14.8, yet
    # Q3-humped). Raise it to demand strength. UNVALIDATED (§11).
    # (v0.9) The bar is now compared against the per-date monotonicity's aligned CI
    # LOWER bound, not its point — so it is the bar a direction claim must be
    # CONVINCINGLY above. The FAIL side stays pinned at 0 (reversal is a fact about
    # the sign, not about how strong this threshold was set); see
    # _monotonicity_direction.
    min_monotonicity_spearman: float = 0.0

    def __post_init__(self) -> None:
        n = self.min_rebalances
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(
                f"VerdictThresholds.min_rebalances must be an int >= 1 (the raw "
                f"floor below which the IC autocorrelation is not estimable); got "
                f"{n!r}."
            )
        eff = self.min_effective_samples
        _check_real("min_effective_samples", eff)
        if eff < 1:
            raise ValueError(
                f"VerdictThresholds.min_effective_samples must be >= 1: N_eff is "
                f"clamped to [1, N], so a threshold below 1 can never fail and "
                f"would SILENTLY disable gate part A. Got {eff!r}."
            )
        span = self.min_span_days
        if isinstance(span, bool) or not isinstance(span, int) or span < 0:
            raise ValueError(
                f"VerdictThresholds.min_span_days must be an int >= 0 calendar days "
                f"(0 disables gate part B explicitly); got {span!r}."
            )
        for name in _MAGNITUDE_THRESHOLDS:
            value = getattr(self, name)
            _check_real(name, value)
            if value < 0:
                raise ValueError(
                    f"VerdictThresholds.{name} is compared against a magnitude and "
                    f"must be non-negative; got {value!r}."
                )
        for name, (low, high) in _BOUNDED_THRESHOLDS.items():
            value = getattr(self, name)
            _check_real(name, value)
            if not low <= value < high:
                raise ValueError(
                    f"VerdictThresholds.{name} must be in [{low}, {high}) — its "
                    f"metric never exceeds {high} and the rule compares with a "
                    f"strict '>', so a threshold of {high} or above can never be "
                    f"exceeded and would SILENTLY make the rule it gates "
                    f"unsatisfiable. Got {value!r}."
                )


@dataclass(frozen=True)
class VerdictInputs:
    """The exact facts the axis rules read (extracted from section payloads).

    Every field is either a hard fact or an explicit unknown (NaN / None /
    False), so the rules never guess.

    ``is_exploratory`` is the one field that is NOT a measurement: it is the
    run's own DECLARATION, read straight off :class:`~analytics.eval.config.
    EvalConfig` rather than out of a section payload. Sourcing the cap from a
    section payload would let an evaluator escape it by omitting the key.
    """

    expected_ic_sign: int              # from FactorSpec: +1 / -1
    settled_rebalances: int = 0        # data_coverage (the RAW count)
    # EvalConfig declaration (NOT a metric): True caps the deployment label at Watch.
    is_exploratory: bool = False
    #: data_coverage: N_eff of the IC series, CLAMPED to [1, N]. NaN = UNKNOWN,
    #: which FAILS the gate (never passes it).
    effective_samples: float = float("nan")
    #: data_coverage: calendar days spanned by the IC series (gate part B).
    span_days: float = float("nan")
    # -- Predictive axis facts -------------------------------------------
    ic_ir: float = float("nan")        # predictive_power (POINT estimate)
    #: 95% CI bounds of ic_ir, computed with N_eff (design §6, v0.6). RAW (the
    #: verdict applies expected_ic_sign); the Predictive PASS gates on the LOWER
    #: bound in the expected direction, NOT the naked point. NaN = UNKNOWN -> the
    #: axis cannot PASS, and a NaN bound never manufactures a FAIL.
    ic_ir_ci_low: float = float("nan")
    ic_ir_ci_high: float = float("nan")
    ic_win_rate: float = float("nan")  # predictive_power (hypothesis-relative)
    ic_nw_t: float = float("nan")      # predictive_power
    #: return_risk: POOLED Spearman(bucket index, cross-date MEAN bucket return),
    #: RAW. Unbounded and magnitude-sensitive; as of v0.8 it is a REPORTED figure
    #: and only the FALLBACK for the gate below.
    monotonicity_spearman: float = float("nan")
    #: return_risk: the v0.8 GATED monotonicity — the mean over dates of each
    #: date's own Spearman across the buckets, each capped in [-1, 1] before
    #: averaging (structurally parallel to the rank IC). RAW. NaN = UNKNOWN, which
    #: falls back to ``monotonicity_spearman`` with the substitution DISCLOSED in
    #: the axis reasons, so an IR built before v0.8 stays judgeable.
    #: (v0.9) This POINT is now only the FALLBACK for the direction gate; the CI
    #: below is what decides when it is available.
    monotonicity_spearman_by_date: float = float("nan")
    #: return_risk: 95% CI bounds of ``monotonicity_spearman_by_date``, computed
    #: with the per-date series' OWN N_eff (design §6, v0.9 — the same ``mean_ci``
    #: the ICIR uses). RAW (the verdict applies ``expected_ic_sign``). This is what
    #: the monotonicity DIRECTION gate reads: the bare point carries no dispersion
    #: estimate, and a cross-date mean of per-date rank correlations is so
    #: attenuated by daily noise that "point > 0.0" was a coin flip dressed as a
    #: criterion. NaN = UNKNOWN -> the gate falls back to the point (v0.8 behaviour)
    #: with the fallback DISCLOSED.
    monotonicity_spearman_by_date_ci_low: float = float("nan")
    monotonicity_spearman_by_date_ci_high: float = float("nan")
    oos_available: bool = False        # oos_generalization
    oos_sign_consistent: bool = False  # expected sign holds in BOTH subperiods
    oos_sign_flipped: bool = False     # explicit flip -> Predictive FAIL
    oos_monotonicity_reversed: bool = False  # independent-cell reversal (I5e)
    # -- Incremental axis facts (v0.5, NEW) ------------------------------
    #: purity: was a known-factor BOOK supplied at all? False (the default) means
    #: the Incremental axis is NOT_ASSESSED. This is a declaration of what was
    #: SUPPLIED, distinct from whether the orthogonalized IC could be computed.
    known_factors_supplied: bool = False
    #: purity: mean(orthIC)/std(orthIC) of the factor's IC AFTER residualizing it
    #: on the WHOLE known-factor book per date (mirror of ``ic_ir``, but on the
    #: residual). NaN = UNKNOWN -> Incremental INSUFFICIENT_DATA, never a FAIL.
    incremental_ic_ir: float = float("nan")
    #: purity: mean of the same orthogonalized IC series (direction + magnitude),
    #: reported alongside; the axis reads direction off ``incremental_ic_ir``.
    incremental_ic_mean: float = float("nan")
    #: 95% CI bounds of incremental_ic_ir, computed with the ORTHOGONALIZED IC
    #: series' OWN N_eff. Same lower-bound PASS rule as ic_ir_ci_* (design §6,
    #: v0.6). NaN never convicts.
    incremental_ic_ir_ci_low: float = float("nan")
    incremental_ic_ir_ci_high: float = float("nan")
    # -- return_risk / execution facts -----------------------------------
    #: return_risk: {cost multiplier: net (top - bottom) return}, RAW sign.
    net_long_short_by_cost: tuple[tuple[float, float], ...] = ()
    #: return_risk: mean GROSS (top - bottom) leg difference, RAW sign, before any
    #: cost. Required to align a spread by hypothesis WITHOUT adding the cost back
    #: (see :func:`_aligned_net`): the per-scenario cost is recovered as
    #: ``gross - net``. NaN = UNKNOWN -> a sign=-1 aligned spread is UNKNOWN too.
    gross_long_short_mean: float = float("nan")
    tradable: bool | None = None       # execution_capacity; None = not supplied
    capacity_sufficient: bool | None = None  # None = not supplied


@dataclass(frozen=True)
class AxisVerdict:
    """One axis's verdict (PASS / FAIL / INSUFFICIENT_DATA / NOT_ASSESSED) + why."""

    verdict: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.verdict not in AXIS_VERDICTS:
            raise ValueError(
                f"axis verdict must be one of {AXIS_VERDICTS}; got {self.verdict!r}."
            )
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True)
class VerdictResult:
    """The three axis-verdicts + the DERIVED deployment label (with its reasons).

    ``verdict`` / ``reasons`` are the deployment-level answer (what a caller reads
    first); ``predictive`` / ``incremental`` / ``tradable`` are the three
    independent axes it was derived from. The axes default to NOT_ASSESSED so a
    hand-built result (e.g. a placeholder verdict on an incomplete report) stays
    constructible positionally.
    """

    verdict: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    predictive: AxisVerdict = field(
        default_factory=lambda: AxisVerdict(AXIS_NOT_ASSESSED)
    )
    incremental: AxisVerdict = field(
        default_factory=lambda: AxisVerdict(AXIS_NOT_ASSESSED)
    )
    tradable: AxisVerdict = field(
        default_factory=lambda: AxisVerdict(AXIS_NOT_ASSESSED)
    )

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {VERDICTS}; got {self.verdict!r}."
            )
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def axes(self) -> dict[str, AxisVerdict]:
        """The three axis verdicts keyed by name, in report order."""
        return {
            "predictive": self.predictive,
            "incremental": self.incremental,
            "tradable": self.tradable,
        }


# -- the axis magnitude/level comparisons (design §6, v0.6: gate on the CI) ------
#
# CHANGE #3 (pre-registered criteria + confidence intervals) IS this change. Every
# axis PASS magnitude comparison funnels through these two helpers, so the CI logic
# is localized here instead of scattered across the three axes.
#
# TWO-LAYER RELATIONSHIP WITH THE SAMPLE GATE — they are DIFFERENT checks, do NOT
# read one as subsuming the other:
#   * the raw-floor / N_eff / span sample gate is the "can we even ESTIMATE a CI?"
#     precondition. Below it the axis is INSUFFICIENT_DATA and the CI is not
#     consulted at all.
#   * the lower-CI comparison here is the "is the estimate CONVINCINGLY above the
#     pre-registered bar?" PASS test, ABOVE the gate. The gate can pass (enough
#     data) while the lower CI still fails — a noisy estimate whose interval
#     straddles the bar even at sufficient N_eff. That run is NOT a PASS, but it is
#     not gated out either; it reads as "promising, unconfirmed".


def _clears_magnitude(
    point_estimate: float, threshold: float, *, lower: float | None = None
) -> bool:
    """Does a magnitude estimate clear ``threshold``?

    Point path (``lower`` omitted — e.g. the Newey-West t, itself an inference
    statistic): ``|point_estimate| > threshold``.

    CI path (design §6, v0.6 — ICIR, incremental ICIR): a supplied ``lower`` is the
    estimate's LOWER CONFIDENCE BOUND already ALIGNED to the expected direction, and
    the estimate clears only if that lower bound exceeds the bar — "convincingly
    above", not the naked point. A thin/noisy estimate -> wide CI -> low lower bound
    -> fails automatically. A NaN ``lower`` cannot clear (unknown never PASSes); and
    it never manufactures a FAIL — the FAIL branches are POINT-based and never call
    this with a ``lower``.
    """
    if lower is not None:
        return math.isfinite(lower) and lower > threshold
    return math.isfinite(point_estimate) and abs(point_estimate) > threshold


def _clears_level(point_estimate: float, threshold: float) -> bool:
    """point_estimate > threshold (a directed level), estimate KNOWN and finite."""
    return math.isfinite(point_estimate) and point_estimate > threshold


def _aligned_bounds(sign: int, ci_low: float, ci_high: float) -> tuple[float, float]:
    """The CI of ``sign * estimate``: ``(lower, upper)`` in the EXPECTED direction.

    ``ci_low``/``ci_high`` bracket the RAW estimate. Multiplying an INTERVAL by a
    negative number REVERSES it, so the endpoints are re-sorted rather than kept in
    place — the min/max swap at ``sign = -1`` is the whole content of this helper,
    and doing it in one place is why the ICIR gate and the (v0.9) monotonicity gate
    cannot drift apart on the convention.

    ``(NaN, NaN)`` when either bound is unknown — the caller then treats the
    estimate as not-convincingly-clearing, and never as a FAIL.
    """
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        nan = float("nan")
        return nan, nan
    return min(sign * ci_low, sign * ci_high), max(sign * ci_low, sign * ci_high)


def _aligned_lower_bound(sign: int, ci_low: float, ci_high: float) -> float:
    """The lower CI bound of ``sign * estimate`` (the bound in the EXPECTED direction).

    The worst-case value in the direction the factor claims — what a POSITIVE claim
    (an axis PASS) must clear. NaN when either bound is unknown.
    """
    return _aligned_bounds(sign, ci_low, ci_high)[0]


# -- shared fact readers -------------------------------------------------


def _sample_gate_failures(
    inputs: VerdictInputs, thr: VerdictThresholds
) -> list[str]:
    """Every part of the three-part sample gate that did NOT pass (design §6).

    Returns ALL failures rather than the first: "which part failed" is the whole
    diagnostic value, and a run can fail more than one. Empty list = gate passed.
    An UNKNOWN (non-finite) fact fails as UNKNOWN rather than being compared.
    """
    failures: list[str] = []

    if inputs.settled_rebalances < thr.min_rebalances:
        failures.append(
            f"raw floor: settled rebalances {inputs.settled_rebalances} < required "
            f"{thr.min_rebalances}; below this the IC autocorrelation — and so the "
            f"effective sample size itself — is not estimable."
        )

    n_eff = inputs.effective_samples
    if not math.isfinite(n_eff):
        failures.append(
            f"effective samples (A): UNKNOWN — data_coverage reported no finite "
            f"'effective_samples' (required >= {thr.min_effective_samples}). A "
            f"sample-adequacy gate cannot pass on a sample size nobody measured."
        )
    elif n_eff < thr.min_effective_samples:
        failures.append(
            f"effective samples (A): {n_eff:.2f} < required "
            f"{thr.min_effective_samples} — from {inputs.settled_rebalances} raw "
            f"settled rebalance(s), whose IC observations are autocorrelated and "
            f"therefore NOT that many independent pieces of evidence."
        )

    span = inputs.span_days
    if not math.isfinite(span):
        failures.append(
            f"calendar span (B): UNKNOWN — data_coverage reported no finite "
            f"'span_days' (required >= {thr.min_span_days})."
        )
    elif span < thr.min_span_days:
        failures.append(
            f"calendar span (B): {span:.0f} calendar day(s) < required "
            f"{thr.min_span_days}; a dense-but-short window is not a sample at any "
            f"rebalance frequency."
        )

    return failures


def _aligned_net(net: float, gross: float, sign: int) -> float:
    """Hypothesis-aligned NET spread: ``sign * gross - cost`` (design §6, v0.8).

    The hypothesis decides WHICH LEG IS LONG. Cost is a drag REGARDLESS of
    direction. So the legs are flipped by the sign FIRST and the cost is
    subtracted AFTER::

        cost        = gross - net          (= fee * multiplier * leg_turnover)
        aligned_net = sign * gross - cost

    Before v0.8 this was computed as ``sign * net``, which at ``sign = -1``
    expands to ``-gross + cost``: the trading cost was ADDED BACK, as though
    reversing the legs also reversed who pays the fees. It flattered every
    sign=-1 factor by exactly ``2 * cost``.

    ``sign = +1`` returns ``net`` UNCHANGED and needs no ``gross`` — algebraically
    ``+1 * gross - (gross - net) == net``, and returning ``net`` directly keeps
    the old path bit-identical instead of round-tripping it through a subtraction.

    An unknown (non-finite) ``gross`` at ``sign = -1`` yields NaN: the cost cannot
    be recovered from ``net`` alone, so the aligned spread is UNKNOWN — and
    unknown is neither evidence for nor against (it never convicts, never passes).
    """
    if sign >= 0:
        return net
    if not (math.isfinite(gross) and math.isfinite(net)):
        return float("nan")
    return -gross - (gross - net)


def _base_spread(inputs: VerdictInputs) -> float:
    """Hypothesis-aligned net long-short return at the BASE (1.0x) cost."""
    for multiplier, value in inputs.net_long_short_by_cost:
        if abs(multiplier - 1.0) < 1e-9:
            return _aligned_net(
                value, inputs.gross_long_short_mean, inputs.expected_ic_sign
            )
    return float("nan")


def _all_spreads_negative(inputs: VerdictInputs) -> bool:
    """True only when EVERY known cost scenario is non-positive (design §6).

    An empty/unknown set is NOT "all negative" — unknown is not evidence. A
    sign=-1 run with an unknown gross therefore yields NO scenarios and cannot
    convict (:func:`_aligned_net` returns NaN, which is filtered out here).
    """
    aligned = [
        _aligned_net(v, inputs.gross_long_short_mean, inputs.expected_ic_sign)
        for _, v in inputs.net_long_short_by_cost
    ]
    known = [a for a in aligned if math.isfinite(a)]
    return bool(known) and all(a <= 0 for a in known)


# -- the (v0.9) three-valued monotonicity DIRECTION gate ------------------
#
# Monotonicity is a DIRECTION gate, not a strength gate (v0.7) — but v0.8 revealed
# that its statistic cannot support even that claim as a bare point. Averaging
# per-date rank correlations across dates is heavily ATTENUATED by daily noise: in
# this project's real eleven-factor runs an empirically perfect quantile ladder
# scores only 0.045-0.106, and two factors sat 0.021 apart across a threshold of
# exactly 0.0 with NO dispersion estimate anywhere in the decision. "Point > 0.0"
# on a noisy mean is a coin flip wearing a criterion's clothes — the same defect
# v0.6 fixed for the ICIR by gating on its N_eff lower bound.
#
# So the direction is decided by the CI, and it is THREE-VALUED:
#
#     aligned CI low  > bar   -> HOLDS         (may PASS)
#     aligned CI high < 0     -> CONTRADICTED  (FAIL — a measured reversal)
#     otherwise               -> UNKNOWN       (neither convicts nor acquits)
#
# WHY THE CONTRADICTED TEST IS PINNED AT 0 AND NOT AT THE BAR: they are different
# claims. "Above the bar" is the positive claim, so it moves with a configurable
# (possibly strength-demanding) bar. "Below zero" is REVERSAL — the buckets are
# ordered the wrong way round — which is a fact about the sign, not about how
# strong the project decided to be. Pinning FAIL to a raised bar would convict a
# correctly-ordered-but-weak factor of being reversed, which it is not.

MONO_HOLDS = "HOLDS"
MONO_CONTRADICTED = "CONTRADICTED"
MONO_UNKNOWN = "UNKNOWN"

#: Level-2 fallback disclosure: judged on the POOLED magnitude statistic (v0.7
#: behaviour). Wording kept verbatim from v0.8 — reports and tests read it.
_MONO_POOLED_FALLBACK_NOTE = (
    "monotonicity gate FELL BACK to the pooled 'monotonicity_spearman': "
    "the per-date figure ('monotonicity_spearman_by_date', design §6 v0.8) "
    "was not supplied or not measurable. The pooled statistic is unbounded "
    "and magnitude-sensitive, so this monotonicity judgement is weaker "
    "than a v0.8 run's."
)

#: Level-1 fallback disclosure: the per-date statistic IS available but its CI is
#: not, so the gate reverts to the bare sign of the point (v0.8 behaviour).
#: DELIBERATELY NOT the pooled note's wording — the two are different degradations
#: (which STATISTIC vs which ESTIMATOR OF ITS UNCERTAINTY) and a reader must be
#: able to tell which one happened.
_MONO_POINT_FALLBACK_NOTE = (
    "monotonicity gate REVERTED to the BARE per-date POINT (v0.8 behaviour): no "
    "'monotonicity_spearman_by_date_ci_low/high' was supplied, so the direction was "
    "decided WITHOUT any dispersion estimate. A cross-date mean of per-date rank "
    "correlations is strongly attenuated by daily noise, so a point that merely "
    "lands on the right side of the bar is NOT the same evidence a v0.9 CI would be."
)


def _monotonicity_direction(
    inputs: VerdictInputs, thr: VerdictThresholds
) -> tuple[str, tuple[str, ...]]:
    """Three-valued monotonicity DIRECTION + its disclosure notes (design §6, v0.9).

    Returns ``(MONO_HOLDS | MONO_CONTRADICTED | MONO_UNKNOWN, notes)``.

    THE FALLBACK CHAIN, each level DISCLOSED in ``notes`` (never silent):
      1. the per-date CI (v0.9)      — three-valued, the intended path.
      2. the per-date POINT (v0.8)   — CI absent. TWO-valued, bit-for-bit the v0.8
         rule: aligned point above the bar HOLDS, anything else (including an
         unknown) CONTRADICTS. It cannot yield UNKNOWN: without a dispersion
         estimate there is nothing to be uncertain WITH, and inventing an UNKNOWN
         here would silently convert every pre-v0.9 FAIL into INSUFFICIENT_DATA.
      3. the POOLED point (v0.7)     — per-date figure absent too. Same two-valued
         rule on the weaker, magnitude-sensitive statistic. A NaN here lands on
         CONTRADICTED, exactly as v0.7/v0.8 treated an unmeasurable monotonicity.

    RAW inputs in, hypothesis applied here: this is one of the two places (with
    :func:`_aligned_net`) that knows what ``expected_ic_sign`` means.
    """
    sign = inputs.expected_ic_sign
    bar = thr.min_monotonicity_spearman

    low, high = _aligned_bounds(
        sign,
        inputs.monotonicity_spearman_by_date_ci_low,
        inputs.monotonicity_spearman_by_date_ci_high,
    )
    if math.isfinite(low) and math.isfinite(high):
        interval = f"[{low:+.4f}, {high:+.4f}]"
        if low > bar:
            return MONO_HOLDS, (
                f"monotonicity direction HOLDS: the hypothesis-aligned per-date "
                f"monotonicity 95% CI {interval} (N_eff-based) lies entirely above "
                f"the bar {bar}.",
            )
        if high < 0.0:
            return MONO_CONTRADICTED, (
                f"monotonicity direction CONTRADICTED: the hypothesis-aligned "
                f"per-date monotonicity 95% CI {interval} (N_eff-based) lies "
                f"entirely BELOW 0 — the quantile buckets are ordered AGAINST the "
                f"stated hypothesis, which is a measured negative finding.",
            )
        return MONO_UNKNOWN, (
            f"monotonicity direction is INDISTINGUISHABLE FROM 0: the "
            f"hypothesis-aligned per-date monotonicity 95% CI {interval} "
            f"(N_eff-based, point {sign * inputs.monotonicity_spearman_by_date:+.4f}) "
            f"straddles the bar {bar} / zero, so the evidence is not sufficient to "
            f"ASSERT that the direction holds. Not a refutation either — this "
            f"neither convicts nor acquits.",
        )

    point = inputs.monotonicity_spearman_by_date
    if math.isfinite(point):
        state = (
            MONO_HOLDS if _clears_level(sign * point, bar) else MONO_CONTRADICTED
        )
        return state, (_MONO_POINT_FALLBACK_NOTE,)

    state = (
        MONO_HOLDS
        if _clears_level(sign * inputs.monotonicity_spearman, bar)
        else MONO_CONTRADICTED
    )
    return state, (_MONO_POOLED_FALLBACK_NOTE, _MONO_POINT_FALLBACK_NOTE)


def _has_core_point_signal(inputs: VerdictInputs, thr: VerdictThresholds) -> bool:
    """POINT-estimate in-sample signal on the MAGNITUDE metrics: |ICIR|, NW-t, win rate.

    Every component must be a finite, known number: an unknown is not a signal.
    This is the v0.5 point check, kept for the FAIL determination — a factor whose
    POINT metrics do not clear has no signal at all (a negative finding). The CI
    lower-bound test (design §6, v0.6) is applied SEPARATELY, only to decide PASS vs
    "promising but unconfirmed" for a factor that DID clear on the point.

    MONOTONICITY IS DELIBERATELY NOT IN HERE (v0.9). It used to be, as a fourth
    ``and`` term, which made it silently indistinguishable from the others — and
    that is exactly what must not happen now that it can come back UNKNOWN. Folding
    a three-valued fact into a boolean would collapse UNKNOWN onto one of the two
    other answers; keeping it separate is what lets the axis route an unknown
    direction to INSUFFICIENT_DATA while an unknown-direction factor that ALSO
    fails here still FAILs.
    """
    return (
        _clears_magnitude(inputs.ic_ir, thr.min_abs_icir)
        and _clears_magnitude(inputs.ic_nw_t, thr.min_abs_nw_t)
        and _clears_level(inputs.ic_win_rate, thr.min_ic_win_rate)
    )


# -- Axis A: Predictive --------------------------------------------------


def _predictive_axis(inputs: VerdictInputs, thr: VerdictThresholds) -> AxisVerdict:
    """Is there a real, OUT-OF-SAMPLE predictive signal? (design §6, Axis A).

    Order (FAIL before the gate — the asymmetric rule):
      1. FAIL   a KNOWN out-of-sample reversal (sign flip or independent-cell
                monotonicity reversal). Decided BEFORE the gate, so it Rejects even
                on a thin sample. Requires oos_available AND a set reversal flag —
                unknown never convicts.
      2. INSUFFICIENT_DATA  the sample gate failed OR there is no OOS evidence.
                In-sample metrics are still REPORTED (in the section), just not a
                PASS: no out-of-sample evidence means no predictive claim. The two
                reasons (gate vs no-OOS) are named distinctly.
      3. FAIL   sufficient data + OOS, but a NEGATIVE finding: the OOS sign is not
                consistent OR the in-sample POINT metrics do not clear (no signal at
                all) OR the monotonicity direction is CONTRADICTED (v0.9: its
                aligned CI lies entirely below 0). POINT-based on the magnitude
                metrics, so a merely wide ICIR CI never manufactures a FAIL.
      3b. INSUFFICIENT_DATA  (v0.9) everything else cleared, but the monotonicity
                direction is UNKNOWN — its aligned CI straddles 0, so the direction
                can be neither asserted nor refuted.
      4. PASS   OOS sign consistent AND point metrics clear AND monotonicity
                direction HOLDS AND the ICIR LOWER CI bound (N_eff-based, expected
                direction) exceeds the bar — design §6, v0.6: convincingly above,
                not the naked point.
      5. INSUFFICIENT_DATA  point metrics clear + OOS consistent, but the ICIR lower
                CI does NOT clear (or is unknown): promising, yet not confirmed at
                the pre-registered bar. This is the "gate passed, CI still straddles
                the bar" case — distinct from the sample gate (step 2).

    WHERE UNKNOWN SITS IN THE ORDER IS THE WHOLE POINT (v0.9). Step 3 runs FIRST and
    is checked against the OTHER criteria in full, so an UNKNOWN monotonicity can
    never RESCUE a factor that fails on ICIR / NW-t / win rate / OOS consistency:
    that factor still FAILs, and the unknown direction is merely disclosed
    alongside. Step 3b is reached only when every other criterion has cleared and
    the direction ALONE cannot be asserted. "We could not tell" is a reason to
    withhold a PASS, never a reason to withhold a FAIL that was independently
    earned.
    """
    sign = inputs.expected_ic_sign
    # 1. FAIL on a KNOWN out-of-sample reversal — before the gate.
    if inputs.oos_available and (
        inputs.oos_sign_flipped or inputs.oos_monotonicity_reversed
    ):
        reasons: list[str] = []
        if inputs.oos_sign_flipped:
            reasons.append(
                "out-of-sample IC sign flipped against the stated hypothesis."
            )
        if inputs.oos_monotonicity_reversed:
            reasons.append(
                "quantile monotonicity reversed on an independent cell."
            )
        return AxisVerdict(AXIS_FAIL, tuple(reasons))

    # 2. INSUFFICIENT_DATA on the sample gate OR no OOS evidence.
    reasons = _sample_gate_failures(inputs, thr)
    if not inputs.oos_available:
        reasons.append(
            "no out-of-sample split: generalization NOT established, so there is "
            "no predictive PASS to claim (in-sample metrics are reported, not a "
            "PASS)."
        )
    if reasons:
        return AxisVerdict(AXIS_INSUFFICIENT_DATA, tuple(reasons))

    # The monotonicity direction actually gated on (design §6, v0.9): three-valued,
    # decided on the per-date CI, with every fallback level DISCLOSED. A silent
    # substitution would hide that the decision rests on a weaker statistic (the
    # pooled magnitude one the v0.8 fix exists to stop gating on) or on a point with
    # no dispersion estimate at all (the v0.9 fix).
    mono_state, mono_notes = _monotonicity_direction(inputs, thr)

    # 3. FAIL: a measured negative finding (OOS sign not consistent, no point signal
    #    at all, or a CONTRADICTED monotonicity direction). Checked BEFORE the
    #    UNKNOWN branch, so an unknown direction never rescues a factor that failed
    #    on its own other evidence. POINT-based on the magnitude metrics — a wide
    #    ICIR CI is handled below, not here.
    point_signal = _has_core_point_signal(inputs, thr)
    if (
        (not inputs.oos_sign_consistent)
        or (not point_signal)
        or mono_state == MONO_CONTRADICTED
    ):
        reasons = []
        if not inputs.oos_sign_consistent:
            reasons.append(
                "out-of-sample: the expected IC sign is NOT consistent across both "
                "holdout subperiods (we looked out-of-sample and the signal was "
                "not there)."
            )
        if not point_signal:
            reasons.append(
                "in-sample POINT metrics do NOT clear the predictive thresholds "
                "(ICIR / NW-t / win rate)."
            )
        if mono_state == MONO_CONTRADICTED:
            reasons.append(
                "quantile monotonicity does not hold in the hypothesized direction."
            )
        return AxisVerdict(AXIS_FAIL, tuple(reasons) + mono_notes)

    # 3b. UNKNOWN monotonicity with EVERYTHING ELSE cleared -> INSUFFICIENT_DATA
    #     (v0.9). Reached only after step 3 declined to fail: this branch withholds
    #     a PASS, it never withholds a FAIL.
    if mono_state == MONO_UNKNOWN:
        return AxisVerdict(
            AXIS_INSUFFICIENT_DATA,
            (
                "every other predictive criterion cleared (ICIR / NW-t / win rate "
                "point metrics, and the out-of-sample sign is consistent), but the "
                "monotonicity DIRECTION cannot be distinguished from 0: its "
                "confidence interval straddles zero, so the evidence is NOT "
                "sufficient to assert that the direction holds. Withheld, not "
                "refuted — an unknown direction is not a PASS and not a FAIL.",
            )
            + mono_notes,
        )

    # 4/5. Point signal present + OOS consistent. The #3 lower-CI test decides
    #      PASS vs "promising but unconfirmed" (INSUFFICIENT_DATA). This is ABOVE
    #      the sample gate and does NOT duplicate it (see _clears_magnitude).
    icir_lower = _aligned_lower_bound(sign, inputs.ic_ir_ci_low, inputs.ic_ir_ci_high)
    if _clears_magnitude(inputs.ic_ir, thr.min_abs_icir, lower=icir_lower):
        return AxisVerdict(
            AXIS_PASS,
            (
                "expected IC sign holds in both out-of-sample subperiods.",
                f"in-sample signal convincingly above the bar: the ICIR lower CI "
                f"bound (N_eff-based, expected direction) {icir_lower:+.3f} > "
                f"{thr.min_abs_icir}; point |ICIR| {abs(inputs.ic_ir):.3f}, "
                f"|NW-t| {abs(inputs.ic_nw_t):.2f}, win rate "
                f"{inputs.ic_win_rate:.3f}.",
            )
            + mono_notes,
        )
    lower_txt = "UNKNOWN" if not math.isfinite(icir_lower) else f"{icir_lower:+.3f}"
    return AxisVerdict(
        AXIS_INSUFFICIENT_DATA,
        (
            f"the point metrics clear and the out-of-sample sign is consistent, but "
            f"the ICIR LOWER CI bound ({lower_txt}, N_eff-based) does not exceed the "
            f"pre-registered bar {thr.min_abs_icir}: promising, but not confirmed. "
            f"The sample size passed the gate; the ESTIMATE is still too imprecise "
            f"to claim a PASS (a wider bar than the raw count implies).",
        )
        + mono_notes,
    )


# -- Axis B: Incremental -------------------------------------------------


def _incremental_axis(inputs: VerdictInputs, thr: VerdictThresholds) -> AxisVerdict:
    """Does it add alpha BEYOND the known factor set? (design §6, Axis B).

    The orthogonalized IC is the factor's IC AFTER residualizing it on the WHOLE
    known-factor book per date (computed in PR-B's ``standard.py``; consumed here
    as ``incremental_ic_ir``, exactly as the Predictive axis consumes ``ic_ir``).

    Order:
      * NOT_ASSESSED     no known_factors were supplied (the default).
      * INSUFFICIENT_DATA  a book was supplied but the sample is too thin to
                         assess it (same N_eff/span gate), OR the orthogonalized
                         IC could not be measured (NaN — unknown, never a FAIL),
                         OR the POINT clears in the expected direction but the LOWER
                         CI bound does not (promising, unconfirmed — design §6, v0.6).
      * PASS             the orthogonalized IC is significantly in the EXPECTED
                         direction: its LOWER CI bound (N_eff-based) clears
                         min_abs_icir.
      * FAIL             book + sufficient sample, but the orthogonalized IC POINT
                         is convincingly ~ 0 or in the WRONG direction (redundant /
                         anti-incremental — the factor duplicates the book). FAIL is
                         POINT-based, so a merely wide CI never manufactures it.
    """
    sign = inputs.expected_ic_sign
    if not inputs.known_factors_supplied:
        return AxisVerdict(
            AXIS_NOT_ASSESSED,
            (
                "no known-factor book supplied: incremental value BEYOND the "
                "existing factors cannot be assessed. Pass EvalContext."
                "known_factors to evaluate this axis.",
            ),
        )

    gate_failures = _sample_gate_failures(inputs, thr)
    if gate_failures:
        return AxisVerdict(
            AXIS_INSUFFICIENT_DATA,
            tuple(gate_failures)
            + (
                "a book was supplied, but the sample is too thin to judge whether "
                "the factor is incremental to it.",
            ),
        )

    if not math.isfinite(inputs.incremental_ic_ir):
        return AxisVerdict(
            AXIS_INSUFFICIENT_DATA,
            (
                "the orthogonalized IC (factor residualized on the known-factor "
                "book) could not be measured — reported as UNKNOWN, which is not a "
                "FAIL: unknown never convicts.",
            ),
        )

    # FAIL is POINT-based (v0.5 semantics unchanged): a measured orthogonalized IC
    # that is ~ 0 or in the wrong direction. A merely WIDE CI is handled below, not
    # here, so an imprecise-but-positive estimate is never mislabelled "redundant".
    aligned_point = sign * inputs.incremental_ic_ir
    if not (aligned_point > 0 and _clears_magnitude(inputs.incremental_ic_ir, thr.min_incremental_abs_icir)):
        direction = (
            "in the WRONG direction (anti-incremental)"
            if aligned_point < 0
            else "~ 0 (redundant with the book)"
        )
        return AxisVerdict(
            AXIS_FAIL,
            (
                f"orthogonalized ICIR {inputs.incremental_ic_ir:+.3f} is "
                f"{direction}: after residualizing on the known-factor book the "
                f"factor adds no signal of magnitude > {thr.min_incremental_abs_icir} in its "
                f"stated direction.",
            ),
        )

    # The point clears in the expected direction. The #3 lower-CI test (design §6,
    # v0.6) decides PASS vs "promising but unconfirmed", ABOVE the sample gate.
    incr_lower = _aligned_lower_bound(
        sign, inputs.incremental_ic_ir_ci_low, inputs.incremental_ic_ir_ci_high
    )
    if _clears_magnitude(inputs.incremental_ic_ir, thr.min_incremental_abs_icir, lower=incr_lower):
        return AxisVerdict(
            AXIS_PASS,
            (
                f"orthogonalized ICIR lower CI bound (N_eff-based, expected "
                f"direction) {incr_lower:+.3f} > {thr.min_incremental_abs_icir}: the factor "
                f"convincingly adds a signal the book does not already carry "
                f"(point {inputs.incremental_ic_ir:+.3f}).",
            ),
        )
    lower_txt = "UNKNOWN" if not math.isfinite(incr_lower) else f"{incr_lower:+.3f}"
    return AxisVerdict(
        AXIS_INSUFFICIENT_DATA,
        (
            f"the orthogonalized ICIR point {inputs.incremental_ic_ir:+.3f} is in "
            f"the expected direction, but its LOWER CI bound ({lower_txt}, "
            f"N_eff-based) does not exceed the pre-registered bar {thr.min_incremental_abs_icir}: "
            f"promising incremental value, but not confirmed. Sufficient sample; the "
            f"estimate is still too imprecise to claim a PASS.",
        ),
    )


# -- Axis C: Tradable ----------------------------------------------------


def _tradable_axis(inputs: VerdictInputs, thr: VerdictThresholds) -> AxisVerdict:
    """Can the signal actually be harvested? (design §6, Axis C).

    Execution facts (I5b fill feasibility / I5f capacity) are MEASURED ELSEWHERE
    and are not IR inputs, so the DEFAULT is NOT_ASSESSED.

      * NOT_ASSESSED     no execution facts supplied (tradable is None AND
                         capacity is None).
      * FAIL             a KNOWN execution failure: ``tradable is False`` OR
                         ``capacity_sufficient is False`` OR net long-short
                         non-positive at EVERY cost scenario. (No sample gate — a
                         "cannot execute" is not a statistical claim.)
      * PASS             tradable AND capacity sufficient AND net positive at base
                         cost.
      * INSUFFICIENT_DATA  partial facts (e.g. tradable known but capacity not, or
                         the base-cost spread unknown): established neither way,
                         and unknown must not be spun into either a PASS or a FAIL.
    """
    if inputs.tradable is None and inputs.capacity_sufficient is None:
        return AxisVerdict(
            AXIS_NOT_ASSESSED,
            (
                "no execution facts supplied (I5b fill feasibility / I5f capacity "
                "are measured elsewhere, not IR inputs): tradability not assessed.",
            ),
        )

    fail_reasons: list[str] = []
    if inputs.tradable is False:
        fail_reasons.append("not tradable: the spread cannot be executed.")
    if inputs.capacity_sufficient is False:
        fail_reasons.append(
            "capacity insufficient at the target notional."
        )
    if _all_spreads_negative(inputs):
        fail_reasons.append(
            "net long-short return is non-positive in EVERY cost scenario."
        )
    if fail_reasons:
        return AxisVerdict(AXIS_FAIL, tuple(fail_reasons))

    base = _base_spread(inputs)
    if (
        inputs.tradable is True
        and inputs.capacity_sufficient is True
        and _clears_level(base, 0.0)
    ):
        return AxisVerdict(
            AXIS_PASS,
            (
                "fills feasible and capacity sufficient at the target notional, "
                f"and the net long-short return is positive at base cost "
                f"({base:+.4f}).",
            ),
        )

    return AxisVerdict(
        AXIS_INSUFFICIENT_DATA,
        (
            "tradability marginal: fills and/or capacity are not both established, "
            "or the base-cost net spread is unknown/non-positive. Established "
            "neither way — unknown is not a PASS and not a FAIL.",
        ),
    )


# -- the deployment derivation -------------------------------------------


_CAP_REASON = (
    "CAPPED AT WATCH: this run declares EvalConfig.is_exploratory=True, so no "
    "Adopt claim is available to it regardless of the axes — Adopt IS a "
    "performance claim and an exploratory run declares it is not making one. The "
    "per-axis PASS evidence below stands on its own; to claim Adopt, re-run as a "
    "non-exploratory confirmation."
)


def _tag(name: str, axis: AxisVerdict) -> list[str]:
    """Per-axis reasons, prefixed with the axis name + its state (for the label)."""
    return [f"[{name} {axis.verdict}] {reason}" for reason in axis.reasons]


def _derive_deployment(
    predictive: AxisVerdict,
    incremental: AxisVerdict,
    tradable: AxisVerdict,
    inputs: VerdictInputs,
) -> tuple[str, tuple[str, ...]]:
    """Derive the single deployment label from the three axes (design §6 table).

    Order preserves the asymmetric gate + exploratory cap:
      any FAIL -> REJECT (FAIL is checked FIRST, so a thin-sample failure still
      Rejects); all three PASS -> ADOPT (capped to WATCH if exploratory);
      >=1 PASS with the rest unresolved -> WATCH; no PASS and no FAIL ->
      INSUFFICIENT-DATA.
    """
    axes = (
        ("predictive", predictive),
        ("incremental", incremental),
        ("tradable", tradable),
    )
    fails = [(name, axis) for name, axis in axes if axis.verdict == AXIS_FAIL]
    passes = [(name, axis) for name, axis in axes if axis.verdict == AXIS_PASS]
    unresolved = [
        (name, axis)
        for name, axis in axes
        if axis.verdict in (AXIS_INSUFFICIENT_DATA, AXIS_NOT_ASSESSED)
    ]

    # 1. any FAIL -> REJECT. Checked first, so it bypasses the sample gate.
    if fails:
        names = ", ".join(name for name, _ in fails)
        reasons = [
            f"REJECT: the {names} axis/axes FAILED. A factor that fails ANY axis "
            f"is rejected, and this decision precedes the sample gate — a "
            f"thin-sample failure still Rejects."
        ]
        for name, axis in fails:
            reasons += _tag(name, axis)
        return REJECT, tuple(reasons)

    # 2. all three PASS -> ADOPT (capped to WATCH if the run is exploratory).
    evidence = [reason for name, axis in passes for reason in _tag(name, axis)]
    if len(passes) == 3:
        if inputs.is_exploratory:
            return WATCH, (_CAP_REASON, *evidence)
        return ADOPT, tuple(evidence)

    # 3. >=1 PASS, rest unresolved -> WATCH.
    if passes:
        pass_names = ", ".join(name for name, _ in passes)
        unresolved_names = ", ".join(
            f"{name} ({axis.verdict})" for name, axis in unresolved
        )
        reasons = [
            f"WATCH: {pass_names} established; unresolved: {unresolved_names}. A "
            f"positive claim on some axes but not all, so no Adopt."
        ]
        for name, axis in passes:
            reasons += _tag(name, axis)
        for name, axis in unresolved:
            reasons += _tag(name, axis)
        return WATCH, tuple(reasons)

    # 4. no FAIL, no PASS -> INSUFFICIENT-DATA.
    reasons = [
        "INSUFFICIENT-DATA: no axis reached a positive PASS and none failed — "
        "nothing was demonstrated and nothing was refuted."
    ]
    for name, axis in unresolved:
        reasons += _tag(name, axis)
    return INSUFFICIENT_DATA, tuple(reasons)


def decide_verdict(
    inputs: VerdictInputs, thresholds: VerdictThresholds | None = None
) -> VerdictResult:
    """Score the three axes and derive the deployment label. Pure + deterministic."""
    thr = thresholds or VerdictThresholds()
    predictive = _predictive_axis(inputs, thr)
    incremental = _incremental_axis(inputs, thr)
    tradable = _tradable_axis(inputs, thr)
    label, reasons = _derive_deployment(predictive, incremental, tradable, inputs)
    return VerdictResult(
        verdict=label,
        reasons=reasons,
        predictive=predictive,
        incremental=incremental,
        tradable=tradable,
    )


__all__ = [
    "INSUFFICIENT_DATA",
    "ADOPT",
    "WATCH",
    "REJECT",
    "VERDICTS",
    "AXIS_PASS",
    "AXIS_FAIL",
    "AXIS_INSUFFICIENT_DATA",
    "AXIS_NOT_ASSESSED",
    "AXIS_VERDICTS",
    "AXIS_NAMES",
    "MONO_HOLDS",
    "MONO_CONTRADICTED",
    "MONO_UNKNOWN",
    "VerdictThresholds",
    "VerdictInputs",
    "AxisVerdict",
    "VerdictResult",
    "decide_verdict",
]
