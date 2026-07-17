"""``Section`` / ``Skipped``: what one report section may be — and nothing else.

A mandatory section is either produced (:class:`Section`) or explicitly skipped
with a reason (:class:`Skipped`). There is no third option: a silently missing
section is what :meth:`FactorEvalReport.validate_all_mandatory_present` raises
on (enforcement layer #2, design §7). This mirrors the project's standing rule
绝不静默降级 — a degraded path must SAY it degraded.

Kept in its own module so both ``report`` (the data model) and ``render`` (the
presentation) can import these types without an import cycle.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

from data.quality.report import sanitize_text

# The 8 evaluator-produced sections, in FIXED order (design §5). They render as
# sections 2..9; sections 0 (provenance) and 1 (verdict) come from the report.
MANDATORY_SECTIONS: tuple[str, ...] = (
    "predictive_power",
    "return_risk",
    "stability_cost",
    "purity",
    "oos_generalization",
    "execution_capacity",
    "data_coverage",
    "caveats",
)

# --- payload keys the verdict reads (the contract PR-B must fill) ------------
# Every key is a RAW fact; the hypothesis (expected_ic_sign) is applied by the
# verdict rules, never by the section. See analytics/eval/verdict.py.
VERDICT_KEYS: dict[str, tuple[str, ...]] = {
    # The three-part sample gate's facts (design §6, v0.3). settled_rebalances is
    # the RAW count of rebalances that settled; effective_samples is the IC
    # series' N_eff = N / (1 + 2*sum_k rho_k), CLAMPED to [1, N]; span_days is
    # the IC series' calendar span. The last two are computed from the IR
    # (ir.ic + its date index) — a raw count is not a sample size under a daily
    # rebalance, so the gate reads all three or reports UNKNOWN and fails.
    "data_coverage": ("settled_rebalances", "effective_samples", "span_days"),
    # ic_ir = mean(IC)/std(IC); ic_win_rate = share of periods whose IC carries
    # the EXPECTED sign (hypothesis-relative BY DEFINITION); ic_nw_t =
    # Newey-West autocorrelation-corrected t of the mean IC.
    # ic_ir_ci_low/high = N_eff-based 95% CI of ic_ir (design §6, v0.6): the
    # Predictive PASS gates on the LOWER bound, not the point.
    "predictive_power": (
        "ic_ir",
        "ic_ir_ci_low",
        "ic_ir_ci_high",
        "ic_win_rate",
        "ic_nw_t",
    ),
    # monotonicity_spearman: RAW Spearman(bucket index, bucket mean return).
    # net_long_short_by_cost: {cost multiplier: net (top - bottom) return}, RAW.
    "return_risk": ("monotonicity_spearman", "net_long_short_by_cost"),
    # The Incremental axis (design §6, v0.5). known_factors_supplied = was a
    # known-factor BOOK supplied at all (False -> the axis is NOT_ASSESSED);
    # incremental_ic_ir / incremental_ic_mean = mean(orthIC)/std(orthIC) and
    # mean(orthIC) of the factor's IC AFTER residualizing it on the WHOLE book per
    # date. RAW: the verdict applies expected_ic_sign, never the section.
    "purity": (
        "known_factors_supplied",
        "incremental_ic_ir",
        "incremental_ic_mean",
        # N_eff-based 95% CI of the orthogonalized ICIR (design §6, v0.6): the
        # Incremental PASS gates on the LOWER bound.
        "incremental_ic_ir_ci_low",
        "incremental_ic_ir_ci_high",
    ),
    "oos_generalization": (
        "oos_available",
        "sign_consistent",
        "sign_flipped",
        "monotonicity_reversed",
    ),
    "execution_capacity": ("tradable", "capacity_sufficient"),
}


@dataclass(frozen=True)
class Section:
    """One produced report section: a name plus its metric payload.

    ``payload`` holds the section's facts (the verdict reads the documented keys
    in :data:`VERDICT_KEYS`; everything else is rendered for the human). It is
    DEEP-copied on construction — a caller mutating its dict afterwards must not
    be able to rewrite a report — and always rendered/exported with sorted keys.
    A shallow copy is not enough: ``net_long_short_by_cost`` (a nested dict the
    verdict READS) would still be aliased to the caller's object.
    """

    name: str
    payload: Mapping[str, object] = field(default_factory=dict)
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Section.name must be a non-empty string; got {self.name!r}.")
        if not isinstance(self.payload, Mapping):
            raise ValueError(
                f"Section({self.name!r}).payload must be a mapping of metric name "
                f"-> value; got {type(self.payload).__name__}."
            )
        object.__setattr__(self, "payload", copy.deepcopy(dict(self.payload)))
        if self.note is not None:
            object.__setattr__(self, "note", sanitize_text(str(self.note)))


@dataclass(frozen=True)
class Skipped:
    """A section that was NOT produced — with a mandatory, explicit reason.

    The reason is the whole point: a skipped section is a DISCLOSED degradation
    ("no OOS split configured", "capacity diagnostic disabled"), never a silent
    hole in the report.
    """

    name: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Skipped.name must be a non-empty string; got {self.name!r}.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"Skipped({self.name!r}) requires a non-empty reason: a section may "
                f"be skipped ONLY with an explicit, disclosed reason — never "
                f"silently."
            )
        object.__setattr__(self, "reason", sanitize_text(self.reason))


SectionLike = Section | Skipped


__all__ = ["MANDATORY_SECTIONS", "VERDICT_KEYS", "Section", "SectionLike", "Skipped"]
