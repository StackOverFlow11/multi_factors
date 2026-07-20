"""``FactorEvalReport``: the standard evaluation output (provenance + sections + verdict).

Enforcement layer #2 (design ``tmp/design/factor_eval_contract_v0.1.md`` §7):
:meth:`FactorEvalReport.validate_all_mandatory_present` raises when a mandatory
section is simply absent. A :class:`~analytics.eval.sections.Skipped` WITH a
reason counts as present — the contract demands DISCLOSURE, not results.

A report is built in three explicit steps (the design's template method):

    assemble  ->  validate_all_mandatory_present  ->  with_verdict

Rendering/exporting REQUIRES the verdict: publishing a report that dodges the
verdict is precisely what this contract exists to prevent.

COMPLETENESS IS ENFORCED TWICE, on purpose (绝不静默漏段):

  1. :meth:`with_verdict` re-runs ``validate_all_mandatory_present`` itself, so a
     caller who skips step 2 still cannot obtain a verdict on an incomplete
     report. ``assemble`` stays permissive (a report under construction is not
     yet a claim); the VERDICT is the gate, because the verdict is the claim.
  2. Since ``render`` / ``to_dict`` / ``to_json`` all require a verdict, they are
     unreachable on an unvalidated report through the public API. The remaining
     path is a hand-rolled ``FactorEvalReport(..., verdict=...)`` that bypasses
     both classmethod and step 2 — for THAT, both renderings mark an absent
     mandatory section EXPLICITLY (Markdown ``_MISSING_``, JSON
     ``{"status": "missing"}``) rather than quietly emitting a short record.
     Silently dropping a section from the machine-readable export was the actual
     hole: the Markdown said ``_MISSING_`` while the JSON just shipped 3 of 8
     sections and looked structurally valid.

The section types live in ``sections.py`` and the two renderings in
``render.py``; this module is the data model + the verdict extraction.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from analytics.eval.config import EvalConfig
from analytics.eval.render import render_report, report_to_dict
from analytics.eval.sections import (
    MANDATORY_SECTIONS,
    VERDICT_KEYS,
    Section,
    SectionLike,
    Skipped,
)
from analytics.eval.verdict import (
    VerdictInputs,
    VerdictResult,
    VerdictThresholds,
    decide_verdict,
)
from factors.spec import FactorSpec

SCHEMA_VERSION = "0.1"


def _payload_number(payload: Mapping[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _payload_flag(payload: Mapping[str, object], key: str, default: bool | None):
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


def extract_verdict_inputs(
    spec: FactorSpec, cfg: EvalConfig, sections: Mapping[str, SectionLike]
) -> VerdictInputs:
    """Pull the verdict's facts out of the section payloads (keys: VERDICT_KEYS).

    A skipped/missing section leaves its facts UNKNOWN (NaN / None / False / 0
    rebalances) rather than inventing them: unknown never earns an Adopt, and
    only an EXPLICIT failure produces a hard Reject (see verdict.py).

    ``cfg`` supplies the one non-measured input: ``is_exploratory``, the run's
    own declaration, which caps the verdict at Watch (design §6). It is read off
    the CONFIG — where EvalConfig makes it mandatory — and never off the caveats
    payload, so an evaluator cannot shed the cap by omitting a key.
    """

    def payload_of(name: str) -> Mapping[str, object]:
        section = sections.get(name)
        return section.payload if isinstance(section, Section) else {}

    coverage = payload_of("data_coverage")
    predictive = payload_of("predictive_power")
    returns = payload_of("return_risk")
    purity = payload_of("purity")
    oos = payload_of("oos_generalization")
    execution = payload_of("execution_capacity")

    raw_costs = returns.get("net_long_short_by_cost", {})
    by_cost: tuple[tuple[float, float], ...] = ()
    if isinstance(raw_costs, Mapping):
        by_cost = tuple(
            (float(k), float(v))
            for k, v in sorted(raw_costs.items(), key=lambda kv: float(kv[0]))
        )

    settled = coverage.get("settled_rebalances", 0)
    return VerdictInputs(
        expected_ic_sign=spec.expected_ic_sign,
        is_exploratory=cfg.is_exploratory,
        settled_rebalances=int(settled) if isinstance(settled, (int, float)) else 0,
        # Gate parts A and B. Absent -> NaN -> reported as UNKNOWN and FAILS the
        # gate; an unmeasured sample size must never satisfy a sample gate.
        effective_samples=_payload_number(coverage, "effective_samples", float("nan")),
        span_days=_payload_number(coverage, "span_days", float("nan")),
        ic_ir=_payload_number(predictive, "ic_ir", float("nan")),
        ic_ir_ci_low=_payload_number(predictive, "ic_ir_ci_low", float("nan")),
        ic_ir_ci_high=_payload_number(predictive, "ic_ir_ci_high", float("nan")),
        ic_win_rate=_payload_number(predictive, "ic_win_rate", float("nan")),
        ic_nw_t=_payload_number(predictive, "ic_nw_t", float("nan")),
        monotonicity_spearman=_payload_number(
            returns, "monotonicity_spearman", float("nan")
        ),
        # design §6 v0.8: the GATED monotonicity. Absent (a pre-v0.8 IR) -> NaN ->
        # the verdict falls back to the pooled field and DISCLOSES it.
        monotonicity_spearman_by_date=_payload_number(
            returns, "monotonicity_spearman_by_date", float("nan")
        ),
        # Incremental axis facts (design §6, v0.5). A Skipped purity section (no
        # book) leaves the payload empty -> known_factors_supplied defaults False
        # -> the axis is NOT_ASSESSED. A supplied-but-unmeasurable orthogonalized
        # IC is NaN -> INSUFFICIENT_DATA, never a FAIL (unknown never convicts).
        known_factors_supplied=bool(
            _payload_flag(purity, "known_factors_supplied", False)
        ),
        incremental_ic_ir=_payload_number(purity, "incremental_ic_ir", float("nan")),
        incremental_ic_mean=_payload_number(
            purity, "incremental_ic_mean", float("nan")
        ),
        incremental_ic_ir_ci_low=_payload_number(
            purity, "incremental_ic_ir_ci_low", float("nan")
        ),
        incremental_ic_ir_ci_high=_payload_number(
            purity, "incremental_ic_ir_ci_high", float("nan")
        ),
        net_long_short_by_cost=by_cost,
        # design §6 v0.8: needed to align a spread by hypothesis without adding the
        # cost back (cost = gross - net). Absent -> NaN -> a sign=-1 aligned spread
        # reads UNKNOWN rather than silently mis-signed.
        gross_long_short_mean=_payload_number(
            returns, "gross_long_short_mean", float("nan")
        ),
        oos_available=bool(_payload_flag(oos, "oos_available", False)),
        oos_sign_consistent=bool(_payload_flag(oos, "sign_consistent", False)),
        oos_sign_flipped=bool(_payload_flag(oos, "sign_flipped", False)),
        oos_monotonicity_reversed=bool(
            _payload_flag(oos, "monotonicity_reversed", False)
        ),
        tradable=_payload_flag(execution, "tradable", None),
        capacity_sufficient=_payload_flag(execution, "capacity_sufficient", None),
    )


@dataclass(frozen=True)
class FactorEvalReport:
    """Provenance (spec + cfg) + sections + verdict. Immutable."""

    spec: FactorSpec
    cfg: EvalConfig
    sections: tuple[SectionLike, ...]
    verdict: VerdictResult | None = None
    thresholds: VerdictThresholds = field(default_factory=VerdictThresholds)
    #: "declared" when the verdict bar came from EvalConfig.success_criteria (a
    #: pre-registered bar), "default" otherwise. Stamped in the provenance box +
    #: JSON so a reader can tell a pre-registered result from one judged against the
    #: global default (design §6, v0.6).
    criteria_source: str = "default"

    #: exported so a stored record can be matched to the schema that wrote it.
    SCHEMA_VERSION = SCHEMA_VERSION

    @classmethod
    def assemble(
        cls,
        spec: FactorSpec,
        cfg: EvalConfig,
        sections: Sequence[SectionLike],
        thresholds: VerdictThresholds | None = None,
    ) -> FactorEvalReport:
        """Collect sections into a report (no verdict yet). Rejects duplicates."""
        if not isinstance(spec, FactorSpec):
            raise TypeError(
                f"FactorEvalReport needs a FactorSpec; got {type(spec).__name__}."
            )
        if not isinstance(cfg, EvalConfig):
            raise TypeError(
                f"FactorEvalReport needs an EvalConfig; got {type(cfg).__name__}."
            )
        collected = tuple(sections)
        bad = [s for s in collected if not isinstance(s, (Section, Skipped))]
        if bad:
            raise TypeError(
                f"every report section must be a Section or an explicit "
                f"Skipped(reason); got {[type(s).__name__ for s in bad]}."
            )
        counts = Counter(s.name for s in collected)
        duplicates = sorted(name for name, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"duplicate report section(s) {duplicates}: a collision would "
                f"silently overwrite one section's findings with another's."
            )
        return cls(
            spec=spec,
            cfg=cfg,
            sections=collected,
            thresholds=thresholds or VerdictThresholds(),
        )

    def by_name(self) -> dict[str, SectionLike]:
        """Sections keyed by name (unique — ``assemble`` guarantees it)."""
        return {s.name: s for s in self.sections}

    def validate_all_mandatory_present(self) -> None:
        """Raise unless every mandatory section is present (enforcement #2)."""
        present = {s.name for s in self.sections}
        missing = [name for name in MANDATORY_SECTIONS if name not in present]
        if missing:
            raise ValueError(
                f"factor evaluation report is missing mandatory section(s) "
                f"{missing} for {self.spec.factor_id!r}. Every section in "
                f"{MANDATORY_SECTIONS} must be produced, or explicitly returned as "
                f"Skipped(name, reason) — a silently missing section is not allowed."
            )

    def with_verdict(
        self, thresholds: VerdictThresholds | None = None
    ) -> FactorEvalReport:
        """Return a NEW report carrying the verdict (this object is unchanged).

        Validates FIRST, so "a verdicted report is a complete report" holds for
        EVERY caller — not just the ones that remember to call
        :meth:`validate_all_mandatory_present` themselves. Judging a factor on a
        report with silently absent sections is exactly the self-deception this
        contract exists to stop, and such a record would poison the cross-run
        factor library (design §11). This is the same enforcement layer #2, just
        made unbypassable.
        """
        self.validate_all_mandatory_present()
        # PRE-REGISTERED criteria win (design §6, v0.6): a bar declared on the frozen
        # EvalConfig BEFORE evaluate() ran is pre-registered by construction. An
        # explicit runtime `thresholds` override (or the assembled default) is used
        # only when the config declared none. The chosen source is stamped so the
        # report cannot hide which bar it was judged against.
        if self.cfg.success_criteria is not None:
            thr = self.cfg.success_criteria
            source = "declared"
        else:
            thr = thresholds or self.thresholds
            source = "default"
        inputs = extract_verdict_inputs(self.spec, self.cfg, self.by_name())
        return replace(
            self,
            verdict=decide_verdict(inputs, thr),
            thresholds=thr,
            criteria_source=source,
        )

    def require_verdict(self) -> VerdictResult:
        """The verdict, or a readable error — a report may not dodge it."""
        if self.verdict is None:
            raise ValueError(
                f"factor evaluation report for {self.spec.factor_id!r} has no "
                f"verdict: call with_verdict() first. Publishing a report without "
                f"a verdict is exactly what this contract forbids."
            )
        return self.verdict

    # -- exports ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Machine-readable record, stable key order (the cross-run record)."""
        return report_to_dict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Deterministic JSON (sorted keys) of :meth:`to_dict`."""
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False
        )

    def render(self) -> str:
        """Deterministic Markdown: the 10 fixed sections of design §5."""
        return render_report(self, MANDATORY_SECTIONS)


__all__ = [
    "MANDATORY_SECTIONS",
    "SCHEMA_VERSION",
    "VERDICT_KEYS",
    "FactorEvalReport",
    "Section",
    "SectionLike",
    "Skipped",
    "extract_verdict_inputs",
]
