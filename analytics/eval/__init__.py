"""Factor-evaluation contract: the frozen acceptance target (PR-A).

Standardizes AND enforces how a factor is evaluated, so "write a new factor ->
evaluate it" cannot route around a set of mandatory sections. Three enforcement
layers (design ``tmp/design/factor_eval_contract_v0.1.md`` §7):

    construction  FactorSpec / EvalConfig __post_init__ + Factor.__init_subclass__
                  -> no hypothesis, no PIT contract, no spec = no object.
    assembly      every section is a Section or an explicit Skipped(reason);
                  validate_all_mandatory_present() raises on a silent hole.
    CI            tests assert the sections and the verdict rule table.

Layering: ``analytics`` is downstream of ``factors``, so importing
``factors.spec`` here is fine; ``FactorSpec`` itself lives in ``factors/``
because the ``Factor`` class must hold it (invariant #3, 分层解耦).

``StandardFactorEvaluator`` + the vectorized eval-IR are PR-B.
"""

from analytics.eval.config import EvalConfig
from analytics.eval.evaluator import EvalIR, FactorEvaluator
from analytics.eval.report import (
    SCHEMA_VERSION,
    FactorEvalReport,
    extract_verdict_inputs,
)
from analytics.eval.sections import (
    MANDATORY_SECTIONS,
    VERDICT_KEYS,
    Section,
    SectionLike,
    Skipped,
)
from analytics.eval.verdict import (
    ADOPT,
    AXIS_FAIL,
    AXIS_INSUFFICIENT_DATA,
    AXIS_NAMES,
    AXIS_NOT_ASSESSED,
    AXIS_PASS,
    AXIS_VERDICTS,
    INSUFFICIENT_DATA,
    REJECT,
    VERDICTS,
    WATCH,
    AxisVerdict,
    VerdictInputs,
    VerdictResult,
    VerdictThresholds,
    decide_verdict,
)

__all__ = [
    "ADOPT",
    "AXIS_FAIL",
    "AXIS_INSUFFICIENT_DATA",
    "AXIS_NAMES",
    "AXIS_NOT_ASSESSED",
    "AXIS_PASS",
    "AXIS_VERDICTS",
    "INSUFFICIENT_DATA",
    "MANDATORY_SECTIONS",
    "REJECT",
    "SCHEMA_VERSION",
    "VERDICTS",
    "VERDICT_KEYS",
    "WATCH",
    "AxisVerdict",
    "EvalConfig",
    "EvalIR",
    "FactorEvalReport",
    "FactorEvaluator",
    "Section",
    "SectionLike",
    "Skipped",
    "VerdictInputs",
    "VerdictResult",
    "VerdictThresholds",
    "decide_verdict",
    "extract_verdict_inputs",
]
