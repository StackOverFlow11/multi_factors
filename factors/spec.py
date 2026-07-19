"""``FactorSpec``: the factor-INTRINSIC half of the factor-evaluation contract.

Two objects together form the provenance an evaluator requires (design doc
``tmp/design/factor_eval_contract_v0.1.md`` §1):

  * :class:`FactorSpec` (HERE) — identity + PIT contract + declared inputs.
    It describes the factor itself, travels with it, and barely ever changes.
  * ``analytics.eval.EvalConfig`` — the per-run parameters + honesty flags.

WHY THIS LIVES IN ``factors/`` (layering invariant #3, 分层解耦):
    the project layering is ``data -> universe -> factors -> alpha -> portfolio
    -> runtime -> analytics``. ``Factor`` must HOLD its spec (the base class
    enforces it at class-definition time), so if ``FactorSpec`` lived in
    ``analytics/`` then ``factors`` would import ``analytics`` — an UPWARD
    dependency. ``analytics`` importing ``factors.spec`` is downstream->upstream
    and therefore fine. This module deliberately imports nothing from the
    project (no pandas either): it is a pure declaration.

The construction-time validators below are enforcement layer #1 of three (design
§7): declare the hypothesis and the PIT contract, or you cannot even build the
object. ``expected_ic_sign`` may NOT be None — a factor author must commit to a
direction BEFORE the run, so the verdict can be a factual sign check rather than
an after-the-fact story.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# The ONLY basis an intraday factor may declare: the minute tail model's holding
# period runs execution-anchor to execution-anchor, exec(T) -> exec(T_next), and
# is NEVER close-to-close (I5a). Cross-checked in _check_intraday_block.
INTRADAY_RETURN_BASIS = "exec_to_exec"

# How the evaluated forward return is measured.
#   close_to_close -> daily close(t+h)/close(t) - 1 (the daily pipeline)
#   exec_to_exec   -> execution-anchored, e.g. the minute tail model's
#                     exec(T_next)/exec(T) - 1 (NEVER close-to-close; I5a).
RETURN_BASES: tuple[str, ...] = ("close_to_close", INTRADAY_RETURN_BASIS)

# Price-adjustment conventions. The framework front-adjusts (qfq) in memory and
# stores raw; no other basis is supported today, so anything else is an error
# rather than a silently-wrong evaluation.
PRICE_ADJUSTMENTS: tuple[str, ...] = ("qfq",)

# The minute block: all five are required when ``is_intraday`` is True and must
# all be None otherwise. A half-declared intraday contract is the exact failure
# mode this guards (three timestamps kept separate all the way through: signal
# cutoff / execution timestamp / holding period — see runtime/intraday_*).
INTRADAY_FIELDS: tuple[str, ...] = (
    "decision_cutoff",
    "data_lag",
    "session_open",
    "execution_model",
    "execution_window",
)


@dataclass(frozen=True)
class FactorSpec:
    """Identity + PIT contract + declared inputs of ONE factor (immutable).

    Attributes
    ----------
    factor_id : canonical unique name; equals the ``Factor.name`` panel column.
    version : bump on any redefinition, so cross-run records stay comparable.
    description : one line — what does this factor measure?
    expected_ic_sign : +1 or -1, the hypothesis, FIXED BEFORE THE RUN. Drives the
        OOS sign check in the verdict. **None is forbidden** by design.
    is_intraday : whether the factor is derived from intraday bars; decides
        whether the minute block is required.
    forward_return_horizon : h > 0, the horizon (in evaluation periods) the
        factor claims to predict; the IC is aligned to it.
    return_basis : one of :data:`RETURN_BASES`.
    input_fields : the panel columns the factor actually reads, so an evaluator
        can check availability + coverage instead of guessing.
    price_adjust : one of :data:`PRICE_ADJUSTMENTS`.
    family : orthogonality grouping (momentum/value/lowvol/microstructure/...).
    min_history_bars : leading warm-up bars that are NaN by construction, so the
        evaluator does not charge the factor for its own warm-up window.
    decision_cutoff, data_lag, session_open, execution_model, execution_window :
        the intraday block (see :data:`INTRADAY_FIELDS`).
    """

    factor_id: str
    version: str
    description: str
    expected_ic_sign: int
    is_intraday: bool
    forward_return_horizon: int
    return_basis: str
    input_fields: tuple[str, ...]
    price_adjust: str = "qfq"
    family: str | None = None
    min_history_bars: int = 0
    decision_cutoff: str | None = None
    data_lag: str | None = None
    session_open: str | None = None
    execution_model: str | None = None
    execution_window: str | None = None

    def __post_init__(self) -> None:
        self._check_identity()
        self._check_hypothesis()
        self._check_measurement()
        self._check_inputs()
        self._check_intraday_block()

    # -- validators (enforcement layer #1) --------------------------------

    def _check_identity(self) -> None:
        for field_name in ("factor_id", "version", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"FactorSpec.{field_name} must be a non-empty string; got "
                    f"{value!r}."
                )

    def _check_hypothesis(self) -> None:
        sign = self.expected_ic_sign
        # Two Python gotchas guarded at once: ``True`` is an int subclass and
        # ``1.0 == 1``, so both would sneak past a bare ``in (1, -1)`` and land a
        # non-int in the declared-int field (and in the exported record).
        if isinstance(sign, bool) or not isinstance(sign, int) or sign not in (1, -1):
            raise ValueError(
                f"FactorSpec.expected_ic_sign must be +1 or -1 (never None): the "
                f"factor author must commit to a direction BEFORE the run so the "
                f"verdict stays a factual sign check. Got {sign!r} for "
                f"{self.factor_id!r}."
            )
        if not isinstance(self.is_intraday, bool):
            raise ValueError(
                f"FactorSpec.is_intraday must be a bool; got {self.is_intraday!r}."
            )

    def _check_measurement(self) -> None:
        horizon = self.forward_return_horizon
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                f"FactorSpec.forward_return_horizon must be a positive int (the "
                f"horizon the factor claims to predict); got {horizon!r} for "
                f"{self.factor_id!r}."
            )
        if self.return_basis not in RETURN_BASES:
            raise ValueError(
                f"FactorSpec.return_basis must be one of {RETURN_BASES}; got "
                f"{self.return_basis!r} for {self.factor_id!r}."
            )
        if self.price_adjust not in PRICE_ADJUSTMENTS:
            raise ValueError(
                f"FactorSpec.price_adjust must be one of {PRICE_ADJUSTMENTS} (the "
                f"only supported basis today); got {self.price_adjust!r} for "
                f"{self.factor_id!r}."
            )
        bars = self.min_history_bars
        if isinstance(bars, bool) or not isinstance(bars, int) or bars < 0:
            raise ValueError(
                f"FactorSpec.min_history_bars must be a non-negative int; got "
                f"{bars!r} for {self.factor_id!r}."
            )
        if self.family is not None and (
            not isinstance(self.family, str) or not self.family.strip()
        ):
            raise ValueError(
                f"FactorSpec.family must be None or a non-empty string; got "
                f"{self.family!r} for {self.factor_id!r}."
            )

    def _check_inputs(self) -> None:
        fields = self.input_fields
        # A bare string would silently become a tuple of single characters.
        if isinstance(fields, str) or not isinstance(fields, Sequence):
            raise ValueError(
                f"FactorSpec.input_fields must be a sequence of panel column "
                f"names (not a bare string); got {fields!r} for {self.factor_id!r}."
            )
        normalized = tuple(fields)
        if not normalized:
            raise ValueError(
                f"FactorSpec.input_fields must be non-empty: an evaluator checks "
                f"availability + coverage of the columns the factor reads "
                f"({self.factor_id!r})."
            )
        bad = [f for f in normalized if not isinstance(f, str) or not f.strip()]
        if bad:
            raise ValueError(
                f"FactorSpec.input_fields entries must be non-empty strings; got "
                f"{bad!r} for {self.factor_id!r}."
            )
        # Store a tuple even when handed a list, so the frozen spec stays
        # immutable + hashable.
        object.__setattr__(self, "input_fields", normalized)

    def _check_intraday_block(self) -> None:
        present = [f for f in INTRADAY_FIELDS if getattr(self, f) is not None]
        if self.is_intraday:
            missing = [f for f in INTRADAY_FIELDS if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"FactorSpec({self.factor_id!r}) is_intraday=True requires the "
                    f"whole minute block {INTRADAY_FIELDS}; missing {missing}. A "
                    f"half-declared intraday contract (cutoff without execution "
                    f"model, ...) is exactly what this guard rejects."
                )
            # Present-but-blank is the SAME half-declared contract as missing: a
            # block of empty strings declares nothing while passing a None check.
            blank = [
                f
                for f in INTRADAY_FIELDS
                if not isinstance(getattr(self, f), str) or not getattr(self, f).strip()
            ]
            if blank:
                raise ValueError(
                    f"FactorSpec({self.factor_id!r}) minute block entries must be "
                    f"non-empty strings; got blank/non-string {blank}. An empty "
                    f"cutoff or execution window declares nothing — it is the same "
                    f"half-declared intraday contract as omitting the field."
                )
            if self.return_basis != INTRADAY_RETURN_BASIS:
                raise ValueError(
                    f"FactorSpec({self.factor_id!r}) is_intraday=True requires "
                    f"return_basis={INTRADAY_RETURN_BASIS!r}; got "
                    f"{self.return_basis!r}. An intraday factor's holding period is "
                    f"execution-anchored (exec(T) -> exec(T_next)) and is NEVER "
                    f"close-to-close (I5a)."
                )
            # NOTE the CONVERSE is deliberately NOT enforced: exec_to_exec does
            # NOT imply is_intraday. A DAILY-computed signal executed on the
            # minute tail is a legitimate combination — it is exactly this
            # project's I5a/I5b path (a daily factor decided at 14:50, filled at
            # 14:51, held exec-to-exec). Locking the converse in would reject it.
            # Please do not "tighten" this into an iff.
        elif present:
            raise ValueError(
                f"FactorSpec({self.factor_id!r}) is_intraday=False requires the "
                f"whole minute block to be None; got {present} set. A daily factor "
                f"declaring an execution window would misdescribe its PIT contract."
            )


__all__ = [
    "FactorSpec",
    "RETURN_BASES",
    "PRICE_ADJUSTMENTS",
    "INTRADAY_FIELDS",
    "INTRADAY_RETURN_BASIS",
]
