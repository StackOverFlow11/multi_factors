"""The factor-evaluation contract's VERSION and its declared identity fields.

The statistical adjudication core (three axes / asymmetric gate / "unknown never
convicts" / N_eff CI / exploratory capped at Watch) is NOT rewritten by the
factor-layer refactor — it is UPGRADED, and this module is the explicit version
statement that upgrade owes (the PR #74 precedent: a change to a frozen contract
is stated, not slipped in).

CONTRACT v1.0 — WHAT CHANGED AND WHY
------------------------------------
v0.9 (the eleven-factor loop's frozen contract) described a factor and one
evaluation of it, but it could not say WHICH INFORMATION SET the factor values
were taken under. That was survivable while every run used one basis; it stopped
being survivable when the same factor started being evaluated on two
(``close_to_close`` and ``exec_to_exec``, PR #79), because two reports then
carried the same verdict word for two different statistical claims.

v1.0 adds exactly two identity fields to ``EvalConfig`` and renders four more off
the ``FactorSpec`` that D1 already made mandatory:

* ``EvalConfig.view`` / ``EvalConfig.return_basis`` — the information-set view and
  the forward-return basis, validated as a LEGAL PAIRING at construction time via
  :func:`data.availability_policy.require_legal_pairing`. A close-view factor
  scored on ``exec_to_exec`` returns is now a readable error, not a convention.
* the provenance box renders ``FactorSpec.requires`` / ``adjustment`` /
  ``overnight_boundary`` / ``lookback_depth`` (contract v1.0/v1.1 of the FACTOR
  spec, PR #86 / #91) as named rows instead of leaving them to the JSON's
  ``vars(spec)`` dump.

WHAT DID NOT CHANGE (and must not, in this step)
-----------------------------------------------
Every decision rule and every threshold. ``VerdictThresholds`` defaults
(``min_abs_icir=0.30``, ``min_incremental_abs_icir=0.15``,
``min_monotonicity_spearman=0.0``, the three-part sample gate) are carried over
BYTE-FOR-BYTE as the UNVALIDATED close-era defaults they have always been
(design v3.2 §十 R24). Re-calibrating a bar is a separate pre-registered run; a
refactor that quietly moved one would make every verdict in this cycle
uninterpretable.

IMPACT SURFACE
--------------
Two new keys in the exported ``eval_config`` block and one new top-level key
(``eval_contract_version``) in every report JSON, plus four new provenance rows in
the Markdown. No metric, no axis input and no threshold moves. The D5 C5
reconciliation registers this as a named, expected artifact drift.
"""

from __future__ import annotations

#: The evaluation contract version this package implements. Bumped ONLY with a
#: written statement of what changed (this module's docstring is that statement).
EVAL_CONTRACT_VERSION = "1.0"

#: The identity fields a v1.0 report must be able to state about itself. Named
#: here once so the config validator, the renderer and the cross-basis summary
#: guard all refer to ONE list instead of three spellings of it.
IDENTITY_FIELDS: tuple[str, ...] = ("view", "return_basis")


def basis_identity_phrase(view: str, return_basis: str) -> str:
    """The ONE sentence that states a report's information set + return basis.

    Author-once (#76/#78/#82): the provenance row, the cross-basis summary header
    and any prose that needs to say which basis a number belongs to COMPOSE this
    string rather than each writing their own. A regex cannot assert that no other
    sentence says this; "there is no other sentence" can.
    """
    return f"view={view} x return_basis={return_basis}"


__all__ = ["EVAL_CONTRACT_VERSION", "IDENTITY_FIELDS", "basis_identity_phrase"]
