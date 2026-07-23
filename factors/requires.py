"""``PanelField``: one endpoint-level data requirement of a factor (D1, R25).

A factor declares WHAT data it needs; the availability policy table
(``data/availability_policy.py``, the D0 single source of truth) declares WHEN
that data is visible per view. The two are deliberately kept apart (design
v3.2 §3.1): availability is a property of the ENDPOINT (field-level only for
``market_daily``, R7), never something a factor writes on itself.

Endpoint-only by decision R25: ``source`` must name an endpoint of the
availability policy table. There are NO factor-id references and NO
topological sort here — the whole repo has zero factor-on-factor consumers,
so that machinery stays unbuilt until the first residual-momentum-style
factor is actually approved (design §11).

Validation happens at CONSTRUCTION time (R8 endpoint closure): an undeclared
``source`` has no availability rule, and any fallback for it would be a
field-level lookahead entry point — so it is a readable error, never a
``dict.get`` default. For ``market_daily`` the FIELD is validated too (R7:
availability there is field-level; an unknown field has no rule). For every
other endpoint field-name validity stays the feed's business, exactly as
``data.availability_policy.rule_for`` documents.

Leaf-module discipline: imports only the stdlib and the availability policy
(itself stdlib-only). Never import pandas, feeds, qt, or the registry — the
registry and the D4 materializer sit ABOVE this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.availability_policy import MARKET_DAILY, require_known_source, rule_for


@dataclass(frozen=True, slots=True)
class PanelField:
    """One (field, source-endpoint) requirement, validated at construction.

    Attributes
    ----------
    field : the column/field name the factor reads (e.g. ``"close"``, ``"pe"``).
    source : the availability-policy endpoint publishing it (e.g.
        ``"market_daily"``, ``"daily_basic"``, ``"stk_mins_1min"``). Must be a
        declared endpoint of the policy table — R8 endpoint closure fires HERE,
        at declaration time, not somewhere downstream.
    """

    field: str
    source: str

    def __post_init__(self) -> None:
        for attr in ("field", "source"):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"PanelField.{attr} must be a non-empty string; got {value!r}."
                )
        # R8 endpoint closure: unknown source -> readable error (single source
        # of truth: data.availability_policy.require_known_source).
        require_known_source(self.source)
        # R7: market_daily availability is FIELD-level, so an unknown field has
        # no availability rule -> the policy module's own resolver rejects it.
        if self.source == MARKET_DAILY:
            rule_for(self.source, self.field)


__all__ = ["PanelField"]
