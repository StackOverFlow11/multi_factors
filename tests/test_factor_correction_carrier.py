"""A correction must survive INTO the machine-readable record, whole.

WHAT WENT WRONG, AND WHY A TEST HAD TO EXIST
--------------------------------------------
The intraday-cutoff correctness fix first wrote its "these values supersede
previously published ones" disclosure as prose appended to
``FactorSpec.description``. The Markdown carried it. The dashboard carried it.
The JSON did not: ``analytics.eval.render.sanitize_payload`` caps every exported
string at ``MAX_VALUE_CHARS`` (200) and appends ``...[truncated]``, and the
disclosure sat past character 200.

That cap is GENERIC, not a quirk of one path — measured across the 44 shipped
eval JSONs, 44/44 carry a truncated ``spec.description``. So the JSON, the copy a
summary layer reads, showed restated numbers with no indication that they
replaced a defective run. A report that fails to state the thing about its own
provenance that it must state is the same shape as the factor defect that
produced the correction: hence a guard, not a fixed sentence.

WHY THESE ASSERTIONS AND NOT SUBSTRING MATCHES
-----------------------------------------------
D5b lost a guard to exactly that: a test that matched tokens in a function's
SOURCE stayed green when the tokens were moved into a comment. So nothing here
greps rendered text for a phrase. Instead each test round-trips the DECLARED
structured object through the real export path and asserts EQUALITY with what the
factor declared — a lexical rewording cannot pass, and a truncation cannot pass,
because the compared thing is the object itself.

The length assertions are the teeth for the specific failure: the fields are
deliberately longer than the cap, so "it round-tripped" is only provable while
the carrier is genuinely outside the capped path.
"""

from __future__ import annotations

import json

import pytest

from analytics.eval.figures import _correction_marker
from analytics.eval.render import (
    MAX_VALUE_CHARS,
    corrections_record,
    report_to_dict,
    sanitize_payload,
)
from factors.compute.minute.jump_amount_corr import JumpAmountCorrFactor
from factors.registry import build as build_factor
from factors.spec import (
    CORRECTION_FIELD_MAX_CHARS,
    FactorCorrection,
    FactorSpec,
    PanelField,
)

_TRUNCATION_MARKER = "[truncated]"


def _long(prefix: str) -> str:
    """A field comfortably past the payload cap (so the cap would show if hit)."""
    return prefix + " " + "x" * (MAX_VALUE_CHARS + 50)


def _correction(**overrides) -> FactorCorrection:
    base = dict(
        from_version="1.0",
        to_version="1.1",
        date="2026-07-25",
        defect=_long("the defect was"),
        effect=_long("the effect was"),
        superseded=_long("superseded artifacts are"),
    )
    base.update(overrides)
    return FactorCorrection(**base)


# --------------------------------------------------------------------------- #
# FactorCorrection: a structured fact, validated at authoring time
# --------------------------------------------------------------------------- #
def test_every_field_is_required():
    for name in FactorCorrection._FIELDS:
        with pytest.raises(ValueError, match=name):
            _correction(**{name: "   "})


def test_over_length_raises_rather_than_trimming():
    """The carrier must not be able to lose what it carries.

    Trimming here would rebuild the very defect this field exists to fix, so the
    bound is enforced LOUDLY at construction instead of quietly at export.
    """
    with pytest.raises(ValueError, match="RAISES instead of trimming"):
        _correction(defect="y" * (CORRECTION_FIELD_MAX_CHARS + 1))


def test_date_must_be_an_iso_date():
    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        _correction(date="July 2026")


def test_spec_rejects_a_correction_that_lands_on_another_version():
    """``spec.version`` alone must discriminate a stored artifact.

    A reader holding ONE json should be able to tell which side of the correction
    it is on without going to find the other one.
    """
    with pytest.raises(ValueError, match="to_version"):
        _spec(version="2.0", corrections=(_correction(to_version="1.1"),))


def test_spec_rejects_free_text_in_place_of_a_correction():
    with pytest.raises(ValueError, match="FactorCorrection"):
        _spec(corrections=("we fixed the cutoff",))


def test_spec_rejects_a_bare_correction_not_in_a_tuple():
    with pytest.raises(ValueError, match="not a bare one"):
        _spec(corrections=_correction())


def _spec(**overrides) -> FactorSpec:
    base = dict(
        factor_id="fixture_factor",
        version="1.1",
        description="fixture",
        expected_ic_sign=1,
        is_intraday=False,
        forward_return_horizon=1,
        return_basis="close_to_close",
        input_fields=("close",),
        requires=(PanelField("close", source="market_daily"),),
        adjustment="returns_invariant",
        overnight_boundary="none",
    )
    base.update(overrides)
    return FactorSpec(**base)


# --------------------------------------------------------------------------- #
# The capped path is real (so the tests below are not shadow-boxing)
# --------------------------------------------------------------------------- #
def test_the_generic_payload_cap_would_have_eaten_this():
    """Pins the mechanism the carrier routes around.

    If this ever stops truncating, the correction field is no longer load-bearing
    for THIS reason and the docstring above needs rewriting rather than the code.
    """
    long_value = "z" * (MAX_VALUE_CHARS + 500)
    got = sanitize_payload({"note": long_value})["note"]
    assert got.endswith(_TRUNCATION_MARKER)
    # the surviving payload is exactly the cap; the marker rides on top of it
    assert got[: -len(_TRUNCATION_MARKER)].rstrip(".") == "z" * MAX_VALUE_CHARS


def test_a_correction_written_into_the_description_would_be_truncated():
    """The exact mistake this file exists to prevent, demonstrated.

    Shows the failure mode directly rather than describing it: prose past the cap
    inside ``description`` does not survive ``vars(spec)`` export.
    """
    spec = _spec(description="definition. " + _long("CORRECTION:"))
    exported = sanitize_payload(vars(spec))["description"]
    assert exported.endswith(_TRUNCATION_MARKER)
    assert "CORRECTION" in spec.description  # it IS in the object ...
    assert exported.count("x") < spec.description.count("x")  # ... and cut on export


# --------------------------------------------------------------------------- #
# Round-trip: declared object == what a machine consumer reads back
# --------------------------------------------------------------------------- #
def test_corrections_record_round_trips_the_declared_object_exactly():
    correction = _correction()
    got = corrections_record(_spec(corrections=(correction,)))
    assert got == [correction.as_dict()]
    assert all(_TRUNCATION_MARKER not in v for v in got[0].values())


def test_a_factor_with_no_corrections_exports_an_empty_list_not_a_blank_entry():
    """"No correction declared" and "known correct" are different statements."""
    assert corrections_record(_spec()) == []
    assert _correction_marker(_spec()) == ""


def _report_for(spec):
    """A real, verdict-bearing report carrying ``spec`` — the actual export path.

    Reuses the contract suite's section fixtures rather than hand-rolling eight
    mandatory sections here; the subject under test is the export, not assembly.
    """
    from analytics.eval.report import FactorEvalReport
    from tests.test_factor_eval_contract import _adopt_grade_sections
    from tests.test_factor_eval_contract import _cfg as _contract_cfg

    return FactorEvalReport.assemble(
        spec, _contract_cfg(), _adopt_grade_sections()
    ).with_verdict()


def test_the_json_export_carries_the_declaration_at_top_level_untruncated():
    """The end-to-end property: parse the exported JSON, compare to the object.

    Uses the REAL ``report_to_dict`` + ``json.dumps``/``loads`` round trip on a
    real report, so a change that reroutes corrections back through the capped
    path — or drops the key — fails here.
    """
    correction = _correction()
    spec = _spec(corrections=(correction,))
    doc = json.loads(json.dumps(report_to_dict(_report_for(spec))))

    assert doc["corrections"] == [correction.as_dict()]
    for entry in doc["corrections"]:
        for value in entry.values():
            assert _TRUNCATION_MARKER not in value
    # ... and the teeth: the fields really are past the cap, so surviving whole
    # is only possible outside the capped path.
    assert max(len(v) for v in correction.as_dict().values()) > MAX_VALUE_CHARS
    # The same report's spec block IS still capped — the cap was routed around,
    # not removed (removing it would let a stray panel repr into the artifact).
    assert doc["spec"]["description"].endswith(_TRUNCATION_MARKER) or len(
        spec.description
    ) <= MAX_VALUE_CHARS


def test_the_shipped_factors_declaration_survives_the_real_export():
    """The SHIPPED spec (not a fixture) round-trips through the real exporter."""
    spec = JumpAmountCorrFactor().spec
    declared = [c.as_dict() for c in spec.corrections]
    assert declared, "the shipped factor declares no correction — test is vacuous"
    doc = json.loads(json.dumps(report_to_dict(_report_for(spec))))
    assert doc["corrections"] == declared
    assert max(len(v) for v in declared[0].values()) > MAX_VALUE_CHARS


# --------------------------------------------------------------------------- #
# The shipped factor declares it, and the registry-built one agrees
# --------------------------------------------------------------------------- #
def test_the_corrected_factor_declares_its_correction_through_the_registry():
    """The path the runners actually use must carry it too.

    A declaration only reachable by importing the class directly would miss every
    artifact, since the runners build through the registry.
    """
    spec = build_factor("jump_amount_corr_20").spec
    assert spec.version == "1.1"
    assert len(spec.corrections) == 1
    correction = spec.corrections[0]
    assert (correction.from_version, correction.to_version) == ("1.0", "1.1")
    assert _correction_marker(spec) == "CORRECTED v1.0 -> v1.1"


def test_report_to_dict_emits_the_key_even_when_nothing_was_corrected():
    """Present on EVERY report, empty or not — an absent key is ambiguous.

    A consumer must be able to ask "was anything corrected?" without having to
    distinguish "no corrections" from "an older contract that had no such key".
    """
    doc = report_to_dict(_report_for(_spec()))
    assert doc["corrections"] == []


def test_the_markdown_and_the_json_cannot_disagree():
    """Both surfaces are built from the SAME tuple, so they move together.

    Asserted as a relationship between the two renderings of one object, not as
    a fixed phrase in either: a reworded label stays green, a correction present
    in one surface and missing from the other does not.
    """
    correction = _correction()
    report = _report_for(_spec(corrections=(correction,)))
    doc = report_to_dict(report)
    markdown = report.render()

    assert len(doc["corrections"]) == 1
    # every field of the declared correction appears whole in the rendered
    # provenance box (the row composes them; it never re-words them)
    for name in ("defect", "effect", "superseded"):
        assert getattr(correction, name) in markdown
    assert _TRUNCATION_MARKER not in markdown

    clean = _report_for(_spec())
    assert report_to_dict(clean)["corrections"] == []
    assert "CORRECTION" not in clean.render()
