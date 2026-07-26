"""Which exported fields lose text to the 200-char cap — asserted, not assumed.

WHY A CENSUS AND NOT A SINGLE-FIELD ASSERTION
----------------------------------------------
The intraday-cutoff correction was first written into ``FactorSpec.description``
and was cut out of the JSON by ``sanitize_payload``'s ``MAX_VALUE_CHARS`` cap. The
obvious guard — "assert the correction survives" — would have confirmed only the
thing already known. This repo has the lesson written down from #82: *a guard that
only scans the file you already fixed can only confirm what you already know*, and
widening that scan immediately produced six more instances.

So this file asserts the WHOLE truncated-field set of the exported record, not one
field. Anything new joining the set fails here; anything leaving it fails here too
(a fix must update the record, which is the point).

WHAT IT FOUND
-------------
Six fields are truncated across the shipped corpus, and none of them is noise —
every one is an explanatory caveat cut mid-sentence:

* ``spec.description`` — 44/44 artifacts;
* ``section[return_risk].payload.monotonicity_spearman_by_date_ci_note`` — 44/44;
* ``section[oos_generalization].payload.monotonicity_reversed_status`` — 44/44,
  cut at "This evaluator scores ONE cell and cannot ...[truncated]", i.e. the half
  that states what the evaluator CANNOT do is the half that is gone;
* ``section[caveats].payload.multiple_testing_note`` — 44/44;
* ``section[purity].payload.vif_status`` — 22/44;
* ``section[data_coverage].payload.forward_return_source`` — 20/44.

This predates the correctness fix (the frozen #79 baseline has it too), so it is
recorded here rather than fixed here — see the PR notes for the cost comparison.
``corrections`` is the one field this branch is responsible for, and it must NEVER
appear in the set.

WHAT IT ALSO RULED OUT
----------------------
Not every long string is at risk. ``verdict.reasons``, the per-axis reasons and the
section ``note`` fields go through ``sanitize_text`` ONLY — no cap — as proved on
disk by a 401-char reason and a 1263-char note surviving whole. A "reasons are 16
characters from the cap" alarm is therefore a false one: that path has no cap to
hit. The genuinely close call is ``coverage_bias_bad_vwap`` at 158 of 200.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from analytics.eval.render import MAX_VALUE_CHARS

MARKER = "...[truncated]"

#: The truncated-field set of the shipped corpus, RECORDED so a seventh cannot
#: join quietly and a fix cannot land without updating this. Field names are the
#: normalized paths produced by :func:`_leaves`.
KNOWN_TRUNCATED_FIELDS: frozenset[str] = frozenset(
    {
        "spec.description",
        "section[return_risk].payload.monotonicity_spearman_by_date_ci_note",
        "section[oos_generalization].payload.monotonicity_reversed_status",
        "section[caveats].payload.multiple_testing_note",
        "section[purity].payload.vif_status",
        "section[data_coverage].payload.forward_return_source",
    }
)

#: Fields this branch is responsible for. A correction that cannot survive export
#: is the exact defect this PR exists to fix, so it is called out separately from
#: the inherited set rather than folded into it.
MUST_NEVER_TRUNCATE: frozenset[str] = frozenset({"corrections"})

_REPORTS = pathlib.Path("artifacts/reports")


def _leaves(obj, path: str = ""):
    """Every leaf string with a stable, index-free path.

    Section payloads are keyed by section NAME rather than list position, so the
    recorded set does not shift when the canonical section order changes.
    """
    if isinstance(obj, dict):
        name = obj.get("name")
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            if name and key == "payload":
                child = f"section[{name}].payload"
            yield from _leaves(value, child)
    elif isinstance(obj, list):
        for value in obj:
            yield from _leaves(value, path)
    elif isinstance(obj, str):
        yield path, obj


def _shipped_reports() -> list[pathlib.Path]:
    return sorted(_REPORTS.glob("eval_*.json")) if _REPORTS.is_dir() else []


def _census(paths) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        hit = {field for field, value in _leaves(doc) if value.endswith(MARKER)}
        for field in hit:
            counts[field] = counts.get(field, 0) + 1
    return counts


requires_corpus = pytest.mark.skipif(
    not _shipped_reports(),
    reason="no eval_*.json on disk (artifacts/ is gitignored and run-generated)",
)


@requires_corpus
def test_the_truncated_field_set_is_exactly_the_recorded_one():
    """Reads the JSON FROM DISK — the exported bytes, not a rebuilt object.

    Asserting against a freshly constructed report would test the constructor and
    could pass while the written file was wrong; the failure being guarded is
    "what a machine consumer opens is missing text", which only the file can show.
    """
    census = _census(_shipped_reports())
    got = frozenset(census)
    unexpected = got - KNOWN_TRUNCATED_FIELDS
    assert not unexpected, (
        f"NEW truncated field(s) in the exported record: {sorted(unexpected)} "
        f"(counts {[(f, census[f]) for f in sorted(unexpected)]}). Something "
        f"load-bearing is being cut at {MAX_VALUE_CHARS} chars."
    )
    fixed = KNOWN_TRUNCATED_FIELDS - got
    assert not fixed, (
        f"field(s) no longer truncated: {sorted(fixed)} — good, but update "
        f"KNOWN_TRUNCATED_FIELDS so the record keeps matching reality."
    )


@requires_corpus
def test_the_correction_carrier_is_never_truncated_on_disk():
    """The one field this branch owns. Separate test, separate reason to fail."""
    census = _census(_shipped_reports())
    offending = MUST_NEVER_TRUNCATE & frozenset(census)
    assert not offending, (
        f"{sorted(offending)} was truncated in the exported record — the "
        f"superseded-values disclosure is exactly what must not be cut."
    )


@requires_corpus
def test_a_corrected_factors_artifact_carries_the_whole_declaration_on_disk():
    """End-to-end on the shipped bytes: parse the file, compare to the spec.

    The comparison is against the DECLARED object, so a reworded field passes and
    a shortened one does not — the property is "whole", not "contains a phrase".
    """
    from factors.registry import build

    declared = [c.as_dict() for c in build("jump_amount_corr_20").spec.corrections]
    assert declared, "the factor declares no correction — this test is vacuous"

    checked = 0
    for path in _shipped_reports():
        if "jump_amount_corr" not in path.name:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["corrections"] == declared, f"{path.name} lost the declaration"
        checked += 1
    assert checked >= 4, f"expected the 4 jump artifacts, checked {checked}"


@requires_corpus
def test_the_uncapped_paths_really_are_uncapped():
    """Rules out the false alarm: reasons/notes are not near any cap.

    ``verdict.reasons`` and section ``note`` go through ``sanitize_text`` only.
    Proved on the corpus rather than by reading the code, because "which strings
    are capped" is precisely what was wrong the first time.
    """
    longest: dict[str, int] = {}
    for path in _shipped_reports():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for field, value in _leaves(doc):
            if not value.endswith(MARKER):
                longest[field] = max(longest.get(field, 0), len(value))
    assert longest.get("verdict.reasons", 0) > MAX_VALUE_CHARS
    assert longest.get("sections.note", 0) > MAX_VALUE_CHARS
