"""run-factor-eval <-> run-registry wiring (D7-PR0): status mapping + book assertion.

D5 C4 deferred the run-registry append to D7's governance surface; this module
is the wiring. Two pieces:

1. THE STATUS MAPPING (:data:`VERDICT_TO_STATUS` + :func:`status_for_run`). A
   run's registry status is derived by an EXPLICIT, code-constant mapping —
   never by a transition machine and never by inferring curation decisions
   from a gate verdict:

   * a BOOK member (``factor_id in book_ids``) records ``book`` regardless of
     verdict — the book is curated, and a single eval run neither promotes
     into it nor demotes out of it;
   * verdict ``Watch``             -> ``watch``;
   * verdict ``Reject``            -> ``exploratory`` — NOT ``retired``.
     Retirement is an explicit HUMAN registry edit (a curated decision to stop
     tracking a factor); a default gate verdict must never be able to derive
     it, or one noisy run would fabricate a retirement;
   * verdict ``INSUFFICIENT-DATA`` -> ``exploratory`` (the run could not tell;
     nothing curated follows from that);
   * verdict ``Adopt``             -> a READABLE ERROR. run-factor-eval declares
     ``is_exploratory=True`` on its EvalConfig, which caps Adopt to Watch
     (``analytics.eval.verdict`` — the exploratory cap), so Adopt is
     unreachable here; and even if a future run lifted the cap, promotion to
     ``book`` is the same kind of human curation decision as retirement, so the
     mapping refuses to derive it instead of silently registering one.

2. THE BOOK-SET ASSERTION (:func:`sync_book_registry`). Design §10 says the
   book is read from the run registry; today the code still reads the frozen
   ``qt.factor_eval_runner.BOOK_IDS`` constant. The two agree by construction
   right now, and this assertion is what keeps them agreeing: at startup the
   runner asserts ``{factor_id: latest status == "book"} == BOOK_IDS``
   (latest-record-wins over the append-only log — a later human ``retired``
   record for a book factor MUST trip the assertion, which any-record
   semantics would hide). A mismatch is a readable error naming both sides
   (same BUG-5 spirit as the config ``factors:`` closure: the file may never
   describe a book the run does not use).

   BOOTSTRAP: an EMPTY registry (zero records — first ever run, or a wiped
   store root, which is regenerable by design) accepts the code-declared
   ``book_ids`` as the declared book and SEEDS it: one ``book`` record per id
   with a note saying where the declaration came from. Thereafter every run
   appends its own record and the assertion governs. A NON-empty registry is
   never silently re-seeded — if it disagrees with ``book_ids`` a human
   aligns the two (append the correcting records, or fix ``BOOK_IDS``).

Layering: this is qt glue — it may import ``analytics.eval.verdict`` (the
verdict labels) and ``factors.*`` (the registry, the store). ``factors.store``
itself never learns about verdicts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from analytics.eval.verdict import ADOPT, INSUFFICIENT_DATA, REJECT, WATCH
from factors import registry as factor_registry
from factors.store import RunRegistry, data_fingerprint, store_key

#: verdict -> run-registry status for NON-book factors (see the module
#: docstring; ``retired`` is deliberately unreachable from ANY verdict).
VERDICT_TO_STATUS: dict[str, str] = {
    WATCH: "watch",
    REJECT: "exploratory",
    INSUFFICIENT_DATA: "exploratory",
}

#: The note stamped on the bootstrap seed records (empty registry only).
BOOTSTRAP_NOTE = (
    "bootstrap: seeded from the code-declared book "
    "(qt.factor_eval_runner.BOOK_IDS — the frozen confirmed trio, lead ruling "
    "Q1) on the first run-factor-eval run against an empty registry; "
    "subsequent runs assert registry == BOOK_IDS instead of re-seeding."
)


def status_for_run(
    factor_id: str, verdict: str, *, book_ids: Sequence[str]
) -> str:
    """The run-registry status one eval run of ``factor_id`` records.

    Book members record ``book`` whatever the verdict (curation is not derived
    from a gate); every other factor maps through :data:`VERDICT_TO_STATUS`.
    An unmapped verdict — including ``Adopt`` — is a readable error, never a
    silent default.
    """
    if factor_id in book_ids:
        return "book"
    try:
        return VERDICT_TO_STATUS[verdict]
    except KeyError:
        extra = (
            " Adopt is unreachable from run-factor-eval (its EvalConfig declares "
            "is_exploratory=True, which caps Adopt to Watch), and promotion to "
            "'book' is a human curation decision the mapping refuses to derive."
            if verdict == ADOPT
            else ""
        )
        raise ValueError(
            f"no run-registry status mapping for verdict {verdict!r} "
            f"(factor {factor_id!r}); the explicit mapping covers "
            f"{sorted(VERDICT_TO_STATUS)}.{extra}"
        ) from None


def current_book_set(records: Iterable[Mapping[str, object]]) -> set[str]:
    """The curated book from an append-only record stream: LATEST record wins.

    The registry is append-only (#74), so a factor's CURRENT status is its
    last line; the book is the set of factor ids whose last line says
    ``book``. Any-record semantics would let a retired book factor keep
    passing the startup assertion — exactly the drift the assertion exists
    to catch.
    """
    latest: dict[str, str] = {}
    for rec in records:
        latest[str(rec["factor_id"])] = str(rec["status"])
    return {fid for fid, status in latest.items() if status == "book"}


def sync_book_registry(registry: RunRegistry, *, book_ids: Sequence[str]) -> str:
    """Seed-or-assert the book at runner startup; returns 'seeded' | 'verified'.

    Empty registry -> seed one ``book`` record per declared id and return
    ``'seeded'``. Non-empty -> assert the registry's current book equals
    ``book_ids`` exactly and return ``'verified'``; a mismatch raises a
    readable error naming the missing/extra ids (a human aligns the two —
    promotion/retirement are explicit registry edits, or ``BOOK_IDS`` is
    wrong and gets fixed in code).
    """
    declared = set(book_ids)
    records = registry.read_all()
    if not records:
        for factor_id in book_ids:
            factor = factor_registry.build(factor_id)
            registry.append_run(
                key=store_key(factor, view="decision", params=None),
                factor=factor,
                status="book",
                fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
                note=BOOTSTRAP_NOTE,
            )
        return "seeded"
    registered = current_book_set(records)
    if registered != declared:
        missing = sorted(declared - registered)
        extra = sorted(registered - declared)
        raise ValueError(
            f"run registry {registry.path} disagrees with the code-declared "
            f"book: registry book={sorted(registered)}, BOOK_IDS="
            f"{sorted(declared)} (missing from registry: {missing}; not in "
            f"BOOK_IDS: {extra}). The book is a curated HUMAN decision: append "
            f"the correcting records to the registry (promotion/retirement are "
            f"explicit edits, never derived from a gate verdict), or fix "
            f"BOOK_IDS in qt/factor_eval_runner.py — the two must agree so the "
            f"file cannot describe a book the run does not use."
        )
    return "verified"


__all__ = [
    "BOOTSTRAP_NOTE",
    "VERDICT_TO_STATUS",
    "current_book_set",
    "status_for_run",
    "sync_book_registry",
]
