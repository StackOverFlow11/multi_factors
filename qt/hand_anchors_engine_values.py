"""Daily hand-anchor engine comparison — VERIFY-ONLY since D5 C6.

``qt.hand_anchors_d2`` / ``qt.hand_anchor_rows`` hand-compute momentum_20 /
reversal_20 / liquidity_20 / overnight_mom_20 anchor rows WITHOUT importing the
engine and leave them in ``hand_anchors_d2.json`` under
``daily_pending_engine``. This companion used to be the one place ALLOWED to
import the engine: it rebuilt the freeze data plane (same config, cache-only),
ran the ops-rewritten factor classes, and wrote each comparison back into the
JSON under ``daily_engine_compared`` / ``all_ok_daily``.

REGENERATION IS RETIRED (owner ruling 2026-07-28, D5 C6)
---------------------------------------------------------
The rebuild imported a legacy runner's precondition check and re-derived the
whole data plane behind it; C6 deletes that runner. Two honest notes about the
scope of this particular retirement, because overstating it would be the very
defect this repository keeps catching:

* the dependency on the legacy runners was THIN here — one precondition helper —
  not the eleven private loaders that ``qt.panel_freeze`` called. Retiring this
  tool therefore gives up a capability that was not, on its own, doomed;
* the four daily factor classes themselves are untouched and keep their own
  tests. What is retired is this hand-vs-engine COMPARISON, not the factors.

What ``--verify`` does
----------------------
It re-derives the recorded comparison: for every row of
``daily_engine_compared`` it recomputes the relative difference from the
recorded ``hand`` / ``engine`` values at the same tolerance and checks the
recorded ``rel_diff`` / ``ok`` / ``all_ok_daily`` against what those values
actually imply, and checks that every pending anchor got a comparison. So a
record edited to turn a mismatch into a pass does not survive; a record whose
NUMBERS were never produced cannot be re-created here at all.

When the record is ABSENT the exit code is non-zero and the wording is "NOT
VERIFIED", not "FAILED" — absence of evidence is not evidence of failure, but a
verification command that reports success without verifying anything is the
empty-reconciliation shape this repository has already committed once.

Run::

    python -m qt.hand_anchors_engine_values --verify
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from qt.hand_anchors_d2 import OUT_JSON, TOL
from qt.panel_freeze import RegenerationRetiredError, retirement_message

#: The four ops-rewritten daily factors this comparison covers.
DAILY_FACTORS = ("momentum_20", "reversal_20", "liquidity_20", "overnight_mom_20")
#: Fields every recorded comparison row must carry.
COMPARED_ROW_FIELDS = ("factor_id", "class", "date", "symbol", "hand", "engine",
                       "rel_diff", "ok")


def relative_difference(hand: float, engine: float) -> float:
    """The recorded comparison's own rule, restated once and used by the checker."""
    if math.isfinite(hand) and math.isfinite(engine):
        denominator = max(abs(hand), abs(engine))
        return abs(hand - engine) / denominator if denominator > 0 else 0.0
    if math.isnan(hand) and math.isnan(engine):
        return 0.0
    return float("inf")


@dataclass(frozen=True)
class AnchorRecordVerification:
    """Outcome of checking the recorded daily hand-anchor comparison."""

    path: Path
    n_compared: int
    n_pending: int
    problems: tuple[str, ...]
    recorded: bool

    @property
    def ok(self) -> bool:
        return self.recorded and not self.problems


def verify_recorded_comparison(path: Path | str = OUT_JSON) -> AnchorRecordVerification:
    """Re-derive the recorded daily hand-anchor comparison from its own numbers."""
    path = Path(path)
    problems: list[str] = []
    if not path.exists():
        return AnchorRecordVerification(
            path=path,
            n_compared=0,
            n_pending=0,
            problems=(f"anchor record not found: {path}",),
            recorded=False,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    pending = payload.get("daily_pending_engine", []) or []
    compared = payload.get("daily_engine_compared", []) or []
    if not compared:
        return AnchorRecordVerification(
            path=path,
            n_compared=0,
            n_pending=len(pending),
            problems=(
                f"{path.name} carries no daily_engine_compared rows "
                f"({len(pending)} anchor(s) still pending). The comparison that "
                "would fill them in is retired, so this record cannot be "
                "completed from this tree.",
            ),
            recorded=False,
        )

    n_bad = 0
    for position, row in enumerate(compared):
        missing = [field for field in COMPARED_ROW_FIELDS if field not in row]
        if missing:
            problems.append(f"row {position}: missing field(s) {missing}")
            continue
        label = f"{row['factor_id']} {row['date']} {row['symbol']}"
        if row["factor_id"] not in DAILY_FACTORS:
            problems.append(f"row {position}: unexpected factor id {row['factor_id']!r}")
        expected_rel = relative_difference(float(row["hand"]), float(row["engine"]))
        recorded_rel = float(row["rel_diff"])
        if not _same_number(expected_rel, recorded_rel):
            problems.append(
                f"{label}: recorded rel_diff {recorded_rel!r} but hand/engine imply "
                f"{expected_rel!r}"
            )
        expected_ok = expected_rel <= TOL
        if bool(row["ok"]) != expected_ok:
            problems.append(
                f"{label}: recorded ok={row['ok']!r} but rel {expected_rel!r} vs "
                f"tolerance {TOL!r} implies {expected_ok}"
            )
        n_bad += 0 if expected_ok else 1

    recorded_all_ok = payload.get("all_ok_daily")
    if recorded_all_ok is None:
        problems.append("all_ok_daily is absent from the record")
    elif bool(recorded_all_ok) != (n_bad == 0):
        problems.append(
            f"all_ok_daily is recorded as {recorded_all_ok!r} but the rows imply "
            f"{n_bad == 0} ({n_bad} mismatching row(s))"
        )
    elif n_bad:
        problems.append(
            f"{n_bad} recorded row(s) exceed the tolerance — the daily anchors did "
            "not agree with the engine."
        )

    compared_keys = {(r.get("factor_id"), r.get("date"), r.get("symbol")) for r in compared}
    for row in pending:
        key = (row.get("factor_id"), row.get("date"), row.get("symbol"))
        if key not in compared_keys:
            problems.append(f"pending anchor never compared: {key}")

    return AnchorRecordVerification(
        path=path,
        n_compared=len(compared),
        n_pending=len(pending),
        problems=tuple(problems),
        recorded=True,
    )


def _same_number(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def run_engine_comparison(*args, **kwargs):
    """RETIRED (D5 C6). Always raises :class:`RegenerationRetiredError`."""
    raise RegenerationRetiredError(
        retirement_message(
            "python -m qt.hand_anchors_engine_values",
            "the daily hand-anchor engine comparison",
            "The rebuild imported a legacy eval runner's precondition check and "
            "re-derived the whole data plane behind it; C6 deletes that runner. "
            "The four daily factor classes are untouched — what is retired is "
            "this comparison, not the factors.",
            "it re-derives the RECORDED comparison in hand_anchors_d2.json from "
            "its own hand/engine numbers. It cannot produce numbers that were "
            "never recorded, and says so rather than reporting success.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", default=str(OUT_JSON))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive the recorded daily hand-anchor comparison from its own numbers",
    )
    args = parser.parse_args(argv)

    if not args.verify:
        try:
            run_engine_comparison()
        except RegenerationRetiredError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = verify_recorded_comparison(args.record)
    print(
        f"{result.path}: {result.n_compared} recorded comparison(s), "
        f"{result.n_pending} pending anchor(s)"
    )
    for problem in result.problems:
        print(f"  PROBLEM {problem}")
    if not result.recorded:
        print("daily hand-anchor verification: NOT VERIFIED (nothing recorded)")
        return 1
    print("daily hand-anchor verification: " + ("OK" if result.ok else "FAILED"))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(main())
