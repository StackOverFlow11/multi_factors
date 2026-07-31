"""Integrity check for the D2 hand-anchor record (``hand_anchors_d2.json``).

The record is the anchors leg's input: ``factor_eval_reconcile.load_anchor_rows``
reads its ``frozen14`` rows and compares service values against the hand-computed
ones. Until D5 C6 nothing checked it at all, and it is the ONE file in
``artifacts/refactor_baseline/`` that no manifest covered -- while also being the
one file in there that demonstrably got overwritten (2026-07-25, by a
``qt.hand_anchor_rows`` rerun, which is how the engine comparison was lost).

Why a plain sha256 rather than the panels' canonical-content hash: the panels are
parquet, where writer metadata makes byte equality too strict to be the
authority. This is JSON written by one writer; its bytes ARE canonical, exactly
as for the frozen exec artifacts.

Where the expectation lives
---------------------------
``docs/factors/d5_hand_anchor_record_manifest.json`` -- in git, while the record
itself is gitignored. Same split, same reason, as everywhere else here: whoever
overwrites the record cannot also move the expectation without it appearing in
``git diff``.

The manifest also states the record's INVENTORY, and the inventory is checked,
not merely written down. That matters because the pinned record is incomplete:
``daily_engine_compared`` is empty and cannot be refilled (the companion was
retired in C6). Pinning an incomplete record is fine. Letting a later reader take
"it is pinned" for "it is complete" is not, so the shortfall is a field in the
manifest and an assertion here rather than a sentence someone might not read.

RANGE OF THE INVENTORY CHECK -- read this before relying on it
--------------------------------------------------------------
Measured from the CURRENT pinned record (70 / 20 / 0 / False), not reasoned:

* it bites wherever the recorded SHAPE moves, which is more than row counts:
  the three counts, the KEY SET, and the ``all_ok_frozen14`` verdict flag are
  all compared. A record that gains ``daily_engine_compared`` rows, grows an
  ``all_ok_daily`` key, or flips that flag is refused even with a matching sha;
* it contributes NOTHING when all five stay identical -- which is what a plain
  ``qt.hand_anchor_rows`` rerun produces today. Same seed and same stratified
  selection give the same counts and key set; the same inputs give the same
  flag; only the values inside move. Such a rerun is REFUSED while the sha is
  stale and ACCEPTED the moment the sha is honestly refreshed, so on that path
  **the protection is the human reading the git diff, not this check**.
  (A rerun whose verdict flag DID flip would be caught -- the exemption is the
  unchanged shape, not the act of rerunning.)

The direction that motivated the check -- ``daily_engine_compared`` silently
going to zero, the 2026-07-25 accident -- is no longer reachable FROM HERE: that
loss already happened, and what is pinned is the state after it. The field is
already on the floor. The check would catch that shape on a record that still
had those rows; this record does not.

Both facts are stated because either alone misleads. "The inventory is checked"
invites more confidence than it earns on the rerun path; "the inventory is
useless" is false in the other direction.

A stronger form exists and is deliberately NOT built yet: have
``hand_anchor_rows`` record the ``panels_d2`` canonical hashes it computed
against, so the record is tied to the panels it describes rather than merely
being unchanged. That needs a rerun, and a rerun would overwrite the very bytes
the C5 anchors leg was verified against. It belongs with the follow-up that
re-verifies that leg (design note: C-2/D-4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

#: Git-tracked expectation for the record.
DEFAULT_MANIFEST = "docs/factors/d5_hand_anchor_record_manifest.json"
#: Manifest schema tag; bumping it invalidates every reader, intentionally.
MANIFEST_SCHEMA = "d5-hand-anchor-record/1"
#: Inventory fields the manifest states and this module re-derives and compares.
INVENTORY_FIELDS = (
    "frozen14_rows",
    "daily_pending_engine_rows",
    "daily_engine_compared_rows",
    "all_ok_frozen14",
    # ``top_level_keys`` was STATED in the manifest but never compared, so a
    # record could grow a key -- ``all_ok_daily``, say -- and be accepted with a
    # refreshed sha, while the manifest's most prominent prose says that key is
    # ABSENT. "Pinned is not complete" was resting on prose, and prose stops
    # nothing. Counts alone cannot see a key appear or vanish; the key SET can.
    "top_level_keys",
)


class AnchorRecordMismatch(RuntimeError):
    """The hand-anchor record is not the one the git-tracked manifest describes."""


@dataclass(frozen=True)
class AnchorRecordCheck:
    """What was verified (no secrets; counts and a hash only)."""

    path: Path
    sha256: str
    inventory: dict


def record_inventory(payload: dict) -> dict:
    """The inventory the manifest states, re-derived from the record itself."""
    return {
        "frozen14_rows": len(payload.get("frozen14", []) or []),
        "daily_pending_engine_rows": len(payload.get("daily_pending_engine", []) or []),
        "daily_engine_compared_rows": len(payload.get("daily_engine_compared", []) or []),
        "all_ok_frozen14": payload.get("all_ok_frozen14"),
        "top_level_keys": sorted(payload),
    }


def verify_anchor_record(
    record_path: Path | str,
    manifest_path: Path | str,
) -> AnchorRecordCheck:
    """Verify the record's bytes AND its inventory against the git-tracked manifest.

    Raises :class:`AnchorRecordMismatch` on any disagreement. The two halves catch
    different things and neither subsumes the other:

    * the sha catches ANY byte change, including the one the inventory cannot see
      -- a rerun that keeps every count identical while the values move;
    * the inventory catches a manifest refreshed to match a record whose SHAPE
      changed: rows gained or lost, a key appearing or vanishing.

    See the module docstring's RANGE section for what this does NOT cover from
    the currently pinned record. Short version: on the rerun-plus-refreshed-sha
    path the inventory adds nothing, and the reviewed git diff is the guard.
    """
    record_path = Path(record_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise AnchorRecordMismatch(
            f"no hand-anchor manifest at {manifest_path}; the record cannot be "
            "verified, and an unverified anchors input is how a drifted record "
            "reconciles silently."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AnchorRecordMismatch(
            f"manifest schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}"
        )
    if not record_path.exists():
        raise AnchorRecordMismatch(f"hand-anchor record not found: {record_path}")

    payload_bytes = record_path.read_bytes()
    actual = hashlib.sha256(payload_bytes).hexdigest()
    expected = str(manifest.get("sha256", ""))
    if actual != expected:
        raise AnchorRecordMismatch(
            f"{record_path.name}: sha256 {actual[:16]} != the git-tracked "
            f"{expected[:16]}. The record changed since it was pinned. If a "
            "`qt.hand_anchor_rows` rerun was intended, update "
            f"{manifest_path.name} in the same commit so the change is reviewed "
            "-- and note that a rerun does NOT restore daily_engine_compared."
        )

    inventory = record_inventory(json.loads(payload_bytes.decode("utf-8")))
    stated = manifest.get("contents", {})
    for field in INVENTORY_FIELDS:
        if field not in stated:
            raise AnchorRecordMismatch(
                f"{manifest_path.name} states no {field!r}; the inventory is what "
                "keeps 'pinned' from being read as 'complete'."
            )
        if stated[field] != inventory[field]:
            raise AnchorRecordMismatch(
                f"{record_path.name}: {field} is {inventory[field]!r} but "
                f"{manifest_path.name} states {stated[field]!r}."
            )
    return AnchorRecordCheck(path=record_path, sha256=actual, inventory=inventory)
