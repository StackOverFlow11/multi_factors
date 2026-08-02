"""D7-PR0: the run-registry wiring for run-factor-eval — network-free.

Covers the explicit status mapping (book member -> book; Watch -> watch;
Reject -> exploratory and NEVER retired; INSUFFICIENT-DATA -> exploratory;
Adopt refused), the book-set assertion (latest-record-wins over the
append-only log), the empty-registry bootstrap seed, and the no-secret
contract on the appended note.
"""

from __future__ import annotations

import pytest

from analytics.eval.verdict import ADOPT, INSUFFICIENT_DATA, REJECT, WATCH
from factors import registry as factor_registry
from factors.store import RunRegistry, data_fingerprint, store_key
from qt.factor_eval_registry import (
    BOOTSTRAP_NOTE,
    VERDICT_TO_STATUS,
    current_book_set,
    status_for_run,
    sync_book_registry,
)
from qt.factor_eval_runner import BOOK_IDS

_TOKEN = "abcd1234deadbeefabcd1234deadbeef"
_NON_BOOK = "jump_amount_corr_20"


# --------------------------------------------------------------------------- #
# the status mapping (a code constant, tested exhaustively)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor_id", BOOK_IDS)
@pytest.mark.parametrize("verdict", [WATCH, REJECT, INSUFFICIENT_DATA, ADOPT])
def test_book_members_record_book_regardless_of_verdict(factor_id, verdict):
    """Curation is never derived from a gate verdict — not even from Reject."""
    assert status_for_run(factor_id, verdict, book_ids=BOOK_IDS) == "book"


def test_watch_maps_to_watch():
    assert status_for_run(_NON_BOOK, WATCH, book_ids=BOOK_IDS) == "watch"


def test_reject_maps_to_exploratory_never_retired():
    """A default gate verdict must never be able to derive a retirement."""
    status = status_for_run(_NON_BOOK, REJECT, book_ids=BOOK_IDS)
    assert status == "exploratory"
    assert status != "retired"
    # ...and no OTHER verdict can derive it either: retirement is a human edit.
    assert "retired" not in VERDICT_TO_STATUS.values()


def test_insufficient_data_maps_to_exploratory():
    assert status_for_run(_NON_BOOK, INSUFFICIENT_DATA, book_ids=BOOK_IDS) == "exploratory"


def test_adopt_is_a_readable_error_not_a_silent_promotion():
    """run-factor-eval is exploratory-capped, so Adopt is unreachable; and
    promotion is a human curation decision the mapping refuses to derive."""
    with pytest.raises(ValueError, match="Adopt is unreachable"):
        status_for_run(_NON_BOOK, ADOPT, book_ids=BOOK_IDS)


def test_an_unknown_verdict_is_a_readable_error():
    with pytest.raises(ValueError, match="no run-registry status mapping"):
        status_for_run(_NON_BOOK, "Promising", book_ids=BOOK_IDS)


# --------------------------------------------------------------------------- #
# current_book_set: latest record wins over the append-only log
# --------------------------------------------------------------------------- #
def test_latest_record_wins_so_a_retirement_trips_the_assertion():
    records = [
        {"factor_id": "value_ep", "status": "book"},
        {"factor_id": "value_bp", "status": "book"},
        {"factor_id": "value_ep", "status": "retired"},  # a later human edit
    ]
    assert current_book_set(records) == {"value_bp"}


# --------------------------------------------------------------------------- #
# bootstrap: empty registry -> seed the declared book, then assert
# --------------------------------------------------------------------------- #
def test_empty_registry_is_seeded_from_the_declared_book(tmp_path):
    reg = RunRegistry(tmp_path)
    assert sync_book_registry(reg, book_ids=BOOK_IDS) == "seeded"
    records = reg.read_all()
    assert len(records) == len(BOOK_IDS)
    assert {r["factor_id"] for r in records} == set(BOOK_IDS)
    assert all(r["status"] == "book" for r in records)
    assert all(r["note"] == BOOTSTRAP_NOTE for r in records)
    assert all(r["view"] == "decision" for r in records)


def test_seeded_book_records_carry_the_same_key_identity_the_service_uses(tmp_path):
    """The seed's params/code hashes equal store_key(build(fid), decision,
    params=None) — the key the runner's own book panel read is served under."""
    reg = RunRegistry(tmp_path)
    sync_book_registry(reg, book_ids=BOOK_IDS)
    by_id = {r["factor_id"]: r for r in reg.read_all()}
    for factor_id in BOOK_IDS:
        key = store_key(
            factor_registry.build(factor_id), view="decision", params=None
        )
        assert by_id[factor_id]["params_hash"] == key.params_hash
        assert by_id[factor_id]["code_hash"] == key.code_hash
        assert by_id[factor_id]["schema_version"] != ""


def test_a_conforming_registry_is_verified_and_never_reseeded(tmp_path):
    reg = RunRegistry(tmp_path)
    sync_book_registry(reg, book_ids=BOOK_IDS)
    # a non-book run record on top must not disturb the assertion
    factor = factor_registry.build(_NON_BOOK)
    reg.append_run(
        key=store_key(factor, view="decision", params=None),
        factor=factor,
        status=status_for_run(_NON_BOOK, WATCH, book_ids=BOOK_IDS),
        fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
        note="run-factor-eval test",
    )
    assert sync_book_registry(reg, book_ids=BOOK_IDS) == "verified"
    assert len(reg.read_all()) == len(BOOK_IDS) + 1  # verified appends nothing


# --------------------------------------------------------------------------- #
# the mismatch paths (readable raise, never a silent reseed)
# --------------------------------------------------------------------------- #
def test_a_retired_book_member_fails_the_assertion(tmp_path):
    reg = RunRegistry(tmp_path)
    sync_book_registry(reg, book_ids=BOOK_IDS)
    factor = factor_registry.build("value_ep")
    reg.append_run(
        key=store_key(factor, view="decision", params=None),
        factor=factor,
        status="retired",  # a human curation edit the code must not paper over
        fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
        note="human: retired value_ep",
    )
    with pytest.raises(ValueError, match="missing from registry"):
        sync_book_registry(reg, book_ids=BOOK_IDS)


def test_an_extra_book_factor_fails_the_assertion(tmp_path):
    reg = RunRegistry(tmp_path)
    sync_book_registry(reg, book_ids=BOOK_IDS)
    factor = factor_registry.build("momentum_20")
    reg.append_run(
        key=store_key(factor, view="decision", params=None),
        factor=factor,
        status="book",
        fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
        note="human: promoted momentum_20",
    )
    with pytest.raises(ValueError, match="not in BOOK_IDS"):
        sync_book_registry(reg, book_ids=BOOK_IDS)


def test_a_non_empty_registry_without_a_book_is_never_silently_reseeded(tmp_path):
    """Bootstrap is EMPTY-registry only; a registry that lost its book records
    is a mismatch a human aligns, not a reason to redeclare the book."""
    reg = RunRegistry(tmp_path)
    factor = factor_registry.build(_NON_BOOK)
    reg.append_run(
        key=store_key(factor, view="decision", params=None),
        factor=factor,
        status="watch",
        fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
    )
    with pytest.raises(ValueError, match="disagrees with the code-declared book"):
        sync_book_registry(reg, book_ids=BOOK_IDS)
    assert len(reg.read_all()) == 1  # the failed assertion appended nothing


# --------------------------------------------------------------------------- #
# no-secret: the appended run note is redacted at the boundary (R22)
# --------------------------------------------------------------------------- #
def test_a_token_like_string_in_the_run_note_is_redacted(tmp_path):
    reg = RunRegistry(tmp_path)
    factor = factor_registry.build(_NON_BOOK)
    reg.append_run(
        key=store_key(factor, view="decision", params=None),
        factor=factor,
        status="watch",
        fingerprint=data_fingerprint(adjustment=factor.spec.adjustment),
        note=(
            f"run-factor-eval factor={_NON_BOOK} token={_TOKEN} "
            f"tushare.token={_TOKEN} "
            "config=/home/x/.config.json verdict_with_book=Watch"
        ),
    )
    text = reg.path.read_text()
    assert _TOKEN not in text
    assert ".config.json" not in text
    assert "[REDACTED]" in text
