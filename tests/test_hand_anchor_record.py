"""Tests for the daily hand-anchor record verification (VERIFY-ONLY since C6).

The comparison that produced ``daily_engine_compared`` is retired, so what is
left to check is whether the RECORD says what its own numbers imply. These tests
pin both directions: an honest record passes, and every way of editing it into a
false pass comes back red.

The absence case gets its own test because it is the one with a judgement call
in it: nothing recorded exits NON-ZERO with the wording "NOT VERIFIED". Absence
of evidence is not evidence of failure, but the exit-code channel is binary and
a verification command that reports success without verifying anything is the
empty-reconciliation shape this repository has already committed once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qt.hand_anchors_d2 import TOL
from qt.hand_anchors_engine_values import (
    DAILY_FACTORS,
    AnchorRecordVerification,
    main,
    relative_difference,
    run_engine_comparison,
    verify_recorded_comparison,
)
from qt.panel_freeze import RegenerationRetiredError


def _row(factor_id="momentum_20", hand=1.0, engine=1.0, **overrides) -> dict:
    rel = relative_difference(hand, engine)
    row = {
        "factor_id": factor_id,
        "class": "random",
        "date": "2024-05-30",
        "symbol": "000001.SZ",
        "hand": hand,
        "engine": engine,
        "rel_diff": rel,
        "ok": rel <= TOL,
    }
    row.update(overrides)
    return row


def _record(tmp_path: Path, compared: list[dict], pending: list[dict] | None = None,
            all_ok: bool | None = None) -> Path:
    if all_ok is None:
        all_ok = all(bool(row.get("ok")) for row in compared)
    path = tmp_path / "hand_anchors_d2.json"
    path.write_text(
        json.dumps(
            {
                "seed": 20260724,
                "tolerance": TOL,
                "daily_pending_engine": pending or [],
                "daily_engine_compared": compared,
                "all_ok_daily": all_ok,
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Retired regeneration entry point
# --------------------------------------------------------------------------- #
def test_the_engine_comparison_raises_instead_of_quietly_doing_nothing():
    with pytest.raises(RegenerationRetiredError, match="RETIRED"):
        run_engine_comparison()


def test_the_retirement_message_does_not_overstate_what_was_lost():
    """This tool leaned on ONE precondition helper, not the eleven private
    loaders, and its factors are untouched. Saying otherwise would be the same
    class of defect as a report claiming a check it did not run."""
    with pytest.raises(RegenerationRetiredError) as caught:
        run_engine_comparison()
    text = str(caught.value)
    assert "precondition check" in text
    assert "four daily factor classes are untouched" in text
    assert "python -m qt.hand_anchors_engine_values --verify" in text
    # It must NOT claim to check frozen bytes against a git-tracked manifest —
    # that is what the two panel tools do, not this one.
    assert "git-tracked manifest" not in text


def test_a_bare_invocation_is_non_zero_and_explains_itself(capsys):
    assert main([]) == 1
    assert "RETIRED" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Verification of the record
# --------------------------------------------------------------------------- #
def test_an_honest_record_verifies_green(tmp_path: Path):
    path = _record(tmp_path, [_row(factor_id=fid) for fid in DAILY_FACTORS])
    result = verify_recorded_comparison(path)
    assert result.ok and result.n_compared == 4 and not result.problems


def test_a_flipped_ok_flag_is_convicted(tmp_path: Path):
    """The record claims a row passed; its own hand/engine numbers say it did
    not. This is the edit that a verify-only tool exists to catch."""
    honest = _row(hand=1.0, engine=2.0)
    assert honest["ok"] is False  # the numbers really do disagree
    path = _record(tmp_path, [{**honest, "ok": True}], all_ok=True)
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("recorded ok=True" in p for p in result.problems)


def test_a_doctored_rel_diff_is_convicted(tmp_path: Path):
    path = _record(tmp_path, [_row(hand=1.0, engine=2.0, rel_diff=0.0, ok=True)],
                   all_ok=True)
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("imply" in p for p in result.problems)


def test_a_recorded_mismatch_is_reported_even_when_labelled_consistently(tmp_path: Path):
    """Row honestly says ok=False and all_ok_daily is honestly False. The record
    is self-consistent — and it records a FAILURE, which must not read green."""
    path = _record(tmp_path, [_row(hand=1.0, engine=2.0)])
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("exceed the tolerance" in p for p in result.problems)


def test_an_all_ok_flag_that_contradicts_the_rows_is_convicted(tmp_path: Path):
    path = _record(tmp_path, [_row(hand=1.0, engine=2.0)], all_ok=True)
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("all_ok_daily is recorded as True" in p for p in result.problems)


def test_a_pending_anchor_that_was_never_compared_is_convicted(tmp_path: Path):
    """A partially filled record must not pass on the strength of the rows that
    ARE there."""
    pending = [{"factor_id": "liquidity_20", "date": "2024-05-30", "symbol": "600000.SH"}]
    path = _record(tmp_path, [_row()], pending=pending)
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("never compared" in p for p in result.problems)


def test_a_row_missing_fields_is_convicted(tmp_path: Path):
    honest = _row()
    honest.pop("engine")
    path = _record(tmp_path, [honest], all_ok=True)
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("missing field(s)" in p for p in result.problems)


def test_an_unexpected_factor_id_is_convicted(tmp_path: Path):
    path = _record(tmp_path, [_row(factor_id="not_a_daily_factor")])
    result = verify_recorded_comparison(path)
    assert not result.ok
    assert any("unexpected factor id" in p for p in result.problems)


def test_nothing_recorded_is_not_verified_and_exits_non_zero(tmp_path: Path, capsys):
    path = _record(tmp_path, [], pending=[{"factor_id": "momentum_20",
                                           "date": "2024-05-30",
                                           "symbol": "000001.SZ"}])
    result = verify_recorded_comparison(path)
    assert isinstance(result, AnchorRecordVerification)
    assert not result.recorded and not result.ok and result.n_pending == 1
    assert main(["--verify", "--record", str(path)]) == 1
    assert "NOT VERIFIED (nothing recorded)" in capsys.readouterr().out


def test_an_absent_record_file_is_not_verified(tmp_path: Path):
    result = verify_recorded_comparison(tmp_path / "nope.json")
    assert not result.recorded and not result.ok


def test_verify_writes_nothing_into_the_record(tmp_path: Path):
    path = _record(tmp_path, [_row(factor_id=fid) for fid in DAILY_FACTORS])
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    assert main(["--verify", "--record", str(path)]) == 0
    assert (path.stat().st_mtime_ns, path.stat().st_size) == before


@pytest.mark.parametrize(
    "hand, engine, expected",
    [
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (float("nan"), float("nan"), 0.0),
        (1.0, float("nan"), float("inf")),
        (2.0, 1.0, 0.5),
    ],
)
def test_the_relative_difference_rule_matches_the_recorded_one(hand, engine, expected):
    assert relative_difference(hand, engine) == expected
