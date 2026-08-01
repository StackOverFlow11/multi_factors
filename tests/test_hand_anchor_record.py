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
import subprocess
from pathlib import Path

import pytest

from qt.hand_anchors_d2 import TOL, pending_engine_line
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


# --------------------------------------------------------------------------- #
# The run-summary pointer (review MEDIUM-1)
# --------------------------------------------------------------------------- #
def test_the_pending_line_does_not_send_anyone_to_a_command_that_raises():
    """`hand_anchor_rows` printed "(run python -m qt.hand_anchors_engine_values)"
    on every successful run — and that command now raises.

    Worse in context: the same function writes `daily_pending_engine` WITHOUT
    `daily_engine_compared`, so it is the thing that empties the record. A user
    had just wiped the comparison and was then pointed down a dead end.
    """
    line = pending_engine_line(20)
    assert "20" in line
    assert "python -m qt.hand_anchors_engine_values --verify" in line
    assert "RETIRED" in line
    assert "no longer be completed" in line
    # the bare invocation must not appear as an instruction
    assert "hand_anchors_engine_values)" not in line
    assert "run python -m qt.hand_anchors_engine_values\n" not in line


def test_the_pointer_is_composed_by_the_printer_not_restated():
    """Author-once, structurally: the printing module must carry NO literal of
    its own. A regex cannot assert that no other sentence points at the retired
    command; "there is no other sentence in this file" can."""
    source = (Path(__file__).resolve().parents[1] / "qt" / "hand_anchor_rows.py").read_text()
    # The INVOCATION is what must exist once. Naming the module in prose is
    # fine and useful; telling someone to run it is the thing that goes stale.
    assert "python -m qt.hand_anchors_engine_values" not in source
    assert "pending_engine_line" in source


def test_the_pointer_lives_where_the_pure_side_can_reach_it():
    """It is in `hand_anchors_d2`, not in the retired module, because the latter
    imports the engine and the hand side must never load it. If this ever moves,
    `hand_anchor_rows` would drag `data.clean` in and break hand-anchor purity."""
    import qt.hand_anchors_d2 as pure

    assert isinstance(pure.ENGINE_COMPARISON_POINTER, str)
    # AST, not substring: the module has a COMMENT naming data.clean to explain
    # why it stays out, and a text match cannot tell that from an import.
    import ast

    for name in ("hand_anchors_d2.py", "hand_anchor_rows.py"):
        path = Path(__file__).resolve().parents[1] / "qt" / name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [node.module or ""]
            for target in targets:
                assert not target.startswith(("factors", "data.clean")), (
                    f"{name} imports the engine ({target}); the hand side must not"
                )


RETIRED_INVOCATIONS = (
    "python -m qt.panel_freeze",
    "python -m qt.panel_reconcile",
    "python -m qt.hand_anchors_engine_values",
)
#: Sites allowed to name a retired invocation WITHOUT `--verify`, each because
#: the tool is the SUBJECT of a retirement statement rather than an instruction.
#: Reviewed one by one; anything new must be looked at, not appended blindly.
NAMES_A_RETIRED_TOOL_AS_SUBJECT = {
    "qt/panel_freeze.py",              # the `tool` argument of retirement_message
    "qt/panel_reconcile.py",           # ditto
    "qt/hand_anchors_engine_values.py",  # ditto
    "docs/factors/d1_panel_freeze_manifest.md",  # "(不带 --verify) 现在是可读的报错"
    "docs/factors/pr_c_cutoff_fix_reference_panel.md",  # archived [RETIRED, C6] block
}


def test_no_new_file_starts_instructing_people_to_run_a_retired_tool():
    """An inventory net, NOT a meaning check — range stated so it is not trusted
    further than it reaches.

    It cannot tell an instruction from a description; it only notices when a NEW
    file starts naming a retired invocation without `--verify`. That is worth
    having because the defect it follows was a pre-existing line in a file the
    author never scanned: fixing the two documents and missing the runtime print
    is exactly the #82 shape — a guard that only looks where you already looked
    confirms only what you already know. `docs/progress/` is excluded as a
    historical archive: it records what was true then.

    SCOPE OF THE FILE LIST. Derived from git (tracked ∪ untracked-not-ignored),
    never from a directory walk. A walk sweeps in gitignored working material —
    `tmp/context/**` holds `git archive` exports of PRE-RETIREMENT revisions, so
    every one of them names the retired tools legitimately — and inside a
    worktree it follows the `artifacts` symlink into a different checkout. Both
    were observed: this guard was red on the main checkout while green in every
    worktree, because a worktree structurally lacks the directories that break
    it. Untracked-but-not-ignored IS included on purpose: a newly written file
    is exactly the case this guard exists for, and it is usually not staged yet.

    WHAT IT STILL CANNOT SEE, all four of them:
      1. gitignored files (deliberate — they are not the codebase);
      2. invocation strings composed at runtime;
      3. any file whose suffix is outside the set below. This one has live
         instances: `deploy/systemd/quant-data-update.service` exists to name a
         command to run (`ExecStart=`), and `pyproject.toml` can too, yet
         neither is scanned. Wiring a retired tool into a unit file or a
         pyproject entry point would NOT turn this red. Widening the suffix set
         is a change of reach, not a fix to this declaration — do it as its own
         change, with its own evidence, or not at all;
      4. `tests/` — excluded because RETIRED_INVOCATIONS itself lives there and
         would self-match. Retired calls inside test files are therefore
         invisible too.
    """
    repo = Path(__file__).resolve().parents[1]

    def _git(*args: str) -> list[str]:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout
        return [line for line in out.splitlines() if line]

    listed = set(_git("ls-files")) | set(_git("ls-files", "--others", "--exclude-standard"))
    offenders: list[str] = []
    for relative in sorted(listed):
        path = repo / relative
        if path.suffix not in {".py", ".md", ".yaml", ".txt"} or not path.is_file():
            continue
        if relative.startswith(("docs/progress/", "tests/")):
            continue
        text = path.read_text(encoding="utf-8")
        for call in RETIRED_INVOCATIONS:
            start = text.find(call)
            while start != -1:
                # WINDOW, not line: these sentences wrap, so `--verify` routinely
                # lands on the following source line.
                window = text[start : start + len(call) + 40]
                if "--verify" not in window and relative not in NAMES_A_RETIRED_TOOL_AS_SUBJECT:
                    number = text.count("\n", 0, start) + 1
                    offenders.append(f"{relative}:{number}: {call}")
                start = text.find(call, start + 1)
    assert not offenders, (
        "these name a retired invocation without --verify, from a file not on the "
        "reviewed subject list:\n  " + "\n  ".join(offenders)
    )
