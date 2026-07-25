"""D5 C1: the frozen exec-basis baseline must be unforgeable and unclobberable.

The frozen 77-artifact tree is the ONLY thing the D5 reconciliation can be
judged against, and the unified runner writes to the same filenames it was
copied from. These tests lock the two properties that make the reconciliation
capable of failing at all:

* the reader serves frozen bytes or raises — it can never silently serve
  freshly generated artifacts (the ``compare_postmerge.py`` empty-reconciliation
  failure mode recorded in CLAUDE.md);
* freezing never overwrites an existing baseline with different content.

Every "X raises" test here is paired with a real, non-vacuous mutation. (The
first draft of the tamper test replaced a byte string that did not occur in the
file, so it mutated nothing and "passed" against a guard that had not been
exercised — the exact trap this suite exists to avoid.)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qt.exec_baseline_freeze import (
    EXPECTED_FILE_COUNT,
    MANIFEST_SCHEMA,
    BaselineIntegrityError,
    FrozenExecBaseline,
    expected_artifact_names,
    freeze,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs/factors/d5_exec_baseline_manifest.json"


def _fake_reports(root: Path) -> Path:
    """A synthetic 'live reports' dir holding the full expected inventory."""
    reports = root / "reports"
    reports.mkdir(parents=True)
    for i, name in enumerate(expected_artifact_names()):
        (reports / name).write_bytes(f"payload-{i}-{name}".encode())
    return reports


def _freeze_fake(tmp_path: Path) -> tuple[Path, Path, Path]:
    reports = _fake_reports(tmp_path)
    frozen = tmp_path / "frozen"
    manifest = tmp_path / "manifest.json"
    freeze(reports, frozen, manifest, repo_root=tmp_path)
    return reports, frozen, manifest


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------


def test_exec_baseline_inventory_is_77_derived_from_the_factor_list() -> None:
    names = expected_artifact_names()
    assert len(names) == EXPECTED_FILE_COUNT == 77
    assert len(set(names)) == len(names)
    assert sum(n.endswith(".json") for n in names) == 22
    assert sum(n.endswith("_dashboard.png") for n in names) == 22
    assert sum(n.endswith("_basis_sanity.md") for n in names) == 11
    # 22 per-book report .md + 11 basis-sanity .md = 33 .md in total
    assert sum(n.endswith(".md") for n in names) == 33


def test_exec_baseline_freeze_refuses_a_short_inventory(tmp_path: Path) -> None:
    """A missing artifact is a loud absence, not a quietly smaller baseline."""
    reports = _fake_reports(tmp_path)
    (reports / "eval_jump_amount_corr_exec_no_book.json").unlink()
    with pytest.raises(FileNotFoundError, match="absent"):
        freeze(reports, tmp_path / "frozen", tmp_path / "m.json", repo_root=tmp_path)


def test_exec_baseline_freeze_refuses_unexpected_members(tmp_path: Path) -> None:
    reports = _fake_reports(tmp_path)
    (reports / "eval_surprise_exec_no_book.json").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="unexpected"):
        freeze(reports, tmp_path / "frozen", tmp_path / "m.json", repo_root=tmp_path)


# --------------------------------------------------------------------------
# freeze: non-destructive, idempotent
# --------------------------------------------------------------------------


def test_exec_baseline_freeze_copies_then_is_idempotent(tmp_path: Path) -> None:
    reports, frozen, manifest = _freeze_fake(tmp_path)
    assert len(list(frozen.iterdir())) == EXPECTED_FILE_COUNT

    again = freeze(reports, frozen, manifest, repo_root=tmp_path)
    assert again.copied == ()
    assert len(again.already_frozen) == EXPECTED_FILE_COUNT


def test_exec_baseline_freeze_never_overwrites_a_differing_baseline(
    tmp_path: Path,
) -> None:
    """MUTATION: change the LIVE artifact after freezing. Re-freeze must refuse.

    This is the scenario that matters — the unified runner rewriting a live
    artifact must never be able to launder itself into the frozen baseline.
    """
    reports, frozen, manifest = _freeze_fake(tmp_path)
    victim = reports / "eval_ridge_minute_return_exec_with_book.json"
    before = (frozen / victim.name).read_bytes()
    victim.write_bytes(victim.read_bytes() + b"!")  # real, non-vacuous change

    with pytest.raises(BaselineIntegrityError, match="differs from the live"):
        freeze(reports, frozen, manifest, repo_root=tmp_path)
    assert (frozen / victim.name).read_bytes() == before  # untouched


def test_exec_baseline_freeze_refuses_to_target_the_source_dir(tmp_path: Path) -> None:
    reports = _fake_reports(tmp_path)
    with pytest.raises(ValueError, match="live reports directory"):
        freeze(reports, reports, tmp_path / "m.json", repo_root=tmp_path)


# --------------------------------------------------------------------------
# reader: structural isolation
# --------------------------------------------------------------------------


def test_exec_baseline_reader_rejects_tampered_bytes(tmp_path: Path) -> None:
    """MUTATION: flip one byte of a frozen file; the reader must refuse it."""
    _, frozen, manifest = _freeze_fake(tmp_path)
    name = "eval_jump_amount_corr_exec_no_book.json"
    victim = frozen / name
    before = victim.read_bytes()
    after = before[:-1] + bytes([before[-1] ^ 0x01])
    assert after != before, "mutation must actually mutate"
    assert hashlib.sha256(after).hexdigest() != hashlib.sha256(before).hexdigest()
    victim.write_bytes(after)

    baseline = FrozenExecBaseline(frozen, manifest)
    with pytest.raises(BaselineIntegrityError, match="on-disk sha256"):
        baseline.read_bytes(name)

    ok, problems = baseline.verify_all()
    assert (ok, len(problems)) == (EXPECTED_FILE_COUNT - 1, 1)
    # the guard is targeted, not a blanket failure
    baseline.read_bytes("eval_valley_price_quantile_exec_no_book.json")


def test_exec_baseline_reader_rejects_a_missing_file(tmp_path: Path) -> None:
    _, frozen, manifest = _freeze_fake(tmp_path)
    (frozen / "eval_volume_peak_count_exec_no_book.json").unlink()
    baseline = FrozenExecBaseline(frozen, manifest)
    with pytest.raises(FileNotFoundError):
        baseline.read_bytes("eval_volume_peak_count_exec_no_book.json")


def test_exec_baseline_reader_refuses_the_live_reports_directory() -> None:
    """Aiming the reader at artifacts/reports/ is refused by path, before hashing."""
    with pytest.raises(BaselineIntegrityError, match="LIVE artifacts directory"):
        FrozenExecBaseline(REPO_ROOT / "artifacts" / "reports", MANIFEST)


def test_exec_baseline_reader_requires_a_manifest(tmp_path: Path) -> None:
    _, frozen, _ = _freeze_fake(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        FrozenExecBaseline(frozen, tmp_path / "nope.json")


def test_exec_baseline_reader_rejects_a_foreign_manifest_schema(tmp_path: Path) -> None:
    _, frozen, manifest = _freeze_fake(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["schema"] = "something-else/9"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(BaselineIntegrityError, match="schema"):
        FrozenExecBaseline(frozen, manifest)


def test_exec_baseline_reader_rejects_unknown_names(tmp_path: Path) -> None:
    _, frozen, manifest = _freeze_fake(tmp_path)
    baseline = FrozenExecBaseline(frozen, manifest)
    with pytest.raises(KeyError):
        baseline.read_bytes("eval_not_a_factor_exec_no_book.json")


# --------------------------------------------------------------------------
# the real, committed baseline
# --------------------------------------------------------------------------


def test_committed_manifest_describes_the_real_frozen_baseline() -> None:
    """The git-tracked manifest is well-formed, complete, and carries provenance.

    Deliberately NOT skipif-guarded: it reads only the committed manifest, so it
    runs on every machine. That matters because the provenance assertions below
    are the ones that catch a manifest whose history was silently restamped —
    a check that skips on the hosts without the gitignored bytes would be a
    check that never runs where it is needed.
    """
    payload = json.loads(MANIFEST.read_text())
    assert payload["schema"] == MANIFEST_SCHEMA
    assert payload["file_count"] == EXPECTED_FILE_COUNT
    assert len(payload["files"]) == EXPECTED_FILE_COUNT
    assert {f["name"] for f in payload["files"]} == set(expected_artifact_names())
    # repo-relative, machine-independent
    assert not payload["source_dir"].startswith("/")
    assert not payload["frozen_root"].startswith("/")

    # --- provenance must be present and well-formed, not merely a field ---
    assert payload["source_note"].strip(), (
        "source_note is empty: the manifest no longer records WHERE the frozen "
        "bytes came from. A re-freeze must inherit provenance, never blank it."
    )
    assert re.fullmatch(r"[0-9a-f]{40}", payload["frozen_at_git_head"]), (
        f"frozen_at_git_head {payload['frozen_at_git_head']!r} is not a full "
        "40-hex commit id"
    )
    assert payload["frozen_at_utc"].startswith("20")

    for entry in payload["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), (
            f"{entry['name']}: sha256 is not 64 hex chars"
        )
        assert int(entry["size_bytes"]) > 0, f"{entry['name']}: non-positive size"


def test_freeze_inherits_provenance_instead_of_restamping_it(tmp_path: Path) -> None:
    """MUTATION: re-freeze an UNCHANGED tree. Provenance must survive verbatim.

    The regression this locks: freeze() used to rebuild frozen_at_utc /
    frozen_at_git_head / source_note on every call, so the documented
    verification command silently repointed the manifest at whoever re-ran it
    and blanked the provenance narrative, while printing `copied: 0`.
    """
    reports, frozen, manifest = _freeze_fake(tmp_path)
    # give it real provenance, as the committed manifest has
    original = json.loads(manifest.read_text())
    original["source_note"] = "PR #79 artifacts; bulk-copied 2026-07-21 15:33:20"
    original["frozen_at_git_head"] = "a" * 40
    original["frozen_at_utc"] = "2026-07-25T00:00:00+00:00"
    manifest.write_text(json.dumps(original, indent=2) + "\n")
    before = manifest.read_bytes()

    result = freeze(reports, frozen, manifest, repo_root=tmp_path)

    assert result.copied == ()
    assert result.manifest_written is False, "an unchanged tree must not rewrite"
    assert manifest.read_bytes() == before, "manifest bytes must be untouched"
    after = json.loads(manifest.read_text())
    assert after["source_note"] == original["source_note"]
    assert after["frozen_at_git_head"] == "a" * 40
    assert after["frozen_at_utc"] == "2026-07-25T00:00:00+00:00"


def test_freeze_lets_an_explicit_source_note_win(tmp_path: Path) -> None:
    """Inheritance is a default, not a lock: an explicit note is still an act."""
    reports, frozen, manifest = _freeze_fake(tmp_path)
    result = freeze(
        reports, frozen, manifest, repo_root=tmp_path, source_note="corrected note"
    )
    assert result.manifest_written is True
    assert json.loads(manifest.read_text())["source_note"] == "corrected note"


@pytest.mark.skipif(
    not (REPO_ROOT / "artifacts/refactor_baseline/exec_baseline").is_dir(),
    reason="frozen baseline bytes are gitignored; present only on the run host",
)
def test_real_frozen_baseline_verifies_against_the_committed_manifest() -> None:
    baseline = FrozenExecBaseline(
        REPO_ROOT / "artifacts/refactor_baseline/exec_baseline", MANIFEST
    )
    ok, problems = baseline.verify_all()
    assert problems == []
    assert ok == EXPECTED_FILE_COUNT

    report = baseline.report_json("jump_amount_corr", "no_book")
    assert set(report) >= {"sections", "spec", "thresholds", "verdict"}


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"x" * (3 << 20) + b"tail"  # spans the 1MiB read chunks
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def _tracked_under(prefix: str) -> list[str]:
    """Paths git TRACKS under ``prefix``, read from the index.

    Queries the index rather than the filesystem on purpose. ``artifacts/`` is
    routinely a symlink into another checkout, and every filesystem-walking git
    command (``add``, ``check-ignore``, ``ls-files --error-unmatch`` on a path)
    refuses such paths with "beyond a symbolic link" — which would make an
    assertion about them pass for the wrong reason, and make it impossible to
    mutate. The index has no such blind spot.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", prefix],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.split()


def test_frozen_bytes_are_untracked_while_the_manifest_is_tracked() -> None:
    """The split that gives the tamper check its teeth, asserted against git.

    Hashes live in version control so they can only move through a reviewable
    diff; the bytes they describe do not. If both lived in git, one commit could
    move the bytes and their hashes together and the tamper check would be
    decorative.

    The previous version of this test claimed exactly this but never consulted
    git at all — it froze a fake tree under tmp_path, deleted it, and asserted a
    file it had just written was still readable. Removing ``artifacts/`` from
    .gitignore left it green.
    """
    assert _tracked_under("artifacts") == [], (
        "no bytes under artifacts/ may be committed; found tracked paths"
    )
    manifest_rel = str(MANIFEST.relative_to(REPO_ROOT))
    assert manifest_rel in _tracked_under("docs/factors"), (
        "the manifest MUST be committed — it is the tamper reference"
    )


def test_manifest_outlives_the_bytes_it_describes(tmp_path: Path) -> None:
    """Losing the frozen tree must not lose the record of what it contained."""
    _, frozen, manifest = _freeze_fake(tmp_path)
    shutil.rmtree(frozen)
    assert json.loads(manifest.read_text())["file_count"] == EXPECTED_FILE_COUNT
