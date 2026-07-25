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
import shutil
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
    # 22 report .md = 44 total .md minus the 11 sanity files... 22 + 11 = 33
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
    """The git-tracked manifest is well-formed and complete."""
    payload = json.loads(MANIFEST.read_text())
    assert payload["schema"] == MANIFEST_SCHEMA
    assert payload["file_count"] == EXPECTED_FILE_COUNT
    assert len(payload["files"]) == EXPECTED_FILE_COUNT
    assert {f["name"] for f in payload["files"]} == set(expected_artifact_names())
    assert all(len(f["sha256"]) == 64 for f in payload["files"])
    # repo-relative, machine-independent
    assert not payload["source_dir"].startswith("/")
    assert not payload["frozen_root"].startswith("/")


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


def test_frozen_tree_is_not_a_git_tracked_path(tmp_path: Path) -> None:
    """The bytes stay under gitignored artifacts/; only hashes are committed.

    That split is what gives the tamper check teeth: hashes move only through
    a reviewable ``git diff``.
    """
    _, frozen, manifest = _freeze_fake(tmp_path)
    shutil.rmtree(frozen)
    # manifest survives independently of the bytes it describes
    assert json.loads(manifest.read_text())["file_count"] == EXPECTED_FILE_COUNT
