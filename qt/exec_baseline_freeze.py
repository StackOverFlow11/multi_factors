"""D5 exec-basis evaluation baseline freeze — the ONLY reconciliation reference.

The 22 exec-basis evaluation artifacts (11 factors x {no_book, with_book}), their
22 rendered Markdown reports, the 11 ``_exec_basis_sanity.md`` coverage
disclosures and the 22 dashboards live in ``artifacts/reports/`` and are
**gitignored, single-copy, and not reproducible from current code**: regenerating
them needs a checkout of the pre-D5 tree plus a ~1.5h real run. The unified D5
runner writes to those same filenames. Running it once would therefore destroy
the only thing D5 can be judged against, and any "reconciliation" performed
afterwards would compare the new output with a copy of itself — the
``compare_postmerge.py`` empty-reconciliation failure mode this repo has already
committed once and recorded in CLAUDE.md.

So: freeze first, reconcile against the frozen copy, never against the live path.

Structural isolation (NOT discipline)
-------------------------------------
``FrozenExecBaseline`` is the only supported reader, and it cannot be aimed at
the live artifacts directory even by accident:

1. it requires a git-tracked manifest describing the tree it is given;
2. **every file it returns is sha256-verified against that manifest at read
   time** — so a live artifact copied over a frozen one is a loud failure, not a
   silent substitution. This is the load-bearing guard: it does not ask whether
   the *path* looks right, it asks whether the *bytes* are the frozen bytes;
3. it refuses outright to resolve to the live reports directory.

The manifest lives in git (``docs/factors/d5_exec_baseline_manifest.json``) while
the bytes live under gitignored ``artifacts/``. That split is deliberate and is
the reason (2) has teeth: an attacker-in-the-form-of-a-tired-engineer who
overwrites the frozen tree cannot also silently move the hashes, because the
hashes are under version control and show up in ``git diff``.

Both entry points are BYTE-idempotent on an unchanged tree: re-running them
writes nothing, including the manifest, so ``git diff`` stays clean. Provenance
fields (``frozen_at_utc`` / ``frozen_at_git_head`` / ``source_note``) are
inherited from an existing manifest rather than restamped — a verification run
must not be able to re-date the history it is verifying.

Usage (needs the gitignored ``artifacts/`` tree; in a worktree without one, pass
``--repo-root`` pointing at a checkout that has it)::

    python -m qt.exec_baseline_freeze            # freeze; refuses to clobber
    python -m qt.exec_baseline_freeze --verify   # re-verify the frozen tree, copy nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Repository-relative default locations.
DEFAULT_REPORTS_DIR = "artifacts/reports"
DEFAULT_FROZEN_ROOT = "artifacts/refactor_baseline/exec_baseline"
DEFAULT_MANIFEST = "docs/factors/d5_exec_baseline_manifest.json"

#: Manifest schema tag. Bumping it invalidates every reader — intentional.
MANIFEST_SCHEMA = "d5-exec-baseline-freeze/1"

#: Only the exec-basis family is frozen. The close-basis artifacts are NOT the
#: D5 reference (the unified runner is exec-only, design v3.2 decision 4) and
#: are left alone.
EXEC_MARKER = "_exec_"
FILENAME_PREFIX = "eval_"
FROZEN_SUFFIXES = (".json", ".md", ".png")

#: The eleven closing minute factors, in the canonical order used by reports.
FACTORS = (
    "jump_amount_corr",
    "minute_ideal_amplitude",
    "amp_marginal_anomaly_vol",
    "volume_peak_count",
    "intraday_amp_cut",
    "peak_interval_kurtosis",
    "valley_relative_vwap",
    "valley_ridge_vwap_ratio",
    "ridge_minute_return",
    "valley_price_quantile",
    "peak_ridge_amount_ratio",
)
BOOKS = ("no_book", "with_book")

#: Expected inventory: 22 JSON + 22 MD + 11 sanity MD + 22 PNG.
EXPECTED_FILE_COUNT = 77


class BaselineIntegrityError(RuntimeError):
    """Raised when frozen bytes do not match the git-tracked manifest."""


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes.

    For opaque artifacts (JSON/Markdown/PNG) the byte hash IS canonical — unlike
    parquet, there is no writer metadata to normalise away.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_names() -> tuple[str, ...]:
    """The 77 exec-basis artifact names this freeze is defined over.

    Derived from the factor list rather than from a directory listing, so a
    missing file is a loud absence instead of a quietly smaller baseline.
    """
    names: list[str] = []
    for factor in FACTORS:
        names.append(f"eval_{factor}_exec_basis_sanity.md")
        for book in BOOKS:
            stem = f"eval_{factor}_exec_{book}"
            names.extend((f"{stem}.json", f"{stem}.md", f"{stem}_dashboard.png"))
    return tuple(sorted(names))


def discover_exec_artifacts(reports_dir: Path) -> list[Path]:
    """List the exec-basis artifacts actually present in ``reports_dir``."""
    found = [
        path
        for path in sorted(reports_dir.iterdir())
        if path.is_file()
        and path.name.startswith(FILENAME_PREFIX)
        and EXEC_MARKER in path.name
        and path.suffix in FROZEN_SUFFIXES
    ]
    return found


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover - env dependent
        return "unknown"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """tmp + os.replace, the repository's durable-write pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover - only on write failure
            tmp.unlink()


@dataclass(frozen=True)
class FreezeResult:
    copied: tuple[str, ...]
    already_frozen: tuple[str, ...]
    manifest_path: Path
    frozen_root: Path
    #: False when an existing manifest was already correct and was left alone.
    manifest_written: bool = True


def _load_manifest_if_present(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None


def freeze(
    reports_dir: Path,
    frozen_root: Path,
    manifest_path: Path,
    *,
    repo_root: Path,
    source_note: str = "",
) -> FreezeResult:
    """Copy the exec-basis artifacts into ``frozen_root`` and write the manifest.

    Idempotent and non-destructive: a destination that already exists is
    verified against the freshly hashed source and left alone; a MISMATCH is an
    error, never an overwrite. Freezing twice is safe; freezing over a different
    baseline is impossible.
    """
    reports_dir = reports_dir.resolve()
    frozen_root = frozen_root.resolve()
    if frozen_root == reports_dir or frozen_root.is_relative_to(reports_dir):
        raise ValueError(
            f"refusing to freeze into the live reports directory: {frozen_root}"
        )

    expected = set(expected_artifact_names())
    present = {path.name: path for path in discover_exec_artifacts(reports_dir)}
    missing = sorted(expected - set(present))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} expected exec-basis artifact(s) absent from "
            f"{reports_dir}: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    unexpected = sorted(set(present) - expected)
    if unexpected:
        raise RuntimeError(
            "unexpected exec-basis artifact(s) present — the freeze inventory is "
            f"defined by {len(expected)} names and refuses to guess: {unexpected}"
        )

    frozen_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    copied: list[str] = []
    already: list[str] = []

    for name in sorted(expected):
        src = present[name]
        payload = src.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        dst = frozen_root / name
        if dst.exists():
            existing = sha256_file(dst)
            if existing != digest:
                raise BaselineIntegrityError(
                    f"{name}: frozen copy ({existing[:16]}) differs from the live "
                    f"artifact ({digest[:16]}). The frozen baseline is NOT "
                    "overwritten. Investigate before proceeding — one of the two "
                    "is not the PR #79 baseline."
                )
            already.append(name)
        else:
            _atomic_write_bytes(dst, payload)
            copied.append(name)

        stat = src.stat()
        entries.append(
            {
                "name": name,
                "sha256": digest,
                "size_bytes": stat.st_size,
                "source_mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )

    # PROVENANCE IS INHERITED, NEVER SILENTLY RESTAMPED.
    #
    # An earlier version rebuilt these three fields on every invocation, so the
    # documented "re-run me to check the freeze" command quietly repointed
    # frozen_at_git_head at whoever re-ran it and blanked source_note — while
    # reporting `copied: 0` and passing every test. That turned the freeze tool
    # into a generator of the exact false provenance claim it exists to prevent
    # (the #76/#78/#82 shape: the behaviour changed, the wording did not).
    #
    # So: an existing manifest supplies these unless the caller explicitly
    # overrides them. Re-freezing verifies bytes; it does not re-date history.
    # Inherit only what is actually there: a manifest that never carried
    # provenance has nothing to preserve, and inheriting "" from it would put an
    # empty field where the always-run manifest assertions expect a real one —
    # trading a restamp for a silent blank.
    previous = _load_manifest_if_present(manifest_path) or {}
    frozen_at_utc = str(previous.get("frozen_at_utc") or datetime.now(timezone.utc).isoformat())
    frozen_at_head = str(previous.get("frozen_at_git_head") or _git_head(repo_root))
    note = source_note or str(previous.get("source_note") or "")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "frozen_at_utc": frozen_at_utc,
        "frozen_at_git_head": frozen_at_head,
        # Logical, repo-relative locations. The on-disk paths resolve through a
        # gitignored symlink on some checkouts; recording the resolved absolute
        # path would make this git-tracked manifest machine-specific.
        "source_dir": DEFAULT_REPORTS_DIR,
        "frozen_root": DEFAULT_FROZEN_ROOT,
        "file_count": len(entries),
        "factors": list(FACTORS),
        "books": list(BOOKS),
        "source_note": note,
        "files": entries,
    }
    # BYTE-idempotent: an unchanged tree rewrites nothing at all, so re-running
    # the documented verification command leaves a clean `git diff`.
    manifest_written = manifest != previous
    if manifest_written:
        _atomic_write_bytes(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode(),
        )
    return FreezeResult(
        copied=tuple(copied),
        already_frozen=tuple(already),
        manifest_path=manifest_path,
        frozen_root=frozen_root,
        manifest_written=manifest_written,
    )


class FrozenExecBaseline:
    """Verified read-only accessor for the frozen exec-basis baseline.

    Every read is checked against the git-tracked manifest, so this object
    cannot serve bytes that are not the frozen bytes — including the case where
    the caller aims it at a directory of freshly generated artifacts.
    """

    def __init__(self, frozen_root: Path, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no frozen-baseline manifest at {manifest_path}; the baseline "
                "must be frozen (and its manifest committed) before any "
                "reconciliation can read it"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise BaselineIntegrityError(
                f"manifest schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}"
            )

        root = Path(frozen_root).resolve()
        # Refuse the live path outright. This is the cheap guard; the hash check
        # below is the one that actually cannot be talked around.
        if root.name == "reports" and root.parent.name == "artifacts":
            raise BaselineIntegrityError(
                f"refusing to read the LIVE artifacts directory as a frozen "
                f"baseline: {root}. Reconciliation reads frozen copies only."
            )
        if not root.is_dir():
            raise FileNotFoundError(f"frozen baseline root not found: {root}")

        self.root = root
        self.manifest_path = manifest_path
        self._expected: dict[str, str] = {
            str(entry["name"]): str(entry["sha256"]) for entry in manifest["files"]
        }
        self.frozen_at_git_head = manifest.get("frozen_at_git_head", "unknown")
        self.file_count = int(manifest.get("file_count", len(self._expected)))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._expected))

    def read_bytes(self, name: str) -> bytes:
        """Return the frozen bytes of ``name``, or raise."""
        if name not in self._expected:
            raise KeyError(f"{name!r} is not part of the frozen exec baseline")
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"frozen artifact missing from disk: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self._expected[name]:
            raise BaselineIntegrityError(
                f"{name}: on-disk sha256 {digest[:16]} != manifest "
                f"{self._expected[name][:16]}. The frozen baseline has been "
                "modified; reconciliation against it would be meaningless."
            )
        return payload

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")

    def read_json(self, name: str) -> dict:
        return json.loads(self.read_text(name))

    def report_json(self, factor: str, book: str) -> dict:
        return self.read_json(f"eval_{factor}_exec_{book}.json")

    def verify_all(self) -> tuple[int, list[str]]:
        """Verify every manifest entry. Returns (ok_count, problems)."""
        problems: list[str] = []
        ok = 0
        for name in self.names:
            try:
                self.read_bytes(name)
                ok += 1
            except (BaselineIntegrityError, FileNotFoundError) as exc:
                problems.append(f"{name}: {exc}")
        return ok, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--frozen-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the frozen tree against the manifest; copy nothing",
    )
    parser.add_argument("--source-note", default="")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    reports_dir = Path(args.reports_dir or repo_root / DEFAULT_REPORTS_DIR)
    frozen_root = Path(args.frozen_root or repo_root / DEFAULT_FROZEN_ROOT)
    manifest_path = Path(args.manifest or repo_root / DEFAULT_MANIFEST)

    if args.verify:
        baseline = FrozenExecBaseline(frozen_root, manifest_path)
        ok, problems = baseline.verify_all()
        print(f"verified {ok}/{baseline.file_count} frozen artifacts")
        print(f"frozen at git head: {baseline.frozen_at_git_head}")
        for problem in problems:
            print(f"  PROBLEM {problem}")
        return 1 if problems else 0

    result = freeze(
        reports_dir,
        frozen_root,
        manifest_path,
        repo_root=repo_root,
        source_note=args.source_note,
    )
    print(f"frozen root : {result.frozen_root}")
    print(f"manifest    : {result.manifest_path}")
    print(f"copied      : {len(result.copied)}")
    print(f"already ok  : {len(result.already_frozen)}")
    print(f"manifest    : {'REWRITTEN' if result.manifest_written else 'unchanged'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
