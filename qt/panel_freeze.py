"""D1 RAW factor-value panel baseline — VERIFY-ONLY since D5 C6.

The 14 closing factors' RAW daily factor-value panels (11 minute-derived + 3
confirmed book factors) were frozen once, on the eleven-factor evaluation data
plane (CSI500 ``000905.SH``, 2021-07-01 .. 2026-06-30, CACHE-ONLY), from ``main``
at the producing SHA recorded in ``docs/factors/d1_panel_freeze_manifest.md``.
They are the ONLY baseline the D5 cell-by-cell reconciliation compares against.

REGENERATION IS RETIRED (owner ruling 2026-07-28, D5 C6)
---------------------------------------------------------
This module used to rebuild those panels by calling the eleven legacy
``qt/eval_*.py`` runners' private ``_load_*_panel`` loaders. C6 deletes those
runners, so the rebuild could not work again; rather than leave a code path that
breaks the moment it is used, the capability is retired outright and the
baseline is **frozen-forever**. What remains is verification: read the frozen
bytes and check them against the manifest.

That direction is the safe one. The provenance rule (design v3.2 §5 leg 4) was
always that regenerating this baseline is legitimate ONLY from a checkout of the
pinned pre-D2 SHA — never from current code, because rebuilding the baseline
from the tree being validated makes the D5 reconciliation structurally unable to
fail (the ``compare_postmerge.py`` empty-reconciliation failure mode this repo
committed once). Retiring the rebuild removes the only way to violate that rule
by accident.

Where the authority lives
-------------------------
The bulk panels are gitignored; **the hashes are in git**, in the manifest
document's table. That split is what gives verification teeth: someone who
regenerates the panels and their machine manifest cannot also silently move the
expected hashes, because those show up in ``git diff``. The gitignored
``manifest.json`` is checked too — as a second, richer witness (row counts,
NaN counts, mean/std, file sha256) that must AGREE with the git-tracked table.

Canonical content hash
----------------------
A parquet file's byte hash depends on writer metadata (library versions, page
layout), so the AUTHORITATIVE fingerprint is a canonical CONTENT hash, defined
as sha256 over, in order:

1. the version tag ``"qt.panel_freeze canonical v1"`` + ``\\n``;
2. the row count (ascii) + ``\\n``;
3. the ``date`` level as little-endian int64 epoch-nanoseconds, in canonical
   row order, + ``\\n``;
4. the ``symbol`` level joined with ``\\x1f`` (utf-8), same order, + ``\\n``;
5. the values as little-endian float64 raw bytes, same order, with every NaN
   rewritten to the single canonical ``np.float64("nan")`` bit pattern.

Canonical row order = sort by (date, symbol); duplicate (date, symbol) keys are
an error (the hash would otherwise be order-ambiguous). The column NAME is not
hashed (content equality, not labelling). NaN payload bits are collapsed on
purpose; +0.0 and -0.0 stay distinct (a real IEEE value difference).

Run (deliberately NOT registered in qt/cli, as before)::

    python -m qt.panel_freeze --verify

There is no partial verification. A frozen-forever baseline verified in part is
a baseline you believe more than you checked, so ``--verify`` always covers
every panel in both frozen trees (the D1 baseline and the PR-C corrected
``jump_amount_corr_20`` reference panel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from qt.frozen_manifest_doc import (
    DocManifestRow,
    parse_doc_manifest,
    parse_doc_producing_sha,
)

CANONICAL_HASH_VERSION = "qt.panel_freeze canonical v1"
DEFAULT_CONFIG = "config/phase_c_jump_amount_corr.yaml"
DEFAULT_OUTPUT_ROOT = "artifacts/refactor_baseline"
#: PanelStore artifact name the freeze used — kept because the frozen
#: ``manifest.json`` header records it and the verifier reads that header.
PANEL_STORE_NAME = "d1_panel_freeze_daily"
#: The git-tracked authority for the D1 baseline (14 panels).
D1_MANIFEST_DOC = "docs/factors/d1_panel_freeze_manifest.md"
#: The git-tracked authority for the PR-C corrected jump reference panel (1).
PR_C_MANIFEST_DOC = "docs/factors/pr_c_cutoff_fix_reference_panel.md"
#: Sub-root of the PR-C reference panel, relative to the baseline root.
PR_C_SUBDIR = "pr_c_cutoff_fix"
#: data_coverage payload fields the freeze reconciled against the shipped eval
#: JSONs BEFORE accepting each panel. The reconciliation itself is history; the
#: field list is not, because ``manifest.json`` records the per-field outcome
#: and verification re-reads that record.
RECONCILE_INT_FIELDS = (
    "panel_rows",
    "evaluation_periods",
    "symbols_evaluated",
    "universe_symbols_declared",
    "dropped_symbols_count",
)
RECONCILE_FLOAT_FIELDS = ("factor_nan_rate",)

#: The single authored statement of what C6 retired. Every tool that retired a
#: regeneration path COMPOSES this rather than restating it: a regex cannot
#: assert that no other sentence says the same thing, but "there is no other
#: sentence" can (CLAUDE.md methodology #2).
RETIREMENT_RULING = "owner ruling 2026-07-28, D5 C6"


class RegenerationRetiredError(RuntimeError):
    """Raised when a retired regeneration entry point is invoked.

    A retired capability must fail LOUDLY and READABLY (redline #9). Letting the
    call succeed as a no-op, or die on a bare ``ImportError`` from a deleted
    runner, both leave the caller guessing whether anything happened.
    """


def retirement_message(tool: str, product: str, why: str, verifies: str) -> str:
    """The retirement error text, composed from one authored core plus two
    per-tool clauses.

    Only what is true of ALL THREE retired tools is shared: the ruling, the fact
    that this repository no longer rebuilds the product, and the pointer to
    ``--verify``. Both ``why`` (what the rebuild depended on) and ``verifies``
    (what verification actually checks) are supplied by the caller, because the
    three tools differ on both counts — they did not lean on the legacy runners
    to the same degree, and they do not verify the same kind of thing. A single
    shared sentence covering those would have to overstate one of them, which is
    this repository's recurring defect: text that describes a check other than
    the one performed.
    """
    return (
        f"{tool}: regeneration is RETIRED ({RETIREMENT_RULING}). This repository "
        f"no longer rebuilds {product}. {why} "
        f"Run `{tool} --verify` instead: {verifies}"
    )


# --------------------------------------------------------------------------- #
# Canonical content hash + atomic write (pure, network-free, unit-tested)
# --------------------------------------------------------------------------- #
def _as_single_series(panel: pd.Series | pd.DataFrame) -> pd.Series:
    """Validate and return the one factor series of ``panel`` (no coercion).

    Requires a 2-level MultiIndex named exactly (``date``, ``symbol``) with a
    ``datetime64[ns]`` date level, unique (date, symbol) keys, and numeric
    values. Anything else raises — the freeze never silently reshapes.
    """
    if isinstance(panel, pd.DataFrame):
        if panel.shape[1] != 1:
            raise ValueError(
                f"expected a single-column factor panel; got {panel.shape[1]} columns "
                f"({list(panel.columns)!r})."
            )
        series = panel.iloc[:, 0]
    elif isinstance(panel, pd.Series):
        series = panel
    else:
        raise TypeError(
            f"expected a pandas Series or single-column DataFrame; got "
            f"{type(panel).__name__}."
        )
    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("factor panel index must be a 2-level MultiIndex(date, symbol).")
    if tuple(index.names) != (DATE_LEVEL, SYMBOL_LEVEL):
        raise ValueError(
            f"factor panel index levels must be named ({DATE_LEVEL!r}, {SYMBOL_LEVEL!r}); "
            f"got {tuple(index.names)!r}."
        )
    dates = index.get_level_values(DATE_LEVEL)
    if str(dates.dtype) != "datetime64[ns]":
        raise ValueError(
            f"date level must be datetime64[ns]; got {dates.dtype}. The freeze does "
            "not coerce dtypes — fix the producer."
        )
    if index.has_duplicates:
        dupes = index[index.duplicated()].tolist()[:3]
        raise ValueError(
            f"factor panel has duplicate (date, symbol) keys (e.g. {dupes!r}); the "
            "canonical hash would be order-ambiguous."
        )
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"factor values must be numeric; got dtype {series.dtype}.")
    return series


def canonical_content_hash(panel: pd.Series | pd.DataFrame) -> str:
    """The authoritative content fingerprint (module docstring definition).

    Row-order independent (canonical sort applied first), sensitive to any
    value or (date, symbol) index change, NaN-payload independent.
    """
    series = _as_single_series(panel).sort_index(kind="mergesort")
    dates = series.index.get_level_values(DATE_LEVEL)
    symbols = series.index.get_level_values(SYMBOL_LEVEL)
    values = np.array(series.to_numpy(), dtype="<f8", copy=True)
    values[np.isnan(values)] = np.float64("nan")  # collapse NaN payload bits
    digest = hashlib.sha256()
    digest.update(CANONICAL_HASH_VERSION.encode("utf-8") + b"\n")
    digest.update(str(len(series)).encode("ascii") + b"\n")
    digest.update(np.ascontiguousarray(dates.asi8, dtype="<i8").tobytes() + b"\n")
    digest.update("\x1f".join(str(s) for s in symbols).encode("utf-8") + b"\n")
    digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_write_parquet(panel: pd.Series | pd.DataFrame, path: Path | str) -> str:
    """Atomically write the panel (canonical row order) as parquet; return file sha256.

    tmp-file + ``os.replace`` in the target directory: readers never observe a
    partial file, and a failed write leaves no tmp residue (and never touches an
    existing target).

    This is the byte-layout contract the frozen panels were laid down with. No
    production caller remains after C6 (the freeze that used it is retired); it
    is kept because it DEFINES the on-disk form the verifier reads, and because
    a verifier that cannot construct a valid frozen tree cannot be given a
    negative test.
    """
    series = _as_single_series(panel).sort_index(kind="mergesort")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        series.to_frame().reset_index().to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)  # failure path only; os.replace consumed it on success
    return file_sha256(target)


def file_sha256(path: Path | str) -> str:
    """sha256 of the file bytes (convenience fingerprint; NOT the authority)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen_panel(path: Path | str, factor_id: str) -> pd.Series:
    """Read one frozen panel back into the canonical MultiIndex(date, symbol) series."""
    frame = pd.read_parquet(path)
    if factor_id not in frame.columns:
        raise ValueError(
            f"frozen panel {Path(path).name} carries no column {factor_id!r} "
            f"(columns: {list(frame.columns)!r})."
        )
    series = frame.set_index([DATE_LEVEL, SYMBOL_LEVEL])[factor_id]
    return _as_single_series(series).sort_index(kind="mergesort")


# --------------------------------------------------------------------------- #
# Manifest rows + rendering (pure, unit-tested)
# --------------------------------------------------------------------------- #
MANIFEST_ROW_FIELDS = (
    "factor_id",
    "kind",
    "rows",
    "date_min",
    "date_max",
    "n_symbols",
    "n_nan",
    "mean",
    "std",
    "canonical_sha256",
    "file_sha256",
    "file",
)


def manifest_row(
    factor_id: str,
    kind: str,
    panel: pd.Series | pd.DataFrame,
    canonical_sha256: str,
    file_sha256_hex: str,
    file_name: str,
) -> dict:
    """One manifest record. mean/std are float64 ``Series.mean()``/``std(ddof=1)``
    with NaN skipped (pandas defaults), reported at full precision.

    Still live after C6: verification RE-DERIVES each row from the frozen panel
    and compares it field by field with the recorded one, so a manifest whose
    hash was patched but whose statistics were not is caught as well.
    """
    series = _as_single_series(panel)
    dates = series.index.get_level_values(DATE_LEVEL)
    return {
        "factor_id": str(factor_id),
        "kind": str(kind),
        "rows": int(len(series)),
        "date_min": dates.min().strftime("%Y-%m-%d") if len(series) else None,
        "date_max": dates.max().strftime("%Y-%m-%d") if len(series) else None,
        "n_symbols": int(series.index.get_level_values(SYMBOL_LEVEL).nunique()),
        "n_nan": int(series.isna().sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "canonical_sha256": canonical_sha256,
        "file_sha256": file_sha256_hex,
        "file": str(file_name),
    }


def render_manifest_markdown(header: dict, rows: list[dict]) -> str:
    """Deterministic Markdown manifest: a header block + one row per factor.

    This is the format definition of the frozen ``manifest.md``; ``--verify
    --show-manifest`` re-renders the table from the panels actually on disk so
    it can be eyeballed (or diffed) against the frozen one.
    """
    lines = ["# D1 panel freeze manifest", ""]
    for key in sorted(header):
        lines.append(f"- **{key}**: {header[key]}")
    lines.append("")
    cols = list(MANIFEST_ROW_FIELDS)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for row in rows:
        cells = []
        for col in cols:
            value = row.get(col)
            cells.append(repr(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelVerification:
    """One frozen panel's verification outcome (no secrets)."""

    factor_id: str
    file: str
    rows: int
    canonical_sha256: str
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class VerifyResult:
    """A whole frozen tree's verification outcome."""

    root: Path
    doc: Path
    panels: tuple[PanelVerification, ...]
    problems: tuple[str, ...]
    producing_git_sha: str | None
    rendered_manifest: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems and all(panel.ok for panel in self.panels)

    @property
    def n_ok(self) -> int:
        return sum(1 for panel in self.panels if panel.ok)


def _load_machine_manifest(root: Path) -> dict:
    """The gitignored ``manifest.json`` document, or ``{}`` when absent."""
    path = root / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check_recorded_reconciliation(document: dict, problems: list[str]) -> None:
    """The freeze reconciled each minute panel against its shipped eval JSON and
    recorded the per-field outcome. Verification re-reads that record.

    This is deliberately a check of the RECORD, not a re-run: re-running it
    needs the processed series, which needs the data plane, which needs the
    loaders C6 deletes. What can still be asserted is that the record says what
    a passing reconciliation says — every declared field present, every check
    ``ok`` — so a manifest edited to paper over a failure is caught.
    """
    recorded = document.get("reconciliation", {})
    if not recorded:
        problems.append(
            "manifest.json carries no reconciliation block; the frozen panels' "
            "agreement with the shipped eval artifacts is unwitnessed."
        )
        return
    for factor_id in sorted(recorded):
        entry = recorded[factor_id] or {}
        artifact = entry.get("artifact")
        if not artifact:
            problems.append(f"reconciliation[{factor_id}] names no eval artifact")
        checks = entry.get("checks", {})
        for field in RECONCILE_INT_FIELDS + RECONCILE_FLOAT_FIELDS:
            check = checks.get(field)
            if check is None:
                problems.append(
                    f"reconciliation[{factor_id}] is missing the {field!r} check"
                )
            elif not check.get("ok"):
                problems.append(
                    f"reconciliation[{factor_id}].{field} was recorded as NOT ok "
                    f"({check.get('artifact')!r} vs {check.get('frozen')!r}) — a "
                    "divergent panel must never have been frozen."
                )


def verify_frozen_panels(
    root: Path | str,
    doc_path: Path | str,
    *,
    show_manifest: bool = False,
) -> VerifyResult:
    """Check a frozen panel tree against its git-tracked manifest document.

    Per panel, four independent statements must hold:

    1. the recomputed canonical content hash equals the hash AUTHORED IN GIT;
    2. it also equals the hash in the gitignored ``manifest.json`` (so a tree
       regenerated together with its machine manifest is still caught: the two
       witnesses would agree with each other and disagree with git);
    3. the whole manifest row re-derived from the panel (rows / date range /
       symbol count / NaN count / mean / std / file sha256) equals the recorded
       one — a patched hash with stale statistics does not survive this;
    4. the document's own columns (rows / n_symbols / n_nan / mean / std where
       the document carries them) agree as well.

    Plus tree-level statements: the set of ``*.parquet`` files equals the set of
    factor ids in the document (an EXTRA panel is a problem, not a bonus), and
    the recorded producing SHA equals the one the document pins.
    """
    root = Path(root)
    doc_path = Path(doc_path)
    problems: list[str] = []

    expected = parse_doc_manifest(doc_path)
    document = _load_machine_manifest(root)
    if not document:
        problems.append(
            f"machine manifest missing: {root / 'manifest.json'} — the frozen "
            "tree is incomplete (the git-tracked document alone cannot witness "
            "row counts, NaN counts, mean/std or the file hashes)."
        )
    else:
        _check_recorded_reconciliation(document, problems)
    machine_rows = {str(row["factor_id"]): row for row in document.get("rows", [])}
    header = document.get("header", {})

    doc_sha = parse_doc_producing_sha(doc_path)
    recorded_sha = header.get("producing_git_sha")
    if doc_sha and recorded_sha and doc_sha != recorded_sha:
        problems.append(
            f"producing SHA mismatch: {doc_path.name} pins {doc_sha} but "
            f"manifest.json records {recorded_sha} — this tree was not produced "
            "by the run the document describes."
        )

    panels_dir = root / "panels"
    if not panels_dir.is_dir():
        problems.append(f"frozen panels directory not found: {panels_dir}")
        return VerifyResult(
            root=root,
            doc=doc_path,
            panels=(),
            problems=tuple(problems),
            producing_git_sha=recorded_sha,
        )

    on_disk = {path.stem for path in panels_dir.glob("*.parquet")}
    for extra in sorted(on_disk - set(expected)):
        problems.append(
            f"unregistered panel on disk: {extra}.parquet is not in "
            f"{doc_path.name}; the frozen inventory is defined by the document."
        )

    verifications: list[PanelVerification] = []
    rendered_rows: list[dict] = []
    for factor_id in sorted(expected):
        verification, rendered = _verify_one_panel(
            factor_id, expected[factor_id], panels_dir, machine_rows
        )
        verifications.append(verification)
        if rendered is not None:
            rendered_rows.append(rendered)

    rendered_manifest = (
        render_manifest_markdown(dict(header), rendered_rows) if show_manifest else None
    )
    return VerifyResult(
        root=root,
        doc=doc_path,
        panels=tuple(verifications),
        problems=tuple(problems),
        producing_git_sha=recorded_sha,
        rendered_manifest=rendered_manifest,
    )


def _verify_one_panel(
    factor_id: str,
    expected: DocManifestRow,
    panels_dir: Path,
    machine_rows: dict[str, dict],
) -> tuple[PanelVerification, dict | None]:
    problems: list[str] = []
    path = panels_dir / f"{factor_id}.parquet"
    if not path.exists():
        return (
            PanelVerification(
                factor_id=factor_id,
                file=path.name,
                rows=0,
                canonical_sha256="",
                problems=(f"frozen panel missing from disk: {path}",),
            ),
            None,
        )
    try:
        panel = read_frozen_panel(path, factor_id)
    except (ValueError, TypeError, OSError) as exc:
        return (
            PanelVerification(
                factor_id=factor_id,
                file=path.name,
                rows=0,
                canonical_sha256="",
                problems=(f"frozen panel unreadable: {type(exc).__name__}: {exc}",),
            ),
            None,
        )

    canonical = canonical_content_hash(panel)
    derived = manifest_row(
        factor_id,
        str(expected.kind or (machine_rows.get(factor_id, {}) or {}).get("kind", "")),
        panel,
        canonical,
        file_sha256(path),
        path.name,
    )

    if canonical != expected.canonical_sha256:
        problems.append(
            f"canonical content hash {canonical} != the git-tracked "
            f"{expected.canonical_sha256} — these are NOT the frozen bytes."
        )
    for field, want in (
        ("rows", expected.rows),
        ("n_symbols", expected.n_symbols),
        ("n_nan", expected.n_nan),
        ("mean", expected.mean),
        ("std", expected.std),
    ):
        if want is not None and derived[field] != want:
            problems.append(
                f"{field}: git-tracked document says {want!r}, panel gives "
                f"{derived[field]!r}"
            )
    if expected.file_sha256 is not None and derived["file_sha256"] != expected.file_sha256:
        problems.append(
            f"file sha256 {derived['file_sha256']} != the git-tracked "
            f"{expected.file_sha256} — the file was rewritten"
            + (
                " (its CONTENT still matches, so this is a re-write, not a value "
                "change — but a frozen-forever file has no legitimate reason to "
                "be rewritten)."
                if canonical == expected.canonical_sha256
                else "."
            )
        )

    recorded = machine_rows.get(factor_id)
    if recorded is None:
        problems.append("no row for this factor in manifest.json")
    else:
        for field in MANIFEST_ROW_FIELDS:
            if field not in recorded:
                problems.append(f"manifest.json row is missing field {field!r}")
                continue
            if recorded[field] != derived[field]:
                problems.append(
                    f"manifest.json {field}: recorded {recorded[field]!r}, panel "
                    f"gives {derived[field]!r}"
                )

    return (
        PanelVerification(
            factor_id=factor_id,
            file=path.name,
            rows=int(derived["rows"]),
            canonical_sha256=canonical,
            problems=tuple(problems),
        ),
        derived,
    )


def verify_all(
    baseline_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    show_manifest: bool = False,
) -> list[VerifyResult]:
    """Verify BOTH frozen panel trees: the D1 baseline and the PR-C reference."""
    baseline_root = Path(baseline_root)
    repo_root = repo_root_for(baseline_root)
    return [
        verify_frozen_panels(
            baseline_root, repo_root / D1_MANIFEST_DOC, show_manifest=show_manifest
        ),
        verify_frozen_panels(
            baseline_root / PR_C_SUBDIR,
            repo_root / PR_C_MANIFEST_DOC,
            show_manifest=show_manifest,
        ),
    ]


def repo_root_for(baseline_root: Path | str) -> Path:
    """Where to look for the git-tracked manifest documents.

    ``docs/`` sits next to the repository root, while the baseline root may be
    an arbitrary directory (a copy under test, say). Walk up from the baseline
    root looking for ``docs/factors``; fall back to this module's own repository
    so a copied tree still verifies against the documents in git.
    """
    for candidate in (baseline_root, *baseline_root.parents):
        if (candidate / D1_MANIFEST_DOC).exists():
            return candidate
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Retired regeneration entry point
# --------------------------------------------------------------------------- #
def run_panel_freeze(*args, **kwargs):
    """RETIRED (D5 C6). Always raises :class:`RegenerationRetiredError`.

    Kept as a stub rather than deleted outright so that a stale caller — a
    script, a notebook, a copied command line — gets the retirement explanation
    instead of an ``AttributeError`` that reads like a packaging accident.
    """
    raise RegenerationRetiredError(
        retirement_message(
            "python -m qt.panel_freeze",
            "the D1 RAW factor-value panel baseline, which is frozen-forever",
            "The rebuild called the eleven legacy eval runners' private "
            "`_load_*_panel` loaders, which C6 deletes; and rebuilding this "
            "baseline from the tree under validation is exactly the "
            "empty-reconciliation failure mode the provenance rule forbids.",
            "it recomputes every frozen panel's canonical content hash and "
            "manifest row and checks them against the hashes authored in "
            "`docs/factors/`.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m qt.panel_freeze --verify``.

    The retired flags are still ACCEPTED by the parser so that someone typing
    the old documented command gets the retirement explanation rather than
    ``unrecognized arguments: --resume``, which reads like a version mismatch.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--only", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="frozen baseline root (default: the --output-root location)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the frozen panels against the git-tracked manifest documents",
    )
    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="also print the manifest table re-derived from the panels on disk",
    )
    args = parser.parse_args(argv)

    if not args.verify:
        try:
            run_panel_freeze()
        except RegenerationRetiredError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    root = Path(args.baseline_root or args.output_root)
    results = verify_all(root, show_manifest=args.show_manifest)
    failed = False
    for result in results:
        print(
            f"{result.root}: verified {result.n_ok}/{len(result.panels)} panels "
            f"against {result.doc.name} (producing sha "
            f"{result.producing_git_sha or 'unrecorded'})"
        )
        for problem in result.problems:
            print(f"  PROBLEM {problem}")
        for panel in result.panels:
            for problem in panel.problems:
                print(f"  PROBLEM {panel.factor_id}: {problem}")
        if result.rendered_manifest:
            print(result.rendered_manifest)
        failed = failed or not result.ok
    print("panel freeze verification: " + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
