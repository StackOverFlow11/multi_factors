"""Builders for a SYNTHETIC frozen-baseline tree (D1 / D2 verify-only tests).

The real baseline is 109 MB of parquet and must never be written to, so the
verification tests operate on a miniature tree with the same shape: a
``panels/`` directory, a machine ``manifest.json`` (rows + reconciliation
block), an optional ``panels_d2/`` with its ``manifest_d2.json``, and a
git-tracked-style Markdown manifest document carrying the expected hashes.

The tree is built through the production writer (``atomic_write_parquet``) and
the production row/renderer helpers, so a synthetic tree is a valid frozen tree
by construction rather than by transcription.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from qt.panel_freeze import (
    RECONCILE_FLOAT_FIELDS,
    RECONCILE_INT_FIELDS,
    atomic_write_parquet,
    canonical_content_hash,
    manifest_row,
)

#: Two factors is enough to show that a problem is attributed to ONE of them.
DEFAULT_FACTORS = {"alpha_20": "minute", "book_x": "book"}
DATES = ("2024-01-02", "2024-01-03", "2024-01-04")
SYMBOLS = ("000001.SZ", "600000.SH")


def make_panel(factor_id: str, *, offset: float = 0.0, nan_last: bool = True) -> pd.Series:
    """A small, deterministic (date, symbol) factor panel."""
    keys = [(pd.Timestamp(d), s) for d in DATES for s in SYMBOLS]
    values = [offset + 0.5 * position for position in range(len(keys))]
    if nan_last:
        values[-1] = float("nan")
    index = pd.MultiIndex.from_tuples(keys, names=[DATE_LEVEL, SYMBOL_LEVEL])
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=factor_id)


def render_doc(rows: list[dict], *, producing_sha: str, include_file_sha: bool = True) -> str:
    """A Markdown manifest document in the shape the real ones use.

    Deliberately includes a decoy table BEFORE the manifest table (the real
    documents carry several), so the header-driven table finder is exercised
    rather than a "first table wins" shortcut.
    """
    columns = ["factor_id", "kind", "rows", "n_symbols", "n_nan", "mean", "std",
               "canonical_sha256"]
    if include_file_sha:
        columns.append("file_sha256")
    lines = [
        "# synthetic manifest",
        "",
        f"- **producing SHA**: `{producing_sha}`",
        "",
        "| item | value |",
        "|---|---|",
        "| universe | synthetic |",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append(repr(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_frozen_tree(
    root: Path,
    *,
    factors: dict[str, str] | None = None,
    producing_sha: str = "0" * 40,
    with_d2: bool = False,
    include_file_sha: bool = True,
) -> Path:
    """Write a complete synthetic frozen tree; return the manifest document path."""
    factors = dict(factors or DEFAULT_FACTORS)
    panels_dir = root / "panels"
    rows: list[dict] = []
    for position, (factor_id, kind) in enumerate(factors.items()):
        panel = make_panel(factor_id, offset=float(position))
        target = panels_dir / f"{factor_id}.parquet"
        file_hash = atomic_write_parquet(panel, target)
        rows.append(
            manifest_row(
                factor_id, kind, panel, canonical_content_hash(panel), file_hash,
                target.name,
            )
        )

    reconciliation = {
        row["factor_id"]: {
            "artifact": f"eval_{row['factor_id']}_no_book.json",
            "checks": {
                field: {"artifact": 1, "frozen": 1, "ok": True}
                for field in RECONCILE_INT_FIELDS + RECONCILE_FLOAT_FIELDS
            },
        }
        for row in rows
        if factors[row["factor_id"]] == "minute"
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "header": {"producing_git_sha": producing_sha, "panel_rows": 6},
                "rows": rows,
                "reconciliation": reconciliation,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if with_d2:
        d2_dir = root / "panels_d2"
        d2_rows = []
        for position, factor_id in enumerate(factors):
            panel = make_panel(factor_id, offset=float(position))
            atomic_write_parquet(panel, d2_dir / f"{factor_id}.parquet")
            d2_rows.append(
                {
                    "factor_id": factor_id,
                    "canonical_sha256": canonical_content_hash(panel),
                    "frozen_sha256": canonical_content_hash(panel),
                    "rows": int(len(panel)),
                }
            )
        (root / "manifest_d2.json").write_text(
            json.dumps(
                {
                    "producing_git_sha": producing_sha,
                    "header": {"producing_git_sha": producing_sha, "all_ok": True},
                    "rows": d2_rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    doc_path = root / "manifest_doc.md"
    doc_path.write_text(
        render_doc(rows, producing_sha=producing_sha, include_file_sha=include_file_sha),
        encoding="utf-8",
    )
    return doc_path


def rewrite_panel(root: Path, factor_id: str, panel: pd.Series, *, d2: bool = False) -> None:
    """Overwrite one panel of a synthetic tree (the tampering primitive)."""
    subdir = "panels_d2" if d2 else "panels"
    atomic_write_parquet(panel.rename(factor_id), root / subdir / f"{factor_id}.parquet")


def patch_manifest(root: Path, name: str, mutate) -> None:
    """Read/modify/write one of the synthetic tree's JSON manifests."""
    path = root / name
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
