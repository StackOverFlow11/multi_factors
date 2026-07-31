"""Parsers for the git-tracked frozen-baseline manifest DOCUMENTS.

The bulk artifacts of the factor-layer refactor baselines are gitignored; their
expected hashes are authored in Markdown documents under ``docs/factors/`` that
call themselves the authoritative manifest. Keeping the hashes in git and the
bytes out of it is what gives verification teeth: whoever regenerates a frozen
tree together with its machine manifest still cannot move the expectations
without it showing up in ``git diff``.

So the expectations are READ FROM those documents rather than transcribed into
code — a transcription is a second copy, and the copy in code is the one nobody
updates.

Parsing is header-driven and fail-closed. The table is located by requiring the
columns ``factor_id`` and ``canonical_sha256`` (documents carry several other
tables), every other column is read BY NAME, and anything ambiguous — two
matching tables, none, a duplicate factor id, a malformed hash — is a readable
error. A parser that silently returned an empty expectation set would make the
verifier pass everything.

Consumers: ``qt.panel_freeze`` (D1 baseline + the PR-C reference panel) and
``qt.panel_reconcile`` (the D2 rebuild, which is expected to be bit-identical to
the D1 baseline and is therefore verified against the SAME document).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA40 = re.compile(r"[0-9a-f]{40}")
#: Cell decorations the documents legitimately carry (bold emphasis on the
#: numbers a reader should notice, code ticks around hashes).
_CELL_NOISE = re.compile(r"[`*]")


@dataclass(frozen=True)
class DocManifestRow:
    """One factor's expectations as authored in the git-tracked manifest doc.

    Fields absent from a given document are ``None`` — the PR-C reference table
    carries no ``kind`` or ``file_sha256`` column, and a verifier that invented
    values for them would be checking its own invention.
    """

    factor_id: str
    canonical_sha256: str
    kind: str | None = None
    rows: int | None = None
    n_symbols: int | None = None
    n_nan: int | None = None
    mean: float | None = None
    std: float | None = None
    file_sha256: str | None = None


def _split_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [_CELL_NOISE.sub("", cell).strip() for cell in cells]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell) <= set("-: ") and "-" in cell for cell in cells)


def parse_doc_manifest(path: Path | str) -> dict[str, DocManifestRow]:
    """Parse the factor-hash table out of a git-tracked manifest document.

    Header-driven, not position-driven: the table is located by requiring the
    columns ``factor_id`` and ``canonical_sha256``, and every other column is
    read by NAME. Two matching tables, zero matching tables, a duplicate factor
    id, or a hash that is not 64 lowercase hex are all readable errors — a
    verifier whose expectations parsed to an empty dict would "pass" everything.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[dict[str, DocManifestRow]] = []
    index = 0
    while index < len(lines) - 1:
        cells = _split_table_row(lines[index]) if "|" in lines[index] else []
        if (
            "factor_id" in cells
            and "canonical_sha256" in cells
            and "|" in lines[index + 1]
            and _is_separator(_split_table_row(lines[index + 1]))
        ):
            columns = {name: position for position, name in enumerate(cells)}
            rows: dict[str, DocManifestRow] = {}
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
                values = _split_table_row(lines[cursor])
                cursor += 1
                if len(values) != len(cells):
                    raise ValueError(
                        f"{path.name}: manifest table row has {len(values)} cells but "
                        f"the header has {len(cells)}: {values!r}"
                    )
                row = _doc_row(path, columns, values)
                if row.factor_id in rows:
                    raise ValueError(
                        f"{path.name}: duplicate factor id {row.factor_id!r} in the "
                        "manifest table."
                    )
                rows[row.factor_id] = row
            if not rows:
                raise ValueError(f"{path.name}: manifest table has no data rows.")
            tables.append(rows)
            index = cursor
            continue
        index += 1
    if len(tables) != 1:
        raise ValueError(
            f"{path.name}: expected exactly ONE table carrying factor_id + "
            f"canonical_sha256; found {len(tables)}. The verifier refuses to guess "
            "which one is the manifest."
        )
    return tables[0]


def _doc_row(path: Path, columns: dict[str, int], values: list[str]) -> DocManifestRow:
    def cell(name: str) -> str | None:
        if name not in columns:
            return None
        text = values[columns[name]]
        return text or None

    canonical = cell("canonical_sha256") or ""
    if not _HEX64.match(canonical):
        raise ValueError(
            f"{path.name}: canonical_sha256 {canonical!r} is not 64 lowercase hex."
        )
    file_hash = cell("file_sha256")
    if file_hash is not None and not _HEX64.match(file_hash):
        raise ValueError(
            f"{path.name}: file_sha256 {file_hash!r} is not 64 lowercase hex."
        )
    factor_id = cell("factor_id")
    if not factor_id:
        raise ValueError(f"{path.name}: manifest table row has an empty factor_id.")

    def as_int(name: str) -> int | None:
        text = cell(name)
        return None if text is None else int(text)

    def as_float(name: str) -> float | None:
        text = cell(name)
        return None if text is None else float(text)

    return DocManifestRow(
        factor_id=factor_id,
        canonical_sha256=canonical,
        kind=cell("kind"),
        rows=as_int("rows"),
        n_symbols=as_int("n_symbols"),
        n_nan=as_int("n_nan"),
        mean=as_float("mean"),
        std=as_float("std"),
        file_sha256=file_hash,
    )


def parse_doc_producing_sha(path: Path | str) -> str | None:
    """The producing SHA the document pins, if it states one.

    Read from the document rather than transcribed into this module: the SHA
    would otherwise exist in two places, and the copy in code is the one nobody
    updates.
    """
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        lowered = line.lower()
        if "producing" in lowered and ("sha" in lowered or "git_sha" in lowered):
            found = _SHA40.search(line)
            if found:
                return found.group(0)
    return None


