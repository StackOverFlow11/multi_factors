"""Cross-basis verdict summaries: basis/view are NON-OMITTABLE columns (R24).

A single report states its own information set and return basis in its provenance
box. A SUMMARY does not: a table listing eleven factors' verdicts side by side is
the surface where a close-era "Watch" and an exec-era "Watch" get read as the same
claim, because nothing on the row says they are not. Design v3.2 §3.6 R24 makes
the rule structural: any multi-verdict rendering carries ``basis`` and ``view``
columns, and a caller that omits them gets a readable error instead of a table.

This is the summary-level form of the #76 lesson ("a rendering that does not state
its provenance"), and it is deliberately a REFUSAL rather than a default: filling
in a missing basis would be the silent degradation the project forbids, and
guessing it from the majority of the rows is worse than refusing.

Layering: pure formatting over plain mappings — no pandas, no qt, no store. The
row dicts come from whoever assembled them (a registry read, a directory of report
JSONs); this module only decides what a table is allowed to omit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from analytics.eval.contract import IDENTITY_FIELDS

#: Columns every cross-basis summary MUST carry. Same tuple the contract declares
#: as a report's identity fields — one list, not two spellings (author once).
REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = IDENTITY_FIELDS

_MISSING = "—"


def require_basis_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return ``columns`` as a tuple, or raise naming the omitted identity column.

    Split out from the renderer so a caller that builds a table some other way
    (HTML, a notebook) can enforce the same rule without re-deriving it.
    """
    resolved = tuple(str(c) for c in columns)
    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in resolved]
    if missing:
        raise ValueError(
            f"a cross-basis verdict summary must carry the identity column(s) "
            f"{missing}: a verdict from one view/return-basis is a DIFFERENT "
            f"statistical claim from the same word under another, and a table that "
            f"omits the column invites them to be read as one (design v3.2 §3.6 "
            f"R24). Add the column — the summary will not infer or default it."
        )
    return resolved


def _cell(row: Mapping[str, object], column: str) -> str:
    value = row.get(column, None)
    return _MISSING if value is None else str(value)


def render_verdict_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    title: str | None = None,
) -> str:
    """A deterministic Markdown table of many verdicts, basis/view guaranteed present.

    ``columns`` fixes the column order (so a diff of two summaries is a diff of the
    data). Every row must SUPPLY the identity columns as non-empty values: declaring
    the column and then leaving it blank would satisfy the letter of R24 and defeat
    its purpose, so it is rejected with the same readable error shape.
    """
    resolved = require_basis_columns(columns)
    for index, row in enumerate(rows):
        blank = [
            c
            for c in REQUIRED_SUMMARY_COLUMNS
            if not str(row.get(c, "") or "").strip()
        ]
        if blank:
            raise ValueError(
                f"cross-basis summary row {index} ({row.get('factor_id', '?')!r}) "
                f"leaves the identity column(s) {blank} empty. Declaring the column "
                f"and not filling it is the same omission R24 forbids."
            )
    lines: list[str] = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| " + " | ".join(resolved) + " |")
    lines.append("|" + "|".join("---" for _ in resolved) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(row, c) for c in resolved) + " |")
    return "\n".join(lines) + "\n"


__all__ = [
    "REQUIRED_SUMMARY_COLUMNS",
    "render_verdict_summary",
    "require_basis_columns",
]
