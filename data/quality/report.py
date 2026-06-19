"""Report-only data-quality findings model + deterministic renderer (D3).

A :class:`QualityFinding` is a small immutable record describing one suspicious
pattern found in an upstream data frame. The helpers here aggregate findings into
a stable frame and render a concise, bounded, deterministic text/Markdown summary
suitable for logs or a future data-update artifact.

This layer is REPORT-ONLY: it never mutates data, never drops/repairs rows, and
never touches cache, feed, factor, alpha, portfolio, or runtime logic. Findings
and rendered text carry only dataset/check metadata + bounded {date/time, symbol,
value} examples — never a token, a secret-file path, or an unbounded symbol dump.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import pandas as pd

# severity levels. "hard" = a severe structural finding in the report; it is NOT
# a default exception (the layer stays report-only unless a caller opts in).
INFO = "info"
WARNING = "warning"
HARD = "hard"

_SEVERITY_ORDER = {HARD: 0, WARNING: 1, INFO: 2}

# bound on how many example rows a finding may carry (keeps reports small + safe).
MAX_EXAMPLES = 5

_FRAME_COLUMNS = ["dataset", "check", "severity", "count", "examples", "note"]


@dataclass(frozen=True)
class QualityFinding:
    """One report-only data-quality finding (immutable)."""

    dataset: str
    check: str
    severity: str
    count: int
    examples: tuple[dict, ...] = field(default_factory=tuple)
    note: str | None = None


def clean_value(value: object) -> object:
    """Coerce a cell to a small, deterministic, JSON-friendly example value.

    Timestamps -> ISO date (or date-time if intraday); numbers -> rounded
    python scalars; everything else -> ``str``. No paths, no secrets.
    """
    if isinstance(value, pd.Timestamp):
        if value == value.normalize():
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return round(float(value), 6)
    return str(value)


def make_finding(
    dataset: str,
    check: str,
    severity: str,
    count: int,
    examples: list[dict] | tuple[dict, ...] = (),
    note: str | None = None,
) -> QualityFinding:
    """Build a finding with bounded, cleaned examples (at most ``MAX_EXAMPLES``)."""
    if severity not in _SEVERITY_ORDER:
        raise ValueError(f"unknown severity {severity!r}; expected info/warning/hard")
    bounded = tuple(
        {k: clean_value(v) for k, v in dict(ex).items()}
        for ex in list(examples)[:MAX_EXAMPLES]
    )
    return QualityFinding(
        dataset=dataset,
        check=check,
        severity=severity,
        count=int(count),
        examples=bounded,
        note=note,
    )


def has_hard(findings: list[QualityFinding]) -> bool:
    """Whether any finding is ``hard`` (a report flag, not a default failure)."""
    return any(f.severity == HARD for f in findings)


def sort_findings(findings: list[QualityFinding]) -> list[QualityFinding]:
    """Deterministic order: severity (hard first), then dataset, then check."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.dataset, f.check),
    )


def findings_to_frame(findings: list[QualityFinding]) -> pd.DataFrame:
    """Collect findings into a stable, deterministically-ordered DataFrame."""
    if not findings:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    rows = [
        {
            "dataset": f.dataset,
            "check": f.check,
            "severity": f.severity,
            "count": f.count,
            "examples": f.examples,
            "note": f.note,
        }
        for f in sort_findings(findings)
    ]
    return pd.DataFrame(rows, columns=_FRAME_COLUMNS)


def _format_examples(examples: tuple[dict, ...]) -> str:
    """Compact deterministic ``k=v`` join of bounded example rows."""
    parts = []
    for ex in examples:
        kv = ", ".join(f"{k}={ex[k]}" for k in ex)
        parts.append(f"{{{kv}}}")
    return "; ".join(parts)


def render_report(
    findings: list[QualityFinding], *, title: str = "Data Quality Report"
) -> str:
    """Render a concise, bounded, deterministic Markdown summary.

    Clean input -> an explicit "no findings" line. Otherwise one bullet per
    finding (severity / dataset / check / count / note) plus its bounded examples.
    """
    lines = [f"# {title}", ""]
    if not findings:
        lines.append("No data-quality findings. ✓")
        return "\n".join(lines)

    ordered = sort_findings(findings)
    n_hard = sum(1 for f in ordered if f.severity == HARD)
    n_warn = sum(1 for f in ordered if f.severity == WARNING)
    n_info = sum(1 for f in ordered if f.severity == INFO)
    lines.append(
        f"{len(ordered)} finding(s): {n_hard} hard / {n_warn} warning / {n_info} info."
    )
    lines.append("")
    for f in ordered:
        head = f"- [{f.severity}] `{f.dataset}` / {f.check}: count={f.count}"
        if f.note:
            head += f" — {f.note}"
        lines.append(head)
        if f.examples:
            lines.append(f"  examples: {_format_examples(f.examples)}")
    return "\n".join(lines)
