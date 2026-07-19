"""Presentation of a :class:`~analytics.eval.report.FactorEvalReport`.

Two deterministic renderings of the same data: the Markdown page (the 10 fixed
sections of design §5) and the machine-readable dict (the cross-run comparable
record). Split out of ``report.py`` so the data model stays the data model.

Import direction: ``report`` imports THIS module at runtime; the report types are
only needed for annotations here, so they are imported under ``TYPE_CHECKING`` —
no import cycle.

Secret-safety: every string that reaches a rendered/exported artifact passes
through the D3 quality layer's ``sanitize_text`` (token values, ``.config.json``
paths), and payload values through its bounded ``clean_value``. ``analytics``
importing ``data.quality`` is downstream->upstream (``analytics/factor.py``
already imports ``data.clean.schema``), so this reuses the redaction rules
instead of re-implementing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from analytics.eval.sections import MANDATORY_SECTIONS, Skipped
from data.quality.report import clean_value, sanitize_text

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from analytics.eval.report import FactorEvalReport
    from analytics.eval.sections import SectionLike

# Bound on ONE rendered payload value, mirroring the D3 quality layer's bounded
# examples (MAX_EXAMPLES=5): ``clean_value``'s fallback is ``str(value)``, so a
# payload that accidentally holds a big object (a whole panel's repr) would dump
# it into the artifact. A report is a summary; redaction has already run inside
# ``clean_value``, so truncating afterwards can never re-expose a secret.
MAX_VALUE_CHARS = 200
_TRUNCATED = "...[truncated]"

# Rendered labels for the three verdict axes (design §6, v0.5).
AXIS_TITLES: dict[str, str] = {
    "predictive": "Predictive",
    "incremental": "Incremental",
    "tradable": "Tradable",
}

# Rendered headings for the 8 mandatory sections (rendered as 2..9).
SECTION_TITLES: dict[str, str] = {
    "predictive_power": "Predictive Power",
    "return_risk": "Return & Risk",
    "stability_cost": "Stability & Cost",
    "purity": "Purity",
    "oos_generalization": "OOS & Generalization",
    "execution_capacity": "Execution & Capacity",
    "data_coverage": "Data & Coverage",
    "caveats": "Caveats & Provenance",
}


def sanitize_payload(value: object) -> object:
    """Recursively coerce a payload value to a deterministic, secret-free form.

    Mappings are key-sorted here, so a section's insertion order can never leak
    into the artifact (a report must be byte-identical across runs).

    KEYS are redacted too, not just values: a payload key is just as capable of
    carrying a token-shaped string into the artifact, and the D3 layer this
    reuses already redacts its example keys (``_format_examples``) — leaving
    keys raw here would be both leaky and inconsistent with it.
    """
    if isinstance(value, Mapping):
        # Sort the SOURCE by raw key first, then re-sort by the redacted key: two
        # distinct secret-shaped keys can redact to the same marker, and this
        # keeps which one survives (and the output order) deterministic.
        cleaned = {
            sanitize_text(str(k)): sanitize_payload(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
        return dict(sorted(cleaned.items()))
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(v) for v in value]
    if value is None or isinstance(value, bool):
        return value
    cleaned = clean_value(value)
    if isinstance(cleaned, str) and len(cleaned) > MAX_VALUE_CHARS:
        return cleaned[:MAX_VALUE_CHARS] + _TRUNCATED
    return cleaned


def format_value(value: object) -> str:
    """Compact deterministic rendering of one payload value."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {format_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    return str(value)


def canonical_sections(
    report: FactorEvalReport, mandatory: tuple[str, ...]
) -> list[tuple[str, SectionLike | None]]:
    """Sections in CANONICAL order, independent of how they were assembled.

    The mandatory set first, in the fixed design §5 order (an absent one yields
    ``None``, so every consumer must decide what to say about it), then any extra
    section sorted by name. Assembly order must never leak into an artifact: two
    reports carrying the same sections in a different order have to produce
    byte-identical output, or a cross-run diff turns into noise.
    """
    indexed = report.by_name()
    ordered: list[tuple[str, SectionLike | None]] = [
        (name, indexed.get(name)) for name in mandatory
    ]
    extras = sorted(
        (s for s in report.sections if s.name not in mandatory), key=lambda s: s.name
    )
    return ordered + [(s.name, s) for s in extras]


def report_to_dict(report: FactorEvalReport) -> dict:
    """Machine-readable record with a STABLE key order and CANONICAL section order."""
    verdict = report.require_verdict()
    sections: list[dict] = []
    for name, section in canonical_sections(report, MANDATORY_SECTIONS):
        if section is None:
            # Unreachable via the public API (with_verdict validates, and every
            # export requires a verdict); kept explicit so a hand-built report
            # can never export a SHORT record that looks structurally valid.
            # This mirrors the Markdown's _MISSING_ marker: the JSON states the
            # hole rather than omitting the section.
            sections.append(
                {
                    "name": name,
                    "reason": "MISSING — the contract requires this section.",
                    "status": "missing",
                }
            )
            continue
        if isinstance(section, Skipped):
            sections.append(
                {"name": section.name, "reason": section.reason, "status": "skipped"}
            )
            continue
        entry: dict[str, object] = {
            "name": section.name,
            "payload": sanitize_payload(section.payload),
            "status": "ok",
        }
        if section.note:
            entry["note"] = section.note
        sections.append(dict(sorted(entry.items())))
    return {
        "criteria_source": report.criteria_source,
        "eval_config": sanitize_payload(vars(report.cfg)),
        "schema_version": report.SCHEMA_VERSION,
        "sections": sections,
        "spec": sanitize_payload(vars(report.spec)),
        # the ACTUAL criteria values used, whatever their source (design §6, v0.6).
        "thresholds": sanitize_payload(vars(report.thresholds)),
        # The DERIVED deployment label + the three axis-verdicts it was derived
        # from (design §6, v0.5), so a cross-run record can compare axes, not just
        # the single label.
        "verdict": {
            "reasons": [sanitize_text(r) for r in verdict.reasons],
            "verdict": verdict.verdict,
            "axes": {
                name: {
                    "reasons": [sanitize_text(r) for r in axis.reasons],
                    "verdict": axis.verdict,
                }
                for name, axis in verdict.axes().items()
            },
        },
    }


def render_report(report: FactorEvalReport, mandatory: tuple[str, ...]) -> str:
    """Deterministic Markdown: the 10 fixed sections of design §5."""
    verdict = report.require_verdict()
    spec = report.spec
    lines = [f"# Factor Evaluation — {spec.factor_id} (v{spec.version})", ""]

    lines += ["## 0. Header & Provenance", ""]
    for label, value in _provenance_rows(report):
        lines.append(f"- {label}: {value}")
    lines.append("")

    lines += ["## 1. Verdict & Scorecard", "", f"**{verdict.verdict}**", ""]
    lines += [f"- {reason}" for reason in verdict.reasons]
    lines.append("")
    # The three axis-verdicts the deployment label was derived from (design §6,
    # v0.5). Rendered under the label so the reader sees WHICH axis carried it.
    lines.append("Axes:")
    for name, axis in verdict.axes().items():
        detail = f" — {sanitize_text(axis.reasons[0])}" if axis.reasons else ""
        lines.append(f"- {AXIS_TITLES.get(name, name)}: {axis.verdict}{detail}")
    lines.append("")
    if report.criteria_source == "declared":
        criteria_prefix = (
            "Success criteria (PRE-REGISTERED via EvalConfig.success_criteria — "
            "declared before the run)"
        )
    else:
        criteria_prefix = (
            "Success criteria (source: DEFAULT global bar — ⚠️ unvalidated defaults, "
            "pending calibration; NOT pre-registered for this run)"
        )
    lines.append(
        criteria_prefix
        + ": "
        + ", ".join(f"{k}={v}" for k, v in sorted(vars(report.thresholds).items()))
    )
    lines.append("")

    for index, (name, section) in enumerate(canonical_sections(report, mandatory)):
        # the mandatory set renders as the fixed sections 2..9; extras follow
        # unnumbered (they are additions to the contract, not part of it).
        number = index + 2 if index < len(mandatory) else None
        lines += _render_section(number, name, section)
    return "\n".join(lines).rstrip() + "\n"


def _provenance_rows(report: FactorEvalReport) -> list[tuple[str, str]]:
    spec, cfg = report.spec, report.cfg
    rows: list[tuple[str, object]] = [
        ("factor_id", spec.factor_id),
        ("description", spec.description),
        ("family", spec.family),
        ("hypothesis (expected IC sign, fixed pre-run)", f"{spec.expected_ic_sign:+d}"),
        ("horizon / return basis", f"h={spec.forward_return_horizon} / {spec.return_basis}"),
        ("input fields", ", ".join(spec.input_fields)),
        ("price adjust / warm-up bars", f"{spec.price_adjust} / {spec.min_history_bars}"),
        ("universe", f"{cfg.universe} (PIT={cfg.universe_is_pit})"),
        ("window / rebalance", f"{cfg.start} .. {cfg.end} / {cfg.rebalance}"),
        ("quantiles / long-short", f"{cfg.n_quantiles} / {cfg.long_short}"),
        ("cost scenarios", ", ".join(f"{c}x" for c in cfg.cost_scenarios)),
        (
            "processing",
            f"winsorize={cfg.winsorize}, standardize={cfg.standardize}, "
            f"neutralization={cfg.neutralization} ({cfg.industry_level})",
        ),
        ("oos split", cfg.oos_split),
        ("independent cells", len(cfg.independent_cells)),
        (
            "honesty flags",
            f"exploratory={cfg.is_exploratory}, post_hoc_selected={cfg.post_hoc_selected}, "
            f"tuned={cfg.tuned}, n_factors_screened={cfg.n_factors_screened}",
        ),
        (
            "verdict criteria source",
            f"{report.criteria_source} "
            + (
                "(pre-registered in EvalConfig.success_criteria)"
                if report.criteria_source == "declared"
                else "(global default bar — not pre-registered for this run)"
            ),
        ),
        ("data snapshot", cfg.data_snapshot_id),
    ]
    if spec.is_intraday:
        rows.append(
            (
                "intraday contract",
                f"cutoff={spec.decision_cutoff}, lag={spec.data_lag}, "
                f"session_open={spec.session_open}, "
                f"execution_model={spec.execution_model}, "
                f"window={spec.execution_window}",
            )
        )
    return [(label, sanitize_text(str(value))) for label, value in rows]


def _render_section(
    number: int | None, name: str, section: SectionLike | None
) -> list[str]:
    title = SECTION_TITLES.get(name, name)
    heading = f"## {number}. {title}" if number is not None else f"## + {title}"
    lines = [heading, ""]
    if section is None:
        # unreachable via the public API (with_verdict validates and render
        # requires a verdict); kept explicit so a hand-built report can never
        # render a silent hole. The JSON export marks it too ("status":
        # "missing") — the two renderings must not disagree about a hole.
        return lines + ["_MISSING — the contract requires this section._", ""]
    if isinstance(section, Skipped):
        return lines + [f"_Skipped: {section.reason}_", ""]
    if section.note:
        lines += [section.note, ""]
    payload = sanitize_payload(section.payload)
    if not payload:
        return lines + ["_No metrics reported._", ""]
    lines += [f"- {key}: {format_value(payload[key])}" for key in sorted(payload)]
    lines.append("")
    return lines


__all__ = [
    "MAX_VALUE_CHARS",
    "AXIS_TITLES",
    "SECTION_TITLES",
    "canonical_sections",
    "format_value",
    "render_report",
    "report_to_dict",
    "sanitize_payload",
]
