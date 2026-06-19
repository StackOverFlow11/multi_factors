"""D3b: report-only data-quality hook for the ``data-update`` job.

Surfaces the accepted D3 ``data/quality`` checks in operations. When the hook is
enabled it runs the D3 STRUCTURAL checks on the frames the updater ALREADY warmed
(market bars + 1min minutes) and writes a deterministic Markdown report. This is
report-only: it NEVER filters / repairs / mutates data, never fails the job, never
changes cache coverage or the per-endpoint request summary, and makes NO extra API
call for quality.

No exchange calendar is synthesized here — the missing-date / missing-minute D3
checks need an explicit calendar and stay off in D3b (structural checks only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.quality import (
    HARD,
    QualityFinding,
    render_report,
    run_adj_factor_checks,
    run_intraday_checks,
    run_market_checks,
)

# Only the structural endpoints the updater loads as in-memory frames can be
# quality-checked without new API calls (mirrors the config-level allow-list).
QUALITY_ENDPOINTS: tuple[str, ...] = ("market_daily", "adj_factor", "stk_mins_1min")

_REPORT_TITLE = "Data Update Quality Report"


@dataclass(frozen=True)
class QualityOutcome:
    """Outcome of the D3b quality hook (immutable)."""

    report_path: Path
    findings_count: int
    hard_count: int
    checked_endpoints: tuple[str, ...]


def collect_findings(
    *,
    selected: list[str],
    warmed: set[str],
    market_frame: pd.DataFrame | None,
    intraday_frame: pd.DataFrame | None,
) -> tuple[list[QualityFinding], list[str]]:
    """Run the D3 checks on already-warmed frames; return (findings, checked).

    A surface is checked only when it is BOTH selected for quality (``selected``)
    AND was warmed by ``data_update.endpoints`` (``warmed``) — so the hook never
    checks a frame the updater did not load. Structural checks only: no calendar
    is passed, so missing-date / missing-minute checks stay off (D3b scope).
    """
    chosen = set(selected)
    findings: list[QualityFinding] = []
    checked: list[str] = []
    if "market_daily" in chosen and "market_daily" in warmed and market_frame is not None:
        findings.extend(run_market_checks(market_frame))
        checked.append("market_daily")
    if "adj_factor" in chosen and "adj_factor" in warmed and market_frame is not None:
        findings.extend(run_adj_factor_checks(market_frame))
        checked.append("adj_factor")
    if (
        "stk_mins_1min" in chosen
        and "stk_mins_1min" in warmed
        and intraday_frame is not None
    ):
        findings.extend(run_intraday_checks(intraday_frame))
        checked.append("stk_mins_1min")
    return findings, checked


def _run_context(
    *, window_start, window_end, n_symbols: int, checked_endpoints: list[str]
) -> str:
    """A small, non-secret run-context block (window / symbol COUNT / endpoints).

    Carries only metadata: the date window, the NUMBER of symbols (never the
    symbol list), and the endpoint names checked. No token, no secret-file path,
    no raw config.
    """
    eps = ", ".join(checked_endpoints) if checked_endpoints else "(none)"
    return (
        "## Run context\n\n"
        f"- window: {pd.Timestamp(window_start).date()} .. "
        f"{pd.Timestamp(window_end).date()}\n"
        f"- symbols checked: {int(n_symbols)}\n"
        f"- endpoints checked: {eps}\n"
    )


def write_quality_report(
    findings: list[QualityFinding],
    *,
    report_dir: str,
    report_name: str,
    window_start,
    window_end,
    n_symbols: int,
    checked_endpoints: list[str],
) -> QualityOutcome:
    """Render + write the deterministic report (even when clean); return the outcome.

    Uses the D3 ``render_report`` (bounded, redacted examples) plus a small
    non-secret run-context block. The report is written even with zero findings so
    operations get an explicit "no findings" artifact. ``report_name`` is assumed
    pre-validated as a bare filename (see ``DataUpdateQualityCfg``).
    """
    body = render_report(findings, title=_REPORT_TITLE)
    context = _run_context(
        window_start=window_start,
        window_end=window_end,
        n_symbols=n_symbols,
        checked_endpoints=checked_endpoints,
    )
    text = f"{body}\n\n{context}"

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / report_name
    path.write_text(text, encoding="utf-8")

    hard_count = sum(1 for f in findings if f.severity == HARD)
    return QualityOutcome(
        report_path=path,
        findings_count=len(findings),
        hard_count=hard_count,
        checked_endpoints=tuple(checked_endpoints),
    )
