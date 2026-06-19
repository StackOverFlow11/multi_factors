"""Report-only data-quality layer (D3).

Pure, library-only checks that flag suspicious upstream daily-market / adj_factor
/ 1min-intraday data near ingestion. This layer NEVER mutates data, filters rows,
repairs values, or touches cache / feed / factor / alpha / portfolio / runtime
logic — it only surfaces findings (with bounded, secret-free examples) for logs or
a future data-update artifact.
"""

from __future__ import annotations

from data.quality.intraday import run_intraday_checks
from data.quality.market import run_adj_factor_checks, run_market_checks
from data.quality.report import (
    HARD,
    INFO,
    WARNING,
    QualityFinding,
    findings_to_frame,
    has_hard,
    make_finding,
    render_report,
)

__all__ = [
    "HARD",
    "INFO",
    "WARNING",
    "QualityFinding",
    "findings_to_frame",
    "has_hard",
    "make_finding",
    "render_report",
    "run_adj_factor_checks",
    "run_intraday_checks",
    "run_market_checks",
]
