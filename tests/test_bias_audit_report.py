"""Slice 13: bias-audit report generation tests.

The bias audit must enumerate every required bias category (未来函数, PIT,
可交易过滤, ann_date, 复权, 交易成本) and record the known P0 downgrades so no
hidden bias slips through (INV-007). The writer is pure markdown; we render to a
``tmp_path`` and never touch the repo-root copy in tests (CONTRACTS §8f).
"""

from __future__ import annotations

from qt.reports import (
    bias_audit_required_sections,
    render_bias_audit,
    write_bias_audit,
)


def test_bias_audit_contains_required_sections(tmp_path):
    """The bias audit names every required section (未来函数/PIT/.../成本)."""
    path = write_bias_audit(tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")

    for section in bias_audit_required_sections():
        assert section in text, f"bias audit missing section: {section}"

    # Spot-check the canonical category keywords are present.
    for keyword in ("lookahead", "PIT", "可交易过滤", "ann_date", "复权", "交易成本"):
        assert keyword in text


def test_bias_audit_records_known_phase0_limitations(tmp_path):
    """The audit records the P0 downgrades (static universe, daily, adj_factor)."""
    text = render_bias_audit()
    lowered = text.lower()

    # Static universe is flagged as a PIT downgrade, not a real PIT universe.
    assert "staticuniverse" in lowered
    assert "降级" in text
    # Tradable filter: only missing_close in P0; ST/suspend/limit deferred.
    assert "missing_close" in text
    # adj_factor retained but full forward-adjust deferred.
    assert "adj_factor" in text
    # Forward returns confined to analytics (no-lookahead boundary).
    assert "analytics" in text


def test_bias_audit_discloses_min_listing_days_enforced_real_noop_demo():
    """min_listing_days is enforced on the real path; the demo no-op is disclosed (P2-2)."""
    text = render_bias_audit()
    assert "min_listing_days" in text
    # Real path enforces it via list_date; demo stays a disclosed no-op.
    assert "list_date" in text
    assert "已执行" in text  # real path
    assert "no-op" in text and "降级" in text  # demo fallback, still disclosed


def test_bias_audit_discloses_direction_aware_execution():
    """Direction-aware execution feasibility (limits/suspension) is disclosed (P2-2)."""
    text = render_bias_audit()
    assert "方向感知" in text
    assert "at_up_limit" in text and "at_down_limit" in text
    assert "carry forward" in text or "carry" in text


def test_bias_audit_discloses_missing_settlement_price_convention():
    """A held symbol with a NaN end close settles flat (0.0) — disclosed (LOW)."""
    text = render_bias_audit()
    # The convention (NaN end close -> 0.0 / flat) must be named, not hidden.
    assert "结算价缺失" in text or "结算价" in text
    assert "NaN" in text
