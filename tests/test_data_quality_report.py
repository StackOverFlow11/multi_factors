"""D3 report-only findings model + renderer — deterministic, bounded, secret-free."""

from __future__ import annotations

import pandas as pd

from data.quality.report import (
    HARD,
    INFO,
    WARNING,
    clean_value,
    findings_to_frame,
    has_hard,
    make_finding,
    render_report,
    sanitize_text,
    sort_findings,
)

# Secret-looking inputs a caller might (wrongly) pass into a finding; the report
# layer must redact them. These are SCAN TARGETS, not real secrets.
_SECRET_PATH = "/home/shaofl/Projects/financial_projects/.config.json"
_SECRET_KEY = "tushare.token"


def test_make_finding_bounds_examples_to_five():
    examples = [{"symbol": f"{i:06d}.SZ", "date": pd.Timestamp("2024-01-03")} for i in range(9)]
    f = make_finding("market_daily", "non_positive_ohlc", HARD, count=9, examples=examples)
    assert len(f.examples) == 5
    assert f.count == 9


def test_make_finding_cleans_values():
    f = make_finding(
        "market_daily", "x", WARNING, count=1,
        examples=[{"date": pd.Timestamp("2024-01-03"), "v": 1.23456789}],
    )
    assert f.examples[0]["date"] == "2024-01-03"
    assert f.examples[0]["v"] == round(1.23456789, 6)


def test_clean_value_formats():
    assert clean_value(pd.Timestamp("2024-01-03")) == "2024-01-03"
    assert clean_value(pd.Timestamp("2024-01-03 09:31:00")) == "2024-01-03 09:31:00"
    assert clean_value(5) == 5
    assert clean_value(1.0 / 3.0) == round(1.0 / 3.0, 6)


def test_has_hard():
    assert has_hard([make_finding("d", "c", HARD, 1)])
    assert not has_hard([make_finding("d", "c", WARNING, 1)])
    assert not has_hard([])


def test_findings_to_frame_stable_columns_and_order():
    findings = [
        make_finding("market_daily", "extreme_close_move", WARNING, 2),
        make_finding("adj_factor", "invalid_adj_factor", HARD, 1),
        make_finding("market_daily", "non_positive_ohlc", HARD, 3),
    ]
    frame = findings_to_frame(findings)
    assert list(frame.columns) == ["dataset", "check", "severity", "count", "examples", "note"]
    # hard first, then by dataset, then check
    assert frame["severity"].tolist() == ["hard", "hard", "warning"]
    assert frame.iloc[0]["dataset"] == "adj_factor"
    assert frame.iloc[1]["check"] == "non_positive_ohlc"


def test_findings_to_frame_empty():
    frame = findings_to_frame([])
    assert list(frame.columns) == ["dataset", "check", "severity", "count", "examples", "note"]
    assert len(frame) == 0


def test_render_clean_is_explicit():
    out = render_report([])
    assert "No data-quality findings" in out
    assert out.startswith("# Data Quality Report")


def test_render_is_deterministic_and_bounded():
    findings = [
        make_finding(
            "market_daily", "non_positive_ohlc", HARD, count=2,
            examples=[{"symbol": "000001.SZ", "date": pd.Timestamp("2024-01-03")}],
            note="open/high/low/close must be > 0",
        ),
        make_finding("market_daily", "extreme_close_move", WARNING, count=1),
    ]
    out1 = render_report(findings)
    out2 = render_report(list(reversed(findings)))
    assert out1 == out2  # deterministic regardless of input order
    assert "[hard] `market_daily` / non_positive_ohlc: count=2" in out1
    assert "1 hard / 1 warning / 0 info" in out1


def test_render_has_no_secret_looking_paths():
    """Secret-looking path/key passed into a finding must be redacted from output."""
    findings = [
        make_finding(
            "market_daily", "non_positive_ohlc", HARD, 1,
            examples=[{"path": _SECRET_PATH, "key": _SECRET_KEY, "symbol": "000001.SZ"}],
            note=f"leaked at {_SECRET_PATH}",
        )
    ]
    out = render_report(findings)
    assert _SECRET_PATH not in out
    assert ".config.json" not in out
    assert _SECRET_KEY not in out
    assert "[REDACTED]" in out
    # benign content in the same finding still renders
    assert "000001.SZ" in out


def test_secret_path_in_example_value_redacted():
    f = make_finding("x", "y", HARD, 1, examples=[{"path": _SECRET_PATH}])
    assert f.examples[0]["path"] == "[REDACTED]"
    assert _SECRET_PATH not in render_report([f])


def test_tushare_token_in_example_value_redacted():
    f = make_finding("x", "y", HARD, 1, examples=[{"key": _SECRET_KEY}])
    assert f.examples[0]["key"] == "[REDACTED]"
    assert _SECRET_KEY not in render_report([f])


def test_secret_path_in_note_redacted():
    f = make_finding("x", "y", HARD, 1, note=f"see {_SECRET_PATH} for the token")
    assert _SECRET_PATH not in (f.note or "")
    assert "[REDACTED]" in (f.note or "")
    assert _SECRET_PATH not in render_report([f])


def test_token_kv_value_redacted():
    out = render_report([make_finding("x", "y", HARD, 1, examples=[{"v": "token=abc123secret"}])])
    assert "abc123secret" not in out


def test_benign_values_still_render():
    f = make_finding(
        "market_daily", "high_lt_low", HARD, 1,
        examples=[{"symbol": "000001.SZ", "date": pd.Timestamp("2024-01-03")}],
    )
    out = render_report([f])
    assert "000001.SZ" in out
    assert "2024-01-03" in out
    assert "[REDACTED]" not in out


def test_sanitize_text_is_idempotent_and_deterministic():
    once = sanitize_text(f"path {_SECRET_PATH} key {_SECRET_KEY}")
    twice = sanitize_text(once)
    assert once == twice
    assert _SECRET_PATH not in once and _SECRET_KEY not in once


def test_secret_inputs_examples_still_bounded_to_five():
    examples = [{"path": _SECRET_PATH, "i": i} for i in range(9)]
    f = make_finding("x", "y", HARD, 9, examples=examples)
    assert len(f.examples) == 5
    assert all(ex["path"] == "[REDACTED]" for ex in f.examples)


def test_sort_findings_severity_first():
    findings = [make_finding("z", "c", INFO, 1), make_finding("a", "c", HARD, 1)]
    ordered = sort_findings(findings)
    assert ordered[0].severity == HARD
    assert ordered[1].severity == INFO
