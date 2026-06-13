"""Slice 12: Phase 0 end-to-end pipeline tests (TEST-007, CLI-002, INV-006/007).

These run the REAL spine (qt.pipeline.run_phase0) against the offline DemoFeed —
no network, no tushare token. Every write is redirected under ``tmp_path`` via a
config whose ``output`` dirs point inside the temp dir, so the suite never
pollutes the repo's ``artifacts/`` (CONTRACTS §8f, SEC-003).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import qt.pipeline as pipeline
from qt.config import load_config
from qt.pipeline import _build_feed, run_phase0


def _write_tmp_config(
    tmp_path: Path,
    example_config_path: str,
    *,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    source: str = "demo",
    external_secret_file: str | None = None,
    name: str = "config.yaml",
) -> Path:
    """Copy the example config but redirect every output dir under ``tmp_path``."""
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    raw["data"]["source"] = source
    # Keep the run small + fast: a short window is plenty for momentum_20.
    raw["data"]["start"] = start
    raw["data"]["end"] = end
    if external_secret_file is not None:
        raw["data"]["external_secret_file"] = external_secret_file
    raw["output"] = {
        "root_dir": str(out),
        "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"),
        "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"),
        "overwrite": True,
    }
    cfg_path = tmp_path / name
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_path


def test_phase0_pipeline_runs_with_demo_data(tmp_path, example_config_path):
    """The full demo pipeline runs and returns a populated result (TEST-007)."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    result = run_phase0(str(cfg_path))

    assert result.panel_rows > 0
    assert result.panel_symbols == 5
    assert result.factor_name == "momentum_20"
    # NAV table has the contract columns and at least one rebalance row.
    assert list(result.nav_table.columns) == [
        "nav",
        "gross_return",
        "cost",
        "turnover",
        "net_return",
    ]
    assert not result.nav_table.empty
    # Performance dict has the P0 metrics.
    for key in ("annual_return", "max_drawdown", "volatility", "sharpe"):
        assert key in result.performance


def test_phase0_pipeline_writes_expected_artifacts(tmp_path, example_config_path):
    """All four expected artifacts land under the configured output dir (§15)."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    result = run_phase0(str(cfg_path))

    expected = [
        result.data_path,
        result.factor_path,
        result.report_path,
        result.log_path,
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        # SEC-003: nothing is written outside the temp output dir.
        assert str(tmp_path) in str(path)


def test_phase0_summary_mentions_static_universe_downgrade(tmp_path, example_config_path):
    """The summary discloses the static-universe PIT downgrade (INV-007)."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    result = run_phase0(str(cfg_path))

    text = result.report_path.read_text(encoding="utf-8")
    assert "DOWNGRADES" in text
    lowered = text.lower()
    assert "static universe" in lowered or "staticuniverse" in lowered
    assert "pit" in lowered
    # The daily-data and simple-fallback downgrades are also disclosed.
    assert "daily" in lowered
    assert "alphalens" in lowered or "quantstats" in lowered


def test_phase0_pipeline_is_reentrant(tmp_path, example_config_path):
    """Re-running over already-written files must not fail (INV-006)."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    first = run_phase0(str(cfg_path))
    assert first.report_path.exists()

    # Second run over the same (now-populated) output dir must succeed.
    second = run_phase0(str(cfg_path))
    assert second.report_path.exists()
    assert second.panel_rows == first.panel_rows
    assert second.factor_name == first.factor_name


def test_phase0_headline_metrics_are_finite_and_sane(tmp_path, example_config_path):
    """Monthly nav must annualize at 12/yr -> finite, sane headline (HIGH-1).

    Before the fix, ``performance_summary`` used the daily default (252/yr) on a
    monthly nav, exploding annual_return to ~2e12 % and sharpe to ~17. With the
    correct cadence the metrics are finite and within a sane band.
    """
    cfg_path = _write_tmp_config(
        tmp_path, example_config_path, start="2024-01-01", end="2024-12-31"
    )
    result = run_phase0(str(cfg_path))
    perf = result.performance

    annual_return = perf["annual_return"]
    sharpe = perf["sharpe"]
    volatility = perf["volatility"]
    assert math.isfinite(annual_return)
    assert math.isfinite(sharpe)
    assert math.isfinite(volatility)
    # Sane band: |annual_return| < 50 (i.e. < 5000%); the old bug was ~2e10.
    assert abs(annual_return) < 50.0
    assert abs(sharpe) < 50.0


def test_phase0_report_has_no_hidden_na_quantile(tmp_path, example_config_path):
    """No quantile bucket is a hidden inf shown as 'n/a' (MEDIUM-1).

    Over the full calendar the falling symbol used to cross zero, producing inf
    forward returns whose bucket mean rendered as 'n/a'. With the fix every
    bucket cell is finite.
    """
    cfg_path = _write_tmp_config(
        tmp_path, example_config_path, start="2024-01-01", end="2024-12-31"
    )
    result = run_phase0(str(cfg_path))

    q = result.quantile_returns
    assert not q.empty
    assert not np.isinf(q.to_numpy()).any()
    # The report's quantile table must not show an inf-driven 'n/a'.
    means = q.mean(axis=0)
    assert np.isfinite(means.to_numpy()).all()


def test_phase0_tushare_source_requires_secret_file(tmp_path, example_config_path):
    """source='tushare' with no secret file -> a readable error, not demo data (HIGH-3)."""
    cfg_path = _write_tmp_config(
        tmp_path, example_config_path, source="tushare", external_secret_file=""
    )
    cfg = load_config(str(cfg_path))
    with pytest.raises(ValueError) as exc:
        _build_feed(cfg)
    msg = str(exc.value)
    assert "external_secret_file" in msg
    assert "tushare" in msg.lower()


def test_phase0_tushare_source_routes_to_tushare_feed(
    tmp_path, example_config_path, monkeypatch
):
    """source='tushare' with a secret path dispatches to TushareFeed (HIGH-3).

    No network or token read happens: TushareFeed is monkeypatched, so this
    asserts the DISPATCH, not the data. A 'tushare' config must never be served
    DemoFeed data silently.
    """
    cfg_path = _write_tmp_config(
        tmp_path,
        example_config_path,
        source="tushare",
        external_secret_file=str(tmp_path / "fake.config.json"),
    )
    cfg = load_config(str(cfg_path))

    captured: dict[str, object] = {}

    class _FakeTushareFeed:
        def __init__(self, secret_file: str, token_key: str = "tushare.token",
                     cache=None):
            captured["secret_file"] = secret_file
            captured["token_key"] = token_key
            captured["cache"] = cache  # P4-1: None unless data.cache.enabled

    monkeypatch.setattr(pipeline, "TushareFeed", _FakeTushareFeed)

    feed = _build_feed(cfg)
    assert isinstance(feed, _FakeTushareFeed)
    # cache disabled by default -> feed built without a cache (unchanged path).
    assert captured["cache"] is None
    assert captured["secret_file"] == str(tmp_path / "fake.config.json")
    # The demo source must still route to the offline DemoFeed.
    demo_cfg = load_config(
        str(_write_tmp_config(tmp_path, example_config_path, name="demo.yaml"))
    )
    from data.feed.demo_feed import DemoFeed

    assert isinstance(_build_feed(demo_cfg), DemoFeed)


def test_phase0_standard_analytics_is_additive_not_replacing(tmp_path, example_config_path):
    """P2-4: alphalens/quantstats are report-only — they never replace the simple
    authoritative metrics or change the trading result."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    result = run_phase0(str(cfg_path))

    # standard-analytics backends are recorded (whatever actually ran).
    assert result.std_performance["backend"] in (
        "quantstats", "error", "unavailable", "skipped",
    )
    assert result.std_factor["backend"] in ("alphalens", "error", "unavailable")
    # the AUTHORITATIVE simple perf is untouched and SEPARATE from the std dict.
    assert "annual_return" in result.performance   # simple perf keys
    assert "cagr" not in result.performance         # cagr lives only in std_performance
    # the report shows the cross-check section and flags the simple metrics authoritative.
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Standard analytics" in text
    assert "authoritative" in text.lower()


def test_phase0_standard_analytics_no_secret_leak(tmp_path, example_config_path):
    """The standard-analytics section must not leak the tushare token (only the
    exception TYPE is ever recorded, never a message)."""
    cfg_path = _write_tmp_config(tmp_path, example_config_path)
    result = run_phase0(str(cfg_path))
    text = result.report_path.read_text(encoding="utf-8")
    assert "token" not in text.lower()
