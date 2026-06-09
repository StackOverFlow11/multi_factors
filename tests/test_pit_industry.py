"""Point-in-time SW industry as-of alignment (P2-3, pure + network-free).

`asof_industry` turns per-symbol SW-L1 membership intervals (industry, in_date,
out_date) into a (date, symbol) industry Series where each row carries the
industry the symbol belonged to AS OF that trade_date — never a future
reclassification. This is the PIT replacement for the `stock_basic.industry`
CURRENT tag that was broadcast to every date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.clean.pit_industry import asof_industry
from qt.config import ConfigError, RootConfig, load_config


# --------------------------------------------------------------------------- #
# config: processing.neutralize.industry_level (L1/L2/L3, default L1)
# --------------------------------------------------------------------------- #
def _min_cfg(level=None):
    base = {
        "data": {"source": "demo", "start": "2024-01-01", "end": "2024-03-01"},
        "universe": {"type": "static", "symbols": ["000001.SZ"]},
        "factors": [{"name": "momentum_20"}],
        "alpha": {"model": "equal_weight"},
        "portfolio": {"top_n": 1},
        "backtest": {},
        "cost": {},
        "output": {},
    }
    if level is not None:
        base["processing"] = {"neutralize": {"enabled": True, "industry_level": level}}
    return base


def test_industry_level_defaults_to_l1():
    cfg = RootConfig(**_min_cfg())
    assert cfg.processing.neutralize.industry_level == "L1"


def test_industry_level_accepts_l1_l2_l3():
    for lvl in ("L1", "L2", "L3"):
        cfg = RootConfig(**_min_cfg(lvl))
        assert cfg.processing.neutralize.industry_level == lvl


def test_industry_level_rejects_invalid(tmp_path):
    import yaml

    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(_min_cfg("L9")), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(p))


def test_phase2_config_sets_l1():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "config" / "phase2_real_baseline.yaml"))
    assert cfg.processing.neutralize.industry_level == "L1"


def _index(dates, symbols):
    return pd.MultiIndex.from_product(
        [pd.to_datetime(dates), symbols], names=["date", "symbol"]
    )


def _ts(s):
    return pd.Timestamp(s) if s is not None else None


def test_asof_switches_at_reclassification_date():
    # X is IndA until 2023-06-01, then IndB from 2023-06-01 (open).
    intervals = {
        "X": [
            ("IndA", _ts("2020-01-01"), _ts("2023-06-01")),
            ("IndB", _ts("2023-06-01"), None),
        ]
    }
    idx = _index(["2023-05-31", "2023-06-01", "2023-07-01"], ["X"])
    out = asof_industry(idx, intervals)
    assert out.loc[(pd.Timestamp("2023-05-31"), "X")] == "IndA"  # before switch
    assert out.loc[(pd.Timestamp("2023-06-01"), "X")] == "IndB"  # on switch -> new
    assert out.loc[(pd.Timestamp("2023-07-01"), "X")] == "IndB"  # after


def test_asof_carries_forward_pre_start_membership():
    # Y joined IndC in 2015 and never left -> covers a 2023 window from the start.
    intervals = {"Y": [("IndC", _ts("2015-01-01"), None)]}
    idx = _index(["2023-01-01", "2024-06-30"], ["Y"])
    out = asof_industry(idx, intervals)
    assert (out == "IndC").all()


def test_asof_missing_symbol_is_nan():
    intervals = {"Y": [("IndC", _ts("2015-01-01"), None)]}
    idx = _index(["2023-01-01"], ["Z"])  # Z absent from intervals
    out = asof_industry(idx, intervals)
    assert pd.isna(out.loc[(pd.Timestamp("2023-01-01"), "Z")])


def test_asof_date_before_first_membership_is_nan():
    # W only enters IndD in 2024 -> a 2023 date has no as-of industry (no future use).
    intervals = {"W": [("IndD", _ts("2024-01-01"), None)]}
    idx = _index(["2023-12-31", "2024-01-01"], ["W"])
    out = asof_industry(idx, intervals)
    assert pd.isna(out.loc[(pd.Timestamp("2023-12-31"), "W")])
    assert out.loc[(pd.Timestamp("2024-01-01"), "W")] == "IndD"


def test_asof_returns_industry_series_aligned_to_index():
    intervals = {"A": [("I1", _ts("2010-01-01"), None)], "B": [("I2", _ts("2010-01-01"), None)]}
    idx = _index(["2023-01-01"], ["A", "B"])
    out = asof_industry(idx, intervals)
    assert list(out.index) == list(idx)
    assert out.loc[(pd.Timestamp("2023-01-01"), "A")] == "I1"
    assert out.loc[(pd.Timestamp("2023-01-01"), "B")] == "I2"


def test_asof_picks_latest_in_date_on_overlap():
    # Defensive: if two intervals both cover the date, the most recent in_date wins.
    intervals = {
        "X": [
            ("Old", _ts("2010-01-01"), None),
            ("New", _ts("2022-01-01"), None),
        ]
    }
    idx = _index(["2023-01-01"], ["X"])
    out = asof_industry(idx, intervals)
    assert out.loc[(pd.Timestamp("2023-01-01"), "X")] == "New"


# --------------------------------------------------------------------------- #
# enrich_pit_industry + pipeline wiring (PIT replaces the current-tag broadcast)
# --------------------------------------------------------------------------- #
def test_enrich_pit_industry_sets_per_date_column(demo_panel):
    from data.clean.covariates import enrich_pit_industry

    idx = demo_panel.index
    # a deterministic as-of Series over the panel index
    series = pd.Series("Bank", index=idx, name="industry")
    out = enrich_pit_industry(demo_panel, series)
    assert "industry" in out.columns
    assert (out["industry"] == "Bank").all()
    assert "industry" not in demo_panel.columns  # input not mutated


def test_pipeline_covariates_uses_pit_industry_varying_by_date(monkeypatch, demo_panel, example_config_path):
    # _maybe_enrich_covariates must build a PIT (per-date) industry, NOT a constant
    # current tag, and leave names with no SW history as NaN.
    import logging
    from pathlib import Path

    from qt.config import load_config
    from qt import pipeline as P

    cfg = load_config(str(Path(example_config_path).parent / "example_tushare.yaml"))
    symbols = list(demo_panel.index.get_level_values("symbol").unique())
    switch = pd.Timestamp("2024-01-20")

    seen = {}

    class _FakeCov:
        def __init__(self, *a, **k):
            pass

        def market_cap(self, syms, start, end):
            return pd.DataFrame(
                {"date": [], "symbol": [], "market_cap": []}
            )  # industry-only test

        def pit_sw_intervals(self, syms, level="L2"):
            seen["level"] = level  # the pipeline must pass the configured SW level
            return {
                symbols[0]: [("IndA", pd.Timestamp("2020-01-01"), switch),
                             ("IndB", switch, None)],          # switches mid-window
                symbols[1]: [("IndC", pd.Timestamp("2010-01-01"), None)],  # carry forward
                # symbols[2..] absent -> NaN (disclosed gap, no current-tag fallback)
            }

    monkeypatch.setattr(P, "TushareCovariatesFeed", _FakeCov)
    out = P._maybe_enrich_covariates(cfg, demo_panel, symbols, logging.getLogger("t"))
    assert seen["level"] == cfg.processing.neutralize.industry_level  # default L1

    ind = out["industry"]
    before = ind.loc[(pd.Timestamp("2024-01-19"), symbols[0])]
    after = ind.loc[(pd.Timestamp("2024-01-22"), symbols[0])]
    assert before == "IndA" and after == "IndB"        # PIT switch, varies by date
    assert (ind.xs(symbols[1], level="symbol") == "IndC").all()  # carried forward
    assert pd.isna(ind.xs(symbols[2], level="symbol")).all()     # missing -> NaN
