"""P3-3: OOS stability validation (network-free).

Locks the validation-layer contract:
  * split boundary carries NO leakage: perturbing every forward return realized
    AFTER the split cannot change any train-period date's weights (the
    walk-forward property restated at the split boundary);
  * subperiod statistics are computed strictly within their period;
  * weight-stability diagnostics (sign flips on trained rows, fallback counts /
    reasons) are correct on synthetic logs;
  * the report disclosed the split dates, OOS metrics, weight stability, the
    small-sample caveat, and leaks no secret;
  * the OOS runner refuses the demo source and a config without an ``oos``
    section; the split date must fall inside the data window.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from alpha.ic_weight import RollingICWeightAlpha
from qt.config import ConfigError, load_config
from qt.oos_stability import (
    fallback_reason_counts,
    ic_period_stats,
    sign_consistent,
    split_nav_by_holding,
    subperiod_perf,
    weight_sign_flips,
)

_OOS_CONFIG = str(
    Path(__file__).resolve().parents[1] / "config" / "phase3_real_oos_stability.yaml"
)


# --------------------------------------------------------------------------- #
# split-boundary no-leakage (the P3-3 red-line)
# --------------------------------------------------------------------------- #
def test_perturbing_post_split_returns_leaves_train_weights_unchanged():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=120)
    syms = [f"S{i}" for i in range(8)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    base = rng.normal(size=len(idx))
    factors = pd.DataFrame({"a": base, "b": rng.normal(size=len(idx))}, index=idx)
    fwd = pd.Series(base, index=idx)

    split = dates[60]
    h = 1
    model = RollingICWeightAlpha(window=30, min_periods=10, horizon=h)
    model.fit(factors, fwd)
    train_dates = [d for d in dates if d < split]
    before = {d: model.weights_for(d) for d in train_dates}

    # poison EVERY forward return of factor dates on/after the split: none of
    # them is realized before the split, so no train-period weight may move.
    poisoned = fwd.copy()
    mask = factors.index.get_level_values("date") >= split
    poisoned[mask] = rng.normal(size=int(mask.sum())) * 1e3
    model2 = RollingICWeightAlpha(window=30, min_periods=10, horizon=h)
    model2.fit(factors, poisoned)
    for d in train_dates:
        w_b, w_a = before[d], model2.weights_for(d)
        if w_b is None:
            assert w_a is None
        else:
            pd.testing.assert_series_equal(w_b, w_a)


# --------------------------------------------------------------------------- #
# subperiod statistics (pure functions)
# --------------------------------------------------------------------------- #
def _nav_table():
    dates = pd.to_datetime(
        ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-28", "2023-05-31"]
    )
    return pd.DataFrame(
        {
            "net_return": [0.01, -0.02, 0.03, 0.01, -0.01],
            "turnover": [1.0, 0.4, 0.5, 0.3, 0.2],
            "cost": [0.001] * 5,
            "gross_return": [0.011, -0.019, 0.031, 0.011, -0.009],
            "nav": (1 + pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])).cumprod().values,
        },
        index=pd.Index(dates, name="date"),
    )


def test_split_nav_by_holding_assigns_rows_by_holding_window():
    """A row's return covers [index[i], index[i+1]] — the split must respect it.

    With split 2023-03-01: the 2023-01-31 row (held to 2023-02-28) is fully
    pre-split -> train; the 2023-02-28 row is held to 2023-03-31, STRADDLING the
    split -> excluded from BOTH and disclosed; rows starting on/after the split
    -> test. The last row's end is unknown (terminal candidate) but it starts
    post-split here, so it is test.
    """
    nav = _nav_table()
    train, test, boundary = split_nav_by_holding(nav, pd.Timestamp("2023-03-01"))
    assert list(train.index) == [pd.Timestamp("2023-01-31")]
    assert list(test.index) == [
        pd.Timestamp("2023-03-31"), pd.Timestamp("2023-04-28"),
        pd.Timestamp("2023-05-31"),
    ]
    assert boundary == [pd.Timestamp("2023-02-28")]  # straddler disclosed


def test_split_nav_by_holding_unknown_end_prestart_is_boundary():
    # split AFTER the last row's start: the last row's holding end is unknown
    # (terminal candidate), so it cannot be proven pre-split -> boundary.
    nav = _nav_table()
    train, test, boundary = split_nav_by_holding(nav, pd.Timestamp("2023-06-15"))
    assert pd.Timestamp("2023-05-31") in boundary
    assert len(test) == 0
    assert list(train.index) == [
        pd.Timestamp("2023-01-31"), pd.Timestamp("2023-02-28"),
        pd.Timestamp("2023-03-31"), pd.Timestamp("2023-04-28"),
    ]


def test_subperiod_perf_stats_within_slice_only():
    nav = _nav_table()
    train, test, _ = split_nav_by_holding(nav, pd.Timestamp("2023-03-01"))
    tr = subperiod_perf(train, periods_per_year=12)
    te = subperiod_perf(test, periods_per_year=12)
    assert tr["n_rebalances"] == 1 and te["n_rebalances"] == 3
    assert tr["avg_turnover"] == pytest.approx(1.0)
    assert te["avg_turnover"] == pytest.approx((0.5 + 0.3 + 0.2) / 3)
    for key in ("annual_return", "volatility", "sharpe", "max_drawdown"):
        assert key in tr and key in te


def test_subperiod_perf_empty_slice_is_nan_not_crash():
    nav = _nav_table()
    out = subperiod_perf(nav.iloc[0:0], periods_per_year=12)
    assert out["n_rebalances"] == 0
    assert math.isnan(out["annual_return"])


def test_ic_period_stats_splits_by_realization_date():
    """IC at factor date t realizes at t+h: train requires realization < split.

    10 trading days, h=1, split = dates[5]: ICs of t in dates[0..3] realize at
    dates[1..4] (< split) -> train; t = dates[4] realizes ON dates[5] (== split,
    not strictly before) -> excluded (boundary); t >= dates[5] -> test.
    """
    dates = pd.bdate_range("2023-01-02", periods=10)
    ic = pd.Series([0.1, 0.2, -0.1, 0.3, 0.1, -0.2, -0.3, -0.1, 0.2, np.nan],
                   index=dates)
    split = dates[5]
    stats = ic_period_stats(ic, split, horizon=1)
    tr, te = stats["train"], stats["test"]
    assert tr["n"] == 4  # dates[0..3] only; dates[4] straddles -> excluded
    assert te["n"] == 4  # dates[5..8]; dates[9] is NaN -> dropped
    assert tr["ic_mean"] == pytest.approx(np.mean([0.1, 0.2, -0.1, 0.3]))
    assert tr["hit_rate"] == pytest.approx(3 / 4)
    assert te["hit_rate"] == pytest.approx(1 / 4)
    assert math.isfinite(tr["ic_ir"]) and math.isfinite(te["ic_ir"])


def test_sign_consistent_requires_same_nonzero_sign():
    assert sign_consistent({"train": {"ic_mean": 0.02}, "test": {"ic_mean": 0.01}})
    assert sign_consistent({"train": {"ic_mean": -0.02}, "test": {"ic_mean": -0.01}})
    assert not sign_consistent({"train": {"ic_mean": 0.02}, "test": {"ic_mean": -0.01}})
    assert not sign_consistent({"train": {"ic_mean": float("nan")},
                                "test": {"ic_mean": 0.01}})


# --------------------------------------------------------------------------- #
# weight stability diagnostics
# --------------------------------------------------------------------------- #
def _weights_log():
    dates = pd.to_datetime(
        ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-28", "2023-05-31"]
    )
    return pd.DataFrame(
        {
            "a": [0.33, 0.5, -0.4, 0.6, -0.2],     # trained signs: +, -, +, - -> 3 flips
            "b": [0.33, -0.5, -0.6, -0.4, -0.8],   # trained signs: -, -, -, - -> 0 flips
            "fallback": [True, False, False, False, False],
        },
        index=pd.Index(dates, name="date"),
    )


def test_weight_sign_flips_counts_trained_rows_only():
    flips = weight_sign_flips(_weights_log())
    # the fallback row (all-positive equal weights) must NOT inject flips
    assert flips == {"a": 3, "b": 0}


def test_fallback_reason_counts_aggregates():
    log = {
        pd.Timestamp("2023-01-31"): "insufficient realized IC history",
        pd.Timestamp("2023-02-28"): None,
        pd.Timestamp("2023-03-31"): "insufficient realized IC history",
    }
    counts = fallback_reason_counts(log)
    assert counts == {"insufficient realized IC history": 2}


# --------------------------------------------------------------------------- #
# config + guards
# --------------------------------------------------------------------------- #
def test_oos_config_validates_with_split_inside_window():
    cfg = load_config(_OOS_CONFIG)
    assert cfg.oos is not None
    split = pd.Timestamp(cfg.oos.split_date)
    assert pd.Timestamp(cfg.data.start) < split < pd.Timestamp(cfg.data.end)
    assert cfg.alpha.model == "ic_weighted"  # carries the ic params for the run


def test_split_outside_window_is_a_config_error(tmp_path):
    raw = yaml.safe_load(Path(_OOS_CONFIG).read_text(encoding="utf-8"))
    raw["oos"]["split_date"] = "2031-01-01"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="split_date"):
        load_config(str(p))


def test_oos_runner_rejects_demo_source(example_config_path):
    from qt.oos_stability import run_phase3_oos

    with pytest.raises(ValueError, match="tushare|REAL"):
        run_phase3_oos(example_config_path)


def test_oos_runner_rejects_non_ic_weighted_alpha(tmp_path):
    """alpha.model=equal_weight would silently produce a FAKE comparison
    (equal_weight vs equal_weight labelled ic_weighted) — must be refused."""
    raw = yaml.safe_load(Path(_OOS_CONFIG).read_text(encoding="utf-8"))
    raw["alpha"] = {"model": "equal_weight", "params": {}}
    p = tmp_path / "eq_alpha.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from qt.oos_stability import run_phase3_oos

    with pytest.raises(ValueError, match="ic_weighted"):
        run_phase3_oos(str(p))


def test_oos_runner_requires_oos_section(tmp_path):
    raw = yaml.safe_load(Path(_OOS_CONFIG).read_text(encoding="utf-8"))
    raw.pop("oos")
    p = tmp_path / "no_oos.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from qt.oos_stability import run_phase3_oos

    with pytest.raises(ValueError, match="oos"):
        run_phase3_oos(str(p))


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #
def _synthetic_oos_result():
    from qt.oos_stability import OOSResult

    cfg = load_config(_OOS_CONFIG)
    period = {"annual_return": -0.05, "volatility": 0.15, "sharpe": -0.3,
              "max_drawdown": -0.12, "avg_turnover": 0.8, "n_rebalances": 11}
    ic_p = {"ic_mean": 0.01, "ic_ir": 0.05, "hit_rate": 0.55, "n": 110}
    return OOSResult(
        config=cfg,
        elapsed_seconds=100.0,
        split_date=pd.Timestamp("2023-07-01"),
        train_start=pd.Timestamp("2022-07-01"), train_end=pd.Timestamp("2023-06-30"),
        test_start=pd.Timestamp("2023-07-03"), test_end=pd.Timestamp("2024-06-28"),
        n_train_days=240, n_test_days=238,
        boundary_dates=(pd.Timestamp("2023-06-30"),),
        factor_names=("momentum_20", "roe", "netprofit_yoy"),
        performance={
            "equal_weight": {"train": dict(period), "test": dict(period)},
            "ic_weighted": {"train": dict(period), "test": dict(period)},
        },
        ic_stats={
            "momentum_20": {"train": dict(ic_p), "test": dict(ic_p)},
            "combo_equal_weight": {"train": dict(ic_p), "test": dict(ic_p)},
            "combo_ic_weighted": {"train": dict(ic_p), "test": dict(ic_p)},
        },
        sign_consistency={"momentum_20": True, "combo_equal_weight": True,
                          "combo_ic_weighted": False},
        weights_at_rebalances=_weights_log(),
        sign_flips={"a": 3, "b": 0},
        n_scored=221, n_fallback=20,
        fallback_reasons={"insufficient realized IC history": 20},
        alpha_summary={"model": "ic_weighted", "window": 60, "min_periods": 20,
                       "horizon": 1, "mode": "rolling"},
        downgrades=("DATA PATH = REAL tushare: ...",),
        report_path=Path("artifacts/reports/phase3_oos_stability.md"),
        log_path=Path("artifacts/logs/run_phase3_oos.log"),
    )


def test_render_oos_report_disclosed_boundaries_and_metrics():
    from qt.oos_stability import render_oos_stability

    md = render_oos_stability(_synthetic_oos_result())
    # split boundaries written into the report
    assert "2023-07-01" in md and "2022-07-01" in md and "2024-06-28" in md
    assert "train" in md.lower() and "test" in md.lower()
    # walk-forward boundary semantics + holding-window slicing stated
    assert "realized" in md.lower()
    assert "holding" in md.lower()
    assert "2023-06-30" in md  # the straddling rebalance is disclosed
    # both models' subperiod performance + IC stability + weight stability
    assert "equal_weight" in md and "ic_weighted" in md
    assert "hit rate" in md.lower()
    assert "sign consistency" in md.lower() or "sign-consistency" in md.lower()
    assert "sign flip" in md.lower()
    assert "fallback" in md.lower()
    # the caveat: a stability CHECK, not a return claim
    assert "not a" in md.lower() and "claim" in md.lower()
    assert "small-sample" in md.lower() or "small sample" in md.lower()


def test_render_oos_report_leaks_no_secret():
    from qt.oos_stability import render_oos_stability

    result = _synthetic_oos_result()
    md = render_oos_stability(result)
    assert result.config.data.external_secret_file not in md
    assert "token" not in md.lower()
