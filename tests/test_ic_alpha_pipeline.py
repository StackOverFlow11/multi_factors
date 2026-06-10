"""P3-2: ic_weighted alpha pipeline wiring (network-free, demo data).

Locks the wiring contract: alpha dispatch by config, forward returns reach ONLY
``alpha.fit`` (and only for models that require them), the equal-weight default
keeps its exact P0 numbers, and the report discloses the model / weights /
fallbacks without leaking any secret.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha.equal_weight import EqualWeightAlpha
from alpha.ic_weight import RollingICWeightAlpha
from qt.config import ConfigError, load_config
from qt.pipeline import _build_alpha, run_phase0


def _write_cfg(tmp_path: Path, example_config_path: str, *, alpha=None,
               factors=None, name="cfg.yaml") -> Path:
    # keep the example config's own window so the demo regression numbers
    # (ic 0.96 / annual 0.84) stay comparable.
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    if alpha is not None:
        raw["alpha"] = alpha
    if factors is not None:
        raw["factors"] = factors
    raw["output"] = {
        "root_dir": str(out), "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"), "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"), "overwrite": True,
    }
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


_TWO_FACTORS = [
    {"name": "momentum_20", "enabled": True, "params": {"window": 20}},
    {"name": "momentum_5", "enabled": True, "params": {"window": 5}},
]
_IC_ALPHA = {"model": "ic_weighted", "params": {"window": 30, "min_periods": 10}}


# --------------------------------------------------------------------------- #
# dispatch + config validation
# --------------------------------------------------------------------------- #
def test_build_alpha_dispatches_equal_weight(tmp_path, example_config_path):
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path)))
    assert isinstance(_build_alpha(cfg), EqualWeightAlpha)


def test_build_alpha_dispatches_ic_weighted_with_params(tmp_path, example_config_path):
    cfg = load_config(str(_write_cfg(tmp_path, example_config_path, alpha=_IC_ALPHA)))
    alpha = _build_alpha(cfg)
    assert isinstance(alpha, RollingICWeightAlpha)
    p = alpha.params()
    assert p["window"] == 30 and p["min_periods"] == 10
    # the IC horizon is tied to the FIRST forward-return period (config: 1).
    assert p["horizon"] == int(cfg.analytics.forward_return_periods[0])


def test_unknown_alpha_model_is_a_config_error(tmp_path, example_config_path):
    path = _write_cfg(tmp_path, example_config_path,
                      alpha={"model": "clairvoyant", "params": {}})
    with pytest.raises(ConfigError, match="alpha"):
        load_config(str(path))


# --------------------------------------------------------------------------- #
# end-to-end demo runs
# --------------------------------------------------------------------------- #
def test_phase0_equal_weight_default_numbers_unchanged(tmp_path, example_config_path):
    """The P0 regression line: the default alpha keeps its exact demo numbers."""
    result = run_phase0(str(_write_cfg(tmp_path, example_config_path)))
    assert result.alpha_summary == {"model": "equal_weight"}
    assert result.alpha_weights is None
    assert result.ic_mean == pytest.approx(0.96, abs=0.005)
    assert result.performance["annual_return"] == pytest.approx(0.8408, abs=0.005)
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Alpha model" in text and "`equal_weight`" in text


def test_phase0_ic_weighted_runs_and_discloses(tmp_path, example_config_path):
    cfg_path = _write_cfg(tmp_path, example_config_path,
                          alpha=_IC_ALPHA, factors=_TWO_FACTORS)
    result = run_phase0(str(cfg_path))

    # summary + weights log populated
    s = result.alpha_summary
    assert s["model"] == "ic_weighted"
    assert s["n_dates"] > 0 and 0 <= s["n_fallback"] <= s["n_dates"]
    log = result.alpha_weights
    assert log is not None and {"momentum_20", "momentum_5", "fallback"} <= set(log.columns)
    # early dates (insufficient realized history) fell back; later ones trained.
    assert bool(log["fallback"].iloc[0]) is True
    assert bool(log["fallback"].iloc[-1]) is False
    # trained rows are L1-normalized
    trained = log.loc[~log["fallback"], ["momentum_20", "momentum_5"]]
    assert (trained.abs().sum(axis=1) - 1.0).abs().max() < 1e-9

    # the run still produces a valid backtest + report disclosure
    assert math.isfinite(result.performance["annual_return"])
    text = result.report_path.read_text(encoding="utf-8")
    assert "`ic_weighted`" in text and "walk-forward" in text
    assert "fallback" in text.lower()
    assert "NOT a tuned-" in text
    assert "token" not in text.lower()  # no secret leak


def test_ic_weighted_scores_differ_from_equal_weight_when_ics_diverge():
    """Trained weights actually change the combination (not a silent no-op).

    Built on a synthetic panel whose two factors rank OPPOSITELY (the demo
    feed's momentum_5/momentum_20 share one ranking, so their rank ICs — and
    hence the IC weights — coincide with equal weight there by construction).
    """
    import numpy as np

    from qt.pipeline import _build_scores

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=60)
    syms = [f"S{i}" for i in range(8)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    base = rng.normal(size=len(idx))
    factors = pd.DataFrame({"good": base, "bad": -base}, index=idx)
    fwd = pd.Series(base, index=idx)

    eq_scores = _build_scores(factors, EqualWeightAlpha())
    ic_alpha = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    ic_scores = _build_scores(factors, ic_alpha, fwd)
    # equal weight of perfectly opposed factors is ~0 everywhere; IC weights
    # restore the signal -> the score panels must differ materially.
    assert (eq_scores.abs() < 1e-12).all()
    trained_dates = ic_alpha.weights_log().query("~fallback").index
    trained_scores = ic_scores[ic_scores.index.get_level_values("date").isin(trained_dates)]
    assert trained_scores.abs().max() > 0.1
