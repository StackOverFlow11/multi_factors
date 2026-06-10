"""P3-2: RollingICWeightAlpha — walk-forward IC-weighted combination.

Locks the lookahead red-line and the fallback contract:
  * weight fitting may use ONLY realized history: a (factor[t], fwd_h[t]) pair
    is admissible at prediction date d only if t + h <= d in TRADING-DAY
    positions (the forward return realizes at t+h);
  * perturbing any not-yet-realized forward return must NOT change the weights;
  * insufficient realized history falls back to EQUAL WEIGHT (logged);
  * weights are L1-normalized and sign-preserving; a single factor degenerates
    to +/-1; all-NaN/zero IC falls back;
  * inputs are never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.ic_weight import RollingICWeightAlpha


def _panel(n_days=80, n_sym=8, seed=7):
    """Synthetic (date, symbol) factor frame + forward returns.

    factor 'good' predicts fwd POSITIVELY on every date; factor 'bad' predicts
    NEGATIVELY. Deterministic, so IC signs are stable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    syms = [f"S{i}" for i in range(n_sym)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    base = rng.normal(size=(n_days, n_sym))
    fwd = pd.Series(base.ravel(), index=idx, name="fwd")
    good = pd.Series(base.ravel(), index=idx)          # corr +1 with fwd
    bad = pd.Series(-base.ravel(), index=idx)          # corr -1 with fwd
    factors = pd.DataFrame({"good": good, "bad": bad}, index=idx)
    return factors, fwd, dates


def _block(factors: pd.DataFrame, date) -> pd.DataFrame:
    dates = factors.index.get_level_values("date")
    return factors.loc[dates == pd.Timestamp(date)]


# --------------------------------------------------------------------------- #
# lookahead red-line
# --------------------------------------------------------------------------- #
def test_perturbing_unrealized_future_does_not_change_weights():
    factors, fwd, dates = _panel()
    h = 5
    d = dates[40]
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=h)
    model.fit(factors, fwd)
    w_before = model.weights_for(d)

    # perturb every forward return whose realization date t+h is AFTER d
    # (factor dates t > d - h): these are NOT visible at d and must not matter.
    poisoned = fwd.copy()
    cutoff_pos = list(dates).index(d) - h
    poison_mask = factors.index.get_level_values("date") > dates[cutoff_pos]
    rng = np.random.default_rng(99)
    poisoned[poison_mask] = rng.normal(size=int(poison_mask.sum())) * 100.0

    model2 = RollingICWeightAlpha(window=20, min_periods=5, horizon=h)
    model2.fit(factors, poisoned)
    w_after = model2.weights_for(d)
    pd.testing.assert_series_equal(w_before, w_after)


def test_realization_cutoff_is_exact_t_plus_h():
    # 11 trading days, horizon 5, predict at the last date (pos 10): only ICs of
    # factor dates t with pos(t) <= 10 - 5 = 5 are realized -> exactly 6 rows.
    factors, fwd, dates = _panel(n_days=11)
    d = dates[10]
    # min_periods = 7 > 6 available -> MUST fall back; 6 -> must NOT.
    strict = RollingICWeightAlpha(window=50, min_periods=7, horizon=5)
    strict.fit(factors, fwd)
    strict.predict(_block(factors, d))
    assert strict.fallback_log()[pd.Timestamp(d)] is not None

    ok = RollingICWeightAlpha(window=50, min_periods=6, horizon=5)
    ok.fit(factors, fwd)
    ok.predict(_block(factors, d))
    assert ok.fallback_log()[pd.Timestamp(d)] is None


# --------------------------------------------------------------------------- #
# fallback + degeneration + normalization
# --------------------------------------------------------------------------- #
def test_insufficient_history_falls_back_to_equal_weight():
    factors, fwd, dates = _panel()
    model = RollingICWeightAlpha(window=20, min_periods=10, horizon=1)
    model.fit(factors, fwd)
    early = dates[3]  # only ~2 realized ICs -> fallback
    scores = model.predict(_block(factors, early))
    block = _block(factors, early)
    expected = block.mean(axis=1)
    expected.index = expected.index.get_level_values("symbol")
    pd.testing.assert_series_equal(
        scores, expected.rename("score"), check_names=True
    )
    assert model.fallback_log()[pd.Timestamp(early)] is not None


def test_single_factor_degenerates_to_unit_weight():
    factors, fwd, dates = _panel()
    pos = factors[["good"]]
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model.fit(pos, fwd)
    d = dates[60]
    w = model.weights_for(d)
    assert w["good"] == pytest.approx(1.0)  # positive IC -> +1
    scores = model.predict(_block(pos, d))
    block = _block(pos, d)["good"]
    block.index = block.index.get_level_values("symbol")
    pd.testing.assert_series_equal(scores, block.rename("score"))

    neg = factors[["bad"]]
    model2 = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model2.fit(neg, fwd)
    w2 = model2.weights_for(d)
    assert w2["bad"] == pytest.approx(-1.0)  # negative IC -> -1 (sign kept)


def test_weights_are_l1_normalized_and_sign_preserving():
    factors, fwd, dates = _panel()
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model.fit(factors, fwd)
    w = model.weights_for(dates[60])
    assert abs(w).sum() == pytest.approx(1.0)
    assert w["good"] > 0 and w["bad"] < 0  # negative-IC factor gets a NEGATIVE weight


def test_degenerate_zero_ic_falls_back():
    # constant forward return -> zero-variance cross-sections -> IC NaN -> fallback.
    factors, fwd, dates = _panel()
    flat = pd.Series(0.0, index=fwd.index)
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model.fit(factors, flat)
    d = dates[60]
    model.predict(_block(factors, d))
    assert model.fallback_log()[pd.Timestamp(d)] is not None


# --------------------------------------------------------------------------- #
# modes + interface contract
# --------------------------------------------------------------------------- #
def test_expanding_mode_uses_all_realized_history():
    factors, fwd, dates = _panel()
    roll = RollingICWeightAlpha(window=10, min_periods=5, horizon=1, mode="rolling")
    expa = RollingICWeightAlpha(window=10, min_periods=5, horizon=1, mode="expanding")
    roll.fit(factors, fwd)
    expa.fit(factors, fwd)
    d = dates[60]
    n_roll = roll.train_size_for(d)
    n_expa = expa.train_size_for(d)
    assert n_roll == 10           # capped by the window
    assert n_expa > n_roll        # expanding sees ALL realized history


def test_fit_requires_forward_returns():
    factors, _, _ = _panel()
    model = RollingICWeightAlpha()
    with pytest.raises(ValueError, match="forward_returns"):
        model.fit(factors, None)


def test_predict_requires_a_dated_cross_section():
    factors, fwd, dates = _panel()
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model.fit(factors, fwd)
    undated = _block(factors, dates[60]).droplevel("date")
    with pytest.raises(ValueError, match="date"):
        model.predict(undated)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        RollingICWeightAlpha(mode="clairvoyant")


def test_inputs_are_not_mutated():
    factors, fwd, dates = _panel()
    f_copy, r_copy = factors.copy(), fwd.copy()
    model = RollingICWeightAlpha(window=20, min_periods=5, horizon=1)
    model.fit(factors, fwd)
    model.predict(_block(factors, dates[60]))
    pd.testing.assert_frame_equal(factors, f_copy)
    pd.testing.assert_series_equal(fwd, r_copy)


def test_weights_log_records_per_date_weights_and_fallback():
    factors, fwd, dates = _panel()
    model = RollingICWeightAlpha(window=20, min_periods=10, horizon=1)
    model.fit(factors, fwd)
    model.predict(_block(factors, dates[3]))    # fallback (early)
    model.predict(_block(factors, dates[60]))   # trained
    log = model.weights_log()
    assert list(log.columns[:2]) == ["good", "bad"]
    assert bool(log.loc[pd.Timestamp(dates[3]), "fallback"]) is True
    assert bool(log.loc[pd.Timestamp(dates[60]), "fallback"]) is False
    assert log.loc[pd.Timestamp(dates[60]), "good"] > 0
