"""I5c: Minute Microstructure Pressure (MMP) factor — formula, PIT, config/runner.

Covers the goal's test groups:

1. MMP formula — exact MMP_t on a controlled 22-bar one-symbol day; rolling
   baselines use t-20..t-1 and exclude t; the first 20 bars have NaN MMP; the daily
   ``intraday_mmp20_ew_0930_1450`` is the arithmetic (equal-weight) mean of valid
   MMP_t (NOT volume-weighted); zero/NaN denominators yield NaN, never inf.
2. PIT/leakage — post-14:50 bars don't change the daily MMP; moving a pre-cutoff
   bar's available_time past the cutoff removes its effect; future bars within a day
   don't change earlier MMP_t baselines; multi-symbol/multi-day isolation; no
   prior-day tail feeds the new day's first-20 baseline.
3. Config/runner — old I5a/I5b configs validate and default to the ``ret`` score;
   the I5c config selects ``mmp_ew``; an invalid score_feature fails readably;
   ``_score_panel`` selects MMP without prefix matching; the report heading names
   I5c; the config Literal mirrors INTRADAY_FEATURE_KEYS (drift guard).
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from data.clean.intraday_aggregate import (
    DEFAULT_FEATURE_KEYS,
    INTRADAY_FEATURE_KEYS,
    asof_daily_features,
    compute_minute_mmp,
    mmp_valid_minute_counts,
)
from data.clean.intraday_schema import normalize_intraday_bars
from qt.config import IntradayCfg, RootConfig, load_config

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_I5A_CONFIG = _CONFIG_DIR / "phase_i5a_intraday_tail_framework.yaml"
_I5B_CONFIG = _CONFIG_DIR / "phase_i5b_intraday_execution_feasibility.yaml"
_I5C_CONFIG = _CONFIG_DIR / "phase_i5c_mmp_minute_factor.yaml"

_MMP_COL = "intraday_mmp20_ew_0930_1450"
_DAY = "2024-01-02"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _mbars(specs: list[tuple]) -> pd.DataFrame:
    """specs = [(time_str, symbol, open, high, low, close, volume), ...]."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([s[0] for s in specs]),
            "symbol": [s[1] for s in specs],
            "open": [s[2] for s in specs],
            "high": [s[3] for s in specs],
            "low": [s[4] for s in specs],
            "close": [s[5] for s in specs],
            "volume": [s[6] for s in specs],
            "amount": [s[5] * s[6] for s in specs],
        }
    )
    return normalize_intraday_bars(df, freq="1min", data_lag="1min")


def _controlled_day(symbol="000001.SZ", n=22, start=f"{_DAY} 09:31:00", day=None):
    """A controlled n-bar session with varied OHLCV (all pre-14:50)."""
    base = pd.Timestamp(start if day is None else f"{day} 09:31:00")
    specs = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        k = i + 1
        specs.append(
            (str(t), symbol, 10.0, 10.0 + 0.1 * k, 10.0 - 0.1 * k,
             10.0 + 0.05 * k, 100.0 + 7.0 * (k % 5))  # varied volume
        )
    return specs


def _arrays(specs):
    o = np.array([s[2] for s in specs], float)
    h = np.array([s[3] for s in specs], float)
    low = np.array([s[4] for s in specs], float)
    c = np.array([s[5] for s in specs], float)
    v = np.array([s[6] for s in specs], float)
    return o, h, low, c, v


# --------------------------------------------------------------------------- #
# 1. MMP formula
# --------------------------------------------------------------------------- #
def test_i5c_mmp_exact_formula_and_first20_nan():
    specs = _controlled_day(n=22)
    o, h, low, c, v = _arrays(specs)
    mmp = compute_minute_mmp(o, h, low, c, v)
    assert len(mmp) == 22
    assert np.all(np.isnan(mmp[:20]))           # first 20 have insufficient lookback
    assert np.all(~np.isnan(mmp[20:]))          # 20, 21 are valid

    t = 20  # manual MMP_20 with baseline = bars 0..19 (excludes bar 20)
    mid = (h[t] + low[t]) / 2.0
    S = (c[t] - mid) / mid
    V = (v[t] / np.median(v[0:20])) ** 0.5
    B = abs(c[t] - o[t]) / (h[t] - low[t] + 1e-6)
    R = (h[t] - low[t]) / (np.mean((h - low)[0:20]) + 1e-6)
    assert mmp[t] == pytest.approx(S * V * B * R)


def test_i5c_rolling_baseline_excludes_bar_t():
    # Bumping ONLY bar t's volume must not change MMP_t's baseline (which is t-20..t-1).
    specs = _controlled_day(n=25)
    o, h, low, c, v = _arrays(specs)
    base = compute_minute_mmp(o, h, low, c, v)
    v2 = v.copy()
    v2[22] *= 100.0  # perturb bar 22's own volume
    bumped = compute_minute_mmp(o, h, low, c, v2)
    # MMP_22 itself changes (its V term), but earlier MMP_t (t<22) baselines are intact.
    assert np.allclose(base[:22], bumped[:22], equal_nan=True)
    assert base[22] != pytest.approx(bumped[22])


def test_i5c_daily_is_equal_weight_not_volume_weighted():
    specs = _controlled_day(n=25)
    o, h, low, c, v = _arrays(specs)
    mmp = compute_minute_mmp(o, h, low, c, v)
    valid = ~np.isnan(mmp)
    ew = float(np.mean(mmp[valid]))
    vw = float(np.average(mmp[valid], weights=v[valid]))
    assert abs(ew - vw) > 1e-6  # the test actually distinguishes the two

    out = asof_daily_features(_mbars(specs), features=["mmp_ew"])
    assert list(out.columns) == [_MMP_COL]
    daily = out[_MMP_COL].iloc[0]
    assert daily == pytest.approx(ew)
    assert not np.isclose(daily, vw)


def test_i5c_invalid_denominators_are_nan_not_inf():
    # zero volume -> median baseline 0 -> V NaN; mid<=0 -> S NaN. Never inf.
    flat = np.array([1.0] * 25)
    v = np.array([0.0] * 25)
    m = compute_minute_mmp(flat, flat, flat, flat, v)
    assert np.all(np.isnan(m)) and not np.any(np.isinf(m))
    # mid <= 0 (degenerate negative prices) -> S NaN -> MMP NaN, not inf
    neg = np.array([-1.0] * 25)
    v2 = np.array([100.0] * 25)
    m2 = compute_minute_mmp(neg, neg, neg, neg, v2)
    assert not np.any(np.isinf(m2))
    assert np.all(np.isnan(m2[20:]))  # where a baseline exists, mid<=0 -> NaN


def test_i5c_daily_nan_when_no_valid_minute():
    # only 5 bars -> never reaches the 20-bar lookback -> daily score NaN.
    specs = _controlled_day(n=5)
    out = asof_daily_features(_mbars(specs), features=["mmp_ew"])
    assert np.isnan(out[_MMP_COL].iloc[0])


# --------------------------------------------------------------------------- #
# 2. PIT / leakage
# --------------------------------------------------------------------------- #
def test_i5c_post_cutoff_bars_do_not_change_daily_mmp():
    specs = _controlled_day(n=22)
    base = asof_daily_features(_mbars(specs), features=["mmp_ew"])
    after = specs + [
        (f"{_DAY} 14:51:00", "000001.SZ", 99.0, 99.0, 99.0, 99.0, 9999.0),
        (f"{_DAY} 14:55:00", "000001.SZ", 99.0, 99.0, 99.0, 99.0, 9999.0),
    ]
    perturbed = asof_daily_features(_mbars(after), features=["mmp_ew"])
    pd.testing.assert_frame_equal(base, perturbed)


def test_i5c_delayed_availability_excludes_bar():
    specs = _controlled_day(n=22)
    bars = _mbars(specs)
    base = asof_daily_features(bars, features=["mmp_ew"])
    # push one pre-cutoff bar's availability past the 14:50 cutoff
    delayed = bars.copy()
    target = pd.Timestamp(f"{_DAY} 09:40:00")
    delayed.loc[delayed["bar_end"] == target, "available_time"] = pd.Timestamp(
        f"{_DAY} 15:00:00"
    )
    got = asof_daily_features(delayed, features=["mmp_ew"])
    dropped = bars[bars["bar_end"] != target]
    expected = asof_daily_features(dropped, features=["mmp_ew"])
    pd.testing.assert_frame_equal(got, expected)
    # and the exclusion actually changed the score (fewer bars -> different baseline/mean)
    assert got[_MMP_COL].iloc[0] != base[_MMP_COL].iloc[0]


def test_i5c_future_bars_do_not_change_earlier_mmp():
    specs = _controlled_day(n=30)
    o, h, low, c, v = _arrays(specs)
    full = compute_minute_mmp(o, h, low, c, v)
    prefix = compute_minute_mmp(o[:25], h[:25], low[:25], c[:25], v[:25])
    # earlier per-bar MMP_t use only t-20..t-1, so adding later bars cannot move them
    assert np.allclose(full[:25], prefix, equal_nan=True)


def test_i5c_multi_day_no_prior_day_tail_carryover():
    # Same symbol, two consecutive days of 22 bars each. Each day's rolling baseline
    # resets, so day-2's first 20 bars are NaN -> exactly 2 valid minutes (not 22).
    s1 = _controlled_day(n=22, day="2024-01-02")
    s2 = _controlled_day(n=22, day="2024-01-03")
    counts = mmp_valid_minute_counts(_mbars(s1 + s2))
    assert counts.loc[(pd.Timestamp("2024-01-02"), "000001.SZ")] == 2
    assert counts.loc[(pd.Timestamp("2024-01-03"), "000001.SZ")] == 2


def test_i5c_multi_symbol_isolation():
    a = _controlled_day(symbol="000001.SZ", n=22)
    b = _controlled_day(symbol="000002.SZ", n=22)
    out = asof_daily_features(_mbars(a + b), features=["mmp_ew"])
    # each symbol independently computed (and equals its solo computation)
    solo_a = asof_daily_features(_mbars(a), features=["mmp_ew"])
    assert out.loc[(pd.Timestamp(_DAY), "000001.SZ"), _MMP_COL] == pytest.approx(
        solo_a[_MMP_COL].iloc[0]
    )
    assert (pd.Timestamp(_DAY), "000002.SZ") in out.index


# --------------------------------------------------------------------------- #
# 3. Config / runner
# --------------------------------------------------------------------------- #
def test_i5c_old_configs_default_to_ret_score():
    for path in (_I5A_CONFIG, _I5B_CONFIG):
        cfg = load_config(str(path))
        assert cfg.intraday is not None
        assert cfg.intraday.score_feature == "ret"  # default preserved


def test_i5c_config_selects_mmp_and_titles_study():
    cfg = load_config(str(_I5C_CONFIG))
    assert cfg.intraday is not None
    assert cfg.intraday.score_feature == "mmp_ew"
    assert cfg.intraday.price_limit_check is True          # I5b feasibility stays ON
    assert cfg.output.intraday_report_name == "phase_i5c_mmp_minute_factor"
    assert "I5c" in (cfg.output.intraday_report_title or "")


def test_i5c_invalid_score_feature_fails_readably():
    d = yaml.safe_load(_I5C_CONFIG.read_text())
    d["intraday"]["score_feature"] = "bogus"
    with pytest.raises(ValidationError, match="score_feature"):
        RootConfig(**d)


def test_i5c_score_feature_literal_mirrors_feature_keys():
    args = typing.get_args(IntradayCfg.model_fields["score_feature"].annotation)
    assert set(args) == set(INTRADAY_FEATURE_KEYS)
    assert "mmp_ew" not in DEFAULT_FEATURE_KEYS  # selectable-only, not a default column


def test_i5c_score_panel_selects_mmp_without_prefix_matching():
    from qt.intraday_tail_framework import _score_panel

    cfg = load_config(str(_I5C_CONFIG))
    bars = _mbars(_controlled_day(n=25))
    series, col = _score_panel(cfg, bars, logging.getLogger("test.i5c"))
    assert col == _MMP_COL
    assert series.name == "score"
    assert (pd.Timestamp(_DAY), "000001.SZ") in series.index


def test_i5c_report_heading_names_i5c_study():
    from qt.intraday_tail_framework import _report_heading

    title, intro = _report_heading("mmp_ew", True)
    assert title.startswith("# Phase I5c")
    assert "exploratory" in intro.lower()
    # title override wins for the H1 (no stale phase label)
    over, _ = _report_heading("mmp_ew", True, "Phase I5c — MMP Minute Factor Study")
    assert over == "# Phase I5c — MMP Minute Factor Study"
    # a non-mmp run with the limit check on still reads as I5b
    i5b_title, _ = _report_heading("ret", True)
    assert i5b_title.startswith("# Phase I5b")
