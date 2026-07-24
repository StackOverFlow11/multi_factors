"""PR-G: intraday amplitude-cut factor (Kaiyuan microstructure #30, SECOND factor).

Each trading day is cut INDEPENDENTLY by the 1-minute return: a bar's amplitude
``amp = high/low - 1``; the sentiment is the WITHIN-DAY-lagged 1-minute return
``r_t = close_t/close_{t-1} - 1`` (each day's first surviving bar has no ``r`` and is not
cut); with ``k = floor(lam * n_day)`` the day's cut is ``V_day = V_high - V_low`` where
V_high / V_low are the mean amp of the top-``k`` / bottom-``k`` bars by ``r``. The
trailing ``N`` VALID days give ``V_mean`` / ``V_std``, which are z-scored per date across
the cross-section and averaged. Sign is pre-registered -1.

Hand cases pin the per-day cut on a controlled within-day close chain (so the top-r /
bottom-r bars are known), assemble identical / two-value day sequences to pin
``V_mean`` / ``V_std``, and pin the cross-sectional z-score on a hand 3-symbol panel.
The cut is per-symbol; the cross-sectional z step is inherently cross-symbol (asserted by
design), matching the report's step 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from factors.compute.minute.intraday_amp_cut import (
    AMP_CUT_LAMBDA,
    AMP_CUT_LOOKBACK_DAYS,
    AMP_CUT_MIN_CROSS_SECTION,
    AMP_CUT_MIN_DAY_MINUTES,
    AMP_CUT_MIN_VALID_DAYS,
    IntradayAmpCutFactor,
    V_MEAN_COL,
    V_STD_COL,
    combine_amp_cut_cross_section,
    compute_amp_cut_stats,
    compute_intraday_amp_cut,
)
from factors.spec import FactorSpec

_SYM = "000001.SZ"

# A within-day close chain for a 6-bar day. Bar 0 (09:31) has no return; bars 1..5 have
# returns [+0.05, -0.02, +0.03, -0.04, +0.01]. So the TOP-return bar is index 1 (09:32,
# +0.05) and the BOTTOM-return bar is index 4 (09:35, -0.04). With lam=0.20 and 5 valid
# bars, k = floor(0.2*5) = 1, so V_day = amp[1] - amp[4].
_CLOSE_CHAIN = [100.0, 105.0, 102.9, 105.987, 101.74752, 102.7649952]
_TOP_IDX = 1
_BOT_IDX = 4


def _bars(rows):
    """rows = [(time, symbol, high, low, close), ...] -> normalized 1min bars.

    ``open`` mirrors ``close`` harmlessly; the factor reads ``high``/``low`` (per-bar
    ``amp = high/low - 1``) and ``close`` (the within-day 1-minute return), decoupled on
    purpose. ``normalize_intraday_bars`` sets ``available_time = bar_end + 1min`` so the
    14:50 PIT cutoff excludes any bar with ``bar_end >= 14:50``.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": [r[4] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1.0] * len(rows),
            "amount": [1.0] * len(rows),
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _amp_day(day, amps, sym=_SYM, start="09:31:00"):
    """One 6-bar day on ``_CLOSE_CHAIN`` with per-bar amplitudes ``amps`` (len 6).

    low = 100 for every bar, high = 100 * (1 + amp), so ``amp = high/low - 1`` is exact.
    With lam=0.20 the day's cut is ``V_day = amps[_TOP_IDX] - amps[_BOT_IDX]``.
    """
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, 100.0 * (1.0 + a), 100.0, c)
        for i, (a, c) in enumerate(zip(amps, _CLOSE_CHAIN))
    ]


def _day_from_returns(day, rets, amps, sym=_SYM, start="09:31:00", c0=100.0):
    """One day whose bars 1..n have returns ``rets`` (bar 0 has none) and amps ``amps``.

    ``len(amps) == len(rets) + 1``. Builds the within-day close chain from ``rets`` so the
    valid-bar returns are exactly ``rets`` and the per-bar amplitude is exactly ``amps``.
    """
    base = pd.Timestamp(day) + pd.Timedelta(start)
    closes = [c0]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    return [
        (base + pd.Timedelta(minutes=i), sym, 100.0 * (1.0 + a), 100.0, c)
        for i, (a, c) in enumerate(zip(amps, closes))
    ]


# Small gates so a short constructed sequence produces a value.
_KW = dict(lam=0.20, min_day_minutes=5, min_valid_days=2, lookback_days=10)
_D1 = pd.Timestamp("2021-07-01")
_D2 = pd.Timestamp("2021-07-02")
_D3 = pd.Timestamp("2021-07-03")


# --------------------------------------------------------------------------- #
# Per-day cut hand values (V_high - V_low by 1-minute return, lambda floor, tie)
# --------------------------------------------------------------------------- #
def test_amp_cut_v_mean_hand_value_two_identical_days():
    # V_day = amp[top-r=idx1] - amp[bottom-r=idx4] = 0.02 - 0.05 = -0.03 on both days.
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    rows = _amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps)
    out = compute_amp_cut_stats(_bars(rows), **_KW)
    # day 1 has only 1 valid trading day in its window -> NaN (min_valid_days=2)
    assert np.isnan(out.loc[(_D1, _SYM), V_MEAN_COL])
    # day 2 averages two identical V_day = -0.03; std of identical values = 0
    assert out.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(-0.03)
    assert out.loc[(_D2, _SYM), V_STD_COL] == pytest.approx(0.0)


def test_amp_cut_v_std_hand_value_two_different_days():
    # V_day1 = 0.02 - 0.05 = -0.03; V_day2 = 0.02 - 0.07 = -0.05.
    amps1 = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    amps2 = [0.10, 0.02, 0.03, 0.04, 0.07, 0.06]
    rows = _amp_day("2021-07-01", amps1) + _amp_day("2021-07-02", amps2)
    out = compute_amp_cut_stats(_bars(rows), **_KW)
    assert out.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(-0.04)  # mean([-0.03, -0.05])
    assert out.loc[(_D2, _SYM), V_STD_COL] == pytest.approx(0.014142135623730951)


def test_amp_cut_lambda_floor_selects_k_bars():
    # 10 valid bars -> k = floor(0.20 * 10) = 2. Top-2 returns are +0.05 (idx5) and
    # +0.04 (idx4); bottom-2 are -0.05 (idx10) and -0.04 (idx9).
    rets = [0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, -0.03, -0.04, -0.05]
    amps = [0.99, 0.50, 0.50, 0.50, 0.10, 0.20, 0.50, 0.50, 0.50, 0.02, 0.04]
    # V_high = mean(amp[idx5]=0.20, amp[idx4]=0.10) = 0.15
    # V_low  = mean(amp[idx10]=0.04, amp[idx9]=0.02) = 0.03  -> V_day = 0.12
    rows = _day_from_returns("2021-07-01", rets, amps) + _day_from_returns(
        "2021-07-02", rets, amps
    )
    out = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=10, min_valid_days=2, lookback_days=10
    )
    assert out.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(0.12)


def test_amp_cut_tie_break_is_r_then_bar_end():
    # Two bars tie at r=+0.05 (idx1 @09:32, idx2 @09:33). The stable order (r, bar_end)
    # ranks the LATER bar (idx2) as the top -> V_high = amp[idx2] = 0.02, NOT amp[idx1] =
    # 0.09. Bottom-r is idx5 (-0.03) -> V_low = 0.05. V_day = -0.03. A wrong tie-break
    # (picking idx1) would give V_day = 0.09 - 0.05 = +0.04.
    rets = [0.05, 0.05, -0.01, -0.02, -0.03]
    amps = [0.99, 0.09, 0.02, 0.03, 0.04, 0.05]
    rows = _day_from_returns("2021-07-01", rets, amps) + _day_from_returns(
        "2021-07-02", rets, amps
    )
    out = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=5, min_valid_days=2, lookback_days=10
    )
    assert out.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(-0.03)


# --------------------------------------------------------------------------- #
# Cross-sectional z-score combine (report step 4) — hand 3-symbol panel
# --------------------------------------------------------------------------- #
def _stats_panel(rows):
    """rows = [(date, symbol, v_mean, v_std), ...] -> a (v_mean, v_std) stats panel."""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(r[0]), r[1]) for r in rows], names=["date", "symbol"]
    )
    return pd.DataFrame(
        {V_MEAN_COL: [r[2] for r in rows], V_STD_COL: [r[3] for r in rows]}, index=idx
    )


def test_amp_cut_cross_section_zscore_hand_value():
    # v_mean = [1, 2, 3] -> mean 2, std(ddof=1) 1 -> z = [-1, 0, 1].
    # v_std  = [1, 3, 5] -> mean 3, std(ddof=1) 2 -> z = [-1, 0, 1].
    # factor = (z_mean + z_std) / 2 = [-1, 0, 1].
    d = "2021-07-05"
    stats = _stats_panel(
        [(d, "AAA.SZ", 1.0, 1.0), (d, "BBB.SZ", 2.0, 3.0), (d, "CCC.SZ", 3.0, 5.0)]
    )
    out = combine_amp_cut_cross_section(
        stats, min_cross_section=3, name="intraday_amp_cut_10"
    )
    dd = pd.Timestamp(d)
    assert out.loc[(dd, "AAA.SZ")] == pytest.approx(-1.0)
    assert out.loc[(dd, "BBB.SZ")] == pytest.approx(0.0)
    assert out.loc[(dd, "CCC.SZ")] == pytest.approx(1.0)
    assert out.name == "intraday_amp_cut_10"


def test_amp_cut_cross_section_below_min_is_all_nan():
    d = "2021-07-05"
    stats = _stats_panel([(d, "AAA.SZ", 1.0, 1.0), (d, "BBB.SZ", 2.0, 3.0)])
    out = combine_amp_cut_cross_section(stats, min_cross_section=3, name="f")
    dd = pd.Timestamp(d)
    assert np.isnan(out.loc[(dd, "AAA.SZ")])
    assert np.isnan(out.loc[(dd, "BBB.SZ")])


def test_amp_cut_cross_section_partial_finite_pair_excluded():
    # CCC has a finite v_mean but NaN v_std -> not a finite pair -> the finite-pair
    # cross-section is only {AAA, BBB} = 2 < 3 -> the whole date is NaN.
    d = "2021-07-05"
    stats = _stats_panel(
        [(d, "AAA.SZ", 1.0, 1.0), (d, "BBB.SZ", 2.0, 3.0), (d, "CCC.SZ", 3.0, np.nan)]
    )
    out = combine_amp_cut_cross_section(stats, min_cross_section=3, name="f")
    dd = pd.Timestamp(d)
    assert np.isnan(out.loc[(dd, "AAA.SZ")])
    assert np.isnan(out.loc[(dd, "BBB.SZ")])
    assert (dd, "CCC.SZ") not in out.index  # NaN-pair row is never emitted


def test_amp_cut_cross_section_degenerate_zero_variance_is_nan():
    # All three v_mean identical -> cross-sectional std 0 -> z undefined -> NaN that date.
    d = "2021-07-05"
    stats = _stats_panel(
        [(d, "AAA.SZ", 2.0, 1.0), (d, "BBB.SZ", 2.0, 3.0), (d, "CCC.SZ", 2.0, 5.0)]
    )
    out = combine_amp_cut_cross_section(stats, min_cross_section=3, name="f")
    assert out.isna().all()


def test_amp_cut_cross_section_per_date_independent():
    # date d1 has 3 finite symbols (z-scored); date d2 has only 2 -> NaN. Each date's
    # z-score uses only that date's cross-section.
    stats = _stats_panel(
        [
            ("2021-07-05", "AAA.SZ", 1.0, 1.0),
            ("2021-07-05", "BBB.SZ", 2.0, 3.0),
            ("2021-07-05", "CCC.SZ", 3.0, 5.0),
            ("2021-07-06", "AAA.SZ", 1.0, 1.0),
            ("2021-07-06", "BBB.SZ", 2.0, 3.0),
        ]
    )
    out = combine_amp_cut_cross_section(stats, min_cross_section=3, name="f")
    assert out.loc[(pd.Timestamp("2021-07-05"), "AAA.SZ")] == pytest.approx(-1.0)
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-06"), "AAA.SZ")])


# --------------------------------------------------------------------------- #
# Within-day lag (no cross-day return) + 14:50 PIT truncation + no lookahead
# --------------------------------------------------------------------------- #
def test_amp_cut_within_day_lag_drops_first_bar_each_day():
    # Each 6-bar day has exactly 5 valid (with-return) bars: the first bar has no return.
    # min_day_minutes=6 -> 5 < 6 -> every day invalid -> empty; min_day_minutes=5 -> valid.
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    rows = _amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps)
    out6 = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=6, min_valid_days=2, lookback_days=10
    )
    assert out6.empty  # no cross-day return makes the first bar valid
    out5 = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=5, min_valid_days=2, lookback_days=10
    )
    assert not out5.dropna().empty


def test_amp_cut_first_bar_amplitude_not_used():
    # The first bar has no return (within-day lag) so it is never in the cut; changing
    # only its amplitude must not change the factor.
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    a = compute_amp_cut_stats(
        _bars(_amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps)), **_KW
    )
    wild = [9.99, 0.02, 0.03, 0.04, 0.05, 0.06]  # only the first bar's amp changed
    b = compute_amp_cut_stats(
        _bars(_amp_day("2021-07-01", wild) + _amp_day("2021-07-02", wild)), **_KW
    )
    assert a.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(b.loc[(_D2, _SYM), V_MEAN_COL])


def test_amp_cut_perturbing_post_1450_bars_does_not_change_factor():
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    rows = _amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps)
    a = compute_amp_cut_stats(_bars(rows), **_KW)
    late = rows + [
        (pd.Timestamp("2021-07-02 14:50"), _SYM, 9_999.0, 100.0, 9_999.0),
        (pd.Timestamp("2021-07-02 14:55"), _SYM, 9_999.0, 100.0, 9_999.0),
    ]
    b = compute_amp_cut_stats(_bars(late), **_KW)
    assert a.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(b.loc[(_D2, _SYM), V_MEAN_COL])


def test_amp_cut_future_day_does_not_change_earlier_factor():
    amps1 = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    amps2 = [0.10, 0.02, 0.03, 0.04, 0.07, 0.06]
    base = _amp_day("2021-07-01", amps1) + _amp_day("2021-07-02", amps2)
    a = compute_amp_cut_stats(_bars(base), **_KW)
    wild = [9.99, 0.90, 0.80, 0.70, 0.01, 0.99]
    future = base + _amp_day("2021-07-03", wild)
    b = compute_amp_cut_stats(_bars(future), **_KW)
    assert np.isfinite(a.loc[(_D2, _SYM), V_MEAN_COL])
    assert a.loc[(_D2, _SYM), V_MEAN_COL] == pytest.approx(b.loc[(_D2, _SYM), V_MEAN_COL])


# --------------------------------------------------------------------------- #
# Per-symbol isolation (cut level) + full-pipeline end-to-end
# --------------------------------------------------------------------------- #
def test_amp_cut_per_symbol_isolation_at_cut_level():
    amps_a = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]  # V_day = -0.03
    amps_b = [0.10, 0.02, 0.03, 0.04, 0.09, 0.06]  # V_day = 0.02 - 0.09 = -0.07
    rows = (
        _amp_day("2021-07-01", amps_a, sym="AAA.SZ")
        + _amp_day("2021-07-02", amps_a, sym="AAA.SZ")
        + _amp_day("2021-07-01", amps_b, sym="BBB.SZ")
        + _amp_day("2021-07-02", amps_b, sym="BBB.SZ")
    )
    out = compute_amp_cut_stats(_bars(rows), **_KW)
    assert out.loc[(_D2, "AAA.SZ"), V_MEAN_COL] == pytest.approx(-0.03)
    assert out.loc[(_D2, "BBB.SZ"), V_MEAN_COL] == pytest.approx(-0.07)
    assert out.loc[(_D2, "AAA.SZ"), V_MEAN_COL] != out.loc[(_D2, "BBB.SZ"), V_MEAN_COL]


def test_amp_cut_full_pipeline_cross_section():
    # Three symbols, two identical valid days each -> on d2 every symbol has a finite
    # (V_mean, V_std) pair (V_std = 0). The cross-section z of a constant column is 0,
    # so a zero-variance V_std makes the whole date NaN -> use distinct V_std by giving
    # each symbol a two-value V_day sequence.
    def sym_rows(sym, a5):
        amps1 = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
        amps2 = [0.10, 0.02, 0.03, 0.04, a5, 0.06]
        return _amp_day("2021-07-01", amps1, sym=sym) + _amp_day(
            "2021-07-02", amps2, sym=sym
        )

    rows = sym_rows("AAA.SZ", 0.05) + sym_rows("BBB.SZ", 0.07) + sym_rows("CCC.SZ", 0.09)
    out = compute_intraday_amp_cut(
        _bars(rows),
        lam=0.20,
        min_day_minutes=5,
        min_valid_days=2,
        min_cross_section=3,
        lookback_days=10,
        name="intraday_amp_cut_10",
    )
    d2 = out.loc[_D2]
    assert d2.notna().all()  # all three symbols z-scored
    assert d2.sum() == pytest.approx(0.0, abs=1e-9)  # z-scores centre at 0
    assert out.name == "intraday_amp_cut_10"


# --------------------------------------------------------------------------- #
# Gates: n_day, valid-day floor, cross-section floor
# --------------------------------------------------------------------------- #
def test_amp_cut_n_day_gate_invalidates_thin_days():
    # min_day_minutes=6 but each day has only 5 valid bars -> no valid day -> no value.
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    rows = _amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps)
    out = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=6, min_valid_days=2, lookback_days=10
    )
    assert out.empty


def test_amp_cut_valid_day_floor_returns_nan_until_enough_days():
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    rows = (
        _amp_day("2021-07-01", amps)
        + _amp_day("2021-07-02", amps)
        + _amp_day("2021-07-03", amps)
    )
    out = compute_amp_cut_stats(
        _bars(rows), lam=0.20, min_day_minutes=5, min_valid_days=3, lookback_days=10
    )
    assert np.isnan(out.loc[(_D1, _SYM), V_MEAN_COL])  # 1 valid day
    assert np.isnan(out.loc[(_D2, _SYM), V_MEAN_COL])  # 2 valid days
    assert out.loc[(_D3, _SYM), V_MEAN_COL] == pytest.approx(-0.03)  # 3 valid days


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_amp_cut_empty_bars_yield_empty_schema():
    stats = compute_amp_cut_stats(empty_intraday_bars())
    assert stats.empty
    assert list(stats.index.names) == ["date", "symbol"]
    assert list(stats.columns) == [V_MEAN_COL, V_STD_COL]
    out = compute_intraday_amp_cut(empty_intraday_bars(), name="f")
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "f"


def test_amp_cut_input_bars_not_mutated():
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    bars = _bars(_amp_day("2021-07-01", amps) + _amp_day("2021-07-02", amps))
    before = bars.copy(deep=True)
    compute_amp_cut_stats(bars, **_KW)
    compute_intraday_amp_cut(bars, min_cross_section=2)
    pd.testing.assert_frame_equal(bars, before)


def test_amp_cut_stats_input_not_mutated_by_combine():
    d = "2021-07-05"
    stats = _stats_panel(
        [(d, "AAA.SZ", 1.0, 1.0), (d, "BBB.SZ", 2.0, 3.0), (d, "CCC.SZ", 3.0, 5.0)]
    )
    before = stats.copy(deep=True)
    combine_amp_cut_cross_section(stats, min_cross_section=3, name="f")
    pd.testing.assert_frame_equal(stats, before)


def test_amp_cut_bad_params_raise():
    amps = [0.10, 0.02, 0.03, 0.04, 0.05, 0.06]
    bars = _bars(_amp_day("2021-07-01", amps))
    with pytest.raises(ValueError):
        compute_amp_cut_stats(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_amp_cut_stats(bars, lam=0.0)
    with pytest.raises(ValueError):
        compute_amp_cut_stats(bars, lam=0.6)
    with pytest.raises(ValueError):
        compute_amp_cut_stats(bars, min_day_minutes=1)
    with pytest.raises(ValueError):
        compute_amp_cut_stats(bars, min_valid_days=1)
    with pytest.raises(ValueError):
        combine_amp_cut_cross_section(_stats_panel([("2021-07-05", "A.SZ", 1.0, 1.0)]),
                                      min_cross_section=1)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_amp_cut_factor_spec_is_valid_and_daily():
    spec = IntradayAmpCutFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "intraday_amp_cut_10"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == -1  # NEGATIVE (report rankIC -0.067)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert spec.input_fields == ("high", "low", "close")
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None
    # the PR-D distinction must be spelled out on the spec description
    desc = spec.description
    assert "PR-D" in desc
    assert "pool" in desc.lower()  # PR-D pools; this factor cuts each day
    assert "EACH DAY" in desc


def test_amp_cut_factor_defaults_match_module_constants():
    assert AMP_CUT_LOOKBACK_DAYS == 10
    assert AMP_CUT_LAMBDA == 0.20
    assert AMP_CUT_MIN_DAY_MINUTES == 100
    assert AMP_CUT_MIN_VALID_DAYS == 6
    assert AMP_CUT_MIN_CROSS_SECTION == 10
    f = IntradayAmpCutFactor()
    assert f.lookback_days == AMP_CUT_LOOKBACK_DAYS
    assert f.name == "intraday_amp_cut_10"


def test_amp_cut_factor_subclass_window_tracks_name():
    f = IntradayAmpCutFactor(lookback_days=5)
    assert f.name == "intraday_amp_cut_5"
    assert f.spec.factor_id == "intraday_amp_cut_5"


def test_amp_cut_factor_compute_selects_preaggregated_column():
    f = IntradayAmpCutFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.5]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.5
    assert out.name == f.name


def test_amp_cut_factor_compute_missing_column_raises():
    f = IntradayAmpCutFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_amp_cut_factor_bad_params_raise():
    with pytest.raises(ValueError):
        IntradayAmpCutFactor(lookback_days=0)
