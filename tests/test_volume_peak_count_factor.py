"""PR-F: volume-peak-minute-count factor.

The factor PIT-truncates 1min bars at 14:50, classifies each visible minute against its
SAME-SLOT strictly-prior baseline (eruptive if ``vol > μ + σ``, else mild), marks the
eruptive minutes whose both 1-minute same-session neighbours are mild as PEAKS, and
returns the peak-minute count over the trailing ``N`` VALID trading days. Sign is
pre-registered +1 (more volume peaks -> higher forward return).

Hand cases build a constant (or two-level) BACKGROUND of prior days so the same-slot
baseline μ/σ is exact, then a test day whose eruptive / mild pattern is engineered so
the peak count is a known small integer. Because the baseline is STRICTLY PRIOR, the
only VALID day (>= min_classifiable classifiable bars) is the test day, so the factor
value on the test day equals that day's peak count (single-valid-day window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from factors.compute.minute.primitives import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
)
from factors.compute.minute.volume_peak_count import (
    VOLUME_PRV_LOOKBACK_DAYS,
    VolumePeakCountFactor,
    compute_volume_peak_count,
)
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, volume), ...] -> normalized 1min bars.

    OHLC are dummy constants (the factor reads only ``volume`` for the slot baseline);
    ``amount`` mirrors volume harmlessly. ``normalize_intraday_bars`` sets
    ``available_time = bar_end + 1min`` so the 14:50 PIT cutoff excludes any bar with
    ``bar_end >= 14:50``.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": [float(r[2]) for r in rows],
            "amount": [float(r[2]) for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, vols, sym=_SYM, start="09:31:00"):
    """One session of CONSECUTIVE 1-minute bars (start, start+1m, ...) with ``vols``.

    Consecutive minutes are exactly 60s apart, so within a session every interior bar
    has both 1-minute neighbours (the 60s-gap neighbour test).
    """
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [(base + pd.Timedelta(minutes=i), sym, v) for i, v in enumerate(vols)]


def _background(n_days, bg_vols, sym=_SYM, start_day="2021-07-01", start="09:31:00"):
    """``n_days`` prior days each with the SAME slots/volumes -> a clean slot baseline.

    Uses consecutive CALENDAR days (the factor treats each date-with-bars as one
    trading-day row; it consults no exchange calendar), so day ``n_days`` is the test
    day and has exactly ``n_days`` strictly-prior same-slot observations.
    """
    rows = []
    for i in range(n_days):
        day = (pd.Timestamp(start_day) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, bg_vols, sym=sym, start=start)
    return rows


def _test_day(n_days, start_day="2021-07-01"):
    return pd.Timestamp(start_day) + pd.Timedelta(days=n_days)


# Small gates so the single test day is the only valid day and its window is length 1.
_KW = dict(
    baseline_days=VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs=VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k=VOLUME_PRV_SIGMA_K,
    lookback_days=VOLUME_PRV_LOOKBACK_DAYS,
    min_valid_days=1,
    min_classifiable=1,
)


# --------------------------------------------------------------------------- #
# Hand-computed peak counts
# --------------------------------------------------------------------------- #
def test_two_clean_peaks_hand_value():
    # background: 10 days, 6 slots all volume 100 -> baseline μ=100, σ=0, thr=100.
    # test day: eruptive (200) at 09:32 and 09:35, both with mild (100) neighbours.
    rows = _background(10, [100.0] * 6)
    rows += _session("2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0])
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(2.0)
    assert out.name == "volume_peak_count"


def test_peak_ridge_and_boundary_hand_value():
    # 7 slots. test day eruptions at 09:32, 09:34, 09:35, 09:37.
    #  09:32 -> peak (mild neighbours 09:31, 09:33)
    #  09:34 -> RIDGE (its next neighbour 09:35 is eruptive) -> not a peak
    #  09:35 -> RIDGE (its prev neighbour 09:34 is eruptive) -> not a peak
    #  09:37 -> last bar of the session (no next neighbour) -> not a peak
    # => exactly 1 peak.
    rows = _background(10, [100.0] * 7)
    rows += _session(
        "2021-07-11", [100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]
    )
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(1.0)


def test_threshold_is_mu_plus_sigma_not_mu():
    # Two-level background so σ > 0: every slot alternates 80 / 120 across 10 days ->
    # μ=100, σ=std([80,120]*5, ddof=1)=21.08, thr=121.08. A test-day slot at 115 is
    # ABOVE μ but BELOW μ+σ -> MILD (a μ-only rule would wrongly call it eruptive and
    # turn the real peak into a ridge, giving 0). The 200 bar is the only eruption.
    bg_pattern = [80.0, 120.0] * 5  # 10 prior days, same at every slot
    rows = []
    for i, vol in enumerate(bg_pattern):
        day = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += _session(day, [vol] * 5)
    rows += _session("2021-07-11", [100.0, 115.0, 200.0, 100.0, 100.0])
    out = compute_volume_peak_count(_bars(rows), **_KW)
    # 09:33 (200) eruptive; neighbours 09:32 (115, mild) and 09:34 (100, mild) -> 1 peak
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(1.0)


def test_boundary_eruptions_at_session_ends_are_not_peaks():
    # eruptions ONLY at the first and last slot of the session -> each lacks a
    # neighbour on one side -> 0 peaks.
    rows = _background(10, [100.0] * 5)
    rows += _session("2021-07-11", [200.0, 100.0, 100.0, 100.0, 200.0])
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Same-slot baseline is STRICTLY PRIOR (no lookahead) + PIT truncation
# --------------------------------------------------------------------------- #
def test_future_day_does_not_change_earlier_factor():
    # The baseline is strictly prior and the count is trailing, so appending a WILD
    # future day must not change the test day's value.
    base = _background(10, [100.0] * 6)
    base += _session("2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0])
    a = compute_volume_peak_count(_bars(base), **_KW)
    future = base + _session("2021-07-12", [9_999.0, 9_999.0, 1.0, 9_999.0, 1.0, 9_999.0])
    b = compute_volume_peak_count(_bars(future), **_KW)
    key = (_test_day(10), _SYM)
    assert np.isfinite(a.loc[key])
    assert a.loc[key] == pytest.approx(b.loc[key])


def test_current_day_volume_not_in_its_own_baseline():
    # A slot's own current-day volume must not enter its baseline. With a σ=0
    # background (thr=100), any test-day value > 100 is eruptive regardless of how big
    # it is; changing 200 -> 5000 at the eruption slot keeps it eruptive (baseline
    # unchanged) and its mild neighbours unchanged -> same peak count. A baseline that
    # wrongly folded in the current day would shift with the perturbation.
    rows_a = _background(10, [100.0] * 6) + _session(
        "2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0]
    )
    rows_b = _background(10, [100.0] * 6) + _session(
        "2021-07-11", [100.0, 5_000.0, 100.0, 100.0, 5_000.0, 100.0]
    )
    a = compute_volume_peak_count(_bars(rows_a), **_KW)
    b = compute_volume_peak_count(_bars(rows_b), **_KW)
    key = (_test_day(10), _SYM)
    assert a.loc[key] == pytest.approx(2.0)
    assert b.loc[key] == pytest.approx(2.0)


def test_perturbing_post_1450_bars_does_not_change_factor():
    # A bar with bar_end >= 14:50 is available only at >= 14:51 (> cutoff) -> excluded;
    # its volume must not affect the factor.
    rows = _background(10, [100.0] * 6) + _session(
        "2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0]
    )
    a = compute_volume_peak_count(_bars(rows), **_KW)
    late = rows + [
        (pd.Timestamp("2021-07-11 14:50"), _SYM, 9_999.0),
        (pd.Timestamp("2021-07-11 14:55"), _SYM, 9_999.0),
    ]
    b = compute_volume_peak_count(_bars(late), **_KW)
    key = (_test_day(10), _SYM)
    assert a.loc[key] == pytest.approx(b.loc[key])


# --------------------------------------------------------------------------- #
# Cross-session non-adjacency (11:30 and 13:01 are NOT neighbours)
# --------------------------------------------------------------------------- #
def test_cross_session_11_30_and_13_01_are_not_neighbours():
    # slots 11:29, 11:30, 13:01, 13:02, 13:03 (morning end + afternoon start).
    # test day eruptions at 11:30 and 13:02.
    #  11:30 -> last morning bar; its "next" 13:01 is 91 min away (lunch) -> NOT a
    #           neighbour -> 11:30 is not a peak (would be a peak if 13:01 counted).
    #  13:02 -> interior afternoon bar with mild 13:01/13:03 neighbours -> a PEAK.
    # => exactly 1 peak (the positive control proves peaks ARE counted).
    slots = ["11:29:00", "11:30:00", "13:01:00", "13:02:00", "13:03:00"]

    def day(d, vols):
        base = pd.Timestamp(d)
        return [(base + pd.Timedelta(s), _SYM, v) for s, v in zip(slots, vols)]

    rows = []
    for i in range(10):
        d = (pd.Timestamp("2021-07-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows += day(d, [100.0] * 5)
    rows += day("2021-07-11", [100.0, 200.0, 100.0, 200.0, 100.0])
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(1.0)


def test_missing_minute_breaks_neighbour_adjacency():
    # If the bar one minute before the eruption is missing, the eruption has no
    # 1-minute prior neighbour (the gap to the surviving earlier bar is 120s) -> the
    # eruption is not a peak. Drop the 09:32 bar; eruption at 09:33 then lacks its
    # prior neighbour -> 0 peaks.
    rows = _background(10, [100.0] * 6)
    full = _session("2021-07-11", [100.0, 100.0, 200.0, 100.0, 100.0, 100.0])
    rows += [r for r in full if r[0] != pd.Timestamp("2021-07-11 09:32")]
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.loc[(_test_day(10), _SYM)] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Per-symbol isolation
# --------------------------------------------------------------------------- #
def test_per_symbol_isolation():
    rows = _background(10, [100.0] * 6, sym="AAA.SZ")
    rows += _session("2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0], sym="AAA.SZ")
    rows += _background(10, [100.0] * 6, sym="BBB.SZ")
    rows += _session("2021-07-11", [100.0, 200.0, 100.0, 100.0, 100.0, 100.0], sym="BBB.SZ")
    out = compute_volume_peak_count(_bars(rows), **_KW)
    d = _test_day(10)
    assert out.loc[(d, "AAA.SZ")] == pytest.approx(2.0)  # two peaks
    assert out.loc[(d, "BBB.SZ")] == pytest.approx(1.0)  # one peak
    assert out.loc[(d, "AAA.SZ")] != out.loc[(d, "BBB.SZ")]


# --------------------------------------------------------------------------- #
# Gates: baseline insufficient, valid-day floor
# --------------------------------------------------------------------------- #
def test_baseline_insufficient_yields_no_value():
    # Only 9 prior days -> fewer than baseline_min_obs (=10) same-slot obs -> the test
    # day has 0 classifiable bars -> not a valid day -> no value at all.
    rows = _background(9, [100.0] * 6)
    rows += _session("2021-07-10", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0])
    out = compute_volume_peak_count(_bars(rows), **_KW)
    assert out.dropna().empty


def test_min_classifiable_gate_invalidates_thin_days():
    # A day with fewer than min_classifiable classifiable bars is not valid. Here every
    # day has only 3 slots; with min_classifiable=5 the test day is invalid -> no value.
    rows = _background(10, [100.0] * 3)
    rows += _session("2021-07-11", [100.0, 200.0, 100.0])
    out = compute_volume_peak_count(
        _bars(rows),
        baseline_days=20, baseline_min_obs=10, sigma_k=1.0,
        lookback_days=20, min_valid_days=1, min_classifiable=5,
    )
    assert out.dropna().empty


def test_valid_day_floor_returns_nan_until_enough_valid_days():
    # 10 background days + 3 valid test days, each with one big eruption (1 peak). With
    # min_valid_days=3: the first two valid days are NaN (only 1 / 2 valid days in the
    # trailing window), the third sums all three peak counts (=3).
    rows = _background(10, [100.0] * 6)
    for k in range(3):
        d = (pd.Timestamp("2021-07-11") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        rows += _session(d, [100.0, 900.0, 100.0, 100.0, 100.0, 100.0])
    out = compute_volume_peak_count(
        _bars(rows),
        baseline_days=20, baseline_min_obs=10, sigma_k=1.0,
        lookback_days=20, min_valid_days=3, min_classifiable=1,
    )
    d0 = pd.Timestamp("2021-07-11")
    d1 = pd.Timestamp("2021-07-12")
    d2 = pd.Timestamp("2021-07-13")
    assert np.isnan(out.loc[(d0, _SYM)])
    assert np.isnan(out.loc[(d1, _SYM)])
    assert out.loc[(d2, _SYM)] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Guards / purity
# --------------------------------------------------------------------------- #
def test_empty_bars_yield_empty_schema_series():
    out = compute_volume_peak_count(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]
    assert out.name == "volume_peak_count"


def test_input_bars_not_mutated():
    bars = _bars(_background(10, [100.0] * 6) + _session(
        "2021-07-11", [100.0, 200.0, 100.0, 100.0, 200.0, 100.0]
    ))
    before = bars.copy(deep=True)
    compute_volume_peak_count(bars, **_KW)
    pd.testing.assert_frame_equal(bars, before)


def test_bad_params_raise():
    bars = _bars(_session("2021-07-01", [100.0, 100.0, 100.0]))
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, lookback_days=0)
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, baseline_days=1)
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, baseline_min_obs=1)
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, sigma_k=-0.5)
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, min_valid_days=0)
    with pytest.raises(ValueError):
        compute_volume_peak_count(bars, min_classifiable=0)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = VolumePeakCountFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "volume_peak_count_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == 1  # POSITIVE (report IC positive)
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    assert spec.input_fields == ("volume",)
    for field in (
        "decision_cutoff", "data_lag", "session_open",
        "execution_model", "execution_window",
    ):
        assert getattr(spec, field) is None


def test_factor_defaults_match_module_constants():
    assert VOLUME_PRV_LOOKBACK_DAYS == 20
    assert VOLUME_PRV_BASELINE_DAYS == 20
    assert VOLUME_PRV_BASELINE_MIN_OBS == 10
    assert VOLUME_PRV_SIGMA_K == 1.0
    assert VOLUME_PRV_MIN_VALID_DAYS == 10
    assert VOLUME_PRV_MIN_CLASSIFIABLE == 100
    f = VolumePeakCountFactor()
    assert f.lookback_days == VOLUME_PRV_LOOKBACK_DAYS
    assert f.name == "volume_peak_count_20"


def test_factor_subclass_window_tracks_name():
    f = VolumePeakCountFactor(lookback_days=10)
    assert f.name == "volume_peak_count_10"
    assert f.spec.factor_id == "volume_peak_count_10"


def test_factor_compute_selects_preaggregated_column():
    f = VolumePeakCountFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [7.0]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 7.0
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = VolumePeakCountFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_factor_bad_params_raise():
    with pytest.raises(ValueError):
        VolumePeakCountFactor(lookback_days=0)
