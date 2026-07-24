"""PR-C: price-jump turnover-correlation factor (aggregation + spec + PIT)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import empty_intraday_bars, normalize_intraday_bars
from factors.compute.minute.jump_amount_corr import (
    JumpAmountCorrFactor,
    compute_jump_amount_corr,
)
from factors.spec import FactorSpec

_SYM = "000001.SZ"


def _bars(rows):
    """rows = [(time, symbol, open, high, low, amount), ...] -> normalized 1min bars.

    ``close``/``volume`` are filled harmlessly; the factor reads open/high/low/amount.
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows]),
            "symbol": [r[1] for r in rows],
            "open": [r[2] for r in rows],
            "high": [r[3] for r in rows],
            "low": [r[4] for r in rows],
            "close": [r[2] for r in rows],
            "volume": [1.0] * len(rows),
            "amount": [r[5] for r in rows],
        }
    )
    return normalize_intraday_bars(df, freq="1min")


def _session(day, amps, amounts, sym=_SYM, start="09:31:00"):
    """One session: amplitude ``a`` at open=100 -> high=100+a*50, low=100-a*50."""
    base = pd.Timestamp(day) + pd.Timedelta(start)
    return [
        (base + pd.Timedelta(minutes=i), sym, 100.0, 100.0 + a * 50.0, 100.0 - a * 50.0, amt)
        for i, (a, amt) in enumerate(zip(amps, amounts))
    ]


def _hand_pairs(amps, amounts, jump_z=1.0):
    """Reference jump-pairs: consecutive bar i with within-day z(amp)>jump_z, i+1 exists."""
    amp = np.asarray(amps, dtype=float)
    z = (amp - amp.mean()) / amp.std(ddof=1)
    return [
        (amounts[i], amounts[i + 1])
        for i in range(len(amps))
        if z[i] > jump_z and i + 1 < len(amps)
    ]


# --------------------------------------------------------------------------- #
# Correctness vs a hand-computed reference
# --------------------------------------------------------------------------- #
def test_corr_matches_hand_reference_two_days():
    amps1 = [0.001, 0.02, 0.001, 0.001, 0.02, 0.001]
    amts1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    amps2 = [0.001, 0.03, 0.001, 0.001, 0.001, 0.001]
    amts2 = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    bars = _bars(_session("2021-07-01", amps1, amts1) + _session("2021-07-02", amps2, amts2))
    out = compute_jump_amount_corr(bars, lookback_days=20, min_pairs=2)

    p1 = _hand_pairs(amps1, amts1)
    all_pairs = p1 + _hand_pairs(amps2, amts2)
    x1 = np.array([a for a, _ in p1])
    y1 = np.array([b for _, b in p1])
    xa = np.array([a for a, _ in all_pairs])
    ya = np.array([b for _, b in all_pairs])
    d1, d2 = pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-02")
    assert out.loc[(d1, _SYM)] == pytest.approx(np.corrcoef(x1, y1)[0, 1])
    assert out.loc[(d2, _SYM)] == pytest.approx(np.corrcoef(xa, ya)[0, 1])
    assert out.name == "jump_amount_corr"


def test_zscore_threshold_selects_only_high_amplitude_bars():
    # Only bar index 1 has a high amplitude -> exactly one jump; index 5 (last) is
    # also large but has no strictly-next minute, so it never pairs.
    amps = [0.001, 0.05, 0.001, 0.001, 0.001, 0.05]
    amts = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    bars = _bars(_session("2021-07-01", amps, amts))
    # exactly one pair -> corr undefined at min_pairs=2 -> NaN; the single pair is
    # (20, 30) as the reference confirms.
    assert _hand_pairs(amps, amts) == [(20.0, 30.0)]
    out = compute_jump_amount_corr(bars, lookback_days=20, min_pairs=2)
    assert out.dropna().empty  # one pair < min_pairs


def test_session_close_jump_is_not_paired_across_days():
    # jump only at the LAST bar of the session; its "next" bar is the next day ->
    # gap != 60s -> excluded.
    amps = [0.001, 0.001, 0.001, 0.001, 0.001, 0.05]
    bars = _bars(
        _session("2021-07-05", amps, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        + _session("2021-07-06", amps, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    out = compute_jump_amount_corr(bars, lookback_days=20, min_pairs=2)
    assert out.dropna().empty


def test_lunch_gap_jump_is_not_paired():
    rows = [
        (pd.Timestamp("2021-07-06 09:31"), _SYM, 100.0, 100.05, 99.95, 10.0),
        (pd.Timestamp("2021-07-06 09:32"), _SYM, 100.0, 103.0, 97.0, 20.0),   # jump
        (pd.Timestamp("2021-07-06 13:01"), _SYM, 100.0, 100.05, 99.95, 30.0),  # afternoon
    ]
    out = compute_jump_amount_corr(_bars(rows), lookback_days=20, min_pairs=2)
    # the 09:32 jump's next bar is 13:01 (gap != 60s) -> no pair.
    assert out.dropna().empty


def test_trailing_window_drops_pairs_older_than_lookback():
    # Day A has 2 jump-pairs; then 3 no-jump days; a lookback_days=2 window at the
    # later days no longer sees day A's pairs.
    amps = [0.001, 0.02, 0.001, 0.001, 0.02, 0.001]
    amts = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    flat = [0.001] * 6
    rows = (
        _session("2021-07-01", amps, amts)
        + _session("2021-07-02", flat, [1.0] * 6)
        + _session("2021-07-05", flat, [1.0] * 6)
        + _session("2021-07-06", flat, [1.0] * 6)
    )
    out = compute_jump_amount_corr(_bars(rows), lookback_days=2, min_pairs=2)
    d1 = pd.Timestamp("2021-07-01")
    d6 = pd.Timestamp("2021-07-06")
    assert np.isfinite(out.loc[(d1, _SYM)])       # window covers day A's 2 pairs
    assert np.isnan(out.loc[(d6, _SYM)])          # day A is now outside the 2-day window


def test_min_pairs_gate_returns_nan():
    amps = [0.001, 0.02, 0.001, 0.001, 0.02, 0.001]  # 2 jump-pairs on the day
    bars = _bars(_session("2021-07-01", amps, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    out = compute_jump_amount_corr(bars, lookback_days=20, min_pairs=3)
    assert np.isnan(out.loc[(pd.Timestamp("2021-07-01"), _SYM)])


def test_pit_perturbing_future_bars_does_not_change_factor_at_d():
    amps1 = [0.001, 0.02, 0.001, 0.001, 0.02, 0.001]
    amts1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    amps2 = [0.001, 0.03, 0.001, 0.001, 0.001, 0.001]
    base = _session("2021-07-01", amps1, amts1)
    a = compute_jump_amount_corr(
        _bars(base + _session("2021-07-02", amps2, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0])),
        min_pairs=2,
    )
    # wildly perturb EVERY day-2 bar (amounts + amplitudes)
    b = compute_jump_amount_corr(
        _bars(base + _session("2021-07-02", [0.09, 0.001, 0.09, 0.001, 0.09, 0.001],
                              [9e5, 1.0, 9e5, 1.0, 9e5, 1.0])),
        min_pairs=2,
    )
    d1 = pd.Timestamp("2021-07-01")
    assert a.loc[(d1, _SYM)] == pytest.approx(b.loc[(d1, _SYM)])


def test_per_symbol_isolation():
    amps = [0.001, 0.02, 0.001, 0.001, 0.02, 0.001]
    amts_a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    amts_b = [60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    rows = _session("2021-07-01", amps, amts_a, sym="AAA.SZ") + _session(
        "2021-07-01", amps, amts_b, sym="BBB.SZ"
    )
    out = compute_jump_amount_corr(_bars(rows), lookback_days=20, min_pairs=2)
    d1 = pd.Timestamp("2021-07-01")
    pa = _hand_pairs(amps, amts_a)
    pb = _hand_pairs(amps, amts_b)
    ea = np.corrcoef([a for a, _ in pa], [b for _, b in pa])[0, 1]
    eb = np.corrcoef([a for a, _ in pb], [b for _, b in pb])[0, 1]
    assert out.loc[(d1, "AAA.SZ")] == pytest.approx(ea)
    assert out.loc[(d1, "BBB.SZ")] == pytest.approx(eb)


def test_empty_bars_yield_empty_schema_series():
    out = compute_jump_amount_corr(empty_intraday_bars())
    assert out.empty
    assert list(out.index.names) == ["date", "symbol"]


def test_open_and_amount_guards():
    rows = [
        (pd.Timestamp("2021-07-01 09:31"), _SYM, 0.0, 100.05, 99.95, 10.0),   # open<=0 dropped
        (pd.Timestamp("2021-07-01 09:32"), _SYM, 100.0, 103.0, 97.0, np.nan),  # amount NaN dropped
        (pd.Timestamp("2021-07-01 09:33"), _SYM, 100.0, 100.05, 99.95, 30.0),
    ]
    # no valid jump-pair survives -> empty (guards drop the bad rows before pairing).
    out = compute_jump_amount_corr(_bars(rows), min_pairs=2)
    assert out.dropna().empty


def test_input_bars_not_mutated():
    bars = _bars(_session("2021-07-01", [0.001, 0.02, 0.001], [10.0, 20.0, 30.0]))
    before = bars.copy(deep=True)
    compute_jump_amount_corr(bars, min_pairs=2)
    pd.testing.assert_frame_equal(bars, before)


# --------------------------------------------------------------------------- #
# Spec + Factor subclass
# --------------------------------------------------------------------------- #
def test_factor_spec_is_valid_and_daily():
    spec = JumpAmountCorrFactor().spec
    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "jump_amount_corr_20"
    assert spec.is_intraday is False
    assert spec.expected_ic_sign == -1
    assert spec.return_basis == "close_to_close"
    assert spec.forward_return_horizon == 1
    # is_intraday=False => the whole minute block MUST be None (validated by FactorSpec).
    for field in ("decision_cutoff", "data_lag", "session_open", "execution_model", "execution_window"):
        assert getattr(spec, field) is None


def test_factor_subclass_passes_init_subclass_and_window_tracks_name():
    f = JumpAmountCorrFactor(lookback_days=10)
    assert f.name == "jump_amount_corr_10"
    assert f.spec.factor_id == "jump_amount_corr_10"


def test_factor_compute_selects_preaggregated_column():
    f = JumpAmountCorrFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    panel = pd.DataFrame({f.name: [0.25]}, index=idx)
    out = f.compute(panel)
    assert out.loc[(pd.Timestamp("2021-07-01"), _SYM)] == 0.25
    assert out.name == f.name


def test_factor_compute_missing_column_raises():
    f = JumpAmountCorrFactor()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-07-01"), _SYM)], names=["date", "symbol"]
    )
    with pytest.raises(ValueError, match="pre-aggregated"):
        f.compute(pd.DataFrame({"other": [1.0]}, index=idx))


def test_bad_params_raise():
    with pytest.raises(ValueError):
        JumpAmountCorrFactor(lookback_days=0)
    bars = _bars(_session("2021-07-01", [0.001, 0.02, 0.001], [10.0, 20.0, 30.0]))
    with pytest.raises(ValueError):
        compute_jump_amount_corr(bars, min_pairs=1)  # Pearson needs >= 2 points
    with pytest.raises(ValueError):
        compute_jump_amount_corr(bars, lookback_days=0)
