"""INTRADAY PIT boundary: no minute factor may read a post-cutoff bar.

Why this file exists (the blind spot it closes)
-----------------------------------------------
Every minute factor already had a leakage test — but all of them guard the
CROSS-DAY boundary ("perturbing day d+1 must not move day d"). The INTRA-day
boundary was untested: the fixtures of those tests are five-or-six-bar morning
sessions that do not even contain a 14:50, so the assertion "the 14:50 cutoff is
applied" was never expressed anywhere.

That boundary became load-bearing when PR #79 moved the entry anchor from
``close(d)`` to the 14:51 execution bar: under the exec_to_exec basis a factor
value at d that reads d's 14:50-15:00 bars is strictly forward-looking — the
information set crosses its own entry anchor, in the single most
volume-concentrated window of the A-share session. ``jump_amount_corr`` was the
one factor of the eleven with no truncation at all (its ``decision_time`` /
``prepare_visible_minute_bars`` reference count was 0 while the other ten were
4-10), and nothing went red.

So the guard here is deliberately GENERIC: it covers every ``Factor`` subclass
defined under ``factors/compute/minute`` (discovered by walking the package, not
by a hand-maintained list — §六.8, and the same lesson D4b learned when dropping
a factor from a binding table left the suite green). D6c registered both I3
session features as Factor classes
(``MmpEwFactor`` / ``IntradaySessionRetFactor``), so the class walk covers them;
D6d deleted the legacy aggregate-feature hook they replaced, so the factor
classes are now the ONLY read path and the only surface this guard needs.

The two knives
--------------
1. ``INVISIBLE`` — every bar the visibility rule excludes
   (``available_time > trade_date + decision_time``; with the 1-minute data lag
   that is ``bar_end >= 14:50``). This is the full set a PIT-truncating factor
   must be blind to.
2. ``EXEC_AND_AFTER`` — only ``bar_end >= 14:51``, i.e. the execution bar itself
   and the nine minutes after it. A strict subset of (1), and the sharper
   statement: a factor that moves under this perturbation has seen the very
   minute it is modelled as trading in.

Both must leave every factor value BIT-identical (NaN mask included).

Not-vacuous, by assertion rather than by hope
---------------------------------------------
Each perturbation is checked BEFORE it is used:

* the mask selects at least one bar (and, for the factors, at least one bar on
  every fixture day);
* the perturbed frame really differs from the base frame on EVERY selected row;
* the perturbed frame is byte-identical to the base frame on EVERY unselected
  row (so a "pass" cannot come from a mutation that changed nothing, or from one
  that changed the visible half too).

And each factor must emit at least one finite value on the base fixture,
otherwise "unchanged" would be the trivially-true statement that NaN == NaN.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import (
    DEFAULT_DECISION_TIME,
    SYMBOL_LEVEL,
    TIME_LEVEL,
    normalize_intraday_bars,
)
from factors.base import Factor
from factors.compute.minute import binding as binding_module
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AmpMarginalAnomalyVolFactor,
    compute_amp_marginal_anomaly_vol,
)
from factors.compute.minute.intraday_amp_cut import (
    IntradayAmpCutFactor,
    compute_intraday_amp_cut,
)
from factors.compute.minute.intraday_session_ret import (
    IntradaySessionRetFactor,
    compute_intraday_session_ret,
)
from factors.compute.minute.jump_amount_corr import (
    JumpAmountCorrFactor,
    compute_jump_amount_corr,
)
from factors.compute.minute.minute_ideal_amplitude import (
    MinuteIdealAmplitudeFactor,
    compute_minute_ideal_amplitude,
)
from factors.compute.minute.mmp import (
    MmpEwFactor,
    compute_intraday_mmp_ew,
)
from factors.compute.minute.peak_interval_kurtosis import (
    PeakIntervalKurtosisFactor,
    compute_peak_interval_kurtosis,
)
from factors.compute.minute.peak_ridge_amount_ratio import (
    PeakRidgeAmountRatioFactor,
    compute_peak_ridge_amount_ratio,
)
from factors.compute.minute.ridge_minute_return import (
    RidgeMinuteReturnFactor,
    compute_ridge_minute_return,
)
from factors.compute.minute.valley_price_quantile import (
    ValleyPriceQuantileFactor,
    compute_valley_price_quantile,
)
from factors.compute.minute.valley_relative_vwap import (
    ValleyRelativeVwapFactor,
    compute_valley_relative_vwap,
)
from factors.compute.minute.valley_ridge_vwap_ratio import (
    ValleyRidgeVwapRatioFactor,
    compute_valley_ridge_vwap_ratio,
)
from factors.compute.minute.volume_peak_count import (
    VolumePeakCountFactor,
    compute_volume_peak_count,
)

# --------------------------------------------------------------------------- #
# Fixture: a realistic A-share session that STRADDLES the cutoff
# --------------------------------------------------------------------------- #
# Morning 09:31..11:30 (120 bars) + afternoon 13:01..15:00 (120 bars) = 240/day.
# The afternoon deliberately runs to 15:00 so bars exist on BOTH sides of 14:50.
_N_DAYS = 40  # >= baseline_days (20) + min_valid_days (10) for the peak family
_N_SYMBOLS = 10  # >= AMP_CUT_MIN_CROSS_SECTION / VALLEY_QUANTILE_MIN_CROSS_SECTION
_SYMBOLS = tuple(f"{600000 + i}.SH" for i in range(_N_SYMBOLS))
_CUTOFF = DEFAULT_DECISION_TIME  # "14:50:00"
_EXEC_MINUTE = "14:51:00"


def _session_minutes(day: pd.Timestamp) -> pd.DatetimeIndex:
    morning = pd.date_range(day + pd.Timedelta("09:31:00"), periods=120, freq="1min")
    afternoon = pd.date_range(day + pd.Timedelta("13:01:00"), periods=120, freq="1min")
    return morning.append(afternoon)


def _build_bars() -> pd.DataFrame:
    """Deterministic pseudo-random 1min bars for ``_N_SYMBOLS`` x ``_N_DAYS``."""
    rng = np.random.default_rng(20260725)
    days = pd.bdate_range("2021-07-01", periods=_N_DAYS)
    frames = []
    for s_i, sym in enumerate(_SYMBOLS):
        for day in days:
            times = _session_minutes(day)
            n = len(times)
            # Random-walk close around a per-symbol level; high/low bracket it.
            drift = rng.normal(0.0, 0.0009, n).cumsum()
            close = (10.0 + s_i) * np.exp(drift)
            spread = np.abs(rng.normal(0.0018, 0.0011, n)) + 1e-4
            high = close * (1.0 + spread)
            low = close * (1.0 - spread)
            open_ = np.concatenate([[close[0]], close[:-1]])
            volume = rng.lognormal(9.0, 0.8, n)
            frames.append(
                pd.DataFrame(
                    {
                        "time": times,
                        "symbol": sym,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": volume * close,
                    }
                )
            )
    return normalize_intraday_bars(pd.concat(frames, ignore_index=True), freq="1min")


BARS = _build_bars()
#: Stand-in for the runner's DAILY front-adjusted close panel (MultiIndex(date,
#: symbol) with a ``close`` column) that ``valley_price_quantile`` neutralizes
#: against. It is a DAILY input, so it is built once from the base bars and held
#: FIXED across every perturbation below.
_DAILY_CLOSE = (
    BARS.reset_index()
    .assign(date=lambda f: f[TIME_LEVEL].dt.normalize())
    .groupby(["date", SYMBOL_LEVEL], sort=True)["close"]
    .last()
    .to_frame("close")
)

# Perturbed columns: every numeric channel any minute factor reads.
_PERTURBED_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def _invisible_mask(bars: pd.DataFrame) -> np.ndarray:
    """Bars the visibility rule excludes: ``available_time > trade_date + cutoff``."""
    trade_date = bars["bar_end"].dt.normalize()
    return (bars["available_time"] > trade_date + pd.Timedelta(_CUTOFF)).to_numpy()


def _exec_and_after_mask(bars: pd.DataFrame) -> np.ndarray:
    """The execution bar itself and every bar after it (``bar_end >= 14:51``)."""
    trade_date = bars["bar_end"].dt.normalize()
    return (bars["bar_end"] >= trade_date + pd.Timedelta(_EXEC_MINUTE)).to_numpy()


_MASKS = {"invisible": _invisible_mask, "exec_and_after": _exec_and_after_mask}


def _perturb(bars: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Wildly move every perturbed channel on the masked rows only.

    The price channels stay internally consistent (``low <= close <= high``,
    all strictly positive) so the perturbed frame still passes
    ``validate_intraday_bars`` — a factor must be blind to these bars because
    they are in the future, not because they are malformed.
    """
    out = bars.copy(deep=True)
    scale = 137.0  # far outside any plausible per-minute move
    out.loc[mask, "high"] = bars.loc[mask, "high"].to_numpy() * scale
    out.loc[mask, "low"] = bars.loc[mask, "low"].to_numpy() / scale
    out.loc[mask, "close"] = bars.loc[mask, "close"].to_numpy() * 7.0
    out.loc[mask, "open"] = bars.loc[mask, "open"].to_numpy() / 5.0
    out.loc[mask, "volume"] = bars.loc[mask, "volume"].to_numpy() * 9.0e5 + 1.0
    out.loc[mask, "amount"] = bars.loc[mask, "amount"].to_numpy() * 9.0e5 + 1.0
    return out


def _assert_mutation_is_real_and_targeted(
    base: pd.DataFrame, perturbed: pd.DataFrame, mask: np.ndarray, label: str
) -> None:
    """The mutation moved EVERY selected row and NO unselected row.

    Without this, a green invariance test proves nothing: the project has six
    recorded instances of a mutation that changed nothing while the test that
    depended on it stayed green.
    """
    assert mask.any(), f"{label}: the mask selected no bar — the fixture is vacuous"
    cols = list(_PERTURBED_COLUMNS)
    changed = (base.loc[mask, cols].to_numpy() != perturbed.loc[mask, cols].to_numpy()).any(
        axis=1
    )
    assert changed.all(), (
        f"{label}: {int((~changed).sum())} selected bar(s) were NOT changed by the "
        f"perturbation — the mutation does not reach what this test claims it does"
    )
    pd.testing.assert_frame_equal(base.loc[~mask], perturbed.loc[~mask])


# --------------------------------------------------------------------------- #
# The covered surface: every minute Factor class
# --------------------------------------------------------------------------- #
def _jump(bars: pd.DataFrame) -> pd.Series:
    return compute_jump_amount_corr(bars)


def _ideal_amp(bars: pd.DataFrame) -> pd.Series:
    return compute_minute_ideal_amplitude(bars)


def _amp_anomaly(bars: pd.DataFrame) -> pd.Series:
    return compute_amp_marginal_anomaly_vol(bars)


def _peak_count(bars: pd.DataFrame) -> pd.Series:
    return compute_volume_peak_count(bars)


def _amp_cut(bars: pd.DataFrame) -> pd.Series:
    return compute_intraday_amp_cut(bars)


def _peak_interval(bars: pd.DataFrame) -> pd.Series:
    return compute_peak_interval_kurtosis(bars)


def _valley_vwap(bars: pd.DataFrame) -> pd.Series:
    return compute_valley_relative_vwap(bars)


def _valley_ridge(bars: pd.DataFrame) -> pd.Series:
    return compute_valley_ridge_vwap_ratio(bars)


def _ridge_return(bars: pd.DataFrame) -> pd.Series:
    return compute_ridge_minute_return(bars)


def _peak_ridge_amount(bars: pd.DataFrame) -> pd.Series:
    return compute_peak_ridge_amount_ratio(bars)


def _valley_quantile(bars: pd.DataFrame) -> pd.Series:
    # The daily close panel is a DAILY input, not a minute one: it is held FIXED
    # across the perturbation on purpose, so the only thing that can move the
    # factor is the minute channel this test is about.
    return compute_valley_price_quantile(bars, _DAILY_CLOSE)


def _mmp_ew_factor(bars: pd.DataFrame) -> pd.Series:
    return compute_intraday_mmp_ew(bars)


def _session_ret(bars: pd.DataFrame) -> pd.Series:
    return compute_intraday_session_ret(bars)


#: minute Factor class -> the free compute that produces its raw daily series.
_FACTOR_COMPUTES: dict[type[Factor], object] = {
    JumpAmountCorrFactor: _jump,
    MinuteIdealAmplitudeFactor: _ideal_amp,
    AmpMarginalAnomalyVolFactor: _amp_anomaly,
    VolumePeakCountFactor: _peak_count,
    IntradayAmpCutFactor: _amp_cut,
    PeakIntervalKurtosisFactor: _peak_interval,
    ValleyRelativeVwapFactor: _valley_vwap,
    ValleyRidgeVwapRatioFactor: _valley_ridge,
    RidgeMinuteReturnFactor: _ridge_return,
    PeakRidgeAmountRatioFactor: _peak_ridge_amount,
    ValleyPriceQuantileFactor: _valley_quantile,
    MmpEwFactor: _mmp_ew_factor,
    IntradaySessionRetFactor: _session_ret,
}

#: The full covered surface: the thirteen factors keyed by name. (Until D6d the
#: legacy ``mmp_ew`` aggregate-feature path was pinned here too as a second read
#: path of the same signal; D6d deleted that hook, and ``MmpEwFactor`` above IS
#: the surviving single definition point.)
_CASES: dict[str, object] = {
    **{cls().name: fn for cls, fn in _FACTOR_COMPUTES.items()},
}


def _minute_factor_classes() -> dict[type, str]:
    """Every ``Factor`` subclass DEFINED under ``factors/compute/minute``.

    The authoritative surface (same walk as the D4b binding-closure test): a new
    minute factor file cannot be added without either being covered here or
    turning this file red.
    """
    package = pathlib.Path(binding_module.__file__).parent
    defined: dict[type, str] = {}
    for path in sorted(package.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = importlib.import_module(f"factors.compute.minute.{path.stem}")
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Factor)
                and obj is not Factor
                and obj.__module__ == module.__name__
            ):
                defined[obj] = path.name
    return defined


def test_every_minute_factor_class_is_covered_by_a_cutoff_case():
    """A minute factor absent from ``_FACTOR_COMPUTES`` fails HERE, not silently.

    Deriving the parametrize list from a hand-written dict only guards the
    direction "listed -> tested". The other direction — a factor that exists but
    is in no list — is exactly how a leakage guard goes quietly vacuous, so the
    source of truth is the package walk, not the dict.
    """
    defined = _minute_factor_classes()
    assert defined, "found no minute factor classes — the walk itself is broken"
    missing = {cls.__name__: defined[cls] for cls in defined if cls not in _FACTOR_COMPUTES}
    assert not missing, (
        f"minute factor(s) with no decision-cutoff leakage case: {missing} — they "
        f"would silently keep the right to read post-14:50 bars"
    )
    stale = {cls.__name__ for cls in _FACTOR_COMPUTES if cls not in defined}
    assert not stale, f"leakage cases for classes no longer defined: {stale}"


# --------------------------------------------------------------------------- #
# The perturbations themselves are real and targeted (pre-conditions)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mask_name", sorted(_MASKS))
def test_perturbation_is_real_and_targeted(mask_name):
    mask = _MASKS[mask_name](BARS)
    _assert_mutation_is_real_and_targeted(BARS, _perturb(BARS, mask), mask, mask_name)


def test_masks_are_nested_and_both_non_trivial():
    """``exec_and_after`` is a STRICT subset of ``invisible`` on this fixture.

    Pins the relationship the two knives are supposed to have: the sharper one
    must select fewer bars (it starts one minute later), and both must leave
    the visible half of every day untouched.
    """
    invisible = _invisible_mask(BARS)
    exec_after = _exec_and_after_mask(BARS)
    assert exec_after.sum() > 0
    assert int(invisible.sum()) - int(exec_after.sum()) == _N_DAYS * _N_SYMBOLS, (
        "expected exactly the 14:50 bar per (day, symbol) to separate the two masks"
    )
    assert not (exec_after & ~invisible).any(), "exec_and_after must be inside invisible"
    # The visible half is a real, large majority — a cutoff that ate the session
    # would make every "unchanged" assertion below trivially true.
    assert (~invisible).sum() > 0.9 * invisible.sum() * 10


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def base_values() -> dict[str, pd.Series]:
    return {name: fn(BARS) for name, fn in _CASES.items()}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_base_fixture_produces_finite_values(case, base_values):
    """Non-vacuity: "unchanged" must not mean "NaN both times"."""
    series = base_values[case]
    assert int(np.isfinite(series.to_numpy(dtype=float)).sum()) > 0, (
        f"{case}: the base fixture produced no finite value, so the invariance "
        f"assertions below would be vacuous"
    )


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("mask_name", sorted(_MASKS))
def test_post_cutoff_bars_cannot_move_a_minute_factor(case, mask_name, base_values):
    """Perturbing bars the factor must not see leaves its values BIT-identical.

    ``jump_amount_corr`` failed both knives before the truncation fix (measured:
    every emitted cell moved, max|diff| > 1.0 on a real-cache sample) while the
    other ten were already exactly 0 — the asymmetry is the whole point of
    running this over the full surface rather than over the factor that was
    known to be broken.
    """
    mask = _MASKS[mask_name](BARS)
    perturbed = _perturb(BARS, mask)
    _assert_mutation_is_real_and_targeted(BARS, perturbed, mask, mask_name)

    got = _CASES[case](perturbed)
    want = base_values[case]

    assert got.index.equals(want.index), (
        f"{case} / {mask_name}: the emitted (date, symbol) grid moved — a "
        f"post-cutoff bar changed WHICH values exist"
    )
    g = got.to_numpy(dtype=float)
    w = want.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(g), np.isnan(w)), (
        f"{case} / {mask_name}: the NaN mask moved"
    )
    finite = ~np.isnan(w)
    if finite.any():
        worst = float(np.max(np.abs(g[finite] - w[finite])))
        assert worst == 0.0, (
            f"{case} / {mask_name}: {int((g[finite] != w[finite]).sum())} of "
            f"{int(finite.sum())} finite values moved (max|diff| = {worst:g}) — the "
            f"factor reads bars it cannot know at the decision time"
        )
