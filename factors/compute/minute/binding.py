"""Minute-factor raw-compute binding for the D4 materializer (design §3.5).

The materializer is factor-agnostic: it loads cutoff-filtered 1min bars and asks
"compute this minute factor's raw daily series from these bars". Each minute
Factor's ``compute(panel)`` only SURFACES a pre-aggregated column (D2), so this
module binds every minute factor to the free ``compute_*`` function that actually
does the aggregation, parameterized by the factor INSTANCE (its ``lookback_days``
and ``name``). All other definition constants stay at their module defaults —
the SAME single-source constants the eval runners / ``qt.panel_freeze`` recipes
pass (they pass them explicitly but equal to these defaults), so a materialized
minute factor matches the frozen baseline's math.

This lives in ``factors/compute/minute`` (the factor layer) rather than duplicated
in ``qt`` because it is FACTOR knowledge; ``qt.panel_freeze`` keeps its own
recipe copy (a frozen D1 tool) and D6 may later fold it onto this binding.

Deferred (readable error, D5): ``valley_price_quantile`` also needs the DAILY
close panel (its reversal neutralization), so it does not fit the pure
``(factor, bars) -> series`` shape and is NOT bound here — the materializer
raises for it rather than silently mis-computing.

Layering: factor layer only (never qt / feeds).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from factors.base import Factor
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AmpMarginalAnomalyVolFactor,
    compute_amp_marginal_anomaly_vol,
)
from factors.compute.minute.intraday_amp_cut import (
    IntradayAmpCutFactor,
    compute_intraday_amp_cut,
)
from factors.compute.minute.jump_amount_corr import (
    JumpAmountCorrFactor,
    compute_jump_amount_corr,
)
from factors.compute.minute.minute_ideal_amplitude import (
    MinuteIdealAmplitudeFactor,
    compute_minute_ideal_amplitude,
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
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.compute.minute.valley_relative_vwap import (
    ValleyRelativeVwapFactor,
    compute_valley_relative_vwap,
)
from factors.compute.minute.valley_ridge_vwap_ratio import (
    ValleyRidgeVwapRatioFactor,
    compute_valley_ridge_vwap_ratio,
)
from factors.compute.minute.primitives import VOLUME_PRV_BASELINE_DAYS
from factors.compute.minute.volume_peak_count import (
    VolumePeakCountFactor,
    compute_volume_peak_count,
)

#: bars -> daily raw factor Series, parameterized by the factor instance. The
#: cutoff is already applied by the materializer (bars are pre-filtered to
#: ``available_time <= decision``), so the compute functions run with their
#: DEFAULT decision_time (14:50) as a redundant-but-consistent internal cutoff.
BindingFn = Callable[[Factor, pd.DataFrame], pd.Series]


def _bind(compute_fn) -> BindingFn:
    """A ``(factor, bars)`` binding that calls ``compute_fn`` with the instance's
    lookback + name and every other definition constant at its module default."""

    def _call(factor: Factor, bars: pd.DataFrame) -> pd.Series:
        return compute_fn(
            bars, lookback_days=factor.lookback_days, name=factor.name  # type: ignore[attr-defined]
        )

    return _call


#: factor class -> its bars-based raw compute. Bound: the 10 minute factors whose
#: raw compute needs ONLY the 1min bars. NOT bound: valley_price_quantile (needs
#: the daily panel too — deferred, readable error below).
_MINUTE_BINDINGS: dict[type[Factor], BindingFn] = {
    JumpAmountCorrFactor: _bind(compute_jump_amount_corr),
    MinuteIdealAmplitudeFactor: _bind(compute_minute_ideal_amplitude),
    AmpMarginalAnomalyVolFactor: _bind(compute_amp_marginal_anomaly_vol),
    VolumePeakCountFactor: _bind(compute_volume_peak_count),
    IntradayAmpCutFactor: _bind(compute_intraday_amp_cut),
    PeakIntervalKurtosisFactor: _bind(compute_peak_interval_kurtosis),
    ValleyRelativeVwapFactor: _bind(compute_valley_relative_vwap),
    ValleyRidgeVwapRatioFactor: _bind(compute_valley_ridge_vwap_ratio),
    RidgeMinuteReturnFactor: _bind(compute_ridge_minute_return),
    PeakRidgeAmountRatioFactor: _bind(compute_peak_ridge_amount_ratio),
}

#: Minute factors deliberately NOT bound (need extra inputs), with the reason.
_DEFERRED: dict[type[Factor], str] = {
    ValleyPriceQuantileFactor: (
        "valley_price_quantile also consumes the DAILY close panel (its reversal "
        "neutralization), so it does not fit the pure (factor, bars) binding; the "
        "materializer defers it to D5 (a readable error, never a silent mis-compute)."
    ),
}


#: VALID-DAY POOLED factors (design §3.3 review HIGH): their trailing window
#: counts VALID days (invalid days do NOT occupy a slot — they roll over a
#: valid-day-indexed series via ``primitives.rolling_valid_days`` or an equivalent
#: ``.loc[valid_days].rolling(...)`` / valid-day python loop), so the CALENDAR
#: lookback depth of a trailing ``lookback_days`` valid-day pool is DATA-DEPENDENT
#: and UNBOUNDED, and loading more history can also re-classify boundary days.
#: A fixed ``lookback_depth`` trim therefore truncates sparse-valid-day symbols'
#: pools, making the stored value depend on LOAD GEOMETRY (red line #6: the value
#: must be f(factor, params, view, data), never the anchor/window shape). The
#: materializer must load these to SATURATION (design's structural terminal =
#: real data start) so single-fill and batch-fill agree. Classification is
#: STRUCTURAL (by the rolling mechanism, code-inspected), NOT by observed
#: divergence: only ridge_minute_return / valley_ridge_vwap_ratio /
#: peak_ridge_amount_ratio happened to diverge on the review's data, but
#: volume_peak_count / peak_interval_kurtosis / intraday_amp_cut /
#: valley_relative_vwap roll over valid days too and are the "happens-to-be-clean"
#: representatives (#82 lesson: never fix only what you observed).
#: valley_price_quantile is DEFERRED (needs the daily panel) but declared HERE so
#: D5 cannot bind it carrying the fixed-depth defect.
#: pooled factor class -> its same-slot BASELINE depth in trading days (the
#: "locking offset", review point 1): a loaded day is CLASSIFICATION-FINAL only
#: once it has this many strictly-prior trading days (the baseline reaches full
#: depth and no earlier history can change it). Derived from each factor's own
#: baseline parameter, NOT hardcoded: the peak family shares
#: ``VOLUME_PRV_BASELINE_DAYS``; ``intraday_amp_cut`` has NO cross-day baseline
#: (its classification is within-day), so its locking offset is 0.
_POOLED_BASELINE_DAYS: dict[type[Factor], int] = {
    VolumePeakCountFactor: VOLUME_PRV_BASELINE_DAYS,
    PeakIntervalKurtosisFactor: VOLUME_PRV_BASELINE_DAYS,
    ValleyRelativeVwapFactor: VOLUME_PRV_BASELINE_DAYS,
    ValleyRidgeVwapRatioFactor: VOLUME_PRV_BASELINE_DAYS,
    RidgeMinuteReturnFactor: VOLUME_PRV_BASELINE_DAYS,
    PeakRidgeAmountRatioFactor: VOLUME_PRV_BASELINE_DAYS,
    ValleyPriceQuantileFactor: VOLUME_PRV_BASELINE_DAYS,
    IntradayAmpCutFactor: 0,  # within-day classification, no cross-day baseline
}

VALID_DAY_POOLED_FACTORS: frozenset[type[Factor]] = frozenset(_POOLED_BASELINE_DAYS)

#: The bounded minute factors (trailing window over ALL trading days, not valid
#: days) — a fixed lookback_depth trim is load-geometry-free for these. Kept as
#: an explicit set so the classification is a CLOSED partition of the minute
#: surface (a new minute factor missing from BOTH sets is a readable error).
_BOUNDED_MINUTE_FACTORS: frozenset[type[Factor]] = frozenset({
    JumpAmountCorrFactor,
    MinuteIdealAmplitudeFactor,
    AmpMarginalAnomalyVolFactor,
})


def pooled_baseline_days(factor: Factor) -> int:
    """The factor's same-slot baseline depth = the LOCKING OFFSET (trading days).

    Expanding the load backward can only change the classification of days that
    sit within ``baseline_days`` trading days of the loaded window's start (the
    rolling baseline takes the ``baseline_days`` most recent STRICTLY-PRIOR
    same-slot observations; once it is at full depth, earlier history cannot
    change it). So the locking is POSITION-MONOTONE: drop the first
    ``baseline_days`` trading days of the loaded window and every later day's
    classification is FINAL. Readable error for a non-pooled factor.
    """
    cls = type(factor)
    if cls not in _POOLED_BASELINE_DAYS:
        raise KeyError(
            f"{factor.name} ({cls.__name__}) is not a valid-day-pooled factor, so "
            f"it has no baseline locking offset."
        )
    return int(_POOLED_BASELINE_DAYS[cls])


def pooled_lookback_days(factor: Factor) -> int:
    """The factor's trailing pool size in VALID days (its ``lookback_days``)."""
    if type(factor) not in _POOLED_BASELINE_DAYS:
        raise KeyError(f"{factor.name} is not a valid-day-pooled factor.")
    return int(factor.lookback_days)  # type: ignore[attr-defined]


def is_valid_day_pooled(factor: Factor) -> bool:
    """True iff ``factor``'s trailing window counts VALID days (unbounded depth).

    Readable error for a minute factor in NEITHER partition set — a new minute
    factor must be classified explicitly (never silently treated as bounded,
    which would reintroduce the load-geometry divergence).
    """
    cls = type(factor)
    if cls in VALID_DAY_POOLED_FACTORS:
        return True
    if cls in _BOUNDED_MINUTE_FACTORS:
        return False
    raise KeyError(
        f"{factor.name} ({cls.__name__}) is a minute factor not classified as "
        f"valid-day-pooled or bounded in factors.compute.minute.binding. Add it to "
        f"exactly one partition set (VALID_DAY_POOLED_FACTORS if its trailing "
        f"window counts VALID days, else _BOUNDED_MINUTE_FACTORS) — a missing "
        f"classification would silently size its saturation load wrong."
    )


def is_minute_bound(factor: Factor) -> bool:
    """True iff ``factor`` has a bars-only raw-compute binding here."""
    return type(factor) in _MINUTE_BINDINGS


def minute_raw_from_bars(factor: Factor, bars: pd.DataFrame) -> pd.Series:
    """Compute ``factor``'s raw daily Series from (cutoff-filtered) 1min ``bars``.

    Readable error for a minute factor that is deferred (needs extra inputs) or a
    non-minute-bound factor — never a silent wrong result.
    """
    binding = _MINUTE_BINDINGS.get(type(factor))
    if binding is not None:
        return binding(factor, bars)
    deferred = _DEFERRED.get(type(factor))
    if deferred is not None:
        raise NotImplementedError(f"{factor.name}: {deferred}")
    raise KeyError(
        f"{factor.name} ({type(factor).__name__}) has no minute-bars binding; it "
        f"is not a minute-derived factor bound in factors.compute.minute.binding."
    )


__all__ = [
    "VALID_DAY_POOLED_FACTORS",
    "BindingFn",
    "is_minute_bound",
    "is_valid_day_pooled",
    "minute_raw_from_bars",
    "pooled_baseline_days",
    "pooled_lookback_days",
]
