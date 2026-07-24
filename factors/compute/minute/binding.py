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


__all__ = ["BindingFn", "is_minute_bound", "minute_raw_from_bars"]
