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

TWO GRANULARITIES (D4b). ``minute_raw_from_bars`` is the WHOLE-factor call: bars
in, daily values out. It requires every symbol's bars in ONE frame, which does
not fit the evaluation plane (995 symbols x 5 years: measured 52.7 GB nominal,
~115 GB peak RSS vs 56 GB available — ``docs/factors/d5_saturation_feasibility.md``).
So the streaming materializer uses the SPLIT pair instead:

* :func:`minute_stats_from_bars` — the PER-SYMBOL stage (memory-bounded: one
  symbol's minute history in, its small daily intermediate out);
* :func:`combine_minute_stats` — the CROSS-SECTIONAL combine, run ONCE on the
  assembled full-universe intermediate.

The cut sits BEFORE the cross-section on purpose. Ten of the eleven minute
factors are per-symbol pure, so their intermediate IS their value and the combine
is the identity; ``intraday_amp_cut`` is not — its step 4 z-scores each date
across the covered universe and needs >= ``AMP_CUT_MIN_CROSS_SECTION`` (=10)
finite pairs, so a ONE-SYMBOL cross-section is all-NaN BY DEFINITION (measured:
1692/1692 rows NaN). Splitting around the whole factor would destroy it; splitting
before the combine reproduces it exactly. Its module already organises the math
that way (``compute_amp_cut_stats`` then ``combine_amp_cut_cross_section``), as do
the eval runners — this binding surfaces that existing seam rather than inventing
one.

CONTRACT the streaming caller relies on: ``combine`` is a PER-DATE reduction (a
date's output depends only on that date's intermediate rows), so restricting the
intermediate to the emit window before combining cannot change a kept date's
value. Both combines satisfy it (identity; ``groupby(date)`` z-score).
``combine(per_symbol(bars)) == minute_raw_from_bars(bars)`` is pinned by test for
all eleven factors, so the two granularities cannot drift apart.

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
from dataclasses import dataclass
from typing import NoReturn

import pandas as pd

from factors.base import Factor
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AmpMarginalAnomalyVolFactor,
    compute_amp_marginal_anomaly_vol,
)
from factors.compute.minute.intraday_amp_cut import (
    V_MEAN_COL,
    V_STD_COL,
    IntradayAmpCutFactor,
    combine_amp_cut_cross_section,
    compute_amp_cut_stats,
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

# --------------------------------------------------------------------------- #
# Two-stage (streaming) binding: per-symbol stage + cross-sectional combine (D4b)
# --------------------------------------------------------------------------- #
#: Column name of the per-symbol intermediate for the factors whose per-symbol
#: stage ALREADY yields the final value (the combine is then the identity).
STATS_VALUE_COL = "value"


@dataclass(frozen=True)
class MinuteStreamBinding:
    """The split of a minute factor into per-symbol work + one cross-section pass.

    ``per_symbol(factor, bars)`` -> ``MultiIndex(date, symbol)`` intermediate frame
    for the ONE symbol carried by ``bars`` (it also accepts many symbols — it is
    the same math, just not memory-bounded then).

    ``combine(factor, stats)`` -> the factor's daily raw Series from the assembled
    full-universe intermediate. It MUST be a per-date reduction (see the module
    docstring's CONTRACT), which is what lets the streaming materializer restrict
    the intermediate to the emit window before combining.
    """

    per_symbol: Callable[[Factor, pd.DataFrame], pd.DataFrame]
    combine: Callable[[Factor, pd.DataFrame], pd.Series]
    #: The COLUMNS ``per_symbol`` produces. Declared because the intermediate of a
    #: cross-sectional factor is what the value store persists (D4c), and a stored
    #: artifact must be validated against the shape the code now expects without
    #: computing anything. Pinned against the columns actually produced by test.
    intermediate_columns: tuple[str, ...] = (STATS_VALUE_COL,)
    #: OPTIONAL per-symbol DAY-LEVEL diagnostics (D5): the gate-attrition frame a
    #: factor's scarcity disclosure reduces. Only the factors that HAVE such a
    #: disclosure set it; ``None`` means "this factor publishes no per-day
    #: diagnostics", which the caller must state rather than infer.
    #:
    #: It is a SEPARATE callable rather than an extra ``per_symbol`` output on
    #: purpose: ``per_symbol`` is the value path D4b verified cell-for-cell, and a
    #: diagnostics request must not be able to change it. The cost is that asking
    #: for diagnostics re-runs the per-symbol math on bars ALREADY in memory (no
    #: extra I/O, no effect on any value); that cost is disclosed at the call site.
    per_symbol_diagnostics: Callable[[Factor, pd.DataFrame], pd.DataFrame] | None = None


def _pure_stream(compute_fn) -> MinuteStreamBinding:
    """Stream binding for a PER-SYMBOL PURE factor: intermediate == value.

    Measured, not assumed: whole-universe vs per-symbol-concatenated agree to
    0.000e+00 on real bars for these (probe leg 4 / the D4b reconciliation).
    """

    def _per_symbol(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
        series = compute_fn(
            bars, lookback_days=factor.lookback_days, name=factor.name  # type: ignore[attr-defined]
        )
        return series.to_frame(STATS_VALUE_COL)

    def _combine(factor: Factor, stats: pd.DataFrame) -> pd.Series:
        if STATS_VALUE_COL not in stats.columns:
            raise ValueError(
                f"{factor.name}: the per-symbol intermediate must carry the "
                f"'{STATS_VALUE_COL}' column; got {list(stats.columns)}."
            )
        return stats[STATS_VALUE_COL].rename(factor.name).sort_index(kind="mergesort")

    return MinuteStreamBinding(per_symbol=_per_symbol, combine=_combine)


def _pure_stream_with_diagnostics(compute_fn) -> MinuteStreamBinding:
    """:func:`_pure_stream` plus the compute function's day-level diagnostics sink."""
    base = _pure_stream(compute_fn)
    return MinuteStreamBinding(
        per_symbol=base.per_symbol,
        combine=base.combine,
        per_symbol_diagnostics=_sink_diagnostics(compute_fn),
    )


def _sink_diagnostics(compute_fn) -> Callable[[Factor, pd.DataFrame], pd.DataFrame]:
    """Per-symbol diagnostics via the compute function's own ``diagnostics_out`` sink.

    The three ridge/peak-family factors already expose the day-level gate-attrition
    frame their scarcity disclosure reduces; this only routes it. The frame is
    returned CONCATENATED and the symbol column is left exactly as the compute
    function wrote it — the summarizers read it, and re-labelling here would be a
    second place for the schema to drift.
    """

    def _call(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
        sink: list[pd.DataFrame] = []
        compute_fn(
            bars,
            lookback_days=factor.lookback_days,  # type: ignore[attr-defined]
            name=factor.name,
            diagnostics_out=sink,
        )
        if not sink:
            return pd.DataFrame()
        return pd.concat(sink, ignore_index=False)

    return _call


def _amp_cut_per_symbol(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
    """Steps 1-3 of ``intraday_amp_cut``: the ``(v_mean, v_std)`` intermediate."""
    return compute_amp_cut_stats(
        bars, lookback_days=factor.lookback_days  # type: ignore[attr-defined]
    )


def _amp_cut_combine(factor: Factor, stats: pd.DataFrame) -> pd.Series:
    """Step 4 of ``intraday_amp_cut``: the per-date cross-sectional z-score.

    Run ONCE on the assembled universe — the ``min_cross_section`` gate (=10) is a
    factor DEFINITION constant and is never relaxed to accommodate streaming.
    """
    return combine_amp_cut_cross_section(stats, name=factor.name)


#: factor class -> its two-stage streaming binding. Same coverage as
#: ``_MINUTE_BINDINGS`` (the ten bars-only factors); a factor bound in one table
#: and not the other is a readable error, pinned by test.
_MINUTE_STREAM_BINDINGS: dict[type[Factor], MinuteStreamBinding] = {
    JumpAmountCorrFactor: _pure_stream(compute_jump_amount_corr),
    MinuteIdealAmplitudeFactor: _pure_stream(compute_minute_ideal_amplitude),
    AmpMarginalAnomalyVolFactor: _pure_stream(compute_amp_marginal_anomaly_vol),
    VolumePeakCountFactor: _pure_stream(compute_volume_peak_count),
    IntradayAmpCutFactor: MinuteStreamBinding(
        per_symbol=_amp_cut_per_symbol,
        combine=_amp_cut_combine,
        intermediate_columns=(V_MEAN_COL, V_STD_COL),
    ),
    PeakIntervalKurtosisFactor: _pure_stream(compute_peak_interval_kurtosis),
    ValleyRelativeVwapFactor: _pure_stream(compute_valley_relative_vwap),
    # The three factors that publish a per-day gate-attrition disclosure.
    ValleyRidgeVwapRatioFactor: _pure_stream_with_diagnostics(
        compute_valley_ridge_vwap_ratio
    ),
    RidgeMinuteReturnFactor: _pure_stream_with_diagnostics(compute_ridge_minute_return),
    PeakRidgeAmountRatioFactor: _pure_stream_with_diagnostics(
        compute_peak_ridge_amount_ratio
    ),
}

#: The factors whose cross-sectional combine is NOT the identity — i.e. the ones
#: for which "split per symbol around the WHOLE factor" would be wrong. Declared
#: so the property is a checkable fact rather than a comment.
CROSS_SECTIONAL_MINUTE_FACTORS: frozenset[type[Factor]] = frozenset({IntradayAmpCutFactor})

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


#: MEASURED exceptions: minute factors whose DAY-d VALUE depends on bars AFTER the
#: 14:50 decision cutoff, i.e. whose compute applies no cutoff of its own and must
#: therefore be handed pre-truncated bars to be decision-view. Under the
#: ``exec_to_exec`` basis a value like that uses information from after its own
#: 14:51 entry anchor, so an evaluation of it CANNOT honestly declare
#: ``view=decision``.
#:
#: This is a DENY LIST OF MEASURED FACTS, never a proof of safety: a factor absent
#: from it has either been measured clean or never measured.
#: ``tests/test_decision_cutoff_visibility.py`` is what does the measuring (perturb
#: only the post-cutoff bars, assert the pre-cutoff rows byte-identical, ask whether
#: the value moves) and it pins this set to exactly its own contents, so a second
#: offender cannot join silently.
#:
#: Declared HERE, next to the other minute-factor partitions, because it is a fact
#: about a factor's values — not about the evaluation path that has to consult it.
#:
#: CURRENTLY EMPTY, and that emptiness is MEASURED rather than assumed: the sole
#: entry was ``JumpAmountCorrFactor``, whose compute applied no truncation at all;
#: it now truncates at 14:50 like its ten siblings, and the measuring test moved it
#: into the clean parametrization (where a regression makes it red again). An empty
#: set still means "measured clean OR never measured" for anything outside
#: ``_MINUTE_STREAM_BINDINGS`` — the measuring test pins the bound ten explicitly.
NOT_DECISION_CUTOFF_SAFE: frozenset[type[Factor]] = frozenset()


def is_decision_cutoff_safe(factor: Factor) -> bool:
    """True iff ``factor``'s value at d uses no bar after d's 14:50 cutoff."""
    return type(factor) not in NOT_DECISION_CUTOFF_SAFE


def is_minute_bound(factor: Factor) -> bool:
    """True iff ``factor`` has a bars-only raw-compute binding here."""
    return type(factor) in _MINUTE_BINDINGS


def minute_raw_from_bars(factor: Factor, bars: pd.DataFrame) -> pd.Series:
    """Compute ``factor``'s raw daily Series from (cutoff-filtered) 1min ``bars``.

    The WHOLE-factor granularity: every symbol's bars must be in ``bars``. Prefer
    :func:`minute_stats_from_bars` + :func:`combine_minute_stats` when the universe
    does not fit in memory (see the module docstring).

    Readable error for a minute factor that is deferred (needs extra inputs) or a
    non-minute-bound factor — never a silent wrong result.
    """
    binding = _MINUTE_BINDINGS.get(type(factor))
    if binding is not None:
        return binding(factor, bars)
    _raise_unbound(factor)


def _stream_binding(factor: Factor) -> MinuteStreamBinding:
    binding = _MINUTE_STREAM_BINDINGS.get(type(factor))
    if binding is None:
        _raise_unbound(factor)
    return binding


def _raise_unbound(factor: Factor) -> NoReturn:
    """The shared readable error for a deferred / unbound minute factor."""
    deferred = _DEFERRED.get(type(factor))
    if deferred is not None:
        raise NotImplementedError(f"{factor.name}: {deferred}")
    raise KeyError(
        f"{factor.name} ({type(factor).__name__}) has no minute-bars binding; it "
        f"is not a minute-derived factor bound in factors.compute.minute.binding."
    )


def minute_stats_from_bars(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
    """``factor``'s PER-SYMBOL intermediate from (cutoff-filtered) 1min ``bars``.

    Memory-bounded stage: ``bars`` normally carries ONE symbol, and the returned
    ``MultiIndex(date, symbol)`` frame is daily-sized, so the caller can discard
    the minute bars before reading the next symbol. For a per-symbol-pure factor
    the frame is the value itself (column ``STATS_VALUE_COL``); for
    ``intraday_amp_cut`` it is the pre-cross-section ``(v_mean, v_std)`` panel.
    """
    return _stream_binding(factor).per_symbol(factor, bars)


def has_minute_diagnostics(factor: Factor) -> bool:
    """True iff ``factor`` publishes a per-symbol day-level diagnostics frame."""
    return _stream_binding(factor).per_symbol_diagnostics is not None


def minute_diagnostics_from_bars(factor: Factor, bars: pd.DataFrame) -> pd.DataFrame:
    """``factor``'s per-symbol day-level diagnostics frame (empty when it has none).

    Returns an EMPTY frame — not an error — for a factor without a diagnostics
    binding, so a caller can ask uniformly; whether the factor HAS one is the
    separate, explicit :func:`has_minute_diagnostics` question, and a caller that
    needs the disclosure must check it rather than read an empty frame as "no
    attrition".
    """
    fn = _stream_binding(factor).per_symbol_diagnostics
    return pd.DataFrame() if fn is None else fn(factor, bars)


def minute_intermediate_columns(factor: Factor) -> tuple[str, ...]:
    """The columns ``factor``'s per-symbol intermediate carries (D4c).

    The value store persists this frame verbatim for a cross-sectional factor, so
    the shape has to be answerable WITHOUT computing anything — a stored artifact
    is validated against it on read.
    """
    return tuple(_stream_binding(factor).intermediate_columns)


def combine_minute_stats(factor: Factor, stats: pd.DataFrame) -> pd.Series:
    """``factor``'s daily raw Series from the ASSEMBLED per-symbol intermediates.

    Run ONCE on the full-universe intermediate: this is where a cross-sectional
    factor's date-wise standardization happens, so its universe is the whole
    covered set exactly as in the single-frame path.
    """
    return _stream_binding(factor).combine(factor, stats)


def is_cross_sectional_minute(factor: Factor) -> bool:
    """True iff ``factor``'s combine is NOT the identity (a real cross-section)."""
    return type(factor) in CROSS_SECTIONAL_MINUTE_FACTORS


__all__ = [
    "CROSS_SECTIONAL_MINUTE_FACTORS",
    "NOT_DECISION_CUTOFF_SAFE",
    "STATS_VALUE_COL",
    "VALID_DAY_POOLED_FACTORS",
    "BindingFn",
    "MinuteStreamBinding",
    "combine_minute_stats",
    "has_minute_diagnostics",
    "is_cross_sectional_minute",
    "is_decision_cutoff_safe",
    "is_minute_bound",
    "is_valid_day_pooled",
    "minute_diagnostics_from_bars",
    "minute_intermediate_columns",
    "minute_raw_from_bars",
    "minute_stats_from_bars",
    "pooled_baseline_days",
    "pooled_lookback_days",
]
