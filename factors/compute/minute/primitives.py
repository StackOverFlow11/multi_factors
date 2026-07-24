"""Shared minute-factor primitives (D2, design v3.2 §3.2).

The ONE home of the machinery every minute factor shares:

* ``prepare_visible_minute_bars`` + ``peak_mask_for_symbol`` — the peak/ridge/
  valley taxonomy, moved house VERBATIM from ``data.clean.intraday_volume_prv``
  (it was already author-once there; the move is relocation, not repair — R1);
* ``visible_minute_frame`` / ``guarded_amplitude`` / ``add_bar_end_ns`` — the
  PIT front half + amplitude price guard of the non-peak factor family
  (jump / ideal-amplitude / amp-anomaly / amp-cut share these blocks);
* ``rolling_valid_days`` / ``pooled_trailing_reduce`` — the two trailing-window
  forms (rolling over VALID days only, and pool-then-reduce over trailing
  days), each preserving the factors' exact float operation order;
* ``positive_trade_mask`` — the valley/ridge family's trade guard for the
  ``Σamount/Σvolume`` VWAP aggregation identity;
* ``symbol_frames`` / ``empty_factor_series`` — the per-symbol isolation split
  and the schema-shaped empty output.

Per-factor DEFINITION parameters (``min_valley_bars=20`` / ``min_ridge_bars=10``
/ "counted AFTER the guard" / …) are pre-registered definitions and stay
explicit parameters of these primitives — parameterized, never unified (§〇
总原则: the refactor changes implementations, not definitions).

Layering: imports ``data.clean.intraday_schema`` (real data-layer code) and
numpy/pandas only. It must NEVER import ``data.clean.intraday_aggregate`` —
that module re-exports the migrated MMP/jump math FROM ``factors.compute.
minute``, so an import back would be a genuine cycle.

Latency note (§八, load-bearing): this layer exists so the 11 factors share one
bars read + one normalization + one classification (measured ≈17.6s shared vs
≈58s naive per-factor re-reads on the CSI500 tail window); the shared-read
orchestration itself arrives with the D4 materializer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Callable

import numpy as np
import pandas as pd

from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    SYMBOL_LEVEL,
)

# --------------------------------------------------------------------------- #
# Peak/ridge/valley classification constants (factor DEFINITION constants,
# pinned interpretations of the Kaiyuan series-27 report; NOT tuned knobs).
# Moved from data/clean/intraday_volume_prv.py together with the taxonomy —
# they are the classification's parameters, shared by the whole peak family.
# --------------------------------------------------------------------------- #
VOLUME_PRV_BASELINE_DAYS = 20  # strictly-prior same-slot baseline window (trading days)
VOLUME_PRV_BASELINE_MIN_OBS = 10  # min same-slot obs for a classifiable bar
VOLUME_PRV_SIGMA_K = 1.0  # eruptive threshold multiplier k in vol > μ + k*σ
VOLUME_PRV_MIN_VALID_DAYS = 10  # min valid days in a trailing window for a finite value
VOLUME_PRV_MIN_CLASSIFIABLE = 100  # a day is valid iff it has >= this many classifiable bars

# The "strictly-next minute" test in seconds: two bars are same-session 1-minute
# neighbours iff their bar_end gap is EXACTLY 60s. The lunch break (11:30 -> 13:01),
# the session close and any missing minute all differ, so they are not neighbours.
ONE_MINUTE_SECONDS = 60.0

# The columns every peak-family factor needs. ``prepare_visible_minute_bars`` always
# emits EXACTLY these (plus the derived ``trade_date`` / ``slot``); anything else a
# caller needs is opt-in via ``extra_columns``, so the default output stays byte-identical
# for the callers that predate the option.
_BASE_VISIBLE_COLUMNS = [SYMBOL_LEVEL, "bar_end", "available_time", "volume"]


def empty_factor_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def visible_minute_frame(
    bars: pd.DataFrame,
    *,
    columns: Sequence[str],
    decision_time: str = DEFAULT_DECISION_TIME,
) -> pd.DataFrame:
    """PIT front half of the non-peak minute factors (jump/amplitude/anomaly/cut).

    Selects ``symbol`` / ``bar_end`` / ``available_time`` plus ``columns`` off
    the normalized bars, derives ``trade_date``, and keeps ONLY the bars whose
    ``available_time`` is at or before their own ``trade_date + decision_time``
    (per-bar timestamps FIRST — history days and the signal day are truncated
    identically, and a post-cutoff bar can never leak into a decision).

    Returns a fresh frame (possibly empty). Pure: never mutates ``bars``.
    """
    work = bars.reset_index()[
        [SYMBOL_LEVEL, "bar_end", "available_time", *columns]
    ].copy()
    work["trade_date"] = work["bar_end"].dt.normalize()
    cutoff = work["trade_date"] + pd.Timedelta(decision_time)
    return work.loc[work["available_time"] <= cutoff].copy()


def guarded_amplitude(visible: pd.DataFrame) -> pd.DataFrame:
    """Amplitude price guard + per-bar amplitude for the amplitude family.

    Drops every bar unless ``low > 0`` and ``high >= low`` (a bar failing the
    guard has no meaningful amplitude), then adds ``amp = high/low - 1`` on the
    survivors. Returns a fresh frame (possibly empty, in which case no ``amp``
    column is added). Pure: never mutates ``visible``.
    """
    low = visible["low"].to_numpy(dtype=float)
    high = visible["high"].to_numpy(dtype=float)
    guard = (low > 0.0) & (high >= low)
    out = visible.loc[guard].copy()
    if out.empty:
        return out
    out["amp"] = out["high"].to_numpy(dtype=float) / out["low"].to_numpy(
        dtype=float
    ) - 1.0
    return out


def add_bar_end_ns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``bar_end_ns``: int64 nanoseconds for a deterministic lexsort tie-break.

    (The view via ``datetime64[ns]`` avoids the datetime->int astype
    deprecation.) Returns ``frame`` itself with the column added — callers own
    the frame (it is a fresh copy from the preparation helpers).
    """
    frame["bar_end_ns"] = frame["bar_end"].to_numpy(dtype="datetime64[ns]").astype(
        "int64"
    )
    return frame


def prepare_visible_minute_bars(
    bars: pd.DataFrame,
    *,
    decision_time: str = DEFAULT_DECISION_TIME,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """PIT-truncate ``bars`` at ``decision_time`` and add ``trade_date`` / ``slot``.

    The shared front half of every peak-taxonomy factor: keep only the bars whose
    ``available_time`` is at or before their own ``trade_date + decision_time`` (so
    history days and the signal day are truncated identically), drop non-finite /
    negative volumes (invalid data that would poison the same-slot μ/σ), and add the
    minute-of-day ``slot`` that aligns the SAME time-of-day across days.

    ``bars`` must ALREADY be schema-validated by the caller
    (:func:`data.clean.intraday_schema.validate_intraday_bars`) — this helper is the
    post-validation preparation step, kept separate so the entry points keep their own
    validate-then-check-params ordering.

    Args:
        bars: normalized 1min bars, ``MultiIndex(time, symbol)``; one or many symbols.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        extra_columns: additional ``bars`` columns to carry through, appended AFTER the
            base columns. The truncation and the volume guard are unaffected — the extra
            values merely ride along on the surviving rows — so the default ``()`` keeps
            the output byte-identical for callers that do not need them (PR-F / PR-H).
            PR-I passes ``("amount",)`` to compute a volume-weighted price.

    Returns:
        A fresh ``RangeIndex`` frame with columns ``symbol`` / ``bar_end`` /
        ``available_time`` / ``volume`` / ``trade_date`` / ``slot`` plus any
        ``extra_columns`` (possibly empty). Pure: never mutates ``bars``.
    """
    extra = list(extra_columns)
    unknown = [c for c in extra if c not in bars.columns]
    if unknown:
        raise ValueError(
            f"prepare_visible_minute_bars extra_columns not present on bars: {unknown}; "
            f"bars has {list(bars.columns)}."
        )
    clashing = [c for c in extra if c in _BASE_VISIBLE_COLUMNS]
    if clashing:
        raise ValueError(
            f"prepare_visible_minute_bars extra_columns are already emitted: {clashing}."
        )
    work = bars.reset_index()[_BASE_VISIBLE_COLUMNS + extra].copy()
    work["trade_date"] = work["bar_end"].dt.normalize()
    # PIT truncation FIRST (per-bar timestamps): each bar's cutoff is its own
    # trade_date + decision_time, so every day is truncated to [open, cutoff].
    cutoff = work["trade_date"] + pd.Timedelta(decision_time)
    visible = work.loc[work["available_time"] <= cutoff].copy()
    if visible.empty:
        return visible

    # Guard bad volume BEFORE anything else: a non-finite / negative volume is invalid
    # data that would poison the same-slot μ/σ. (Split-day magnitude jumps are NOT
    # corrected — disclosed, matching the report's Wind treatment.)
    vol = visible["volume"].to_numpy(dtype=float)
    visible = visible.loc[np.isfinite(vol) & (vol >= 0.0)].copy()
    if visible.empty:
        return visible

    # slot = minute-of-day (minutes since midnight); bars are minute-aligned so this is
    # exact and aligns the SAME time-of-day across days for the same-slot baseline.
    visible["slot"] = (
        (visible["bar_end"] - visible["trade_date"]) // pd.Timedelta(minutes=1)
    ).astype(int)
    return visible


def peak_mask_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
) -> pd.DataFrame:
    """Per-minute PEAK / classifiable mask for ONE symbol's PIT-visible bars.

    This is THE volume-peak identification of the report (§2), shared by every factor in
    the peak family so the taxonomy is defined in exactly one place: same-slot
    strictly-prior μ/σ baseline -> eruptive vs mild -> a peak is an eruptive minute whose
    both 1-minute same-session neighbours are mild.

    ``g`` holds columns ``trade_date`` / ``slot`` / ``bar_end`` / ``volume`` for a
    SINGLE symbol (a fresh ``RangeIndex``, as produced by
    :func:`prepare_visible_minute_bars`). The same-slot STRICTLY-PRIOR baseline is
    computed on a day x slot pivot (rolling over the day axis then ``shift(1)``), so no
    per-``(day, slot)`` python loop is needed and the current day never enters its own
    baseline. Peak detection is done back in long form with a within-day 60s-gap
    neighbour test (which enforces "same continuous session" and "the 1-minute
    neighbour" at once). No cross-symbol leakage — ``g`` is one symbol's slice — and no
    cross-day leakage — the baseline is strictly prior and the neighbour test never
    crosses a day boundary (grouped by ``trade_date``).

    Returns:
        ``g`` sorted by ``(trade_date, bar_end)`` on a fresh ``RangeIndex`` with the
        added boolean columns ``classifiable`` (a strictly-prior baseline existed),
        ``valley`` (classifiable and NOT eruptive — the report's 量谷, what PR-I prices),
        ``peak`` and ``ridge`` (eruptive but NOT an isolated peak — the report's 量岭,
        what PR-J prices). The three masks PARTITION the classifiable bars exactly:
        ``valley`` holds the mild ones and ``peak`` / ``ridge`` split the eruptive ones,
        so ``valley | peak | ridge == classifiable`` and no two overlap. Pure: never
        mutates ``g``.
    """
    # day x slot volume matrix: rows are the symbol's trading days, columns are the
    # minute-of-day slots; a missing (day, slot) cell is NaN (no bar that minute).
    v = g.pivot(index="trade_date", columns="slot", values="volume")
    v = v.sort_index().sort_index(axis=1)

    # Strictly-prior same-slot baseline: rolling over the DAY axis (per slot column),
    # requiring >= baseline_min_obs actual same-slot observations, THEN shift(1) so
    # day t uses days t-baseline_days .. t-1 only (never day t itself).
    roll = v.rolling(baseline_days, min_periods=baseline_min_obs)
    mu = roll.mean().shift(1)
    sigma = roll.std().shift(1)  # ddof=1 (pandas rolling default)
    thr = mu + sigma_k * sigma

    thr_long = thr.reset_index().melt(
        id_vars="trade_date", var_name="slot", value_name="thr"
    )
    # melt can hand back the slot column labels as object dtype; keep it int so the
    # merge below matches g's int ``slot`` (an object/int mismatch would silently miss).
    thr_long["slot"] = thr_long["slot"].astype(int)
    work = g.merge(thr_long, on=["trade_date", "slot"], how="left")
    work = work.sort_values(["trade_date", "bar_end"], kind="mergesort").reset_index(
        drop=True
    )

    vol = work["volume"].to_numpy(dtype=float)
    thr_arr = work["thr"].to_numpy(dtype=float)
    # Classifiable iff the strictly-prior baseline is finite (>= baseline_min_obs obs);
    # eruptive iff strictly above μ + k*σ; mild is any other classifiable bar.
    classifiable = np.isfinite(thr_arr)
    eruptive = classifiable & (vol > thr_arr)
    mild = classifiable & ~eruptive
    work["classifiable"] = classifiable
    # VALLEY (量谷 in the report's peak/ridge/valley taxonomy) == MILD: a classifiable,
    # non-eruptive minute. Exposed as a first-class boolean so the valley-PRICE family
    # (PR-I) consumes THE SAME classification instead of re-deriving it and drifting.
    # Purely additive — ``classifiable`` and ``peak`` below are computed exactly as
    # before and no consumer of this frame reads columns positionally.
    work["valley"] = mild
    # Carry mild as 0/1 so the within-day neighbour shift stays numeric (a shifted bool
    # column would go through an object-dtype fillna and warn).
    work["mild_i"] = mild.astype(np.int8)

    by_day = work.groupby("trade_date", sort=False)
    prev_end = by_day["bar_end"].shift(1)
    next_end = by_day["bar_end"].shift(-1)
    gap_prev = (work["bar_end"] - prev_end).dt.total_seconds().to_numpy()
    gap_next = (next_end - work["bar_end"]).dt.total_seconds().to_numpy()
    prev_mild = by_day["mild_i"].shift(1).fillna(0).to_numpy() > 0
    next_mild = by_day["mild_i"].shift(-1).fillna(0).to_numpy() > 0
    # A peak is an eruptive bar whose BOTH 1-minute neighbours (exactly 60s away, so
    # same session) exist and are mild. Missing / lunch-break / cutoff neighbours fail
    # the 60s gap; eruptive neighbours (ridge) and unclassifiable ones fail the mild
    # test — all correctly block the peak.
    peak = (
        eruptive
        & (gap_prev == ONE_MINUTE_SECONDS)
        & prev_mild
        & (gap_next == ONE_MINUTE_SECONDS)
        & next_mild
    )
    work["peak"] = peak
    # RIDGE (量岭) == eruptive AND NOT an isolated peak. Exposed as a first-class boolean
    # so the ridge-PRICE family (PR-J) consumes THE SAME classification instead of
    # re-deriving it and drifting. Purely additive: ``classifiable`` / ``valley`` /
    # ``peak`` above are computed exactly as before and no consumer of this frame reads
    # columns positionally.
    #
    # PINNED, and wider than the literal "eruptive next to an eruptive": an eruptive bar
    # is a ridge whenever its isolation is NOT PROVABLE, which also covers the
    # session-boundary bars (a neighbour missing across the lunch break / the cutoff /
    # a gap) and the bars whose neighbour is unclassifiable. That is the exact complement
    # of PR-F's deliberately conservative peak rule, so the taxonomy stays a clean
    # partition (valley | peak | ridge == classifiable) with no bar silently dropped.
    work["ridge"] = eruptive & ~peak
    return work


def positive_trade_mask(volume: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """The valley/ridge family's POSITIVE-TRADE guard.

    A bar with non-finite or non-positive ``volume`` OR ``amount`` carries no
    price information (``amount/volume`` is meaningless or degenerate), so it is
    dropped from BOTH sums of a ``Σamount/Σvolume`` VWAP. Applied at the
    summation step only, never before classification — the same-slot baseline
    must stay bit-identical across the family.
    """
    return np.isfinite(volume) & (volume > 0.0) & np.isfinite(amount) & (amount > 0.0)


def rolling_valid_days(
    obj: pd.Series | pd.DataFrame, *, lookback_days: int, min_valid_days: int
):
    """Trailing window over VALID days only (the peak family's reduction form).

    ``obj`` is indexed by ``trade_date`` and carries ONLY the valid days
    (invalid days are ABSENT, not NaN, so they never occupy a window slot).
    Returns the pandas ``Rolling`` object over the ascending-sorted days with
    ``min_periods=min_valid_days`` — the caller applies its own reduction
    (``.mean()`` / ``.sum()``), keeping each factor's float operation order
    exactly as before the D2 move.
    """
    return obj.sort_index().rolling(lookback_days, min_periods=min_valid_days)


def pooled_trailing_reduce(
    day_channels: Sequence[Sequence[np.ndarray]],
    *,
    lookback_days: int,
    reducer: Callable[..., float],
) -> list[float]:
    """Pool-then-reduce over trailing days (the amplitude family's form).

    ``day_channels`` holds one sequence per channel; each sequence has one
    ``np.ndarray`` per trading day, all channels aligned on the same day axis.
    For every day ``j`` the trailing ``lookback_days`` days (including ``j``)
    of each channel are concatenated into ONE pooled array and passed to
    ``reducer(*pooled_channels) -> float`` — exactly the original per-factor
    loop, so the float operation order is unchanged.
    """
    if not day_channels:
        return []
    n_days = len(day_channels[0])
    values: list[float] = []
    for j in range(n_days):
        lo = max(0, j - lookback_days + 1)
        pooled = tuple(
            np.concatenate(list(channel[lo : j + 1])) for channel in day_channels
        )
        values.append(reducer(*pooled))
    return values


def symbol_frames(visible: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Split the visible bars into ONE FRAME PER SYMBOL, in sorted symbol order.

    The whole cross-symbol isolation guarantee lives here: every downstream step (the
    same-slot baseline, the classification, the per-day legs, the trailing window) runs
    on a single symbol's rows, so one symbol's bars can never reach another's factor
    value.
    """
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        yield str(sym), g.reset_index(drop=True)


__all__ = [
    "ONE_MINUTE_SECONDS",
    "VOLUME_PRV_BASELINE_DAYS",
    "VOLUME_PRV_BASELINE_MIN_OBS",
    "VOLUME_PRV_MIN_CLASSIFIABLE",
    "VOLUME_PRV_MIN_VALID_DAYS",
    "VOLUME_PRV_SIGMA_K",
    "add_bar_end_ns",
    "empty_factor_series",
    "guarded_amplitude",
    "peak_mask_for_symbol",
    "pooled_trailing_reduce",
    "positive_trade_mask",
    "prepare_visible_minute_bars",
    "rolling_valid_days",
    "symbol_frames",
    "visible_minute_frame",
]
