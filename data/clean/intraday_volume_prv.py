"""Volume-peak-count factor (PR-F).

Reproduces the Kaiyuan market-microstructure series #27 (开源证券《高频成交量的峰、岭、
谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417) flagship
"volume-peak-minute-count" factor (§2) as a daily PIT-safe column derived DIRECTLY
from the 1min cache (no coarser resampling — the peak/ridge/valley taxonomy lives at
the 1-minute grain). Kept in the DATA-clean layer (like
:func:`data.clean.intraday_amp_anomaly.compute_amp_marginal_anomaly_vol`) so the
``factors`` layer only SELECTS the pre-aggregated column and never fetches or sees a
forward return.

The report is under-specified about the PIT boundary; the interpretations below are
deliberate, DISCLOSED choices PINNED in the task card (task_card_pr_f_*.md §1), not
tuned knobs. They are reproduced on the factor spec so a reader can see exactly what
was assumed:

  1. PIT truncation (standing authorization): each day keeps only the 1min bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50) —
     history days AND the signal day are truncated identically, so the same-slot
     cross-day baseline is measured on a consistent window (≈ the morning session plus
     the afternoon up to the cutoff).
  2. Same-slot baseline (PINNED as STRICTLY PRIOR — the report does not say whether
     the current day is included, and strictly-prior is the more PIT-stable reading):
     for minute slot ``s`` and day ``t`` the baseline ``μ_s`` / ``σ_s`` (ddof=1) is
     the symbol's SAME-SLOT volume over the trailing ``baseline_days`` (=20) trading
     days STRICTLY BEFORE ``t``; fewer than ``baseline_min_obs`` (=10) same-slot
     observations in that window -> the ``(t, s)`` bar is NOT classifiable.
  3. classify: ``vol > μ_s + k*σ_s`` (k = 1) -> ERUPTIVE, else MILD (a "valley").
  4. a slot is a PEAK iff it is eruptive AND both its 1-minute neighbours in the SAME
     continuous session exist and are MILD. An eruptive bar whose neighbour is also
     eruptive is a RIDGE (not a peak); a session-boundary bar (each session's first /
     last visible bar, so a neighbour is missing across the lunch break or the cutoff)
     is likewise NOT a peak (its isolation is unprovable — PINNED disclosure). A
     neighbour that is not classifiable (baseline too thin) also blocks the peak.
     "Same continuous session" and "the 1-minute neighbour" are enforced by requiring
     the adjacent bar to be EXACTLY 60s away (the lunch break 11:30->13:01 and any
     missing minute fail this, so they are correctly not neighbours).

Factor value: ``volume_peak_count_20`` = the total count of peak minutes over the
symbol's most recent ``lookback_days`` (=20) VALID trading days INCLUDING ``d`` (a
simple count — after truncation the per-day slot count is uniform, so counts are
cross-sectionally comparable without normalization). A day is VALID iff it has at
least ``min_classifiable`` (=100) classifiable bars; fewer than ``min_valid_days``
(=10) valid days in the trailing window -> NaN (honest missing; the runner discloses
the coverage). Values are emitted only on valid days.

Pre-registered sign = +1 (more volume peaks = more informed-trading participation =
higher future returns; the report's full-market RankIC is +10.62% / RankICIR 4.36 and
its CSI500 sub-domain long-short is +14.96%/yr — the CSI500 line is the direct anchor
for our eval cell). NOTE the report is a monthly, market-cap + industry neutral series
on Wind data; our eval cell is CSI500 daily with industry + size neutral, so the
report numbers are a LOOSE reference only (disclosed, never mislabeled). Raw minute
volume (cached as-is) has magnitude jumps across split days that pollute the 20-day σ;
the report (Wind) does not adjust for this either, so we disclose it and do NOT correct
it. The value at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees
a future bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.

This module is DATA-layer only: it does not fetch, does not touch factors / alpha /
portfolio / runtime, and never sees a token.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import DAILY_INDEX_NAMES, DEFAULT_DECISION_TIME
from data.clean.intraday_schema import SYMBOL_LEVEL, validate_intraday_bars

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
VOLUME_PRV_LOOKBACK_DAYS = 20  # trailing VALID trading-day count window (N), includes d
VOLUME_PRV_BASELINE_DAYS = 20  # strictly-prior same-slot baseline window (trading days)
VOLUME_PRV_BASELINE_MIN_OBS = 10  # min same-slot obs for a classifiable bar
VOLUME_PRV_SIGMA_K = 1.0  # eruptive threshold multiplier k in vol > μ + k*σ
VOLUME_PRV_MIN_VALID_DAYS = 10  # min valid days in the trailing window for a finite value
VOLUME_PRV_MIN_CLASSIFIABLE = 100  # a day is valid iff it has >= this many classifiable bars

# The "strictly-next minute" test in seconds: two bars are same-session 1-minute
# neighbours iff their bar_end gap is EXACTLY 60s. The lunch break (11:30 -> 13:01),
# the session close and any missing minute all differ, so they are not neighbours.
_ONE_MINUTE_SECONDS = 60.0

# The columns every peak-family factor needs. ``prepare_visible_minute_bars`` always
# emits EXACTLY these (plus the derived ``trade_date`` / ``slot``); anything else a
# caller needs is opt-in via ``extra_columns``, so the default output stays byte-identical
# for the callers that predate the option.
_BASE_VISIBLE_COLUMNS = [SYMBOL_LEVEL, "bar_end", "available_time", "volume"]


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


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
    (:func:`~data.clean.intraday_schema.validate_intraday_bars`) — this helper is the
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
        ``valley`` (classifiable and NOT eruptive — the report's 量谷, what PR-I prices)
        and ``peak``. ``valley`` and ``peak`` are disjoint: a peak is eruptive by
        construction. Pure: never mutates ``g``.
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
        & (gap_prev == _ONE_MINUTE_SECONDS)
        & prev_mild
        & (gap_next == _ONE_MINUTE_SECONDS)
        & next_mild
    )
    work["peak"] = peak
    return work


def _peak_count_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily volume-peak-count values for ONE symbol from its PIT-visible bars.

    Identifies the peaks with the shared :func:`peak_mask_for_symbol` and reduces them to
    the trailing-``lookback_days``-VALID-day count.
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )

    peak_count = work.groupby("trade_date")["peak"].sum()
    classifiable_count = work.groupby("trade_date")["classifiable"].sum()
    valid_days = classifiable_count.index[classifiable_count >= min_classifiable]
    if len(valid_days) == 0:
        return [], []

    # Count over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Values are emitted only on valid days.
    pc_valid = peak_count.loc[valid_days].astype(float).sort_index()
    factor_valid = pc_valid.rolling(lookback_days, min_periods=min_valid_days).sum()
    days = [pd.Timestamp(d).normalize() for d in factor_valid.index]
    return days, list(factor_valid.to_numpy(dtype=float))


def compute_volume_peak_count(
    bars: pd.DataFrame,
    *,
    lookback_days: int = VOLUME_PRV_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "volume_peak_count",
) -> pd.Series:
    """PIT-safe daily "volume-peak-minute-count" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    classifies every visible minute against its SAME-SLOT strictly-prior baseline
    (eruptive vs mild), marks the eruptive minutes whose both 1-minute same-session
    neighbours are mild as PEAKS, and returns the trailing-``lookback_days``-VALID-day
    peak count. See the module docstring for the LOCKED definition.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping
            is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window for the count (definition).
        baseline_days: strictly-prior same-slot baseline window in trading days.
        baseline_min_obs: minimum same-slot observations for a classifiable bar.
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ``.
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day is VALID iff it has at least this many classifiable bars.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if baseline_days < 2:
        # Need >= 2 baseline observations so a ddof=1 std of the same-slot volume is
        # defined.
        raise ValueError(f"baseline_days must be >= 2; got {baseline_days!r}.")
    if baseline_min_obs < 2:
        raise ValueError(f"baseline_min_obs must be >= 2; got {baseline_min_obs!r}.")
    if sigma_k < 0.0:
        raise ValueError(f"sigma_k must be >= 0; got {sigma_k!r}.")
    if min_valid_days < 1:
        raise ValueError(f"min_valid_days must be >= 1; got {min_valid_days!r}.")
    if min_classifiable < 1:
        raise ValueError(f"min_classifiable must be >= 1; got {min_classifiable!r}.")
    if len(bars) == 0:
        return _empty_series(name)

    visible = prepare_visible_minute_bars(bars, decision_time=decision_time)
    if visible.empty:
        return _empty_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _peak_count_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "VOLUME_PRV_BASELINE_DAYS",
    "VOLUME_PRV_BASELINE_MIN_OBS",
    "VOLUME_PRV_LOOKBACK_DAYS",
    "VOLUME_PRV_MIN_CLASSIFIABLE",
    "VOLUME_PRV_MIN_VALID_DAYS",
    "VOLUME_PRV_SIGMA_K",
    "compute_volume_peak_count",
    "peak_mask_for_symbol",
    "prepare_visible_minute_bars",
]
