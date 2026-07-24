"""Amplitude marginal-anomaly relative-volatility factor (PR-E): math + surface (D2).

Reproduces the Changjiang high-frequency-factor series #19 (长江证券《高频因子（十九）》,
2026-06-03, reportId 5462994) "振幅边际异常相对波动因子" as a daily PIT-safe column
derived from 5min bars (themselves DERIVED from the 1min cache).

The source report is UNDER-SPECIFIED; the five choices below are deliberate,
DISCLOSED interpretations pinned in the task card (task_card_pr_e_*.md §0), not tuned
knobs. They are reproduced verbatim on the factor spec so a reader can see exactly
what was assumed:

  1. bar frequency = 5min, DERIVED from the 1min cache via
     :func:`data.clean.intraday_aggregate.resample_intraday_bars` (a derived bar
     inherits ``available_time = max(source_1min.available_time)`` — PIT-faithful).
  2. lookback window N = 20 trading days (the symbol's own days that have bars,
     trailing and INCLUDING ``d``).
  3. anomaly threshold = ``1{|Δamp_t| > μ + σ}`` with μ / σ = mean / ddof=1 std of
     ALL pooled ``|Δamp|`` in the 20-day pool (k = 1).
  4. weighted volatility = the ddof=1 SAMPLE std of the RETURNS on the selected
     (weight-1) bars.
  5. bar return ``r_t = close_t/close_{t-1} - 1`` and ``Δamp_t = amp_t - amp_{t-1}``
     are BOTH WITHIN-DAY lagged — each day's FIRST bar has no return / no Δamp — so
     the overnight gap never contaminates a pair.

Definition (per symbol, per panel date ``d``):

  1. Take the symbol's most recent ``N`` (=20) trading days INCLUDING ``d`` of 5min
     bars. PIT truncation (standing authorization): keep only bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50),
     so every day is truncated to its own [session-open, 14:50].
  2. Per bar: ``amp = high/low - 1`` (drop a bar unless ``low > 0`` and
     ``high >= low``). WITHIN each (symbol, day), sorted by ``bar_end``, form
     ``Δamp_t = amp_t - amp_{t-1}`` and ``r_t = close_t/close_{t-1} - 1``; the first
     surviving bar of each day has neither (no cross-day lag).
  3. Pool ALL surviving ``(|Δamp|, r)`` pairs of the ``N``-day window into ONE set.
     If the pool has fewer than ``min_pool`` (=460 ≈ 46 x 20 / 2) valid pairs, the
     value is NaN (honest missing — coverage is disclosed by the runner).
  4. Pool ``μ = mean(|Δamp|)`` and ``σ = std(|Δamp|, ddof=1)``; select the bars with
     ``|Δamp_t| > μ + k*σ`` (k = 1). If fewer than ``min_selected`` (=20) bars are
     selected, the value is NaN (an anomaly std over too few bars is unreliable).
  5. factor(d, s) = ``std(r_t | selected, ddof=1)`` (column
     ``amp_marginal_anomaly_vol_{N}``).

Pre-registered sign = +1 (the report's IC is positive across its universes:
raw CSI800 +4.47% / full-market +4.92%; market-cap + industry neutral +4.11% /
+5.56%, ICIR 65.85% / 108.71%). NOTE the report's sample is CSI800 / full-market on
a MONTHLY series; our eval cell is CSI500 daily — a LOOSE reference only. The value
at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees a future
bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_aggregate import resample_intraday_bars
from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)
from factors.base import Factor
from factors.compute.minute.primitives import (
    empty_factor_series,
    guarded_amplitude,
    pooled_trailing_reduce,
    visible_minute_frame,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
AMP_ANOMALY_LOOKBACK_DAYS = 20  # trailing trading-day window (N), includes date d
AMP_ANOMALY_FREQ = "5min"  # bar frequency, DERIVED from the 1min cache
AMP_ANOMALY_SIGMA_K = 1.0  # anomaly threshold multiplier k in |Δamp| > μ + k*σ
AMP_ANOMALY_MIN_POOL = 460  # minimum valid pooled (|Δamp|, r) pairs for a finite value
AMP_ANOMALY_MIN_SELECTED = 20  # minimum selected (anomaly) bars for a finite std


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def _anomaly_vol_cut(
    dabs: np.ndarray,
    ret: np.ndarray,
    min_pool: int,
    min_selected: int,
    sigma_k: float,
) -> float:
    """Selected-bar return std over one pooled window; NaN if a gate fails.

    ``dabs`` / ``ret`` are equal-length arrays of the VALID ``(|Δamp|, r)`` pairs in
    ONE pooled window (already NaN-free). Fewer than ``min_pool`` pairs -> NaN; select
    the bars whose ``|Δamp|`` exceeds ``mean + sigma_k * std(ddof=1)`` of the pooled
    ``|Δamp|``; fewer than ``min_selected`` selected -> NaN (an unreliable std).
    """
    n = dabs.size
    if n < min_pool:
        return float("nan")
    mu = float(dabs.mean())
    sigma = float(dabs.std(ddof=1))
    threshold = mu + sigma_k * sigma
    mask = dabs > threshold
    n_sel = int(np.count_nonzero(mask))
    if n_sel < min_selected:
        return float("nan")
    return float(ret[mask].std(ddof=1))


def _anomaly_vol_for_symbol(
    g: pd.DataFrame,
    lookback_days: int,
    min_pool: int,
    min_selected: int,
    sigma_k: float,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily factor values for ONE symbol from its within-day-lagged bars.

    ``g`` holds columns ``trade_date`` / ``dabs`` (``|Δamp_t|``) / ``ret`` (``r_t``)
    for a single symbol; ``dabs`` / ``ret`` are NaN on each day's first surviving bar
    (no cross-day lag). Per day the VALID (finite) pairs are collected once, then each
    date pools the trailing ``lookback_days`` days (including that date) via the shared
    :func:`~factors.compute.minute.primitives.pooled_trailing_reduce` — no
    cross-symbol leakage because ``g`` is a single symbol's slice, and no cross-day
    leakage because each day's first-bar pair is already NaN and dropped here.
    """
    days: list[pd.Timestamp] = []
    day_dabs: list[np.ndarray] = []
    day_ret: list[np.ndarray] = []
    for day, sub in g.groupby("trade_date", sort=True):
        days.append(pd.Timestamp(day).normalize())
        d = sub["dabs"].to_numpy(dtype=float)
        r = sub["ret"].to_numpy(dtype=float)
        valid = np.isfinite(d) & np.isfinite(r)
        day_dabs.append(d[valid])
        day_ret.append(r[valid])

    values = pooled_trailing_reduce(
        (day_dabs, day_ret),
        lookback_days=lookback_days,
        reducer=lambda dabs, ret: _anomaly_vol_cut(
            dabs, ret, min_pool, min_selected, sigma_k
        ),
    )
    return days, values


def compute_amp_marginal_anomaly_vol(
    bars: pd.DataFrame,
    *,
    lookback_days: int = AMP_ANOMALY_LOOKBACK_DAYS,
    min_pool: int = AMP_ANOMALY_MIN_POOL,
    min_selected: int = AMP_ANOMALY_MIN_SELECTED,
    sigma_k: float = AMP_ANOMALY_SIGMA_K,
    decision_time: str = DEFAULT_DECISION_TIME,
    freq: str = AMP_ANOMALY_FREQ,
    name: str = "amp_marginal_anomaly_vol",
) -> pd.Series:
    """PIT-safe daily "amplitude marginal-anomaly relative-volatility" factor.

    Takes normalized 1min ``bars``, DERIVES ``freq`` (default 5min) bars from them via
    :func:`resample_intraday_bars` (so a derived bar is usable only once EVERY
    constituent 1min bar is available), PIT-truncates them at ``decision_time``, forms
    the within-day ``|Δamp|`` / ``r`` pairs, and returns the trailing-``lookback_days``
    anomaly-bar return std. See the module docstring for the LOCKED definition.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping
            is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing trading-day window length (part of the definition).
        min_pool: minimum valid pooled ``(|Δamp|, r)`` pairs for a finite value.
        min_selected: minimum selected (anomaly) bars for a finite return std.
        sigma_k: anomaly threshold multiplier ``k`` in ``|Δamp| > μ + k*σ``.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        freq: derived bar frequency (default 5min); part of the definition.
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if min_pool < 2:
        # Need >= 2 pooled points so a ddof=1 std of |Δamp| is defined.
        raise ValueError(f"min_pool must be >= 2; got {min_pool!r}.")
    if min_selected < 2:
        # Need >= 2 selected bars so a ddof=1 std of their returns is defined.
        raise ValueError(f"min_selected must be >= 2; got {min_selected!r}.")
    if sigma_k < 0.0:
        raise ValueError(f"sigma_k must be >= 0; got {sigma_k!r}.")
    if len(bars) == 0:
        return empty_factor_series(name)

    # DERIVE the coarse (5min) bars from 1min FIRST: available_time = max source, so a
    # coarse bar enters only once all its 1min constituents are available.
    coarse = resample_intraday_bars(bars, freq)
    if len(coarse) == 0:
        return empty_factor_series(name)

    visible = visible_minute_frame(
        coarse, columns=("high", "low", "close"), decision_time=decision_time
    )
    if visible.empty:
        return empty_factor_series(name)

    visible = guarded_amplitude(visible)
    if visible.empty:
        return empty_factor_series(name)

    # Sort so the within-day lag sees bars in chronological order within each
    # (symbol, day); mergesort keeps it stable.
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")
    # Δamp and r are WITHIN-DAY lagged: grouping by (symbol, trade_date) makes each
    # day's FIRST surviving bar NaN, so no pair ever crosses the overnight gap.
    by_session = visible.groupby([SYMBOL_LEVEL, "trade_date"], sort=False)
    visible["dabs"] = by_session["amp"].diff().abs()
    visible["ret"] = by_session["close"].pct_change()

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _anomaly_vol_for_symbol(
            g, lookback_days, min_pool, min_selected, sigma_k
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class AmpMarginalAnomalyVolFactor(Factor):
    """Amplitude marginal-anomaly relative-volatility factor (daily, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_amp_marginal_anomaly_vol`); it does NO minute work of
    its own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (a pinned interpretation of the report), not a tuned knob. It only names
            the column so a non-default window cannot silently mislabel it.
    """

    name: str = f"amp_marginal_anomaly_vol_{AMP_ANOMALY_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = AMP_ANOMALY_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"amp-marginal-anomaly-vol lookback_days must be a positive integer; "
                f"got {lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"amp_marginal_anomaly_vol_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's IC is POSITIVE across its universes (raw
        CSI800 +4.47% / full-market +4.92%; market-cap + industry neutral +4.11% /
        +5.56%) — a HIGH anomaly-bar relative volatility predicts HIGHER forward
        returns. The sign is fixed BEFORE the run (a validated prototype must
        reproduce it). NOTE the report's sample is CSI800 / full-market on a MONTHLY
        series while our eval cell is CSI500 daily, so the report numbers are a LOOSE
        reference only (disclosed, never mislabeled). is_intraday=False by the module
        docstring's reasoning: minute INPUT but a DAILY signal traded close-to-close.
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once
        >= ``AMP_ANOMALY_MIN_POOL`` valid pooled pairs accumulate in the trailing
        window), not a fixed leading count — the honest NaN rate is reported by
        data_coverage.

        The description spells out the FIVE pinned interpretations of the
        under-specified report (bar freq, lookback, threshold, weighted-vol operator,
        within-day lag) so a reader sees exactly what was assumed.

        D1 declarations (D0 pre-assignment table row 3): adjustment=
        returns_invariant — amp is the same-day ratio high/low - 1 and the
        selected-bar statistic is a std of minute RETURNS (ratios), so the
        anchor cancels throughout. overnight_boundary=none — Δamp and the
        bar-return are BOTH within-day lagged, "the overnight gap never
        contaminates a pair" (module docstring pinned choice 5).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Amplitude marginal-anomaly relative volatility (Changjiang HF-factor "
                f"series #19). PINNED interpretations of an under-specified report: "
                f"(1) {AMP_ANOMALY_FREQ} bars DERIVED from the 1min cache "
                f"(available_time = max source, PIT-faithful); (2) trailing "
                f"{self._lookback_days} trading days (PIT-truncated at 14:50 per bar); "
                f"(3) select bars with |Δamp| > μ + {AMP_ANOMALY_SIGMA_K:g}σ of the "
                f"pooled |Δamp|; (4) factor = ddof=1 std of the RETURNS on the selected "
                f"bars; (5) Δamp and bar-return are WITHIN-DAY lagged (each day's first "
                f"bar has neither). amp = high/low - 1. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close; >= {AMP_ANOMALY_MIN_POOL} valid "
                f"pooled pairs and >= {AMP_ANOMALY_MIN_SELECTED} selected bars required "
                f"else NaN."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "close"),
            requires=_minute_requires("high", "low", "close"),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily amp-marginal-anomaly-vol column off ``panel``.

        The runner runs ``compute_amp_marginal_anomaly_vol`` on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"AmpMarginalAnomalyVolFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_amp_marginal_anomaly_vol and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "AMP_ANOMALY_FREQ",
    "AMP_ANOMALY_LOOKBACK_DAYS",
    "AMP_ANOMALY_MIN_POOL",
    "AMP_ANOMALY_MIN_SELECTED",
    "AMP_ANOMALY_SIGMA_K",
    "AmpMarginalAnomalyVolFactor",
    "compute_amp_marginal_anomaly_vol",
]
