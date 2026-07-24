"""Minute "ideal amplitude" factor (PR-D): math + surface (D2).

Reproduces the Kaiyuan report §30 (市场微观结构系列 30) 分钟理想振幅因子 as a daily
PIT-safe column derived from 1min bars.

Definition (LOCKED — reproduced from the report; N/lambda/min-minutes are part of
the factor DEFINITION, not tuned knobs). For each symbol and each panel date ``d``:

  1. Take the symbol's most recent ``N`` (=10) trading days INCLUDING ``d`` (the
     symbol's own minute-trading days — mirrors the jump factor's trailing window).
  2. PIT truncation (standing authorization): keep only bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50).
     This reuses the I3 per-bar cutoff path, so EVERY day in the window is truncated
     to its own [session-open, 14:50] and post-14:50 / close data is never touched.
  3. Per-bar minute amplitude ``amp = high/low - 1``; a bar is dropped unless
     ``low > 0`` and ``high >= low``.
  4. Pool ALL surviving bars of the window into ONE set ("merged cut", not a
     per-day cut). If the pool has fewer than ``min_minutes`` (=1150 ≈ half of
     10 x ~230) valid minutes, the value is NaN (honest missing — no fabricated
     warm-up; coverage is disclosed by the runner).
  5. Rank the pooled minutes by RAW minute close (unadjusted — amplitude is a ratio
     so it needs no adjustment, and the report ranks on the raw minute price), with
     ``(close, bar_end)`` as a stable total order. With ``k = floor(lambda * n)``
     (lambda=0.25):
        V_high = mean amp of the ``k`` HIGHEST-close minutes
        V_low  = mean amp of the ``k`` LOWEST-close minutes
  6. factor(d, s) = ``V_high - V_low`` (column ``minute_ideal_amp_{N}``).

Pre-registered sign = -1 (report full-market IC -0.059 / ICIR -3.1 / rankIC -0.076).
The value at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees a
future bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DEFAULT_DECISION_TIME,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)
from factors.base import Factor
from factors.compute.minute.primitives import (
    add_bar_end_ns,
    empty_factor_series,
    guarded_amplitude,
    pooled_trailing_reduce,
    visible_minute_frame,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (report terminal parameters; NOT tuned knobs).
IDEAL_AMP_LOOKBACK_DAYS = 10  # trailing trading-day window (N), includes date d
IDEAL_AMP_LAMBDA = 0.25  # top/bottom fraction by close (lambda) that forms V_high/V_low
IDEAL_AMP_MIN_MINUTES = 1150  # minimum valid pooled minutes for a finite value


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def _rank_cut(
    closes: np.ndarray, amps: np.ndarray, bar_ends: np.ndarray, lam: float, min_minutes: int
) -> float:
    """V_high - V_low over one pooled window; NaN if too few minutes or k < 1.

    ``closes``/``amps``/``bar_ends`` are equal-length arrays for ONE pooled window.
    Ranking is a stable total order ``(close, bar_end)`` (lexsort with close as the
    primary key), so the top-k / bottom-k selection is fully deterministic even when
    two minutes share a close.
    """
    n = closes.size
    if n < min_minutes:
        return float("nan")
    k = int(np.floor(lam * n))
    if k < 1:
        return float("nan")
    order = np.lexsort((bar_ends, closes))  # close primary, bar_end tie-break
    a = amps[order]
    return float(a[-k:].mean() - a[:k].mean())


def _amplitude_for_symbol(
    g: pd.DataFrame, lookback_days: int, lam: float, min_minutes: int
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily factor values for ONE symbol from its PIT-filtered, guarded bars.

    ``g`` holds columns ``trade_date`` / ``close`` / ``amp`` / ``bar_end_ns`` for a
    single symbol. Per-day arrays are built once, then each date pools the trailing
    ``lookback_days`` days (including that date) via the shared
    :func:`~factors.compute.minute.primitives.pooled_trailing_reduce` — no
    cross-symbol leakage because ``g`` is a single symbol's slice.
    """
    days: list[pd.Timestamp] = []
    day_close: list[np.ndarray] = []
    day_amp: list[np.ndarray] = []
    day_be: list[np.ndarray] = []
    for day, sub in g.groupby("trade_date", sort=True):
        days.append(pd.Timestamp(day).normalize())
        day_close.append(sub["close"].to_numpy(dtype=float))
        day_amp.append(sub["amp"].to_numpy(dtype=float))
        day_be.append(sub["bar_end_ns"].to_numpy(dtype="int64"))

    values = pooled_trailing_reduce(
        (day_close, day_amp, day_be),
        lookback_days=lookback_days,
        reducer=lambda closes, amps, bes: _rank_cut(closes, amps, bes, lam, min_minutes),
    )
    return days, values


def compute_minute_ideal_amplitude(
    bars: pd.DataFrame,
    *,
    lookback_days: int = IDEAL_AMP_LOOKBACK_DAYS,
    lam: float = IDEAL_AMP_LAMBDA,
    min_minutes: int = IDEAL_AMP_MIN_MINUTES,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "minute_ideal_amp",
) -> pd.Series:
    """PIT-safe daily "minute ideal amplitude" factor from 1min ``bars``.

    See the module docstring for the LOCKED definition. The heavy per-symbol loop is
    memory-bounded (the runner feeds one symbol at a time), but this function also
    accepts a multi-symbol frame and keeps symbols strictly isolated.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``.
        lookback_days: trailing trading-day window length (part of the definition).
        lam: top/bottom close fraction defining V_high/V_low (0 < lam <= 0.5).
        min_minutes: minimum valid pooled minutes for a finite value.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if not (0.0 < lam <= 0.5):
        raise ValueError(f"lam must be in (0, 0.5]; got {lam!r}.")
    if min_minutes < 2:
        # Need at least 2 minutes so a non-empty top/bottom cut can exist.
        raise ValueError(f"min_minutes must be >= 2; got {min_minutes!r}.")
    if len(bars) == 0:
        return empty_factor_series(name)

    visible = visible_minute_frame(
        bars, columns=("high", "low", "close"), decision_time=decision_time
    )
    if visible.empty:
        return empty_factor_series(name)

    visible = guarded_amplitude(visible)
    if visible.empty:
        return empty_factor_series(name)

    # int64 nanoseconds for a deterministic lexsort tie-break.
    visible = add_bar_end_ns(visible)
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _amplitude_for_symbol(g, lookback_days, lam, min_minutes)
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class MinuteIdealAmplitudeFactor(Factor):
    """Minute ideal-amplitude factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_minute_ideal_amplitude`); it does NO minute work of its
    own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the column
            so a non-default window cannot silently mislabel it.
    """

    name: str = f"minute_ideal_amp_{IDEAL_AMP_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = IDEAL_AMP_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"minute-ideal-amplitude lookback_days must be a positive integer; "
                f"got {lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"minute_ideal_amp_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's RankIC mean is -7.6% (full market, N=10,
        lambda=25%) — a HIGH minute ideal amplitude predicts LOWER forward returns.
        The sign is fixed BEFORE the run (a validated prototype must reproduce it).
        is_intraday=False by the module docstring's reasoning: minute INPUT but a
        DAILY signal traded close-to-close. min_history_bars=0: the warm-up is
        DATA-dependent (a value appears once >= ``IDEAL_AMP_MIN_MINUTES`` valid
        pooled minutes accumulate in the trailing window), not a fixed leading
        count — the honest NaN rate is reported by data_coverage.

        D1 declarations (D0 pre-assignment table row 2 + note 1, the table's
        flagged judgment call): adjustment=returns_invariant — the amplitude
        itself is the same-day ratio high/low - 1 (anchor cancels; module
        docstring step 3). overnight_boundary=CROSSED_DISCLOSED — the pooled
        ranking key is the RAW minute close across the trailing multi-day
        window, so when the window contains an ex-date the pooled ordering
        interleaves bars on two different price bases; the definition is
        pinned to the report (Wind ranks on the raw price) and deliberately
        kept. OPEN OBLIGATION, tracked here per D0 note 1: of the three
        CROSSED_DISCLOSED requirements (crossing is real / values kept by
        definition / deviation MEASURED and disclosed) the third — a
        PR-L-style ex-date deviation measurement (share of pooling windows
        containing a true ex-date + realized rank-perturbation magnitude) —
        is still MISSING and is owed in D2 alongside that stage's
        overnight-boundary property tests.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Minute ideal amplitude (Kaiyuan report §30): pool the 1min bars of "
                f"the trailing {self._lookback_days} trading days (PIT-truncated at "
                f"14:50 per bar), rank the pooled minutes by RAW close, and return "
                f"V_high - V_low where V_high/V_low are the mean per-minute amplitude "
                f"(high/low - 1) of the top / bottom floor({IDEAL_AMP_LAMBDA:g}*n) "
                f"minutes by close. Derived from 1min bars but a DAILY signal traded "
                f"close-to-close; >= {IDEAL_AMP_MIN_MINUTES} valid pooled minutes "
                f"required else NaN."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared
            # for honest provenance disclosure (data_coverage lists them); the daily
            # panel surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "close"),
            requires=_minute_requires("high", "low", "close"),
            adjustment="returns_invariant",
            overnight_boundary="crossed_disclosed",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily minute-ideal-amplitude column off ``panel``.

        The runner runs ``compute_minute_ideal_amplitude`` on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so
        this factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"MinuteIdealAmplitudeFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_minute_ideal_amplitude and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "IDEAL_AMP_LAMBDA",
    "IDEAL_AMP_LOOKBACK_DAYS",
    "IDEAL_AMP_MIN_MINUTES",
    "MinuteIdealAmplitudeFactor",
    "compute_minute_ideal_amplitude",
]
