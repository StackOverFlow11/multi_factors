"""Intraday amplitude-cut factor (PR-G): math + surface (D2).

Reproduces the SECOND factor of the Kaiyuan market-microstructure series #30 (开源证券
《高频振幅因子的内部切割——市场微观结构系列（30）》, 2025-08-09, reportId 4988549) — the
"日内振幅切割因子" (§3, Table 3, thought (2)) — as a daily PIT-safe column derived from
the 1min cache.

DISTINCTION FROM PR-D (must be stated on the factor spec): PR-D (``minute_ideal_amp``,
the report's FLAGSHIP, thought (1)) POOLS the trailing 10 days' 1min bars into ONE set
and cuts by minute CLOSE PRICE. THIS factor (thought (2)) cuts EACH DAY independently by
the 1-MINUTE RETURN, produces a daily ``V_day`` series, then takes its trailing-10-valid-
day mean / std and combines them cross-sectionally. The report finds the two are only
~30% correlated — they are mathematically distinct constructions, not variants.

Definition (report Table 3 four steps + §3 terminal parameters). For each symbol and each
panel date ``d`` (a DAILY signal, close-to-close, ``is_intraday=False``):

  1. PIT truncation (standing authorization): each day keeps only the 1min bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50), so
     every day (history AND signal day) is truncated to its own [session-open, 14:50].
  2. PER-DAY CUT (each trading day in the window, independent):
       - per-bar amplitude ``amp = high/low - 1`` (drop a bar unless ``low > 0`` and
         ``high >= low``);
       - the sentiment indicator is the 1-MINUTE RETURN ``r_t = close_t/close_{t-1} - 1``
         (the report's terminal pick), WITHIN-DAY lagged: each day's FIRST surviving bar
         has no ``r`` and is NOT in that day's cut (no overnight gap; PR-E precedent);
       - a bar is VALID iff both ``amp`` and ``r`` are finite; ``n_day`` = the valid-bar
         count; ``n_day < min_day_minutes`` (=100) -> the day is INVALID (no ``V_day``);
       - ``k = floor(lam * n_day)`` (lam=0.20, report terminal), ``k >= 1``; a stable
         total order ``(r, bar_end)`` makes the selection deterministic;
       - ``V_high`` = mean ``amp`` of the ``k`` HIGHEST-``r`` bars, ``V_low`` = mean
         ``amp`` of the ``k`` LOWEST-``r`` bars; the day's cut value is
         ``V_day = V_high - V_low``.
  3. TIME-SERIES aggregation: take the most recent ``lookback_days`` (=10) VALID trading
     days INCLUDING ``d``; fewer than ``min_valid_days`` (=6) valid days -> NaN (honest
     missing). ``V_mean = mean(V_day)`` and ``V_std = std(V_day, ddof=1)``.
  4. CROSS-SECTIONAL standardization (report step 4): for each panel date, z-score
     ``V_mean`` and ``V_std`` SEPARATELY over the covered-universe cross-section (ddof=1
     std denominator; a date whose finite-pair cross-section is smaller than
     ``min_cross_section`` (=10) -> all NaN that date), then the factor value is
     ``(z(V_mean) + z(V_std)) / 2`` (column ``intraday_amp_cut_{N}``).

Pre-registered sign = -1 (report: rankIC mean -0.067, rankICIR -3.82, quintile
long-short 16.7%/yr at N=10, lambda=20%, 1-minute-return indicator; the V_mean and V_std
sub-factors are each negative too). Low-volatility family reading: when high-return
minutes have a MUCH larger amplitude than low-return minutes, future returns are lower.

PINNED interpretation (disclosed, not a tuned knob): the report cross-standardizes on the
FULL market; our eval cell cross-standardizes on the CSI500 covered set. The factor value
at ``(d, s)`` uses only bars at dates <= d, so a value never sees a future bar
(invariant #1).

The heavy per-symbol work (steps 1-3, producing the ``(V_mean, V_std)`` panel) is split
from the cross-sectional combine (step 4) on purpose: the runner streams one symbol at a
time through :func:`compute_amp_cut_stats` (memory-bounded), assembles the full-universe
two-column panel, then calls :func:`combine_amp_cut_cross_section` ONCE — because step 4
needs every symbol's ``(V_mean, V_std)`` present before it can z-score a date's
cross-section. :func:`compute_intraday_amp_cut` chains the two for a single-call path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DATE_LEVEL,
    DEFAULT_DECISION_TIME,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)
from factors.base import Factor
from factors.compute.minute.primitives import (
    add_bar_end_ns,
    empty_factor_series,
    guarded_amplitude,
    visible_minute_frame,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (report terminal parameters; NOT tuned knobs).
AMP_CUT_LOOKBACK_DAYS = 10  # trailing VALID trading-day window (N), includes date d
AMP_CUT_LAMBDA = 0.20  # top/bottom fraction by 1-min return that forms V_high/V_low
AMP_CUT_MIN_DAY_MINUTES = 100  # a day is valid iff it has >= this many valid (amp & r) bars
AMP_CUT_MIN_VALID_DAYS = 6  # min valid days in the trailing window for a finite V_mean/V_std
AMP_CUT_MIN_CROSS_SECTION = 10  # min finite-pair cross-section for a finite z-score on a date

# Internal two-column stats panel labels (V_mean / V_std of the daily V_day series).
V_MEAN_COL = "v_mean"
V_STD_COL = "v_std"


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def _empty_stats() -> pd.DataFrame:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` ``(v_mean, v_std)`` panel."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.DataFrame(
        {V_MEAN_COL: pd.Series([], dtype=float), V_STD_COL: pd.Series([], dtype=float)},
        index=index,
    )


def _day_cut(
    amps: np.ndarray, rets: np.ndarray, bar_ends: np.ndarray, lam: float, min_day_minutes: int
) -> float:
    """``V_high - V_low`` for ONE day's valid bars; NaN if the day is invalid or k < 1.

    ``amps`` / ``rets`` / ``bar_ends`` are equal-length arrays of the VALID (amp & r
    finite) bars of ONE trading day. The bars are ranked by the 1-minute return ``r``
    with ``(r, bar_end)`` as a stable total order (lexsort with ``r`` primary), so the
    top-k / bottom-k selection is deterministic even when two bars share a return.
    """
    n = amps.size
    if n < min_day_minutes:
        return float("nan")
    k = int(np.floor(lam * n))
    if k < 1:
        return float("nan")
    order = np.lexsort((bar_ends, rets))  # r primary, bar_end tie-break
    a = amps[order]
    return float(a[-k:].mean() - a[:k].mean())  # V_high (top-r) - V_low (bottom-r)


def _amp_cut_stats_for_symbol(
    g: pd.DataFrame,
    lookback_days: int,
    lam: float,
    min_day_minutes: int,
    min_valid_days: int,
) -> tuple[list[pd.Timestamp], list[float], list[float]]:
    """Trailing V_mean / V_std for ONE symbol from its within-day-lagged bars.

    ``g`` holds columns ``trade_date`` / ``amp`` / ``ret`` (the within-day 1-minute
    return, NaN on each day's first surviving bar) / ``bar_end_ns`` for a single symbol.
    Per day the VALID bars are cut into ``V_day``; only VALID days (``V_day`` finite)
    enter the trailing series, so the rolling window spans the most recent
    ``lookback_days`` VALID days (report "取最近 10 个有效交易日"). Values are emitted only
    on valid days; a window with fewer than ``min_valid_days`` valid days -> NaN (both
    stats). No cross-symbol leakage — ``g`` is one symbol's slice — and no cross-day
    leakage — each day's first-bar return is already NaN and excluded from that day's cut.
    """
    days: list[pd.Timestamp] = []
    vdays: list[float] = []
    for day, sub in g.groupby("trade_date", sort=True):
        amp = sub["amp"].to_numpy(dtype=float)
        ret = sub["ret"].to_numpy(dtype=float)
        be = sub["bar_end_ns"].to_numpy(dtype="int64")
        valid = np.isfinite(amp) & np.isfinite(ret)
        vdays.append(_day_cut(amp[valid], ret[valid], be[valid], lam, min_day_minutes))
        days.append(pd.Timestamp(day).normalize())

    series = pd.Series(vdays, index=pd.DatetimeIndex(days)).sort_index()
    valid_series = series.dropna()  # only VALID days carry a V_day
    if valid_series.empty:
        return [], [], []
    roll = valid_series.rolling(lookback_days, min_periods=min_valid_days)
    vmean = roll.mean()
    vstd = roll.std()  # ddof=1 (pandas rolling default)
    out_days = [pd.Timestamp(d).normalize() for d in vmean.index]
    return out_days, list(vmean.to_numpy(dtype=float)), list(vstd.to_numpy(dtype=float))


def compute_amp_cut_stats(
    bars: pd.DataFrame,
    *,
    lookback_days: int = AMP_CUT_LOOKBACK_DAYS,
    lam: float = AMP_CUT_LAMBDA,
    min_day_minutes: int = AMP_CUT_MIN_DAY_MINUTES,
    min_valid_days: int = AMP_CUT_MIN_VALID_DAYS,
    decision_time: str = DEFAULT_DECISION_TIME,
) -> pd.DataFrame:
    """Per-symbol trailing ``(V_mean, V_std)`` panel (steps 1-3; NO cross-section yet).

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``, forms
    the within-day 1-minute returns, cuts each valid day by return into ``V_day``, and
    returns the trailing-``lookback_days``-VALID-day mean / std of ``V_day``. The
    cross-sectional standardization (step 4) is deliberately NOT applied here — it needs
    the full-universe panel and is done by :func:`combine_amp_cut_cross_section`.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the per-day cut
            and trailing aggregation are strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window (definition).
        lam: top/bottom 1-minute-return fraction defining V_high/V_low (0 < lam <= 0.5).
        min_day_minutes: a day is valid iff it has >= this many valid (amp & r) bars.
        min_valid_days: minimum valid days in the trailing window for a finite value.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).

    Returns:
        ``MultiIndex(date, symbol)`` DataFrame with columns ``v_mean`` / ``v_std``
        (midnight-normalized dates), sorted. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if not (0.0 < lam <= 0.5):
        raise ValueError(f"lam must be in (0, 0.5]; got {lam!r}.")
    if min_day_minutes < 2:
        # Need at least 2 valid minutes so a non-empty top/bottom cut can exist.
        raise ValueError(f"min_day_minutes must be >= 2; got {min_day_minutes!r}.")
    if min_valid_days < 2:
        # Need >= 2 valid days so a ddof=1 std of the V_day series is defined.
        raise ValueError(f"min_valid_days must be >= 2; got {min_valid_days!r}.")
    if len(bars) == 0:
        return _empty_stats()

    visible = visible_minute_frame(
        bars, columns=("high", "low", "close"), decision_time=decision_time
    )
    if visible.empty:
        return _empty_stats()

    visible = guarded_amplitude(visible)
    if visible.empty:
        return _empty_stats()

    # int64 nanoseconds for a deterministic lexsort tie-break.
    visible = add_bar_end_ns(visible)
    # Sort so the within-day lag sees bars in chronological order within each
    # (symbol, day); mergesort keeps it stable.
    visible = visible.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")
    # r is WITHIN-DAY lagged: grouping by (symbol, trade_date) makes each day's FIRST
    # surviving bar NaN, so no return ever crosses the overnight gap.
    visible["ret"] = visible.groupby([SYMBOL_LEVEL, "trade_date"], sort=False)[
        "close"
    ].pct_change()

    index_tuples: list[tuple] = []
    vmean_vals: list[float] = []
    vstd_vals: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vmean, vstd = _amp_cut_stats_for_symbol(
            g, lookback_days, lam, min_day_minutes, min_valid_days
        )
        for day, m, s in zip(days, vmean, vstd):
            index_tuples.append((day, str(sym)))
            vmean_vals.append(m)
            vstd_vals.append(s)

    if not index_tuples:
        return _empty_stats()
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.DataFrame(
        {V_MEAN_COL: vmean_vals, V_STD_COL: vstd_vals}, index=index
    ).sort_index()


def combine_amp_cut_cross_section(
    stats: pd.DataFrame,
    *,
    min_cross_section: int = AMP_CUT_MIN_CROSS_SECTION,
    name: str = "intraday_amp_cut",
) -> pd.Series:
    """Cross-sectional z-score combine of the ``(V_mean, V_std)`` panel (step 4).

    For each panel date, the cross-section is the symbols with BOTH ``V_mean`` and
    ``V_std`` finite. If that count is below ``min_cross_section`` (=10), every symbol on
    that date is NaN (honest missing). Otherwise ``V_mean`` and ``V_std`` are each
    z-scored (ddof=1 denominator) across the cross-section and the factor value is
    ``(z(V_mean) + z(V_std)) / 2``. A degenerate date (a column's cross-sectional std is
    zero or non-finite) is likewise NaN. The pre-registered sign (-1) lives on the factor
    spec, not here — this returns the RAW combined score.

    Args:
        stats: ``MultiIndex(date, symbol)`` panel with ``v_mean`` / ``v_std`` columns,
            typically the full covered universe assembled by the runner.
        min_cross_section: minimum finite-pair cross-section for a finite z-score.
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the combined
        factor value, sorted, named ``name``. Pure: never mutates ``stats``.
    """
    if min_cross_section < 2:
        # Need >= 2 cross-section members so a ddof=1 std across the section is defined.
        raise ValueError(f"min_cross_section must be >= 2; got {min_cross_section!r}.")
    if stats is None or stats.empty:
        return empty_factor_series(name)
    if V_MEAN_COL not in stats.columns or V_STD_COL not in stats.columns:
        raise ValueError(
            f"stats must carry '{V_MEAN_COL}' and '{V_STD_COL}' columns; got "
            f"{list(stats.columns)}."
        )

    vm_all = stats[V_MEAN_COL].to_numpy(dtype=float)
    vs_all = stats[V_STD_COL].to_numpy(dtype=float)
    finite_pair = np.isfinite(vm_all) & np.isfinite(vs_all)
    valid = stats.loc[finite_pair, [V_MEAN_COL, V_STD_COL]]
    if valid.empty:
        return empty_factor_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for date, grp in valid.groupby(level=DATE_LEVEL, sort=True):
        syms = grp.index.get_level_values(SYMBOL_LEVEL)
        n = len(grp)
        if n < min_cross_section:
            vals = np.full(n, np.nan)
        else:
            vm = grp[V_MEAN_COL].to_numpy(dtype=float)
            vs = grp[V_STD_COL].to_numpy(dtype=float)
            sm = float(vm.std(ddof=1))
            ss = float(vs.std(ddof=1))
            if not (np.isfinite(sm) and np.isfinite(ss)) or sm == 0.0 or ss == 0.0:
                vals = np.full(n, np.nan)
            else:
                zm = (vm - vm.mean()) / sm
                zs = (vs - vs.mean()) / ss
                vals = (zm + zs) / 2.0
        day = pd.Timestamp(date).normalize()
        for sym, v in zip(syms, vals):
            index_tuples.append((day, str(sym)))
            values.append(float(v))

    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


def compute_intraday_amp_cut(
    bars: pd.DataFrame,
    *,
    lookback_days: int = AMP_CUT_LOOKBACK_DAYS,
    lam: float = AMP_CUT_LAMBDA,
    min_day_minutes: int = AMP_CUT_MIN_DAY_MINUTES,
    min_valid_days: int = AMP_CUT_MIN_VALID_DAYS,
    min_cross_section: int = AMP_CUT_MIN_CROSS_SECTION,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "intraday_amp_cut",
) -> pd.Series:
    """Single-call intraday amplitude-cut factor: stats (steps 1-3) then combine (step 4).

    Convenience that chains :func:`compute_amp_cut_stats` and
    :func:`combine_amp_cut_cross_section`. The runner does NOT use this (it needs the
    per-symbol stats loop for memory-boundedness) but tests and any all-in-memory caller
    can. The cross-section is whatever ``bars`` carry — see the module docstring's PINNED
    interpretation (the runner scopes it to the CSI500 covered set).

    See the module docstring for the LOCKED definition. Pure: never mutates ``bars``.
    """
    stats = compute_amp_cut_stats(
        bars,
        lookback_days=lookback_days,
        lam=lam,
        min_day_minutes=min_day_minutes,
        min_valid_days=min_valid_days,
        decision_time=decision_time,
    )
    return combine_amp_cut_cross_section(
        stats, min_cross_section=min_cross_section, name=name
    )


class IntradayAmpCutFactor(Factor):
    """Intraday amplitude-cut factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_amp_cut_stats` + :func:`combine_amp_cut_cross_section`);
    it does NO minute work of its own, mirroring its siblings and the value / financial
    factors that surface an enriched column.

    Args:
        lookback_days: trailing VALID trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the column so a
            non-default window cannot silently mislabel it.
    """

    name: str = f"intraday_amp_cut_{AMP_CUT_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = AMP_CUT_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"intraday-amp-cut lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"intraday_amp_cut_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's rankIC mean is -0.067 (rankICIR -3.82, quintile
        long-short 16.7%/yr at N=10, lambda=20%, 1-minute-return indicator) — a HIGH
        intraday amplitude-cut value predicts LOWER forward returns; the V_mean and V_std
        sub-factors are each negative too. The sign is fixed BEFORE the run (a validated
        prototype must reproduce it). is_intraday=False by the module docstring's
        reasoning: minute INPUT but a DAILY signal traded close-to-close. min_history_bars=
        0: the warm-up is DATA-dependent (a value appears once >= ``AMP_CUT_MIN_VALID_DAYS``
        valid days accumulate AND the cross-section has >= ``AMP_CUT_MIN_CROSS_SECTION``
        finite pairs), not a fixed leading count — the honest NaN rate is reported by
        data_coverage.

        The description spells out the DISTINCTION FROM PR-D (``minute_ideal_amp``): PR-D
        pools the 10-day minutes into ONE set and cuts by minute CLOSE PRICE; this factor
        cuts EACH DAY by the 1-MINUTE RETURN, then takes the trailing-10-valid-day mean /
        std of the daily cut and combines them cross-sectionally (the report finds the two
        only ~30% correlated).

        D1 declarations (D0 pre-assignment table row 5): adjustment=
        returns_invariant — amp is the same-day ratio high/low - 1 and the
        cut key r = close_t/close_{t-1} - 1 is a within-day ratio, so the
        anchor cancels in both. overnight_boundary=none — r is WITHIN-DAY
        lagged, "no overnight gap; PR-E precedent" (module docstring step 2).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Intraday amplitude cut (Kaiyuan microstructure series #30, SECOND "
                f"factor 日内振幅切割). DISTINCT FROM PR-D minute_ideal_amp, which pools the "
                f"trailing days' minutes into ONE set and cuts by minute CLOSE PRICE: "
                f"this factor cuts EACH DAY independently by the 1-MINUTE RETURN "
                f"r=close_t/close_{{t-1}}-1 (within-day lagged, first bar of each day has "
                f"no r), taking V_day = V_high - V_low where V_high/V_low are the mean "
                f"amp (high/low-1) of the top/bottom floor({AMP_CUT_LAMBDA:g}*n_day) bars "
                f"by return (day valid iff >= {AMP_CUT_MIN_DAY_MINUTES} valid bars). "
                f"Trailing {self._lookback_days} VALID days give V_mean / V_std (>= "
                f"{AMP_CUT_MIN_VALID_DAYS} valid days else NaN); per date they are each "
                f"cross-sectionally z-scored over the covered universe (>= "
                f"{AMP_CUT_MIN_CROSS_SECTION} finite pairs else NaN) and averaged: factor "
                f"= (z(V_mean) + z(V_std))/2. Derived from 1min bars but a DAILY signal "
                f"traded close-to-close (report finds ~30% corr with minute_ideal_amp)."
            ),
            expected_ic_sign=-1,
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
        """Select the pre-aggregated daily intraday-amp-cut column off ``panel``.

        The runner runs ``compute_amp_cut_stats`` per symbol on the minute cache upstream,
        assembles the full-universe ``(V_mean, V_std)`` panel, applies
        ``combine_amp_cut_cross_section``, and joins the result as ``self.name``; here we
        only surface it, so this factor does no temporal logic and cannot introduce
        lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"IntradayAmpCutFactor needs the pre-aggregated '{self.name}' column on "
                f"the panel (produced upstream by compute_amp_cut_stats + "
                f"combine_amp_cut_cross_section and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "AMP_CUT_LAMBDA",
    "AMP_CUT_LOOKBACK_DAYS",
    "AMP_CUT_MIN_CROSS_SECTION",
    "AMP_CUT_MIN_DAY_MINUTES",
    "AMP_CUT_MIN_VALID_DAYS",
    "IntradayAmpCutFactor",
    "V_MEAN_COL",
    "V_STD_COL",
    "combine_amp_cut_cross_section",
    "compute_amp_cut_stats",
    "compute_intraday_amp_cut",
]
