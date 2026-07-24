"""PEAK/RIDGE AMOUNT-RATIO factor (PR-M): math + surface (D2).

Reproduces the SEVENTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§7.2 「峰岭成交比因子多空年化收益 27.13%」): "本小节通过计算 20 日量峰总成交额与量岭总成交额，
二者做比作为峰岭成交比因子，衡量知情交易相对个人投资者交易的相对参与程度".

This factor carries NO price information at all — it is a pure TRADED-VALUE mix between
the two eruptive groups — so it is the one test that separates "only price information
survives in this taxonomy" from "the peak/ridge split itself carries alpha".

The classification is the SHARED taxonomy in :mod:`factors.compute.minute.primitives`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``) — same same-slot
strictly-prior μ+kσ eruptive test, same classifiable rule. Nothing about the taxonomy is
re-implemented here, so the family can never drift apart.

AGGREGATION — THE REPORT'S FORM, WHICH IS *NOT* THE MEAN OF DAILY RATIOS
------------------------------------------------------------------------
The report says "计算 20 日量峰总成交额与量岭总成交额，二者做比": sum the peak amount over
20 days, sum the ridge amount over 20 days, THEN divide. That is a RATIO OF SUMS. Contrast
§7.1 (PR-J), which explicitly says "计算 20 日价格比均值" — a MEAN OF RATIOS. The report
draws the distinction itself in adjacent sections, so it is a deliberate difference in the
source and is followed here.

The two forms are genuinely different, not a rounding detail: a ratio of sums is
AMOUNT-WEIGHTED (busy days dominate) while a mean of ratios weights every valid day
equally. The ratio of sums is also structurally far better behaved for THIS quantity —
``peak_amt / ridge_amt`` on a single day has a small, noisy denominator and a heavy right
tail, and a single day whose ridge amount nearly vanishes would dominate a 20-day mean of
ratios. Pooling both legs first is the natural estimator of a 20-day participation MIX,
which is exactly what the report says the factor measures.

PINNED choices (deliberate and DISCLOSED, not tuned knobs; reproduced on the factor spec):

  1. PEAK IS THE NUMERATOR, RIDGE THE DENOMINATOR — "峰岭成交比", peak first, and the
     report's stated semantics ("衡量知情交易相对个人投资者交易的相对参与程度", informed
     trading RELATIVE TO retail) fix the direction independently of the name's word order.
  2. THE RIDGE MASK IS ``eruptive & ~peak`` and the PEAK MASK is PR-F's isolated-eruption
     test — the identical masks PR-J used, so ``valley | peak | ridge == classifiable``
     stays an exact partition and no eruptive bar is silently counted on both sides. A
     VALLEY bar enters NEITHER leg: this factor reads only the two ERUPTIVE groups.
  3. POSITIVE-TRADE GUARD. A bar with non-finite or non-positive ``amount`` carries no
     traded value, so it is dropped from both legs and from both bar counts. The guard runs
     at the summation step only — never before classification — because the same-slot μ/σ
     baseline is the shared taxonomy's and must stay bit-identical. Unlike PR-I/PR-J this
     factor never divides by volume, so ``volume`` is NOT part of the guard: a bar with a
     real amount contributes its amount regardless of how its volume is recorded.
  4. RAW (UNADJUSTED) AMOUNTS. ``amount`` is traded VALUE in RMB, which no split or
     dividend adjustment factor rescales — the adjustment moves prices and share counts in
     compensating directions. Both legs are therefore free of the ex-date caveat PR-L had
     to disclose, and free even of PR-I/PR-J's weaker "cancels within the day" argument:
     there is nothing to cancel.
  5. BOTH LEGS COVER THE PIT-VISIBLE WINDOW ONLY (09:31–14:50), not the full session. A
     NECESSARY DEVIATION from the report, which uses the whole day: the standing 14:50
     decision cutoff truncates history days and the signal day identically, and reading the
     closing auction would be lookahead at our decision time.
  6. AN ASYMMETRIC BAR FLOOR — ``min_peak_bars`` (=5) against ``min_ridge_bars`` (=10).
     PEAKS ARE THE SCARCER LEG HERE, the reverse of PR-J's valley/ridge asymmetry: a peak
     must erupt AND be ISOLATED, and isolation is a strong condition on a liquid name whose
     eruptions cluster. Holding the peak leg to the ridge floor would discard sound days
     and bias the surviving sample toward names whose eruptions happen to arrive alone.
     The floor is therefore set LOW, deliberately, and both the realized peak-bar
     distribution and the resulting day-validity rate are REPORTED
     (``with_diagnostics=True``) rather than left implicit — together with the
     counterfactual valid-day count at a peak floor of 10.
  7. BOTH BAR COUNTS ARE TAKEN AFTER THE GUARD, so a day cannot qualify on the strength of
     bars that traded nothing.

Factor value: ``peak_ridge_amount_ratio_20`` = ``Σ peak_amt / Σ ridge_amt`` where both sums
run over the symbol's most recent ``lookback_days`` (=20) VALID trading days INCLUDING
``d``. A day is VALID iff it clears four gates: at least ``min_classifiable`` (=100)
classifiable bars (PR-F's gate, unchanged), at least ``min_peak_bars`` (=5) tradable peak
bars, at least ``min_ridge_bars`` (=10) tradable ridge bars, and strictly positive amount in
BOTH legs. Fewer than ``min_valid_days`` (=10) valid days in the trailing window -> NaN
(honest missing; the runner discloses the coverage). Values are emitted only on valid days.

Pre-registered sign = +1, READ FROM THE REPORT, not from data and not from semantics: §7.2
states "峰岭成交比因子 RankIC 均值 10.28%，RankICIR 4.07" — an explicitly POSITIVE RankIC —
with a long leg of 16.06%/yr, long-short 27.13%/yr, IR 2.89, max drawdown 7.88% and a 74.3%
monthly win rate (every year positive 2013–2025). NOTE the report is a MONTHLY, market-cap
+ industry neutral full-market series on Wind data while our eval cell is CSI500 daily with
industry + size neutralization, so its numbers are a LOOSE reference only.

The value at ``d`` uses only bars at dates <= d, so a factor value never sees a future bar
(invariant #1); it is a DAILY signal traded close-to-close from d+1.
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
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    empty_factor_series,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
    symbol_frames,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The classification constants live with the taxonomy in ``primitives``.
PEAK_RIDGE_LOOKBACK_DAYS = 20  # trailing VALID trading-day window for BOTH sums, includes d
# PINNED LOWER than the ridge floor (module docstring §6): a peak must erupt AND be
# ISOLATED, which makes peaks the structurally scarcer leg of this pair.
PEAK_RIDGE_MIN_PEAK_BARS = 5  # min TRADABLE peak bars for a valid day
PEAK_RIDGE_MIN_RIDGE_BARS = 10  # min TRADABLE ridge bars for a valid day

# The extra 1min column this factor needs on top of the taxonomy's (volume): the traded
# value. Unlike PR-I / PR-J this factor never divides by volume — amount is the whole
# quantity.
_AMOUNT = "amount"

# Per-day diagnostic columns (the peak-scarcity disclosure the task card requires).
DIAGNOSTIC_COLUMNS = ("classifiable_bars", "peak_bars", "ridge_bars", "valid")


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


# --------------------------------------------------------------------------- #
# The three load-bearing steps, each a NAMED function.
#
# They are small enough to inline, but each one carries a correctness property that the
# test-suite has to be able to BREAK on purpose: the positive-trade guard, the trailing
# (never forward-looking) window, and the per-symbol split. A test that asserts "perturbing
# X changes nothing" proves nothing unless the defective implementation can be substituted
# and shown to FAIL it, so each property gets its own substitutable seam. The per-symbol
# split is the shared ``primitives.symbol_frames``, aliased into THIS module's globals so
# the seam stays patchable here.
# --------------------------------------------------------------------------- #
def _tradable_amount(amt: np.ndarray) -> np.ndarray:
    """Positive-trade guard (module docstring §3): finite, strictly positive amount.

    ``volume`` is deliberately absent — this factor never divides by volume, so a bar with
    a real traded value contributes it regardless of how its volume is recorded.
    """
    return np.isfinite(amt) & (amt > 0.0)


def _trailing_ratio_of_sums(
    legs: pd.DataFrame, *, lookback_days: int, min_valid_days: int
) -> pd.Series:
    """The report's ratio of 20-day sums over a STRICTLY TRAILING valid-day window.

    Both legs are pooled with the SAME window and the SAME ``min_periods``, so numerator
    and denominator always cover exactly the same days. The window is trailing and the
    index is sorted ascending first, so a value at ``d`` can never absorb a later day.
    """
    ordered = legs.sort_index()
    roll = ordered.rolling(lookback_days, min_periods=min_valid_days).sum()
    return roll["peak_amt"] / roll["ridge_amt"]


# The per-symbol split seam (see the block comment above): the shared primitive bound to
# a module-local name so tests can substitute a defective split HERE.
_symbol_frames = symbol_frames


def peak_ridge_amount_by_day(
    work: pd.DataFrame,
    *,
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS,
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    with_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Daily peak / ridge traded-AMOUNT totals for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~factors.compute.minute.primitives.peak_mask_for_symbol`, which must have been
    built from bars prepared with ``extra_columns=("amount",)`` so the traded value is
    available alongside the ``peak`` / ``ridge`` / ``classifiable`` masks.

    Returns the two LEGS rather than their daily ratio, because the factor is the report's
    RATIO OF 20-DAY SUMS (module docstring), not the mean of daily ratios — the caller
    pools each leg over the trailing window first and divides once at the end.

    Args:
        work: one symbol's classified minute frame (see above).
        min_peak_bars: minimum TRADABLE peak bars for a valid day (PINNED lower than the
            ridge floor — see §6; peaks are the scarcer leg of this pair).
        min_ridge_bars: minimum TRADABLE ridge bars for a valid day.
        min_classifiable: minimum classifiable bars for a valid day (PR-F's gate).
        with_diagnostics: also return the per-day bar-count frame, so the caller can
            REPORT the peak-scarcity distribution instead of only gating on it.

    Returns:
        DataFrame indexed by ``trade_date`` (ascending) with ``peak_amt`` / ``ridge_amt``
        columns for the days that clear all gates — invalid days are ABSENT, not NaN, so
        they do not occupy a slot in the caller's trailing window (the same rule
        PR-F / PR-H / PR-I / PR-J use). With ``with_diagnostics=True``, a
        ``(legs, diagnostics)`` pair where ``diagnostics`` is indexed by EVERY day present
        in ``work`` and carries :data:`DIAGNOSTIC_COLUMNS`.
    """
    amt = work[_AMOUNT].to_numpy(dtype=float)
    # Positive-trade guard: a bar with no finite positive traded value contributes nothing.
    # Applied HERE, at the summation step, never before classification — the shared
    # same-slot baseline must stay bit-identical.
    tradable = _tradable_amount(amt)
    peak = work["peak"].to_numpy(dtype=bool) & tradable
    ridge = work["ridge"].to_numpy(dtype=bool) & tradable

    per_bar = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            "peak_amt": np.where(peak, amt, 0.0),
            "ridge_amt": np.where(ridge, amt, 0.0),
            "peak_bars": peak.astype(np.int64),
            "ridge_bars": ridge.astype(np.int64),
            "classifiable_bars": work["classifiable"]
            .to_numpy(dtype=bool)
            .astype(np.int64),
        }
    )
    agg = per_bar.groupby("trade_date", sort=True).sum()

    # Four validity gates (module docstring): PR-F's classifiable floor (unchanged),
    # enough TRADABLE peak bars, enough TRADABLE ridge bars, and strictly positive amount
    # on both legs (a day contributing 0 to the denominator pool is never fabricated).
    valid = (
        (agg["classifiable_bars"] >= min_classifiable)
        & (agg["peak_bars"] >= min_peak_bars)
        & (agg["ridge_bars"] >= min_ridge_bars)
        & (agg["peak_amt"] > 0.0)
        & (agg["ridge_amt"] > 0.0)
    )
    legs = agg.loc[valid, ["peak_amt", "ridge_amt"]].astype(float)

    if not with_diagnostics:
        return legs
    diagnostics = agg[["classifiable_bars", "peak_bars", "ridge_bars"]].copy()
    diagnostics["valid"] = valid
    return legs, diagnostics


def _peak_ridge_amount_ratio_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
    min_peak_bars: int,
    min_ridge_bars: int,
    collect_diagnostics: bool = False,
) -> tuple[list[pd.Timestamp], list[float], pd.DataFrame | None]:
    """Daily peak/ridge amount-ratio values for ONE symbol from its PIT-visible bars.

    Classifies the minutes with the SHARED :func:`peak_mask_for_symbol`, reduces each valid
    day to its two amount legs, then divides the trailing-``lookback_days``-valid-day SUM of
    the peak leg by that of the ridge leg (the report's ratio-of-sums form). No cross-symbol
    leakage (``g`` is one symbol's slice) and no lookahead (the baseline is strictly prior,
    both rolling windows are trailing).
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )
    result = peak_ridge_amount_by_day(
        work,
        min_peak_bars=min_peak_bars,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        with_diagnostics=collect_diagnostics,
    )
    if collect_diagnostics:
        legs, diagnostics = result
    else:
        legs, diagnostics = result, None
    if legs.empty:
        return [], [], diagnostics

    # RATIO OF SUMS over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated.
    ratio = _trailing_ratio_of_sums(
        legs, lookback_days=lookback_days, min_valid_days=min_valid_days
    )
    days = [pd.Timestamp(d).normalize() for d in ratio.index]
    return days, list(ratio.to_numpy(dtype=float)), diagnostics


def compute_peak_ridge_amount_ratio(
    bars: pd.DataFrame,
    *,
    lookback_days: int = PEAK_RIDGE_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS,
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "peak_ridge_amount_ratio",
    diagnostics_out: list | None = None,
) -> pd.Series:
    """PIT-safe daily "peak/ridge traded-amount ratio" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``, classifies
    every visible minute with the SHARED PR-F taxonomy, totals each valid day's peak and
    ridge traded amount, and returns the ratio of the trailing-``lookback_days``-VALID-day
    SUMS of the two legs. See the module docstring for the LOCKED definition, the report's
    ratio-of-sums wording, and the seven pinned choices.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping is
            strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window POOLED by both legs (definition).
        baseline_days: strictly-prior same-slot baseline window in trading days (PR-F).
        baseline_min_obs: minimum same-slot observations for a classifiable bar (PR-F).
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ`` (PR-F).
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day needs at least this many classifiable bars (PR-F).
        min_peak_bars: a day needs at least this many TRADABLE peak bars (PINNED lower than
            the ridge floor — see the module docstring §6).
        min_ridge_bars: a day needs at least this many TRADABLE ridge bars.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).
        diagnostics_out: optional list the per-symbol day-level bar-count frames are
            APPENDED to (each carries a ``symbol`` column), so a caller can report the
            peak-scarcity distribution. Purely observational — supplying it does not change
            the returned factor.

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily factor
        value, sorted, named ``name``. Pure: never mutates ``bars``.
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
    if min_peak_bars < 1:
        raise ValueError(f"min_peak_bars must be >= 1; got {min_peak_bars!r}.")
    if min_ridge_bars < 1:
        raise ValueError(f"min_ridge_bars must be >= 1; got {min_ridge_bars!r}.")
    if len(bars) == 0:
        return empty_factor_series(name)

    # extra_columns=("amount",) is the ONLY difference from the PR-F / PR-H entry points:
    # the traded value rides along on the surviving rows, and the truncation / volume guard
    # / slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_AMOUNT,)
    )
    if visible.empty:
        return empty_factor_series(name)

    collect = diagnostics_out is not None
    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in _symbol_frames(visible):
        days, vals, diagnostics = _peak_ridge_amount_ratio_for_symbol(
            g,
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_peak_bars=min_peak_bars,
            min_ridge_bars=min_ridge_bars,
            collect_diagnostics=collect,
        )
        if collect and diagnostics is not None and not diagnostics.empty:
            frame = diagnostics.copy()
            frame[SYMBOL_LEVEL] = sym
            diagnostics_out.append(frame)
        for day, val in zip(days, vals):
            index_tuples.append((day, sym))
            values.append(val)

    if not index_tuples:
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class PeakRidgeAmountRatioFactor(Factor):
    """Peak/ridge traded-amount ratio factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_peak_ridge_amount_ratio`); it does NO minute work of its
    own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing VALID trading-day window POOLED by both amount legs; part
            of the factor DEFINITION (reproduced from the report), not a tuned knob. It
            only names the column so a non-default window cannot silently mislabel it.
    """

    name: str = f"peak_ridge_amount_ratio_{PEAK_RIDGE_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = PEAK_RIDGE_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"peak-ridge-amount-ratio lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"peak_ridge_amount_ratio_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1 READ FROM THE REPORT, not inferred from data and not merely
        argued from semantics: §7.2 states "峰岭成交比因子 RankIC 均值 10.28%，RankICIR
        4.07" — an explicitly POSITIVE RankIC — with a long leg of 16.06%/yr, long-short
        27.13%/yr, IR 2.89, max drawdown 7.88%, 74.3% monthly win rate and every year
        positive 2013–2025. The report's §1 taxonomy summary agrees independently ("对于
        量峰时点，分钟数、成交额类因子更加有效，反映知情交易参与度，为正向因子", with the
        ridge leg's amount factors marked negative-alpha), so peak-over-ridge is positive
        on both counts. The sign is fixed BEFORE the run (a validated prototype must
        reproduce it). NOTE the report is a MONTHLY, market-cap + industry neutral
        full-market series on Wind data while our eval cell is CSI500 daily with industry
        + size neutral, so the report numbers are a LOOSE reference only (disclosed, never
        mislabeled, never written in as an expected value). is_intraday=False by the module
        docstring's reasoning: minute INPUT but a DAILY signal traded close-to-close.
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once enough
        VALID days accumulate), not a fixed leading count — the honest NaN rate is reported
        by data_coverage.

        The description spells out the AGGREGATION FORM, which is the one place this
        factor's definition departs from its siblings: the report specifies a RATIO OF
        20-DAY SUMS, not the mean of daily ratios that §7.1 (PR-J) specifies.

        D1 declarations (D0 pre-assignment table row 11): adjustment=none —
        "``amount`` is traded VALUE in RMB, which no split or dividend
        adjustment factor rescales ... there is nothing to cancel" (module
        docstring PINNED §4); volume only drives the shared classification,
        and no price is ever read (zero price content — one disclaimer covers
        both axes). overnight_boundary=none — no raw-price comparison exists.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Peak/ridge traded-amount ratio (Kaiyuan microstructure series #27, "
                f"SEVENTH factor 峰岭成交比, §7.2). SAME minute classification as PR-F "
                f"volume_peak_count / PR-H peak_interval_kurtosis / PR-I "
                f"valley_relative_vwap / PR-J valley_ridge_vwap_ratio (SHARED taxonomy "
                f"in factors.compute.minute.primitives, not re-implemented): 1min bars "
                f"PIT-truncated at 14:50, a minute is ERUPTIVE if vol > μ + "
                f"{VOLUME_PRV_SIGMA_K:g}σ of its SAME-SLOT strictly-prior "
                f"{VOLUME_PRV_BASELINE_DAYS}-day baseline; a PEAK is an ISOLATED eruption "
                f"and a RIDGE is 'eruptive AND NOT a peak'. This is the first factor of "
                f"the loop carrying NO price information: both legs are pure traded VALUE, "
                f"so it separates 'only price information survives in this taxonomy' from "
                f"'the peak/ridge split itself carries alpha'. AGGREGATION = RATIO OF "
                f"SUMS: Σ peak amount over the trailing {self._lookback_days} VALID days "
                f"divided by Σ ridge amount over the SAME days — the report's literal "
                f"wording ('计算 20 日量峰总成交额与量岭总成交额，二者做比'), which "
                f"DIFFERS from §7.1's mean-of-daily-ratios and is followed as written; it "
                f"is also the better-behaved estimator, since a single day with a nearly "
                f"vanishing ridge amount would dominate a mean of daily ratios. PINNED "
                f"choices: (1) PEAK is the numerator, RIDGE the denominator, fixed both by "
                f"the name 峰岭 and by the report's stated semantics (informed trading "
                f"RELATIVE TO retail participation); (2) the masks are PR-J's exactly, so "
                f"valley|peak|ridge stays an exact partition of the classifiable bars and "
                f"a VALLEY bar enters NEITHER leg — this factor reads only the two "
                f"ERUPTIVE groups; (3) bars with non-finite or non-positive amount are "
                f"dropped from both legs and both bar counts (guard applied at summation "
                f"only, so PR-F's baseline is untouched); volume is deliberately NOT in "
                f"the guard because this factor never divides by volume; (4) RAW "
                f"unadjusted amounts are exactly correct: traded VALUE in RMB is not "
                f"rescaled by any split/dividend adjustment factor, so unlike PR-L there "
                f"is no ex-date caveat and unlike PR-I/PR-J no within-day cancellation "
                f"argument is even needed; (5) DEVIATION FROM THE REPORT, disclosed: both "
                f"legs span the PIT-VISIBLE window 09:31-14:50 only, not the full session "
                f"— reading the close would be lookahead at our 14:50 decision time; (6) a "
                f"day is VALID iff it has >= {VOLUME_PRV_MIN_CLASSIFIABLE} classifiable "
                f"bars AND >= {PEAK_RIDGE_MIN_PEAK_BARS} TRADABLE peak bars AND >= "
                f"{PEAK_RIDGE_MIN_RIDGE_BARS} TRADABLE ridge bars (both counted AFTER the "
                f"guard) AND positive amount in both legs — the PEAK floor is deliberately "
                f"LOWER, the REVERSE of PR-J's asymmetry, because a peak must erupt AND be "
                f"ISOLATED, making peaks the scarcer leg of THIS pair; the realized "
                f"peak-bar distribution, the day-validity rate and the counterfactual "
                f"valid-day count at a peak floor of 10 are all REPORTED by the runner "
                f"rather than left implicit; NaN below {VOLUME_PRV_MIN_VALID_DAYS} valid "
                f"days. Derived from 1min bars but a DAILY signal traded close-to-close."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from: volume drives
            # the SHARED classification, amount is the factor's whole quantity. Declared
            # for honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("volume", "amount"),
            requires=_minute_requires("volume", "amount"),
            adjustment="none",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily peak/ridge amount-ratio column off ``panel``.

        The runner runs ``compute_peak_ridge_amount_ratio`` per symbol on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"PeakRidgeAmountRatioFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_peak_ridge_amount_ratio and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "PEAK_RIDGE_LOOKBACK_DAYS",
    "PEAK_RIDGE_MIN_PEAK_BARS",
    "PEAK_RIDGE_MIN_RIDGE_BARS",
    "PeakRidgeAmountRatioFactor",
    "compute_peak_ridge_amount_ratio",
    "peak_ridge_amount_by_day",
]
