"""VALLEY-RELATIVE VWAP factor (PR-I).

Reproduces the THIRD factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§4): "参考聪明钱因子的构建方式，计算每日量谷的成交量加权价格，与当日总成交量加权价格做比，
并计算 20 日均值作为量谷相对加权价格因子".

Same MACHINE as PR-F / PR-H, different FAMILY. The volume classification is REUSED
verbatim from :mod:`data.clean.intraday_volume_prv` (``prepare_visible_minute_bars`` +
``peak_mask_for_symbol``) — same-slot strictly-prior μ+kσ eruptive/mild test, same
classifiable rule, same valid-day floor. A VALLEY (量谷) is exactly that module's
``valley`` column: a classifiable, NON-eruptive minute. Nothing about the taxonomy is
re-implemented here, so the three factors can never drift apart. Where PR-F counts peaks
and PR-H measures the TIMING of peaks, this module measures the PRICE LEVEL at the
valleys — a different statistic on a different part of the taxonomy, which is why it is
worth running after two null results on the peak-counting family.

PINNED choices (deliberate and DISCLOSED, not tuned knobs; reproduced on the factor spec
so a reader sees exactly what was assumed):

  1. VWAP VIA THE AGGREGATION IDENTITY. A bar's volume-weighted price is
     ``p = amount / volume``, so a set's volume-weighted price
     ``Σ(p_i·v_i) / Σv_i`` collapses EXACTLY to ``Σamount / Σvolume``. Both legs use
     that identity: valley VWAP = Σamount(valley bars)/Σvolume(valley bars), day VWAP
     = Σamount(all visible bars)/Σvolume(all visible bars). This is the day's REAL
     VWAP, strictly better than approximating each bar by its close.
  2. POSITIVE-TRADE GUARD. A bar with non-finite or non-positive ``volume`` OR
     ``amount`` carries no price information (``amount/volume`` is meaningless or
     degenerate), so it is dropped from BOTH sums. The guard runs at the summation
     step only — never before classification — because the same-slot μ/σ baseline is
     PR-F's and must stay bit-identical. A zero-volume minute is therefore still
     classified (and still a valley) but contributes nothing to either VWAP.
  3. RAW (UNADJUSTED) PRICES. The cached minute bars are unadjusted. Both legs are
     sums over the SAME trading day, and a split/dividend adjustment factor is constant
     within a day, so it cancels exactly in the ratio — no adjustment is needed (the
     same reasoning PR-D's amplitude and I5b's price-limit checks rely on).
  4. THE DAY VWAP COVERS THE PIT-VISIBLE WINDOW ONLY (09:31–14:50), not the full
     session. This is a NECESSARY DEVIATION from the report, which uses the whole day:
     the standing 14:50 decision cutoff truncates history days and the signal day
     identically, and reading the closing auction would be lookahead at our decision
     time. Disclosed, not silently equated to the report's construction.
  5. VALLEY-BAR COUNT IS TAKEN AFTER THE GUARD. The ``min_valley_bars`` floor counts
     the valley minutes that actually CONTRIBUTE to the numerator, so a day cannot
     qualify on the strength of valley bars that traded nothing.

Factor value: ``valley_relative_vwap_20`` = the MEAN of the daily ratio
``valley VWAP / day VWAP`` over the symbol's most recent ``lookback_days`` (=20) VALID
trading days INCLUDING ``d``. A day is VALID iff it clears three gates: at least
``min_classifiable`` (=100) classifiable bars (PR-F's gate, unchanged), at least
``min_valley_bars`` (=20) tradable valley bars, and strictly positive volume in BOTH
denominators. Fewer than ``min_valid_days`` (=10) valid days in the trailing window ->
NaN (honest missing; the runner discloses the coverage). Values are emitted only on
valid days.

Pre-registered sign = +1. The report's full-market RankIC is +8.69% / RankICIR 4.44
(long-short 25.35%/yr, IR 3.04, monthly win rate 79.7%) — the strongest factor in the
report; its CSI500 sub-domain long-short is 9.94% / IR 1.26, the closest comparable to
our eval cell. Semantics per the report: valley minutes are moments of subdued trading
sentiment where prices are unlikely to have over-reacted, so a HIGH relative valley
price means the calm, informed part of the day was bid up -> higher future return. NOTE
the report is a MONTHLY, market-cap + industry neutral full-market series on Wind data
while our eval cell is CSI500 daily with industry + size neutralization, so its numbers
are a LOOSE reference only (disclosed, never mislabeled, never written in as an expected
value).

The value at ``d`` uses only bars at dates <= d, so a factor value never sees a future
bar (invariant #1); it is a DAILY signal traded close-to-close from d+1. This module is
DATA-layer only: it does not fetch, does not touch factors / alpha / portfolio /
runtime, and never sees a token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import DAILY_INDEX_NAMES, DEFAULT_DECISION_TIME
from data.clean.intraday_schema import SYMBOL_LEVEL, validate_intraday_bars
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
    peak_mask_for_symbol,
    prepare_visible_minute_bars,
)

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The classification constants are IMPORTED from PR-F, never redefined.
VALLEY_VWAP_LOOKBACK_DAYS = 20  # trailing VALID trading-day mean window, includes d
VALLEY_VWAP_MIN_VALLEY_BARS = 20  # min TRADABLE valley bars for a valid day

# The extra 1min column this family needs on top of PR-F's (volume): the traded value,
# which turns the bar set into a volume-weighted price via Σamount/Σvolume.
_AMOUNT = "amount"


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def valley_vwap_ratio_by_day(
    work: pd.DataFrame,
    *,
    min_valley_bars: int = VALLEY_VWAP_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> pd.Series:
    """Daily ``valley VWAP / day VWAP`` ratio for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~data.clean.intraday_volume_prv.peak_mask_for_symbol`, which must have been
    built from bars prepared with ``extra_columns=("amount",)`` so the traded value is
    available alongside the ``valley`` / ``classifiable`` masks.

    Both VWAPs use the ``Σamount / Σvolume`` aggregation identity (PINNED §1 of the
    module docstring). The POSITIVE-TRADE GUARD (§2) drops non-finite / non-positive
    volume or amount bars from both sums; the day leg spans ALL visible bars that clear
    the guard (eruptive minutes included — it is the WHOLE visible day's VWAP), while
    the valley leg spans only the guarded valley bars.

    Returns:
        Series indexed by ``trade_date`` (ascending) holding the ratio for the days that
        clear all three validity gates — invalid days are ABSENT, not NaN, so they do
        not occupy a slot in the caller's trailing window (the same rule PR-F/PR-H use).
    """
    vol = work["volume"].to_numpy(dtype=float)
    amt = work[_AMOUNT].to_numpy(dtype=float)
    # Positive-trade guard: no price information in a bar that traded nothing (or whose
    # fields are non-finite / negative). Applied HERE, at the summation step, never
    # before classification — PR-F's same-slot baseline must stay bit-identical.
    tradable = np.isfinite(vol) & (vol > 0.0) & np.isfinite(amt) & (amt > 0.0)
    valley = work["valley"].to_numpy(dtype=bool) & tradable

    per_bar = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            "day_amt": np.where(tradable, amt, 0.0),
            "day_vol": np.where(tradable, vol, 0.0),
            "valley_amt": np.where(valley, amt, 0.0),
            "valley_vol": np.where(valley, vol, 0.0),
            "valley_bars": valley.astype(np.int64),
            "classifiable_bars": work["classifiable"].to_numpy(dtype=bool).astype(
                np.int64
            ),
        }
    )
    agg = per_bar.groupby("trade_date", sort=True).sum()

    # Three validity gates (§ module docstring / task card §1.4): PR-F's classifiable
    # floor (unchanged), enough TRADABLE valley bars, and a positive denominator on both
    # legs (a 0/0 ratio is never fabricated).
    valid = (
        (agg["classifiable_bars"] >= min_classifiable)
        & (agg["valley_bars"] >= min_valley_bars)
        & (agg["day_vol"] > 0.0)
        & (agg["valley_vol"] > 0.0)
    )
    ok = agg.loc[valid]
    if ok.empty:
        return pd.Series(
            [], index=pd.DatetimeIndex([], name="trade_date"), dtype=float
        )
    valley_vwap = ok["valley_amt"] / ok["valley_vol"]
    day_vwap = ok["day_amt"] / ok["day_vol"]
    return (valley_vwap / day_vwap).astype(float)


def _valley_ratio_mean_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
    min_valley_bars: int,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily valley-relative-VWAP values for ONE symbol from its PIT-visible bars.

    Classifies the minutes with the REUSED :func:`peak_mask_for_symbol`, reduces each
    valid day to its VWAP ratio, then takes the trailing-``lookback_days``-valid-day
    mean. No cross-symbol leakage (``g`` is one symbol's slice) and no lookahead (the
    baseline is strictly prior, the mean window is trailing).
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )
    ratio = valley_vwap_ratio_by_day(
        work, min_valley_bars=min_valley_bars, min_classifiable=min_classifiable
    )
    if ratio.empty:
        return [], []

    # Mean over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Emitted only on valid days.
    rolled = ratio.sort_index().rolling(lookback_days, min_periods=min_valid_days).mean()
    days = [pd.Timestamp(d).normalize() for d in rolled.index]
    return days, list(rolled.to_numpy(dtype=float))


def compute_valley_relative_vwap(
    bars: pd.DataFrame,
    *,
    lookback_days: int = VALLEY_VWAP_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_valley_bars: int = VALLEY_VWAP_MIN_VALLEY_BARS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "valley_relative_vwap",
) -> pd.Series:
    """PIT-safe daily "valley-relative VWAP" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    classifies every visible minute with the REUSED PR-F taxonomy, computes each valid
    day's ``valley VWAP / whole-visible-day VWAP`` ratio via the ``Σamount/Σvolume``
    identity, and returns the trailing-``lookback_days``-VALID-day mean of that ratio.
    See the module docstring for the LOCKED definition and the five pinned choices.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping is
            strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window averaged (definition).
        baseline_days: strictly-prior same-slot baseline window in trading days (PR-F).
        baseline_min_obs: minimum same-slot observations for a classifiable bar (PR-F).
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ`` (PR-F).
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day needs at least this many classifiable bars (PR-F).
        min_valley_bars: a day needs at least this many TRADABLE valley bars.
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
    if min_valley_bars < 1:
        raise ValueError(f"min_valley_bars must be >= 1; got {min_valley_bars!r}.")
    if len(bars) == 0:
        return _empty_series(name)

    # extra_columns=("amount",) is the ONLY difference from the PR-F / PR-H entry
    # points: the traded value rides along on the surviving rows, and the truncation /
    # volume guard / slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_AMOUNT,)
    )
    if visible.empty:
        return _empty_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _valley_ratio_mean_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_valley_bars=min_valley_bars,
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "VALLEY_VWAP_LOOKBACK_DAYS",
    "VALLEY_VWAP_MIN_VALLEY_BARS",
    "compute_valley_relative_vwap",
    "valley_vwap_ratio_by_day",
]
