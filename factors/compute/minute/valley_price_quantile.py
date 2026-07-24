"""VALLEY WEIGHTED-PRICE-QUANTILE factor (PR-L): math + surface (D2).

Reproduces the SIXTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§5): "将『日内最高价、日内最低价、昨日收盘价』三者的最高值与最低值作为区间的价格高低位，并计算
每日量谷成交量加权价格的相对分位点，将 20 日的分位点均值作为量谷加权价格分位点因子…考虑到因子
计算中正向暴露 20 日反转因子，我们进一步对该因子做反转中性化处理".

Same MACHINE as PR-F / PR-H / PR-I / PR-J / PR-K, a new STATISTIC FAMILY. The minute
classification is the SHARED taxonomy in :mod:`factors.compute.minute.primitives`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``) — same same-slot
strictly-prior μ+kσ eruptive test, same classifiable rule. A VALLEY (量谷) is exactly
that taxonomy's ``valley`` column. Where PR-F counted peaks, PR-H measured their timing,
PR-I / PR-J took price RATIOS and PR-K a RETURN, this module measures a price POSITION:
WHERE in the day's price range the valley VWAP sits.

Definition, per symbol and per panel date ``d`` (a DAILY signal traded close-to-close,
``is_intraday=False`` like all eight precedents):

  1. PIT truncation (standing authorization): each day keeps only the 1min bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50).
  2. PRICE RANGE from three prices — the visible day's high, the visible day's low, and
     the PREVIOUS trading day's close: ``hi = max(visible high, prev_close)`` and
     ``lo = min(visible low, prev_close)``.
  3. VALLEY VWAP = ``Σamount(valley bars) / Σvolume(valley bars)`` (PR-I's identity).
  4. DAILY QUANTILE ``q_day = (valley VWAP - lo) / (hi - lo)``, deliberately NOT clipped.
  5. TRAILING MEAN ``qbar`` over the most recent ``lookback_days`` (=20) VALID days
     including ``d``; fewer than ``min_valid_days`` (=10) valid days -> NaN.
  6. REVERSAL NEUTRALIZATION: per panel date, the cross-sectional OLS residual of
     ``qbar`` on a 20-day reversal factor.

PINNED choices (deliberate and DISCLOSED, reproduced on the factor spec):

  1. PREV_CLOSE IS THE LAST VISIBLE (<=14:50) RAW CLOSE OF THE PREVIOUS TRADING DAY, not
     that day's true 15:00 daily close. DEVIATION FROM THE REPORT, which uses the real
     previous close. The previous day's 15:00 close is NOT itself a lookahead at a 14:50
     decision on day d — it is already past — so this is not a leakage fix; it is a
     SINGLE-VISIBILITY-DEFINITION choice, so that every price entering the factor (the
     range, the VWAP, the previous close) comes from one source under one cutoff rather
     than mixing a minute-cache window with a daily-close feed. Disclosed, never silently
     equated to the report's construction.
  2. NO CLIPPING of ``q_day``. If the valley VWAP falls outside ``[lo, hi]`` the range
     construction is wrong, and clipping to [0,1] would hide exactly the defect worth
     seeing. Under a correct range a traded price cannot escape it, so an out-of-range
     value is a signal to investigate, not a value to sanitize.
  3. THE REVERSAL IS TAKEN AT T-1: ``rev20 = -(close_{d-1}/close_{d-21} - 1)`` on
     FRONT-ADJUSTED (qfq) daily closes. The report's naive 20-day reversal uses
     ``close_d``, which is 15:00 information and WOULD be a lookahead at our 14:50
     decision time. This is the direct application of the standing authorization that
     implicitly-leaking fields be taken from pre-14:50 data. The closes come from the
     panel the runner already loads — NO new data source. Front-adjusted closes are
     REQUIRED here (unlike the within-day quantities below) because the ratio spans 20
     trading days and would otherwise break across any split or dividend.
  4. RAW (UNADJUSTED) MINUTE PRICES for the range and the VWAP, with ONE DISCLOSED
     IMPERFECTION. PR-I/PR-J/PR-K could argue the adjustment factor cancels because both
     of their legs sat inside ONE trading day. That argument does NOT fully hold here:
     ``prev_close`` crosses the overnight boundary, so on an EX-DIVIDEND / SPLIT date the
     previous raw close sits on the OLD (higher) price scale while the day's own bars sit
     on the new one. The DIRECTION is therefore one-sided, not symmetric: ``prev_close``
     pulls ``hi`` UP (measured on real cache: 42 top-widening vs 4 bottom-widening among
     true ex-dates), inflating the denominator ``hi - lo`` while the valley VWAP and
     ``lo`` stay on the new scale — which compresses that day's ``q_day`` toward the LOW
     END of an artificially wide range, NOT toward the middle.
     MEASURED on the 30-name prototype panel: 15.6% of valid days show ANY range
     widening from ``prev_close``, but that is overwhelmingly ordinary overnight gapping,
     which is the INTENDED behaviour (the report's range includes the previous close
     precisely so a gap counts); only 0.73% of valid days are true ex-dates, and only
     ~0.21% are BOTH a true ex-date AND actually range-distorted — before the 20-day mean
     dilutes it further. DISCLOSED here rather than silently corrected, because
     correcting it would mix an adjusted price into an otherwise raw, single-visibility
     construction.
  5. THE RANGE USES A PRICE GUARD, THE VWAP USES A TRADE GUARD. A bar enters the
     high/low range if its prices are finite with ``high >= low > 0`` (a zero-volume
     minute still quotes a real price); it enters the VWAP only if BOTH ``volume`` and
     ``amount`` are finite and strictly positive (PR-I's positive-trade guard). Both
     guards run at the summation step only, never before classification, so the shared
     same-slot baseline — and therefore all merged factors — stay bit-identical.
  6. DAY VALIDITY needs all of: >= ``min_classifiable`` (=100) classifiable bars (PR-F's
     gate, unchanged), >= ``min_valley_bars`` (=20) TRADABLE valley bars (PR-I's floor),
     strictly positive valley volume, an available ``prev_close``, and ``hi > lo``. A
     symbol's FIRST visible day therefore never produces a value (no previous day).
  7. THE RESIDUALIZATION IS PER DATE ON THE COVERED CROSS-SECTION, mirroring PR-G's
     per-date cross-sectional post-processing. A date with fewer than
     ``min_cross_section`` (=10) symbols carrying BOTH ``qbar`` and ``rev20`` is all-NaN;
     a symbol missing ``rev20`` is dropped from the regression and its residual is NaN
     (never a silent zero fill); a degenerate cross-section (zero-variance ``rev20``,
     which cannot identify a slope) is likewise NaN rather than an unresidualized
     ``qbar`` passed off as neutralized.
  8. EVERYTHING SPANS THE PIT-VISIBLE WINDOW 09:31-14:50 ONLY, while the report uses the
     full session. Reading the closing auction would be lookahead at our decision time.
     Disclosed, not silently equated.

Pre-registered sign = +1. The report's full-market RankIC is +6.34% / RankICIR 4.32
(long leg 13.1%/yr, long-short 20.22%/yr, IR 3.29, max drawdown 10.18%, monthly win rate
80.4%); its CSI500 sub-domain long-short is 11.71% / IR 1.76, the closest comparable to
our eval cell. Semantics per the report: a HIGH position of the calm-minute (valley)
price within the day's range means the informed, unhurried part of the day traded near
the top of the range -> higher future return.

The value at ``d`` uses only bars at dates <= d and closes at dates <= d-1, so a factor
value never sees a future bar (invariant #1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import MARKET_DAILY, STK_MINS_1MIN
from data.clean.intraday_schema import (
    DAILY_INDEX_NAMES,
    DATE_LEVEL,
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
    positive_trade_mask,
    prepare_visible_minute_bars,
    rolling_valid_days,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The classification constants live with the taxonomy in ``primitives``.
VALLEY_QUANTILE_LOOKBACK_DAYS = 20  # trailing VALID trading-day mean window, includes d
VALLEY_QUANTILE_MIN_VALLEY_BARS = 20  # min TRADABLE valley bars for a valid day (PR-I's)
VALLEY_QUANTILE_REVERSAL_DAYS = 20  # span of the reversal factor neutralized against
VALLEY_QUANTILE_MIN_CROSS_SECTION = 10  # min paired cross-section for a finite residual

# The extra 1min columns this family needs on top of the taxonomy's (volume): the traded
# value for the VWAP, and the prices that form the day's range / the previous close.
_EXTRA_COLUMNS = ("amount", "high", "low", "close")


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def valley_price_quantile_by_day(
    work: pd.DataFrame,
    *,
    min_valley_bars: int = VALLEY_QUANTILE_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> pd.Series:
    """Daily valley-VWAP price quantile for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~factors.compute.minute.primitives.peak_mask_for_symbol`, built from bars
    prepared with ``extra_columns=("amount", "high", "low", "close")`` so the traded
    value and the prices ride alongside the ``valley`` / ``classifiable`` masks.

    Implements pinned choices 1-6 of the module docstring: the range is
    ``[min(visible low, prev_close), max(visible high, prev_close)]`` where ``prev_close``
    is the previous trading day's LAST VISIBLE usable close; the valley VWAP is
    ``Σamount/Σvolume`` over guarded valley bars; the quantile is NOT clipped.

    Returns:
        Series indexed by ``trade_date`` (ascending) holding ``q_day`` for the days that
        clear every validity gate — invalid days are ABSENT, not NaN, so they do not
        occupy a slot in the caller's trailing window (the rule PR-F..PR-K use).
    """
    # Chronological within the symbol so "the day's last visible close" is well defined.
    ordered = work.sort_values("bar_end", kind="mergesort")

    vol = ordered["volume"].to_numpy(dtype=float)
    amt = ordered["amount"].to_numpy(dtype=float)
    high = ordered["high"].to_numpy(dtype=float)
    low = ordered["low"].to_numpy(dtype=float)
    close = ordered["close"].to_numpy(dtype=float)

    # PINNED §5: a TRADE guard for the VWAP, a PRICE guard for the range. A zero-volume
    # minute still quotes a real price, so it may set the day's high/low while
    # contributing nothing to the volume-weighted price.
    tradable = positive_trade_mask(vol, amt)
    priced = np.isfinite(high) & np.isfinite(low) & (low > 0.0) & (high >= low)
    valley = ordered["valley"].to_numpy(dtype=bool) & tradable

    per_bar = pd.DataFrame(
        {
            "trade_date": ordered["trade_date"].to_numpy(),
            "valley_amt": np.where(valley, amt, 0.0),
            "valley_vol": np.where(valley, vol, 0.0),
            "valley_bars": valley.astype(np.int64),
            "classifiable_bars": ordered["classifiable"]
            .to_numpy(dtype=bool)
            .astype(np.int64),
            # +/-inf so a day with no priced bar reduces to a non-finite range and fails
            # the hi > lo gate rather than fabricating one.
            "day_high": np.where(priced, high, -np.inf),
            "day_low": np.where(priced, low, np.inf),
        }
    )
    grouped = per_bar.groupby("trade_date", sort=True)
    agg = grouped[
        ["valley_amt", "valley_vol", "valley_bars", "classifiable_bars"]
    ].sum()
    agg["day_high"] = grouped["day_high"].max()
    agg["day_low"] = grouped["day_low"].min()

    # PINNED §1: prev_close = the previous trading day's LAST VISIBLE usable close. Taken
    # from the SAME visible (<=14:50) bars as everything else — a post-cutoff bar on the
    # previous day can never become it.
    usable_close = np.isfinite(close) & (close > 0.0)
    closes = pd.DataFrame(
        {
            "trade_date": ordered["trade_date"].to_numpy()[usable_close],
            "close": close[usable_close],
        }
    )
    last_close = closes.groupby("trade_date", sort=True)["close"].last()
    # shift(1) over the symbol's own trading days: day d takes day d-1's last visible
    # close, and the FIRST day has none (NaN) -> that day is invalid.
    prev_close = last_close.reindex(agg.index).shift(1)

    pc = prev_close.to_numpy(dtype=float)
    hi = np.maximum(agg["day_high"].to_numpy(dtype=float), pc)
    lo = np.minimum(agg["day_low"].to_numpy(dtype=float), pc)

    valid = (
        (agg["classifiable_bars"].to_numpy() >= min_classifiable)
        & (agg["valley_bars"].to_numpy() >= min_valley_bars)
        & (agg["valley_vol"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(pc)
        & np.isfinite(hi)
        & np.isfinite(lo)
        & (hi > lo)
    )
    if not valid.any():
        return pd.Series(
            [], index=pd.DatetimeIndex([], name="trade_date"), dtype=float
        )

    valley_vwap = (
        agg["valley_amt"].to_numpy(dtype=float)[valid]
        / agg["valley_vol"].to_numpy(dtype=float)[valid]
    )
    # PINNED §2: NOT clipped — an out-of-range value must be visible, not sanitized.
    q = (valley_vwap - lo[valid]) / (hi[valid] - lo[valid])
    return pd.Series(q, index=agg.index[valid], dtype=float)


def _quantile_mean_for_symbol(
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
    """Trailing-mean daily quantile for ONE symbol from its PIT-visible bars.

    Classifies the minutes with the SHARED :func:`peak_mask_for_symbol`, reduces each
    valid day to its price quantile, then takes the trailing-``lookback_days``-valid-day
    mean. No cross-symbol leakage (``g`` is one symbol's slice) and no lookahead (the
    baseline is strictly prior, the mean window is trailing, the previous close is the
    previous day's).
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )
    q = valley_price_quantile_by_day(
        work, min_valley_bars=min_valley_bars, min_classifiable=min_classifiable
    )
    if q.empty:
        return [], []
    rolled = rolling_valid_days(
        q, lookback_days=lookback_days, min_valid_days=min_valid_days
    ).mean()
    days = [pd.Timestamp(d).normalize() for d in rolled.index]
    return days, list(rolled.to_numpy(dtype=float))


def compute_valley_price_quantile_stats(
    bars: pd.DataFrame,
    *,
    lookback_days: int = VALLEY_QUANTILE_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_valley_bars: int = VALLEY_QUANTILE_MIN_VALLEY_BARS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "valley_price_quantile_raw",
) -> pd.Series:
    """RAW trailing-mean price quantile panel (steps 1-5; NO neutralization yet).

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    classifies every visible minute with the SHARED PR-F taxonomy, computes each valid
    day's quantile of the valley VWAP within the prev-close-extended price range, and
    returns the trailing-``lookback_days``-VALID-day mean of that quantile.

    The REVERSAL NEUTRALIZATION (step 6) is deliberately NOT applied here — it needs the
    full-universe panel plus daily closes and is done by :func:`residualize_on_reversal`.
    The returned values are therefore the RAW ``qbar``, which is exactly what the
    prototype's before/after neutralization comparison needs.

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
        name: the returned Series name.

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the raw
        trailing-mean quantile, sorted, named ``name``. Pure: never mutates ``bars``.
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
        return empty_factor_series(name)

    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=_EXTRA_COLUMNS
    )
    if visible.empty:
        return empty_factor_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _quantile_mean_for_symbol(
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
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


def reversal_20(
    closes: pd.DataFrame | pd.Series,
    *,
    days: int = VALLEY_QUANTILE_REVERSAL_DAYS,
    close_col: str = "close",
    name: str = "rev20",
) -> pd.Series:
    """T-1-based ``days``-day reversal factor from FRONT-ADJUSTED daily closes.

    ``rev20(d) = -(close_{d-1} / close_{d-(days+1)} - 1)`` — PINNED §3 of the module
    docstring. The report's naive form uses ``close_d``, which is 15:00 information and
    would be a LOOKAHEAD at our 14:50 decision time; taking the whole ratio one day
    earlier makes every input strictly visible when the factor is formed.

    The closes MUST be front-adjusted: the ratio spans ``days`` trading days and a raw
    series would break across any split or dividend inside the window.

    Args:
        closes: ``MultiIndex(date, symbol)`` daily panel (DataFrame with ``close_col``,
            or a Series of closes). Typically ``panel[["close"]]`` from the runner —
            NO new data source.
        days: reversal span in trading days (definition, not a tuned knob).
        close_col: the close column name when ``closes`` is a DataFrame.
        name: the returned Series name.

    Returns:
        ``MultiIndex(date, symbol)`` Series over the SAME rows as the input, NaN where
        the T-1 window is incomplete or a close is non-positive / non-finite (honest
        missing — never fabricated). Pure: never mutates ``closes``.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1; got {days!r}.")
    if isinstance(closes, pd.DataFrame):
        if close_col not in closes.columns:
            raise ValueError(
                f"reversal_20 needs a '{close_col}' column; got {list(closes.columns)}."
            )
        series = closes[close_col]
    else:
        series = closes
    if series.empty:
        return empty_factor_series(name)

    s = series.astype(float).sort_index()
    # Non-positive / non-finite closes cannot form a ratio; blank them BEFORE shifting so
    # a bad print poisons only the windows that actually touch it.
    clean = s.where(np.isfinite(s.to_numpy(dtype=float)) & (s.to_numpy(dtype=float) > 0.0))
    by_symbol = clean.groupby(level=SYMBOL_LEVEL, sort=False)
    prior = by_symbol.shift(1)  # close_{d-1}
    base = by_symbol.shift(days + 1)  # close_{d-(days+1)}
    rev = -(prior / base - 1.0)
    return rev.rename(name)


def residualize_on_reversal(
    qbar: pd.Series,
    rev: pd.Series,
    *,
    min_cross_section: int = VALLEY_QUANTILE_MIN_CROSS_SECTION,
    name: str = "valley_price_quantile",
) -> pd.Series:
    """Per-date cross-sectional OLS residual of ``qbar`` on ``rev`` (step 6).

    The report neutralizes this factor against the 20-day reversal because the raw
    quantile is positively exposed to it. For each panel date the cross-section is the
    symbols carrying BOTH a finite ``qbar`` and a finite ``rev``; the residual is
    ``qbar - (a + b*rev)`` from an intercept OLS fit on that cross-section. Mirrors PR-G's
    per-date cross-sectional post-processing, and follows the project's neutralization
    convention that an under-determined cross-section yields NaN rather than a fabricated
    value.

    PINNED §7: a date with fewer than ``min_cross_section`` paired symbols is all-NaN; a
    symbol missing ``rev`` is dropped from the fit and its residual is NaN (never a silent
    zero fill); a degenerate cross-section (zero-variance ``rev``, which cannot identify a
    slope) is likewise NaN rather than an unresidualized ``qbar`` passed off as
    neutralized.

    Args:
        qbar: ``MultiIndex(date, symbol)`` raw trailing-mean quantile panel.
        rev: ``MultiIndex(date, symbol)`` reversal panel (from :func:`reversal_20`).
        min_cross_section: minimum paired cross-section for a finite residual.
        name: the returned Series name (the factor-panel column name).

    Returns:
        A Series over EXACTLY ``qbar``'s rows, in ``qbar``'s order, holding the residual
        (or NaN). Pure: never mutates its inputs.
    """
    if min_cross_section < 2:
        # Need >= 2 cross-section members before an intercept + slope fit is determined.
        raise ValueError(
            f"min_cross_section must be >= 2; got {min_cross_section!r}."
        )
    if qbar is None or qbar.empty:
        return empty_factor_series(name)

    y_all = qbar.to_numpy(dtype=float)
    # Align the reversal onto qbar's rows WITHOUT reordering them (the output must line
    # up with the caller's panel row-for-row).
    x_all = rev.reindex(qbar.index).to_numpy(dtype=float)
    out = np.full(y_all.shape, np.nan, dtype=float)

    dates = qbar.index.get_level_values(DATE_LEVEL).to_numpy()
    paired = np.isfinite(y_all) & np.isfinite(x_all)
    for date in pd.unique(dates):
        on_date = dates == date
        fit_rows = on_date & paired
        n = int(fit_rows.sum())
        if n < min_cross_section:
            continue  # honest missing: the whole date stays NaN
        x = x_all[fit_rows]
        y = y_all[fit_rows]
        x_mean = x.mean()
        sxx = float(((x - x_mean) ** 2).sum())
        if not np.isfinite(sxx) or sxx <= 0.0:
            continue  # degenerate: no slope is identified -> NaN, not a raw qbar
        y_mean = y.mean()
        slope = float(((x - x_mean) * (y - y_mean)).sum()) / sxx
        out[fit_rows] = y - (y_mean + slope * (x - x_mean))

    return pd.Series(out, index=qbar.index, name=name)


def compute_valley_price_quantile(
    bars: pd.DataFrame,
    closes: pd.DataFrame | pd.Series,
    *,
    lookback_days: int = VALLEY_QUANTILE_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_valley_bars: int = VALLEY_QUANTILE_MIN_VALLEY_BARS,
    min_cross_section: int = VALLEY_QUANTILE_MIN_CROSS_SECTION,
    reversal_days: int = VALLEY_QUANTILE_REVERSAL_DAYS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "valley_price_quantile",
) -> pd.Series:
    """Single-call factor: raw quantile stats (1-5) then reversal neutralization (6).

    Convenience that chains :func:`compute_valley_price_quantile_stats`,
    :func:`reversal_20` and :func:`residualize_on_reversal`. The runner does NOT use this
    (it needs the per-symbol stats loop for memory-boundedness, and it reports the
    before/after-neutralization diagnostics) but tests and any all-in-memory caller can.

    ``closes`` must be the FRONT-ADJUSTED daily close panel — see PINNED §3. Pure: never
    mutates its inputs.
    """
    stats = compute_valley_price_quantile_stats(
        bars,
        lookback_days=lookback_days,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
        min_valid_days=min_valid_days,
        min_classifiable=min_classifiable,
        min_valley_bars=min_valley_bars,
        decision_time=decision_time,
    )
    if stats.empty:
        return empty_factor_series(name)
    rev = reversal_20(closes, days=reversal_days)
    return residualize_on_reversal(
        stats, rev, min_cross_section=min_cross_section, name=name
    )


class ValleyPriceQuantileFactor(Factor):
    """Valley weighted-price-quantile factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_valley_price_quantile_stats` and then
    reversal-neutralized by :func:`residualize_on_reversal`); it does NO minute work of
    its own, mirroring its siblings.

    Args:
        lookback_days: trailing VALID trading-day window averaged; part of the factor
            DEFINITION (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"valley_price_quantile_{VALLEY_QUANTILE_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = VALLEY_QUANTILE_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"valley-price-quantile lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"valley_price_quantile_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1 (report full-market RankIC +6.34%, RankICIR 4.32, long leg
        13.1%/yr, long-short 20.22%/yr, IR 3.29, max drawdown 10.18%, monthly win rate
        80.4%; CSI500 sub-domain long-short 11.71% / IR 1.76 — the closest comparable to
        our eval cell). Semantics per the report: a HIGH position of the calm-minute
        (valley) price within the day's range means the informed, unhurried part of the
        day traded near the top of the range -> higher future return. The sign is fixed
        BEFORE the run (a validated prototype must reproduce it). NOTE the report is a
        MONTHLY, market-cap + industry neutral full-market series on Wind data while our
        eval cell is CSI500 daily with industry + size neutral, so the report numbers are
        a LOOSE reference only (disclosed, never mislabeled, never written in as an
        expected value). is_intraday=False: minute INPUT but a DAILY signal traded
        close-to-close. min_history_bars=0: the warm-up is DATA-dependent, not a fixed
        leading count — the honest NaN rate is reported by data_coverage.

        The description spells out the two subtleties that make this factor harder than
        its five siblings: the PREV_CLOSE-extended range read at OUR 14:50 visibility
        rather than the report's true daily close, and the reversal neutralization taken
        at T-1 so day d's 15:00 close is never an input to day d's value.

        D1 declarations (D0 pre-assignment table row 10 + note 2 — the
        two-axes-differ exemplar): adjustment=returns_invariant — the rev20
        leg is a ratio of FRONT-ADJUSTED daily closes (anchor cancels; this
        module's ``reversal_20`` PINNED §3) and the minute legs are raw but
        fully same-day, so an ANCHOR perturbation does not move the value.
        overnight_boundary=CROSSED_DISCLOSED — prev_close is a raw close on
        the PREVIOUS day's basis entering day d's range, the pinned definition
        keeps the ex-date values, and the deviation is MEASURED and disclosed
        (0.73% true ex-dates / ~0.21% ex-date AND range-distorted; module
        docstring PINNED §4, pinned in
        tests/test_valley_price_quantile_factor.py). All three
        CROSSED_DISCLOSED requirements are met — unlike PR-D, nothing is owed.
        ``requires`` adds market_daily close beyond the minute fields: the T-1
        rev20 neutralization reads the qfq DAILY close panel (a real second
        endpoint, declared truthfully rather than mirrored off input_fields).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Valley weighted-price-quantile (Kaiyuan microstructure series #27, "
                f"SIXTH factor 量谷加权价格分位点). SAME minute classification as PR-F "
                f"volume_peak_count / PR-H peak_interval_kurtosis / PR-I "
                f"valley_relative_vwap / PR-J valley_ridge_vwap_ratio / PR-K "
                f"ridge_minute_return (SHARED taxonomy in factors.compute.minute."
                f"primitives, not re-implemented): 1min bars PIT-truncated at 14:50, a "
                f"minute is ERUPTIVE if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of its "
                f"SAME-SLOT strictly-prior {VOLUME_PRV_BASELINE_DAYS}-day baseline, and "
                f"a VALLEY (量谷) is a classifiable NON-eruptive minute. NEW STATISTIC "
                f"vs PR-F..PR-K: a price POSITION rather than a count, a timing moment, "
                f"a price RATIO or a return — each valid day scores WHERE the valley "
                f"VWAP sits inside the day's price range, and the factor averages that "
                f"over the trailing {self._lookback_days} VALID days before a "
                f"cross-sectional reversal neutralization. PINNED choices: (1) the "
                f"range is [min(visible low, prev_close), max(visible high, "
                f"prev_close)] where prev_close is the LAST VISIBLE (<=14:50) RAW CLOSE "
                f"of the PREVIOUS trading day — a DISCLOSED DEVIATION from the report, "
                f"which uses the true previous daily close; the previous 15:00 close is "
                f"not itself a lookahead at a 14:50 decision, so this is a "
                f"SINGLE-VISIBILITY-DEFINITION choice (every price in the factor comes "
                f"from one source under one cutoff), not a leakage fix; (2) the daily "
                f"quantile (valley VWAP - lo)/(hi - lo) is NOT clipped — a VWAP outside "
                f"the range means the range is wrong and must be visible rather than "
                f"sanitized; (3) THE REVERSAL NEUTRALIZATION IS TAKEN AT T-1: rev20 = "
                f"-(close_(d-1)/close_(d-{VALLEY_QUANTILE_REVERSAL_DAYS + 1}) "
                f"- 1) on FRONT-ADJUSTED daily closes from the panel (NO new data source). "
                f"The report's naive form uses close_d, which is 15:00 information and "
                f"WOULD be a lookahead at our 14:50 decision — day d's close is therefore "
                f"never an input to day d's factor value; front-adjusted closes are "
                f"required because the ratio spans {VALLEY_QUANTILE_REVERSAL_DAYS} "
                f"trading days; (4) RAW minute prices for the range and the VWAP, with a "
                f"DISCLOSED imperfection: prev_close crosses the overnight boundary, so on "
                f"an ex-dividend / split date the previous raw close sits on the OLD "
                f"(higher) scale while the day's own bars sit on the new one. The bias is "
                f"ONE-SIDED: prev_close pulls hi UP (measured 42 top-widening vs 4 "
                f"bottom-widening among true ex-dates), inflating hi - lo while the valley "
                f"VWAP and lo stay on the new scale, which compresses that day's quantile "
                f"toward the LOW END of an artificially wide range — NOT toward the middle. "
                f"Measured: 15.6% of valid days show any prev_close widening (overwhelmingly "
                f"ordinary overnight gaps, which is INTENDED), only 0.73% are true ex-dates "
                f"and only ~0.21% are both a true ex-date AND range-distorted, before the "
                f"{self._lookback_days}-day mean dilutes it; disclosed rather than silently "
                f"corrected; (5) a PRICE guard (finite, high >= low > 0) admits a bar to "
                f"the range while a TRADE guard (finite positive volume AND amount) "
                f"admits it to the VWAP, both applied at the summation step only so PR-F's "
                f"baseline stays bit-identical; (6) a day is VALID iff it has >= "
                f"{VOLUME_PRV_MIN_CLASSIFIABLE} classifiable bars AND >= "
                f"{VALLEY_QUANTILE_MIN_VALLEY_BARS} TRADABLE valley bars AND positive "
                f"valley volume AND an available prev_close AND hi > lo — so a symbol's "
                f"FIRST visible day never produces a value; (7) the residualization is "
                f"PER DATE on the covered cross-section, with < "
                f"{VALLEY_QUANTILE_MIN_CROSS_SECTION} paired symbols, a missing rev20, or "
                f"a degenerate (zero-variance) rev20 all yielding NaN rather than a "
                f"zero-filled or unresidualized value passed off as neutralized; (8) "
                f"DEVIATION FROM THE REPORT, disclosed: everything spans the PIT-VISIBLE "
                f"window 09:31-14:50 only, not the full session. NaN below "
                f"{VOLUME_PRV_MIN_VALID_DAYS} valid days. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from, plus the daily
            # close the T-1 reversal neutralization regresses against. Declared for honest
            # provenance disclosure (data_coverage lists them); the daily panel surfaces
            # the pre-aggregated column itself.
            input_fields=("volume", "amount", "high", "low", "close"),
            requires=(
                *_minute_requires("volume", "amount", "high", "low", "close"),
                # The T-1 rev20 neutralization reads the front-adjusted DAILY
                # close panel — a genuine market_daily requirement on top of
                # the minute fields (see the docstring's D1 paragraph).
                PanelField("close", source=MARKET_DAILY),
            ),
            adjustment="returns_invariant",
            overnight_boundary="crossed_disclosed",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily valley-price-quantile column off ``panel``.

        The runner runs ``compute_valley_price_quantile_stats`` per symbol on the minute
        cache and ``residualize_on_reversal`` once on the assembled panel upstream, then
        joins the result as ``self.name``; here we only surface it, so this factor does no
        temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"ValleyPriceQuantileFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_valley_price_quantile_stats + residualize_on_reversal and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "VALLEY_QUANTILE_LOOKBACK_DAYS",
    "VALLEY_QUANTILE_MIN_CROSS_SECTION",
    "VALLEY_QUANTILE_MIN_VALLEY_BARS",
    "VALLEY_QUANTILE_REVERSAL_DAYS",
    "ValleyPriceQuantileFactor",
    "compute_valley_price_quantile",
    "compute_valley_price_quantile_stats",
    "residualize_on_reversal",
    "reversal_20",
    "valley_price_quantile_by_day",
]
