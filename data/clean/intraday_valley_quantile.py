"""VALLEY WEIGHTED-PRICE-QUANTILE factor (PR-L).

Reproduces the SIXTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§5): "将『日内最高价、日内最低价、昨日收盘价』三者的最高值与最低值作为区间的价格高低位，并计算
每日量谷成交量加权价格的相对分位点，将 20 日的分位点均值作为量谷加权价格分位点因子…考虑到因子
计算中正向暴露 20 日反转因子，我们进一步对该因子做反转中性化处理".

Same MACHINE as PR-F / PR-H / PR-I / PR-J / PR-K, a new STATISTIC FAMILY. The minute
classification is REUSED verbatim from :mod:`data.clean.intraday_volume_prv`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``) — same same-slot
strictly-prior μ+kσ eruptive test, same classifiable rule. A VALLEY (量谷) is exactly
that module's ``valley`` column. Where PR-F counted peaks, PR-H measured their timing,
PR-I / PR-J took price RATIOS and PR-K a RETURN, this module measures a price POSITION:
WHERE in the day's price range the valley VWAP sits. That is the scientific point of the
run — PR-I and PR-J both passed, and a position statistic tests whether that success came
from the ratio FORM or from price-level INFORMATION generally.

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
     previous raw close sits on the old price scale and artificially widens the range,
     biasing that day's ``q_day`` toward the middle. The effect is confined to the
     handful of ex-dates per symbol per year and is diluted by the 20-day mean; it is
     DISCLOSED here rather than silently corrected, because correcting it would mix an
     adjusted price into an otherwise raw, single-visibility construction. Callers who
     care can read the ex-date share off the coverage diagnostics.
  5. THE RANGE USES A PRICE GUARD, THE VWAP USES A TRADE GUARD. A bar enters the
     high/low range if its prices are finite with ``high >= low > 0`` (a zero-volume
     minute still quotes a real price); it enters the VWAP only if BOTH ``volume`` and
     ``amount`` are finite and strictly positive (PR-I's positive-trade guard). Both
     guards run at the summation step only, never before classification, so PR-F's
     same-slot baseline — and therefore all five merged factors — stay bit-identical.
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
our eval cell. Semantics per the report: a HIGH relative position of the calm-minute
price within the day's range means the informed, unhurried part of the day traded near
the top of the range -> higher future return. NOTE the report is a MONTHLY, market-cap +
industry neutral full-market series on Wind data while our eval cell is CSI500 daily with
industry + size neutralization, so its numbers are a LOOSE reference only (disclosed,
never mislabeled, never written in as an expected value).

The value at ``d`` uses only bars at dates <= d and closes at dates <= d-1, so a factor
value never sees a future bar (invariant #1). This module is DATA-layer only: it does not
fetch, does not touch factors / alpha / portfolio / runtime, and never sees a token.

The heavy per-symbol work (steps 1-5) is split from the cross-sectional residualization
(step 6) on purpose — the same split PR-G uses: the runner streams one symbol at a time
through :func:`compute_valley_price_quantile_stats` (memory-bounded), assembles the
full-universe ``qbar`` panel, computes ``rev20`` once from the daily panel, then calls
:func:`residualize_on_reversal` ONCE. :func:`compute_valley_price_quantile` chains the
three for a single-call path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.intraday_aggregate import (
    DAILY_INDEX_NAMES,
    DATE_LEVEL,
    DEFAULT_DECISION_TIME,
)
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
VALLEY_QUANTILE_LOOKBACK_DAYS = 20  # trailing VALID trading-day mean window, includes d
VALLEY_QUANTILE_MIN_VALLEY_BARS = 20  # min TRADABLE valley bars for a valid day (PR-I's)
VALLEY_QUANTILE_REVERSAL_DAYS = 20  # span of the reversal factor neutralized against
VALLEY_QUANTILE_MIN_CROSS_SECTION = 10  # min paired cross-section for a finite residual

# The extra 1min columns this family needs on top of PR-F's (volume): the traded value
# for the VWAP, and the prices that form the day's range / the previous close.
_EXTRA_COLUMNS = ("amount", "high", "low", "close")


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def valley_price_quantile_by_day(
    work: pd.DataFrame,
    *,
    min_valley_bars: int = VALLEY_QUANTILE_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> pd.Series:
    """Daily valley-VWAP price quantile for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~data.clean.intraday_volume_prv.peak_mask_for_symbol`, built from bars
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
    tradable = np.isfinite(vol) & (vol > 0.0) & np.isfinite(amt) & (amt > 0.0)
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

    Classifies the minutes with the REUSED :func:`peak_mask_for_symbol`, reduces each
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
    rolled = q.sort_index().rolling(lookback_days, min_periods=min_valid_days).mean()
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
    classifies every visible minute with the REUSED PR-F taxonomy, computes each valid
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
        return _empty_series(name)

    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=_EXTRA_COLUMNS
    )
    if visible.empty:
        return _empty_series(name)

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
        return _empty_series(name)
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
        return _empty_series(name)

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
        return _empty_series(name)

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
        return _empty_series(name)
    rev = reversal_20(closes, days=reversal_days)
    return residualize_on_reversal(
        stats, rev, min_cross_section=min_cross_section, name=name
    )


__all__ = [
    "VALLEY_QUANTILE_LOOKBACK_DAYS",
    "VALLEY_QUANTILE_MIN_CROSS_SECTION",
    "VALLEY_QUANTILE_MIN_VALLEY_BARS",
    "VALLEY_QUANTILE_REVERSAL_DAYS",
    "compute_valley_price_quantile",
    "compute_valley_price_quantile_stats",
    "residualize_on_reversal",
    "reversal_20",
    "valley_price_quantile_by_day",
]
