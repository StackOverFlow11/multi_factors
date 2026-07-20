"""RIDGE MINUTE-RETURN factor (PR-K).

Reproduces the FIFTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§3): "计算过去 20 日量岭时点的累计收益作为量岭分钟收益因子，衡量个人投资者交易的收益贡献,
个人投资者交易的过度反应，导致量岭分钟收益因子为显著负向因子".

Same MACHINE as PR-F / PR-H / PR-I / PR-J — what changes is the STATISTIC. The four prior
reproductions from this report covered a COUNT (weak), a TIMING moment (null) and two
PRICE LEVELS (both passed). This is the RETURN family, a fourth statistic type, and the
report's ONLY NEGATIVE peak/ridge/valley factor — so it also tests whether the SIGN
transfers along with the family.

The classification is REUSED verbatim from :mod:`data.clean.intraday_volume_prv`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``, whose ``ridge`` column PR-J
already exposed) — same same-slot strictly-prior μ+kσ eruptive test, same classifiable
rule, same valid-day floor. Nothing about the taxonomy is re-implemented or modified here,
so the five factors can never drift apart.

PINNED choices (deliberate and DISCLOSED, not tuned knobs; the report is silent on every
one of them, and each is reproduced on the factor spec so a reader sees what was assumed):

  1. THE RIDGE MASK IS ``eruptive & ~peak`` (PR-J's, unchanged). A RIDGE (量岭) is an
     eruptive minute that is NOT an isolated peak — wider than the literal "eruptive next
     to an eruptive", because it also covers session-boundary eruptions and eruptions
     whose neighbour is unclassifiable. It is the exact COMPLEMENT of PR-F's conservative
     peak test, so ``valley | peak | ridge == classifiable`` stays an exact partition and
     an isolated PEAK's return is counted on NEITHER side.
  2. THE MINUTE RETURN IS ``close_t / close_{t-1} - 1`` WITH A WITHIN-DAY LAG. The
     predecessor is the previous VISIBLE bar of the SAME trade date, so each day's FIRST
     visible bar has no return and never enters the sum. The lag deliberately does NOT
     require exact 60s adjacency: a bar that opens a new session block (in practice 13:01,
     after the lunch break) returns against the last bar before the gap, which is a
     genuine price change of the stock over that interval. Dropping such bars instead
     would silently discard the post-lunch minute whenever it is a ridge. The obvious
     selection worry is defused by the taxonomy itself: the 13:01 slot is compared against
     its OWN same-slot history, so a systematically busy post-lunch minute is not
     systematically eruptive.
  3. RAW (UNADJUSTED) CLOSES. The cached minute bars are unadjusted, and that is CORRECT
     here: a split/dividend adjustment factor is constant WITHIN a day, so it cancels
     exactly in ``close_t / close_{t-1}``; and because the within-day lag already excludes
     each day's first bar, no return ever straddles an ex-date boundary. (Same reasoning
     PR-I / PR-J's ratios, PR-D's amplitude and I5b's price-limit checks rely on.)
  4. POSITIVE-CLOSE GUARD. A return is formed only when BOTH closes are finite and
     strictly positive; otherwise the bar contributes nothing and is not counted towards
     the ridge floor. The guard runs at the return step only — never before classification
     — because PR-F's same-slot μ/σ baseline must stay bit-identical.
  5. THE DAILY AGGREGATE IS A SIMPLE SUM ``s_day = Σ r_t`` over the selected ridge bars,
     NOT ``Π(1+r_t) - 1``. The report says only "累计收益" (cumulative return). Ridge
     minutes are NON-CONTIGUOUS within the day — they are scattered eruptive moments, not
     a held position — so a holding-period/compounding reading does not apply to them; and
     at minute scale the two conventions differ negligibly anyway. The choice is therefore
     the simple sum, disclosed rather than silently equated to the report's wording.
  6. THE TRAILING AGGREGATE IS ALSO A SUM. "过去 20 日累计" is read as the sum of the daily
     sums over the trailing window, i.e. the factor accumulates across days rather than
     averaging (which is what PR-J's ratio did).
  7. A RIDGE-BAR FLOOR OF 10, counted AFTER the return guard. Ridge bars are STRUCTURALLY
     scarce: a minute must erupt (a minority event by construction, since the threshold is
     its own strictly-prior same-slot μ+σ) AND fail the isolation test. This is the SAME
     floor PR-J pinned for its ridge leg, so the two runs' coverage is directly
     comparable, and the realized ridge-bar distribution plus the day-validity rate are
     REPORTED by the runner rather than left implicit.
  8. BOTH THE SUM AND THE COUNT COVER THE PIT-VISIBLE WINDOW ONLY (09:31–14:50), not the
     full session. A NECESSARY DEVIATION from the report, which uses the whole day: the
     standing 14:50 decision cutoff truncates history days and the signal day identically,
     and reading the closing auction would be lookahead at our decision time.

Factor value: ``ridge_minute_return_20`` = the SUM of ``s_day`` over the symbol's most
recent ``lookback_days`` (=20) VALID trading days INCLUDING ``d``. A day is VALID iff it
has at least ``min_classifiable`` (=100) classifiable bars (PR-F's gate, unchanged) and at
least ``min_ridge_bars`` (=10) ridge bars carrying a valid return. Fewer than
``min_valid_days`` (=10) valid days in the trailing window -> NaN (honest missing; the
runner discloses the coverage). Values are emitted only on valid days.

Pre-registered sign = -1. The report's full-market RankIC is -6.29% / RankICIR -3.55 (long
leg 7.47%/yr, long-short 14.98%/yr, IR 1.73, max drawdown 13.84%, monthly win rate 70.3%);
it gives NO CSI500 sub-domain figure for this factor, so none is quoted. Semantics per the
report: ridge minutes are retail follow-the-crowd trading, and their return contribution
measures that crowd's OVER-REACTION — the higher the accumulated ridge-minute return, the
more over-extended the stock and the worse it performs going forward. NOTE the report is a
MONTHLY, market-cap + industry neutral full-market series on Wind data while our eval cell
is CSI500 daily with industry + size neutralization, so its numbers are a LOOSE reference
only (disclosed, never mislabeled, never written in as an expected value).

The value at ``d`` uses only bars at dates <= d, so a factor value never sees a future bar
(invariant #1); it is a DAILY signal traded close-to-close from d+1. This module is
DATA-layer only: it does not fetch, does not touch factors / alpha / portfolio / runtime,
and never sees a token.
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
RIDGE_RETURN_LOOKBACK_DAYS = 20  # trailing VALID trading-day SUM window, includes d
# The SAME scarcity floor PR-J pinned for its ridge leg (module docstring §7), so the two
# runs' ridge coverage is directly comparable.
RIDGE_RETURN_MIN_RIDGE_BARS = 10  # min RETURN-CARRYING ridge bars for a valid day

# The extra 1min column this family needs on top of PR-F's (volume): the close, which
# turns consecutive bars into a minute return.
_CLOSE = "close"

# Per-day diagnostic columns (the ridge-scarcity disclosure the task card requires).
# ``ridge_bars`` is every ridge minute; ``ridge_return_bars`` is the subset that carries a
# valid return — the one the floor actually gates on — so the attrition is visible.
DIAGNOSTIC_COLUMNS = ("classifiable_bars", "ridge_bars", "ridge_return_bars", "valid")


def _empty_series(name: str) -> pd.Series:
    """Schema-shaped empty ``MultiIndex(date, symbol)`` factor Series."""
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=DAILY_INDEX_NAMES,
    )
    return pd.Series([], index=index, dtype=float, name=name)


def ridge_minute_return_by_day(
    work: pd.DataFrame,
    *,
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    with_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Daily summed ridge-minute return for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~data.clean.intraday_volume_prv.peak_mask_for_symbol`, which must have been
    built from bars prepared with ``extra_columns=("close",)`` so the close is available
    alongside the ``ridge`` / ``classifiable`` masks. ``peak_mask_for_symbol`` returns the
    rows sorted by ``(trade_date, bar_end)``, which is exactly the order the WITHIN-DAY lag
    needs.

    The minute return is ``close_t / close_{t-1} - 1`` against the previous VISIBLE bar of
    the SAME day (PINNED §2), guarded so both closes are finite and strictly positive
    (§4), and the day's value is the SIMPLE SUM of those returns over the ridge bars (§5)
    — an isolated PEAK's return is never included (§1).

    Args:
        work: one symbol's classified minute frame (see above).
        min_ridge_bars: minimum ridge bars CARRYING A VALID RETURN for a valid day.
        min_classifiable: minimum classifiable bars for a valid day (PR-F's gate).
        with_diagnostics: also return the per-day bar-count frame, so the caller can
            REPORT the ridge-scarcity distribution instead of only gating on it.

    Returns:
        Series indexed by ``trade_date`` (ascending) holding the summed ridge-minute
        return for the days that clear both gates — invalid days are ABSENT, not NaN, so
        they do not occupy a slot in the caller's trailing window (the same rule PR-F /
        PR-H / PR-I / PR-J use). With ``with_diagnostics=True``, a ``(daily, diagnostics)``
        pair where ``diagnostics`` is indexed by EVERY day present in ``work`` and carries
        :data:`DIAGNOSTIC_COLUMNS`.
    """
    close = work[_CLOSE].to_numpy(dtype=float)
    # WITHIN-DAY lag: the previous visible bar of the SAME trade date. grouping by
    # trade_date is what stops a return from ever crossing a day boundary, so each day's
    # first visible bar gets a NaN predecessor and drops out below.
    prev_close = (
        work.groupby("trade_date", sort=False)[_CLOSE].shift(1).to_numpy(dtype=float)
    )
    # Positive-close guard: a non-finite or non-positive close on EITHER side carries no
    # usable return. Applied HERE, at the return step, never before classification —
    # PR-F's same-slot baseline must stay bit-identical.
    has_return = (
        np.isfinite(close) & (close > 0.0)
        & np.isfinite(prev_close) & (prev_close > 0.0)
    )
    # Denominator neutralized where the guard failed, so no divide-by-zero / inf is ever
    # formed; those entries are discarded by the np.where immediately after.
    safe_prev = np.where(has_return, prev_close, 1.0)
    ret = np.where(has_return, close / safe_prev - 1.0, 0.0)

    ridge = work["ridge"].to_numpy(dtype=bool)
    ridge_return = ridge & has_return

    per_bar = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            # SIMPLE SUM of the selected returns (PINNED §5) — not Π(1+r)-1.
            "ridge_return_sum": np.where(ridge_return, ret, 0.0),
            "ridge_bars": ridge.astype(np.int64),
            "ridge_return_bars": ridge_return.astype(np.int64),
            "classifiable_bars": work["classifiable"]
            .to_numpy(dtype=bool)
            .astype(np.int64),
        }
    )
    agg = per_bar.groupby("trade_date", sort=True).sum()

    # Two validity gates (§ module docstring / task card §1.5): PR-F's classifiable floor
    # (unchanged) and enough RETURN-CARRYING ridge bars.
    valid = (agg["classifiable_bars"] >= min_classifiable) & (
        agg["ridge_return_bars"] >= min_ridge_bars
    )
    ok = agg.loc[valid]
    if ok.empty:
        daily = pd.Series([], index=pd.DatetimeIndex([], name="trade_date"), dtype=float)
    else:
        daily = ok["ridge_return_sum"].astype(float)

    if not with_diagnostics:
        return daily
    diagnostics = agg[["classifiable_bars", "ridge_bars", "ridge_return_bars"]].copy()
    diagnostics["valid"] = valid
    return daily, diagnostics


def _ridge_return_sum_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
    min_ridge_bars: int,
    collect_diagnostics: bool = False,
) -> tuple[list[pd.Timestamp], list[float], pd.DataFrame | None]:
    """Daily ridge-minute-return values for ONE symbol from its PIT-visible bars.

    Classifies the minutes with the REUSED :func:`peak_mask_for_symbol`, reduces each valid
    day to its summed ridge-minute return, then takes the trailing-``lookback_days``-
    valid-day SUM. No cross-symbol leakage (``g`` is one symbol's slice) and no lookahead
    (the baseline is strictly prior, the window is trailing).
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )
    result = ridge_minute_return_by_day(
        work,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        with_diagnostics=collect_diagnostics,
    )
    if collect_diagnostics:
        daily, diagnostics = result
    else:
        daily, diagnostics = result, None
    if daily.empty:
        return [], [], diagnostics

    # SUM over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Emitted only on valid days.
    rolled = daily.sort_index().rolling(lookback_days, min_periods=min_valid_days).sum()
    days = [pd.Timestamp(d).normalize() for d in rolled.index]
    return days, list(rolled.to_numpy(dtype=float)), diagnostics


def compute_ridge_minute_return(
    bars: pd.DataFrame,
    *,
    lookback_days: int = RIDGE_RETURN_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "ridge_minute_return",
    diagnostics_out: list | None = None,
) -> pd.Series:
    """PIT-safe daily "ridge minute-return" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``, classifies
    every visible minute with the REUSED PR-F taxonomy, sums each valid day's ridge-minute
    returns, and returns the trailing-``lookback_days``-VALID-day SUM of those daily sums.
    See the module docstring for the LOCKED definition and the eight pinned choices.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping is
            strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window summed (definition).
        baseline_days: strictly-prior same-slot baseline window in trading days (PR-F).
        baseline_min_obs: minimum same-slot observations for a classifiable bar (PR-F).
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ`` (PR-F).
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day needs at least this many classifiable bars (PR-F).
        min_ridge_bars: a day needs at least this many ridge bars CARRYING A VALID RETURN.
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).
        diagnostics_out: optional list the per-symbol day-level bar-count frames are
            APPENDED to (each carries a ``symbol`` column), so a caller can report the
            ridge-scarcity distribution. Purely observational — supplying it does not
            change the returned factor.

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
    if min_ridge_bars < 1:
        raise ValueError(f"min_ridge_bars must be >= 1; got {min_ridge_bars!r}.")
    if len(bars) == 0:
        return _empty_series(name)

    # extra_columns=("close",) is the ONLY difference from the PR-F / PR-H entry points:
    # the close rides along on the surviving rows, and the truncation / volume guard /
    # slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_CLOSE,)
    )
    if visible.empty:
        return _empty_series(name)

    collect = diagnostics_out is not None
    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals, diagnostics = _ridge_return_sum_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_ridge_bars=min_ridge_bars,
            collect_diagnostics=collect,
        )
        if collect and diagnostics is not None and not diagnostics.empty:
            frame = diagnostics.copy()
            frame[SYMBOL_LEVEL] = str(sym)
            diagnostics_out.append(frame)
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return _empty_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "RIDGE_RETURN_LOOKBACK_DAYS",
    "RIDGE_RETURN_MIN_RIDGE_BARS",
    "compute_ridge_minute_return",
    "ridge_minute_return_by_day",
]
