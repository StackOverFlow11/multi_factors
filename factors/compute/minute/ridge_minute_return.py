"""RIDGE MINUTE-RETURN factor (PR-K): math + surface (D2).

Reproduces the FIFTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§3): "计算过去 20 日量岭时点的累计收益作为量岭分钟收益因子，衡量个人投资者交易的收益贡献,
个人投资者交易的过度反应，导致量岭分钟收益因子为显著负向因子".

Same MACHINE as PR-F / PR-H / PR-I / PR-J — what changes is the STATISTIC. The four prior
reproductions from this report covered a COUNT (weak), a TIMING moment (null) and two
PRICE LEVELS (both passed). This is the RETURN family, a fourth statistic type, and the
report's ONLY NEGATIVE peak/ridge/valley factor — so it also tests whether the SIGN
transfers along with the family.

The classification is the SHARED taxonomy in :mod:`factors.compute.minute.primitives`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``, whose ``ridge`` column PR-J
already exposed) — same same-slot strictly-prior μ+kσ eruptive test, same classifiable
rule, same valid-day floor. Nothing about the taxonomy is re-implemented or modified here,
so the family can never drift apart.

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
     — because the shared same-slot μ/σ baseline must stay bit-identical.
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
    rolling_valid_days,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The classification constants live with the taxonomy in ``primitives``.
RIDGE_RETURN_LOOKBACK_DAYS = 20  # trailing VALID trading-day SUM window, includes d
# The SAME scarcity floor PR-J pinned for its ridge leg (module docstring §7), so the two
# runs' ridge coverage is directly comparable.
RIDGE_RETURN_MIN_RIDGE_BARS = 10  # min RETURN-CARRYING ridge bars for a valid day

# The extra 1min column this family needs on top of the taxonomy's (volume): the close,
# which turns consecutive bars into a minute return.
_CLOSE = "close"

# Per-day diagnostic columns (the ridge-scarcity disclosure the task card requires).
# ``ridge_bars`` is every ridge minute; ``ridge_return_bars`` is the subset that carries a
# valid return — the one the floor actually gates on — so the attrition is visible.
DIAGNOSTIC_COLUMNS = ("classifiable_bars", "ridge_bars", "ridge_return_bars", "valid")


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def ridge_minute_return_by_day(
    work: pd.DataFrame,
    *,
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    with_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Daily summed ridge-minute return for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~factors.compute.minute.primitives.peak_mask_for_symbol`, which must have been
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
    # the shared same-slot baseline must stay bit-identical.
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

    Classifies the minutes with the SHARED :func:`peak_mask_for_symbol`, reduces each valid
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
    rolled = rolling_valid_days(
        daily, lookback_days=lookback_days, min_valid_days=min_valid_days
    ).sum()
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
    every visible minute with the SHARED PR-F taxonomy, sums each valid day's ridge-minute
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
        return empty_factor_series(name)

    # extra_columns=("close",) is the ONLY difference from the PR-F / PR-H entry points:
    # the close rides along on the surviving rows, and the truncation / volume guard /
    # slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_CLOSE,)
    )
    if visible.empty:
        return empty_factor_series(name)

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
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class RidgeMinuteReturnFactor(Factor):
    """Ridge minute-return factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_ridge_minute_return`); it does NO minute work of its
    own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing VALID trading-day window summed; part of the factor
            DEFINITION (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"ridge_minute_return_{RIDGE_RETURN_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = RIDGE_RETURN_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"ridge-minute-return lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"ridge_minute_return_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: this is the report's ONLY NEGATIVE peak/ridge/valley factor
        (full-market RankIC -6.29%, RankICIR -3.55, long leg 7.47%/yr, long-short
        14.98%/yr, IR 1.73, max drawdown 13.84%, monthly win rate 70.3%). The report gives
        NO CSI500 sub-domain figure for this factor, so none is quoted here. Semantics per
        the report: ridge minutes are retail follow-the-crowd trading and their
        accumulated return measures that crowd's OVER-REACTION — the more ridge-minute
        return a stock has piled up, the more over-extended it is and the worse it
        performs going forward. The sign is fixed BEFORE the run (a validated prototype
        must reproduce it). NOTE the report is a MONTHLY, market-cap + industry neutral
        full-market series on Wind data while our eval cell is CSI500 daily with industry
        + size neutral, so the report numbers are a LOOSE reference only (disclosed, never
        mislabeled, never written in as an expected value). is_intraday=False by the
        module docstring's reasoning: minute INPUT but a DAILY signal traded
        close-to-close. min_history_bars=0: the warm-up is DATA-dependent (a value appears
        once enough VALID days accumulate), not a fixed leading count — the honest NaN
        rate is reported by data_coverage.

        The description spells out the RELATION TO PR-F..PR-J (same shared classification,
        a RETURN statistic instead of a count / timing moment / price level) plus the
        pinned choices — above all the SIMPLE-SUM convention, the within-day return lag and
        the raw closes, none of which the report specifies.

        D1 declarations (D0 pre-assignment table row 9): adjustment=
        returns_invariant — the minute return close_t/close_(t-1) - 1 is a
        within-day ratio, and "a split/dividend adjustment factor is constant
        WITHIN a day, so it cancels exactly" (module docstring PINNED §3).
        overnight_boundary=none — the same pinned choice's own claim, "no
        return ever straddles an ex-date boundary" (the within-day lag drops
        each day's first visible bar).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Ridge minute-return (Kaiyuan microstructure series #27, FIFTH factor "
                f"量岭分钟收益). SAME minute classification as PR-F volume_peak_count / "
                f"PR-H peak_interval_kurtosis / PR-I valley_relative_vwap / PR-J "
                f"valley_ridge_vwap_ratio (SHARED taxonomy in factors.compute.minute."
                f"primitives, not re-implemented): 1min bars PIT-truncated at 14:50, a "
                f"minute is ERUPTIVE if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of its "
                f"SAME-SLOT strictly-prior {VOLUME_PRV_BASELINE_DAYS}-day baseline, and "
                f"a RIDGE (量岭) is an eruptive minute that is NOT an isolated peak. NEW "
                f"STATISTIC vs PR-F..PR-J: a RETURN rather than a count, a timing moment "
                f"or a price level — each day sums the minute returns of its ridge bars "
                f"and the factor sums those daily sums over the trailing "
                f"{self._lookback_days} VALID days. PINNED choices (the report specifies "
                f"none of them): (1) the RIDGE mask is 'eruptive AND NOT an isolated "
                f"peak', keeping valley|peak|ridge an exact partition of the classifiable "
                f"bars — an isolated PEAK's return is counted on NEITHER side; (2) the "
                f"minute return is close_t/close_(t-1) - 1 with a WITHIN-DAY lag against "
                f"the previous VISIBLE bar of the same date, so each day's FIRST visible "
                f"bar carries no return and no return ever crosses a day boundary; exact "
                f"60s adjacency is deliberately NOT required, so a bar opening a new "
                f"session block (13:01, after lunch) returns against the last bar before "
                f"the gap — a genuine price change rather than a discarded one; (3) RAW "
                f"unadjusted closes are correct here because the adjustment factor is "
                f"constant within a day and cancels in the ratio, and the within-day lag "
                f"already excludes the one bar that could straddle an ex-date; (4) a "
                f"return is formed only when both closes are finite and strictly positive "
                f"(guard applied at the return step only, so PR-F's baseline is "
                f"untouched); (5) the daily aggregate is a SIMPLE SUM Σr, NOT a compound "
                f"Π(1+r)-1 — ridge minutes are non-contiguous within the day so a "
                f"holding-period reading does not apply, and at minute scale the two "
                f"differ negligibly; (6) the trailing aggregate is likewise a SUM across "
                f"days; (7) a day is VALID iff it has >= {VOLUME_PRV_MIN_CLASSIFIABLE} "
                f"classifiable bars AND >= {RIDGE_RETURN_MIN_RIDGE_BARS} ridge bars "
                f"CARRYING A VALID RETURN (counted AFTER the guard) — the same scarcity "
                f"floor PR-J pinned for its ridge leg, since a ridge bar must erupt AND "
                f"fail the isolation test; the realized ridge-bar distribution and "
                f"day-validity rate are REPORTED by the runner rather than left implicit; "
                f"(8) DEVIATION FROM THE REPORT, disclosed: everything spans the "
                f"PIT-VISIBLE window 09:31-14:50 only, not the full session — reading the "
                f"close would be lookahead at our 14:50 decision time. NaN below "
                f"{VOLUME_PRV_MIN_VALID_DAYS} valid days. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("volume", "close"),
            requires=_minute_requires("volume", "close"),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily ridge-minute-return column off ``panel``.

        The runner runs ``compute_ridge_minute_return`` per symbol on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"RidgeMinuteReturnFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_ridge_minute_return and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "RIDGE_RETURN_LOOKBACK_DAYS",
    "RIDGE_RETURN_MIN_RIDGE_BARS",
    "RidgeMinuteReturnFactor",
    "compute_ridge_minute_return",
    "ridge_minute_return_by_day",
]
