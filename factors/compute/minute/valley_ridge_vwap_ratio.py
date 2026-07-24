"""VALLEY/RIDGE VWAP-RATIO factor (PR-J): math + surface (D2).

Reproduces the FOURTH factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§7.1): "计算每日量谷成交量加权价格，与量岭成交量加权价格做比，并计算 20 日价格比均值作为
谷岭加权价格比因子".

Same MACHINE as PR-F / PR-H / PR-I and the same VWAP identity as PR-I — what changes is
the DENOMINATOR. PR-I divided the valley VWAP by the WHOLE VISIBLE DAY's VWAP; this
factor divides it by the RIDGE VWAP, so the two behavioural groups of the report's
peak/ridge/valley taxonomy are contrasted head-on rather than one being compared against
an aggregate that contains it. That makes PR-J the natural robustness test of PR-I: if
the price-level signal survives the swap, it is a property of the FAMILY; if it does not,
PR-I depended on its specific denominator. Either outcome is a legitimate result.

The classification is the SHARED taxonomy in :mod:`factors.compute.minute.primitives`
(``prepare_visible_minute_bars`` + ``peak_mask_for_symbol``) — same same-slot
strictly-prior μ+kσ eruptive test, same classifiable rule, same valid-day floor. Nothing
about the taxonomy is re-implemented here, so the family can never drift apart.

PINNED choices (deliberate and DISCLOSED, not tuned knobs; reproduced on the factor spec
so a reader sees exactly what was assumed):

  1. THE RIDGE MASK IS ``eruptive & ~peak``. A RIDGE (量岭) is an eruptive minute that is
     NOT an isolated peak. This is WIDER than the literal "eruptive next to an eruptive":
     it also covers the session-boundary eruptions (a neighbour missing across the lunch
     break / the cutoff / a data gap) and the eruptions whose neighbour is unclassifiable.
     The rule is the exact COMPLEMENT of PR-F's deliberately conservative peak test, so
     the taxonomy stays a clean partition — ``valley | peak | ridge == classifiable`` —
     and no eruptive bar is silently dropped on either side. Disclosed rather than
     silently equated to the report's wording.
  2. VWAP VIA THE AGGREGATION IDENTITY (same as PR-I). A bar's volume-weighted price is
     ``p = amount / volume``, so a set's volume-weighted price ``Σ(p_i·v_i) / Σv_i``
     collapses EXACTLY to ``Σamount / Σvolume``. Both legs use that identity: valley VWAP
     = Σamount(valley bars)/Σvolume(valley bars), ridge VWAP = Σamount(ridge
     bars)/Σvolume(ridge bars).
  3. POSITIVE-TRADE GUARD. A bar with non-finite or non-positive ``volume`` OR ``amount``
     carries no price information, so it is dropped from BOTH sums. The guard runs at the
     summation step only — never before classification — because the same-slot μ/σ
     baseline is the shared taxonomy's and must stay bit-identical. A zero-volume minute
     is therefore still classified (and still a valley) but contributes nothing to either
     VWAP.
  4. RAW (UNADJUSTED) PRICES. The cached minute bars are unadjusted. Both legs are sums
     over the SAME trading day, and a split/dividend adjustment factor is constant within
     a day, so it cancels exactly in the ratio — no adjustment is needed (the same
     reasoning PR-I, PR-D's amplitude and I5b's price-limit checks rely on).
  5. BOTH LEGS COVER THE PIT-VISIBLE WINDOW ONLY (09:31–14:50), not the full session.
     A NECESSARY DEVIATION from the report, which uses the whole day: the standing 14:50
     decision cutoff truncates history days and the signal day identically, and reading
     the closing auction would be lookahead at our decision time.
  6. AN ASYMMETRIC BAR FLOOR — ``min_valley_bars`` (=20) but ``min_ridge_bars`` (=10).
     Ridge bars are STRUCTURALLY far scarcer than valley bars: a minute must erupt (a
     minority event by construction, since the threshold is its own strictly-prior
     same-slot μ+σ) AND fail the isolation test. Holding the ridge leg to the valley
     leg's floor would throw away a large share of otherwise sound days and bias the
     surviving sample towards unusually turbulent sessions. The floor is therefore
     LOWERED, deliberately, and both the realized ridge-bar distribution and the
     resulting day-validity rate are REPORTED (``with_diagnostics=True``) rather than
     left implicit — if coverage comes out materially worse than PR-I's, that is a
     finding to surface, not to hide.
  7. BOTH BAR COUNTS ARE TAKEN AFTER THE GUARD, so a day cannot qualify on the strength
     of bars that traded nothing.

Factor value: ``valley_ridge_vwap_ratio_20`` = the MEAN of the daily ratio
``valley VWAP / ridge VWAP`` over the symbol's most recent ``lookback_days`` (=20) VALID
trading days INCLUDING ``d``. A day is VALID iff it clears four gates: at least
``min_classifiable`` (=100) classifiable bars (PR-F's gate, unchanged), at least
``min_valley_bars`` (=20) tradable valley bars, at least ``min_ridge_bars`` (=10)
tradable ridge bars, and strictly positive volume in BOTH denominators. Fewer than
``min_valid_days`` (=10) valid days in the trailing window -> NaN (honest missing; the
runner discloses the coverage). Values are emitted only on valid days.

Pre-registered sign = +1. The report's full-market RankIC is +6.98% / RankICIR 3.56
(long-short 15.83%/yr, IR 1.83, monthly win rate 72.3%, only 2023 negative in 13 years);
its CSI500 sub-domain long-short is 10.49% / IR 1.34, the closest comparable to our eval
cell. Semantics per the report: a HIGH valley/ridge price ratio means retail
over-reaction pushed the eruptive minutes' price DOWN relative to the calm ones, so the
stock is depressed and performs better going forward. NOTE the report is a MONTHLY,
market-cap + industry neutral full-market series on Wind data while our eval cell is
CSI500 daily with industry + size neutralization, so its numbers are a LOOSE reference
only (disclosed, never mislabeled, never written in as an expected value).

The value at ``d`` uses only bars at dates <= d, so a factor value never sees a future
bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.
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
    positive_trade_mask,
    prepare_visible_minute_bars,
    rolling_valid_days,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (pinned interpretations of the report; NOT tuned knobs).
# The classification constants live with the taxonomy in ``primitives``.
VALLEY_RIDGE_LOOKBACK_DAYS = 20  # trailing VALID trading-day mean window, includes d
VALLEY_RIDGE_MIN_VALLEY_BARS = 20  # min TRADABLE valley bars for a valid day
# PINNED LOWER than the valley floor (module docstring §6): ridge bars are structurally
# far scarcer, so holding both legs to 20 would bias the surviving sample.
VALLEY_RIDGE_MIN_RIDGE_BARS = 10  # min TRADABLE ridge bars for a valid day

# The extra 1min column this family needs on top of the taxonomy's (volume): the traded
# value, which turns the bar set into a volume-weighted price via Σamount/Σvolume.
_AMOUNT = "amount"

# Per-day diagnostic columns (the ridge-scarcity disclosure the task card requires).
DIAGNOSTIC_COLUMNS = ("classifiable_bars", "valley_bars", "ridge_bars", "valid")


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def valley_ridge_vwap_ratio_by_day(
    work: pd.DataFrame,
    *,
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS,
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    with_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Daily ``valley VWAP / ridge VWAP`` ratio for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~factors.compute.minute.primitives.peak_mask_for_symbol`, which must have been
    built from bars prepared with ``extra_columns=("amount",)`` so the traded value is
    available alongside the ``valley`` / ``ridge`` / ``classifiable`` masks.

    Both VWAPs use the ``Σamount / Σvolume`` aggregation identity (PINNED §2 of the
    module docstring). The POSITIVE-TRADE GUARD (§3) drops non-finite / non-positive
    volume or amount bars from both sums. Unlike PR-I, the denominator is NOT the whole
    visible day but the RIDGE bars alone (§1: ``eruptive & ~peak``), so an isolated PEAK
    contributes to NEITHER leg.

    Args:
        work: one symbol's classified minute frame (see above).
        min_valley_bars: minimum TRADABLE valley bars for a valid day.
        min_ridge_bars: minimum TRADABLE ridge bars for a valid day (PINNED lower than
            the valley floor — see §6).
        min_classifiable: minimum classifiable bars for a valid day (PR-F's gate).
        with_diagnostics: also return the per-day bar-count frame, so the caller can
            REPORT the ridge-scarcity distribution instead of only gating on it.

    Returns:
        Series indexed by ``trade_date`` (ascending) holding the ratio for the days that
        clear all gates — invalid days are ABSENT, not NaN, so they do not occupy a slot
        in the caller's trailing window (the same rule PR-F / PR-H / PR-I use). With
        ``with_diagnostics=True``, a ``(ratio, diagnostics)`` pair where ``diagnostics``
        is indexed by EVERY day present in ``work`` and carries
        :data:`DIAGNOSTIC_COLUMNS`.
    """
    vol = work["volume"].to_numpy(dtype=float)
    amt = work[_AMOUNT].to_numpy(dtype=float)
    # Positive-trade guard: no price information in a bar that traded nothing (or whose
    # fields are non-finite / negative). Applied HERE, at the summation step, never
    # before classification — the shared same-slot baseline must stay bit-identical.
    tradable = positive_trade_mask(vol, amt)
    valley = work["valley"].to_numpy(dtype=bool) & tradable
    ridge = work["ridge"].to_numpy(dtype=bool) & tradable

    per_bar = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            "valley_amt": np.where(valley, amt, 0.0),
            "valley_vol": np.where(valley, vol, 0.0),
            "ridge_amt": np.where(ridge, amt, 0.0),
            "ridge_vol": np.where(ridge, vol, 0.0),
            "valley_bars": valley.astype(np.int64),
            "ridge_bars": ridge.astype(np.int64),
            "classifiable_bars": work["classifiable"]
            .to_numpy(dtype=bool)
            .astype(np.int64),
        }
    )
    agg = per_bar.groupby("trade_date", sort=True).sum()

    # Four validity gates (§ module docstring / task card §1.4): PR-F's classifiable
    # floor (unchanged), enough TRADABLE valley bars, enough TRADABLE ridge bars, and a
    # positive denominator on both legs (a 0/0 ratio is never fabricated).
    valid = (
        (agg["classifiable_bars"] >= min_classifiable)
        & (agg["valley_bars"] >= min_valley_bars)
        & (agg["ridge_bars"] >= min_ridge_bars)
        & (agg["valley_vol"] > 0.0)
        & (agg["ridge_vol"] > 0.0)
    )
    ok = agg.loc[valid]
    if ok.empty:
        ratio = pd.Series([], index=pd.DatetimeIndex([], name="trade_date"), dtype=float)
    else:
        valley_vwap = ok["valley_amt"] / ok["valley_vol"]
        ridge_vwap = ok["ridge_amt"] / ok["ridge_vol"]
        ratio = (valley_vwap / ridge_vwap).astype(float)

    if not with_diagnostics:
        return ratio
    diagnostics = agg[["classifiable_bars", "valley_bars", "ridge_bars"]].copy()
    diagnostics["valid"] = valid
    return ratio, diagnostics


def _valley_ridge_ratio_mean_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
    min_valley_bars: int,
    min_ridge_bars: int,
    collect_diagnostics: bool = False,
) -> tuple[list[pd.Timestamp], list[float], pd.DataFrame | None]:
    """Daily valley/ridge VWAP-ratio values for ONE symbol from its PIT-visible bars.

    Classifies the minutes with the SHARED :func:`peak_mask_for_symbol`, reduces each
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
    result = valley_ridge_vwap_ratio_by_day(
        work,
        min_valley_bars=min_valley_bars,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        with_diagnostics=collect_diagnostics,
    )
    if collect_diagnostics:
        ratio, diagnostics = result
    else:
        ratio, diagnostics = result, None
    if ratio.empty:
        return [], [], diagnostics

    # Mean over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Emitted only on valid days.
    rolled = rolling_valid_days(
        ratio, lookback_days=lookback_days, min_valid_days=min_valid_days
    ).mean()
    days = [pd.Timestamp(d).normalize() for d in rolled.index]
    return days, list(rolled.to_numpy(dtype=float)), diagnostics


def compute_valley_ridge_vwap_ratio(
    bars: pd.DataFrame,
    *,
    lookback_days: int = VALLEY_RIDGE_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS,
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "valley_ridge_vwap_ratio",
    diagnostics_out: list | None = None,
) -> pd.Series:
    """PIT-safe daily "valley/ridge VWAP ratio" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    classifies every visible minute with the SHARED PR-F taxonomy, computes each valid
    day's ``valley VWAP / ridge VWAP`` ratio via the ``Σamount/Σvolume`` identity, and
    returns the trailing-``lookback_days``-VALID-day mean of that ratio. See the module
    docstring for the LOCKED definition and the seven pinned choices.

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
        min_ridge_bars: a day needs at least this many TRADABLE ridge bars (PINNED lower
            than the valley floor — see the module docstring §6).
        decision_time: per-bar PIT cutoff time-of-day (default 14:50:00).
        name: the returned Series name (the factor-panel column name).
        diagnostics_out: optional list the per-symbol day-level bar-count frames are
            APPENDED to (each carries a ``symbol`` column), so a caller can report the
            ridge-scarcity distribution. Purely observational — supplying it does not
            change the returned factor.

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
    if min_ridge_bars < 1:
        raise ValueError(f"min_ridge_bars must be >= 1; got {min_ridge_bars!r}.")
    if len(bars) == 0:
        return empty_factor_series(name)

    # extra_columns=("amount",) is the ONLY difference from the PR-F / PR-H entry
    # points: the traded value rides along on the surviving rows, and the truncation /
    # volume guard / slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_AMOUNT,)
    )
    if visible.empty:
        return empty_factor_series(name)

    collect = diagnostics_out is not None
    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals, diagnostics = _valley_ridge_ratio_mean_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
            min_valley_bars=min_valley_bars,
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


class ValleyRidgeVwapRatioFactor(Factor):
    """Valley/ridge VWAP-ratio factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_valley_ridge_vwap_ratio`); it does NO minute work of its
    own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing VALID trading-day window averaged; part of the factor
            DEFINITION (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"valley_ridge_vwap_ratio_{VALLEY_RIDGE_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = VALLEY_RIDGE_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"valley-ridge-vwap-ratio lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"valley_ridge_vwap_ratio_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's full-market RankIC is +6.98% (RankICIR 3.56,
        long-short 15.83%/yr, IR 1.83, monthly win rate 72.3%, only 2023 negative across
        13 years), and its CSI500 sub-domain long-short is 10.49% / IR 1.34, the closest
        comparable to our eval cell. Semantics per the report: a HIGH valley/ridge price
        ratio means retail over-reaction pushed the eruptive minutes' price DOWN relative
        to the calm ones, so the stock is depressed and performs better going forward.
        The sign is fixed BEFORE the run (a validated prototype must reproduce it). NOTE
        the report is a MONTHLY, market-cap + industry neutral full-market series on Wind
        data while our eval cell is CSI500 daily with industry + size neutral, so the
        report numbers are a LOOSE reference only (disclosed, never mislabeled, never
        written in as an expected value). is_intraday=False by the module docstring's
        reasoning: minute INPUT but a DAILY signal traded close-to-close.
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once enough
        VALID days accumulate), not a fixed leading count — the honest NaN rate is
        reported by data_coverage.

        The description spells out the RELATION TO PR-I (same shared classification, same
        VWAP identity, DENOMINATOR swapped from the whole visible day to the ridge bars)
        plus the pinned choices — above all the ASYMMETRIC bar floor, which exists
        because ridge bars are structurally far scarcer than valley bars.

        D1 declarations (D0 pre-assignment table row 8): adjustment=
        returns_invariant — the same same-day two-leg cancellation argument as
        PR-I, with the denominator swapped to the ridge VWAP (module docstring
        PINNED §4); price arrives via Σamount/Σvolume, so requires lists no
        OHLC field. overnight_boundary=none — both legs are same-day.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Valley/ridge VWAP ratio (Kaiyuan microstructure series #27, FOURTH "
                f"factor 谷岭加权价格比). SAME minute classification as PR-F "
                f"volume_peak_count / PR-H peak_interval_kurtosis / PR-I "
                f"valley_relative_vwap (SHARED taxonomy in factors.compute.minute."
                f"primitives, not re-implemented): 1min bars PIT-truncated at 14:50, a "
                f"minute is ERUPTIVE if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of its "
                f"SAME-SLOT strictly-prior {VOLUME_PRV_BASELINE_DAYS}-day baseline, "
                f"else it is a VALLEY (量谷). DENOMINATOR SWAPPED vs PR-I: instead of "
                f"the whole visible day's VWAP the divisor is the RIDGE (量岭) VWAP, so "
                f"the two behavioural groups are contrasted head-on — daily ratio = "
                f"(valley VWAP) / (ridge VWAP), averaged over the trailing "
                f"{self._lookback_days} VALID days. PINNED choices: (1) the RIDGE mask "
                f"is 'eruptive AND NOT an isolated peak', which is WIDER than 'eruptive "
                f"next to an eruptive' — it also covers session-boundary eruptions and "
                f"eruptions with an unclassifiable neighbour, keeping valley|peak|ridge "
                f"an exact partition of the classifiable bars; an isolated PEAK "
                f"contributes to NEITHER leg; (2) each VWAP uses the aggregation "
                f"identity Σ(p·v)/Σv = Σamount/Σvolume; (3) bars with non-finite or "
                f"non-positive volume or amount are dropped from BOTH sums (guard "
                f"applied at summation only, so PR-F's baseline is untouched); (4) RAW "
                f"unadjusted prices are correct here because the adjustment factor is "
                f"constant within a day and cancels in the ratio; (5) DEVIATION FROM "
                f"THE REPORT, disclosed: both legs span the PIT-VISIBLE window "
                f"09:31-14:50 only, not the full session — reading the close would be "
                f"lookahead at our 14:50 decision time; (6) a day is VALID iff it has "
                f">= {VOLUME_PRV_MIN_CLASSIFIABLE} classifiable bars AND >= "
                f"{VALLEY_RIDGE_MIN_VALLEY_BARS} TRADABLE valley bars AND >= "
                f"{VALLEY_RIDGE_MIN_RIDGE_BARS} TRADABLE ridge bars (both counted AFTER "
                f"the guard) AND positive volume in both denominators — the ridge floor "
                f"is deliberately LOWER because a ridge bar must erupt AND fail the "
                f"isolation test, making ridges structurally far scarcer than valleys, "
                f"and the realized ridge-bar distribution plus day-validity rate are "
                f"REPORTED by the runner rather than left implicit; NaN below "
                f"{VOLUME_PRV_MIN_VALID_DAYS} valid days. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("volume", "amount"),
            requires=_minute_requires("volume", "amount"),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily valley/ridge VWAP-ratio column off ``panel``.

        The runner runs ``compute_valley_ridge_vwap_ratio`` per symbol on the minute
        cache upstream and joins the result as ``self.name``; here we only surface it, so
        this factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"ValleyRidgeVwapRatioFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_valley_ridge_vwap_ratio and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "VALLEY_RIDGE_LOOKBACK_DAYS",
    "VALLEY_RIDGE_MIN_RIDGE_BARS",
    "VALLEY_RIDGE_MIN_VALLEY_BARS",
    "ValleyRidgeVwapRatioFactor",
    "compute_valley_ridge_vwap_ratio",
    "valley_ridge_vwap_ratio_by_day",
]
