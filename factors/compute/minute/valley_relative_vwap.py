"""VALLEY-RELATIVE VWAP factor (PR-I): math + surface (D2).

Reproduces the THIRD factor of the Kaiyuan market-microstructure series #27 (开源证券
《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417,
§4): "参考聪明钱因子的构建方式，计算每日量谷的成交量加权价格，与当日总成交量加权价格做比，
并计算 20 日均值作为量谷相对加权价格因子".

Same MACHINE as PR-F / PR-H, different FAMILY. The volume classification is the SHARED
taxonomy in :mod:`factors.compute.minute.primitives` (``prepare_visible_minute_bars`` +
``peak_mask_for_symbol``) — same-slot strictly-prior μ+kσ eruptive/mild test, same
classifiable rule, same valid-day floor. A VALLEY (量谷) is exactly that taxonomy's
``valley`` column: a classifiable, NON-eruptive minute. Nothing about the taxonomy is
re-implemented here, so the family can never drift apart. Where PR-F counts peaks
and PR-H measures the TIMING of peaks, this module measures the PRICE LEVEL at the
valleys.

PINNED choices (deliberate and DISCLOSED, not tuned knobs; reproduced on the factor spec
so a reader sees exactly what was assumed):

  1. VWAP VIA THE AGGREGATION IDENTITY. A bar's volume-weighted price is
     ``p = amount / volume``, so a set's volume-weighted price
     ``Σ(p_i·v_i) / Σv_i`` collapses EXACTLY to ``Σamount / Σvolume``. Both legs use
     that identity: valley VWAP = Σamount(valley bars)/Σvolume(valley bars), day VWAP
     = Σamount(all visible bars)/Σvolume(all visible bars). This is the day's REAL
     VWAP, strictly better than approximating each bar by its close.
  2. POSITIVE-TRADE GUARD. A bar with non-finite or non-positive ``volume`` OR
     ``amount`` carries no price information, so it is dropped from BOTH sums
     (:func:`~factors.compute.minute.primitives.positive_trade_mask`). The guard runs
     at the summation step only — never before classification — because the same-slot
     μ/σ baseline is the shared taxonomy's and must stay bit-identical.
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
VALLEY_VWAP_LOOKBACK_DAYS = 20  # trailing VALID trading-day mean window, includes d
VALLEY_VWAP_MIN_VALLEY_BARS = 20  # min TRADABLE valley bars for a valid day

# The extra 1min column this family needs on top of the taxonomy's (volume): the traded
# value, which turns the bar set into a volume-weighted price via Σamount/Σvolume.
_AMOUNT = "amount"


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def valley_vwap_ratio_by_day(
    work: pd.DataFrame,
    *,
    min_valley_bars: int = VALLEY_VWAP_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> pd.Series:
    """Daily ``valley VWAP / day VWAP`` ratio for ONE symbol, on VALID days only.

    ``work`` is one symbol's frame as returned by
    :func:`~factors.compute.minute.primitives.peak_mask_for_symbol`, which must have been
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
    # before classification — the shared same-slot baseline must stay bit-identical.
    tradable = positive_trade_mask(vol, amt)
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
    ratio = valley_vwap_ratio_by_day(
        work, min_valley_bars=min_valley_bars, min_classifiable=min_classifiable
    )
    if ratio.empty:
        return [], []

    # Mean over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Emitted only on valid days.
    rolled = rolling_valid_days(
        ratio, lookback_days=lookback_days, min_valid_days=min_valid_days
    ).mean()
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
    classifies every visible minute with the SHARED PR-F taxonomy, computes each valid
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
        return empty_factor_series(name)

    # extra_columns=("amount",) is the ONLY difference from the PR-F / PR-H entry
    # points: the traded value rides along on the surviving rows, and the truncation /
    # volume guard / slot assignment are untouched.
    visible = prepare_visible_minute_bars(
        bars, decision_time=decision_time, extra_columns=(_AMOUNT,)
    )
    if visible.empty:
        return empty_factor_series(name)

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
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class ValleyRelativeVwapFactor(Factor):
    """Valley-relative VWAP factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_valley_relative_vwap`); it does NO minute work of its
    own, mirroring its siblings and the value / financial factors that surface an
    enriched column.

    Args:
        lookback_days: trailing VALID trading-day window averaged; part of the factor
            DEFINITION (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"valley_relative_vwap_{VALLEY_VWAP_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = VALLEY_VWAP_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"valley-relative-vwap lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"valley_relative_vwap_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's full-market RankIC is +8.69% (RankICIR 4.44,
        long-short 25.35%/yr, IR 3.04, monthly win rate 79.7% — the strongest factor in
        the report), and its CSI500 sub-domain long-short is 9.94% / IR 1.26, the closest
        comparable to our eval cell. Semantics per the report: valley minutes are moments
        of subdued sentiment where prices are unlikely to have over-reacted, so a HIGH
        relative valley price predicts HIGHER forward returns. The sign is fixed BEFORE
        the run (a validated prototype must reproduce it). NOTE the report is a MONTHLY,
        market-cap + industry neutral full-market series on Wind data while our eval cell
        is CSI500 daily with industry + size neutral, so the report numbers are a LOOSE
        reference only (disclosed, never mislabeled, never written in as an expected
        value). is_intraday=False by the module docstring's reasoning: minute INPUT but a
        DAILY signal traded close-to-close. min_history_bars=0: the warm-up is
        DATA-dependent (a value appears once enough VALID days accumulate), not a fixed
        leading count — the honest NaN rate is reported by data_coverage.

        The description spells out the DIFFERENT FAMILY vs PR-F / PR-H (the same shared
        classification, but a PRICE LEVEL at the VALLEYS rather than a count or timing of
        the PEAKS) plus the pinned choices, including the one place we KNOWINGLY DEVIATE
        from the report: our day VWAP covers the PIT-visible window only.

        D1 declarations (D0 pre-assignment table row 7): adjustment=
        returns_invariant — both VWAP legs are sums over the SAME trading day
        and "a split/dividend adjustment factor is constant within a day, so
        it cancels exactly in the ratio" (module docstring PINNED §3).
        Price information arrives via Σamount/Σvolume, which is why requires
        lists no OHLC field — the anomaly is real and intended.
        overnight_boundary=none — both legs are same-day; nothing crosses the
        overnight boundary.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Valley-relative VWAP (Kaiyuan microstructure series #27, THIRD factor "
                f"量谷相对加权价格). SAME minute classification as PR-F volume_peak_count "
                f"and PR-H peak_interval_kurtosis (SHARED taxonomy in factors.compute."
                f"minute.primitives, not re-implemented): 1min bars PIT-truncated at "
                f"14:50, a minute is ERUPTIVE if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of "
                f"its SAME-SLOT strictly-prior {VOLUME_PRV_BASELINE_DAYS}-day baseline, "
                f"else it is a VALLEY (量谷). DIFFERENT FAMILY: instead of counting or "
                f"timing the PEAKS this factor prices the VALLEYS — daily ratio = "
                f"(valley VWAP) / (whole visible day VWAP), averaged over the trailing "
                f"{self._lookback_days} VALID days. PINNED choices: (1) each VWAP uses "
                f"the aggregation identity Σ(p·v)/Σv = Σamount/Σvolume, the day's REAL "
                f"volume-weighted price rather than a close approximation; (2) bars with "
                f"non-finite or non-positive volume or amount are dropped from BOTH "
                f"sums (guard applied at summation only, so PR-F's baseline is "
                f"untouched); (3) RAW unadjusted prices are correct here because the "
                f"adjustment factor is constant within a day and cancels in the ratio; "
                f"(4) DEVIATION FROM THE REPORT, disclosed: our day VWAP spans the "
                f"PIT-VISIBLE window 09:31-14:50 only, not the full session — reading "
                f"the close would be lookahead at our 14:50 decision time; (5) a day is "
                f"VALID iff it has >= {VOLUME_PRV_MIN_CLASSIFIABLE} classifiable bars "
                f"AND >= {VALLEY_VWAP_MIN_VALLEY_BARS} TRADABLE valley bars (counted "
                f"after the guard) AND positive volume in both denominators; NaN below "
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
        """Select the pre-aggregated daily valley-relative-VWAP column off ``panel``.

        The runner runs ``compute_valley_relative_vwap`` per symbol on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"ValleyRelativeVwapFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_valley_relative_vwap and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "VALLEY_VWAP_LOOKBACK_DAYS",
    "VALLEY_VWAP_MIN_VALLEY_BARS",
    "ValleyRelativeVwapFactor",
    "compute_valley_relative_vwap",
    "valley_vwap_ratio_by_day",
]
