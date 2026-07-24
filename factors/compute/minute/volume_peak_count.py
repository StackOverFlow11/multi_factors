"""Volume-peak-count factor (PR-F): math + surface (D2 one-factor-one-file).

Reproduces the Kaiyuan market-microstructure series #27 (开源证券《高频成交量的峰、岭、
谷信息——市场微观结构研究系列（27）》, 2025-07-20, reportId 4957417) flagship
"volume-peak-minute-count" factor (§2) as a daily PIT-safe column derived DIRECTLY
from the 1min cache (no coarser resampling — the peak/ridge/valley taxonomy lives at
the 1-minute grain). The taxonomy itself (``prepare_visible_minute_bars`` +
``peak_mask_for_symbol``) lives in :mod:`factors.compute.minute.primitives`; this
file reduces it to the peak COUNT.

The report is under-specified about the PIT boundary; the interpretations below are
deliberate, DISCLOSED choices PINNED in the task card (task_card_pr_f_*.md §1), not
tuned knobs. They are reproduced on the factor spec so a reader can see exactly what
was assumed:

  1. PIT truncation (standing authorization): each day keeps only the 1min bars with
     ``available_time <= (that bar's trade_date + decision_time)`` (default 14:50) —
     history days AND the signal day are truncated identically, so the same-slot
     cross-day baseline is measured on a consistent window (≈ the morning session plus
     the afternoon up to the cutoff).
  2. Same-slot baseline (PINNED as STRICTLY PRIOR — the report does not say whether
     the current day is included, and strictly-prior is the more PIT-stable reading):
     for minute slot ``s`` and day ``t`` the baseline ``μ_s`` / ``σ_s`` (ddof=1) is
     the symbol's SAME-SLOT volume over the trailing ``baseline_days`` (=20) trading
     days STRICTLY BEFORE ``t``; fewer than ``baseline_min_obs`` (=10) same-slot
     observations in that window -> the ``(t, s)`` bar is NOT classifiable.
  3. classify: ``vol > μ_s + k*σ_s`` (k = 1) -> ERUPTIVE, else MILD (a "valley").
  4. a slot is a PEAK iff it is eruptive AND both its 1-minute neighbours in the SAME
     continuous session exist and are MILD (see ``peak_mask_for_symbol``).

Factor value: ``volume_peak_count_20`` = the total count of peak minutes over the
symbol's most recent ``lookback_days`` (=20) VALID trading days INCLUDING ``d`` (a
simple count — after truncation the per-day slot count is uniform, so counts are
cross-sectionally comparable without normalization). A day is VALID iff it has at
least ``min_classifiable`` (=100) classifiable bars; fewer than ``min_valid_days``
(=10) valid days in the trailing window -> NaN (honest missing; the runner discloses
the coverage). Values are emitted only on valid days.

Pre-registered sign = +1 (more volume peaks = more informed-trading participation =
higher future returns; the report's full-market RankIC is +10.62% / RankICIR 4.36 and
its CSI500 sub-domain long-short is +14.96%/yr — the CSI500 line is the direct anchor
for our eval cell). NOTE the report is a monthly, market-cap + industry neutral series
on Wind data; our eval cell is CSI500 daily with industry + size neutral, so the
report numbers are a LOOSE reference only (disclosed, never mislabeled). Raw minute
volume (cached as-is) has magnitude jumps across split days that pollute the 20-day σ;
the report (Wind) does not adjust for this either, so we disclose it and do NOT correct
it. The value at ``(d, s)`` uses only bars at dates <= d, so a factor value never sees
a future bar (invariant #1); it is a DAILY signal traded close-to-close from d+1.

The factor math is pure and never fetches; the ``VolumePeakCountFactor`` surface
class only SELECTS the pre-aggregated column off the enriched panel.
"""

from __future__ import annotations

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
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constant (pinned interpretation of the report; NOT a tuned knob).
# The classification constants live with the taxonomy in ``primitives``.
VOLUME_PRV_LOOKBACK_DAYS = 20  # trailing VALID trading-day count window (N), includes d


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def _peak_count_for_symbol(
    g: pd.DataFrame,
    *,
    baseline_days: int,
    baseline_min_obs: int,
    sigma_k: float,
    lookback_days: int,
    min_valid_days: int,
    min_classifiable: int,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Daily volume-peak-count values for ONE symbol from its PIT-visible bars.

    Identifies the peaks with the shared :func:`peak_mask_for_symbol` and reduces them to
    the trailing-``lookback_days``-VALID-day count.
    """
    work = peak_mask_for_symbol(
        g,
        baseline_days=baseline_days,
        baseline_min_obs=baseline_min_obs,
        sigma_k=sigma_k,
    )

    peak_count = work.groupby("trade_date")["peak"].sum()
    classifiable_count = work.groupby("trade_date")["classifiable"].sum()
    valid_days = classifiable_count.index[classifiable_count >= min_classifiable]
    if len(valid_days) == 0:
        return [], []

    # Count over the trailing lookback_days VALID days (including d); NaN until
    # min_valid_days valid days have accumulated. Values are emitted only on valid days.
    pc_valid = peak_count.loc[valid_days].astype(float).sort_index()
    factor_valid = pc_valid.rolling(lookback_days, min_periods=min_valid_days).sum()
    days = [pd.Timestamp(d).normalize() for d in factor_valid.index]
    return days, list(factor_valid.to_numpy(dtype=float))


def compute_volume_peak_count(
    bars: pd.DataFrame,
    *,
    lookback_days: int = VOLUME_PRV_LOOKBACK_DAYS,
    baseline_days: int = VOLUME_PRV_BASELINE_DAYS,
    baseline_min_obs: int = VOLUME_PRV_BASELINE_MIN_OBS,
    sigma_k: float = VOLUME_PRV_SIGMA_K,
    min_valid_days: int = VOLUME_PRV_MIN_VALID_DAYS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    decision_time: str = DEFAULT_DECISION_TIME,
    name: str = "volume_peak_count",
) -> pd.Series:
    """PIT-safe daily "volume-peak-minute-count" factor from 1min ``bars``.

    Takes normalized 1min ``bars``, PIT-truncates each day at ``decision_time``,
    classifies every visible minute against its SAME-SLOT strictly-prior baseline
    (eruptive vs mild), marks the eruptive minutes whose both 1-minute same-session
    neighbours are mild as PEAKS, and returns the trailing-``lookback_days``-VALID-day
    peak count. See the module docstring for the LOCKED definition.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the grouping
            is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing VALID trading-day window for the count (definition).
        baseline_days: strictly-prior same-slot baseline window in trading days.
        baseline_min_obs: minimum same-slot observations for a classifiable bar.
        sigma_k: eruptive threshold multiplier ``k`` in ``vol > μ + k*σ``.
        min_valid_days: minimum valid days in the trailing window for a finite value.
        min_classifiable: a day is VALID iff it has at least this many classifiable bars.
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
    if len(bars) == 0:
        return empty_factor_series(name)

    visible = prepare_visible_minute_bars(bars, decision_time=decision_time)
    if visible.empty:
        return empty_factor_series(name)

    index_tuples: list[tuple] = []
    values: list[float] = []
    for sym, g in visible.groupby(SYMBOL_LEVEL, sort=True):
        days, vals = _peak_count_for_symbol(
            g.reset_index(drop=True),
            baseline_days=baseline_days,
            baseline_min_obs=baseline_min_obs,
            sigma_k=sigma_k,
            lookback_days=lookback_days,
            min_valid_days=min_valid_days,
            min_classifiable=min_classifiable,
        )
        for day, val in zip(days, vals):
            index_tuples.append((day, str(sym)))
            values.append(val)

    if not index_tuples:
        return empty_factor_series(name)
    index = pd.MultiIndex.from_tuples(index_tuples, names=DAILY_INDEX_NAMES)
    return pd.Series(values, index=index, name=name).sort_index()


class VolumePeakCountFactor(Factor):
    """Volume-peak-minute-count factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by :func:`compute_volume_peak_count`); it does NO minute work of its
    own, mirroring the value / financial factors that surface an enriched column.

    Args:
        lookback_days: trailing VALID trading-day count window; part of the factor
            DEFINITION (a pinned interpretation of the report), not a tuned knob. It
            only names the column so a non-default window cannot silently mislabel it.
    """

    name: str = f"volume_peak_count_{VOLUME_PRV_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = VOLUME_PRV_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"volume-peak-count lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"volume_peak_count_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's IC is POSITIVE (full-market RankIC +10.62% /
        RankICIR 4.36; CSI500 sub-domain long-short +14.96%/yr) — more volume peaks
        (informed-trading participation) predicts HIGHER forward returns. The sign is
        fixed BEFORE the run (a validated prototype must reproduce it). NOTE the report
        is a MONTHLY, market-cap + industry neutral series on Wind data while our eval
        cell is CSI500 daily with industry + size neutral, so the report numbers are a
        LOOSE reference only (disclosed, never mislabeled). is_intraday=False by the
        module docstring's reasoning: minute INPUT but a DAILY signal traded
        close-to-close. min_history_bars=0: the warm-up is DATA-dependent (a value
        appears once >= ``VOLUME_PRV_MIN_VALID_DAYS`` valid days accumulate in the
        trailing window, and a day is valid only once its same-slot baselines fill in),
        not a fixed leading count — the honest NaN rate is reported by data_coverage.

        The description spells out the pinned interpretations of the under-specified
        report (PIT truncation, strictly-prior same-slot baseline, μ+σ eruptive
        threshold, mild-neighbour peak rule, valid-day gate) so a reader sees exactly
        what was assumed.

        D1 declarations (D0 pre-assignment table row 4 + note 3): adjustment=
        none — the factor reads the pure volume channel, never a price
        (this module's docstring; taxonomy in factors/compute/minute/
        primitives.py). overnight_boundary=none — no raw-price comparison
        exists. The KNOWN raw-volume magnitude jump across split days is
        disclosed at the definition site and deliberately NOT corrected
        (report alignment); per D0 note 3 it belongs to NEITHER taxonomy
        axis — do not "fix" it via these declarations.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Volume-peak-minute count (Kaiyuan microstructure series #27). PINNED "
                f"interpretations of an under-specified report: (1) 1min bars "
                f"PIT-truncated at 14:50 per bar; (2) same-slot baseline = μ/σ (ddof=1) "
                f"of the STRICTLY-PRIOR {VOLUME_PRV_BASELINE_DAYS} trading days' "
                f"same-slot volume, needing >= {VOLUME_PRV_BASELINE_MIN_OBS} obs else "
                f"unclassifiable; (3) a minute is ERUPTIVE if vol > μ + "
                f"{VOLUME_PRV_SIGMA_K:g}σ else MILD; (4) a PEAK is an eruptive minute "
                f"whose both 1-minute same-session neighbours exist and are mild "
                f"(ridge / session-boundary / unclassifiable-neighbour minutes are not "
                f"peaks); (5) factor = peak-minute count over the trailing "
                f"{self._lookback_days} VALID days (>= {VOLUME_PRV_MIN_CLASSIFIABLE} "
                f"classifiable bars) including d, NaN below "
                f"{VOLUME_PRV_MIN_VALID_DAYS} valid days. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar field the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists it); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("volume",),
            requires=_minute_requires("volume"),
            adjustment="none",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily volume-peak-count column off ``panel``.

        The runner runs ``compute_volume_peak_count`` on the minute cache upstream and
        joins the result as ``self.name``; here we only surface it, so this factor does
        no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"VolumePeakCountFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_volume_peak_count and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "VOLUME_PRV_LOOKBACK_DAYS",
    "VolumePeakCountFactor",
    "compute_volume_peak_count",
]
