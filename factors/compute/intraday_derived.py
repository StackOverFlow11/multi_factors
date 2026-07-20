"""Daily factors DERIVED from intraday (minute) bars but executed close-to-close.

The members today are :class:`JumpAmountCorrFactor` (PR-C, the Kaiyuan report §6
"price-jump turnover correlation" factor), :class:`MinuteIdealAmplitudeFactor`
(PR-D, the Kaiyuan report §30 "minute ideal amplitude" factor),
:class:`AmpMarginalAnomalyVolFactor` (PR-E, the Changjiang high-frequency-factor
series #19 "amplitude marginal-anomaly relative-volatility" factor) and
:class:`VolumePeakCountFactor` (PR-F, the Kaiyuan microstructure series #27
"volume-peak-minute-count" factor). Like the value / MMP factors, the heavy
computation runs UPSTREAM (``data.clean.intraday_aggregate`` /
``data.clean.intraday_amplitude`` / ``data.clean.intraday_amp_anomaly`` /
``data.clean.intraday_volume_prv`` aggregate 1min bars into a daily
``MultiIndex(date, symbol)`` column) and the Factor here simply SELECTS its column off
the panel the runner already enriched. Keeping the minute aggregation in the data-clean
layer preserves the layering: ``factors`` never fetches and never sees a forward return.

WHY ``is_intraday=False`` FOR A MINUTE-DERIVED FACTOR (deliberate, documented):
    ``FactorSpec.is_intraday`` flags an intraday-EXECUTION contract — the minute
    tail model that DECIDES at 14:50 and FILLS at 14:51, whose holding period runs
    exec(T) -> exec(T_next) (I5a). This factor has minute INPUT but its signal is a
    DAILY value traded at the daily close and held close-to-close (t -> t+1), just
    like the report's monthly rebalance on the daily close and the project's daily
    default. It carries no 14:50 decision cutoff, no execution window, no exec-to-
    exec holding period — so ``is_intraday`` is False, ``return_basis`` is
    ``"close_to_close"``, and the five minute-block spec fields are all None. (The
    base ``FactorSpec`` deliberately does NOT force is_intraday from a minute
    provenance; a daily signal computed from minute data is a legitimate case.)
"""

from __future__ import annotations

import pandas as pd

from data.clean.intraday_aggregate import (
    JUMP_LOOKBACK_DAYS,
    JUMP_MIN_PAIRS,
)
from data.clean.intraday_amp_anomaly import (
    AMP_ANOMALY_FREQ,
    AMP_ANOMALY_LOOKBACK_DAYS,
    AMP_ANOMALY_MIN_POOL,
    AMP_ANOMALY_MIN_SELECTED,
    AMP_ANOMALY_SIGMA_K,
)
from data.clean.intraday_amplitude import (
    IDEAL_AMP_LAMBDA,
    IDEAL_AMP_LOOKBACK_DAYS,
    IDEAL_AMP_MIN_MINUTES,
)
from data.clean.intraday_volume_prv import (
    VOLUME_PRV_BASELINE_DAYS,
    VOLUME_PRV_BASELINE_MIN_OBS,
    VOLUME_PRV_LOOKBACK_DAYS,
    VOLUME_PRV_MIN_CLASSIFIABLE,
    VOLUME_PRV_MIN_VALID_DAYS,
    VOLUME_PRV_SIGMA_K,
)
from factors.base import Factor
from factors.spec import FactorSpec


class JumpAmountCorrFactor(Factor):
    """Price-jump turnover-correlation factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the
    panel (produced by ``compute_jump_amount_corr``); it does NO minute work of its
    own, mirroring the value / financial factors that surface an enriched column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"jump_amount_corr_{JUMP_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = JUMP_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"jump-amount-corr lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"jump_amount_corr_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's RankIC mean is -10.23% (full A, market-cap
        + industry neutral) — high jump-amount-correlation predicts LOWER forward
        returns. The sign is fixed BEFORE the run; a validated prototype reproduced
        it (mean RankIC -0.074 on 2022-2024 sampled names). is_intraday=False by the
        module docstring's reasoning (daily signal traded close-to-close).
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once
        >= ``JUMP_MIN_PAIRS`` jump-pairs accumulate in the trailing window), not a
        fixed leading count — the honest NaN rate is reported by data_coverage
        rather than hidden behind a fabricated warm-up window.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Price-jump turnover correlation (Kaiyuan report §6): trailing "
                f"{self._lookback_days}-trading-day lagged Pearson corr between the "
                f"traded amount at price-JUMP minutes (within-day amplitude z-score "
                f">1) and the amount at the strictly-next minute. Derived from 1min "
                f"bars but a DAILY signal traded close-to-close; >= {JUMP_MIN_PAIRS} "
                f"jump-pairs required else NaN."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. These
            # are declared for honest provenance disclosure (data_coverage lists
            # them); the daily panel surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "open", "amount"),
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily jump-amount-corr column off ``panel``.

        The runner runs ``compute_jump_amount_corr`` on the minute cache upstream
        and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"JumpAmountCorrFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_jump_amount_corr and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


class MinuteIdealAmplitudeFactor(Factor):
    """Minute ideal-amplitude factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_amplitude.compute_minute_ideal_amplitude``);
    it does NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` and
    the value / financial factors that surface an enriched column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the column
            so a non-default window cannot silently mislabel it.
    """

    name: str = f"minute_ideal_amp_{IDEAL_AMP_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = IDEAL_AMP_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"minute-ideal-amplitude lookback_days must be a positive integer; "
                f"got {lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"minute_ideal_amp_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's RankIC mean is -7.6% (full market, N=10,
        lambda=25%) — a HIGH minute ideal amplitude predicts LOWER forward returns.
        The sign is fixed BEFORE the run (a validated prototype must reproduce it).
        is_intraday=False by the module docstring's reasoning: minute INPUT but a
        DAILY signal traded close-to-close. min_history_bars=0: the warm-up is
        DATA-dependent (a value appears once >= ``IDEAL_AMP_MIN_MINUTES`` valid
        pooled minutes accumulate in the trailing window), not a fixed leading
        count — the honest NaN rate is reported by data_coverage.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Minute ideal amplitude (Kaiyuan report §30): pool the 1min bars of "
                f"the trailing {self._lookback_days} trading days (PIT-truncated at "
                f"14:50 per bar), rank the pooled minutes by RAW close, and return "
                f"V_high - V_low where V_high/V_low are the mean per-minute amplitude "
                f"(high/low - 1) of the top / bottom floor({IDEAL_AMP_LAMBDA:g}*n) "
                f"minutes by close. Derived from 1min bars but a DAILY signal traded "
                f"close-to-close; >= {IDEAL_AMP_MIN_MINUTES} valid pooled minutes "
                f"required else NaN."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared
            # for honest provenance disclosure (data_coverage lists them); the daily
            # panel surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "close"),
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily minute-ideal-amplitude column off ``panel``.

        The runner runs ``compute_minute_ideal_amplitude`` on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so
        this factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"MinuteIdealAmplitudeFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_minute_ideal_amplitude and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


class AmpMarginalAnomalyVolFactor(Factor):
    """Amplitude marginal-anomaly relative-volatility factor (daily, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_amp_anomaly.compute_amp_marginal_anomaly_vol``);
    it does NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` /
    :class:`MinuteIdealAmplitudeFactor` and the value / financial factors that surface
    an enriched column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (a pinned interpretation of the report), not a tuned knob. It only names
            the column so a non-default window cannot silently mislabel it.
    """

    name: str = f"amp_marginal_anomaly_vol_{AMP_ANOMALY_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = AMP_ANOMALY_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"amp-marginal-anomaly-vol lookback_days must be a positive integer; "
                f"got {lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"amp_marginal_anomaly_vol_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's IC is POSITIVE across its universes (raw
        CSI800 +4.47% / full-market +4.92%; market-cap + industry neutral +4.11% /
        +5.56%) — a HIGH anomaly-bar relative volatility predicts HIGHER forward
        returns. The sign is fixed BEFORE the run (a validated prototype must
        reproduce it). NOTE the report's sample is CSI800 / full-market on a MONTHLY
        series while our eval cell is CSI500 daily, so the report numbers are a LOOSE
        reference only (disclosed, never mislabeled). is_intraday=False by the module
        docstring's reasoning: minute INPUT but a DAILY signal traded close-to-close.
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once
        >= ``AMP_ANOMALY_MIN_POOL`` valid pooled pairs accumulate in the trailing
        window), not a fixed leading count — the honest NaN rate is reported by
        data_coverage.

        The description spells out the FIVE pinned interpretations of the
        under-specified report (bar freq, lookback, threshold, weighted-vol operator,
        within-day lag) so a reader sees exactly what was assumed.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Amplitude marginal-anomaly relative volatility (Changjiang HF-factor "
                f"series #19). PINNED interpretations of an under-specified report: "
                f"(1) {AMP_ANOMALY_FREQ} bars DERIVED from the 1min cache "
                f"(available_time = max source, PIT-faithful); (2) trailing "
                f"{self._lookback_days} trading days (PIT-truncated at 14:50 per bar); "
                f"(3) select bars with |Δamp| > μ + {AMP_ANOMALY_SIGMA_K:g}σ of the "
                f"pooled |Δamp|; (4) factor = ddof=1 std of the RETURNS on the selected "
                f"bars; (5) Δamp and bar-return are WITHIN-DAY lagged (each day's first "
                f"bar has neither). amp = high/low - 1. Derived from 1min bars but a "
                f"DAILY signal traded close-to-close; >= {AMP_ANOMALY_MIN_POOL} valid "
                f"pooled pairs and >= {AMP_ANOMALY_MIN_SELECTED} selected bars required "
                f"else NaN."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "close"),
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily amp-marginal-anomaly-vol column off ``panel``.

        The runner runs ``compute_amp_marginal_anomaly_vol`` on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"AmpMarginalAnomalyVolFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_amp_marginal_anomaly_vol and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


class VolumePeakCountFactor(Factor):
    """Volume-peak-minute-count factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_volume_prv.compute_volume_peak_count``); it does
    NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` /
    :class:`MinuteIdealAmplitudeFactor` / :class:`AmpMarginalAnomalyVolFactor` and the
    value / financial factors that surface an enriched column.

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
    "AmpMarginalAnomalyVolFactor",
    "JumpAmountCorrFactor",
    "MinuteIdealAmplitudeFactor",
    "VolumePeakCountFactor",
]
