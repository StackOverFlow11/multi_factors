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
from data.clean.intraday_amp_cut import (
    AMP_CUT_LAMBDA,
    AMP_CUT_LOOKBACK_DAYS,
    AMP_CUT_MIN_CROSS_SECTION,
    AMP_CUT_MIN_DAY_MINUTES,
    AMP_CUT_MIN_VALID_DAYS,
)
from data.clean.intraday_amplitude import (
    IDEAL_AMP_LAMBDA,
    IDEAL_AMP_LOOKBACK_DAYS,
    IDEAL_AMP_MIN_MINUTES,
)
from data.clean.intraday_peak_interval import (
    PEAK_INTERVAL_LOOKBACK_DAYS,
    PEAK_INTERVAL_MIN_INTERVALS,
)
from data.clean.intraday_valley_ridge_vwap import (
    VALLEY_RIDGE_LOOKBACK_DAYS,
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
)
from data.clean.intraday_valley_vwap import (
    VALLEY_VWAP_LOOKBACK_DAYS,
    VALLEY_VWAP_MIN_VALLEY_BARS,
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


class IntradayAmpCutFactor(Factor):
    """Intraday amplitude-cut factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_amp_cut.compute_amp_cut_stats`` +
    ``combine_amp_cut_cross_section``); it does NO minute work of its own, mirroring
    :class:`JumpAmountCorrFactor` / :class:`MinuteIdealAmplitudeFactor` /
    :class:`AmpMarginalAnomalyVolFactor` / :class:`VolumePeakCountFactor` and the value /
    financial factors that surface an enriched column.

    Args:
        lookback_days: trailing VALID trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the column so a
            non-default window cannot silently mislabel it.
    """

    name: str = f"intraday_amp_cut_{AMP_CUT_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = AMP_CUT_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"intraday-amp-cut lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"intraday_amp_cut_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's rankIC mean is -0.067 (rankICIR -3.82, quintile
        long-short 16.7%/yr at N=10, lambda=20%, 1-minute-return indicator) — a HIGH
        intraday amplitude-cut value predicts LOWER forward returns; the V_mean and V_std
        sub-factors are each negative too. The sign is fixed BEFORE the run (a validated
        prototype must reproduce it). is_intraday=False by the module docstring's
        reasoning: minute INPUT but a DAILY signal traded close-to-close. min_history_bars=
        0: the warm-up is DATA-dependent (a value appears once >= ``AMP_CUT_MIN_VALID_DAYS``
        valid days accumulate AND the cross-section has >= ``AMP_CUT_MIN_CROSS_SECTION``
        finite pairs), not a fixed leading count — the honest NaN rate is reported by
        data_coverage.

        The description spells out the DISTINCTION FROM PR-D (``minute_ideal_amp``): PR-D
        pools the 10-day minutes into ONE set and cuts by minute CLOSE PRICE; this factor
        cuts EACH DAY by the 1-MINUTE RETURN, then takes the trailing-10-valid-day mean /
        std of the daily cut and combines them cross-sectionally (the report finds the two
        only ~30% correlated).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Intraday amplitude cut (Kaiyuan microstructure series #30, SECOND "
                f"factor 日内振幅切割). DISTINCT FROM PR-D minute_ideal_amp, which pools the "
                f"trailing days' minutes into ONE set and cuts by minute CLOSE PRICE: "
                f"this factor cuts EACH DAY independently by the 1-MINUTE RETURN "
                f"r=close_t/close_{{t-1}}-1 (within-day lagged, first bar of each day has "
                f"no r), taking V_day = V_high - V_low where V_high/V_low are the mean "
                f"amp (high/low-1) of the top/bottom floor({AMP_CUT_LAMBDA:g}*n_day) bars "
                f"by return (day valid iff >= {AMP_CUT_MIN_DAY_MINUTES} valid bars). "
                f"Trailing {self._lookback_days} VALID days give V_mean / V_std (>= "
                f"{AMP_CUT_MIN_VALID_DAYS} valid days else NaN); per date they are each "
                f"cross-sectionally z-scored over the covered universe (>= "
                f"{AMP_CUT_MIN_CROSS_SECTION} finite pairs else NaN) and averaged: factor "
                f"= (z(V_mean) + z(V_std))/2. Derived from 1min bars but a DAILY signal "
                f"traded close-to-close (report finds ~30% corr with minute_ideal_amp)."
            ),
            expected_ic_sign=-1,
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
        """Select the pre-aggregated daily intraday-amp-cut column off ``panel``.

        The runner runs ``compute_amp_cut_stats`` per symbol on the minute cache upstream,
        assembles the full-universe ``(V_mean, V_std)`` panel, applies
        ``combine_amp_cut_cross_section``, and joins the result as ``self.name``; here we
        only surface it, so this factor does no temporal logic and cannot introduce
        lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"IntradayAmpCutFactor needs the pre-aggregated '{self.name}' column on "
                f"the panel (produced upstream by compute_amp_cut_stats + "
                f"combine_amp_cut_cross_section and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


class PeakIntervalKurtosisFactor(Factor):
    """Volume-peak interval-kurtosis factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_peak_interval.compute_peak_interval_kurtosis``);
    it does NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` /
    :class:`MinuteIdealAmplitudeFactor` / :class:`AmpMarginalAnomalyVolFactor` /
    :class:`VolumePeakCountFactor` / :class:`IntradayAmpCutFactor` and the value /
    financial factors that surface an enriched column.

    Args:
        lookback_days: trailing VALID trading-day window pooled for the kurtosis; part of
            the factor DEFINITION (reproduced from the report), not a tuned knob. It only
            names the column so a non-default window cannot silently mislabel it.
    """

    name: str = f"peak_interval_kurtosis_{PEAK_INTERVAL_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = PEAK_INTERVAL_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"peak-interval-kurtosis lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"peak_interval_kurtosis_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=+1: the report's full-market RankIC is +7.19% (RankICIR 4.63,
        long-short 23.3%/yr, IR 3.39, max drawdown 7.37%, 13/13 positive years — the most
        stable factor in the report). A peaky, fat-tailed peak-interval distribution means
        informed trading arrives in BURSTS rather than evenly through the session. The
        sign is fixed BEFORE the run (a validated prototype must reproduce it).
        is_intraday=False for the same reason as the siblings: minute INPUT but a DAILY
        signal traded close-to-close. min_history_bars=0: the warm-up is DATA-dependent (a
        value appears once enough VALID days AND enough pooled intervals accumulate), not
        a fixed leading count — the honest NaN rate is reported by data_coverage.

        The description spells out the DISTINCTION FROM PR-F (``volume_peak_count``): the
        SAME peak identification (reused, not re-implemented) reduced by a DIFFERENT
        statistic — the shape of the gap distribution rather than the peak count — plus
        the two interpretations the report leaves open (interval unit, kurtosis
        convention).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Volume-peak interval kurtosis (Kaiyuan microstructure series #27, "
                f"SECOND factor 量峰间隔峰度). SAME peak identification as PR-F "
                f"volume_peak_count (REUSED from data.clean.intraday_volume_prv, not "
                f"re-implemented): 1min bars PIT-truncated at 14:50, a minute is ERUPTIVE "
                f"if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of its SAME-SLOT strictly-prior "
                f"{VOLUME_PRV_BASELINE_DAYS}-day baseline, and a PEAK is an eruptive "
                f"minute whose both 1-minute same-session neighbours are mild. DIFFERENT "
                f"STATISTIC: the gaps between consecutive same-day peaks, pooled over the "
                f"trailing {self._lookback_days} VALID days (a day with < 2 peaks "
                f"contributes 0 intervals but is still valid), reduced to their kurtosis. "
                f"PINNED interpretations of an under-specified report: (1) an interval is "
                f"measured in TRADING MINUTES — the tradable-slot difference inside the "
                f"day's visible bar sequence, so the lunch break costs nothing (11:29 and "
                f"13:02 peaks are 3 apart, not 93) and a wall-clock ~90-minute spike can "
                f"never dominate the distribution; (2) kurtosis = FISHER excess, "
                f"bias-corrected (the pandas .kurt() / scipy fisher=True bias=False "
                f"convention; normal = 0). NaN unless >= "
                f"{PEAK_INTERVAL_MIN_INTERVALS} intervals pooled and the pool has "
                f"non-zero variance. Derived from 1min bars but a DAILY signal traded "
                f"close-to-close."
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
        """Select the pre-aggregated daily peak-interval-kurtosis column off ``panel``.

        The runner runs ``compute_peak_interval_kurtosis`` per symbol on the minute cache
        upstream and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"PeakIntervalKurtosisFactor needs the pre-aggregated '{self.name}' "
                f"column on the panel (produced upstream by "
                f"compute_peak_interval_kurtosis and joined by the runner); panel has "
                f"{list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


class ValleyRelativeVwapFactor(Factor):
    """Valley-relative VWAP factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by ``data.clean.intraday_valley_vwap.compute_valley_relative_vwap``); it
    does NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` /
    :class:`MinuteIdealAmplitudeFactor` / :class:`AmpMarginalAnomalyVolFactor` /
    :class:`VolumePeakCountFactor` / :class:`IntradayAmpCutFactor` /
    :class:`PeakIntervalKurtosisFactor` and the value / financial factors that surface an
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

        The description spells out the DIFFERENT FAMILY vs PR-F / PR-H (the same reused
        classification, but a PRICE LEVEL at the VALLEYS rather than a count or timing of
        the PEAKS) plus the pinned choices, including the one place we KNOWINGLY DEVIATE
        from the report: our day VWAP covers the PIT-visible window only.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Valley-relative VWAP (Kaiyuan microstructure series #27, THIRD factor "
                f"量谷相对加权价格). SAME minute classification as PR-F volume_peak_count "
                f"and PR-H peak_interval_kurtosis (REUSED from "
                f"data.clean.intraday_volume_prv, not re-implemented): 1min bars "
                f"PIT-truncated at 14:50, a minute is ERUPTIVE if vol > μ + "
                f"{VOLUME_PRV_SIGMA_K:g}σ of its SAME-SLOT strictly-prior "
                f"{VOLUME_PRV_BASELINE_DAYS}-day baseline, else it is a VALLEY (量谷). "
                f"DIFFERENT FAMILY: instead of counting or timing the PEAKS this factor "
                f"prices the VALLEYS — daily ratio = (valley VWAP) / (whole visible day "
                f"VWAP), averaged over the trailing {self._lookback_days} VALID days. "
                f"PINNED choices: (1) each VWAP uses the aggregation identity Σ(p·v)/Σv "
                f"= Σamount/Σvolume, the day's REAL volume-weighted price rather than a "
                f"close approximation; (2) bars with non-finite or non-positive volume "
                f"or amount are dropped from BOTH sums (guard applied at summation only, "
                f"so PR-F's baseline is untouched); (3) RAW unadjusted prices are correct "
                f"here because the adjustment factor is constant within a day and cancels "
                f"in the ratio; (4) DEVIATION FROM THE REPORT, disclosed: our day VWAP "
                f"spans the PIT-VISIBLE window 09:31-14:50 only, not the full session — "
                f"reading the close would be lookahead at our 14:50 decision time; "
                f"(5) a day is VALID iff it has >= {VOLUME_PRV_MIN_CLASSIFIABLE} "
                f"classifiable bars AND >= {VALLEY_VWAP_MIN_VALLEY_BARS} TRADABLE valley "
                f"bars (counted after the guard) AND positive volume in both "
                f"denominators; NaN below {VOLUME_PRV_MIN_VALID_DAYS} valid days. Derived "
                f"from 1min bars but a DAILY signal traded close-to-close."
            ),
            expected_ic_sign=1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. Declared for
            # honest provenance disclosure (data_coverage lists them); the daily panel
            # surfaces the pre-aggregated column itself.
            input_fields=("volume", "amount"),
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


class ValleyRidgeVwapRatioFactor(Factor):
    """Valley/ridge VWAP-ratio factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the panel
    (produced by
    ``data.clean.intraday_valley_ridge_vwap.compute_valley_ridge_vwap_ratio``); it does
    NO minute work of its own, mirroring :class:`JumpAmountCorrFactor` /
    :class:`MinuteIdealAmplitudeFactor` / :class:`AmpMarginalAnomalyVolFactor` /
    :class:`VolumePeakCountFactor` / :class:`IntradayAmpCutFactor` /
    :class:`PeakIntervalKurtosisFactor` / :class:`ValleyRelativeVwapFactor` and the value
    / financial factors that surface an enriched column.

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

        The description spells out the RELATION TO PR-I (same reused classification, same
        VWAP identity, DENOMINATOR swapped from the whole visible day to the ridge bars)
        plus the pinned choices — above all the ASYMMETRIC bar floor, which exists
        because ridge bars are structurally far scarcer than valley bars.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Valley/ridge VWAP ratio (Kaiyuan microstructure series #27, FOURTH "
                f"factor 谷岭加权价格比). SAME minute classification as PR-F "
                f"volume_peak_count / PR-H peak_interval_kurtosis / PR-I "
                f"valley_relative_vwap (REUSED from data.clean.intraday_volume_prv, not "
                f"re-implemented): 1min bars PIT-truncated at 14:50, a minute is ERUPTIVE "
                f"if vol > μ + {VOLUME_PRV_SIGMA_K:g}σ of its SAME-SLOT strictly-prior "
                f"{VOLUME_PRV_BASELINE_DAYS}-day baseline, else it is a VALLEY (量谷). "
                f"DENOMINATOR SWAPPED vs PR-I: instead of the whole visible day's VWAP "
                f"the divisor is the RIDGE (量岭) VWAP, so the two behavioural groups are "
                f"contrasted head-on — daily ratio = (valley VWAP) / (ridge VWAP), "
                f"averaged over the trailing {self._lookback_days} VALID days. PINNED "
                f"choices: (1) the RIDGE mask is 'eruptive AND NOT an isolated peak', "
                f"which is WIDER than 'eruptive next to an eruptive' — it also covers "
                f"session-boundary eruptions and eruptions with an unclassifiable "
                f"neighbour, keeping valley|peak|ridge an exact partition of the "
                f"classifiable bars; an isolated PEAK contributes to NEITHER leg; "
                f"(2) each VWAP uses the aggregation identity Σ(p·v)/Σv = Σamount/Σvolume; "
                f"(3) bars with non-finite or non-positive volume or amount are dropped "
                f"from BOTH sums (guard applied at summation only, so PR-F's baseline is "
                f"untouched); (4) RAW unadjusted prices are correct here because the "
                f"adjustment factor is constant within a day and cancels in the ratio; "
                f"(5) DEVIATION FROM THE REPORT, disclosed: both legs span the "
                f"PIT-VISIBLE window 09:31-14:50 only, not the full session — reading the "
                f"close would be lookahead at our 14:50 decision time; (6) a day is VALID "
                f"iff it has >= {VOLUME_PRV_MIN_CLASSIFIABLE} classifiable bars AND >= "
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
    "AmpMarginalAnomalyVolFactor",
    "IntradayAmpCutFactor",
    "JumpAmountCorrFactor",
    "MinuteIdealAmplitudeFactor",
    "PeakIntervalKurtosisFactor",
    "ValleyRelativeVwapFactor",
    "ValleyRidgeVwapRatioFactor",
    "VolumePeakCountFactor",
]
