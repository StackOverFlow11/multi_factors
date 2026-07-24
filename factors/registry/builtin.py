"""Explicit builtin registration list (design §6 pit #3: NO pkgutil discovery).

Importing this module registers every shipped factor into the default
registry. The list is EXPLICIT on purpose: auto-discovery would turn an
import-order accident into a dispatch-table change, and the whole point of
red line #4 is that this file is the ONE place the name -> class mapping
lives. Add a factor = add its import and its ``register`` call here.

Registration order is the retired ``_build_factors`` chain's order (financial
and value exact names first, then the name-family prefixes), preserving its
dispatch semantics verbatim: exact names always won over prefixes there
(membership tests preceded ``startswith``), and the family prefixes are
mutually non-overlapping (the registry enforces that at registration time).
Builders reproduce the chain's params coercion EXACTLY — same ``int(...)`` /
``str(...)`` calls, same defaults — so a config that built a factor before D1
builds a byte-identical factor after it.

The 11 minute-derived surface factors are registered too (their families were
never in the chain — eval runners construct them directly today), so the
registry covers the FULL factor surface the D4 service will serve; their
builders follow the same window-named convention with ``lookback_days``.
"""

from __future__ import annotations

from collections.abc import Mapping

from factors.compute.candidates import (
    VALUE_FIELDS,
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import SUPPORTED_FIELDS, FinancialFactor
from factors.compute.minute.amp_marginal_anomaly_vol import (
    AMP_ANOMALY_LOOKBACK_DAYS,
    AmpMarginalAnomalyVolFactor,
)
from factors.compute.minute.intraday_amp_cut import (
    AMP_CUT_LOOKBACK_DAYS,
    IntradayAmpCutFactor,
)
from factors.compute.minute.jump_amount_corr import (
    JUMP_LOOKBACK_DAYS,
    JumpAmountCorrFactor,
)
from factors.compute.minute.minute_ideal_amplitude import (
    IDEAL_AMP_LOOKBACK_DAYS,
    MinuteIdealAmplitudeFactor,
)
from factors.compute.minute.peak_interval_kurtosis import (
    PEAK_INTERVAL_LOOKBACK_DAYS,
    PeakIntervalKurtosisFactor,
)
from factors.compute.minute.peak_ridge_amount_ratio import (
    PEAK_RIDGE_LOOKBACK_DAYS,
    PeakRidgeAmountRatioFactor,
)
from factors.compute.minute.ridge_minute_return import (
    RIDGE_RETURN_LOOKBACK_DAYS,
    RidgeMinuteReturnFactor,
)
from factors.compute.minute.valley_price_quantile import (
    VALLEY_QUANTILE_LOOKBACK_DAYS,
    ValleyPriceQuantileFactor,
)
from factors.compute.minute.valley_relative_vwap import (
    VALLEY_VWAP_LOOKBACK_DAYS,
    ValleyRelativeVwapFactor,
)
from factors.compute.minute.valley_ridge_vwap_ratio import (
    VALLEY_RIDGE_LOOKBACK_DAYS,
    ValleyRidgeVwapRatioFactor,
)
from factors.compute.minute.volume_peak_count import (
    VOLUME_PRV_LOOKBACK_DAYS,
    VolumePeakCountFactor,
)
from factors.compute.momentum import MomentumFactor
from factors.registry.registry import register

Params = Mapping[str, object]


def _window(params: Params) -> int:
    """The chain's exact window coercion: ``int(params.get("window", 20))``."""
    return int(params.get("window", 20))  # type: ignore[arg-type]


# -- exact names (the chain's membership branches, in chain order) ---------

register(
    FinancialFactor,
    exact=SUPPORTED_FIELDS,
    builder=lambda name, params: FinancialFactor(field=name),
)
register(
    ValueFactor,
    exact=VALUE_FIELDS,
    builder=lambda name, params: ValueFactor(name),
)

# -- name-family prefixes (the chain's startswith branches, in chain order) -

register(
    OvernightMomentumFactor,
    prefix="overnight_mom",
    builder=lambda name, params: OvernightMomentumFactor(
        window=_window(params),
        open_col=str(params.get("open_col", "open")),
        close_col=str(params.get("close_col", "close")),
    ),
)
register(
    MomentumFactor,
    prefix="momentum",
    builder=lambda name, params: MomentumFactor(
        window=_window(params), price_col=str(params.get("price_col", "close"))
    ),
)
register(
    ReversalFactor,
    prefix="reversal",
    builder=lambda name, params: ReversalFactor(
        window=_window(params), price_col=str(params.get("price_col", "close"))
    ),
)
register(
    VolatilityFactor,
    prefix="volatility",
    builder=lambda name, params: VolatilityFactor(
        window=_window(params), price_col=str(params.get("price_col", "close"))
    ),
)
register(
    LiquidityFactor,
    prefix="liquidity",
    builder=lambda name, params: LiquidityFactor(
        window=_window(params), amount_col=str(params.get("amount_col", "amount"))
    ),
)

# -- minute-derived surface factors (never in the chain; registered so the --
# -- registry covers the full factor surface for D4) ------------------------


def _lookback_builder(cls, default_days: int):
    """Builder for the ``{family}_{lookback_days}`` minute-derived factors."""

    def _build(name: str, params: Params):
        return cls(lookback_days=int(params.get("lookback_days", default_days)))  # type: ignore[arg-type]

    return _build


register(
    JumpAmountCorrFactor,
    prefix="jump_amount_corr",
    builder=_lookback_builder(JumpAmountCorrFactor, JUMP_LOOKBACK_DAYS),
)
register(
    MinuteIdealAmplitudeFactor,
    prefix="minute_ideal_amp",
    builder=_lookback_builder(MinuteIdealAmplitudeFactor, IDEAL_AMP_LOOKBACK_DAYS),
)
register(
    AmpMarginalAnomalyVolFactor,
    prefix="amp_marginal_anomaly_vol",
    builder=_lookback_builder(AmpMarginalAnomalyVolFactor, AMP_ANOMALY_LOOKBACK_DAYS),
)
register(
    VolumePeakCountFactor,
    prefix="volume_peak_count",
    builder=_lookback_builder(VolumePeakCountFactor, VOLUME_PRV_LOOKBACK_DAYS),
)
register(
    IntradayAmpCutFactor,
    prefix="intraday_amp_cut",
    builder=_lookback_builder(IntradayAmpCutFactor, AMP_CUT_LOOKBACK_DAYS),
)
register(
    PeakIntervalKurtosisFactor,
    prefix="peak_interval_kurtosis",
    builder=_lookback_builder(PeakIntervalKurtosisFactor, PEAK_INTERVAL_LOOKBACK_DAYS),
)
register(
    ValleyRelativeVwapFactor,
    prefix="valley_relative_vwap",
    builder=_lookback_builder(ValleyRelativeVwapFactor, VALLEY_VWAP_LOOKBACK_DAYS),
)
register(
    ValleyRidgeVwapRatioFactor,
    prefix="valley_ridge_vwap_ratio",
    builder=_lookback_builder(ValleyRidgeVwapRatioFactor, VALLEY_RIDGE_LOOKBACK_DAYS),
)
register(
    RidgeMinuteReturnFactor,
    prefix="ridge_minute_return",
    builder=_lookback_builder(RidgeMinuteReturnFactor, RIDGE_RETURN_LOOKBACK_DAYS),
)
register(
    ValleyPriceQuantileFactor,
    prefix="valley_price_quantile",
    builder=_lookback_builder(ValleyPriceQuantileFactor, VALLEY_QUANTILE_LOOKBACK_DAYS),
)
register(
    PeakRidgeAmountRatioFactor,
    prefix="peak_ridge_amount_ratio",
    builder=_lookback_builder(PeakRidgeAmountRatioFactor, PEAK_RIDGE_LOOKBACK_DAYS),
)
