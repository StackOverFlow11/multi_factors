"""D2 re-export shim — the 11 minute-factor surface classes moved house.

Single definition point from D2 on (design v3.2 §6.4): each surface class now
lives WITH its factor math in its own ``factors.compute.minute.{factor}``
module (one factor per file). This shim keeps the pre-D2 import path working
for the eval runners and tests; it is deleted in D6d. Import-only by contract
(locked by the shim purity test).
"""

from factors.compute.minute.amp_marginal_anomaly_vol import AmpMarginalAnomalyVolFactor
from factors.compute.minute.intraday_amp_cut import IntradayAmpCutFactor
from factors.compute.minute.jump_amount_corr import JumpAmountCorrFactor
from factors.compute.minute.minute_ideal_amplitude import MinuteIdealAmplitudeFactor
from factors.compute.minute.peak_interval_kurtosis import PeakIntervalKurtosisFactor
from factors.compute.minute.peak_ridge_amount_ratio import PeakRidgeAmountRatioFactor
from factors.compute.minute.ridge_minute_return import RidgeMinuteReturnFactor
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.compute.minute.valley_relative_vwap import ValleyRelativeVwapFactor
from factors.compute.minute.valley_ridge_vwap_ratio import ValleyRidgeVwapRatioFactor
from factors.compute.minute.volume_peak_count import VolumePeakCountFactor

__all__ = [
    "AmpMarginalAnomalyVolFactor",
    "IntradayAmpCutFactor",
    "JumpAmountCorrFactor",
    "MinuteIdealAmplitudeFactor",
    "PeakIntervalKurtosisFactor",
    "PeakRidgeAmountRatioFactor",
    "RidgeMinuteReturnFactor",
    "ValleyPriceQuantileFactor",
    "ValleyRelativeVwapFactor",
    "ValleyRidgeVwapRatioFactor",
    "VolumePeakCountFactor",
]
