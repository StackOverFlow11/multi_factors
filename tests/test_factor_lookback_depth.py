"""D4 Commit 1: the 14 closing factors' pre-registered ``lookback_depth``.

The D3 tail-recompute engine HARD-REQUIRES ``spec.lookback_depth`` (a readable
error, never a silent 0/headline fallback) to size its incremental overlap
window. This test pins the pre-registered transitive lookback depth of every
closing-set factor to a specific value derived from its CODE CONSTANTS (design
§六.18: headline + nested baseline, NOT the headline window).

Pinning is the point: once declared, the value is a pre-registration — changing
it is a definition-adjacent change that must be disclosed separately. A wrong
declaration under-samples the overlap and produces FALSE mismatches every night
(the D3 engine already has teeth for that); this test is the up-front guard.

The parameterization also proves the declaration is DERIVED from the constructor
param, not a frozen literal: a non-default window/lookback yields the matching
depth (so a re-parameterized factor cannot silently keep the default depth).
"""

from __future__ import annotations

import pytest

from factors import registry as factor_registry
from factors.compute.candidates import ValueFactor, VolatilityFactor
from factors.compute.financial import FinancialFactor
from factors.compute.minute.amp_marginal_anomaly_vol import AmpMarginalAnomalyVolFactor
from factors.compute.minute.intraday_amp_cut import IntradayAmpCutFactor
from factors.compute.minute.jump_amount_corr import JumpAmountCorrFactor
from factors.compute.minute.minute_ideal_amplitude import MinuteIdealAmplitudeFactor
from factors.compute.minute.peak_interval_kurtosis import PeakIntervalKurtosisFactor
from factors.compute.minute.peak_ridge_amount_ratio import PeakRidgeAmountRatioFactor
from factors.compute.minute.primitives import VOLUME_PRV_BASELINE_DAYS
from factors.compute.minute.ridge_minute_return import RidgeMinuteReturnFactor
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.compute.minute.valley_relative_vwap import ValleyRelativeVwapFactor
from factors.compute.minute.valley_ridge_vwap_ratio import ValleyRidgeVwapRatioFactor
from factors.compute.minute.volume_peak_count import VolumePeakCountFactor
from factors.compute.momentum import MomentumFactor

# The 14 closing factors: (factor instance, expected lookback_depth).
# 3 book factors + 11 minute-derived factors. Baseline (nested-lookback) minute
# factors = lookback_days (20) + VOLUME_PRV_BASELINE_DAYS (20) = 40.
_BASELINE = VOLUME_PRV_BASELINE_DAYS  # 20
_CLOSING_FACTORS = [
    # -- book factors -----------------------------------------------------
    (ValueFactor("value_ep"), 1),
    (ValueFactor("value_bp"), 1),
    (VolatilityFactor(window=20), 21),  # window + 1 (pct_change adds a day)
    # -- minute factors, no nested baseline -------------------------------
    (JumpAmountCorrFactor(), 20),
    (MinuteIdealAmplitudeFactor(), 10),
    (AmpMarginalAnomalyVolFactor(), 20),
    (IntradayAmpCutFactor(), 10),
    # -- minute factors, nested baseline (lookback 20 + baseline 20) ------
    (VolumePeakCountFactor(), 20 + _BASELINE),
    (PeakIntervalKurtosisFactor(), 20 + _BASELINE),
    (ValleyRelativeVwapFactor(), 20 + _BASELINE),
    (ValleyRidgeVwapRatioFactor(), 20 + _BASELINE),
    (RidgeMinuteReturnFactor(), 20 + _BASELINE),
    (ValleyPriceQuantileFactor(), 20 + _BASELINE),
    (PeakRidgeAmountRatioFactor(), 20 + _BASELINE),
]


@pytest.mark.parametrize(
    "factor, expected",
    _CLOSING_FACTORS,
    ids=[f.name for f, _ in _CLOSING_FACTORS],
)
def test_closing_factor_lookback_depth(factor, expected):
    """Each closing factor declares its exact pre-registered transitive depth."""
    depth = factor.spec.lookback_depth
    assert depth is not None, (
        f"{factor.name}: lookback_depth is None — the store engine would refuse "
        f"to size its overlap window (D3 hard-require)."
    )
    assert depth == expected, f"{factor.name}: lookback_depth {depth} != {expected}"
    # It must be a positive int (contract v1.1 validates this at construction).
    assert isinstance(depth, int) and not isinstance(depth, bool) and depth >= 1


def test_closing_factors_count_is_fourteen():
    """The pinned table is exactly the 14-factor closing set (no drift)."""
    assert len(_CLOSING_FACTORS) == 14


@pytest.mark.parametrize(
    "factor, expected",
    [
        # window-parameterized: depth tracks the constructor param, not a literal.
        (VolatilityFactor(window=10), 11),
        (MomentumFactor(window=30), 31),
        (JumpAmountCorrFactor(lookback_days=30), 30),
        (MinuteIdealAmplitudeFactor(lookback_days=15), 15),
        (VolumePeakCountFactor(lookback_days=15), 15 + _BASELINE),
        (ValleyRidgeVwapRatioFactor(lookback_days=25), 25 + _BASELINE),
    ],
)
def test_lookback_depth_tracks_the_constructor_param(factor, expected):
    """A non-default window/lookback yields the matching depth (derived, not frozen)."""
    assert factor.spec.lookback_depth == expected


def test_non_closing_daily_factors_also_declare_depth():
    """The remaining registered factors declare a depth too (no latent store break)."""
    from factors.compute.candidates import (
        LiquidityFactor,
        OvernightMomentumFactor,
        ReversalFactor,
    )

    assert MomentumFactor(window=20).spec.lookback_depth == 21
    assert ReversalFactor(window=20).spec.lookback_depth == 21  # inherits momentum
    assert LiquidityFactor(window=20).spec.lookback_depth == 20
    assert OvernightMomentumFactor(window=20).spec.lookback_depth == 21
    assert FinancialFactor("roe").spec.lookback_depth == 1


def test_registry_built_factor_carries_the_same_depth():
    """Building through the registry (the D4 service path) yields the same depth."""
    built = factor_registry.build("jump_amount_corr_20")
    assert built.spec.lookback_depth == 20
    built_vol = factor_registry.build("volatility_20")
    assert built_vol.spec.lookback_depth == 21
