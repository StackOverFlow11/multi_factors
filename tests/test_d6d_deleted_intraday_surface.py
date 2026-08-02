"""D6d anti-resurrection lock: the deleted intraday surface stays deleted.

The D2 shims (ten ``data/clean/intraday_*`` factor re-export modules +
``factors/compute/intraday_derived``) and the aggregate's factor-math surface
(the ``factors.compute.minute`` re-exports + the ``mmp_ew`` feature hook) are
DELETED. This file locks them out:

1. every deleted module name is NOT importable;
2. ``data.clean.intraday_aggregate`` exports no factor math (``hasattr`` per
   name, plus ``mmp_ew`` absent from ``INTRADAY_FEATURE_KEYS`` and the
   ``epsilon`` parameter gone from ``asof_daily_features``);
3. anti-vacuity — the same probe DOES import a module that exists and DOES
   find a name that exists, so the negative assertions above are not vacuously
   green (e.g. from a broken ``importlib`` usage).

Network-free.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

_DELETED_MODULE_NAMES = [
    "data.clean.intraday_amount_ratio",
    "data.clean.intraday_amp_anomaly",
    "data.clean.intraday_amp_cut",
    "data.clean.intraday_amplitude",
    "data.clean.intraday_peak_interval",
    "data.clean.intraday_ridge_return",
    "data.clean.intraday_valley_quantile",
    "data.clean.intraday_valley_ridge_vwap",
    "data.clean.intraday_valley_vwap",
    "data.clean.intraday_volume_prv",
    "factors.compute.intraday_derived",
]

#: The factor-math names the aggregate re-exported through D6c (homes:
#: ``factors.compute.minute.mmp`` / ``factors.compute.minute.jump_amount_corr``).
_DELETED_AGGREGATE_EXPORTS = [
    "DEFAULT_EPSILON",
    "MMP_LOOKBACK",
    "compute_minute_mmp",
    "mmp_valid_minute_counts",
    "JUMP_LOOKBACK_DAYS",
    "JUMP_MIN_PAIRS",
    "JUMP_Z",
    "compute_jump_amount_corr",
]


@pytest.mark.parametrize("module_name", _DELETED_MODULE_NAMES)
def test_deleted_module_is_not_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_aggregate_exports_no_factor_math():
    agg = importlib.import_module("data.clean.intraday_aggregate")
    for name in _DELETED_AGGREGATE_EXPORTS:
        assert not hasattr(agg, name), (
            f"data.clean.intraday_aggregate.{name} resurrected — factor math "
            f"lives in factors/compute/minute only (D6d)."
        )
    assert "mmp_ew" not in agg.INTRADAY_FEATURE_KEYS
    assert "epsilon" not in inspect.signature(agg.asof_daily_features).parameters


def test_probe_is_not_vacuous():
    """The same import/getattr mechanism SUCCEEDS on existing modules/names."""
    schema = importlib.import_module("data.clean.intraday_schema")  # kept real
    assert schema is not None
    mmp = importlib.import_module("factors.compute.minute.mmp")  # factor home
    assert hasattr(mmp, "compute_minute_mmp")  # the name IS found at its home
    agg = importlib.import_module("data.clean.intraday_aggregate")
    assert hasattr(agg, "asof_daily_features")  # and the kept generic core
    assert "ret" in agg.INTRADAY_FEATURE_KEYS
