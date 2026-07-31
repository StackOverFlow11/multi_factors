"""Every DAILY factor's value must not depend on which universe was loaded.

WHY THIS IS LOAD-BEARING, AND WHY IT BECAME SO IN D6a. The factor store key is
``(factor_id, params, code, view)``; it has NO universe dimension, deliberately
(design revision A2: a PIT universe over a date range is a time-varying SET, so
"the universe" is not a value a key dimension could hold). That is only sound
while the stored value is universe-INDEPENDENT.

On the minute plane this was learned the hard way: ``intraday_amp_cut``'s
cross-sectional z-score ran during materialization, so its stored value WAS a
function of the loaded universe while the key said otherwise — measured, 12 names
filled then 24 read served 24 wrong cells with zero recompute and zero error.
D4c fixed it by storing the per-symbol intermediate and moving the combine to
read-assembly.

THE DAILY PLANE NEVER INHERITED THAT FIX, because until D6a no daily factor value
was ever stored. D6a is the step that puts them in the shared store, so the
invariant that keeps the daily plane sound — every daily factor is per-symbol —
becomes load-bearing here. It held before by accident of what was written; from
here it has to hold on purpose.

RED MEANS STOP AND REPORT. A daily factor whose value moves when the universe
changes cannot be stored under this key. The fix is D4c's (store the per-symbol
intermediate, combine at read-assembly) — NOT adding the factor to an exemption
list, which would restore exactly the silent-wrong-value failure D4c closed.

WHY BEHAVIOURAL AND NOT A SOURCE SCAN. Scanning ``factors/compute`` for
``groupby(level="date")`` / ``.rank(`` is both too strict and too loose:
``groupby(level="symbol").rank()`` is a per-symbol time-series rank and would be
a false positive, while a cross-sectional standardization written as
``x.unstack().sub(x.unstack().mean(axis=1), axis=0).stack()`` contains neither
token and would sail through. Computing the factor over two universes and
comparing the shared symbols tests the property itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.schema import CORE_COLUMNS, normalize_panel
from factors import registry as factor_registry
from factors.materialize import is_minute_factor

_SYMBOLS = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
_SUBSET = ["000001.SZ", "000003.SZ", "000005.SZ"]
_N_DAYS = 60


def _registered_factors() -> list:
    """One instance per registry entry — DERIVED, never a hand-kept list.

    Reads the default registry's dispatch tables directly rather than through a
    public accessor: adding one would be a production change, and this step is a
    path switch that has to stay provably behaviour-neutral. The point is that a
    newly registered factor lands in this census without anyone remembering to
    add it — including the case this test exists for, a new DAILY factor.

    An entry's builder is called with the entry key: the exact-name builders
    (financial fields, value ratios) use the name, and the prefix builders ignore
    it and take their window from params defaults, so one call shape serves both.
    """
    registry = factor_registry.DEFAULT_REGISTRY
    entries = list(registry._exact.values()) + list(registry._prefixes)
    return [entry.builder(entry.key, {}) for entry in entries]


def _daily_factors() -> list:
    return [f for f in _registered_factors() if not is_minute_factor(f)]


def _panel(symbols: list[str], extra_fields: tuple[str, ...]) -> pd.DataFrame:
    """A deterministic panel whose columns differ ACROSS symbols.

    Detection power depends on that: a cross-sectional operator only changes a
    value when the peers it standardizes against change, so every symbol gets its
    own level and slope.

    A symbol's data is keyed off its identity (its position in ``_SYMBOLS``), NOT
    off its position in ``symbols``. The first draft used ``enumerate(symbols)``
    and every one of the ten daily factors "failed": dropping two names shifted
    the survivors' own prices, so the two panels held different DATA and the
    comparison was measuring the fixture. The census caught it instantly, which
    is the one piece of evidence that it has power at all — but it is also the
    reminder that a probe reporting "everything is broken" is usually broken
    itself.
    """
    dates = pd.bdate_range("2024-01-01", periods=_N_DAYS)
    rows = []
    for symbol in symbols:
        i = _SYMBOLS.index(symbol)
        t = np.arange(_N_DAYS, dtype=float)
        close = 50.0 + 10.0 * i + (1.0 + 0.5 * i) * t + np.sin(t / 3.0) * (1.0 + i)
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.97,
                "close": close,
                "volume": 1_000.0 + 100.0 * i + t,
                "amount": (1_000.0 + 100.0 * i + t) * close,
                "adj_factor": 1.0,
            }
        )
        for k, field in enumerate(extra_fields):
            if field in frame.columns:
                continue
            frame[field] = 0.5 + 0.1 * i + 0.01 * k + t / 1000.0
        rows.append(frame)
    return normalize_panel(pd.concat(rows, ignore_index=True))


def _extra_fields(factor) -> tuple[str, ...]:
    """Panel columns the factor reads beyond the core OHLCV set."""
    return tuple(f for f in (factor.spec.input_fields or ()) if f not in CORE_COLUMNS)


def test_the_census_is_not_empty_and_covers_the_known_daily_families():
    """A census that enumerates nothing passes vacuously; this refuses to.

    Not an exhaustive expected list (that would be the hand-kept register this
    test exists to avoid) — just enough anchors that an enumeration returning
    nothing, or losing whole families, cannot read as success.
    """
    names = {f.name for f in _daily_factors()}
    assert len(names) >= 8, f"suspiciously small daily-factor census: {sorted(names)}"
    for anchor in ("momentum_20", "volatility_20", "value_ep", "roe"):
        assert anchor in names, f"{anchor} missing from the census: {sorted(names)}"


def test_no_minute_factor_leaked_into_the_daily_census():
    """The split is what makes the daily claim meaningful; pin it."""
    assert not [f for f in _daily_factors() if f.spec.is_intraday]


@pytest.mark.parametrize("factor", _daily_factors(), ids=lambda f: f.name)
def test_a_daily_factor_value_does_not_depend_on_the_loaded_universe(factor):
    """Drop 2 of 5 symbols; the surviving 3 must keep bit-identical values."""
    extra = _extra_fields(factor)
    full = factor.compute(_panel(_SYMBOLS, extra))
    subset = factor.compute(_panel(_SUBSET, extra))

    shared = subset.index
    missing = shared.difference(full.index)
    assert not len(missing), f"{factor.name}: subset produced rows the full run did not"

    a = full.reindex(shared)
    b = subset
    nan_mismatch = int((a.isna() != b.isna()).sum())
    both = a.notna() & b.notna()
    worst = float((a[both] - b[both]).abs().max()) if both.any() else 0.0

    assert nan_mismatch == 0 and worst == 0.0, (
        f"{factor.name} is CROSS-SECTIONAL on the daily plane: dropping two "
        f"symbols moved {nan_mismatch} NaN cell(s) and up to {worst:.3e} in value "
        f"for the symbols that stayed.\n"
        f"STOP AND REPORT — do not add an exemption. Its value is a function of "
        f"the loaded universe, but the factor store key "
        f"(factor_id, params, code, view) does not name a universe, so a value "
        f"filled under one universe would be served to another (the D5b defect, "
        f"measured on the minute plane before D4c fixed it). The fix is D4c's: "
        f"store the per-symbol intermediate and run the cross-sectional combine "
        f"at read-assembly."
    )


def test_at_least_one_finite_value_is_actually_compared(request):
    """Guards the parametrized test above from passing on all-NaN output."""
    for factor in _daily_factors():
        extra = _extra_fields(factor)
        values = factor.compute(_panel(_SUBSET, extra))
        assert values.notna().any(), (
            f"{factor.name} produced no finite value on the census fixture, so the "
            f"universe-independence comparison for it is vacuous."
        )
