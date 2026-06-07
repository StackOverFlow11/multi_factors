"""Slice 2: panel schema tests (backlog section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.schema import CORE_COLUMNS, INDEX_NAMES, normalize_panel, validate_panel


def _raw_rows() -> pd.DataFrame:
    """Two symbols, two dates, deliberately out of order."""
    rows = [
        ("2024-01-02", "000002.SZ"),
        ("2024-01-01", "000002.SZ"),
        ("2024-01-02", "000001.SZ"),
        ("2024-01-01", "000001.SZ"),
    ]
    data = []
    for d, s in rows:
        rec = {"date": d, "symbol": s}
        for c in CORE_COLUMNS:
            rec[c] = 1.0
        data.append(rec)
    return pd.DataFrame(data)


def test_normalize_panel_creates_multiindex() -> None:
    panel = normalize_panel(_raw_rows())
    assert isinstance(panel.index, pd.MultiIndex)
    assert list(panel.index.names) == INDEX_NAMES
    date_level = panel.index.get_level_values("date")
    assert pd.api.types.is_datetime64_any_dtype(date_level)
    assert all(isinstance(s, str) for s in panel.index.get_level_values("symbol"))
    # No exceptions from the full validator either.
    validate_panel(panel)


def test_normalize_panel_sorts_index() -> None:
    panel = normalize_panel(_raw_rows())
    assert panel.index.is_monotonic_increasing
    first = panel.index[0]
    assert first == (pd.Timestamp("2024-01-01"), "000001.SZ")


def test_normalize_panel_rejects_duplicate_index() -> None:
    raw = _raw_rows()
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as exc:
        normalize_panel(dup)
    assert "duplicate" in str(exc.value).lower()


def test_normalize_panel_requires_core_columns() -> None:
    raw = _raw_rows().drop(columns=["close"])
    with pytest.raises(ValueError) as exc:
        normalize_panel(raw)
    assert "close" in str(exc.value).lower()


def test_normalize_panel_allows_nan_cells() -> None:
    raw = _raw_rows()
    raw.loc[0, "close"] = np.nan
    panel = normalize_panel(raw)  # must not raise
    assert panel["close"].isna().any()


def test_normalize_panel_accepts_already_indexed() -> None:
    panel = normalize_panel(_raw_rows())
    # Round-trip: feeding a normalized panel back in is idempotent.
    again = normalize_panel(panel)
    pd.testing.assert_frame_equal(panel, again)
