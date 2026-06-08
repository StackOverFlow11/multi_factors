"""Slice 6: factor preprocessing pipeline (PROC-001..004).

Cross-sectional, by-date processing: drop_missing + z-score standardization.
Zero-variance / single-name cross-sections must not crash. Output keeps the
canonical MultiIndex(date, symbol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.schema import INDEX_NAMES
from factors.process.pipeline import ProcessingPipeline


def _factor_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a MultiIndex(date, symbol) factor frame from (date, symbol, value)."""
    dates = [pd.Timestamp(d) for d, _, _ in rows]
    symbols = [s for _, s, _ in rows]
    values = [v for _, _, v in rows]
    idx = pd.MultiIndex.from_arrays([dates, symbols], names=INDEX_NAMES)
    return pd.DataFrame({"f": values}, index=idx)


def test_zscore_is_applied_by_date() -> None:
    # Two dates, each its own cross-section. Day 1 values {1,2,3}; day 2 {10,20,30}.
    # Z-score is computed independently per date, so both days map to the same
    # standardized triple even though their raw scales differ.
    frame = _factor_frame(
        [
            ("2024-01-01", "000001.SZ", 1.0),
            ("2024-01-01", "000002.SZ", 2.0),
            ("2024-01-01", "000003.SZ", 3.0),
            ("2024-01-02", "000001.SZ", 10.0),
            ("2024-01-02", "000002.SZ", 20.0),
            ("2024-01-02", "000003.SZ", 30.0),
        ]
    )
    out = ProcessingPipeline().transform(frame)

    for date in frame.index.get_level_values("date").unique():
        cross = out.xs(date, level="date")["f"]
        # Population std (ddof=0) -> mean 0, std 1 per date.
        assert cross.mean() == pytest_approx(0.0)
        assert cross.std(ddof=0) == pytest_approx(1.0)

    day1 = out.xs(pd.Timestamp("2024-01-01"), level="date")["f"].to_numpy()
    day2 = out.xs(pd.Timestamp("2024-01-02"), level="date")["f"].to_numpy()
    # Identical standardized shape across the two differently-scaled days.
    np.testing.assert_allclose(day1, day2)


def test_zscore_ignores_nan() -> None:
    # One name is NaN on this date. drop_missing removes it before standardizing,
    # so the surviving {2,4} cross-section standardizes on its own mean/std and
    # the NaN row does not appear in the output for that date.
    frame = _factor_frame(
        [
            ("2024-01-01", "000001.SZ", 2.0),
            ("2024-01-01", "000002.SZ", np.nan),
            ("2024-01-01", "000003.SZ", 4.0),
        ]
    )
    out = ProcessingPipeline().transform(frame)
    cross = out.xs(pd.Timestamp("2024-01-01"), level="date")["f"]

    assert "000002.SZ" not in cross.index
    assert len(cross) == 2
    assert cross.mean() == pytest_approx(0.0)
    assert cross.std(ddof=0) == pytest_approx(1.0)


def test_zscore_handles_zero_std() -> None:
    # Zero-variance date (all equal) and single-name date must NOT raise; the
    # documented rule is they map to 0.0 (de-meaned, no scaling).
    frame = _factor_frame(
        [
            ("2024-01-01", "000001.SZ", 5.0),
            ("2024-01-01", "000002.SZ", 5.0),
            ("2024-01-01", "000003.SZ", 5.0),
            ("2024-01-02", "000001.SZ", 7.0),  # single-name cross-section
        ]
    )
    out = ProcessingPipeline().transform(frame)

    zero_var = out.xs(pd.Timestamp("2024-01-01"), level="date")["f"]
    assert (zero_var == 0.0).all()

    single = out.xs(pd.Timestamp("2024-01-02"), level="date")["f"]
    assert (single == 0.0).all()


def test_processor_keeps_multiindex(factor_panel: pd.DataFrame) -> None:
    # Output must keep the canonical MultiIndex(date, symbol) and the same
    # factor columns; input must not be mutated.
    before = factor_panel.copy(deep=True)
    out = ProcessingPipeline().transform(factor_panel)

    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == INDEX_NAMES
    assert list(out.columns) == list(factor_panel.columns)
    assert out.index.is_monotonic_increasing
    # Immutability: the input frame is untouched.
    pd.testing.assert_frame_equal(factor_panel, before)


def pytest_approx(value: float):
    """Local tolerance helper (avoids importing pytest at module top)."""
    import pytest

    return pytest.approx(value, abs=1e-9)
