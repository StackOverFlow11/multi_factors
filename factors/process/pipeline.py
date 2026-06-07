"""Cross-sectional factor preprocessing pipeline (Slice 6, PROC-001..004).

``ProcessingPipeline`` is a concrete :class:`FactorProcessor`. It processes a
factor panel *by date* (each cross-section independently). P0 steps:

    drop_missing  -> per-date, drop rows whose factor value is NaN
    standardize   -> per-date z-score (mean ~0, std ~1)

Z-score uses the population standard deviation (``ddof=0``). A zero-variance or
single-name cross-section has no spread to scale by, so the documented rule is
that it standardizes to ``0.0`` (de-meaned, no scaling) rather than producing
NaN/inf -- it never raises.

P1 steps (winsorize, neutralize) are present as optional no-op hooks so the
config surface (``ProcessingCfg``) is honoured, but they are intentionally not
implemented in P0.

Design notes:
    * Pure / immutable: ``transform`` never mutates its input; it returns a new
      frame with the canonical ``MultiIndex(date, symbol)`` preserved.
    * Decisions are cross-sectional: every column is standardized within each
      date's cross-section, independent of other dates (no time-series leakage).
"""

from __future__ import annotations

import pandas as pd

from factors.process.base import FactorProcessor


class ProcessingPipeline(FactorProcessor):
    """Configurable by-date factor processor (drop_missing + z-score for P0)."""

    def __init__(
        self,
        *,
        drop_missing: bool = True,
        standardize: bool = True,
        winsorize: bool = False,
        neutralize: bool = False,
    ) -> None:
        # Step toggles. P1 hooks default off and are no-ops in P0.
        self._drop_missing = drop_missing
        self._standardize = standardize
        self._winsorize = winsorize
        self._neutralize = neutralize

    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Process every column cross-sectionally, per date.

        Args:
            factors: ``MultiIndex(date, symbol)`` frame; columns are factor names.

        Returns:
            A new ``MultiIndex(date, symbol)`` frame, same columns, each date's
            cross-section processed independently. Never mutates the input.
        """
        if not isinstance(factors.index, pd.MultiIndex):
            raise ValueError(
                "ProcessingPipeline.transform expects a MultiIndex(date, symbol) "
                f"factor panel; got index type {type(factors.index).__name__}."
            )
        if factors.empty:
            return factors.copy()

        date_level = factors.index.names[0]
        # Operate on a copy so the caller's frame is never mutated.
        out = factors.copy()

        if self._drop_missing:
            out = self._apply_drop_missing(out)
        if self._winsorize:
            out = self._apply_winsorize(out, date_level)
        if self._standardize:
            out = self._apply_zscore(out, date_level)
        if self._neutralize:
            out = self._apply_neutralize(out, date_level)

        return out.sort_index()

    @staticmethod
    def _apply_drop_missing(frame: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with any NaN factor value (per-date is implicit per-row)."""
        return frame.dropna(axis=0, how="any")

    @staticmethod
    def _apply_zscore(frame: pd.DataFrame, date_level: str) -> pd.DataFrame:
        """Z-score each column within each date's cross-section (ddof=0).

        Zero-variance / single-name cross-sections -> 0.0 (no spread to scale).
        """
        grouped = frame.groupby(level=date_level)
        # Per-date mean and population std, broadcast back to every row via
        # ``transform`` (keeps the full MultiIndex aligned, no manual join).
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)
        demeaned = frame - mean
        # Where std == 0 (constant column or single name), keep the de-meaned
        # value (which is 0) instead of dividing -> avoids NaN/inf.
        scaled = demeaned.divide(std).where(std != 0, other=demeaned)
        return scaled[list(frame.columns)]

    @staticmethod
    def _apply_winsorize(frame: pd.DataFrame, date_level: str) -> pd.DataFrame:
        """P1 hook: clip extremes per date. No-op in P0 (returns unchanged)."""
        return frame

    @staticmethod
    def _apply_neutralize(frame: pd.DataFrame, date_level: str) -> pd.DataFrame:
        """P1 hook: industry/size neutralization per date. No-op in P0."""
        return frame
