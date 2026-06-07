"""Thin P0 risk helpers for portfolio weights.

These are minimal, pure functions consumed later by P1 constructors (max-weight
cap, turnover cap, industry constraints). Each returns a NEW Series and never
mutates its input (immutable style). Keep this module small.
"""

from __future__ import annotations

import pandas as pd


def enforce_long_only(weights: pd.Series) -> pd.Series:
    """Return a copy of ``weights`` with all negative entries clipped to 0.

    P0 portfolios are long-only (PF-009). This does NOT renormalize; callers that
    need weights to sum to 1 should renormalize afterwards.
    """
    return weights.clip(lower=0.0)


def cap_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """Cap each weight at ``max_weight`` and renormalize to sum to 1 (PF-006).

    Args:
        weights: symbol-indexed weights (assumed non-negative).
        max_weight: per-name upper bound, in (0, 1].

    Returns:
        A NEW symbol-indexed Series whose entries are <= ``max_weight`` and that
        sums to ~1.0 when the input had positive mass. Returns the input
        unchanged (as a copy) when it is empty or sums to 0.

    Notes:
        A single capping + renormalize pass can push previously-uncapped names
        back over the cap. This thin P0 helper does a single pass; the P1
        constructor that adopts it will iterate to a feasible solution.
    """
    if max_weight <= 0.0:
        raise ValueError(f"max_weight must be positive, got {max_weight}")
    if weights.empty:
        return weights.copy()

    capped = weights.clip(upper=max_weight)
    total = capped.sum()
    if total <= 0.0:
        return capped
    return capped / total
