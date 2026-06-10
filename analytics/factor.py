"""Factor checks: forward returns, IC, IC summary, quantile returns.

This module is the ONLY place in the framework allowed to compute forward
(future-looking) returns. The factor layer must never receive them — see
CLAUDE.md invariant #1 and CONTRACTS.md §3.

Implementation note: the IC / quantile logic here is a simple numpy/pandas
implementation — deterministic, dependency-light, easy to audit — and it remains
the AUTHORITATIVE result that drives the run. Since P2-4, alphalens-reloaded is
wired in as a report-only cross-check via ``analytics/alphalens_adapter.py``
(backend disclosed in the report); it never replaces these numbers.

All functions are pure: inputs are never mutated.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data.clean.schema import DATE_LEVEL, validate_panel

_DEFAULT_PERIODS: tuple[int, ...] = (1, 5, 20)
_VALID_IC_METHODS = ("spearman", "pearson")


def forward_returns(
    panel: pd.DataFrame,
    periods: tuple[int, ...] = _DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Per-symbol forward returns from FUTURE close prices.

    For each ``n`` in ``periods``, column ``forward_return_{n}d`` holds, for each
    (date, symbol):

        close[t + n] / close[t] - 1

    computed within each symbol's own time series (no cross-symbol leakage). The
    last ``n`` dates of every symbol are NaN (no future bar exists yet).

    Allowed here only — analytics is the forward-return boundary (INV-001).

    Parameters
    ----------
    panel : canonical market panel, MultiIndex(date, symbol), must contain
        ``close``. Validated via the shared schema helper.
    periods : forward horizons in trading days; must all be positive ints.

    Returns
    -------
    DataFrame aligned to ``panel.index`` with one column per requested period.
    """
    validate_panel(panel)
    if "close" not in panel.columns:
        raise ValueError("forward_returns requires a 'close' column in the panel.")
    if not periods:
        raise ValueError("forward_returns: 'periods' must be a non-empty tuple of ints.")
    bad = [p for p in periods if not isinstance(p, int) or p <= 0]
    if bad:
        raise ValueError(f"forward_returns: periods must be positive ints; got {bad}.")

    # A non-positive denominator (close <= 0, or NaN) makes ``future/close - 1``
    # produce +/-inf, which would silently pollute IC / quantile stats. Mask the
    # denominator to NaN first so an undefined return is correctly NaN, never inf.
    close = panel["close"]
    safe_close = close.where(close > 0.0)
    # group by symbol so shift(-n) never reaches across symbols
    grouped = safe_close.groupby(level="symbol", sort=False, group_keys=False)
    out: dict[str, pd.Series] = {}
    for n in periods:
        future = grouped.shift(-n)
        out[f"forward_return_{n}d"] = future / safe_close - 1.0

    result = pd.DataFrame(out, index=panel.index)
    return result


def _cross_section_corr(factor: pd.Series, fwd_return: pd.Series, method: str) -> float:
    """Correlate one date's cross-section, dropping non-finite pairs. NaN if < 2."""
    # Treat +/-inf like NaN so a polluted return can never bias the correlation.
    pair = pd.DataFrame({"f": factor, "r": fwd_return}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair["f"].nunique() < 2 or pair["r"].nunique() < 2:
        # zero variance on either side -> correlation undefined
        return float("nan")
    return float(pair["f"].corr(pair["r"], method=method))


def compute_ic(
    factor: pd.Series,
    fwd_return: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Per-date cross-sectional information coefficient (IC).

    For each date, correlate the factor cross-section against the forward-return
    cross-section. ``method="spearman"`` gives rank IC (default); ``"pearson"``
    gives linear IC. NaN factor/return pairs are dropped within each date before
    correlating; a date with fewer than 2 valid pairs (or zero variance) yields
    NaN for that date.

    Parameters
    ----------
    factor, fwd_return : MultiIndex(date, symbol) Series, aligned on the same
        index. They are inner-joined on the index, so partial overlap is fine.
    method : "spearman" or "pearson".

    Returns
    -------
    Series indexed by date, one IC per date, sorted by date.
    """
    if method not in _VALID_IC_METHODS:
        raise ValueError(
            f"compute_ic: method must be one of {_VALID_IC_METHODS}; got {method!r}."
        )
    aligned = pd.DataFrame({"f": factor, "r": fwd_return})
    if not isinstance(aligned.index, pd.MultiIndex) or aligned.index.nlevels != 2:
        raise ValueError(
            "compute_ic: factor and fwd_return must share a MultiIndex(date, symbol)."
        )

    dates = aligned.index.get_level_values(DATE_LEVEL)
    ic_by_date: dict[pd.Timestamp, float] = {}
    for date, block in aligned.groupby(dates, sort=True):
        ic_by_date[date] = _cross_section_corr(block["f"], block["r"], method)

    ic = pd.Series(ic_by_date, name="ic")
    ic.index.name = DATE_LEVEL
    return ic.sort_index()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Summarize an IC series.

    Returns ``{"ic_mean": <mean>, "ic_ir": <mean/std>}`` where the information
    ratio uses the sample std (ddof=1). NaN ICs are ignored. If fewer than two
    finite ICs exist, ``ic_ir`` is NaN (std undefined).
    """
    clean = ic.dropna()
    if clean.empty:
        return {"ic_mean": float("nan"), "ic_ir": float("nan")}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ic_ir = mean / std if math.isfinite(std) and std != 0 else float("nan")
    return {"ic_mean": mean, "ic_ir": float(ic_ir)}


def quantile_returns(
    factor: pd.Series,
    fwd_return: pd.Series,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return per (date, quantile bucket).

    On each date the cross-section is split into ``quantiles`` buckets by factor
    rank (bucket 1 = lowest factor, bucket ``quantiles`` = highest). Each cell is
    the mean forward return of the symbols in that bucket on that date.

    Shape contract: a DataFrame with one ROW per date and one COLUMN per bucket
    (columns ``1..quantiles``, integer-labelled). Buckets with no members on a
    date are NaN. NaN factor/return pairs are dropped before bucketing.

    Parameters
    ----------
    factor, fwd_return : aligned MultiIndex(date, symbol) Series.
    quantiles : number of buckets (>= 2).

    Returns
    -------
    DataFrame, index = date, columns = bucket labels 1..quantiles.
    """
    if quantiles < 2:
        raise ValueError(f"quantile_returns: 'quantiles' must be >= 2; got {quantiles}.")
    # Drop non-finite (+/-inf as well as NaN) before bucketing so a polluted
    # return cannot turn a whole bucket mean into inf (shown as 'n/a').
    aligned = (
        pd.DataFrame({"f": factor, "r": fwd_return})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if aligned.empty:
        return pd.DataFrame()

    dates = aligned.index.get_level_values(DATE_LEVEL)
    rows: dict[pd.Timestamp, pd.Series] = {}
    bucket_labels = list(range(1, quantiles + 1))
    for date, block in aligned.groupby(dates, sort=True):
        buckets = _assign_buckets(block["f"], quantiles)
        means = block["r"].groupby(buckets).mean()
        rows[date] = means.reindex(bucket_labels)

    out = pd.DataFrame(rows).T
    out.index.name = DATE_LEVEL
    out.columns = bucket_labels
    return out


def _assign_buckets(factor_cs: pd.Series, quantiles: int) -> pd.Series:
    """Assign 1..quantiles bucket labels to a single date's factor cross-section.

    Uses rank-based qcut so duplicate factor values are spread deterministically.
    If the cross-section is too small/degenerate for ``quantiles`` distinct
    edges, falls back to a uniform rank split so every symbol still gets a label.
    """
    n = len(factor_cs)
    if n == 0:
        return pd.Series([], dtype="int64", index=factor_cs.index)
    ranks = factor_cs.rank(method="first")
    try:
        labels = pd.qcut(ranks, q=quantiles, labels=bucket_range(quantiles))
        return labels.astype("int64")
    except ValueError:
        # too few points for that many edges -> uniform split by order statistic
        order = (ranks - 1) * quantiles // n + 1
        return order.clip(upper=quantiles).astype("int64")


def bucket_range(quantiles: int) -> list[int]:
    """Bucket labels 1..quantiles (lowest factor = 1)."""
    return list(range(1, quantiles + 1))


__all__ = [
    "forward_returns",
    "compute_ic",
    "ic_summary",
    "quantile_returns",
]
