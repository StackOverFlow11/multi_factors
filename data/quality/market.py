"""Report-only data-quality checks for daily market / adj_factor frames (D3).

Each check is a pure function: it takes a canonical daily frame (raw cache-shaped
``[date, symbol, open, high, low, close, volume, amount]`` or the MultiIndex
``(date, symbol)`` panel) and returns one :class:`~data.quality.report.
QualityFinding` (or ``None`` when clean). Nothing here filters the panel, repairs
values, or touches ``front_adjust`` / cache coverage — it only surfaces findings.
"""

from __future__ import annotations

import pandas as pd

from data.quality._frames import reset_keys, row_examples
from data.quality.report import HARD, WARNING, QualityFinding, make_finding

_OHLC = ["open", "high", "low", "close"]
_DATE = "date"
_SYMBOL = "symbol"


def check_duplicate_keys(
    df: pd.DataFrame, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """Duplicate ``(date, symbol)`` rows (corrupts joins / coverage assumptions)."""
    d = reset_keys(df)
    if _DATE not in d.columns or _SYMBOL not in d.columns:
        return None
    mask = d.duplicated(subset=[_DATE, _SYMBOL], keep=False)
    if not mask.any():
        return None
    keys = (
        d.loc[mask, [_DATE, _SYMBOL]]
        .drop_duplicates()
        .sort_values([_SYMBOL, _DATE])
        .head(5)
    )
    examples = [{_DATE: r[_DATE], _SYMBOL: r[_SYMBOL]} for _, r in keys.iterrows()]
    return make_finding(
        dataset, "duplicate_keys", HARD, count=int(mask.sum()),
        examples=examples, note="duplicate (date, symbol) rows",
    )


def check_non_positive_ohlc(
    df: pd.DataFrame, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """Any of open/high/low/close <= 0 (or NaN)."""
    d = reset_keys(df)
    present = [c for c in _OHLC if c in d.columns]
    if not present or _DATE not in d.columns or _SYMBOL not in d.columns:
        return None
    vals = d[present].apply(pd.to_numeric, errors="coerce")
    mask = ~(vals > 0).all(axis=1)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, _DATE])
    return make_finding(
        dataset, "non_positive_ohlc", HARD, count=int(mask.sum()),
        examples=examples, note="open/high/low/close must be > 0",
    )


def check_high_low_inversion(
    df: pd.DataFrame, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """``high < low``."""
    d = reset_keys(df)
    if "high" not in d.columns or "low" not in d.columns:
        return None
    hi = pd.to_numeric(d["high"], errors="coerce")
    lo = pd.to_numeric(d["low"], errors="coerce")
    mask = (hi < lo).fillna(False)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, _DATE])
    return make_finding(
        dataset, "high_lt_low", HARD, count=int(mask.sum()),
        examples=examples, note="high must be >= low",
    )


def check_close_outside_range(
    df: pd.DataFrame, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """``close`` outside ``[low, high]``."""
    d = reset_keys(df)
    needed = {"low", "high", "close"}
    if not needed.issubset(d.columns):
        return None
    hi = pd.to_numeric(d["high"], errors="coerce")
    lo = pd.to_numeric(d["low"], errors="coerce")
    cl = pd.to_numeric(d["close"], errors="coerce")
    mask = ((cl < lo) | (cl > hi)).fillna(False)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, _DATE], extra="close")
    return make_finding(
        dataset, "close_outside_low_high", HARD, count=int(mask.sum()),
        examples=examples, note="close must be within [low, high]",
    )


def check_negative_volume_amount(
    df: pd.DataFrame, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """Negative ``volume`` or ``amount``."""
    d = reset_keys(df)
    cols = [c for c in ("volume", "amount") if c in d.columns]
    if not cols:
        return None
    vals = d[cols].apply(pd.to_numeric, errors="coerce")
    mask = (vals < 0).any(axis=1)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, _DATE])
    return make_finding(
        dataset, "negative_volume_amount", HARD, count=int(mask.sum()),
        examples=examples, note="volume / amount must be >= 0",
    )


def check_adj_factor(
    df: pd.DataFrame, *, dataset: str = "adj_factor"
) -> QualityFinding | None:
    """Non-positive / invalid ``adj_factor``."""
    d = reset_keys(df)
    if "adj_factor" not in d.columns:
        return None
    vals = pd.to_numeric(d["adj_factor"], errors="coerce")
    mask = ~(vals > 0)
    if not mask.any():
        return None
    key_cols = [c for c in (_SYMBOL, _DATE) if c in d.columns]
    examples = row_examples(d, mask, key_cols, extra="adj_factor") if key_cols else []
    return make_finding(
        dataset, "invalid_adj_factor", HARD, count=int(mask.sum()),
        examples=examples, note="adj_factor must be > 0",
    )


def check_extreme_returns(
    df: pd.DataFrame, *, dataset: str = "market_daily", threshold: float = 0.5
) -> QualityFinding | None:
    """Suspicious raw ``close`` day-over-day moves: ``abs(pct_change) > threshold``.

    Per symbol, sorted by date. WARNING severity (suspicious, not proven wrong —
    e.g. an un-adjusted split or a real streak); never used to filter the panel.
    """
    d = reset_keys(df)
    if not {"close", _SYMBOL, _DATE}.issubset(d.columns):
        return None
    d = d.sort_values([_SYMBOL, _DATE]).copy()
    close_num = pd.to_numeric(d["close"], errors="coerce")
    d["pct_change"] = close_num.groupby(d[_SYMBOL]).pct_change().round(4)
    mask = (d["pct_change"].abs() > threshold).fillna(False)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, _DATE], extra="pct_change")
    return make_finding(
        dataset, "extreme_close_move", WARNING, count=int(mask.sum()),
        examples=examples, note=f"|close pct_change| > {threshold}",
    )


def check_missing_dates(
    df: pd.DataFrame, expected_dates, *, dataset: str = "market_daily"
) -> QualityFinding | None:
    """Rows missing per symbol vs an explicitly-provided ``expected_dates`` calendar.

    WARNING severity (a gap may be a legitimate halt); reported only when a
    calendar is passed. Examples are bounded ``{symbol, date}`` of missing rows.
    """
    if expected_dates is None:
        return None
    d = reset_keys(df)
    if _DATE not in d.columns or _SYMBOL not in d.columns:
        return None
    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_dates))).normalize().unique()
    if len(expected) == 0:
        return None
    total = 0
    examples: list[dict] = []
    for sym in sorted(d[_SYMBOL].astype(str).unique()):
        present = set(
            pd.to_datetime(d.loc[d[_SYMBOL].astype(str) == sym, _DATE]).dt.normalize()
        )
        missing = [dt for dt in expected if dt not in present]
        total += len(missing)
        for dt in missing:
            if len(examples) < 5:
                examples.append({_SYMBOL: sym, _DATE: dt})
    if total == 0:
        return None
    return make_finding(
        dataset, "missing_dates", WARNING, count=total,
        examples=examples, note="rows missing vs expected_dates calendar",
    )


def run_market_checks(
    df: pd.DataFrame,
    *,
    dataset: str = "market_daily",
    expected_dates=None,
    return_threshold: float = 0.5,
) -> list[QualityFinding]:
    """Run the OHLCV daily checks; return all non-clean findings (deterministic)."""
    findings = []
    for fn in (
        check_duplicate_keys,
        check_non_positive_ohlc,
        check_high_low_inversion,
        check_close_outside_range,
        check_negative_volume_amount,
    ):
        f = fn(df, dataset=dataset)
        if f is not None:
            findings.append(f)
    rf = check_extreme_returns(df, dataset=dataset, threshold=return_threshold)
    if rf is not None:
        findings.append(rf)
    if expected_dates is not None:
        mf = check_missing_dates(df, expected_dates, dataset=dataset)
        if mf is not None:
            findings.append(mf)
    return findings


def run_adj_factor_checks(
    df: pd.DataFrame, *, dataset: str = "adj_factor"
) -> list[QualityFinding]:
    """Run the adj_factor frame checks (duplicate keys + positivity)."""
    findings = []
    dup = check_duplicate_keys(df, dataset=dataset)
    if dup is not None:
        findings.append(dup)
    adj = check_adj_factor(df, dataset=dataset)
    if adj is not None:
        findings.append(adj)
    return findings
