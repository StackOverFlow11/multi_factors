"""Report-only data-quality checks for normalized 1min intraday frames (D3).

Pure functions over a normalized intraday frame (raw cache-shaped columns or the
MultiIndex ``(time, symbol)`` panel; the bar timestamp is ``bar_end`` when
present, else ``time``). Report-only: nothing here enforces raw frequency, drops
rows, or replaces the existing intraday schema / cache guards — it surfaces
suspicious patterns alongside them.
"""

from __future__ import annotations

import pandas as pd

from data.quality._frames import reset_keys, row_examples
from data.quality.report import HARD, WARNING, QualityFinding, make_finding

_OHLC = ["open", "high", "low", "close"]
_SYMBOL = "symbol"
_DEFAULT_DATASET = "stk_mins_1min"


def _time_col(d: pd.DataFrame) -> str | None:
    """The bar-timestamp column: ``bar_end`` (preferred) else ``time``."""
    if "bar_end" in d.columns:
        return "bar_end"
    if "time" in d.columns:
        return "time"
    return None


def check_duplicate_bars(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """Duplicate ``(bar_end/time, symbol)`` rows."""
    d = reset_keys(df)
    tcol = _time_col(d)
    if tcol is None or _SYMBOL not in d.columns:
        return None
    mask = d.duplicated(subset=[tcol, _SYMBOL], keep=False)
    if not mask.any():
        return None
    keys = (
        d.loc[mask, [tcol, _SYMBOL]]
        .drop_duplicates()
        .sort_values([_SYMBOL, tcol])
        .head(5)
    )
    examples = [{_SYMBOL: r[_SYMBOL], tcol: r[tcol]} for _, r in keys.iterrows()]
    return make_finding(
        dataset, "duplicate_bars", HARD, count=int(mask.sum()),
        examples=examples, note=f"duplicate ({tcol}, symbol) rows",
    )


def check_non_monotonic_time(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """Bar timestamps that step backwards within a symbol (in row order)."""
    d = reset_keys(df)
    tcol = _time_col(d)
    if tcol is None or _SYMBOL not in d.columns:
        return None
    d = d.assign(_t=pd.to_datetime(d[tcol], errors="coerce"))
    backward = d.groupby(_SYMBOL, sort=False)["_t"].diff() < pd.Timedelta(0)
    backward = backward.fillna(False)
    if not backward.any():
        return None
    examples = row_examples(d, backward, [_SYMBOL, tcol])
    return make_finding(
        dataset, "non_monotonic_time", HARD, count=int(backward.sum()),
        examples=examples, note=f"{tcol} steps backwards within a symbol",
    )


def check_non_positive_ohlc(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """Any of open/high/low/close <= 0 (or NaN)."""
    d = reset_keys(df)
    tcol = _time_col(d)
    present = [c for c in _OHLC if c in d.columns]
    if not present or tcol is None or _SYMBOL not in d.columns:
        return None
    vals = d[present].apply(pd.to_numeric, errors="coerce")
    mask = ~(vals > 0).all(axis=1)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, tcol])
    return make_finding(
        dataset, "non_positive_ohlc", HARD, count=int(mask.sum()),
        examples=examples, note="open/high/low/close must be > 0",
    )


def check_high_low_inversion(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """``high < low``."""
    d = reset_keys(df)
    tcol = _time_col(d)
    if tcol is None or "high" not in d.columns or "low" not in d.columns:
        return None
    hi = pd.to_numeric(d["high"], errors="coerce")
    lo = pd.to_numeric(d["low"], errors="coerce")
    mask = (hi < lo).fillna(False)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, tcol])
    return make_finding(
        dataset, "high_lt_low", HARD, count=int(mask.sum()),
        examples=examples, note="high must be >= low",
    )


def check_close_outside_range(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """``close`` outside ``[low, high]``."""
    d = reset_keys(df)
    tcol = _time_col(d)
    if tcol is None or not {"low", "high", "close"}.issubset(d.columns):
        return None
    hi = pd.to_numeric(d["high"], errors="coerce")
    lo = pd.to_numeric(d["low"], errors="coerce")
    cl = pd.to_numeric(d["close"], errors="coerce")
    mask = ((cl < lo) | (cl > hi)).fillna(False)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, tcol], extra="close")
    return make_finding(
        dataset, "close_outside_low_high", HARD, count=int(mask.sum()),
        examples=examples, note="close must be within [low, high]",
    )


def check_negative_volume_amount(
    df: pd.DataFrame, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """Negative ``volume`` or ``amount``."""
    d = reset_keys(df)
    tcol = _time_col(d)
    cols = [c for c in ("volume", "amount") if c in d.columns]
    if tcol is None or not cols:
        return None
    vals = d[cols].apply(pd.to_numeric, errors="coerce")
    mask = (vals < 0).any(axis=1)
    if not mask.any():
        return None
    examples = row_examples(d, mask, [_SYMBOL, tcol])
    return make_finding(
        dataset, "negative_volume_amount", HARD, count=int(mask.sum()),
        examples=examples, note="volume / amount must be >= 0",
    )


def check_missing_minutes(
    df: pd.DataFrame, expected_times, *, dataset: str = _DEFAULT_DATASET
) -> QualityFinding | None:
    """Bars missing per symbol vs an explicitly-provided minute calendar/window.

    WARNING severity (a gap may be a real halt / illiquid minute); reported only
    when a calendar is passed. Examples are bounded ``{symbol, time}``.
    """
    if expected_times is None:
        return None
    d = reset_keys(df)
    tcol = _time_col(d)
    if tcol is None or _SYMBOL not in d.columns:
        return None
    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_times))).unique()
    if len(expected) == 0:
        return None
    total = 0
    examples: list[dict] = []
    for sym in sorted(d[_SYMBOL].astype(str).unique()):
        present = set(pd.to_datetime(d.loc[d[_SYMBOL].astype(str) == sym, tcol]))
        missing = [t for t in expected if t not in present]
        total += len(missing)
        for t in missing:
            if len(examples) < 5:
                examples.append({_SYMBOL: sym, tcol: t})
    if total == 0:
        return None
    return make_finding(
        dataset, "missing_minutes", WARNING, count=total,
        examples=examples, note="bars missing vs expected minute calendar",
    )


def run_intraday_checks(
    df: pd.DataFrame,
    *,
    dataset: str = _DEFAULT_DATASET,
    expected_times=None,
) -> list[QualityFinding]:
    """Run the 1min intraday checks; return all non-clean findings (deterministic)."""
    findings = []
    for fn in (
        check_duplicate_bars,
        check_non_monotonic_time,
        check_non_positive_ohlc,
        check_high_low_inversion,
        check_close_outside_range,
        check_negative_volume_amount,
    ):
        f = fn(df, dataset=dataset)
        if f is not None:
            findings.append(f)
    if expected_times is not None:
        mf = check_missing_minutes(df, expected_times, dataset=dataset)
        if mf is not None:
            findings.append(mf)
    return findings
