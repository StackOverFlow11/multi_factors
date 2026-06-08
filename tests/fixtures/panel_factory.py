"""Deterministic test-data factory (backlog section 2).

Builds a tiny but adversarial market panel that exposes the classic bugs:
rising / falling / flat trends, a mid-series NaN, and an extreme jump. Fully
deterministic: no randomness, no ``datetime.now`` — the same input always
yields byte-identical output, so tests are reproducible.

Symbols and their patterns:
    000001.SZ  close strictly rising      -> high momentum
    000002.SZ  close strictly falling     -> low momentum
    000003.SZ  close flat                 -> ~zero momentum
    000004.SZ  one NaN close mid-series   -> missing-data handling
    000005.SZ  one extreme jump           -> later winsorize target
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.schema import normalize_panel

SYMBOLS: list[str] = [
    "000001.SZ",
    "000002.SZ",
    "000003.SZ",
    "000004.SZ",
    "000005.SZ",
]
N_DAYS: int = 45
START_DATE: str = "2024-01-01"
BASE_PRICE: float = 100.0

# Index (0-based) where the special events happen.
NAN_DAY: int = 22          # 000004.SZ close is NaN here
JUMP_DAY: int = 30         # 000005.SZ close jumps here
JUMP_MULTIPLIER: float = 3.0


def _trading_dates() -> pd.DatetimeIndex:
    """45 business days starting at START_DATE (deterministic, no weekends)."""
    return pd.bdate_range(start=START_DATE, periods=N_DAYS)


def _close_series_for(symbol: str) -> np.ndarray:
    """Deterministic close path per symbol pattern."""
    t = np.arange(N_DAYS, dtype=float)
    if symbol == "000001.SZ":
        # Strictly rising: +1 per day.
        close = BASE_PRICE + t
    elif symbol == "000002.SZ":
        # Strictly falling: -1 per day (stays positive over 45 days).
        close = BASE_PRICE + 50.0 - t
    elif symbol == "000003.SZ":
        # Flat.
        close = np.full(N_DAYS, BASE_PRICE)
    elif symbol == "000004.SZ":
        # Mildly rising, with one NaN inserted mid-series.
        close = BASE_PRICE + 0.5 * t
        close[NAN_DAY] = np.nan
    elif symbol == "000005.SZ":
        # Flat-ish, then a single extreme upward jump that persists.
        close = np.full(N_DAYS, BASE_PRICE)
        close[JUMP_DAY:] = BASE_PRICE * JUMP_MULTIPLIER
    else:  # pragma: no cover - defensive
        close = np.full(N_DAYS, BASE_PRICE)
    return close


def make_demo_panel() -> pd.DataFrame:
    """Return a NORMALIZED demo market panel (via schema.normalize_panel).

    Shape: MultiIndex(date, symbol), CORE_COLUMNS present, adj_factor == 1.0,
    positive volume/amount, open/high/low derived sanely from close. The single
    NaN close (000004.SZ @ NAN_DAY) is preserved (NaN cells are legal).
    """
    dates = _trading_dates()
    rows: list[dict] = []
    for symbol in SYMBOLS:
        close = _close_series_for(symbol)
        for i, d in enumerate(dates):
            c = close[i]
            if np.isnan(c):
                # Missing close -> whole OHLC row is missing, volume 0.
                open_ = high = low = np.nan
                volume = 0.0
                amount = 0.0
            else:
                open_ = c  # simple, deterministic OHLC around close
                high = c * 1.01
                low = c * 0.99
                volume = 1_000_000.0 + i * 1_000.0  # positive, controllable
                amount = volume * c
            rows.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": c,
                    "volume": volume,
                    "amount": amount,
                    "adj_factor": 1.0,
                }
            )
    raw = pd.DataFrame(rows)
    return normalize_panel(raw)


def make_factor_panel() -> pd.DataFrame:
    """Return a small deterministic factor panel for processing/alpha tests.

    Two columns (``momentum_20``, ``volatility_20``) over the same (date, symbol)
    index as the demo panel. Values are deterministic placeholders, not the real
    factor computation (that belongs to the factors agent); 000004.SZ keeps a NaN
    where its close was missing, so NaN-handling paths get exercised.
    """
    panel = make_demo_panel()
    idx = panel.index
    close = panel["close"]
    # Deterministic, monotone-ish stand-ins derived from price so downstream
    # tests have realistic-shaped numbers without depending on the real factor.
    mom = (close / BASE_PRICE - 1.0).rename("momentum_20")
    vol = (close.abs() / BASE_PRICE * 0.1).rename("volatility_20")
    out = pd.DataFrame({"momentum_20": mom, "volatility_20": vol}, index=idx)
    return out


def make_scores(date: str | pd.Timestamp) -> pd.Series:
    """Return a deterministic symbol-indexed score Series for one date.

    Useful for portfolio tests. Scores rank symbols 000001..000005 ascending
    (000005 highest), with 000004 set to NaN to exercise NaN handling.
    """
    ts = pd.Timestamp(date).normalize()
    values = {
        "000001.SZ": 0.1,
        "000002.SZ": 0.2,
        "000003.SZ": 0.3,
        "000004.SZ": np.nan,
        "000005.SZ": 0.5,
    }
    s = pd.Series(values, name="score")
    s.index.name = "symbol"
    s.attrs["date"] = ts
    return s
