"""DemoFeed: a deterministic, network-free DataFeed (DATA-002, CFG-005).

Produces a canonical market panel from a fixed, reproducible price path — no
randomness, no ``datetime.now``. It lets the whole pipeline (and tests) run with
zero network and zero credentials. The price patterns mirror the adversarial
fixture (rising / falling / flat) so downstream factor/portfolio behaviour is
predictable, but this module is self-contained: it must not import ``tests``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean.schema import normalize_panel
from data.feed.base import DataFeed

# Built-in demo universe and its deterministic close patterns.
_DEMO_SYMBOLS: tuple[str, ...] = (
    "000001.SZ",  # strictly rising
    "000002.SZ",  # strictly falling
    "000003.SZ",  # flat
    "000004.SZ",  # mild rise
    "000005.SZ",  # flat then jump
)
_BASE_PRICE: float = 100.0
_JUMP_DAY: int = 30
_JUMP_MULTIPLIER: float = 3.0
# Daily multiplicative decay for the falling symbol: strictly < 1 (so price
# strictly falls each day) yet strictly > 0 (so it never crosses zero, keeping
# forward returns finite over the full multi-hundred-day demo calendar).
_FALL_DECAY: float = 0.99


def _close_path(symbol: str, n: int) -> np.ndarray:
    """Deterministic close path for ``symbol`` over ``n`` trading days."""
    t = np.arange(n, dtype=float)
    if symbol == "000001.SZ":
        return _BASE_PRICE + t
    if symbol == "000002.SZ":
        # Strictly falling but ALWAYS positive over the full calendar: a small
        # multiplicative daily decay (never crosses zero, so forward returns stay
        # finite — no inf pollution). Keeps the qualitative "low momentum" shape.
        return _BASE_PRICE * (_FALL_DECAY**t)
    if symbol == "000003.SZ":
        return np.full(n, _BASE_PRICE)
    if symbol == "000004.SZ":
        return _BASE_PRICE + 0.5 * t
    if symbol == "000005.SZ":
        close = np.full(n, _BASE_PRICE)
        if n > _JUMP_DAY:
            close[_JUMP_DAY:] = _BASE_PRICE * _JUMP_MULTIPLIER
        return close
    # Unknown symbol: a gentle deterministic ramp keyed off its hash, so any
    # requested code still yields a stable, positive, reproducible series.
    seed = abs(hash(symbol)) % 7 + 1
    return _BASE_PRICE + (t * seed) / 10.0


class DemoFeed(DataFeed):
    """Deterministic offline market-data source.

    Honors the requested ``symbols`` and the inclusive ``[start, end]`` window.
    Output is a canonical panel (``normalize_panel``) with all CORE_COLUMNS and
    ``adj_factor == 1.0``. No network, no randomness, no wall-clock.
    """

    def __init__(self, calendar_start: str = "2024-01-01", calendar_days: int = 250) -> None:
        # A fixed business-day calendar the demo prices live on. Requests outside
        # it simply return the overlapping slice (possibly empty).
        self._calendar_start = calendar_start
        self._calendar_days = int(calendar_days)

    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        freq: str = "D",
    ) -> pd.DataFrame:
        """Return a normalized demo panel for ``symbols`` over [start, end]."""
        if freq not in ("D", "1d", "daily"):
            raise ValueError(
                f"DemoFeed only supports daily bars (freq='D'); got freq={freq!r}."
            )
        if not symbols:
            raise ValueError("DemoFeed.get_bars requires a non-empty symbol list.")

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if start_ts > end_ts:
            raise ValueError(
                f"start ({start}) must be on or before end ({end}) for DemoFeed.get_bars."
            )

        calendar = pd.bdate_range(start=self._calendar_start, periods=self._calendar_days)
        rows: list[dict] = []
        for symbol in symbols:
            close = _close_path(symbol, len(calendar))
            for i, day in enumerate(calendar):
                if day < start_ts or day > end_ts:
                    continue
                c = float(close[i])
                rows.append(
                    {
                        "date": day,
                        "symbol": symbol,
                        "open": c,
                        "high": c * 1.01,
                        "low": c * 0.99,
                        "close": c,
                        "volume": 1_000_000.0 + i * 1_000.0,
                        "amount": (1_000_000.0 + i * 1_000.0) * c,
                        "adj_factor": 1.0,
                    }
                )

        raw = pd.DataFrame(
            rows,
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "adj_factor",
            ],
        )
        return normalize_panel(raw)
