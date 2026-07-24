"""Intraday (minute) bar schema — SEPARATE from the daily panel contract.

The daily canonical panel (:mod:`data.clean.schema`) normalizes its ``date``
level to midnight; minute bars must NOT reuse that — for intraday data the
intra-day ``time`` is meaningful and must be preserved. This module defines the
minute-bar shape and the helpers that enforce it. It is pure: every function
returns a new object and never mutates its input. It is also source-agnostic:
raw-vendor field renaming (e.g. tushare ``vol`` -> ``volume``) belongs in the
feed, not here.

Canonical intraday shape::

    index: MultiIndex(time, symbol)
        time   -> pandas.Timestamp at minute precision, NOT normalized to midnight
        symbol -> str, tushare style, e.g. "000001.SZ"
    sorted by (time, symbol), no duplicate (time, symbol) pairs.

Required columns: open, high, low, close, volume, amount, freq, bar_start,
bar_end, available_time. Raw persisted intraday bars use ``freq="1min"``; coarser
intraday bars, if needed later, are derived views rebuilt from 1min data. Extra
columns such as ``source_trade_time`` are kept for audit.

Minute-level PIT time rules (see tmp/context/intraday_pit_checkpoints/
02_minute_pit_semantics.md) — the upstream ``trade_time`` is the END of the
interval the bar represents::

    raw_freq       = 1min
    bar_end        = source trade_time
    bar_start      = bar_end - freq
    available_time = bar_end + data_lag   (earliest a strategy may use the bar)

``available_time`` is what enforces "an observation may be used for a decision
iff available_time <= decision_time". A conservative default ``data_lag`` of
``1min`` is used for backtests unless a tighter measured feed delay is known.

IMPORTANT (design invariant, CLAUDE.md #1): this module is intraday *shape* only.
It must NOT compute forward returns or anything that touches the future.
"""

from __future__ import annotations

import pandas as pd

# Minimal required OHLCV columns for an intraday bar.
INTRADAY_CORE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

# Derived PIT/time columns the normalizer adds.
INTRADAY_TIME_COLUMNS: list[str] = [
    "freq",
    "bar_start",
    "bar_end",
    "available_time",
]

# Canonical index level names.
TIME_LEVEL = "time"
SYMBOL_LEVEL = "symbol"
INTRADAY_INDEX_NAMES: list[str] = [TIME_LEVEL, SYMBOL_LEVEL]

# Daily-aggregation view of minute bars: the (date, symbol) grid every
# minute-DERIVED daily factor emits. Homed HERE since D2 (moved from
# data.clean.intraday_aggregate, which re-exports them) so that both the
# data-layer generic core (intraday_aggregate) and the factor math in
# factors.compute.minute can import one definition without an import cycle —
# the aggregate module re-exports the migrated MMP/jump math FROM factors, so
# the factor modules must never import the aggregate module back.
DATE_LEVEL = "date"
DAILY_INDEX_NAMES: list[str] = [DATE_LEVEL, SYMBOL_LEVEL]

# Session time-of-day defaults shared by the PIT cutoff machinery (I3) and the
# minute factors (D2). Same single-home rationale as DAILY_INDEX_NAMES above.
DEFAULT_DECISION_TIME = "14:50:00"
DEFAULT_SESSION_OPEN = "09:30:00"

RAW_INTRADAY_FREQ = "1min"

# The schema can represent derived coarser bars, but raw feeds/cache should only
# persist 1min bars. Coarser bars are built from the 1min cache in a later
# resampling stage, not fetched as separate upstream products.
DERIVED_INTRADAY_FREQS: tuple[str, ...] = ("5min", "15min", "30min", "60min")
SUPPORTED_INTRADAY_FREQS: tuple[str, ...] = (
    RAW_INTRADAY_FREQ,
    *DERIVED_INTRADAY_FREQS,
)


def ensure_supported_freq(freq: str) -> None:
    """Raise a readable ValueError unless ``freq`` is a supported schema freq."""
    if freq not in SUPPORTED_INTRADAY_FREQS:
        raise ValueError(
            f"Unsupported intraday freq {freq!r}; must be one of "
            f"{SUPPORTED_INTRADAY_FREQS}."
        )


def ensure_raw_intraday_freq(freq: str) -> None:
    """Raise unless a raw upstream/cache request uses the only raw freq: 1min."""
    if freq != RAW_INTRADAY_FREQ:
        raise ValueError(
            f"Raw intraday bars support only freq={RAW_INTRADAY_FREQ!r}; "
            "build coarser bars from cached 1min data via intraday resampling."
        )


def _freq_to_timedelta(freq: str) -> pd.Timedelta:
    """Convert a supported minute ``freq`` string into a ``pd.Timedelta``."""
    ensure_supported_freq(freq)
    return pd.Timedelta(freq)


def _extract_time_symbol(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Pull (time, symbol) out of a frame whether they are columns or the index.

    Returns (time_series, symbol_series, payload_frame_without_keys). Raises a
    readable ValueError if neither layout is present. Mirrors
    :func:`data.clean.schema._extract_date_symbol` but never normalizes ``time``.
    """
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        names = list(df.index.names)
        if names == INTRADAY_INDEX_NAMES or set(names) == set(INTRADAY_INDEX_NAMES):
            reset = df.reset_index()
            return (
                reset[TIME_LEVEL],
                reset[SYMBOL_LEVEL],
                reset.drop(columns=INTRADAY_INDEX_NAMES),
            )
        if all(n is None for n in names):
            time = df.index.get_level_values(0).to_series(index=range(len(df)))
            symbol = df.index.get_level_values(1).to_series(index=range(len(df)))
            return time, symbol, df.reset_index(drop=True)

    if TIME_LEVEL in df.columns and SYMBOL_LEVEL in df.columns:
        reset = df.reset_index(drop=True)
        return (
            reset[TIME_LEVEL],
            reset[SYMBOL_LEVEL],
            reset.drop(columns=INTRADAY_INDEX_NAMES),
        )

    raise ValueError(
        "Intraday bars must provide 'time' and 'symbol' either as a "
        "MultiIndex(time, symbol) or as two columns named 'time' and 'symbol'. "
        f"Got index names={list(df.index.names)} and columns={list(df.columns)}."
    )


def normalize_intraday_bars(
    df: pd.DataFrame, freq: str, data_lag: str = "1min"
) -> pd.DataFrame:
    """Return a new intraday panel in canonical shape.

    Accepts ``df`` with (time, symbol) either as columns or a MultiIndex, plus
    the OHLCV columns. Derives ``freq``/``bar_start``/``bar_end``/
    ``available_time`` from ``time`` (treated as the bar END) and produces:

      - MultiIndex(time, symbol), ``time`` at its original minute precision
        (NEVER normalized to midnight), ``symbol`` str;
      - rows sorted ascending by (time, symbol);
      - duplicates collapsed by the natural key (symbol, freq, bar_end), with the
        LATER input row winning (upstream may return reverse-chronological rows).

    Raises ValueError for an unsupported ``freq`` or missing OHLCV columns. Extra
    input columns (e.g. ``source_trade_time``) are preserved for audit. The input
    is never mutated.
    """
    bar_span = _freq_to_timedelta(freq)
    lag = pd.Timedelta(data_lag)

    time, symbol, payload = _extract_time_symbol(df)

    missing = [c for c in INTRADAY_CORE_COLUMNS if c not in payload.columns]
    if missing:
        raise ValueError(
            f"Intraday bars are missing required OHLCV columns: {missing}. "
            f"INTRADAY_CORE_COLUMNS = {INTRADAY_CORE_COLUMNS}."
        )

    bar_end = pd.to_datetime(time).reset_index(drop=True)

    out = payload.copy()
    out[TIME_LEVEL] = bar_end.to_numpy()
    out[SYMBOL_LEVEL] = symbol.astype(str).to_numpy()
    out["freq"] = freq
    out["bar_end"] = bar_end.to_numpy()
    out["bar_start"] = (bar_end - bar_span).to_numpy()
    out["available_time"] = (bar_end + lag).to_numpy()

    # Dedup BEFORE sorting so "later input row wins" is by input order.
    out = out.drop_duplicates(
        subset=[SYMBOL_LEVEL, "freq", "bar_end"], keep="last"
    )
    out = out.sort_values([TIME_LEVEL, SYMBOL_LEVEL], kind="mergesort")

    index = pd.MultiIndex.from_arrays(
        [out[TIME_LEVEL].to_numpy(), out[SYMBOL_LEVEL].to_numpy()],
        names=INTRADAY_INDEX_NAMES,
    )
    out = out.drop(columns=[TIME_LEVEL, SYMBOL_LEVEL])
    out.index = index

    ordered = [*INTRADAY_CORE_COLUMNS, *INTRADAY_TIME_COLUMNS]
    extras = [c for c in out.columns if c not in ordered]
    return out[[*ordered, *extras]]


def empty_intraday_bars() -> pd.DataFrame:
    """An empty but schema-shaped intraday panel (used when a feed returns none).

    Built with proper dtypes so :func:`validate_intraday_bars` accepts it.
    """
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
        names=INTRADAY_INDEX_NAMES,
    )
    data = {
        "open": pd.Series([], dtype=float),
        "high": pd.Series([], dtype=float),
        "low": pd.Series([], dtype=float),
        "close": pd.Series([], dtype=float),
        "volume": pd.Series([], dtype=float),
        "amount": pd.Series([], dtype=float),
        "freq": pd.Series([], dtype=object),
        "bar_start": pd.Series([], dtype="datetime64[ns]"),
        "bar_end": pd.Series([], dtype="datetime64[ns]"),
        "available_time": pd.Series([], dtype="datetime64[ns]"),
    }
    out = pd.DataFrame(data)
    out.index = index
    return out


def validate_intraday_bars(df: pd.DataFrame) -> None:
    """Assert that ``df`` already satisfies the canonical intraday contract.

    Raises a readable ValueError on any violation; returns None on success. Does
    not mutate or return a frame — use :func:`normalize_intraday_bars` to fix
    shape.
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            "Intraday index must be a 2-level MultiIndex(time, symbol); "
            f"got {type(df.index).__name__} with names={list(df.index.names)}."
        )
    if list(df.index.names) != INTRADAY_INDEX_NAMES:
        raise ValueError(
            f"Intraday index level names must be {INTRADAY_INDEX_NAMES}; "
            f"got {list(df.index.names)}."
        )

    time_level = df.index.get_level_values(TIME_LEVEL)
    if not pd.api.types.is_datetime64_any_dtype(time_level):
        raise ValueError(
            f"Intraday 'time' level must be datetime (pandas.Timestamp); "
            f"got dtype {time_level.dtype}."
        )

    symbol_level = df.index.get_level_values(SYMBOL_LEVEL)
    if not all(isinstance(s, str) for s in symbol_level):
        raise ValueError("Intraday 'symbol' level must contain only str values.")

    required = [*INTRADAY_CORE_COLUMNS, *INTRADAY_TIME_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Intraday bars are missing required columns: {missing}.")

    if df.index.duplicated().any():
        raise ValueError("Intraday bars have duplicate (time, symbol) index entries.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Intraday bars must be sorted by (time, symbol).")

    if len(df) == 0:
        return

    for col in ("bar_start", "bar_end", "available_time"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(
                f"Intraday '{col}' column must be datetime; got dtype {df[col].dtype}."
            )
    if (df["available_time"] < df["bar_end"]).any():
        raise ValueError(
            "Intraday available_time must be >= bar_end "
            "(PIT data lag cannot be negative)."
        )
    if (df["bar_start"] >= df["bar_end"]).any():
        raise ValueError(
            "Intraday bar_start must be < bar_end (a bar spans a positive interval)."
        )
