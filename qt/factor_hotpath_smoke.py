"""Hot-path timing smoke for the D4 materializer (design §八 R20).

Measures, on the REAL intraday cache (cache-only, ``stk_mins_live_calls=0``), the
magnitude gap between:

* NAIVE per-factor loading — each factor materialized separately, so the 1min
  read + normalize is paid ONCE PER FACTOR (what the D4 service does today: each
  ``materialize_range`` call loads its own bars); and
* SHARED loading — the 1min read + normalize + cutoff paid ONCE, then every
  factor computed from the shared bars (the ``factors.compute.minute.primitives``
  amortization the design §3.2 calls the load-bearing hot-path mechanism).

The budget is MEASURED, not claimed (§八): the caliber (universe sample, window,
factor count) and the two wall times are printed as-is, with an explicit caveat
that the numbers do NOT extrapolate to full-A (~6x the CSI500 read).

Run: ``python -m qt.factor_hotpath_smoke`` (deliberately NOT in qt/cli). It reads
``artifacts/cache/tushare/v1`` (symlink the main checkout's artifacts into the
worktree first; remove the symlink after so ``git status`` stays clean).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.cache.intraday_cache import READ_COLUMNS
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import (
    RAW_INTRADAY_FREQ,
    empty_intraday_bars,
    normalize_intraday_bars,
)
from factors import registry as factor_registry
from factors.compute.minute.binding import minute_raw_from_bars
from factors.view_lag import minute_decision_cutoff

DEFAULT_CACHE_ROOT = "artifacts/cache/tushare/v1"
DEFAULT_FACTORS = (
    "jump_amount_corr_20",
    "minute_ideal_amp_10",
    "amp_marginal_anomaly_vol_20",
    "volume_peak_count_20",
    "valley_relative_vwap_20",
    "peak_ridge_amount_ratio_20",
)
DEFAULT_START = "2024-03-01"
DEFAULT_END = "2024-04-05"
DEFAULT_N_SYMBOLS = 40


class CacheMinuteProvider:
    """Cache-only MinuteBarProvider: per-symbol read + normalize, zero live calls."""

    def __init__(self, root: str) -> None:
        self._store = IntradayParquetStore(root)
        self.calls = 0
        self.live_calls = 0  # provably 0 — read_range has no fetch closure

    def minute_bars(self, symbols, start, end):
        self.calls += 1
        if not symbols:
            return empty_intraday_bars()
        parts = []
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for sym in symbols:
            part = self._store.read_range(INTRADAY_ENDPOINT, sym, RAW_INTRADAY_FREQ, s, e)
            if part.empty:
                continue
            parts.append(
                normalize_intraday_bars(
                    part.rename(columns={"bar_end": "time"})[READ_COLUMNS], freq=RAW_INTRADAY_FREQ
                )
            )
        if not parts:
            return empty_intraday_bars()
        return pd.concat(parts).sort_index(kind="mergesort")


def _sample_symbols(root: str, n: int) -> list[str]:
    """First ``n`` symbols the intraday store has, in a stable directory order."""
    base = Path(root) / f"{INTRADAY_ENDPOINT}" / f"freq={RAW_INTRADAY_FREQ}"
    syms: list[str] = []
    for prefix_dir in sorted(base.glob("symbol_prefix=*")):
        for sym_dir in sorted(prefix_dir.glob("symbol=*")):
            syms.append(sym_dir.name.split("=", 1)[1])
            if len(syms) >= n:
                return syms
    return syms


@dataclass(frozen=True)
class SmokeResult:
    n_symbols: int
    n_factors: int
    window: str
    naive_seconds: float
    shared_seconds: float
    ratio: float
    naive_provider_calls: int
    shared_provider_calls: int
    live_calls: int
    raw_bar_rows: int


def run_hotpath_smoke(
    *,
    cache_root: str = DEFAULT_CACHE_ROOT,
    factor_ids=DEFAULT_FACTORS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    n_symbols: int = DEFAULT_N_SYMBOLS,
    cutoff: str = "14:50:00",
) -> SmokeResult:
    factor_ids = list(factor_ids)
    symbols = _sample_symbols(cache_root, n_symbols)
    factors = [factor_registry.build(fid) for fid in factor_ids]
    emit_start = pd.Timestamp(start).normalize()
    emit_end = pd.Timestamp(end).normalize()

    # The smoke isolates the READ amortization only, so BOTH paths use the same
    # BOUNDED load window (never the valid-day-pooled SATURATION expansion, which
    # is a separate correctness path — its multi-load cost would confound a pure
    # read-cost comparison). The bounded window comfortably covers each factor's
    # trailing depth; the shared load uses the max depth so every factor is served.
    max_depth = max(int(f.spec.lookback_depth) for f in factors)
    end_bar = emit_end + pd.Timedelta(days=1)

    # -- NAIVE: read + normalize + cutoff ONCE PER FACTOR (its own bounded window).
    naive_provider = CacheMinuteProvider(cache_root)
    t0 = time.monotonic()
    for factor in factors:
        w = int(factor.spec.lookback_depth)
        ls = emit_start - pd.Timedelta(days=w * 2 + 25)
        bars = minute_decision_cutoff(
            naive_provider.minute_bars(symbols, ls, end_bar), decision_time=cutoff
        )
        raw = minute_raw_from_bars(factor, bars)
        dd = raw.index.get_level_values("date")
        _ = raw[(dd >= emit_start) & (dd <= emit_end)]
    naive_seconds = time.monotonic() - t0

    # -- SHARED: read + normalize + cutoff ONCE, then compute every factor.
    shared_provider = CacheMinuteProvider(cache_root)
    load_start = emit_start - pd.Timedelta(days=max_depth * 2 + 25)
    t0 = time.monotonic()
    bars = shared_provider.minute_bars(symbols, load_start, end_bar)
    bars = minute_decision_cutoff(bars, decision_time=cutoff)
    raw_rows = len(bars)
    for factor in factors:
        raw = minute_raw_from_bars(factor, bars)
        dd = raw.index.get_level_values("date")
        _ = raw[(dd >= emit_start) & (dd <= emit_end)]
    shared_seconds = time.monotonic() - t0

    return SmokeResult(
        n_symbols=len(symbols),
        n_factors=len(factors),
        window=f"{start}..{end}",
        naive_seconds=round(naive_seconds, 2),
        shared_seconds=round(shared_seconds, 2),
        ratio=round(naive_seconds / shared_seconds, 2) if shared_seconds else float("nan"),
        naive_provider_calls=naive_provider.calls,
        shared_provider_calls=shared_provider.calls,
        live_calls=naive_provider.live_calls + shared_provider.live_calls,
        raw_bar_rows=raw_rows,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--n-symbols", type=int, default=DEFAULT_N_SYMBOLS)
    args = parser.parse_args(argv)
    r = run_hotpath_smoke(
        cache_root=args.cache_root, start=args.start, end=args.end, n_symbols=args.n_symbols
    )
    print(f"caliber: {r.n_symbols} symbols x {r.n_factors} minute factors, window {r.window}")
    print(f"raw 1min bar rows (shared load, post-cutoff): {r.raw_bar_rows}")
    print(f"NAIVE  (per-factor load): {r.naive_seconds}s  ({r.naive_provider_calls} provider loads)")
    print(f"SHARED (load once):       {r.shared_seconds}s  ({r.shared_provider_calls} provider load)")
    print(f"naive / shared ratio:     {r.ratio}x")
    print(f"stk_mins_live_calls:      {r.live_calls}  (cache-only)")
    print(
        "CAVEAT: measured on this CSI500 sample/window only; does NOT extrapolate "
        "to full-A (~6x the read). The budget is measured, not claimed (§八)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
