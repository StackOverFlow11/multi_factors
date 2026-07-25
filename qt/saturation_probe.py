"""D5 feasibility probe: can the D4 materializer run the real evaluation universe?

D4's materializer was accepted against a 40-symbol hot-path smoke. D5 needs it on
the eleven-factor evaluation plane (CSI500, 995 symbols, 2021-07-01..2026-06-30),
so before committing to a multi-hour run this probe MEASURES, on the real cache,
the four quantities that decide whether that run is possible at all:

1. **listing depth** — per-symbol first 1min month, read from the parquet store's
   directory layout (no data is read). A symbol whose bars begin at or after
   ``emit_start`` can never accumulate ``lookback_days`` valid days before it, so
   the pooled saturation criterion can never terminate early and the load expands
   to the declared floor;
2. **on-disk volume** at the evaluation window vs at the floor;
3. **in-memory cost** of a whole-universe single-frame load, measured on a sample
   and extrapolated linearly (the materializer loads every symbol into ONE frame:
   ``sources.minute.minute_bars(list(symbols), ...)``);
4. **per-symbol equivalence** — whether materializing symbol-by-symbol and
   concatenating reproduces the whole-universe values exactly. This is the
   decisive question for the fix, and it is answered per factor rather than
   assumed: ten factors are per-symbol pure, ``intraday_amp_cut`` is not (it
   z-scores across the loaded cross-section, ``AMP_CUT_MIN_CROSS_SECTION=10``).

Run: ``python -m qt.saturation_probe`` (cache-only; zero live calls).
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import pandas as pd

from data.cache.intraday_cache import ENDPOINT as INTRADAY_ENDPOINT
from data.clean.intraday_schema import RAW_INTRADAY_FREQ
from factors import registry as factor_registry
from factors.compute.minute.binding import minute_raw_from_bars
from qt.factor_hotpath_smoke import CACHE_MINUTE_DATA_START, CacheMinuteProvider

DEFAULT_CACHE_ROOT = "artifacts/cache/tushare/v1"
#: The frozen D1 panel is on the exact evaluation data plane, so its symbol list
#: IS the evaluation universe — no need to re-resolve PIT membership here.
DEFAULT_UNIVERSE_PANEL = "artifacts/refactor_baseline/panels/ridge_minute_return_20.parquet"
EVAL_START = pd.Timestamp("2021-07-01")
EVAL_END = pd.Timestamp("2026-06-30")
GB = 1024**3


def universe_from_frozen_panel(path: str) -> list[str]:
    return sorted(pd.read_parquet(path)["symbol"].unique().tolist())


def first_minute_month(root: str, symbols: set[str]) -> pd.Series:
    """Per-symbol earliest 1min month, from DIRECTORY STRUCTURE only."""
    base = Path(root) / INTRADAY_ENDPOINT / f"freq={RAW_INTRADAY_FREQ}"
    first: dict[str, pd.Timestamp] = {}
    for prefix in base.glob("symbol_prefix=*"):
        for sym_dir in prefix.glob("symbol=*"):
            sym = sym_dir.name.split("=", 1)[1]
            if sym not in symbols:
                continue
            months = [
                pd.Timestamp(
                    year=int(yd.name.split("=", 1)[1]),
                    month=int(mf.name.split("=", 1)[1].replace(".parquet", "")),
                    day=1,
                )
                for yd in sym_dir.glob("year=*")
                for mf in yd.glob("month=*")
            ]
            if months:
                first[sym] = min(months)
    return pd.Series(first).sort_values()


def partition_bytes(root: str, symbols: set[str]) -> tuple[int, int]:
    """(evaluation-window bytes, floor-depth bytes) of 1min month partitions."""
    base = Path(root) / INTRADAY_ENDPOINT / f"freq={RAW_INTRADAY_FREQ}"
    ev = full = 0
    lo = (EVAL_START.year, EVAL_START.month)
    hi = (EVAL_END.year, EVAL_END.month)
    for prefix in base.glob("symbol_prefix=*"):
        for sym_dir in prefix.glob("symbol=*"):
            if sym_dir.name.split("=", 1)[1] not in symbols:
                continue
            for yd in sym_dir.glob("year=*"):
                year = int(yd.name.split("=", 1)[1])
                for mf in yd.glob("month=*"):
                    month = int(mf.name.split("=", 1)[1].replace(".parquet", ""))
                    size = mf.stat().st_size
                    full += size
                    if lo <= (year, month) <= hi:
                        ev += size
    return ev, full


def measure_load(provider, symbols: list[str], start, end) -> dict[str, float]:
    t0 = time.time()
    bars = provider.minute_bars(symbols, start, end)
    elapsed = time.time() - t0
    out = {
        "rows": float(len(bars)),
        "frame_gb": float(bars.memory_usage(deep=True).sum()) / GB,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "seconds": elapsed,
    }
    del bars
    return out


def per_symbol_equivalence(
    provider, symbols: list[str], start, end, factor_names: tuple[str, ...]
) -> list[dict]:
    """Whole-universe vs per-symbol-concatenated, per factor, on real bars."""
    bars = provider.minute_bars(symbols, start, end)
    rows = []
    for name in factor_names:
        factor = factor_registry.build(name, {})
        whole = minute_raw_from_bars(factor, bars).sort_index()
        parts = []
        for sym in symbols:
            one = bars[bars.index.get_level_values("symbol") == sym]
            if one.empty:
                continue
            parts.append(minute_raw_from_bars(factor, one))
        per = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
        entry: dict = {"factor": name, "rows": len(whole)}
        if whole.index.equals(per.index):
            both = whole.notna() & per.notna()
            entry["index_same"] = True
            entry["max_abs_diff"] = (
                float((whole[both] - per[both]).abs().max()) if both.any() else 0.0
            )
            entry["nan_set_diff"] = int((whole.isna() != per.isna()).sum())
        else:
            entry["index_same"] = False
            entry["max_abs_diff"] = float("nan")
            entry["nan_set_diff"] = -1
        entry["per_symbol_safe"] = bool(
            entry["index_same"]
            and entry["nan_set_diff"] == 0
            and entry["max_abs_diff"] == 0.0
        )
        rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--universe-panel", default=DEFAULT_UNIVERSE_PANEL)
    parser.add_argument("--sample-symbols", type=int, default=20)
    args = parser.parse_args(argv)

    symbols = universe_from_frozen_panel(args.universe_panel)
    symset = set(symbols)
    floor = pd.Timestamp(CACHE_MINUTE_DATA_START)
    print(f"universe: {len(symbols)} symbols | emit {EVAL_START.date()}..{EVAL_END.date()}")
    print(f"declared minute floor: {floor.date()}\n")

    print("== 1. listing depth ==")
    first = first_minute_month(args.cache_root, symset)
    late = int((first >= EVAL_START).sum())
    print(first.dt.year.value_counts().sort_index().to_string())
    print(
        f"\nsymbols whose first 1min bar is ON/AFTER emit_start: {late}"
        f"  -> pooled saturation CANNOT terminate early; it expands to the floor\n"
    )

    print("== 2. on-disk volume ==")
    ev, full = partition_bytes(args.cache_root, symset)
    print(f"evaluation window : {ev / GB:7.2f} GB")
    print(f"floor depth       : {full / GB:7.2f} GB  ({full / max(ev, 1):.2f}x)\n")

    print("== 3. whole-universe single-frame load (measured, then extrapolated) ==")
    provider = CacheMinuteProvider(args.cache_root)
    sample = symbols[: args.sample_symbols]
    scale = len(symbols) / len(sample)
    for label, start in (("eval-window", EVAL_START), ("floor-depth", floor)):
        m = measure_load(provider, sample, start, EVAL_END)
        print(
            f"{label:12s} {len(sample):3d} syms: rows={m['rows']:>12,.0f} "
            f"frame={m['frame_gb']:6.2f} GB peakRSS={m['peak_rss_gb']:6.2f} GB "
            f"load={m['seconds']:6.1f}s"
        )
        print(
            f"{'':12s} x{scale:5.2f} -> rows={m['rows'] * scale / 1e6:7.0f}M "
            f"frame={m['frame_gb'] * scale:7.1f} GB load={m['seconds'] * scale / 60:5.1f} min"
        )
    print(f"\nlive minute calls so far: {provider.live_calls} (cache-only)\n")

    print("== 4. per-symbol equivalence (the fix's decisive question) ==")
    probe_syms = symbols[:12]
    rows = per_symbol_equivalence(
        provider,
        probe_syms,
        pd.Timestamp("2021-03-01"),
        pd.Timestamp("2021-09-30"),
        ("ridge_minute_return_20", "volume_peak_count_20", "intraday_amp_cut_10"),
    )
    for r in rows:
        verdict = "SAFE" if r["per_symbol_safe"] else "NOT SAFE"
        print(
            f"{r['factor']:26s} rows={r['rows']:>6,} index_same={r['index_same']!s:5s} "
            f"max|diff|={r['max_abs_diff']:.3e} nan_set_diff={r['nan_set_diff']:>5d}  {verdict}"
        )
    print(
        "\nintraday_amp_cut is expected NOT SAFE at whole-factor granularity: its "
        "step-4 cross-sectional z-score needs >= AMP_CUT_MIN_CROSS_SECTION symbols "
        "on a date, and a one-symbol cross-section is all-NaN by definition. The "
        "per-symbol split must therefore happen BEFORE that combine, not around it."
    )
    print(f"live minute calls total: {provider.live_calls}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
