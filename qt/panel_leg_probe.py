"""D5 C5 fourth-leg PROBE: the new engine's panel vs the frozen D1 baseline.

Not the acceptance run — a MEASUREMENT taken before committing to one, in the same
spirit as ``qt.saturation_probe`` (which is why the multi-hour D5 run was not
started blind). It answers ONE question the C5 plan needs an answer to first:

    how big, and of what shape, is the difference between the values the D4/D4b
    materializer produces on the real evaluation plane and the D1 frozen panel the
    fourth leg compares against?

The question matters because the two are NOT expected to agree cell for cell, and
the reason is structural rather than numerical. The frozen panel was produced by the
old runners' loaders, which read exactly ``[data.start, data.end]`` and did NO
warm-up extension (``docs/factors/d5_runner_difference_catalogue.md`` §四). The
materializer loads ``lookback_depth`` trailing trading days of warm-up for a bounded
factor and expands to saturation for a valid-day-pooled one. So the early part of
the window is structurally under-warmed in the BASELINE and correctly warmed in the
new engine, and the fourth leg's "relative <= 1e-12" phrasing cannot apply to it.

What this probe reports, per factor:

* how many cells differ, and the NaN-set changes split by direction (NaN -> finite is
  the warm-up being filled in; finite -> NaN would be the alarming direction);
* the difference profile BY MONTH — the warm-up explanation predicts a front-loaded
  profile decaying to zero, and a difference that does NOT decay is a different
  animal that needs its own attribution;
* the residual on the cells both sides call finite OUTSIDE the early region, which is
  where the "float re-association only" claim can actually be tested.

Cache-only (``stk_mins_live_calls`` is structurally 0: the store read has no fetch
closure). Resumable: results are appended per factor to ``--out`` as JSON lines and a
re-run skips factors already present unless ``--force``.

Run (needs the gitignored ``artifacts/`` tree — see
``docs/factors/d5_exec_baseline_freeze.md`` for the symlink note):
``python -m qt.panel_leg_probe --factors jump_amount_corr_20``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.availability_policy import View
from factors import registry as factor_registry
from factors.materialize import MaterializeSources, materialize_range
from qt.factor_hotpath_smoke import CacheMinuteProvider

DEFAULT_CACHE_ROOT = "artifacts/cache/tushare/v1"
DEFAULT_PANEL_DIR = "artifacts/refactor_baseline/panels"
EVAL_START = pd.Timestamp("2021-07-01")
EVAL_END = pd.Timestamp("2026-06-30")


def frozen_panel(panel_dir: str, factor_id: str) -> pd.Series:
    """The frozen D1 baseline as a ``MultiIndex(date, symbol)`` Series."""
    frame = pd.read_parquet(Path(panel_dir) / f"{factor_id}.parquet")
    return (
        frame.set_index(["date", "symbol"])[factor_id]
        .sort_index(kind="mergesort")
    )


def compare(factor_id: str, symbols: list[str], cache_root: str, panel_dir: str) -> dict:
    """Materialize ``factor_id`` on the real plane and profile the difference."""
    base = frozen_panel(panel_dir, factor_id)
    provider = CacheMinuteProvider(cache_root)
    started = time.monotonic()
    fresh = materialize_range(
        factor_registry.build(factor_id),
        view=View.DECISION,
        symbols=symbols,
        emit_start=EVAL_START,
        emit_end=EVAL_END,
        sources=MaterializeSources(minute=provider),
    ).sort_index(kind="mergesort")
    wall = time.monotonic() - started

    joined = pd.DataFrame({"base": base}).join(
        pd.DataFrame({"fresh": fresh}), how="outer"
    )
    b, f = joined["base"].to_numpy(), joined["fresh"].to_numpy()
    b_nan, f_nan = np.isnan(b), np.isnan(f)
    both = ~b_nan & ~f_nan
    filled = b_nan & ~f_nan          # warm-up filled in (the expected direction)
    lost = ~b_nan & f_nan            # the alarming direction
    # Relative difference on the cells both sides call finite.
    denom = np.maximum(np.abs(b), np.abs(f))
    rel = np.zeros(len(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel[both] = np.where(
            denom[both] > 0, np.abs(b[both] - f[both]) / denom[both], 0.0
        )
    differing = both & (rel > 1e-12)

    months = pd.PeriodIndex(
        joined.index.get_level_values("date"), freq="M"
    ).astype(str)
    by_month: dict[str, dict[str, int]] = {}
    for label in sorted(set(months)):
        m = months == label
        entry = {
            "filled": int((filled & m).sum()),
            "lost": int((lost & m).sum()),
            "differing_finite": int((differing & m).sum()),
        }
        if any(entry.values()):
            by_month[label] = entry

    # The residual OUTSIDE the first year, where warm-up cannot be the explanation.
    late = joined.index.get_level_values("date") >= EVAL_START + pd.DateOffset(years=1)
    late_both = both & late
    return {
        "factor": factor_id,
        "wall_seconds": round(wall, 1),
        "live_calls": provider.live_calls,
        "rows_base": int(len(base)),
        "rows_fresh": int(len(fresh)),
        "rows_union": int(len(joined)),
        "both_finite": int(both.sum()),
        "nan_filled": int(filled.sum()),
        "nan_lost": int(lost.sum()),
        "differing_finite": int(differing.sum()),
        "max_rel_diff": float(rel[both].max()) if both.any() else None,
        "max_rel_diff_after_year1": (
            float(rel[late_both].max()) if late_both.any() else None
        ),
        "differing_finite_after_year1": int((differing & late).sum()),
        "nan_filled_after_year1": int((filled & late).sum()),
        "by_month": by_month,
    }


def _done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {
        json.loads(line)["factor"]
        for line in out_path.read_text().splitlines()
        if line.strip()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument(
        "--symbols-file",
        required=True,
        help="one symbol per line — the REQUESTED universe, which must be the "
        "runner's (996 for the CSI500 plane), NOT the frozen panel's symbol list "
        "(995: 300114.SZ has no minute bars and never entered the panel). The "
        "distinction is load-bearing for the cross-sectional factor.",
    )
    parser.add_argument("--factors", nargs="+", required=True)
    parser.add_argument("--out", default="panel_leg_probe.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    symbols = [
        s.strip() for s in Path(args.symbols_file).read_text().splitlines() if s.strip()
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.force else _done(out_path)
    print(f"universe: {len(symbols)} symbols | emit {EVAL_START.date()}..{EVAL_END.date()}")
    for factor_id in args.factors:
        if factor_id in done:
            print(f"{factor_id}: SKIP (already in {out_path})")
            continue
        print(f"{factor_id}: running...", flush=True)
        record = compare(factor_id, symbols, args.cache_root, args.panel_dir)
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"{factor_id}: both_finite={record['both_finite']:,} "
            f"NaN->finite={record['nan_filled']:,} finite->NaN={record['nan_lost']:,} "
            f"differing_finite={record['differing_finite']:,} "
            f"max_rel={record['max_rel_diff']} "
            f"| after year 1: differing={record['differing_finite_after_year1']:,} "
            f"filled={record['nan_filled_after_year1']:,} "
            f"max_rel={record['max_rel_diff_after_year1']} "
            f"| wall={record['wall_seconds']}s live_calls={record['live_calls']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
