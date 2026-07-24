"""Engine-side comparison for the four daily hand anchors (D2, R12 companion).

``qt.hand_anchors_d2`` / ``qt.hand_anchor_rows`` hand-compute momentum_20 /
reversal_20 / liquidity_20 / overnight_mom_20 anchor rows WITHOUT importing the
engine and leave them in ``hand_anchors_d2.json`` under
``daily_pending_engine``. This companion is the one place ALLOWED to import
the engine: it rebuilds the freeze data plane (same config, cache-only), runs
the ops-rewritten factor classes, and compares each pending row at the same
<= 1e-12 relative tolerance. The independence requirement binds the HAND side,
not the comparer (module docstring of hand_anchors_d2).

Run AFTER ``python -m qt.hand_anchor_rows``:

    python -m qt.hand_anchors_engine_values
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from qt.hand_anchors_d2 import OUT_JSON, TOL


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from factors.compute.candidates import (
        LiquidityFactor,
        OvernightMomentumFactor,
        ReversalFactor,
    )
    from factors.compute.momentum import MomentumFactor
    from qt.config import load_config
    from qt.eval_jump_amount_corr import _check_preconditions
    from qt.panel_freeze import DEFAULT_CONFIG, PANEL_STORE_NAME
    from qt.pipeline import _build_cache, _build_universe, _load_panel, _make_logger

    payload = json.loads(Path(OUT_JSON).read_text(encoding="utf-8"))
    pending = payload.get("daily_pending_engine", [])
    if not pending:
        print("no pending daily rows; run qt.hand_anchor_rows first")
        return 1

    cfg = load_config(DEFAULT_CONFIG)
    _check_preconditions(cfg)
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"output_name": PANEL_STORE_NAME})}
    )
    logger = _make_logger(
        Path(cfg.output.log_dir) / "hand_anchors_engine.log",
        name="qt.hand_anchors_engine",
    )
    cache = _build_cache(cfg)
    _, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)

    factories = {
        "momentum_20": MomentumFactor(window=20),
        "reversal_20": ReversalFactor(window=20),
        "liquidity_20": LiquidityFactor(window=20),
        "overnight_mom_20": OvernightMomentumFactor(window=20),
    }
    values = {name: f.compute(panel) for name, f in factories.items()}

    n_bad = 0
    out_rows = []
    for row in pending:
        series = values[row["factor_id"]]
        key = (pd.Timestamp(row["date"]), row["symbol"])
        engine = float(series.loc[key]) if key in series.index else float("nan")
        hand = row["hand"]
        if np.isfinite(hand) and np.isfinite(engine):
            denom = max(abs(hand), abs(engine))
            rel = abs(hand - engine) / denom if denom > 0 else 0.0
        elif np.isnan(hand) and np.isnan(engine):
            rel = 0.0
        else:
            rel = float("inf")
        ok = rel <= TOL
        n_bad += 0 if ok else 1
        out_rows.append({**row, "engine": engine, "rel_diff": rel, "ok": bool(ok)})
        print(
            f"{'OK  ' if ok else 'FAIL'} {row['factor_id']:18s} {row['class']:12s} "
            f"{row['date']} {row['symbol']} hand={hand!r} engine={engine!r} rel={rel:.2e}"
        )

    payload["daily_engine_compared"] = out_rows
    payload["all_ok_daily"] = n_bad == 0
    Path(OUT_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"daily anchors: {len(out_rows)} rows, {n_bad} mismatches -> {OUT_JSON}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(main())
