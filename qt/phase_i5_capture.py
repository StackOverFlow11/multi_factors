"""D6c I5 legacy capture + score reconciliation harness (PR-1).

D6c switches the two intraday runners — ``qt.intraday_tail_framework`` (the
I5a/I5b/I5c/I5f tail runner) and ``qt.intraday_group_backtest`` (the I5d/I5e
quintile runner) — from the legacy score path (``data.clean.intraday_aggregate.
asof_daily_features``, formerly wrapped by the now-retired
``intraday_tail_framework._score_panel``) to the factor service
(``factors.service`` over the D6c-registered
``intraday_ret_0930_1450`` / ``intraday_mmp20_ew_0930_1450`` factors). Before
any runner was touched, this module captured what the LEGACY code produced —
the same "freeze first, reconcile against the frozen copy" discipline as D6b's
``qt.phase3_capture``.

Two capture legs
----------------
RUNNER leg (:func:`capture_runner`): call ``run_phase_i5a_intraday`` /
``run_phase_i5d_intraday_groups`` and export the returned result object as
full-precision JSON — every leaf of NAV / event / feasibility / holdings logs,
group NAV/metrics/spread/monotonicity, limit counts, liquidity diagnostics,
score coverage and factor diagnostics. The ONLY excluded fields are the
explicit :data:`EXCLUDED_RESULT_FIELDS` list (wall-clock timing, the config
object which carries the secret-file path, and output paths — locations, not
values). The runner's own writes (report/log/figures) still happen — this is a
capture OF the legacy runner, not a reimplementation.

SCORE leg (:func:`capture_score`) — the load-bearing evidence captured BEFORE
the switch: on the SAME bars the runner loads, the legacy score (derived here
by :func:`_legacy_score_panel`, the retired runner helper's exact semantics)
is compared CELL BY CELL against the service path (``factors.service.panel``
over the registered factor, ``view=DECISION, basis=EXEC_TO_EXEC``, per
decision date, served through the shared value store with the cache-only
``CacheMinuteProvider``). The reconcile JSON records max abs diff / NaN-mask
mismatch / index-set differences with a 0.0 verdict — value-level proof that
the newly registered factor IS the legacy hook on real data.

Data-plane evidence
-------------------
Both legs record their live-API footprint. The runners' minute reads are
structurally read-only (``stk_mins_live_calls=0``); every daily endpoint flows
through the P4 read-through cache the runner builds via ``qt.pipeline.
_build_cache`` — the harness wraps that single builder in-process (a pure
passthrough recorder, no behavior change) and diffs ``cache.stats()`` per run.
Any non-zero gap-fetch invalidates the run: STOP and report, never fall back
to another data plane. The score leg additionally records the
``CacheMinuteProvider`` call/live-call counters (``live_calls`` is provably 0 —
``read_range`` has no fetch closure).

Reconcilers
-----------
``compare_json`` / ``compare_panels`` / ``to_jsonable`` are REUSED from
``qt.phase3_capture`` (author-once: 0.0 float tolerance, NaN == NaN, explicit
exclusion list). :func:`compare_score_series` adds the score-leg verdict on
top of ``compare_panels``: legacy cells must ALL be served (a missing legacy
cell is a finding), served-only cells are the store's explicit-NaN footprint
rows (expected, but asserted all-NaN — a finite served-only value is a
finding).

CLI (what the L1/L1p capture runs invoke)::

    python -m qt.phase_i5_capture run --runner tail|group --config CFG --out result.json
    python -m qt.phase_i5_capture capture-score --runner tail|group --config CFG --out-dir DIR
    python -m qt.phase_i5_capture compare-json A.json B.json [--out report.json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.availability_policy import ReturnBasis, View
from factors import registry as factor_registry
from factors.materialize import MaterializeSources
from factors.service import DecisionPoint
from factors.service import panel as serve_panel
from qt.config import RootConfig, load_config
from qt.factor_source import open_factor_value_store
from data.clean.intraday_aggregate import asof_daily_features
from qt.intraday_group_backtest import _load_anchor_minute_bars
from qt.intraday_tail_framework import _load_minute_bars_cache_only
from qt.phase3_capture import compare_json, compare_panels, to_jsonable
from qt.pipeline import _build_cache, _build_universe, _load_panel, _make_logger

__all__ = [
    "EXCLUDED_RESULT_FIELDS",
    "capture_runner",
    "capture_score",
    "compare_score_series",
]

_LOGGER_NAME = "qt.phase_i5_capture"

#: Result-object fields EXCLUDED from the runner-leg capture (same discipline
#: as D6b): timings are not values; the config object carries the secret-file
#: path; report/log/figure paths are locations, not values. Everything else is
#: captured leaf-by-leaf.
EXCLUDED_RESULT_FIELDS = frozenset(
    {
        "config",
        "elapsed",
        "report_path",
        "log_path",
        "figure_paths",
    }
)

#: Runner kind -> (module, entry point). Both runners share the score/loading
#: helpers; what differs is the minute slicing (full window vs anchor dates).
_RUNNERS = {
    "tail": ("qt.intraday_tail_framework", "run_phase_i5a_intraday"),
    "group": ("qt.intraday_group_backtest", "run_phase_i5d_intraday_groups"),
}

#: The D6c-registered factor ids the score leg may serve. The legacy score's
#: resolved column IS the factor id byte-for-byte (pinned by the D6c factor
#: tests); anything else is a loud error, never a silent skip.
_KNOWN_SCORE_FACTORS = frozenset(
    {
        "intraday_ret_0930_1450",
        "intraday_mmp20_ew_0930_1450",
    }
)


# --------------------------------------------------------------------------- #
# Data-plane recorder (in-process passthrough; no behavior change)
# --------------------------------------------------------------------------- #
class _BuildCacheRecorder:
    """Wrap a runner module's ``_build_cache`` to collect the instances it builds.

    The runner creates its read-through cache internally; ``cache.stats()`` is
    per-instance, so the harness can only diff the run's gap-fetch counts by
    observing construction. The wrapper is a pure passthrough — it records and
    returns what the original returned (``None`` included, when caching is
    disabled).
    """

    def __init__(self, module) -> None:
        self._module = module
        self._original = module._build_cache
        self.instances: list = []

    def __enter__(self) -> "_BuildCacheRecorder":
        def wrapper(cfg):
            instance = self._original(cfg)
            self.instances.append(instance)
            return instance

        self._module._build_cache = wrapper
        return self

    def __exit__(self, *exc) -> None:
        self._module._build_cache = self._original

    def gap_fetches(self) -> dict[str, int]:
        """Endpoint -> total gap fetches (live API calls) across the instances."""
        totals: dict[str, int] = {}
        for instance in self.instances:
            if instance is None:
                continue
            for endpoint, count in instance.stats().items():
                totals[endpoint] = totals.get(endpoint, 0) + int(count)
        return totals


# --------------------------------------------------------------------------- #
# Runner-leg capture
# --------------------------------------------------------------------------- #
def capture_runner(runner: str, config_path: str, out_path: str) -> dict:
    """Run one I5 runner and export its result object as full-precision JSON.

    The payload carries a ``data_plane`` section (daily-endpoint gap fetches
    per runner-built cache, plus the result's own ``minute_live_calls`` /
    ``stk_limit_gap_fetches`` leaves inside ``result``) — a non-zero gap-fetch
    count INVALIDATES the capture (the window must be cache-only). Returns the
    payload dict that was written.
    """
    if runner not in _RUNNERS:
        raise ValueError(f"unknown runner {runner!r}; known: {sorted(_RUNNERS)}")
    module_name, entry = _RUNNERS[runner]
    module = importlib.import_module(module_name)
    run = getattr(module, entry)
    with _BuildCacheRecorder(module) as recorder:
        result = run(config_path)
    payload = {
        "runner": runner,
        "config_path": str(config_path),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_fields": sorted(EXCLUDED_RESULT_FIELDS),
        "data_plane": {
            "daily_gap_fetches": recorder.gap_fetches(),
            "n_cache_instances": len(recorder.instances),
        },
        "result": to_jsonable(result, EXCLUDED_RESULT_FIELDS),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, allow_nan=True), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- #
# Score-leg comparison (network-free; unit-tested)
# --------------------------------------------------------------------------- #
def compare_score_series(legacy: pd.Series, served: pd.Series) -> dict:
    """Cell-by-cell compare of the legacy daily score vs the service-served one.

    Cell compare is ``qt.phase3_capture.compare_panels`` (exact float equality,
    two NaNs equal) on the one-column frames — reused, never re-implemented.
    The verdict adds the score-leg's two structural rules:

    * every legacy cell MUST be served (``legacy_only`` empty) — a missing
      legacy cell means the service cannot reproduce the runner's score;
    * served-only cells are the store's explicit-NaN footprint rows for
      requested-but-barless (date, symbol) cells — expected, but asserted
      all-NaN: a FINITE served-only value would mean the service scores a cell
      the legacy path never saw.
    """
    base = compare_panels(legacy.to_frame("score"), served.to_frame("score"))
    column = base["per_column"].get(
        "score",
        {"n_compared": 0, "n_value_mismatch": 0, "nan_mask_mismatch": 0, "max_abs_diff": 0.0},
    )
    served_only = served.index.difference(legacy.index)
    served_only_all_nan = bool(served.loc[served_only].isna().all()) if len(served_only) else True
    verdict_pass = (
        column["max_abs_diff"] == 0.0
        and column["nan_mask_mismatch"] == 0
        and base["n_left_only_index"] == 0
        and served_only_all_nan
    )
    return {
        "n_legacy_rows": base["n_rows_left"],
        "n_served_rows": base["n_rows_right"],
        "n_compared": column["n_compared"],
        "n_value_mismatch": column["n_value_mismatch"],
        "nan_mask_mismatch": column["nan_mask_mismatch"],
        "max_abs_diff": column["max_abs_diff"],
        "n_legacy_only_index": base["n_left_only_index"],
        "legacy_only_index": base["left_only_index"],
        "n_served_only_index": int(len(served_only)),
        "served_only_all_nan": served_only_all_nan,
        "verdict_pass": verdict_pass,
    }


# --------------------------------------------------------------------------- #
# Score-leg capture (replays the runner's load; fills the shared value store)
# --------------------------------------------------------------------------- #
def _load_run_bars(runner: str, cfg: RootConfig, panel: pd.DataFrame, symbols, logger):
    """The runner's EXACT cache-only minute load (full window vs anchor-sliced)."""
    if runner == "tail":
        bars, covered, _uncovered, live_calls = _load_minute_bars_cache_only(cfg, symbols, logger)
        return bars, covered, int(live_calls)
    from runtime.backtest.events import monthly_anchor_pairs, trading_calendar

    pairs = monthly_anchor_pairs(trading_calendar(panel))
    anchor_dates = sorted({pd.Timestamp(d).normalize() for pair in pairs for d in pair})
    load = _load_anchor_minute_bars(cfg, symbols, anchor_dates, logger)
    return load.bars, load.covered, int(load.live_calls)


def _legacy_score_panel(cfg: RootConfig, bars: pd.DataFrame, logger) -> tuple[pd.Series, str]:
    """The RETIRED runner-side legacy score, kept here as the capture reference.

    Byte-for-byte the semantics of the ``_score_panel`` D6c PR-2 deleted from
    ``qt.intraday_tail_framework``: one configured feature through
    ``asof_daily_features`` (per-bar PIT cutoff BEFORE daily grouping), its
    single returned column used exactly. The capture harness owns this legacy
    reference so the score leg still runs after the runners switched to the
    factor service — the D6b ``phase3_capture.legacy_factor_panel`` precedent.
    """
    ic = cfg.intraday
    assert ic is not None
    feats = asof_daily_features(
        bars,
        decision_time=ic.decision_time,
        session_open=ic.session_open,
        features=[ic.score_feature],
    )
    if feats.shape[1] != 1:
        raise ValueError(
            f"expected exactly one feature column for score_feature="
            f"{ic.score_feature!r}, got {list(feats.columns)}."
        )
    col = feats.columns[0]
    logger.info(
        "legacy score (capture reference): feature_key=%s, column=%s, %d rows",
        ic.score_feature, col, len(feats),
    )
    return feats[col].rename("score"), col


def capture_score(runner: str, config_path: str, out_dir: str) -> dict:
    """Score-leg capture: the legacy score vs the factor service, per cell.

    Replays the runner's load sequence (shared cache -> universe -> daily panel
    -> the runner's own cache-only minute load), computes the legacy score,
    then serves the SAME (date, symbol) cells from ``factors.service`` — ONE
    decision date per call, so the store fill covers exactly the cells the
    runner scored (a single range fill would materialize every trading day
    between the outermost anchors — the multi-year minute read the grouped
    runner's anchor slicing exists to avoid). Writes ``legacy_score.parquet``,
    ``served_score.parquet`` and ``score_reconcile.json`` into ``out_dir`` and
    returns the reconcile report.

    The LEGACY side is derived here from ``asof_daily_features`` directly —
    exactly what the retired ``_score_panel`` delegated to (D6b's
    ``phase3_capture.legacy_factor_panel`` precedent: the capture harness owns
    the legacy reference, the switched runners do not).
    """
    if runner not in _RUNNERS:
        raise ValueError(f"unknown runner {runner!r}; known: {sorted(_RUNNERS)}")
    cfg = load_config(config_path)
    ic = cfg.intraday
    assert ic is not None  # the I5 runners' precondition; same configs here
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out / "capture_score.log", name=_LOGGER_NAME)

    cache = _build_cache(cfg)
    _universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    bars, covered, minute_live_calls = _load_run_bars(runner, cfg, panel, symbols, logger)
    legacy, column = _legacy_score_panel(cfg, bars, logger)
    if column not in _KNOWN_SCORE_FACTORS:
        raise ValueError(
            f"score column {column!r} is not a D6c-registered factor id "
            f"({sorted(_KNOWN_SCORE_FACTORS)}); the score leg reconciles the "
            "registered factor against its legacy hook, nothing else."
        )
    factor = factor_registry.build(column, None)
    spec_open = factor.spec.session_open
    if spec_open is not None and spec_open != ic.session_open:
        raise ValueError(
            f"config session_open={ic.session_open!r} != factor spec "
            f"session_open={spec_open!r}; the served values would not be the "
            "legacy score's values."
        )

    dates = pd.DatetimeIndex(
        sorted(pd.Timestamp(d).normalize() for d in legacy.index.get_level_values("date").unique())
    )
    from qt.factor_eval_providers import CacheMinuteProvider, DailyEvalPanelProvider

    provider = CacheMinuteProvider(cfg.data.cache.root_dir)
    sources = MaterializeSources(daily=DailyEvalPanelProvider(panel), minute=provider)
    served_parts: list[pd.DataFrame] = []
    with open_factor_value_store(cfg, logger) as store:
        for date in dates:
            frame = serve_panel(
                [column],
                covered,
                [DecisionPoint(date=date, cutoff=ic.decision_time)],
                store=store,
                sources=sources,
                view=View.DECISION,
                basis=ReturnBasis.EXEC_TO_EXEC,
            )
            served_parts.append(frame)
    served = (
        pd.concat(served_parts)[column]
        if served_parts
        else pd.Series([], dtype=float, name=column, index=legacy.index[:0])
    )
    logger.info(
        "score leg served: %d decision dates x %d covered symbols -> %d rows "
        "(minute provider calls=%d, live_calls=%d)",
        len(dates),
        len(covered),
        len(served),
        provider.calls,
        provider.live_calls,
    )

    legacy.to_frame().to_parquet(out / "legacy_score.parquet")
    served.to_frame().to_parquet(out / "served_score.parquet")
    reconcile = compare_score_series(legacy, served)
    reconcile.update(
        {
            "runner": runner,
            "config_path": str(config_path),
            "factor_id": column,
            "n_decision_dates": int(len(dates)),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_plane": {
                "daily_gap_fetches": cache.stats() if cache is not None else {},
                "runner_minute_live_calls": minute_live_calls,
                "provider_calls": int(provider.calls),
                "provider_live_calls": int(provider.live_calls),
            },
        }
    )
    (out / "score_reconcile.json").write_text(
        json.dumps(reconcile, indent=1, allow_nan=True), encoding="utf-8"
    )
    logger.info(
        "score reconcile: verdict_pass=%s max_abs_diff=%r",
        reconcile["verdict_pass"],
        reconcile["max_abs_diff"],
    )
    return reconcile


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    payload = capture_runner(args.runner, args.config, args.out)
    gap = payload["data_plane"]["daily_gap_fetches"]
    print(f"OK capture runner={args.runner}: daily_gap_fetches={gap} -> {args.out}")
    return 0


def _cmd_capture_score(args: argparse.Namespace) -> int:
    report = capture_score(args.runner, args.config, args.out_dir)
    print(
        f"OK capture-score runner={args.runner}: factor={report['factor_id']} "
        f"max_abs_diff={report['max_abs_diff']!r} "
        f"verdict_pass={report['verdict_pass']} -> {args.out_dir}"
    )
    return 0 if report["verdict_pass"] else 1


def _cmd_compare_json(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    report = compare_json(left.get("result", left), right.get("result", right))
    text = json.dumps(report, indent=1, allow_nan=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"compare-json: n_diffs={report['n_diffs']}")
    return 0 if report["n_diffs"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qt.phase_i5_capture", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Runner-leg capture (runs the legacy runner).")
    p_run.add_argument("--runner", required=True, choices=sorted(_RUNNERS))
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser(
        "capture-score",
        help="Score-leg capture (legacy score reference vs the factor service).",
    )
    p_score.add_argument("--runner", required=True, choices=sorted(_RUNNERS))
    p_score.add_argument("--config", required=True)
    p_score.add_argument("--out-dir", required=True)
    p_score.set_defaults(func=_cmd_capture_score)

    p_cj = sub.add_parser("compare-json", help="Leaf-wise compare of two captures.")
    p_cj.add_argument("left")
    p_cj.add_argument("right")
    p_cj.add_argument("--out", default=None)
    p_cj.set_defaults(func=_cmd_compare_json)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
