"""D1 end-of-stage RAW factor-value panel freeze (the pre-D2 baseline for D5).

Freezes the closing 14 factors' RAW daily factor-value panels (11 minute-derived
factors + the 3 confirmed book factors) on the eleven-factor evaluation data
plane (CSI500 ``000905.SH``, 2021-07-01 .. 2026-06-30, CACHE-ONLY). ``main`` at
the producing SHA still carries the pre-refactor factor math bit-for-bit (the D1
registry PR was dispatch-only), so these panels are the ONLY baseline the D5
cell-by-cell reconciliation can compare against once D2 rewrites the math.

NO FACTOR MATH IS REIMPLEMENTED HERE. Every panel is produced by calling the
SAME loader chain the ``qt/eval_*.py`` runners call, with the SAME constants,
imported FROM the runner modules themselves (identity with the runner call
sites, not a transcription of them):

* minute factors — each runner's private ``_load_*_panel`` (per-symbol
  cache-only 1min read -> ``data.clean`` ``compute_*`` -> daily aggregation),
  then the runner's own restriction to the daily panel dates
  (``load.factor[dates.isin(panel_dates)]``). The frozen value is the RAW
  restricted series, BEFORE ``_process_factors`` (z-score / neutralization are
  shared machinery, not factor math).
* book factors — the runners' ``_build_book_factors()`` +
  ``factor.compute(panel)`` on the enriched daily panel, frozen RAW.

PROVENANCE RULE (design v3.2 §5 leg 4): regenerating this baseline is only
legitimate from a checkout of the pinned pre-D2 producing SHA recorded in the
manifest — NEVER from current code. Rebuilding the baseline from the same tree
that is being validated would make the D5 reconciliation structurally unable to
fail (the ``compare_postmerge.py`` empty-reconciliation failure mode).

Canonical content hash
----------------------
A parquet file's byte hash depends on writer metadata (library versions, page
layout), so the AUTHORITATIVE fingerprint is a canonical CONTENT hash, defined
as sha256 over, in order:

1. the version tag ``"qt.panel_freeze canonical v1"`` + ``\\n``;
2. the row count (ascii) + ``\\n``;
3. the ``date`` level as little-endian int64 epoch-nanoseconds, in canonical
   row order, + ``\\n``;
4. the ``symbol`` level joined with ``\\x1f`` (utf-8), same order, + ``\\n``;
5. the values as little-endian float64 raw bytes, same order, with every NaN
   rewritten to the single canonical ``np.float64("nan")`` bit pattern.

Canonical row order = sort by (date, symbol); duplicate (date, symbol) keys are
an error (the hash would otherwise be order-ambiguous). The column NAME is not
hashed (content equality, not labelling). NaN payload bits are collapsed on
purpose; +0.0 and -0.0 stay distinct (a real IEEE value difference).

Run: ``python -m qt.panel_freeze`` (deliberately NOT registered in qt/cli).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL

CANONICAL_HASH_VERSION = "qt.panel_freeze canonical v1"
DEFAULT_CONFIG = "config/phase_c_jump_amount_corr.yaml"
DEFAULT_OUTPUT_ROOT = "artifacts/refactor_baseline"
#: PanelStore artifact name for THIS run — content-identical to the runners'
#: per-run panel (same cache, same window), under a freeze-specific name so the
#: freeze never clobbers an accepted eval run's ``artifacts/data`` artifact.
PANEL_STORE_NAME = "d1_panel_freeze_daily"
#: Determinism double-run subjects (task §2): two minute factors — including
#: valley_price_quantile, the one loader that also consumes the daily panel —
#: plus one book factor.
DETERMINISM_FACTORS = ("jump_amount_corr_20", "valley_price_quantile_20", "value_ep")
#: data_coverage payload fields reconciled against the shipped eval JSONs.
#: They describe the PROCESSED series at the evaluator boundary (see
#: ``analytics/eval/standard.py::data_coverage``), so the reconciliation
#: re-derives that boundary from the frozen RAW panel via the runners' own
#: ``_process_factors`` — the frozen artifact itself stays raw.
RECONCILE_INT_FIELDS = (
    "panel_rows",
    "evaluation_periods",
    "symbols_evaluated",
    "universe_symbols_declared",
    "dropped_symbols_count",
)
RECONCILE_FLOAT_FIELDS = ("factor_nan_rate",)


# --------------------------------------------------------------------------- #
# Canonical content hash + atomic write (pure, network-free, unit-tested)
# --------------------------------------------------------------------------- #
def _as_single_series(panel: pd.Series | pd.DataFrame) -> pd.Series:
    """Validate and return the one factor series of ``panel`` (no coercion).

    Requires a 2-level MultiIndex named exactly (``date``, ``symbol``) with a
    ``datetime64[ns]`` date level, unique (date, symbol) keys, and numeric
    values. Anything else raises — the freeze never silently reshapes.
    """
    if isinstance(panel, pd.DataFrame):
        if panel.shape[1] != 1:
            raise ValueError(
                f"expected a single-column factor panel; got {panel.shape[1]} columns "
                f"({list(panel.columns)!r})."
            )
        series = panel.iloc[:, 0]
    elif isinstance(panel, pd.Series):
        series = panel
    else:
        raise TypeError(
            f"expected a pandas Series or single-column DataFrame; got "
            f"{type(panel).__name__}."
        )
    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("factor panel index must be a 2-level MultiIndex(date, symbol).")
    if tuple(index.names) != (DATE_LEVEL, SYMBOL_LEVEL):
        raise ValueError(
            f"factor panel index levels must be named ({DATE_LEVEL!r}, {SYMBOL_LEVEL!r}); "
            f"got {tuple(index.names)!r}."
        )
    dates = index.get_level_values(DATE_LEVEL)
    if str(dates.dtype) != "datetime64[ns]":
        raise ValueError(
            f"date level must be datetime64[ns]; got {dates.dtype}. The freeze does "
            "not coerce dtypes — fix the producer."
        )
    if index.has_duplicates:
        dupes = index[index.duplicated()].tolist()[:3]
        raise ValueError(
            f"factor panel has duplicate (date, symbol) keys (e.g. {dupes!r}); the "
            "canonical hash would be order-ambiguous."
        )
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"factor values must be numeric; got dtype {series.dtype}.")
    return series


def canonical_content_hash(panel: pd.Series | pd.DataFrame) -> str:
    """The authoritative content fingerprint (module docstring definition).

    Row-order independent (canonical sort applied first), sensitive to any
    value or (date, symbol) index change, NaN-payload independent.
    """
    series = _as_single_series(panel).sort_index(kind="mergesort")
    dates = series.index.get_level_values(DATE_LEVEL)
    symbols = series.index.get_level_values(SYMBOL_LEVEL)
    values = np.array(series.to_numpy(), dtype="<f8", copy=True)
    values[np.isnan(values)] = np.float64("nan")  # collapse NaN payload bits
    digest = hashlib.sha256()
    digest.update(CANONICAL_HASH_VERSION.encode("utf-8") + b"\n")
    digest.update(str(len(series)).encode("ascii") + b"\n")
    digest.update(np.ascontiguousarray(dates.asi8, dtype="<i8").tobytes() + b"\n")
    digest.update("\x1f".join(str(s) for s in symbols).encode("utf-8") + b"\n")
    digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_write_parquet(panel: pd.Series | pd.DataFrame, path: Path | str) -> str:
    """Atomically write the panel (canonical row order) as parquet; return file sha256.

    tmp-file + ``os.replace`` in the target directory: readers never observe a
    partial file, and a failed write leaves no tmp residue (and never touches an
    existing target).
    """
    series = _as_single_series(panel).sort_index(kind="mergesort")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        series.to_frame().reset_index().to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)  # failure path only; os.replace consumed it on success
    return file_sha256(target)


def file_sha256(path: Path | str) -> str:
    """sha256 of the file bytes (convenience fingerprint; NOT the authority)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen_panel(path: Path | str, factor_id: str) -> pd.Series:
    """Read one frozen panel back into the canonical MultiIndex(date, symbol) series."""
    frame = pd.read_parquet(path)
    if factor_id not in frame.columns:
        raise ValueError(
            f"frozen panel {Path(path).name} carries no column {factor_id!r} "
            f"(columns: {list(frame.columns)!r})."
        )
    series = frame.set_index([DATE_LEVEL, SYMBOL_LEVEL])[factor_id]
    return _as_single_series(series).sort_index(kind="mergesort")


# --------------------------------------------------------------------------- #
# Manifest rows + rendering (pure, unit-tested)
# --------------------------------------------------------------------------- #
MANIFEST_ROW_FIELDS = (
    "factor_id",
    "kind",
    "rows",
    "date_min",
    "date_max",
    "n_symbols",
    "n_nan",
    "mean",
    "std",
    "canonical_sha256",
    "file_sha256",
    "file",
)


def manifest_row(
    factor_id: str,
    kind: str,
    panel: pd.Series | pd.DataFrame,
    canonical_sha256: str,
    file_sha256_hex: str,
    file_name: str,
) -> dict:
    """One manifest record. mean/std are float64 ``Series.mean()``/``std(ddof=1)``
    with NaN skipped (pandas defaults), reported at full precision."""
    series = _as_single_series(panel)
    dates = series.index.get_level_values(DATE_LEVEL)
    return {
        "factor_id": str(factor_id),
        "kind": str(kind),
        "rows": int(len(series)),
        "date_min": dates.min().strftime("%Y-%m-%d") if len(series) else None,
        "date_max": dates.max().strftime("%Y-%m-%d") if len(series) else None,
        "n_symbols": int(series.index.get_level_values(SYMBOL_LEVEL).nunique()),
        "n_nan": int(series.isna().sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "canonical_sha256": canonical_sha256,
        "file_sha256": file_sha256_hex,
        "file": str(file_name),
    }


def render_manifest_markdown(header: dict, rows: list[dict]) -> str:
    """Deterministic Markdown manifest: a header block + one row per factor."""
    lines = ["# D1 panel freeze manifest", ""]
    for key in sorted(header):
        lines.append(f"- **{key}**: {header[key]}")
    lines.append("")
    cols = list(MANIFEST_ROW_FIELDS)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for row in rows:
        cells = []
        for col in cols:
            value = row.get(col)
            cells.append(repr(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def atomic_write_text(text: str, path: Path | str) -> None:
    """tmp + ``os.replace`` text write (same atomicity contract as the parquet)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Recipes: the runner call sites, referenced (not transcribed)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MinuteRecipe:
    """One minute factor's runner-owned build chain.

    ``build`` closes over the runner module's loader AND its constants, so the
    call is identical to the runner's own call site by construction.
    """

    factor_id: str
    stem: str  # the runner's _REPORT_STEM (names the eval JSON artifacts)
    spec: object
    build: Callable[[object, list[str], pd.DataFrame, logging.Logger], object]


def minute_recipes() -> list[MinuteRecipe]:
    """The 11 minute factors, each bound to its runner's loader + constants.

    Imports live inside the function so importing ``qt.panel_freeze`` for the
    network-free unit tests stays light.
    """
    import qt.eval_amp_marginal_anomaly_vol as e
    import qt.eval_intraday_amp_cut as g
    import qt.eval_jump_amount_corr as c
    import qt.eval_minute_ideal_amplitude as d
    import qt.eval_peak_interval_kurtosis as h
    import qt.eval_peak_ridge_amount_ratio as m
    import qt.eval_ridge_minute_return as k
    import qt.eval_valley_price_quantile as ell
    import qt.eval_valley_relative_vwap as i
    import qt.eval_valley_ridge_vwap_ratio as j
    import qt.eval_volume_peak_count as f

    recipes: list[MinuteRecipe] = []

    def add(module, factor, build) -> None:
        spec = factor.spec
        recipes.append(
            MinuteRecipe(
                factor_id=spec.factor_id,
                stem=module._REPORT_STEM,
                spec=spec,
                build=build,
            )
        )

    jac = c.JumpAmountCorrFactor(lookback_days=c.JUMP_LOOKBACK_DAYS)
    add(c, jac, lambda cfg, symbols, panel, logger, _s=jac.spec: c._load_jump_factor_panel(
        cfg, symbols, _s, logger,
        lookback_days=c.JUMP_LOOKBACK_DAYS, min_pairs=c.JUMP_MIN_PAIRS,
    ))

    mia = d.MinuteIdealAmplitudeFactor(lookback_days=d.IDEAL_AMP_LOOKBACK_DAYS)
    add(d, mia, lambda cfg, symbols, panel, logger, _s=mia.spec: d._load_minute_ideal_amp_panel(
        cfg, symbols, _s, logger,
        lookback_days=d.IDEAL_AMP_LOOKBACK_DAYS, lam=d.IDEAL_AMP_LAMBDA,
        min_minutes=d.IDEAL_AMP_MIN_MINUTES,
    ))

    amav = e.AmpMarginalAnomalyVolFactor(lookback_days=e.AMP_ANOMALY_LOOKBACK_DAYS)
    add(e, amav, lambda cfg, symbols, panel, logger, _s=amav.spec: e._load_amp_anomaly_vol_panel(
        cfg, symbols, _s, logger,
        lookback_days=e.AMP_ANOMALY_LOOKBACK_DAYS, min_pool=e.AMP_ANOMALY_MIN_POOL,
        min_selected=e.AMP_ANOMALY_MIN_SELECTED, sigma_k=e.AMP_ANOMALY_SIGMA_K,
    ))

    vpc = f.VolumePeakCountFactor(lookback_days=f.VOLUME_PRV_LOOKBACK_DAYS)
    add(f, vpc, lambda cfg, symbols, panel, logger, _s=vpc.spec: f._load_volume_peak_count_panel(
        cfg, symbols, _s, logger,
        lookback_days=f.VOLUME_PRV_LOOKBACK_DAYS, baseline_days=f.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=f.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=f.VOLUME_PRV_SIGMA_K,
        min_valid_days=f.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=f.VOLUME_PRV_MIN_CLASSIFIABLE,
    ))

    iac = g.IntradayAmpCutFactor(lookback_days=g.AMP_CUT_LOOKBACK_DAYS)
    add(g, iac, lambda cfg, symbols, panel, logger, _s=iac.spec: g._load_amp_cut_panel(
        cfg, symbols, _s, logger,
        lookback_days=g.AMP_CUT_LOOKBACK_DAYS, lam=g.AMP_CUT_LAMBDA,
        min_day_minutes=g.AMP_CUT_MIN_DAY_MINUTES, min_valid_days=g.AMP_CUT_MIN_VALID_DAYS,
        min_cross_section=g.AMP_CUT_MIN_CROSS_SECTION,
    ))

    pik = h.PeakIntervalKurtosisFactor(lookback_days=h.PEAK_INTERVAL_LOOKBACK_DAYS)
    add(h, pik, lambda cfg, symbols, panel, logger, _s=pik.spec: h._load_peak_interval_kurtosis_panel(
        cfg, symbols, _s, logger,
        lookback_days=h.PEAK_INTERVAL_LOOKBACK_DAYS, baseline_days=h.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=h.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=h.VOLUME_PRV_SIGMA_K,
        min_valid_days=h.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=h.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_intervals=h.PEAK_INTERVAL_MIN_INTERVALS,
    ))

    vrv = i.ValleyRelativeVwapFactor(lookback_days=i.VALLEY_VWAP_LOOKBACK_DAYS)
    add(i, vrv, lambda cfg, symbols, panel, logger, _s=vrv.spec: i._load_valley_relative_vwap_panel(
        cfg, symbols, _s, logger,
        lookback_days=i.VALLEY_VWAP_LOOKBACK_DAYS, baseline_days=i.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=i.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=i.VOLUME_PRV_SIGMA_K,
        min_valid_days=i.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=i.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_valley_bars=i.VALLEY_VWAP_MIN_VALLEY_BARS,
    ))

    vrr = j.ValleyRidgeVwapRatioFactor(lookback_days=j.VALLEY_RIDGE_LOOKBACK_DAYS)
    add(j, vrr, lambda cfg, symbols, panel, logger, _s=vrr.spec: j._load_valley_ridge_vwap_ratio_panel(
        cfg, symbols, _s, logger,
        lookback_days=j.VALLEY_RIDGE_LOOKBACK_DAYS, baseline_days=j.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=j.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=j.VOLUME_PRV_SIGMA_K,
        min_valid_days=j.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=j.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_valley_bars=j.VALLEY_RIDGE_MIN_VALLEY_BARS,
        min_ridge_bars=j.VALLEY_RIDGE_MIN_RIDGE_BARS,
    ))

    rmr = k.RidgeMinuteReturnFactor(lookback_days=k.RIDGE_RETURN_LOOKBACK_DAYS)
    add(k, rmr, lambda cfg, symbols, panel, logger, _s=rmr.spec: k._load_ridge_minute_return_panel(
        cfg, symbols, _s, logger,
        lookback_days=k.RIDGE_RETURN_LOOKBACK_DAYS, baseline_days=k.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=k.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=k.VOLUME_PRV_SIGMA_K,
        min_valid_days=k.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=k.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_ridge_bars=k.RIDGE_RETURN_MIN_RIDGE_BARS,
    ))

    vpq = ell.ValleyPriceQuantileFactor(lookback_days=ell.VALLEY_QUANTILE_LOOKBACK_DAYS)
    # The ONE loader that also consumes the daily panel (its reversal
    # neutralization needs daily closes) — the runner passes it positionally.
    add(ell, vpq, lambda cfg, symbols, panel, logger, _s=vpq.spec: ell._load_valley_price_quantile_panel(
        cfg, symbols, _s, panel, logger,
        lookback_days=ell.VALLEY_QUANTILE_LOOKBACK_DAYS,
        baseline_days=ell.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=ell.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=ell.VOLUME_PRV_SIGMA_K,
        min_valid_days=ell.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=ell.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_valley_bars=ell.VALLEY_QUANTILE_MIN_VALLEY_BARS,
        min_cross_section=ell.VALLEY_QUANTILE_MIN_CROSS_SECTION,
        reversal_days=ell.VALLEY_QUANTILE_REVERSAL_DAYS,
    ))

    pra = m.PeakRidgeAmountRatioFactor(lookback_days=m.PEAK_RIDGE_LOOKBACK_DAYS)
    add(m, pra, lambda cfg, symbols, panel, logger, _s=pra.spec: m._load_peak_ridge_amount_ratio_panel(
        cfg, symbols, _s, logger,
        lookback_days=m.PEAK_RIDGE_LOOKBACK_DAYS, baseline_days=m.VOLUME_PRV_BASELINE_DAYS,
        baseline_min_obs=m.VOLUME_PRV_BASELINE_MIN_OBS, sigma_k=m.VOLUME_PRV_SIGMA_K,
        min_valid_days=m.VOLUME_PRV_MIN_VALID_DAYS,
        min_classifiable=m.VOLUME_PRV_MIN_CLASSIFIABLE,
        min_peak_bars=m.PEAK_RIDGE_MIN_PEAK_BARS,
        min_ridge_bars=m.PEAK_RIDGE_MIN_RIDGE_BARS,
    ))

    return recipes


# --------------------------------------------------------------------------- #
# Artifact reconciliation (against the shipped eval JSONs)
# --------------------------------------------------------------------------- #
def reconcile_with_eval_artifact(
    stem: str,
    processed: pd.Series,
    declared_symbols: list[str],
    reports_dir: Path,
) -> dict:
    """Check the frozen panel's coverage against ``{stem}_no_book.json``.

    The JSON's ``data_coverage`` fields describe the PROCESSED factor series at
    the evaluator boundary; ``processed`` here was re-derived from the frozen
    RAW panel through the runners' own ``_process_factors``, so equality proves
    the frozen panel is the same series the shipped evaluation consumed. Any
    mismatch raises — a divergent panel must NOT be recorded (task: stop and
    investigate, never freeze a discrepancy).
    """
    json_path = reports_dir / f"{stem}_no_book.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"eval artifact {json_path.name} not found under {reports_dir} — the "
            "reconciliation target is missing; refusing to freeze unreconciled."
        )
    document = json.loads(json_path.read_text(encoding="utf-8"))
    coverage = None
    for section in document.get("sections", []):
        if section.get("name") == "data_coverage":
            coverage = section.get("payload", {})
    if not coverage:
        raise ValueError(f"{json_path.name} carries no data_coverage payload.")

    values = np.asarray(processed.to_numpy(), dtype=float)
    finite = np.isfinite(values)
    total = int(len(processed))
    evaluated = pd.unique(processed.index.get_level_values(SYMBOL_LEVEL))
    ours: dict[str, object] = {
        "panel_rows": total,
        "evaluation_periods": int(
            pd.unique(processed.index.get_level_values(DATE_LEVEL)).size
        ),
        "symbols_evaluated": int(len(evaluated)),
        "universe_symbols_declared": int(len(declared_symbols)),
        "dropped_symbols_count": len(
            set(map(str, declared_symbols)) - set(map(str, evaluated))
        ),
        # the eval JSON writer rounds floats to 6 decimals (data.quality clean_value)
        "factor_nan_rate": round(1.0 - finite.sum() / total, 6) if total else None,
    }
    checks: dict[str, dict] = {}
    mismatches: list[str] = []
    for field in RECONCILE_INT_FIELDS + RECONCILE_FLOAT_FIELDS:
        expected = coverage.get(field)
        actual = ours[field]
        ok = expected == actual
        checks[field] = {"artifact": expected, "frozen": actual, "ok": bool(ok)}
        if not ok:
            mismatches.append(f"{field}: artifact={expected!r} frozen={actual!r}")
    if mismatches:
        raise ValueError(
            f"frozen panel disagrees with {json_path.name} — NOT freezing: "
            + "; ".join(mismatches)
        )
    return {"artifact": json_path.name, "checks": checks}


# --------------------------------------------------------------------------- #
# Freeze orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FreezeResult:
    """Immutable summary of one panel-freeze run (no secrets)."""

    output_root: Path
    manifest_json: Path
    manifest_md: Path
    rows: tuple[dict, ...]
    header: dict
    elapsed: float


def _git_head_sha() -> str:
    """Best-effort producing SHA (the manifest doc records the authoritative one)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort here
        return "UNKNOWN"


def run_panel_freeze(
    config_path: str = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    resume: bool = False,
) -> FreezeResult:
    """Freeze all 14 RAW factor panels + verify determinism + reconcile artifacts.

    ``resume=True`` completes an interrupted freeze: a minute factor whose panel
    file already exists is NOT rebuilt from the minute cache — its RAW series is
    read back FROM the frozen file and then pushed through the SAME processing +
    eval-artifact reconciliation before being accepted into the manifest (a
    stale or corrupted file fails loudly; nothing is reused unverified). Its
    canonical hash is therefore the hash of the file CONTENT. Book factors are
    always recomputed (cheap). The determinism double-run still rebuilds its
    subjects end-to-end via the runner loaders, so a resumed subject is compared
    fresh-build-vs-frozen-file — a strictly stronger check than two in-process
    builds. Resumed factor ids are disclosed in the manifest header.
    """
    from qt.config import load_config
    from qt.eval_jump_amount_corr import _build_book_factors, _check_preconditions
    from qt.pipeline import (
        _build_cache,
        _build_universe,
        _load_panel,
        _log_run_cache_stats,
        _make_logger,
        _maybe_enrich_covariates,
        _maybe_enrich_value,
        _process_factors,
    )

    started = time.monotonic()
    cfg = load_config(config_path)
    _check_preconditions(cfg)  # tushare + cache-only + PIT index + neutralize, as the runners
    # Freeze-specific PanelStore name: identical content, never clobbers an
    # accepted eval run's per-run panel artifact.
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"output_name": PANEL_STORE_NAME})}
    )

    out_root = Path(output_root)
    panels_dir = out_root / "panels"
    log_path = Path(cfg.output.log_dir) / "panel_freeze.log"
    logger = _make_logger(log_path, name="qt.panel_freeze")
    logger.info("panel freeze: config=%s output_root=%s", config_path, out_root)

    # -- shared data plane, the runners' exact preamble --------------------- #
    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    book_factors = _build_book_factors()
    panel = _maybe_enrich_value(cfg, panel, symbols, book_factors, logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    _log_run_cache_stats(cache, logger)
    panel_dates = pd.Index(
        pd.unique(panel.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    )

    rows: list[dict] = []
    reconciliations: dict[str, dict] = {}
    canonical_by_id: dict[str, str] = {}
    live_calls_total = 0

    # -- book factors (raw, pre-processing) --------------------------------- #
    book_raw = pd.concat(
        [factor.compute(panel).rename(factor.name) for factor in book_factors], axis=1
    )
    for factor in book_factors:
        raw = book_raw[factor.name].rename(factor.name)
        canonical = canonical_content_hash(raw)
        target = panels_dir / f"{factor.name}.parquet"
        sha = atomic_write_parquet(raw, target)
        rows.append(manifest_row(factor.name, "book", raw, canonical, sha, target.name))
        canonical_by_id[factor.name] = canonical
        logger.info("frozen book %s: rows=%d canonical=%s", factor.name, len(raw), canonical)

    # -- minute factors (raw, the runners' own loader chains) ---------------- #
    resumed: list[str] = []
    for recipe in minute_recipes():
        target = panels_dir / f"{recipe.factor_id}.parquet"
        if resume and target.exists():
            raw = read_frozen_panel(target, recipe.factor_id)
            resumed.append(recipe.factor_id)
            covered_note = "resumed-from-file"
        else:
            load = recipe.build(cfg, symbols, panel, logger)
            live_calls = int(load.live_calls)
            live_calls_total += live_calls
            if live_calls != 0:
                raise RuntimeError(
                    f"{recipe.factor_id}: stk_mins_live_calls={live_calls} != 0 — the "
                    "freeze is cache-only by contract."
                )
            raw = load.factor[
                load.factor.index.get_level_values(DATE_LEVEL).isin(panel_dates)
            ].rename(recipe.factor_id)
            covered_note = f"covered={len(load.covered)}/{load.requested}"
        # reconcile BEFORE accepting (a divergent panel is never frozen/kept)
        processed = _process_factors(
            cfg, raw.to_frame(recipe.factor_id), panel
        )[recipe.factor_id]
        reconciliations[recipe.factor_id] = reconcile_with_eval_artifact(
            recipe.stem, processed, symbols, Path(cfg.output.report_dir)
        )
        canonical = canonical_content_hash(raw)
        if recipe.factor_id in resumed:
            sha = file_sha256(target)
        else:
            sha = atomic_write_parquet(raw, target)
        rows.append(
            manifest_row(recipe.factor_id, "minute", raw, canonical, sha, target.name)
        )
        canonical_by_id[recipe.factor_id] = canonical
        logger.info(
            "frozen minute %s: rows=%d %s canonical=%s (reconciled vs %s)",
            recipe.factor_id, len(raw), covered_note, canonical,
            reconciliations[recipe.factor_id]["artifact"],
        )

    # -- determinism double-run (2 minute + 1 book) -------------------------- #
    determinism: dict[str, dict] = {}
    for factor_id in DETERMINISM_FACTORS:
        rebuilt = _rebuild_for_determinism(
            factor_id, cfg, symbols, panel, panel_dates, logger
        )
        second = canonical_content_hash(rebuilt)
        first = canonical_by_id[factor_id]
        determinism[factor_id] = {"first": first, "second": second, "ok": second == first}
        if second != first:
            raise RuntimeError(
                f"determinism double-run FAILED for {factor_id}: {first} != {second}"
            )
        logger.info("determinism double-run ok: %s %s", factor_id, first)

    header = {
        "producing_git_sha": _git_head_sha(),
        "config": str(config_path),
        "universe": f"{cfg.universe.type}:{cfg.universe.index_code}",
        "window": f"{cfg.data.start}..{cfg.data.end}",
        "cache_root": str(cfg.data.cache.root_dir),
        "panel_store_name": PANEL_STORE_NAME,
        "panel_rows": int(len(panel)),
        "panel_dates": int(len(panel_dates)),
        "universe_symbols": int(len(symbols)),
        "stk_mins_live_calls": int(live_calls_total),
        "resumed_factors": ",".join(resumed) if resumed else "none",
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "canonical_hash_version": CANONICAL_HASH_VERSION,
        "determinism_double_run": "; ".join(
            f"{fid}:{'ok' if determinism[fid]['ok'] else 'FAIL'}"
            for fid in DETERMINISM_FACTORS
        ),
        "artifact_reconciliation": (
            f"{len(reconciliations)}/11 minute factors reconciled against "
            "eval_*_no_book.json data_coverage (book factors carry no coverage "
            "fields in the eval artifacts — disclosed, not skipped silently)"
        ),
    }

    manifest = {
        "header": header,
        "rows": rows,
        "reconciliation": reconciliations,
        "determinism": determinism,
    }
    manifest_json = out_root / "manifest.json"
    manifest_md = out_root / "manifest.md"
    atomic_write_text(json.dumps(manifest, indent=2, sort_keys=True), manifest_json)
    atomic_write_text(render_manifest_markdown(header, rows), manifest_md)
    logger.info(
        "panel freeze complete: %d panels, %ss, manifest=%s",
        len(rows), header["elapsed_seconds"], manifest_json,
    )
    return FreezeResult(
        output_root=out_root,
        manifest_json=manifest_json,
        manifest_md=manifest_md,
        rows=tuple(rows),
        header=header,
        elapsed=time.monotonic() - started,
    )


def _rebuild_for_determinism(
    factor_id: str,
    cfg,
    symbols: list[str],
    panel: pd.DataFrame,
    panel_dates: pd.Index,
    logger: logging.Logger,
) -> pd.Series:
    """Second, independent build of one factor's RAW panel (same chains).

    Minute factors re-run the runner loader end-to-end (a fresh cache-store
    read); book factors re-instantiate ``_build_book_factors()`` and recompute
    from the shared enriched panel (the panel load itself is once-per-process,
    exactly as in the runners; its own determinism is covered by the cache
    equivalence suites).
    """
    from qt.eval_jump_amount_corr import _build_book_factors

    for factor in _build_book_factors():
        if factor.name == factor_id:
            return factor.compute(panel).rename(factor_id)
    for recipe in minute_recipes():
        if recipe.factor_id == factor_id:
            load = recipe.build(cfg, symbols, panel, logger)
            return load.factor[
                load.factor.index.get_level_values(DATE_LEVEL).isin(panel_dates)
            ].rename(factor_id)
    raise KeyError(f"unknown determinism factor_id {factor_id!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m qt.panel_freeze [--config ...] [--output-root ...]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "complete an interrupted freeze: existing minute panels are read back "
            "from their frozen files and re-verified (processing + eval-artifact "
            "reconciliation) instead of being rebuilt from the minute cache"
        ),
    )
    args = parser.parse_args(argv)
    result = run_panel_freeze(args.config, args.output_root, resume=args.resume)
    print(f"panels frozen: {len(result.rows)} -> {result.output_root / 'panels'}")
    print(f"manifest: {result.manifest_json}")
    print(f"stk_mins_live_calls: {result.header['stk_mins_live_calls']}")
    print(f"elapsed_seconds: {result.header['elapsed_seconds']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
