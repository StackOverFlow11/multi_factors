"""run-factor-eval-reconcile: the four-leg reconciliation harness (D5 C4, commit 3).

Built in C4, consumed by C5 (the full four-leg audit). Three modes plus ONE hard
gate that runs at every mode's entry:

* HARD GATE — the frozen exec baseline (``qt.exec_baseline_freeze``) must verify
  **77/77**. Baseline bytes that are unreadable or modified are a HARD ERROR,
  never a skip: without the baseline there is nothing to reconcile against, and
  a reconcile that "passes" without one is the compare_postmerge failure shape.

* ``--mode panels`` — the raw factor panel from the factor SERVICE
  (``factors.service.panel``, decision view x exec_to_exec) vs the frozen D1
  panel, cell by cell. ``jump_amount_corr_20`` compares against the
  ``pr_c_cutoff_fix`` reference instead (its D1 panel encodes the pre-cutoff
  definition and is kept untouched). Differences are only allowed in the three
  NAMED, machine-checkable classes pre-registered in the handoff §3:

  1. ``per_symbol_trim_fix`` — bounded factor, the row sits in the symbol's
     first ``lookback_depth`` trading dates, direction frozen-NaN -> new-finite
     (the D4 union-trim warmup repair; vs the D1 baseline the expected count is
     zero, because the old runners read per-symbol full windows too).
  2. ``saturation_vs_anchor_truncation`` — valid-day-POOLED factor, early
     region [2021-07-01, 2021-10-31], direction frozen-NaN -> new-finite, and
     the per-month counts must be non-increasing ("按月递减至零"). The minute
     cache's real start is 2015-01-05 while the D1 baseline anchors at
     2021-07-01, so the saturated engine warms pools the baseline never saw.
  3. the jump cutoff reference — handled by the reference-path selection above.

  Anything else (finite->NaN, finite-vs-finite beyond tolerance, a finite value
  on a row the frozen panel does not have) is UNCLASSIFIED and fails the mode.
  Extra all-NaN rows in the served panel are the D4c NaN footprint (registered
  drift #3) and are counted, not failed.

* ``--mode reports`` — the new ``factor_eval_*`` JSON/Markdown artifacts vs the
  frozen exec artifacts, VALUE-LEVEL (never byte-level: §七/§七之二 registered
  the contract-v1.0/v1.1 additions, so byte equality is known-false and must
  not be the pass condition). Every flattened leaf difference must be one of
  the registered items; everything else must be EQUAL (numerics within
  METRIC_REL_TOL). Two registered relaxations:

  * the with-book run in the default ``decision`` book mode carries the
    decision-view book — its Incremental-axis numerics legitimately differ from
    the frozen close-view-book artifact. Those leaves are REPORTED in full
    (class ``book_view_effect``), not gated; the strict with-book gate lives on
    the ``_bookclose`` artifact (legacy-faithful book), which isolates the
    engine from the intended book-view change (the handoff's (a)/(b) split).
  * ``jump_amount_corr_20``: the frozen exec artifact is PRE-cutoff-fix (its
    IC -0.030840 encodes the lookahead), so value differences ARE the declared
    correction. They are only accepted when the new JSON carries the
    ``corrections`` block (contract v1.1); without it they fail. Jump's
    value-level verification lives in the panels leg (cutoff reference) and in
    the post-fix restated numbers, not here.

  Cross-check within the NEW pair: no-book vs with-book ``eval_config`` may
  differ ONLY in ``book_view`` (None vs the run's book mode) — the first time
  the two exec artifacts legally differ (§七之二), not a regression.

* ``--mode anchors`` — the service path produces engine values for the
  ``hand_anchors_d2.json`` rows (same real-cache bars) and reconciles against
  the HAND side. EXPECTED SIGNAL (do not misread as a regression): the five
  ``jump_amount_corr_20`` rows that mismatch the FROZEN panels_d2 engine
  (hand truncated at 14:50 vs the frozen un-truncated engine) must go GREEN
  here — the service carries the corrected, truncated definition. A jump
  anchor mismatch FAILS this mode. A mismatch on a valid-day-pooled factor in
  the early region (hand side clipped at the 2021-07-01 anchor, the service
  saturating to 2015-01-05) is the registered saturation class; anything else
  fails.

Layering: this module is qt-side orchestration (it may wire caches and the
service); the classification rules are pure functions so the unit tests need
no cache, no network, and no real baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from factors import registry as factor_registry
from factors import service as factor_service
from factors.compute.minute.binding import is_valid_day_pooled
from qt.exec_baseline_freeze import (
    DEFAULT_FROZEN_ROOT,
    DEFAULT_MANIFEST,
    FACTORS as REPORT_FACTOR_NAMES,
    FrozenExecBaseline,
)

# --------------------------------------------------------------------------- #
# Constants (each pre-registered in the handoff §3 / catalogue §七/§七之二)
# --------------------------------------------------------------------------- #
FROZEN_PANELS_DIR = Path("artifacts/refactor_baseline/panels")
JUMP_CUTOFF_REFERENCE_DIR = Path("artifacts/refactor_baseline/pr_c_cutoff_fix/panels")
ANCHORS_JSON = Path("artifacts/refactor_baseline/hand_anchors_d2.json")

#: JC1 ruling: rolling-mechanism factors reconcile to <= 1e-12 relative
#: (attributable float reordering; the NaN mask must match EXACTLY).
PANEL_REL_TOL = 1e-12
#: Aggregated metrics (IC / ICIR / spreads) pass through more summation stages
#: than a single factor cell; the gate is still a hard number, reported per run.
METRIC_REL_TOL = 1e-9

EARLY_REGION_LO = pd.Timestamp("2021-07-01")
EARLY_REGION_HI = pd.Timestamp("2021-10-31")

#: factor_id -> the frozen exec artifact's report name (qt.exec_baseline_freeze
#: FACTORS). Closed map: an unknown factor id is a readable error, never a guess.
_FACTOR_TO_REPORT_NAME: dict[str, str] = {
    "jump_amount_corr_20": "jump_amount_corr",
    "minute_ideal_amp_10": "minute_ideal_amplitude",
    "amp_marginal_anomaly_vol_20": "amp_marginal_anomaly_vol",
    "volume_peak_count_20": "volume_peak_count",
    "intraday_amp_cut_10": "intraday_amp_cut",
    "peak_interval_kurtosis_20": "peak_interval_kurtosis",
    "valley_relative_vwap_20": "valley_relative_vwap",
    "valley_ridge_vwap_ratio_20": "valley_ridge_vwap_ratio",
    "ridge_minute_return_20": "ridge_minute_return",
    "valley_price_quantile_20": "valley_price_quantile",
    "peak_ridge_amount_ratio_20": "peak_ridge_amount_ratio",
}
assert set(_FACTOR_TO_REPORT_NAME.values()) == set(REPORT_FACTOR_NAMES)

#: Registered JSON additions (catalogue §七 spec 16->20 keys; §七之二 contract
#: v1.0/v1.1). ``corrections`` is matched as a PREFIX (its leaves are indexed).
ALLOWED_ADDED_JSON_PATHS: frozenset[str] = frozenset({
    "eval_config.view",
    "eval_config.return_basis",
    "eval_config.book_view",
    "eval_contract_version",
    "spec.requires",
    "spec.adjustment",
    "spec.overnight_boundary",
    "spec.lookback_depth",
})
ALLOWED_ADDED_JSON_PREFIXES: tuple[str, ...] = ("corrections",)

#: Registered Markdown additions (§七之二 #3: the four provenance lines; #5: one
#: corrections provenance line per declared correction).
ALLOWED_ADDED_MD_PREFIXES: tuple[str, ...] = (
    "- evaluation contract:",
    "- requires (endpoint inputs):",
    "- adjustment / overnight boundary:",
    "- lookback depth (trailing trading days):",
    "- ⚠️ CORRECTION (",
)

MAX_EXAMPLES = 10


class ReconciliationError(RuntimeError):
    """A reconciliation precondition failed (baseline unreadable, bad input)."""


# --------------------------------------------------------------------------- #
# Hard gate
# --------------------------------------------------------------------------- #
def require_baseline_verified(baseline: FrozenExecBaseline) -> None:
    """The 77/77 hard gate. Any problem is a hard error, never a skip."""
    ok, problems = baseline.verify_all()
    if problems or ok != baseline.file_count:
        raise ReconciliationError(
            f"frozen exec baseline did NOT verify: {ok}/{baseline.file_count} ok; "
            f"{len(problems)} problem(s): {problems[:5]}. Reconciliation without "
            "the baseline is meaningless — fix the baseline first."
        )


def load_verified_baseline(repo_root: Path) -> FrozenExecBaseline:
    """Construct the frozen-baseline reader and run the hard gate."""
    baseline = FrozenExecBaseline(
        Path(repo_root) / DEFAULT_FROZEN_ROOT, Path(repo_root) / DEFAULT_MANIFEST
    )
    require_baseline_verified(baseline)
    return baseline


# --------------------------------------------------------------------------- #
# Shared leaf diff helpers
# --------------------------------------------------------------------------- #
def _flatten(obj: object, prefix: str = "") -> dict[str, object]:
    """Flatten a JSON-like structure to {dotted.path[index]: leaf}."""
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _numeric_rel_diff(old: object, new: object) -> float | None:
    """Relative difference for genuine numerics (bool is NOT numeric here)."""
    if isinstance(old, bool) or isinstance(new, bool):
        return None
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if math.isnan(old) and math.isnan(new):
            return 0.0
        denom = max(abs(old), abs(new))
        return abs(old - new) / denom if denom > 0 else 0.0
    return None


# --------------------------------------------------------------------------- #
# reports mode: value-level JSON/Markdown diff with the registered whitelist
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeafDiff:
    path: str
    old: object
    new: object
    classification: str  # registered_* | unregistered_* | book_view_effect


@dataclass
class ReportDiff:
    """One JSON pair diff: every differing leaf, classified; ok = gate result."""

    name: str
    strict: bool
    diffs: list[LeafDiff] = field(default_factory=list)
    max_numeric_rel_diff: float = 0.0
    ok: bool = True

    def by_class(self, classification: str) -> list[LeafDiff]:
        return [d for d in self.diffs if d.classification == classification]


def _is_registered_addition(path: str) -> bool:
    return path in ALLOWED_ADDED_JSON_PATHS or any(
        path == p or path.startswith(f"{p}.") or path.startswith(f"{p}[")
        for p in ALLOWED_ADDED_JSON_PREFIXES
    )


def diff_report_json(
    old: dict, new: dict, *, name: str, strict: bool, correction_expected: bool
) -> ReportDiff:
    """Diff one (frozen, new) JSON pair leaf by leaf against the registered list.

    ``strict`` gates numeric/string equality (no-book; with-book bookclose).
    Non-strict mode (decision-view book) reports numeric differences as
    ``book_view_effect`` without failing them — the book-view change is
    registered; hiding it would be, gating it would false-positive.
    ``correction_expected`` (jump): value differences are the declared
    correction — accepted ONLY if the new JSON carries a ``corrections`` block.
    """
    result = ReportDiff(name=name, strict=strict)
    old_flat, new_flat = _flatten(old), _flatten(new)
    corrections_present = "corrections" in new and bool(new["corrections"])
    for path in sorted(set(old_flat) | set(new_flat)):
        in_old, in_new = path in old_flat, path in new_flat
        if in_old and in_new:
            old_v, new_v = old_flat[path], new_flat[path]
            rel = _numeric_rel_diff(old_v, new_v)
            if rel is not None:
                result.max_numeric_rel_diff = max(result.max_numeric_rel_diff, rel)
                if rel <= METRIC_REL_TOL:
                    continue
                if correction_expected and corrections_present:
                    cls = "registered_correction_effect"
                elif strict:
                    cls = "unregistered_change"
                else:
                    cls = "book_view_effect"
            elif old_v == new_v:
                continue
            elif (
                path == "spec.version"
                and correction_expected
                and corrections_present
                and old_v == "1.0"
            ):
                cls = "registered_correction_effect"
            else:
                cls = "unregistered_change"
            result.diffs.append(LeafDiff(path, old_v, new_v, cls))
        elif in_new:
            cls = (
                "registered_addition"
                if _is_registered_addition(path)
                else "unregistered_addition"
            )
            result.diffs.append(LeafDiff(path, None, new_flat[path], cls))
        else:
            result.diffs.append(LeafDiff(path, old_flat[path], None, "unregistered_removal"))
    result.ok = not any(d.classification.startswith("unregistered") for d in result.diffs)
    if correction_expected and not corrections_present:
        # Value drift without the structured correction carrier is unexplained.
        result.ok = False
    return result


def diff_report_md(
    old_text: str, new_text: str, *, name: str, correction_expected: bool
) -> ReportDiff:
    """Markdown: every line unique to the NEW file must be a registered addition.

    Lines unique to the OLD file are never registered (the contract additions
    are pure additions); for jump they are the correction's prose effect and
    are reported, not gated.
    """
    result = ReportDiff(name=name, strict=True)
    old_lines = set(old_text.splitlines())
    new_lines = set(new_text.splitlines())
    for line in sorted(new_lines - old_lines):
        cls = (
            "registered_addition"
            if line.startswith(ALLOWED_ADDED_MD_PREFIXES)
            else ("registered_correction_effect" if correction_expected else "unregistered_addition")
        )
        result.diffs.append(LeafDiff("<md>", None, line, cls))
    for line in sorted(old_lines - new_lines):
        cls = "registered_correction_effect" if correction_expected else "unregistered_removal"
        result.diffs.append(LeafDiff("<md>", line, None, cls))
    result.ok = not any(d.classification.startswith("unregistered") for d in result.diffs)
    return result


def check_new_pair_consistency(no_book: dict, with_book: dict) -> list[LeafDiff]:
    """The new no-book vs with-book eval_config may differ ONLY in book_view."""
    problems: list[LeafDiff] = []
    old_cfg = _flatten(no_book.get("eval_config", {}), "eval_config")
    new_cfg = _flatten(with_book.get("eval_config", {}), "eval_config")
    for path in sorted(set(old_cfg) | set(new_cfg)):
        if old_cfg.get(path) != new_cfg.get(path) and path != "eval_config.book_view":
            problems.append(
                LeafDiff(path, old_cfg.get(path), new_cfg.get(path), "unregistered_change")
            )
    return problems


# --------------------------------------------------------------------------- #
# panels mode: cell-by-cell with the three named classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelCellDiff:
    date: str
    symbol: str
    frozen: object
    new: object
    classification: str  # per_symbol_trim_fix | saturation_vs_anchor_truncation | unclassified_*


@dataclass
class PanelDiff:
    factor_id: str
    rows_frozen: int = 0
    rows_new: int = 0
    equal: int = 0
    within_tolerance: int = 0
    nan_footprint_rows: int = 0
    max_rel_diff: float = 0.0
    diffs: list[PanelCellDiff] = field(default_factory=list)
    saturation_by_month: dict[str, int] = field(default_factory=dict)
    saturation_monotonic: bool = True
    ok: bool = True

    def by_class(self, classification: str) -> list[PanelCellDiff]:
        return [d for d in self.diffs if d.classification == classification]


def classify_panel_differences(
    new_values: pd.Series,
    frozen: pd.DataFrame,
    *,
    factor_id: str,
    is_pooled: bool,
    lookback_depth: int,
    tol: float = PANEL_REL_TOL,
    early_lo: pd.Timestamp = EARLY_REGION_LO,
    early_hi: pd.Timestamp = EARLY_REGION_HI,
) -> PanelDiff:
    """Classify every cell difference between the served and the frozen panel.

    ``new_values``: (date, symbol) -> raw value from the service (MultiIndex).
    ``frozen``: the frozen parquet with columns date / symbol / <factor_id>.
    Pure function — the unit tests drive it with synthetic frames.
    """
    result = PanelDiff(factor_id=factor_id)
    frozen = frozen.copy()
    frozen["date"] = pd.to_datetime(frozen["date"])
    frozen_s = frozen.set_index(["date", "symbol"])[factor_id]
    result.rows_frozen = int(len(frozen_s))
    result.rows_new = int(len(new_values))

    # First-lookback_depth trading dates per symbol (for the trim-fix class).
    dates_by_symbol = frozen.groupby("symbol")["date"].agg(lambda s: sorted(set(s)))
    early_dates = {
        sym: set(dates[:lookback_depth]) for sym, dates in dates_by_symbol.items()
    }

    new_idx = new_values.index
    frozen_idx = frozen_s.index

    # Extra served rows: allowed only as the all-NaN D4c footprint (drift #3).
    extra = new_idx.difference(frozen_idx)
    for key in extra:
        value = new_values.loc[key]
        if pd.isna(value):
            result.nan_footprint_rows += 1
        else:
            result.diffs.append(
                PanelCellDiff(str(key[0].date()), str(key[1]), None, float(value),
                              "unregistered_new_finite_row")
            )

    for (date, symbol), frozen_v in frozen_s.items():
        new_v = new_values.loc[(date, symbol)] if (date, symbol) in new_idx else float("nan")
        if pd.isna(frozen_v) and pd.isna(new_v):
            result.equal += 1
            continue
        if pd.notna(frozen_v) and pd.notna(new_v):
            denom = max(abs(frozen_v), abs(new_v))
            rel = abs(frozen_v - new_v) / denom if denom > 0 else 0.0
            result.max_rel_diff = max(result.max_rel_diff, rel)
            if rel <= tol:
                result.within_tolerance += 1
            else:
                result.diffs.append(
                    PanelCellDiff(str(date.date()), str(symbol), float(frozen_v),
                                  float(new_v), "unclassified_finite_vs_finite")
                )
            continue
        if pd.notna(frozen_v):  # finite -> NaN: never in an allowed class
            result.diffs.append(
                PanelCellDiff(str(date.date()), str(symbol), float(frozen_v), None,
                              "unclassified_frozen_finite_new_nan")
            )
            continue
        # frozen NaN -> new finite: the two named classes.
        if not is_pooled and date in early_dates.get(symbol, set()):
            cls = "per_symbol_trim_fix"
        elif is_pooled and early_lo <= date <= early_hi:
            cls = "saturation_vs_anchor_truncation"
            month = str(date.to_period("M"))
            result.saturation_by_month[month] = result.saturation_by_month.get(month, 0) + 1
        else:
            cls = "unclassified_nan_to_finite"
        result.diffs.append(
            PanelCellDiff(str(date.date()), str(symbol), None, float(new_v), cls)
        )

    counts = [result.saturation_by_month[m] for m in sorted(result.saturation_by_month)]
    result.saturation_monotonic = all(b <= a for a, b in zip(counts, counts[1:]))
    result.ok = (
        not any(d.classification.startswith("unclassified") or
                d.classification.startswith("unregistered") for d in result.diffs)
        and result.saturation_monotonic
    )
    return result


def frozen_panel_path(factor_id: str, repo_root: Path) -> Path:
    """The frozen reference for the panels leg (jump -> the cutoff reference)."""
    if factor_id == "jump_amount_corr_20":
        return Path(repo_root) / JUMP_CUTOFF_REFERENCE_DIR / f"{factor_id}.parquet"
    return Path(repo_root) / FROZEN_PANELS_DIR / f"{factor_id}.parquet"


# --------------------------------------------------------------------------- #
# anchors mode: the service path vs the hand-computed anchor rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorRowResult:
    factor_id: str
    cls: str
    date: str
    symbol: str
    hand: float
    service: float
    rel_diff: float
    classification: str  # ok | saturation_vs_anchor_truncation | failed


@dataclass
class AnchorsDiff:
    factor_id: str
    rows: list[AnchorRowResult] = field(default_factory=list)
    ok: bool = True

    def by_class(self, classification: str) -> list[AnchorRowResult]:
        return [r for r in self.rows if r.classification == classification]


def classify_anchor_row(
    *,
    factor_id: str,
    cls: str,
    date: pd.Timestamp,
    symbol: str,
    hand: float,
    service: float,
    is_pooled: bool,
    tol: float,
) -> AnchorRowResult:
    """One anchor row. jump MUST reconcile (the corrected-definition signal)."""
    if math.isnan(hand) and math.isnan(service):
        rel = 0.0
    elif math.isnan(hand) or math.isnan(service):
        rel = math.inf
    else:
        denom = max(abs(hand), abs(service))
        rel = abs(hand - service) / denom if denom > 0 else 0.0
    if rel <= tol:
        classification = "ok"
    elif factor_id == "jump_amount_corr_20":
        # The five frozen-engine mismatches must go GREEN on the service path;
        # a miss here means the service does not carry the truncated definition.
        classification = "failed"
    elif is_pooled and EARLY_REGION_LO <= date <= EARLY_REGION_HI:
        classification = "saturation_vs_anchor_truncation"
    else:
        classification = "failed"
    return AnchorRowResult(
        factor_id, cls, str(date.date()), symbol, hand, service, rel, classification
    )


def load_anchor_rows(factor_id: str, repo_root: Path) -> list[dict]:
    """The factor's hand-anchor rows (placeholder rows carry no date/symbol)."""
    path = Path(repo_root) / ANCHORS_JSON
    if not path.exists():
        raise ReconciliationError(f"hand anchors not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        row
        for row in payload.get("frozen14", [])
        if row.get("factor_id") == factor_id and row.get("symbol") and row.get("date")
    ]


# --------------------------------------------------------------------------- #
# The real-cache orchestration (qt-side wiring; NOT unit-tested)
# --------------------------------------------------------------------------- #
def _report_name(factor_id: str) -> str:
    try:
        return _FACTOR_TO_REPORT_NAME[factor_id]
    except KeyError:
        raise ReconciliationError(
            f"{factor_id!r} has no frozen exec artifact name mapping."
        ) from None


def _build_bundle(cfg, logger, symbols=None, value_factors=(), store_root=None):
    """The service bundle for panels/anchors (same wiring as the runner)."""
    from qt.factor_eval_providers import (
        DEFAULT_STORE_ROOT,
        CacheMinuteProvider,
        DailyEvalPanelProvider,
    )
    from factors.materialize import MaterializeSources
    from factors.store import FactorValueStore
    from qt.pipeline import _build_cache, _load_panel, _maybe_enrich_value

    cache = _build_cache(cfg)
    if symbols is None:
        from qt.pipeline import _build_universe

        _universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, list(symbols), logger, cache)
    panel = _maybe_enrich_value(cfg, panel, list(symbols), list(value_factors), logger, cache)
    sources = MaterializeSources(
        daily=DailyEvalPanelProvider(panel),
        minute=CacheMinuteProvider(cfg.data.cache.root_dir),
    )
    store = FactorValueStore(store_root or DEFAULT_STORE_ROOT)
    return store, sources, panel, list(symbols), cache


def run_panels_mode(config_path: str, factor_id: str, repo_root: Path) -> PanelDiff:
    from data.availability_policy import ReturnBasis, View
    from qt.config import load_config
    from qt.pipeline import _make_logger

    load_verified_baseline(repo_root)
    cfg = load_config(config_path)
    logger = _make_logger(
        Path(cfg.output.log_dir) / f"factor_eval_reconcile_{factor_id}.log",
        name="qt.factor_eval_reconcile",
    )
    store, sources, daily_panel, symbols, _cache = _build_bundle(cfg, logger)
    decisions = [
        factor_service.DecisionPoint(d)
        for d in pd.Index(pd.unique(daily_panel.index.get_level_values("date"))).sort_values()
    ]
    served = factor_service.panel(
        [factor_id], symbols, decisions, store=store, sources=sources,
        view=View.DECISION, basis=ReturnBasis.EXEC_TO_EXEC,
    )
    live_calls = int(getattr(sources.minute, "live_calls", 0))
    if live_calls != 0:
        raise ReconciliationError(
            f"cache-only violated: stk_mins_live_calls={live_calls}. ABORT."
        )
    factor = factor_registry.build(factor_id)
    frozen = pd.read_parquet(frozen_panel_path(factor_id, repo_root))
    result = classify_panel_differences(
        served[factor_id],
        frozen,
        factor_id=factor_id,
        is_pooled=is_valid_day_pooled(factor),
        lookback_depth=int(factor.spec.lookback_depth),
    )
    logger.info(
        "panels %s: rows frozen=%d new=%d equal=%d tol=%d trim_fix=%d saturation=%d "
        "footprint=%d unclassified=%d max_rel=%.3e live_calls=%d ok=%s",
        factor_id, result.rows_frozen, result.rows_new, result.equal,
        result.within_tolerance, len(result.by_class("per_symbol_trim_fix")),
        len(result.by_class("saturation_vs_anchor_truncation")),
        result.nan_footprint_rows,
        len([d for d in result.diffs if d.classification.startswith("un")]),
        result.max_rel_diff, live_calls, result.ok,
    )
    return result


def run_reports_mode(
    config_path: str, factor_id: str, repo_root: Path, *, report_dir: Path | None = None
) -> list[ReportDiff]:
    from qt.config import load_config

    baseline = load_verified_baseline(repo_root)
    cfg = load_config(config_path)
    report_dir = report_dir or Path(cfg.output.report_dir)
    report_name = _report_name(factor_id)
    correction_expected = factor_id == "jump_amount_corr_20"
    stem = f"factor_eval_{factor_id}"

    results: list[ReportDiff] = []
    new_no_book = json.loads((report_dir / f"{stem}_exec_no_book.json").read_text())
    new_with_book = json.loads((report_dir / f"{stem}_exec_with_book.json").read_text())

    pairs = [
        ("no_book", new_no_book, True),
        # decision-view book: registered numeric drift, reported not gated.
        ("with_book(decision)", new_with_book, False),
    ]
    bookclose = report_dir / f"{stem}_exec_with_book_bookclose.json"
    if bookclose.exists():
        pairs.append(
            ("with_book(bookclose)", json.loads(bookclose.read_text()), True)
        )
    for label, new_json, strict in pairs:
        book = "no_book" if label == "no_book" else "with_book"
        frozen_json = baseline.report_json(report_name, book)
        results.append(
            diff_report_json(
                frozen_json, new_json, name=f"{stem}_exec_{book}.json[{label}]",
                strict=strict, correction_expected=correction_expected,
            )
        )
        new_md = (report_dir / f"{stem}_exec_{book}{'_bookclose' if 'bookclose' in label else ''}.md").read_text()
        frozen_md = baseline.read_text(f"eval_{report_name}_exec_{book}.md")
        results.append(
            diff_report_md(
                frozen_md, new_md, name=f"{stem}_exec_{book}.md[{label}]",
                correction_expected=correction_expected,
            )
        )
    problems = check_new_pair_consistency(new_no_book, new_with_book)
    if problems:
        results.append(
            ReportDiff(
                name="new-pair eval_config consistency", strict=True,
                diffs=problems, ok=False,
            )
        )
    return results


def run_anchors_mode(config_path: str, factor_id: str, repo_root: Path) -> AnchorsDiff:
    from data.availability_policy import ReturnBasis, View
    from qt.config import load_config
    from qt.hand_anchors_d2 import TOL
    from qt.pipeline import _make_logger

    load_verified_baseline(repo_root)
    rows = load_anchor_rows(factor_id, repo_root)
    if not rows:
        raise ReconciliationError(f"no hand-anchor rows for {factor_id!r}.")
    factor = factor_registry.build(factor_id)
    try:
        pooled = is_valid_day_pooled(factor)
    except KeyError:
        pooled = False  # daily factors are not in the minute partition sets
    cfg = load_config(config_path)
    logger = _make_logger(
        Path(cfg.output.log_dir) / f"factor_eval_reconcile_anchors_{factor_id}.log",
        name="qt.factor_eval_reconcile",
    )
    symbols = sorted({row["symbol"] for row in rows})
    value_factors = ()
    if factor_id in ("value_ep", "value_bp"):
        from qt.factor_eval_runner import _build_book_factors

        value_factors = tuple(_build_book_factors())
    store, sources, _panel, symbols, _cache = _build_bundle(
        cfg, logger, symbols=symbols, value_factors=value_factors
    )
    decisions = [factor_service.DecisionPoint(pd.Timestamp(row["date"])) for row in rows]
    served = factor_service.panel(
        [factor_id], symbols, decisions, store=store, sources=sources,
        view=View.DECISION, basis=ReturnBasis.EXEC_TO_EXEC,
    )
    live_calls = int(getattr(sources.minute, "live_calls", 0))
    if live_calls != 0:
        raise ReconciliationError(
            f"cache-only violated: stk_mins_live_calls={live_calls}. ABORT."
        )
    result = AnchorsDiff(factor_id=factor_id)
    series = served[factor_id]
    for row in rows:
        key = (pd.Timestamp(row["date"]), row["symbol"])
        service_v = float(series.loc[key]) if key in series.index else float("nan")
        result.rows.append(
            classify_anchor_row(
                factor_id=factor_id, cls=row["class"], date=key[0], symbol=key[1],
                hand=float(row["hand"]), service=service_v,
                is_pooled=pooled, tol=TOL,
            )
        )
    result.ok = all(r.classification != "failed" for r in result.rows)
    logger.info(
        "anchors %s: %d rows, ok=%d saturation=%d failed=%d live_calls=%d -> %s",
        factor_id, len(result.rows), len(result.by_class("ok")),
        len(result.by_class("saturation_vs_anchor_truncation")),
        len(result.by_class("failed")), live_calls, result.ok,
    )
    return result


__all__ = [
    "ALLOWED_ADDED_JSON_PATHS",
    "ALLOWED_ADDED_MD_PREFIXES",
    "AnchorsDiff",
    "EARLY_REGION_HI",
    "EARLY_REGION_LO",
    "LeafDiff",
    "METRIC_REL_TOL",
    "PANEL_REL_TOL",
    "PanelDiff",
    "ReconciliationError",
    "ReportDiff",
    "check_new_pair_consistency",
    "classify_anchor_row",
    "classify_panel_differences",
    "diff_report_json",
    "diff_report_md",
    "frozen_panel_path",
    "load_anchor_rows",
    "load_verified_baseline",
    "require_baseline_verified",
    "run_anchors_mode",
    "run_panels_mode",
    "run_reports_mode",
]
