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
  definition and is kept untouched). Differences are only allowed in the
  NAMED, machine-checkable classes registered in catalogue §七之三:

  1. ``warmup_left_extension`` — the old runners anchored their left edge at
     ``data.start`` (2021-07-01) and never extended the window, so the frozen
     panel's first w-1 trading dates are under-warmed (NaN or a partial pool);
     the materializer saturates left to the minute cache's real start
     (2015-01-05). This is a loading-geometry difference, not a value bug:
     * bounded factor: the differing row sits in the grid's first
       ``lookback_depth - 1`` trading dates — ALL THREE directions allowed
       (frozen-NaN -> new-finite, finite -> finite as a partial pool fills,
       and a finite value on a row the frozen panel does not have);
     * valid-day-POOLED factor: the early region [2021-07-01, 2021-10-31],
       same three directions, and the per-month counts must be
       non-increasing ("按月递减至零"; a violation fails the mode).
  2. ``float_reordering_tail`` — scattered finite-vs-finite cells with rel
     diff <= 5e-12 (rolling-correlation summation order; the JC1 1e-12 gate
     is the attributable floor, this is the measured tail above it). The
     class is CAPPED at 101 cells (the measured jump count); more fails.
  3. ``threshold_flip_tail`` — count factors (volume_peak): rolling-sigma
     float noise (~4e-10) times an integer volume sitting on the peak
     threshold flips the count by EXACTLY +/-1 on a sparse (symbol, day)
     cluster (measured: 20 cells, 600623.SH 2023-06-15..07-14). Bounds:
     |delta| == 1 exactly, rel <= 1e-2, at most 25 cells; more fails.
  4. the jump cutoff reference — handled by the reference-path selection above.

  Anything else (finite->NaN, finite-vs-finite beyond every named tail, a
  finite value on a row the frozen panel does not have outside the warmup
  boundary) is UNCLASSIFIED and fails the mode. Extra all-NaN rows in the
  served panel are the D4c NaN footprint (registered drift #3) and are
  counted, not failed.

* ``--mode reports`` — the new ``factor_eval_*`` JSON/Markdown artifacts vs the
  frozen exec artifacts, VALUE-LEVEL (never byte-level: §七/§七之二 registered
  the contract-v1.0/v1.1 additions, so byte equality is known-false and must
  not be the pass condition). Every flattened leaf difference must be one of
  the registered items; everything else must be EQUAL (numerics within
  METRIC_REL_TOL). Registered relaxations (catalogue §七之三):

  * ``warmup_aggregate_effect`` — aggregate-metric leaves (``sections[*]``,
    ``verdict.reasons[*]``, ``verdict.axes.*``) legitimately move: they
    aggregate a panel whose warmup cells moved (the panels leg is the value
    gate). Only DIGIT-CARRYING changes qualify — a pure label flip (verdict
    PASS -> FAIL carries no digit) stays unregistered and fails.
  * ``registered_sanity_stem_rename`` — the runner's deliberate
    ``eval_<name>_`` -> ``factor_eval_<factor_id>_`` sanity-report rename
    (anti-collision).
  * ``registered_run_order_artifact`` — ``exec_price_artifact_reused``
    False -> True within one session (later runs reuse the artifact the
    first run built).
  * the with-book run in the default ``decision`` book mode carries the
    decision-view book — its Incremental-axis numerics legitimately differ from
    the frozen close-view-book artifact. Those leaves are REPORTED in full
    (class ``book_view_effect``), not gated; the strict with-book gate lives on
    the ``_bookclose`` artifact (legacy-faithful book), which isolates the
    engine from the intended book-view change (the handoff's (a)/(b) split).
  * ``jump_amount_corr_20``: the frozen exec artifact is PRE-cutoff-fix (its
    IC -0.030840 encodes the lookahead), so value differences ARE the declared
    correction — including ``spec.version`` / ``spec.description`` /
    ``sections[7].payload.factor_version``. They are only accepted when the
    new JSON carries the ``corrections`` block (contract v1.1); without it
    they fail. Jump's value-level verification lives in the panels leg
    (cutoff reference) and in the post-fix restated numbers, not here.
  * ``valley_price_quantile_20`` (§七之四): the unified runner publishes the
    NeutralizationCoverage disclosure as an add-Section; the legacy runner
    only LOGGED it (catalogue §三 mechanism B), so the frozen artifact has no
    such section. The whole ``neutralization_coverage`` subtree (JSON leaves +
    the section's MD lines) is a registered addition, matched by section NAME
    (``REGISTERED_EXTRA_SECTIONS``), never by index.

  Markdown is diffed as a line set, then same-key lines are PAIRED into
  row-level changes before classification (a bare set diff reports every
  value change twice — one removal + one addition with the same ``- key:``
  head).

  Cross-check within the NEW pair: no-book vs with-book ``eval_config`` may
  differ ONLY in ``book_view`` (None vs the run's book mode) — the first time
  the two exec artifacts legally differ (§七之二), not a regression.

* ``--mode anchors`` — the service path produces engine values for the
  ``hand_anchors_d2.json`` rows (same real-cache bars) and reconciles against
  the HAND side. A cross-sectional factor (``stores_intermediate`` — its
  served value is a function of the requested universe, D4c) is served over
  the FULL config universe, because the hand side was computed over it and a
  5-name request both empties the panel (below the combine's cross-section
  floor) and asks a different question; per-symbol factors keep the cheap
  anchor-symbols-only request. The hand side was computed with the OLD
  loading geometry (left edge anchored at 2021-07-01), so a mismatch inside
  the warmup boundary is the registered ``warmup_left_extension`` class —
  for bounded factors the frozen grid's first ``lookback_depth - 1`` trading
  dates (the harness never had this class for bounded factors before §七之三;
  the first-run failures were exactly this asymmetry). EXPECTED SIGNAL (do
  not misread as a regression): jump's NON-warmup rows must reconcile — the
  service carries the corrected, truncated definition, so the frozen-engine
  mismatches on random dates go GREEN here (rel ~1e-15). A jump mismatch
  outside the warmup boundary FAILS this mode.

Layering: this module is qt-side orchestration (it may wire caches and the
service); the classification rules are pure functions so the unit tests need
no cache, no network, and no real baseline.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
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

#: ``float_reordering_tail`` (catalogue §七之三): scattered finite-vs-finite
#: cells just above the JC1 floor (measured jump tail: 101 cells, rel
#: 1.0e-12..2.9e-12, rolling-correlation summation order). Bounded on BOTH
#: axes — the global tolerance is NOT widened, so a real regression keeps
#: failing; a flood of float-noise cells fails the cap.
FLOAT_TAIL_REL_TOL = 5e-12
FLOAT_TAIL_MAX_CELLS = 101

#: ``threshold_flip_tail`` (catalogue §七之三): count factors (volume_peak) —
#: rolling-sigma float noise (~4e-10) times an integer volume sitting on the
#: peak threshold flips the count by EXACTLY +/-1. Measured: 20 cells,
#: 600623.SH 2023-06-15..07-14. An amplitude > 1 or a larger cluster fails.
THRESHOLD_FLIP_REL_TOL = 1e-2
THRESHOLD_FLIP_MAX_CELLS = 25

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
#: v1.0/v1.1). ``corrections`` and ``spec.requires`` are matched as PREFIXES
#: (their leaves are indexed: ``corrections[0]...``, ``spec.requires[0]...``).
ALLOWED_ADDED_JSON_PATHS: frozenset[str] = frozenset({
    "eval_config.view",
    "eval_config.return_basis",
    "eval_config.book_view",
    "eval_contract_version",
    "spec.adjustment",
    "spec.overnight_boundary",
    "spec.lookback_depth",
})
ALLOWED_ADDED_JSON_PREFIXES: tuple[str, ...] = ("corrections", "spec.requires")

#: Aggregate-metric leaves: numeric/prose changes HERE are the downstream
#: effect of the registered panel-level ``warmup_left_extension`` differences
#: (the panels leg is the value gate; reports are aggregates of the panel).
#: Anything OUTSIDE these paths (verdict labels, spec, eval_config, criteria)
#: still fails on change.
AGGREGATE_CHANGE_PATH_PREFIXES: tuple[str, ...] = (
    "sections[",
    "verdict.reasons[",
    "verdict.axes.",
)

#: Registered Markdown additions (§七之二 #3: the four provenance lines; #5: one
#: corrections provenance line per declared correction).
ALLOWED_ADDED_MD_PREFIXES: tuple[str, ...] = (
    "- evaluation contract:",
    "- requires (endpoint inputs):",
    "- adjustment / overnight boundary:",
    "- lookback depth (trailing trading days):",
    "- ⚠️ CORRECTION (",
)

#: factor_id -> the add-Sections the unified runner publishes that the frozen
#: artifact does NOT carry (§七之四, D5 C4b). valley_price_quantile's
#: NeutralizationCoverage was only LOGGED by the legacy runner (catalogue §三
#: mechanism B — never an add-Section), so the unified runner's new section is
#: an ADDITION against the frozen artifact, registered here per factor; any
#: other added section stays unregistered and fails.
REGISTERED_EXTRA_SECTIONS: dict[str, tuple[str, ...]] = {
    "valley_price_quantile_20": ("neutralization_coverage",),
}

#: The MD note-line prefix of ``NeutralizationCoverage.render()`` — the
#: disclosure's one-line summary, rendered as the section's note. Kept as a
#: constant with a pinning test (the alternative — re-rendering a zero
#: instance — would couple the reconcile gate to the renderer's format by
#: construction instead of by assertion).
_NEUTRALIZATION_NOTE_PREFIX = "neutralization (T-1 rev20):"


def _registered_section_md_prefixes(section_names: tuple[str, ...]) -> tuple[str, ...]:
    """The MD line prefixes belonging to a registered add-Section's rendering.

    An extra section renders (``analytics/eval/render.py._render_section``) as
    an unnumbered ``## + <name>`` heading, the note line, and one
    ``- <field>:`` payload line per dataclass field (derived from the
    dataclass, so a field rename breaks the pinning test instead of silently
    unregistering). An unknown section name is a readable error — registering
    a section whose MD rendering is not catalogued would be a guess.
    """
    from dataclasses import fields as dc_fields

    from qt.factor_eval_disclosures import (
        NEUTRALIZATION_SECTION_NAME,
        NeutralizationCoverage,
    )

    prefixes: list[str] = []
    for name in section_names:
        if name == NEUTRALIZATION_SECTION_NAME:
            prefixes.append(f"## + {NEUTRALIZATION_SECTION_NAME}")
            prefixes.append(_NEUTRALIZATION_NOTE_PREFIX)
            prefixes.extend(f"- {f.name}:" for f in dc_fields(NeutralizationCoverage))
        else:
            raise ValueError(
                f"no MD rendering is registered for added section {name!r}."
            )
    return tuple(prefixes)


def _is_registered_section_addition(
    path: str, new: dict, section_names: tuple[str, ...]
) -> bool:
    """True iff ``path`` is a leaf of one of ``section_names`` in ``new``.

    The check reads the NEW JSON's section at the path's index and matches its
    NAME — never the index alone — so a different extra section landing at the
    same index stays unregistered.
    """
    match = re.match(r"sections\[(\d+)\]", path)
    if match is None or not section_names:
        return False
    sections = new.get("sections") or []
    idx = int(match.group(1))
    if idx >= len(sections) or not isinstance(sections[idx], dict):
        return False
    return sections[idx].get("name") in section_names


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


def _carries_digit(value: object) -> bool:
    return any(ch.isdigit() for ch in str(value))


def _classify_value_change(
    path: str,
    old_v: object,
    new_v: object,
    *,
    correction_expected: bool,
    corrections_present: bool,
    strict: bool,
) -> str:
    """Classify one changed leaf (both sides present, values differ).

    Ladder (catalogue §七之三):
    1. the runner's deliberate ``sanity_report`` stem rename
       (``eval_<name>_`` -> ``factor_eval_<factor_id>_``) — anti-collision;
    2. ``exec_price_artifact_reused`` False -> True — a same-session run-order
       artifact (the first run builds the exec-price artifact, later runs
       reuse it). True -> False is NOT registered;
    3. jump: any value change while the structured ``corrections`` carrier is
       present IS the declared correction (contract v1.1) — this covers
       ``spec.version``, ``spec.description``,
       ``sections[7].payload.factor_version`` and the restated numerics;
    4. the decision-view with-book pair: registered numeric drift, reported
       not gated (``book_view_effect``);
    5. digit-carrying changes on AGGREGATE paths — the downstream effect of
       the registered panel-level warmup differences
       (``warmup_aggregate_effect``). The digit requirement keeps a pure
       label flip (a verdict PASS -> FAIL carries no digit) UNREGISTERED.
    6. everything else: unregistered.
    """
    if (
        path.endswith(".sanity_report")
        and isinstance(old_v, str)
        and isinstance(new_v, str)
        and old_v.endswith("_exec_basis_sanity.md")
        and new_v.endswith("_exec_basis_sanity.md")
    ):
        return "registered_sanity_stem_rename"
    if path.endswith(".exec_price_artifact_reused") and old_v is False and new_v is True:
        return "registered_run_order_artifact"
    if correction_expected and corrections_present:
        return "registered_correction_effect"
    if not strict:
        return "book_view_effect"
    if path.startswith(AGGREGATE_CHANGE_PATH_PREFIXES) and (
        _carries_digit(old_v) or _carries_digit(new_v)
    ):
        return "warmup_aggregate_effect"
    return "unregistered_change"


def diff_report_json(
    old: dict,
    new: dict,
    *,
    name: str,
    strict: bool,
    correction_expected: bool,
    registered_sections: tuple[str, ...] = (),
) -> ReportDiff:
    """Diff one (frozen, new) JSON pair leaf by leaf against the registered list.

    ``strict`` gates numeric/string equality (no-book; with-book bookclose).
    Non-strict mode (decision-view book) reports numeric differences as
    ``book_view_effect`` without failing them — the book-view change is
    registered; hiding it would be, gating it would false-positive.
    ``correction_expected`` (jump): value differences are the declared
    correction — accepted ONLY if the new JSON carries a ``corrections`` block.
    ``registered_sections``: add-Section names whose whole subtree is a
    registered addition (§七之四 — the frozen artifact predates the section).
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
            elif old_v == new_v:
                continue
            cls = _classify_value_change(
                path, old_v, new_v,
                correction_expected=correction_expected,
                corrections_present=corrections_present,
                strict=strict,
            )
            result.diffs.append(LeafDiff(path, old_v, new_v, cls))
        elif in_new:
            if _is_registered_addition(path):
                cls = "registered_addition"
            elif _is_registered_section_addition(path, new, registered_sections):
                cls = "registered_section_addition"
            else:
                cls = "unregistered_addition"
            result.diffs.append(LeafDiff(path, None, new_flat[path], cls))
        else:
            result.diffs.append(LeafDiff(path, old_flat[path], None, "unregistered_removal"))
    result.ok = not any(d.classification.startswith("unregistered") for d in result.diffs)
    if correction_expected and not corrections_present:
        # Value drift without the structured correction carrier is unexplained.
        result.ok = False
    return result


#: Pairing-key length cap. Long prose bullets legitimately pair by their
#: (digit-normalized) head — e.g. the incremental-axis reason line is ~103
#: chars before its first colon — so the cap only excludes pathological
#: lines, not real prose.
_MD_KEY_MAX_CHARS = 200


def _md_line_key(line: str) -> str | None:
    """The pairing key of a ``- key: value`` Markdown line (None = unpairable)."""
    head, sep, _value = line.partition(":")
    return head if sep and len(head) <= _MD_KEY_MAX_CHARS else None


def _md_line_key_digit_normalized(line: str) -> str | None:
    """Fallback key: some prose lines carry the changed number BEFORE the
    first colon (``- [incremental FAIL] orthogonalized ICIR +0.120 is ~ 0
    (redundant with the book): after residualizing ...``), so the exact head
    can never pair them. Normalizing digit runs in the head pairs exactly
    those; a change in the prose WORDS still does not pair (the normalized
    heads differ)."""
    key = _md_line_key(line)
    return re.sub(r"\d+", "#", key) if key is not None else None


def _pair_md_lines(
    removed: list[str], added: list[str]
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Pair removals with additions into row-level changes.

    Two passes: exact ``- key:`` head first, then the digit-normalized head
    for the leftovers. Returns (pairs, unpaired_removals, unpaired_additions).
    """
    pairs: list[tuple[str, str]] = []
    rem, add = list(removed), list(added)
    for keyfn in (_md_line_key, _md_line_key_digit_normalized):
        next_rem: list[str] = []
        next_add: list[str] = []
        rem_by: dict[str, list[str]] = {}
        add_by: dict[str, list[str]] = {}
        for line in rem:
            key = keyfn(line)
            (rem_by.setdefault(key, []) if key is not None else next_rem).append(line)
        for line in add:
            key = keyfn(line)
            (add_by.setdefault(key, []) if key is not None else next_add).append(line)
        for key in sorted(set(rem_by) | set(add_by)):
            olds, news = rem_by.get(key, []), add_by.get(key, [])
            for old_line, new_line in zip(olds, news):
                pairs.append((old_line, new_line))
            next_rem.extend(olds[len(news):])
            next_add.extend(news[len(olds):])
        rem, add = next_rem, next_add
    return pairs, rem, add


def _classify_md_change(
    old_line: str, new_line: str, *, correction_expected: bool, strict: bool
) -> str:
    """One KEY-PAIRED Markdown change; the ladder mirrors the JSON one."""
    old_v, new_v = old_line.partition(":")[2], new_line.partition(":")[2]
    if (
        old_line.startswith("- sanity_report:")
        and old_v.strip().endswith("_exec_basis_sanity.md")
        and new_v.strip().endswith("_exec_basis_sanity.md")
    ):
        return "registered_sanity_stem_rename"
    if correction_expected:
        return "registered_correction_effect"
    if not strict:
        return "book_view_effect"
    if _carries_digit(old_line) and _carries_digit(new_line):
        # A change carrying numbers (in the key OR the value — prose bullets
        # can carry the number before the colon) follows the numeric
        # attribution of the registered panel-level warmup differences. A
        # pure label flip (no digits on either side) stays UNREGISTERED.
        return "warmup_aggregate_effect"
    return "unregistered_change"


def diff_report_md(
    old_text: str,
    new_text: str,
    *,
    name: str,
    strict: bool = True,
    correction_expected: bool,
    registered_section_lines: tuple[str, ...] = (),
) -> ReportDiff:
    """Markdown: line-set diff, then pair same-key lines into row-level CHANGES.

    A bare set difference reports every value change twice (one removal + one
    addition with the same ``- key:`` head) — 83-96 phantom "removals" in the
    first run were exactly that. Pairing first, then classifying the pair with
    the numeric-attribution ladder (the same one as the JSON leg), keeps the
    teeth where they belong: UNPAIRED additions must be registered additions,
    UNPAIRED removals are never registered (except jump's correction prose).
    ``registered_section_lines``: the line prefixes of a registered
    add-Section's rendering (§七之四).
    """
    result = ReportDiff(name=name, strict=strict)
    old_counts = Counter(old_text.splitlines())
    new_counts = Counter(new_text.splitlines())
    removed = sorted((old_counts - new_counts).elements())
    added = sorted((new_counts - old_counts).elements())

    pairs, unpaired_removals, unpaired_additions = _pair_md_lines(removed, added)
    for old_line, new_line in pairs:
        cls = _classify_md_change(
            old_line, new_line,
            correction_expected=correction_expected, strict=strict,
        )
        result.diffs.append(LeafDiff(f"<md:{_md_line_key(old_line)}>", old_line, new_line, cls))
    for line in unpaired_removals:
        cls = "registered_correction_effect" if correction_expected else "unregistered_removal"
        result.diffs.append(LeafDiff("<md>", line, None, cls))
    for line in unpaired_additions:
        if line.startswith(ALLOWED_ADDED_MD_PREFIXES):
            cls = "registered_addition"
        elif registered_section_lines and (
            line == "" or line.startswith(registered_section_lines)
        ):
            # the section's rendering includes its blank separators; blank-line
            # additions register ONLY alongside a registered section (they are
            # unregistered everywhere else).
            cls = "registered_section_addition"
        elif correction_expected:
            cls = "registered_correction_effect"
        else:
            cls = "unregistered_addition"
        result.diffs.append(LeafDiff("<md>", None, line, cls))
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
# panels mode: cell-by-cell with the named classes (catalogue §七之三)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelCellDiff:
    date: str
    symbol: str
    frozen: object
    new: object
    classification: str  # warmup_left_extension | float_reordering_tail |
    # threshold_flip_tail | unclassified_* | unregistered_*


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
    warmup_by_month: dict[str, int] = field(default_factory=dict)
    warmup_by_direction: dict[str, int] = field(default_factory=dict)
    warmup_monotonic: bool = True
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
    float_tail_tol: float = FLOAT_TAIL_REL_TOL,
    float_tail_max: int = FLOAT_TAIL_MAX_CELLS,
    flip_rel_tol: float = THRESHOLD_FLIP_REL_TOL,
    flip_max: int = THRESHOLD_FLIP_MAX_CELLS,
) -> PanelDiff:
    """Classify every cell difference between the served and the frozen panel.

    ``new_values``: (date, symbol) -> raw value from the service (MultiIndex).
    ``frozen``: the frozen parquet with columns date / symbol / <factor_id>.
    Pure function — the unit tests drive it with synthetic frames.

    The warmup boundary is the LEFT-EXTENSION geometry difference: the old
    runners anchored at ``data.start`` (the frozen grid's first w-1 trading
    dates are under-warmed), the materializer saturates to the cache's real
    start. Bounded: the grid's first ``lookback_depth - 1`` trading dates.
    Pooled: the early region [early_lo, early_hi] with non-increasing
    per-month counts.
    """
    result = PanelDiff(factor_id=factor_id)
    frozen = frozen.copy()
    frozen["date"] = pd.to_datetime(frozen["date"])
    frozen_s = frozen.set_index(["date", "symbol"])[factor_id]
    result.rows_frozen = int(len(frozen_s))
    result.rows_new = int(len(new_values))

    # Bounded warmup boundary: the frozen grid's first (lookback_depth - 1)
    # trading dates (measured: exactly where the under-warmed cells live; the
    # w-th date is already fully warmed). The boundary is the GRID's left
    # edge, not per-symbol: the old runners anchored every symbol's load at
    # ``data.start``, so only the evaluation window's left edge is
    # under-warmed — a late-starting symbol's first dates see the same bars
    # in both geometries and cannot differ there.
    grid_dates = sorted(frozen["date"].unique())
    n_warm = max(int(lookback_depth) - 1, 0)
    warmup_dates = set(grid_dates[:n_warm])

    def _in_warmup(date: pd.Timestamp, symbol: str) -> bool:
        if is_pooled:
            return early_lo <= date <= early_hi
        return date in warmup_dates

    def _warmup_cell(date, symbol, frozen_v, new_v, direction: str) -> None:
        result.warmup_by_direction[direction] = (
            result.warmup_by_direction.get(direction, 0) + 1
        )
        if is_pooled:
            month = str(date.to_period("M"))
            result.warmup_by_month[month] = result.warmup_by_month.get(month, 0) + 1
        result.diffs.append(
            PanelCellDiff(
                str(date.date()), str(symbol),
                None if frozen_v is None or pd.isna(frozen_v) else float(frozen_v),
                None if new_v is None or pd.isna(new_v) else float(new_v),
                "warmup_left_extension",
            )
        )

    new_idx = new_values.index
    frozen_idx = frozen_s.index
    frozen_symbols = set(frozen["symbol"].unique())

    # Extra served rows: allowed as the all-NaN D4c footprint (drift #3), or
    # as new-only finite rows INSIDE the warmup boundary for a symbol the
    # frozen panel knows (a grid gap the old runner did not emit). A finite
    # value for a symbol the frozen panel does not have at all is never
    # registered.
    extra = new_idx.difference(frozen_idx)
    for key in extra:
        value = new_values.loc[key]
        if pd.isna(value):
            result.nan_footprint_rows += 1
        elif key[1] in frozen_symbols and _in_warmup(key[0], key[1]):
            _warmup_cell(key[0], key[1], None, value, "new_only_finite")
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
            elif _in_warmup(date, symbol):
                _warmup_cell(date, symbol, frozen_v, new_v, "finite_to_finite")
            elif rel <= float_tail_tol:
                result.diffs.append(
                    PanelCellDiff(str(date.date()), str(symbol), float(frozen_v),
                                  float(new_v), "float_reordering_tail")
                )
            elif abs(frozen_v - new_v) == 1.0 and rel <= flip_rel_tol:
                result.diffs.append(
                    PanelCellDiff(str(date.date()), str(symbol), float(frozen_v),
                                  float(new_v), "threshold_flip_tail")
                )
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
        # frozen NaN -> new finite: the warmup class or unclassified.
        if _in_warmup(date, symbol):
            _warmup_cell(date, symbol, frozen_v, new_v, "nan_to_finite")
        else:
            result.diffs.append(
                PanelCellDiff(str(date.date()), str(symbol), None, float(new_v),
                              "unclassified_nan_to_finite")
            )

    counts = [result.warmup_by_month[m] for m in sorted(result.warmup_by_month)]
    result.warmup_monotonic = all(b <= a for a, b in zip(counts, counts[1:]))
    result.ok = (
        not any(d.classification.startswith("unclassified") or
                d.classification.startswith("unregistered") for d in result.diffs)
        and result.warmup_monotonic
        and len(result.by_class("float_reordering_tail")) <= float_tail_max
        and len(result.by_class("threshold_flip_tail")) <= flip_max
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
    classification: str  # ok | warmup_left_extension | failed


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
    warmup_dates: frozenset | None = None,
) -> AnchorRowResult:
    """One anchor row. The hand side was computed with the OLD loading
    geometry (left edge anchored at 2021-07-01), the service with the new
    (saturating) one — so a warmup-region row legitimately differs, for
    bounded factors exactly as in the panels leg (``warmup_dates`` = the
    frozen grid's first ``lookback_depth - 1`` trading dates).

    jump's NON-warmup rows MUST reconcile: they are the signal that the
    service carries the corrected, truncated definition (the five frozen-
    engine mismatches go green here); a miss outside the warmup boundary
    means the truncation is not carried and fails the mode.
    """
    if math.isnan(hand) and math.isnan(service):
        rel = 0.0
    elif math.isnan(hand) or math.isnan(service):
        rel = math.inf
    else:
        denom = max(abs(hand), abs(service))
        rel = abs(hand - service) / denom if denom > 0 else 0.0
    if rel <= tol:
        classification = "ok"
    elif warmup_dates is not None and date in warmup_dates:
        classification = "warmup_left_extension"
    elif factor_id == "jump_amount_corr_20":
        classification = "failed"
    elif is_pooled and EARLY_REGION_LO <= date <= EARLY_REGION_HI:
        classification = "warmup_left_extension"
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
        "panels %s: rows frozen=%d new=%d equal=%d tol=%d warmup=%d(%s) "
        "float_tail=%d threshold_flip=%d footprint=%d unclassified=%d "
        "max_rel=%.3e live_calls=%d ok=%s",
        factor_id, result.rows_frozen, result.rows_new, result.equal,
        result.within_tolerance, len(result.by_class("warmup_left_extension")),
        result.warmup_by_direction,
        len(result.by_class("float_reordering_tail")),
        len(result.by_class("threshold_flip_tail")),
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
    registered_sections = REGISTERED_EXTRA_SECTIONS.get(factor_id, ())
    section_md_prefixes = _registered_section_md_prefixes(registered_sections)
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
                registered_sections=registered_sections,
            )
        )
        new_md = (report_dir / f"{stem}_exec_{book}{'_bookclose' if 'bookclose' in label else ''}.md").read_text()
        frozen_md = baseline.read_text(f"eval_{report_name}_exec_{book}.md")
        results.append(
            diff_report_md(
                frozen_md, new_md, name=f"{stem}_exec_{book}.md[{label}]",
                strict=strict, correction_expected=correction_expected,
                registered_section_lines=section_md_prefixes,
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
        is_minute = True
    except KeyError:
        pooled = False  # daily factors are not in the minute partition sets
        is_minute = False
    warmup_dates: frozenset | None = None
    if is_minute and not pooled:
        # Bounded factors: the same warmup boundary as the panels leg — the
        # frozen grid's first (lookback_depth - 1) trading dates.
        panel_path = frozen_panel_path(factor_id, repo_root)
        if panel_path.exists():
            grid = pd.Index(
                pd.unique(pd.to_datetime(pd.read_parquet(panel_path)["date"]))
            ).sort_values()
            n_warm = max(int(factor.spec.lookback_depth) - 1, 0)
            warmup_dates = frozenset(grid[:n_warm])
    cfg = load_config(config_path)
    logger = _make_logger(
        Path(cfg.output.log_dir) / f"factor_eval_reconcile_anchors_{factor_id}.log",
        name="qt.factor_eval_reconcile",
    )
    anchor_symbols = sorted({row["symbol"] for row in rows})
    value_factors = ()
    if factor_id in ("value_ep", "value_bp"):
        from qt.factor_eval_runner import _build_book_factors

        value_factors = tuple(_build_book_factors())
    # A CROSS-SECTIONAL factor's served value is a function of the REQUESTED
    # universe (D4c: the combine runs at read-assembly over exactly what was
    # asked for) — and the hand anchors were computed over the FULL evaluation
    # universe. Requesting just the anchor symbols both empties the panel
    # (5 names < the combine's cross-section floor -> all-NaN, measured on
    # valley_price_quantile) and asks a different question than the hand side
    # answered. So such a factor is served over the full config universe and
    # the anchor cells are looked up inside it; per-symbol factors keep the
    # cheap anchor-only request (their values are universe-independent).
    from factors.materialize import stores_intermediate

    request_symbols = None if stores_intermediate(factor) else anchor_symbols
    store, sources, _panel, symbols, _cache = _build_bundle(
        cfg, logger, symbols=request_symbols, value_factors=value_factors
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
                is_pooled=pooled, tol=TOL, warmup_dates=warmup_dates,
            )
        )
    result.ok = all(r.classification != "failed" for r in result.rows)
    logger.info(
        "anchors %s: %d rows, ok=%d warmup=%d failed=%d live_calls=%d -> %s",
        factor_id, len(result.rows), len(result.by_class("ok")),
        len(result.by_class("warmup_left_extension")),
        len(result.by_class("failed")), live_calls, result.ok,
    )
    return result


__all__ = [
    "AGGREGATE_CHANGE_PATH_PREFIXES",
    "ALLOWED_ADDED_JSON_PATHS",
    "ALLOWED_ADDED_MD_PREFIXES",
    "AnchorsDiff",
    "EARLY_REGION_HI",
    "EARLY_REGION_LO",
    "FLOAT_TAIL_MAX_CELLS",
    "FLOAT_TAIL_REL_TOL",
    "LeafDiff",
    "METRIC_REL_TOL",
    "PANEL_REL_TOL",
    "PanelDiff",
    "REGISTERED_EXTRA_SECTIONS",
    "ReconciliationError",
    "ReportDiff",
    "THRESHOLD_FLIP_MAX_CELLS",
    "THRESHOLD_FLIP_REL_TOL",
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
