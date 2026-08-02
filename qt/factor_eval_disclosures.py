"""The SINGLE home of the heterogeneous factor-eval coverage disclosures (D5 C4).

Four of the eleven legacy eval runners each carried a scarcity / neutralization
disclosure of a DIFFERENT shape (catalogue
``docs/factors/d5_runner_difference_catalogue.md`` §三): no shared base class,
four independent frozen dataclasses, each with its own summarizer and
``render()``. They are deliberately NOT unified into one object — these are the
measured facts of each factor's OWN gates, with different units and semantics,
and forcing one shape would lose information. This module is where they now
live, MOVED VERBATIM from:

* ``RidgeCoverage`` + ``summarize_ridge_coverage`` (from
  ``qt/eval_valley_ridge_vwap_ratio.py``),
* ``RidgeReturnCoverage`` + ``summarize_ridge_return_coverage`` (from
  ``qt/eval_ridge_minute_return.py``),
* ``PeakCoverage`` + ``summarize_peak_coverage`` (from
  ``qt/eval_peak_ridge_amount_ratio.py``),
* ``NeutralizationCoverage`` + ``summarize_neutralization`` (from
  ``qt/eval_valley_price_quantile.py``).

The four legacy runners re-export from here, so every historical import path
keeps working (their test files are untouched).

The TWO intentional additions on top of the verbatim move, both catalogue
items, and both small:

1. ``NeutralizationCoverage.render()`` — the one disclosure the CLI rendered
   as an INLINE f-string (catalogue §三 "一处归一": PR-L at ``qt/cli.py``
   while its three siblings already went through ``.render()``). The format is
   exactly the line the CLI used to inline.
2. :func:`to_section` — packs a coverage dataclass into an add-Section, the
   contract's §3.6 extension point ("may ADD sections but never drop a
   mandatory one"). The renderer natively supports extras
   (``analytics/eval/render.py::canonical_sections``: mandatory first, extras
   sorted by name after), and the verdict reads only the mandatory payloads,
   so appending an extra section is VERDICT-LAZY by construction (pinned by
   test).

GATE-DEFAULT SINGLE SOURCE: the summarizers' floor defaults are the same
``factors.compute.minute`` module constants the minute-binding's compute calls
apply
(``factors/compute/minute/binding.py`` calls every compute function with all
gate parameters at their module defaults). The disclosure therefore reports
the floors the run ACTUALLY applied by construction — and a test pins the
defaults against those constants so a hand-edit on one side cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analytics.eval.sections import Section
from data.clean.schema import DATE_LEVEL
from factors.compute.minute.peak_ridge_amount_ratio import (
    PEAK_RIDGE_MIN_PEAK_BARS,
    PEAK_RIDGE_MIN_RIDGE_BARS,
    PeakRidgeAmountRatioFactor,
)
from factors.compute.minute.primitives import VOLUME_PRV_MIN_CLASSIFIABLE
from factors.compute.minute.ridge_minute_return import (
    RIDGE_RETURN_MIN_RIDGE_BARS,
    RidgeMinuteReturnFactor,
)
from factors.compute.minute.valley_price_quantile import ValleyPriceQuantileFactor
from factors.compute.minute.valley_ridge_vwap_ratio import (
    VALLEY_RIDGE_MIN_RIDGE_BARS,
    VALLEY_RIDGE_MIN_VALLEY_BARS,
    ValleyRidgeVwapRatioFactor,
)

# Percentiles reported for the realized ridge/peak-bar distributions.
_RIDGE_PCTL = (0, 10, 25, 50, 75, 90, 100)
_PEAK_PCTL = (0, 10, 25, 50, 75, 90, 100)

# The counterfactual floor the ridge-return coverage disclosure also reports, so
# PR-K's ridge coverage is directly comparable to PR-J's (which used 20 for its
# VALLEY leg).
_COMPARISON_FLOOR = 20

# The counterfactual peak floor the disclosure quantifies: how many days would
# still be valid if the PEAK leg were held to the RIDGE leg's floor instead of
# its own. DERIVED from the ridge gate, not hardcoded.
_COUNTERFACTUAL_PEAK_FLOOR = PEAK_RIDGE_MIN_RIDGE_BARS

# --------------------------------------------------------------------------- #
# Ridge-scarcity coverage (measured, never assumed) — PR-J valley/ridge VWAP
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RidgeCoverage:
    """Realized ridge-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the
    days the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible
    bars, including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The
    headline ``validity_rate`` is taken over ``classifiable_days``, because a day with
    no baseline fails for a PR-F warm-up reason rather than a ridge-scarcity one and
    would otherwise make the ridge gate look worse than it is — both denominators are
    reported so the reader can check that framing. The gate-failure counts are NOT
    mutually exclusive (a thin day can fail several gates at once) and are reported for
    shape, not as a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    ridge_percentiles: tuple[tuple[int, float], ...]
    ridge_mean: float
    valley_median: float
    days_below_ridge_gate: int
    days_below_valley_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive if the ridge leg were held to the
    # VALLEY floor. Quantifies exactly what the lowered threshold buys.
    valid_days_at_valley_floor: int
    # The gates this run actually applied, so the disclosure can never describe the
    # module defaults while the run used something else.
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.ridge_percentiles)
        return (
            f"ridge scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"ridge_bars[{pctl} mean={self.ridge_mean:.1f}] "
            f"valley_bars_median={self.valley_median:.0f} "
            f"below_ridge_gate({self.min_ridge_bars})="
            f"{self.days_below_ridge_gate} "
            f"below_valley_gate({self.min_valley_bars})="
            f"{self.days_below_valley_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_ridge_floor_were_{self.min_valley_bars}="
            f"{self.valid_days_at_valley_floor}"
        )



def summarize_ridge_coverage(
    frames: list[pd.DataFrame],
    *,
    min_ridge_bars: int = VALLEY_RIDGE_MIN_RIDGE_BARS,
    min_valley_bars: int = VALLEY_RIDGE_MIN_VALLEY_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
) -> RidgeCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The three floors must be the ones the RUN applied, not the module defaults —
    otherwise the disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_ridge_bars=min_ridge_bars,
        min_valley_bars=min_valley_bars,
        min_classifiable=min_classifiable,
    )
    empty = tuple((p, float("nan")) for p in _RIDGE_PCTL)
    if not frames:
        return RidgeCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            ridge_percentiles=empty,
            ridge_mean=float("nan"),
            valley_median=float("nan"),
            days_below_ridge_gate=0,
            days_below_valley_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_valley_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the
    # percentiles towards zero for a reason that has nothing to do with ridge scarcity.
    scored = classifiable >= min_classifiable
    ridge = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    valley = diag.loc[scored, "valley_bars"].to_numpy(dtype=float)
    # The counterfactual holds the ridge leg to the valley floor, leaving every other
    # gate exactly as it was.
    at_valley_floor = valid & (
        diag["ridge_bars"].to_numpy(dtype=float) >= min_valley_bars
    )
    return RidgeCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        ridge_percentiles=(
            tuple((p, float(np.percentile(ridge, p))) for p in _RIDGE_PCTL)
            if ridge.size
            else empty
        ),
        ridge_mean=float(ridge.mean()) if ridge.size else float("nan"),
        valley_median=float(np.median(valley)) if valley.size else float("nan"),
        days_below_ridge_gate=int((ridge < min_ridge_bars).sum()),
        days_below_valley_gate=int((valley < min_valley_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_valley_floor=int(at_valley_floor.sum()),
        **gates,
    )



# --------------------------------------------------------------------------- #
# Ridge-scarcity coverage (measured, never assumed) — PR-K ridge minute return
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RidgeReturnCoverage:
    """Realized ridge-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the days
    the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible bars,
    including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The headline
    ``validity_rate`` is taken over ``classifiable_days``, because a day with no baseline
    fails for a PR-F warm-up reason rather than a ridge-scarcity one and would otherwise
    make the ridge gate look worse than it is — both denominators are reported so the
    reader can check that framing.

    TWO ridge counts are tracked, because this factor gates on the narrower one: total
    ``ridge_bars`` and the ``ridge_return_bars`` subset that carries a valid minute return
    (the day's first visible bar is excluded by the within-day lag even when it is a
    ridge). Reporting both makes the return-guard attrition visible instead of implicit.
    The gate-failure counts are NOT mutually exclusive and are reported for shape, not as
    a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    ridge_return_percentiles: tuple[tuple[int, float], ...]
    ridge_return_mean: float
    ridge_bars_mean: float
    ridge_bars_median: float
    days_below_ridge_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive at the higher floor PR-J used for its
    # VALLEY leg. Quantifies exactly what the scarcity-driven threshold buys.
    valid_days_at_comparison_floor: int
    # The gates this run actually applied, so the disclosure can never describe the module
    # defaults while the run used something else.
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE
    comparison_floor: int = _COMPARISON_FLOOR

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    @property
    def return_guard_attrition(self) -> float:
        """Share of ridge bars LOST to the return guard (mean over classifiable days)."""
        if not np.isfinite(self.ridge_bars_mean) or self.ridge_bars_mean <= 0.0:
            return float("nan")
        return 1.0 - self.ridge_return_mean / self.ridge_bars_mean

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.ridge_return_percentiles)
        return (
            f"ridge scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"ridge_return_bars[{pctl} mean={self.ridge_return_mean:.1f}] "
            f"ridge_bars_mean={self.ridge_bars_mean:.1f} "
            f"ridge_bars_median={self.ridge_bars_median:.0f} "
            f"return_guard_attrition={self.return_guard_attrition:.1%} "
            f"below_ridge_gate({self.min_ridge_bars})={self.days_below_ridge_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_floor_were_{self.comparison_floor}="
            f"{self.valid_days_at_comparison_floor}"
        )



def summarize_ridge_return_coverage(
    frames: list[pd.DataFrame],
    *,
    min_ridge_bars: int = RIDGE_RETURN_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    comparison_floor: int = _COMPARISON_FLOOR,
) -> RidgeReturnCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The floors must be the ones the RUN applied, not the module defaults — otherwise the
    disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        comparison_floor=comparison_floor,
    )
    empty = tuple((p, float("nan")) for p in _RIDGE_PCTL)
    if not frames:
        return RidgeReturnCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            ridge_return_percentiles=empty,
            ridge_return_mean=float("nan"),
            ridge_bars_mean=float("nan"),
            ridge_bars_median=float("nan"),
            days_below_ridge_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_comparison_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the
    # percentiles towards zero for a reason that has nothing to do with ridge scarcity.
    scored = classifiable >= min_classifiable
    ridge_ret = diag.loc[scored, "ridge_return_bars"].to_numpy(dtype=float)
    ridge_all = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    # The counterfactual raises the ridge floor, leaving every other gate exactly as it was.
    at_comparison = valid & (
        diag["ridge_return_bars"].to_numpy(dtype=float) >= comparison_floor
    )
    return RidgeReturnCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        ridge_return_percentiles=(
            tuple((p, float(np.percentile(ridge_ret, p))) for p in _RIDGE_PCTL)
            if ridge_ret.size
            else empty
        ),
        ridge_return_mean=float(ridge_ret.mean()) if ridge_ret.size else float("nan"),
        ridge_bars_mean=float(ridge_all.mean()) if ridge_all.size else float("nan"),
        ridge_bars_median=float(np.median(ridge_all)) if ridge_all.size else float("nan"),
        days_below_ridge_gate=int((ridge_ret < min_ridge_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_comparison_floor=int(at_comparison.sum()),
        **gates,
    )



# --------------------------------------------------------------------------- #
# Peak-scarcity coverage (measured, never assumed) — PR-M peak/ridge amount
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PeakCoverage:
    """Realized peak-bar distribution + day-validity rate over the whole universe.

    Built from the per-day diagnostics the factor emits, so the numbers describe the days
    the factor actually saw. ``symbol_days`` counts EVERY symbol-day with visible bars,
    including the leading warm-up days that have no same-slot baseline yet;
    ``classifiable_days`` counts those that clear PR-F's classifiable floor. The headline
    ``validity_rate`` is taken over ``classifiable_days``, because a day with no baseline
    fails for a PR-F warm-up reason rather than a peak-scarcity one and would otherwise
    make the peak gate look worse than it is — both denominators are reported so the reader
    can check that framing. The gate-failure counts are NOT mutually exclusive (a thin day
    can fail several gates at once) and are reported for shape, not as a partition.
    """

    symbol_days: int
    classifiable_days: int
    valid_days: int
    peak_percentiles: tuple[tuple[int, float], ...]
    peak_mean: float
    ridge_median: float
    days_below_peak_gate: int
    days_below_ridge_gate: int
    days_below_classifiable_gate: int
    # Counterfactual: how many days would survive if the PEAK leg were held to the RIDGE
    # floor. Quantifies exactly what the lowered threshold buys.
    valid_days_at_ridge_floor: int
    # The gates this run actually applied, so the disclosure can never describe the module
    # defaults while the run used something else.
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE
    counterfactual_peak_floor: int = _COUNTERFACTUAL_PEAK_FLOOR

    @property
    def validity_rate(self) -> float:
        """Valid days as a share of CLASSIFIABLE days (see the class docstring)."""
        if not self.classifiable_days:
            return float("nan")
        return self.valid_days / self.classifiable_days

    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI."""
        pctl = " ".join(f"p{p}={v:.0f}" for p, v in self.peak_percentiles)
        return (
            f"peak scarcity: symbol_days={self.symbol_days} "
            f"classifiable_days={self.classifiable_days} "
            f"valid_days={self.valid_days} ({self.validity_rate:.1%} of classifiable) "
            f"peak_bars[{pctl} mean={self.peak_mean:.1f}] "
            f"ridge_bars_median={self.ridge_median:.0f} "
            f"below_peak_gate({self.min_peak_bars})={self.days_below_peak_gate} "
            f"below_ridge_gate({self.min_ridge_bars})={self.days_below_ridge_gate} "
            f"below_classifiable_gate({self.min_classifiable})="
            f"{self.days_below_classifiable_gate} "
            f"valid_if_peak_floor_were_{self.counterfactual_peak_floor}="
            f"{self.valid_days_at_ridge_floor}"
        )



def summarize_peak_coverage(
    frames: list[pd.DataFrame],
    *,
    min_peak_bars: int = PEAK_RIDGE_MIN_PEAK_BARS,
    min_ridge_bars: int = PEAK_RIDGE_MIN_RIDGE_BARS,
    min_classifiable: int = VOLUME_PRV_MIN_CLASSIFIABLE,
    counterfactual_peak_floor: int = _COUNTERFACTUAL_PEAK_FLOOR,
) -> PeakCoverage:
    """Reduce the per-symbol day-level diagnostics to the scarcity disclosure.

    The three floors must be the ones the RUN applied, not the module defaults — otherwise
    the disclosure would describe gates that were never enforced.
    """
    gates = dict(
        min_peak_bars=min_peak_bars,
        min_ridge_bars=min_ridge_bars,
        min_classifiable=min_classifiable,
        counterfactual_peak_floor=counterfactual_peak_floor,
    )
    empty = tuple((p, float("nan")) for p in _PEAK_PCTL)
    if not frames:
        return PeakCoverage(
            symbol_days=0,
            classifiable_days=0,
            valid_days=0,
            peak_percentiles=empty,
            peak_mean=float("nan"),
            ridge_median=float("nan"),
            days_below_peak_gate=0,
            days_below_ridge_gate=0,
            days_below_classifiable_gate=0,
            valid_days_at_ridge_floor=0,
            **gates,
        )
    diag = pd.concat(frames, ignore_index=True)
    classifiable = diag["classifiable_bars"].to_numpy(dtype=float)
    valid = diag["valid"].to_numpy(dtype=bool)
    # The bar-count distributions describe the days that had a fair chance: a warm-up day
    # with no same-slot baseline has zero of everything and would only drag the percentiles
    # towards zero for a reason that has nothing to do with peak scarcity.
    scored = classifiable >= min_classifiable
    peak = diag.loc[scored, "peak_bars"].to_numpy(dtype=float)
    ridge = diag.loc[scored, "ridge_bars"].to_numpy(dtype=float)
    # The counterfactual raises the PEAK floor, leaving every other gate exactly as it was.
    at_ridge_floor = valid & (
        diag["peak_bars"].to_numpy(dtype=float) >= counterfactual_peak_floor
    )
    return PeakCoverage(
        symbol_days=int(len(diag)),
        classifiable_days=int(scored.sum()),
        valid_days=int(valid.sum()),
        peak_percentiles=(
            tuple((p, float(np.percentile(peak, p))) for p in _PEAK_PCTL)
            if peak.size
            else empty
        ),
        peak_mean=float(peak.mean()) if peak.size else float("nan"),
        ridge_median=float(np.median(ridge)) if ridge.size else float("nan"),
        days_below_peak_gate=int((peak < min_peak_bars).sum()),
        days_below_ridge_gate=int((ridge < min_ridge_bars).sum()),
        days_below_classifiable_gate=int((~scored).sum()),
        valid_days_at_ridge_floor=int(at_ridge_floor.sum()),
        **gates,
    )



# --------------------------------------------------------------------------- #
# Neutralization coverage (the reversal-neutralized factor's diagnostic) — PR-L
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NeutralizationCoverage:
    """What the reversal neutralization actually did, measured rather than assumed.

    A neutralization can fail quietly in two ways: the reversal can be unavailable for
    most of the panel (so most residuals are NaN), or the cross-section can be too thin
    to regress on many dates. Both are counted here and logged, so a coverage regression
    is a number in the run record instead of an unexplained drop in sample size.
    """

    raw_rows: int  # finite RAW qbar values
    rev_rows: int  # finite rev20 values on those rows
    residual_rows: int  # finite residuals (the shipped factor)
    dates_total: int
    dates_residualized: int  # dates that cleared min_cross_section AND were non-degenerate
    cross_section_min: int
    cross_section_median: float
    cross_section_max: int
    raw_rev_spearman_mean: float  # mean per-date exposure of the RAW factor to rev20
    def render(self) -> str:
        """One-line, secret-free summary for the run log and the CLI.

        THE one intentional addition on top of the verbatim move: this disclosure
        was the only one of the four the CLI rendered as an INLINE f-string
        (catalogue section 3 "one-site normalization"), while its three siblings
        already went through ``.render()``. The format below is exactly the line
        the CLI used to inline, so the rendered output cannot drift from what
        runs already print.
        """
        return (
            f"neutralization (T-1 rev20): raw_rows={self.raw_rows} "
            f"rev_paired={self.rev_rows} residual_rows={self.residual_rows} "
            f"dates={self.dates_residualized}/{self.dates_total} "
            f"cross_section min/med/max={self.cross_section_min}/"
            f"{self.cross_section_median:.1f}/{self.cross_section_max} "
            f"mean_spearman(raw,rev20)={self.raw_rev_spearman_mean:+.4f}"
        )



def summarize_neutralization(
    raw: pd.Series,
    rev: pd.Series,
    residual: pd.Series,
    *,
    min_cross_section: int,
) -> NeutralizationCoverage:
    """Reduce the raw / reversal / residual panels to the coverage diagnostic."""
    raw_finite = raw.dropna()
    rev_on_raw = rev.reindex(raw.index)
    paired = pd.DataFrame({"f": raw, "r": rev_on_raw}).dropna()

    sizes: list[int] = []
    exposures: list[float] = []
    for _, g in paired.groupby(level=DATE_LEVEL, sort=True):
        sizes.append(len(g))
        if len(g) >= min_cross_section:
            f = g["f"].to_numpy(dtype=float)
            r = g["r"].to_numpy(dtype=float)
            fr = pd.Series(f).rank().to_numpy()
            rr = pd.Series(r).rank().to_numpy()
            if fr.std() > 0.0 and rr.std() > 0.0:
                exposures.append(float(np.corrcoef(fr, rr)[0, 1]))

    resid_finite = residual.dropna()
    dates_resid = int(
        resid_finite.index.get_level_values(DATE_LEVEL).unique().size
    )
    return NeutralizationCoverage(
        raw_rows=int(len(raw_finite)),
        rev_rows=int(len(paired)),
        residual_rows=int(len(resid_finite)),
        dates_total=int(raw.index.get_level_values(DATE_LEVEL).unique().size),
        dates_residualized=dates_resid,
        cross_section_min=int(min(sizes)) if sizes else 0,
        cross_section_median=float(np.median(sizes)) if sizes else float("nan"),
        cross_section_max=int(max(sizes)) if sizes else 0,
        raw_rev_spearman_mean=(
            float(np.mean(exposures)) if exposures else float("nan")
        ),
    )



# --------------------------------------------------------------------------- #
# The add-Section bridge (contract §3.6: may ADD, never drop a mandatory one)
# --------------------------------------------------------------------------- #
#: Payload keys that are computed PROPERTIES rather than dataclass fields, so
#: ``asdict`` alone does not produce them. Author-once: the reconcile harness
#: derives an add-Section's Markdown line prefixes from this same tuple, and a
#: property listed in only one of the two places would silently unregister a
#: rendered line (measured: three such lines failed the reports leg).
DERIVED_PAYLOAD_PROPERTIES: tuple[str, ...] = (
    "validity_rate",
    "return_guard_attrition",
)


def to_section(name: str, coverage) -> Section:
    """Pack a coverage disclosure dataclass into an add-Section.

    The payload is the dataclass's fields (asdict) plus its derived properties
    (``validity_rate`` / ``return_guard_attrition`` where defined); the note is
    the disclosure's one-line ``render()``, so the artifact and the run log can
    never state different numbers. The verdict reads ONLY the mandatory section
    payloads (``analytics/eval/report.py::extract_verdict_inputs``), so an extra
    section added this way cannot move a verdict — pinned by test.
    """
    payload: dict[str, object] = dict(asdict(coverage))
    for prop in DERIVED_PAYLOAD_PROPERTIES:
        if hasattr(coverage, prop):
            payload[prop] = getattr(coverage, prop)
    return Section(name=name, payload=payload, note=coverage.render())


@dataclass(frozen=True)
class DisclosureBinding:
    """Which disclosure a factor publishes: the section name + its summarizer."""

    section_name: str
    summarize: Callable[[list[pd.DataFrame]], object]


#: factor class -> its day-level gate-attrition disclosure. Only the factors
#: that HAVE such a disclosure appear (catalogue §三 mechanism A: the
#: diagnostics sink); every other factor publishes NO per-day disclosure, which
#: is stated (``None``) rather than inferred from an empty frame.
_DISCLOSURE_BY_CLASS: dict[type, DisclosureBinding] = {
    ValleyRidgeVwapRatioFactor: DisclosureBinding(
        "ridge_scarcity_coverage", summarize_ridge_coverage
    ),
    RidgeMinuteReturnFactor: DisclosureBinding(
        "ridge_scarcity_coverage", summarize_ridge_return_coverage
    ),
    PeakRidgeAmountRatioFactor: DisclosureBinding(
        "peak_scarcity_coverage", summarize_peak_coverage
    ),
}


def disclosure_binding_for(factor) -> DisclosureBinding | None:
    """The disclosure binding for ``factor``, or None when it publishes none.

    Keyed by the factor CLASS (the same keying
    ``factors/compute/minute/binding.py`` uses for its diagnostics bindings), so
    the two tables agree by construction; a factor with a diagnostics binding
    but no summarizer here is a readable error at the call site, never a
    silently reduced mixture.
    """
    return _DISCLOSURE_BY_CLASS.get(type(factor))


#: The add-Section name of the ONE mechanism-B disclosure (catalogue §三).
NEUTRALIZATION_SECTION_NAME = "neutralization_coverage"

#: factor class -> publishes the NeutralizationCoverage disclosure. Mechanism B
#: (catalogue §三): NO diagnostics sink — the disclosure is reduced from the
#: raw + reversal + residual panels AFTER the loop, so it cannot ride
#: :func:`disclosure_binding_for`'s sink table. Class-keyed like that table,
#: so the two mechanisms agree by construction.
_NEUTRALIZATION_DISCLOSURE_CLASSES: frozenset[type] = frozenset(
    {ValleyPriceQuantileFactor}
)


def publishes_neutralization_disclosure(factor) -> bool:
    """True iff ``factor`` publishes the mechanism-B NeutralizationCoverage."""
    return type(factor) in _NEUTRALIZATION_DISCLOSURE_CLASSES


__all__ = [
    "DERIVED_PAYLOAD_PROPERTIES",
    "DisclosureBinding",
    "NEUTRALIZATION_SECTION_NAME",
    "NeutralizationCoverage",
    "PeakCoverage",
    "RidgeCoverage",
    "RidgeReturnCoverage",
    "disclosure_binding_for",
    "publishes_neutralization_disclosure",
    "summarize_neutralization",
    "summarize_peak_coverage",
    "summarize_ridge_coverage",
    "summarize_ridge_return_coverage",
    "to_section",
]
