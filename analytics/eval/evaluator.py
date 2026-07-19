"""``FactorEvaluator``: the standard interface every factor evaluation walks.

The mandatory report sections are ABSTRACT METHODS, and the flow is a template
method — so a custom evaluator may ADD sections but can never drop a mandatory
one (design ``tmp/design/factor_eval_contract_v0.1.md`` §5). Together with
:meth:`FactorEvalReport.validate_all_mandatory_present` that is the double lock:
the ABC stops you at class definition, the report validation stops you at
assembly.

SCOPE (PR-A = the contract only): this module defines the interface, the fixed
section order and the eval-IR seam. ``StandardFactorEvaluator`` and the
vectorized ``build_ir`` are **PR-B's job** and are deliberately absent here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from analytics.eval.config import EvalConfig
from analytics.eval.report import FactorEvalReport
from analytics.eval.sections import MANDATORY_SECTIONS, Section, Skipped
from analytics.eval.verdict import VerdictThresholds
from factors.spec import FactorSpec


class EvalIR(Protocol):
    """The four vectorized intermediates every mandatory metric derives from.

    Design §8: computing these ONCE and reducing them is what turns the report
    from a per-period python loop into two ``groupby`` passes. **PR-B implements
    the builder**; PR-A only fixes the shape so the reduction code and the
    speed-up have a stable target.

    Attributes
    ----------
    factor : (1) the processed factor panel ``F``, MultiIndex(date, symbol).
    forward_returns : (2) ``R_h`` at the spec's horizon, MultiIndex(date, symbol),
        computed at the analytics boundary — the factor layer never sees it.
    ic : (3) the per-date IC series, indexed by date.
    quantile_returns : (4) the per-(date, quantile) mean-return matrix.
    """

    factor: pd.Series
    forward_returns: pd.Series
    ic: pd.Series
    quantile_returns: pd.DataFrame


class FactorEvaluator(ABC):
    """Standard factor evaluation: 8 mandatory sections + a fixed flow.

    Each section returns a :class:`Section` or an explicit
    :class:`Skipped` (with a reason) — never None, never a silent omission.
    """

    #: the fixed section order the template method calls (design §5). Aliased to
    #: the report's mandatory set on purpose: ONE source of truth, so the caller
    #: order and the presence check can never drift apart.
    SECTION_ORDER: tuple[str, ...] = MANDATORY_SECTIONS

    @abstractmethod
    def build_ir(
        self,
        factor_panel: pd.Series | pd.DataFrame,
        spec: FactorSpec,
        cfg: EvalConfig,
        ctx: object | None = None,
    ) -> EvalIR:
        """Build the vectorized eval-IR once (design §8). **PR-B implements this.**

        ``ctx`` carries whatever a section needs beyond the factor panel (price
        panel, universe, minute bars, ...); PR-B defines its shape.
        """

    @abstractmethod
    def predictive_power(self, ir: EvalIR) -> Section | Skipped:
        """Rank IC / ICIR / IC win rate / Newey-West t / IC decay."""

    @abstractmethod
    def return_risk(self, ir: EvalIR) -> Section | Skipped:
        """Quantile NAV / long-short spread / monotonicity / Sharpe / maxDD / vol."""

    @abstractmethod
    def stability_cost(self, ir: EvalIR) -> Section | Skipped:
        """Turnover / autocorrelation (half-life) / cost-sensitivity gradient."""

    @abstractmethod
    def purity(self, ir: EvalIR) -> Section | Skipped:
        """Correlation with known factors / orthogonalized IC / post-neutralization / VIF."""

    @abstractmethod
    def oos_generalization(self, ir: EvalIR) -> Section | Skipped:
        """Train/test sign consistency / independent-universe verdict (the project's core lesson)."""

    @abstractmethod
    def execution_capacity(self, ir: EvalIR) -> Section | Skipped:
        """Fill feasibility (price-limit gating, I5b) / capacity ratio (I5f)."""

    @abstractmethod
    def data_coverage(self, ir: EvalIR) -> Section | Skipped:
        """Coverage / dropped symbols / survivorship / NaN — and the sample size."""

    @abstractmethod
    def caveats(self, ir: EvalIR) -> Section | Skipped:
        """Post-hoc / exploratory / sample size / multiple testing."""

    def evaluate(
        self,
        factor_panel: pd.Series | pd.DataFrame,
        spec: FactorSpec,
        cfg: EvalConfig,
        ctx: object | None = None,
        thresholds: VerdictThresholds | None = None,
    ) -> FactorEvalReport:
        """Build the IR once, run the 8 sections IN ORDER, assemble, validate, verdict.

        The three contract steps stay visible on purpose: assemble ->
        validate_all_mandatory_present (raises on a silently missing section) ->
        with_verdict (a report may not be published without one).
        """
        ir = self.build_ir(factor_panel, spec, cfg, ctx)
        sections = [getattr(self, name)(ir) for name in self.SECTION_ORDER]
        report = FactorEvalReport.assemble(spec, cfg, sections, thresholds=thresholds)
        report.validate_all_mandatory_present()
        return report.with_verdict(thresholds)


__all__ = ["EvalIR", "FactorEvaluator"]
