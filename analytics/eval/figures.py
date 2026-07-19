"""Sell-side-research-style visual dashboard for a factor evaluation report.

ONE PNG that renders a :class:`~analytics.eval.report.FactorEvalReport` (the frozen
section payloads + three-axis verdict) together with the two time series only the
:class:`~analytics.eval.ir.StandardEvalIR` carries (per-rebalance RankIC and the
quantile return matrix). Pure plotting: :func:`render_factor_dashboard` takes an
already-evaluated report + IR and writes the figure, so it is decoupled from the
run logic and unit-testable with a toy :class:`DashboardData` (mirroring
``qt.intraday_group_figures``).

Design language references the Kaiyuan (开源证券) layered-backtest charts: a white
canvas, thin dark spines, no heavy gridlines, the SIGNATURE thin per-quantile NAV
curves plus a THICK dark-maroon long-short curve on a secondary axis, a dense
YYYYMMDD date axis, and a top multi-column legend.

MANDATORY factor-definition panel (contract constraint): every dashboard renders a
"Factor definition" band sourced from the report's :class:`~factors.spec.FactorSpec`
— what the factor measures, its inputs, hypothesis sign, horizon / return basis,
family, and (when intraday) the minute execution block. A factor may not be shown
without stating how it is computed.

matplotlib is imported lazily here and this module is deliberately NOT re-exported
from ``analytics.eval.__init__`` so importing the eval package never pulls a
plotting backend.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a display.
import matplotlib.pyplot as plt  # noqa: E402  (must follow use("Agg"))
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from analytics.eval.report import FactorEvalReport  # noqa: E402
from analytics.eval.verdict import VerdictResult  # noqa: E402
from factors.spec import FactorSpec  # noqa: E402

# -- palette (Kaiyuan-report inspired) -------------------------------------
_INK = "#222222"
_GRID = "#E9E9E9"
_BAR_BLUE = "#4472C4"
_BAR_RED = "#C0392B"
_GOLD = "#E1A730"
_LS_MAROON = "#8B1A1A"  # the signature long-short curve
#: ordered quantile palette Q1(low) -> QN(high): deep-blue -> gray -> red.
_Q_COLORS = ("#2E5B9C", "#6FA8DC", "#9AA0A6", "#E69138", "#CC0000")
_DEP_COLORS = {"Adopt": "#2E8B57", "Watch": "#D98C00", "Reject": "#C0392B"}
_AXIS_COLORS = {
    "PASS": "#2E8B57", "FAIL": "#C0392B",
    "NOT_ASSESSED": "#8A8A8A", "INSUFFICIENT_DATA": "#D98C00",
}
_DEF_BG = "#F4F6F9"  # factor-definition band background


# --------------------------------------------------------------------------- #
# Normalized input (decoupled from report/IR internals -> unit-testable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashboardData:
    """Everything the dashboard renders, pulled out of a report + its IR.

    Building this explicitly (rather than reaching into a report/IR inside every
    panel) keeps :func:`_render` testable with toy data.
    """

    spec: FactorSpec
    verdict: VerdictResult
    payloads: dict[str, dict]  # section name -> payload dict (empty for Skipped)
    ic: pd.Series              # per-rebalance RankIC (may be empty)
    quantile_returns: pd.DataFrame  # date x 1..n quantile returns (may be empty)

    @classmethod
    def from_report(cls, report: FactorEvalReport, ir: object) -> "DashboardData":
        payloads = {
            s.name: dict(getattr(s, "payload", None) or {}) for s in report.sections
        }
        ic = getattr(ir, "ic", pd.Series(dtype=float))
        qr = getattr(ir, "quantile_returns", pd.DataFrame())
        return cls(
            spec=report.spec,
            verdict=report.require_verdict(),
            payloads=payloads,
            ic=ic if isinstance(ic, pd.Series) else pd.Series(dtype=float),
            quantile_returns=qr if isinstance(qr, pd.DataFrame) else pd.DataFrame(),
        )


# --------------------------------------------------------------------------- #
# styling helpers
# --------------------------------------------------------------------------- #
def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": _INK,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _style(ax, title: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRID, linewidth=0.7, zorder=0)
    ax.tick_params(colors=_INK, labelsize=8, length=3)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold",
                     color="#1a1a1a", pad=8)


def _date_axis(ax, index, n_ticks: int = 16) -> None:
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return
    step = max(1, len(idx) // n_ticks)
    pos = list(range(0, len(idx), step))
    ax.set_xticks([idx[p] for p in pos])
    ax.set_xticklabels([idx[p].strftime("%Y%m%d") for p in pos],
                       rotation=90, fontsize=6.8)


def _empty(ax, title: str, msg: str = "not available") -> None:
    _style(ax, title)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
            color="#999999", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _qcols(qr: pd.DataFrame) -> list:
    return sorted(qr.columns, key=lambda c: int(str(c)))


# --------------------------------------------------------------------------- #
# header + verdict chips + MANDATORY factor-definition band
# --------------------------------------------------------------------------- #
def _header_text(ax, data: DashboardData) -> None:
    ax.axis("off")
    cov = data.payloads.get("data_coverage", {})
    pp = data.payloads.get("predictive_power", {})
    cav = data.payloads.get("caveats", {})
    spec = data.spec
    win = cav.get("window", "")
    ax.text(0.0, 0.72, f"Factor Evaluation — {spec.factor_id}",
            fontsize=17, fontweight="bold", color="#111111", transform=ax.transAxes)
    n = cov.get("settled_rebalances")
    neff = cov.get("effective_samples")
    sub = f"{cav.get('universe', '')} · {win} · {cav.get('rebalance', '')} rebalance"
    if n is not None:
        sub += f" · N={int(n)} settled"
    if isinstance(neff, (int, float)):
        sub += f" (N_eff={neff:.0f})"
    if cav.get("is_exploratory"):
        sub += " · EXPLORATORY (cap: Watch)"
    ax.text(0.0, 0.16, sub, fontsize=10, color="#444444", transform=ax.transAxes)
    if pp:
        ax.text(1.0, 0.72,
                f"RankIC {pp.get('ic_mean', float('nan')):+.4f}    "
                f"ICIR {pp.get('ic_ir', float('nan')):+.3f}    "
                f"NW-t {pp.get('ic_nw_t', float('nan')):+.1f}",
                fontsize=11, color="#111111", ha="right", transform=ax.transAxes)
        lo, hi = pp.get("ic_ir_ci_low"), pp.get("ic_ir_ci_high")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            ax.text(1.0, 0.16,
                    f"ICIR 95% CI [{lo:+.3f}, {hi:+.3f}]    "
                    f"win rate {pp.get('ic_win_rate', float('nan')):.1%}",
                    fontsize=10, color="#444444", ha="right", transform=ax.transAxes)


def _chips_row(ax, data: DashboardData) -> None:
    ax.axis("off")
    ax.axhline(0.97, color="#DDDDDD", lw=1.0)
    v = data.verdict
    axes_v = v.axes()
    chips = [
        ("DEPLOYMENT", v.verdict, _DEP_COLORS.get(v.verdict, "#8A8A8A"), True),
        ("Predictive", axes_v["predictive"].verdict.replace("_", " "),
         _AXIS_COLORS.get(axes_v["predictive"].verdict, "#8A8A8A"), False),
        ("Incremental", axes_v["incremental"].verdict.replace("_", " "),
         _AXIS_COLORS.get(axes_v["incremental"].verdict, "#8A8A8A"), False),
        ("Tradable", axes_v["tradable"].verdict.replace("_", " "),
         _AXIS_COLORS.get(axes_v["tradable"].verdict, "#8A8A8A"), False),
    ]
    for (label, value, color, big), x in zip(chips, (0.02, 0.28, 0.52, 0.76)):
        ax.text(x, 0.62, label, fontsize=9.5, color="#666666", transform=ax.transAxes)
        ax.text(x, 0.12, f" {value} ", fontsize=15 if big else 12, fontweight="bold",
                color="white", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.34", fc=color, ec="none"))


def _definition_meta_line(spec: FactorSpec) -> str:
    """The structured one-line factor-definition metadata (pure -> unit-testable)."""
    return (
        f"inputs: {', '.join(spec.input_fields)}   ·   "
        f"expected sign: {spec.expected_ic_sign:+d}   ·   "
        f"horizon: {spec.forward_return_horizon} ({spec.return_basis})   ·   "
        f"family: {spec.family or '—'}   ·   "
        f"min-history: {spec.min_history_bars} bars   ·   "
        f"price: {spec.price_adjust}   ·   "
        f"intraday: {spec.is_intraday}"
    )


def _definition_block_line(spec: FactorSpec) -> str | None:
    """The intraday minute-execution block line, or None for a daily factor."""
    if not spec.is_intraday:
        return None
    return (
        f"minute block —  cutoff: {spec.decision_cutoff}  ·  lag: {spec.data_lag}"
        f"  ·  session-open: {spec.session_open}  ·  exec: {spec.execution_model}"
        f"  ·  window: {spec.execution_window}"
    )


def _definition_band(ax, data: DashboardData) -> None:
    """MANDATORY panel: how the factor is computed, from the FactorSpec.

    Stacked vertically so nothing overlaps: header + id on one line, the full
    computation description wrapped full-width below it, then the structured
    metadata line (and the minute block for an intraday factor).
    """
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=_DEF_BG, edgecolor="#D8DEE7", lw=1.0,
                               zorder=0))
    spec = data.spec
    ax.text(0.012, 0.90, "FACTOR DEFINITION", fontsize=10.5, fontweight="bold",
            color="#33475B", transform=ax.transAxes, va="top")
    ax.text(0.185, 0.90, f"{spec.factor_id}  (v{spec.version})", fontsize=9.5,
            fontweight="bold", color="#111111", transform=ax.transAxes, va="top",
            family="DejaVu Sans Mono")
    ax.text(0.012, 0.62, textwrap.fill(spec.description, width=150), fontsize=9.2,
            color="#222222", transform=ax.transAxes, va="top")
    block = _definition_block_line(spec)
    meta_y = 0.20 if block is not None else 0.10
    ax.text(0.012, meta_y, _definition_meta_line(spec), fontsize=8.6,
            color="#41505F", transform=ax.transAxes, va="bottom",
            family="DejaVu Sans Mono")
    if block is not None:
        ax.text(0.012, 0.04, block, fontsize=8.4, color="#7A5C00",
                transform=ax.transAxes, va="bottom", family="DejaVu Sans Mono")


# --------------------------------------------------------------------------- #
# data panels
# --------------------------------------------------------------------------- #
def _panel_hero_nav(ax, data: DashboardData) -> None:
    qr = data.quantile_returns
    if qr.empty:
        _empty(ax, "Layered backtest — quantile NAV & long-short")
        return
    qcols = _qcols(qr)
    nav = (1.0 + qr[qcols].fillna(0.0)).cumprod()
    for i, q in enumerate(qcols):
        tag = (" (low)" if i == 0 else " (high)" if i == len(qcols) - 1 else "")
        ax.plot(nav.index, nav[q].values, color=_Q_COLORS[i % len(_Q_COLORS)],
                lw=1.15, label=f"Q{q}{tag}")
    ax.set_ylabel("Quantile NAV (long-only, equal-weight)", fontsize=9, color=_INK)
    sign = data.spec.expected_ic_sign
    _style(ax, f"Layered backtest — quantile NAV & long-short  "
               f"(Q1 lowest factor · aligned sign {sign:+d})")
    # long Q1 / short Q_top when sign is -1 (else flip) -> GROSS aligned spread.
    lo, hi = (qcols[0], qcols[-1]) if sign < 0 else (qcols[-1], qcols[0])
    ls = qr[lo].fillna(0.0) - qr[hi].fillna(0.0)
    ls_nav = (1.0 + ls).cumprod()
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ls_nav.index, ls_nav.values, color=_LS_MAROON, lw=2.3,
             label=f"Long-Short  Q{lo}−Q{hi}  (gross, right)")
    ax2.set_ylabel("Long-Short NAV (gross)", fontsize=9, color=_LS_MAROON)
    ax2.tick_params(colors=_LS_MAROON, labelsize=8, length=3)
    _date_axis(ax, nav.index)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8.5,
              frameon=False, handlelength=1.6, columnspacing=1.4)


def _panel_ic(ax, data: DashboardData) -> None:
    ic = data.ic.dropna()
    if ic.empty:
        _empty(ax, "Information coefficient")
        return
    ic.index = pd.to_datetime(ic.index)
    ax.bar(ic.index, ic.values, width=2.0, color="#B7C7E2", zorder=1,
           label="RankIC per rebalance")
    ax.axhline(0, color=_INK, lw=0.7)
    ax.set_ylabel("RankIC", fontsize=9, color=_INK)
    _style(ax, "Information coefficient — per-period & cumulative")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ic.index, ic.cumsum().values, color=_LS_MAROON, lw=2.0,
             label="Cumulative RankIC (right)")
    ax2.set_ylabel("Cumulative RankIC", fontsize=9, color=_LS_MAROON)
    ax2.tick_params(colors=_LS_MAROON, labelsize=8, length=3)
    _date_axis(ax, ic.index)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2, fontsize=8.5, frameon=False)


def _panel_quantile_bar(ax, data: DashboardData) -> None:
    fnav = data.payloads.get("return_risk", {}).get("quantile_final_nav")
    if not fnav:
        _empty(ax, "Quantile final NAV  (monotonicity)")
        return
    qs = sorted(fnav, key=lambda c: int(str(c)))
    vals = [fnav[q] for q in qs]
    colors = [_Q_COLORS[i % len(_Q_COLORS)] for i in range(len(qs))]
    ax.bar([f"Q{q}" for q in qs], vals, color=colors, zorder=2, width=0.66)
    ax.axhline(1.0, color=_INK, lw=0.7, ls="--")
    for i, val in enumerate(vals):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                color=_INK)
    ax.set_ylabel("Final NAV", fontsize=9)
    _style(ax, "Quantile final NAV  (monotonicity)")


def _panel_oos(ax, data: DashboardData) -> None:
    oos = data.payloads.get("oos_generalization", {})
    keys = [("train_ic_mean", "Train IC"), ("test_ic_mean", "Test IC"),
            ("holdout_subperiod_1_ic_mean", "Holdout-1"),
            ("holdout_subperiod_2_ic_mean", "Holdout-2")]
    labels = [lbl for k, lbl in keys if oos.get(k) is not None]
    vals = [oos[k] for k, _ in keys if oos.get(k) is not None]
    if not vals:
        _empty(ax, "Out-of-sample sign consistency")
        return
    colors = [_BAR_BLUE if (v or 0) < 0 else _BAR_RED for v in vals]
    ax.bar(labels, vals, color=colors, zorder=2, width=0.62)
    ax.axhline(0, color=_INK, lw=0.7)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:+.4f}", ha="center",
                va="top" if val < 0 else "bottom", fontsize=7.5, color=_INK)
    ax.set_ylabel("mean RankIC", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    _style(ax, "Out-of-sample sign consistency")


def _panel_decay(ax, data: DashboardData) -> None:
    pp = data.payloads.get("predictive_power", {})
    lags = [1, 2, 3, 5]
    pairs = [(k, pp.get(f"ic_decay_stale_lag_{k}_mean")) for k in lags]
    pairs = [(k, v) for k, v in pairs if v is not None]
    if not pairs:
        _empty(ax, "IC decay (signal persistence)")
        return
    xs, vals = [k for k, _ in pairs], [v for _, v in pairs]
    ax.plot(xs, vals, marker="o", color=_BAR_BLUE, lw=1.8, ms=6, zorder=2)
    ax.axhline(0, color=_INK, lw=0.7)
    ax.set_xlabel("stale lag (periods)", fontsize=9)
    ax.set_ylabel("mean RankIC", fontsize=9)
    ax.set_xticks(xs)
    _style(ax, "IC decay (signal persistence)")


def _panel_purity(ax, data: DashboardData) -> None:
    purity = data.payloads.get("purity", {})
    corr = purity.get("mean_rank_corr_with_anchor")
    if not corr:
        _empty(ax, "Purity — corr with book",
               "no known-factor book\n(Incremental NOT_ASSESSED)")
        return
    names = list(corr)
    vals = [corr[n] for n in names]
    colors = [_BAR_RED if v > 0 else _BAR_BLUE for v in vals]
    ax.barh(names, vals, color=colors, zorder=2, height=0.55)
    ax.axvline(0, color=_INK, lw=0.7)
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.10 * max(hi - lo, 0.1)
    ax.set_xlim(lo - pad - 0.12, hi + pad + 0.12)
    for i, val in enumerate(vals):
        ax.text(val + (pad * 0.3 if val >= 0 else -pad * 0.3), i, f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8,
                color=_INK)
    incr = purity.get("incremental_ic_ir")
    ax.set_xlabel("mean rank-corr with book factor", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)
    title = "Purity — corr w/ book"
    if isinstance(incr, (int, float)):
        title += f" (incr ICIR {incr:+.3f})"
    _style(ax, title)


def _panel_cost(ax, data: DashboardData) -> None:
    sc = data.payloads.get("stability_cost", {})
    drag = sc.get("annualized_cost_drag_by_scenario")
    if not drag:
        _empty(ax, "Cost sensitivity")
        return
    ks = sorted(drag, key=lambda c: float(str(c)))
    vals = [drag[k] for k in ks]
    labels = [f"{float(k):.0f}x fee" for k in ks]
    ax.bar(labels, vals, color=[_GOLD, "#D98C00", _BAR_RED][: len(ks)], zorder=2,
           width=0.6)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.1%}", ha="center", va="bottom", fontsize=8.5,
                color=_INK)
    ax.set_ylabel("annualized cost drag", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1.0)
    _style(ax, "Cost sensitivity (turnover × fee)")


def _panel_coverage(ax, data: DashboardData) -> None:
    ax.axis("off")
    cov = data.payloads.get("data_coverage", {})
    sc = data.payloads.get("stability_cost", {})
    ac = sc.get("factor_rank_autocorr_by_lag", {}) or {}
    # payload dict keys may be int (live report) or str (round-tripped JSON).
    autocorr_1 = ac.get(1, ac.get("1"))
    ax.text(0.0, 0.95, "Coverage & turnover", fontsize=11.5, fontweight="bold",
            color="#1a1a1a", transform=ax.transAxes, va="top")
    rows = [
        ("symbols evaluated", cov.get("symbols_evaluated")),
        ("cross-section med", cov.get("cross_section_size_median")),
        ("factor NaN rate", cov.get("factor_nan_rate")),
        ("rank autocorr(1)", autocorr_1),
        ("half-life (periods)", sc.get("half_life_periods")),
        ("PIT universe", cov.get("universe_is_pit")),
    ]

    def _fmt(k, v):
        if v is None:
            return f"{k:<18}: —"
        if k == "factor NaN rate" and isinstance(v, (int, float)):
            return f"{k:<18}: {v:.1%}"
        if isinstance(v, float):
            return f"{k:<18}: {v:.3f}" if abs(v) < 100 else f"{k:<18}: {v:.0f}"
        return f"{k:<18}: {v}"

    ax.text(0.0, 0.72, "\n".join(_fmt(k, v) for k, v in rows), fontsize=9.5,
            color="#333333", family="DejaVu Sans Mono", transform=ax.transAxes,
            va="top")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _render(data: DashboardData, out_path: str | Path) -> Path:
    _apply_rc()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 21.5))
    gs = GridSpec(7, 3, figure=fig,
                  height_ratios=[0.30, 0.38, 0.80, 1.5, 1.1, 1.0, 1.0],
                  hspace=0.62, wspace=0.28,
                  left=0.06, right=0.945, top=0.975, bottom=0.04)
    _header_text(fig.add_subplot(gs[0, :]), data)
    _chips_row(fig.add_subplot(gs[1, :]), data)
    _definition_band(fig.add_subplot(gs[2, :]), data)
    _panel_hero_nav(fig.add_subplot(gs[3, :]), data)
    _panel_ic(fig.add_subplot(gs[4, :]), data)
    _panel_quantile_bar(fig.add_subplot(gs[5, 0]), data)
    _panel_oos(fig.add_subplot(gs[5, 1]), data)
    _panel_decay(fig.add_subplot(gs[5, 2]), data)
    _panel_purity(fig.add_subplot(gs[6, 0]), data)
    _panel_cost(fig.add_subplot(gs[6, 1]), data)
    _panel_coverage(fig.add_subplot(gs[6, 2]), data)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render_factor_dashboard(
    report: FactorEvalReport, ir: object, out_path: str | Path
) -> Path:
    """Render ``report`` (+ its ``ir`` series) to a one-PNG dashboard at ``out_path``.

    ``ir`` is the :class:`~analytics.eval.ir.StandardEvalIR` produced alongside the
    report (see :meth:`StandardFactorEvaluator.evaluate_with_ir`); only its ``ic``
    and ``quantile_returns`` are read.
    """
    return _render(DashboardData.from_report(report, ir), out_path)


__all__ = ["DashboardData", "render_factor_dashboard"]
