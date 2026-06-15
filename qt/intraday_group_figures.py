"""Matplotlib (Agg) figures for the I5d MMP quintile grouped backtest.

Pure plotting helpers: each takes already-computed frames/series and an output
path, renders ONE PNG with a non-interactive ``Agg`` backend (no display needed),
and returns the written path. They are deliberately decoupled from the run logic
so they can be unit-tested with toy NAV frames (goal §Tests 5).

Direction convention is made explicit on every figure: **Q1 = lowest MMP score,
QN = highest** — the legend/labels say "low"/"high" so the quantile direction can
never be misread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a display.

import matplotlib.pyplot as plt  # noqa: E402  (must follow the use("Agg") call)
import pandas as pd  # noqa: E402

# Required metric columns for the grouped-bar figure (one bar group per metric).
METRIC_COLUMNS: tuple[str, ...] = (
    "annual_return",
    "max_drawdown",
    "mean_turnover",
    "total_cost",
)
_METRIC_TITLES = {
    "annual_return": "Annualized return",
    "max_drawdown": "Max drawdown",
    "mean_turnover": "Mean turnover",
    "total_cost": "Total cost",
}


def _group_label(group: int, n_groups: int) -> str:
    """``Q{g}`` with a low/high tag on the extremes (direction is unmissable)."""
    if group == 1:
        return f"Q{group} (low)"
    if group == n_groups:
        return f"Q{group} (high)"
    return f"Q{group}"


def plot_quintile_nav(
    nav_by_group: dict[int, pd.DataFrame], out_path: str | Path, n_groups: int
) -> Path:
    """NAV curves for Q1..QN on one chart (x = rebalance date, y = NAV)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in sorted(nav_by_group):
        nav = nav_by_group[group]
        if nav is None or nav.empty or "nav" not in nav.columns:
            continue
        ax.plot(
            [pd.Timestamp(d) for d in nav.index],
            nav["nav"].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            label=_group_label(group, n_groups),
        )
    ax.set_title("MMP quintile group NAV (Q1 = lowest score, QN = highest)")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("NAV (start = 1.0)")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_spread_curve(
    spread_curve: pd.Series,
    out_path: str | Path,
    *,
    low_label: str = "Q1",
    high_label: str = "Q5",
) -> Path:
    """Cumulative synthetic ``high − low`` group spread curve (long-only legs)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if spread_curve is not None and len(spread_curve) > 0:
        ax.plot(
            [pd.Timestamp(d) for d in spread_curve.index],
            spread_curve.to_numpy(dtype=float),
            color="tab:purple",
            marker="o",
            markersize=3,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Cumulative synthetic {high_label}−{low_label} spread "
        "(long-only leg difference, NOT a dollar-neutral book)"
    )
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel(f"Cumulative {high_label}−{low_label} net return")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_group_metrics(
    metrics: pd.DataFrame, out_path: str | Path, n_groups: int
) -> Path:
    """Compact 2x2 grouped-bar panel: annual return / maxDD / turnover / cost."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted(metrics.index)
    labels = [_group_label(int(g), n_groups) for g in groups]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, METRIC_COLUMNS):
        if col in metrics.columns:
            values = [float(metrics.loc[g, col]) for g in groups]
        else:
            values = [float("nan")] * len(groups)
        ax.bar(labels, values, color="tab:blue")
        ax.set_title(_METRIC_TITLES.get(col, col))
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    fig.suptitle(
        "MMP quintile group metrics (Q1 = lowest score, QN = highest)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
