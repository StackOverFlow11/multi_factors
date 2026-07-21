"""Implementation-INDEPENDENT sanity checks on the ``exec_to_exec`` return series.

A wrong return series does not announce itself: every factor scored against it
still produces a plausible-looking IC, a plausible-looking quantile spread and a
confident verdict. Unit tests cannot catch this on their own — they check the code
against the same understanding that wrote it. So these four checks are deliberately
built from a DIFFERENT direction:

  1. **Agreement.** Two one-day return series for the same stocks must move
     together. The per-date cross-sectional correlation against ``close_to_close``
     should sit far above :data:`CLOSE_CORR_FLOOR`; a median below it means the
     series is not measuring what it claims, and :func:`check_exec_basis` RAISES
     rather than letting eleven factors be scored against it.
  2. **Magnitude.** The distribution (mean / std / p1 / p99) must be the same order
     as the daily cross-section, not 10x or 0.1x it.
  3. **Hand computation.** Five (date, symbol) pairs are recomputed straight from
     the raw cached 1min parquet with plain arithmetic — this module never calls
     ``resolve_fill`` / ``bar_execution_price`` / ``build_execution_prices``, so it
     cannot inherit a bug from them. Agreement to 1e-9 is checked and the five rows
     are published so a reader can redo them.
  4. **Ex-date.** At least one holding period spanning a corporate action, shown
     BOTH ways: the raw VWAP ratio (which reads the mechanical drop as a loss) and
     the adjusted return (which does not).

Everything here is read-only: no cache is warmed, no artifact is rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.eval.ir import cross_section_corr
from data.cache.intraday_cache import ENDPOINT as MINUTE_ENDPOINT
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from qt.exec_forward_returns import ExecBasisParams

#: Median per-date cross-sectional correlation with ``close_to_close`` below which
#: the exec series is treated as broken. Two one-day return series over the same
#: names differ only by their intraday measurement point; ~0.9 is a floor, not a
#: target (the observed value should be well above it).
CLOSE_CORR_FLOOR = 0.90

#: How many pairs get hand-recomputed. Five is what a reader will actually check.
HAND_CHECK_ROWS = 5

#: Tolerance for "the hand computation agrees" (double-precision round-trip slack).
HAND_CHECK_TOL = 1e-9

#: Deterministic sample: the same five rows every run, so the published check is
#: reproducible rather than a different lottery each time.
HAND_CHECK_SEED = 20260721


@dataclass(frozen=True)
class ExecBasisSanity:
    """The four §3 checks, as facts."""

    corr_median: float
    corr_p10: float
    corr_p90: float
    corr_dates: int
    corr_median_spearman: float
    exec_stats: dict[str, float]
    close_stats: dict[str, float]
    hand_checks: tuple[dict, ...]
    hand_check_max_abs_diff: float
    ex_date_checks: tuple[dict, ...]
    corr_floor: float = CLOSE_CORR_FLOOR

    @property
    def corr_ok(self) -> bool:
        return bool(np.isfinite(self.corr_median) and self.corr_median >= self.corr_floor)

    def headline(self) -> dict[str, object]:
        """The compact form that travels INSIDE every exec-basis report."""
        return {
            "sanity_corr_vs_close_to_close_median": self.corr_median,
            "sanity_corr_vs_close_to_close_p10": self.corr_p10,
            "sanity_corr_vs_close_to_close_p90": self.corr_p90,
            "sanity_corr_spearman_median": self.corr_median_spearman,
            "sanity_corr_dates": self.corr_dates,
            "sanity_corr_floor": self.corr_floor,
            "sanity_exec_return_mean": self.exec_stats["mean"],
            "sanity_exec_return_std": self.exec_stats["std"],
            "sanity_exec_return_p1": self.exec_stats["p1"],
            "sanity_exec_return_p99": self.exec_stats["p99"],
            "sanity_close_return_mean": self.close_stats["mean"],
            "sanity_close_return_std": self.close_stats["std"],
            "sanity_close_return_p1": self.close_stats["p1"],
            "sanity_close_return_p99": self.close_stats["p99"],
            "sanity_hand_checked_rows": len(self.hand_checks),
            "sanity_hand_check_max_abs_diff": self.hand_check_max_abs_diff,
            "sanity_ex_date_pairs_shown": len(self.ex_date_checks),
        }


def _stats(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p1", "p99", "n")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "p1": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "n": float(values.size),
    }


def exit_anchor_dates(index: pd.MultiIndex, horizon: int) -> pd.Series:
    """Per row, the date its forward return is measured AT — ``h`` periods later.

    Uses the same positional ``groupby(symbol).shift(-h)`` the return itself uses,
    so an anchor reported here is the anchor the return actually used.
    """
    dates = pd.Series(index.get_level_values(DATE_LEVEL), index=index)
    return dates.groupby(level=SYMBOL_LEVEL, sort=False, group_keys=False).shift(
        -horizon
    )


def _hand_price(
    store: IntradayParquetStore,
    symbol: str,
    date: pd.Timestamp,
    params: ExecBasisParams,
) -> dict[str, object]:
    """Recompute one execution bar's VWAP from the raw parquet, by hand.

    Plain pandas + arithmetic on the STORED columns: pick the bars whose
    ``bar_end`` falls in the window, take the earliest, divide ``amount`` by
    ``volume``. No execution-layer helper is imported, on purpose.
    """
    day = pd.Timestamp(date).normalize()
    stored = store.read_range(
        MINUTE_ENDPOINT,
        symbol,
        RAW_INTRADAY_FREQ,
        day,
        day + pd.Timedelta("23:59:59"),
    )
    lo = day + pd.Timedelta(params.execution_window[0])
    hi = day + pd.Timedelta(params.execution_window[1])
    bar_end = pd.to_datetime(stored["bar_end"]) if len(stored) else pd.Series(dtype="datetime64[ns]")
    window = stored.loc[(bar_end >= lo) & (bar_end <= hi)] if len(stored) else stored
    if window.empty:
        return {"bar_time": None, "amount": float("nan"), "volume": float("nan"),
                "vwap": float("nan")}
    row = window.sort_values("bar_end").iloc[0]
    amount = float(row["amount"])
    volume = float(row["volume"])
    vwap = amount / volume if volume > 0 else float("nan")
    return {
        "bar_time": pd.Timestamp(row["bar_end"]),
        "amount": amount,
        "volume": volume,
        "vwap": vwap,
    }


def _hand_check_rows(
    frame: pd.DataFrame,
    exec_returns: pd.Series,
    exits: pd.Series,
    store: IntradayParquetStore,
    params: ExecBasisParams,
    n_rows: int,
) -> tuple[list[dict], float]:
    """Recompute ``n_rows`` returns end-to-end from raw bars and compare."""
    finite = exec_returns[np.isfinite(exec_returns.to_numpy(dtype=float))]
    if finite.empty:
        return [], float("nan")
    rng = np.random.default_rng(HAND_CHECK_SEED)
    take = min(n_rows, len(finite))
    positions = rng.choice(len(finite), size=take, replace=False)
    positions.sort()

    rows: list[dict] = []
    worst = 0.0
    for pos in positions:
        key = finite.index[pos]
        entry_date, symbol = pd.Timestamp(key[0]).normalize(), str(key[1])
        exit_date = pd.Timestamp(exits.loc[key])
        entry = _hand_price(store, symbol, entry_date, params)
        exit_ = _hand_price(store, symbol, exit_date, params)
        af_entry = float(frame.loc[(entry_date, symbol), "adj_factor"])
        af_exit = float(frame.loc[(exit_date, symbol), "adj_factor"])
        hand = (exit_["vwap"] * af_exit) / (entry["vwap"] * af_entry) - 1.0
        generated = float(finite.iloc[pos])
        diff = abs(hand - generated)
        worst = max(worst, diff if np.isfinite(diff) else float("inf"))
        rows.append(
            {
                "symbol": symbol,
                "entry_date": str(entry_date.date()),
                "entry_bar": str(entry["bar_time"]),
                "entry_amount": entry["amount"],
                "entry_volume": entry["volume"],
                "entry_vwap": entry["vwap"],
                "entry_adj_factor": af_entry,
                "exit_date": str(exit_date.date()),
                "exit_bar": str(exit_["bar_time"]),
                "exit_amount": exit_["amount"],
                "exit_volume": exit_["volume"],
                "exit_vwap": exit_["vwap"],
                "exit_adj_factor": af_exit,
                "hand_return": hand,
                "generated_return": generated,
                "abs_diff": diff,
            }
        )
    return rows, worst


def _ex_date_rows(
    frame: pd.DataFrame,
    exec_returns: pd.Series,
    exits: pd.Series,
    n_rows: int,
) -> list[dict]:
    """Holding periods that straddle a corporate action, shown adjusted and not."""
    finite = exec_returns[np.isfinite(exec_returns.to_numpy(dtype=float))]
    if finite.empty:
        return []
    af = frame["adj_factor"]
    entry_af = af.reindex(finite.index).to_numpy(dtype=float)
    exit_keys = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex(exits.reindex(finite.index).to_numpy()),
            finite.index.get_level_values(SYMBOL_LEVEL),
        ],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )
    exit_af = af.reindex(exit_keys).to_numpy(dtype=float)
    raw = frame["raw_exec_price"]
    entry_raw = raw.reindex(finite.index).to_numpy(dtype=float)
    exit_raw = raw.reindex(exit_keys).to_numpy(dtype=float)

    ratio = np.where(np.isfinite(entry_af) & (entry_af > 0), exit_af / entry_af, np.nan)
    moved = np.isfinite(ratio) & (np.abs(ratio - 1.0) > 1e-9)
    if not moved.any():
        return []
    order = np.argsort(-np.abs(ratio - 1.0))
    rows: list[dict] = []
    for pos in order:
        if not moved[pos]:
            break
        key = finite.index[pos]
        unadjusted = exit_raw[pos] / entry_raw[pos] - 1.0
        rows.append(
            {
                "symbol": str(key[1]),
                "entry_date": str(pd.Timestamp(key[0]).date()),
                "exit_date": str(pd.Timestamp(exit_keys[pos][0]).date()),
                "entry_raw_vwap": float(entry_raw[pos]),
                "exit_raw_vwap": float(exit_raw[pos]),
                "entry_adj_factor": float(entry_af[pos]),
                "exit_adj_factor": float(exit_af[pos]),
                "unadjusted_return": float(unadjusted),
                "adjusted_return": float(finite.iloc[pos]),
                "correction": float(finite.iloc[pos] - unadjusted),
            }
        )
        if len(rows) >= n_rows:
            break
    return rows


def check_exec_basis(
    frame: pd.DataFrame,
    exec_returns: pd.Series,
    close_returns: pd.Series,
    params: ExecBasisParams,
    cache_root: str,
    horizon: int,
    *,
    n_hand_checks: int = HAND_CHECK_ROWS,
    n_ex_date_checks: int = 1,
    corr_floor: float = CLOSE_CORR_FLOOR,
    enforce: bool = True,
) -> ExecBasisSanity:
    """Run the four checks; RAISE when the agreement check fails.

    ``enforce=False`` is for tests that need to inspect a deliberately broken
    series. A production caller leaves it True: a return basis that disagrees with
    close-to-close is a bug signal, and the eleven verdicts that would be built on
    it are worth less than nothing.
    """
    aligned_close = close_returns.reindex(exec_returns.index)
    ic_dates = pd.Index(
        pd.unique(exec_returns.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    ).sort_values()
    pearson = cross_section_corr(
        exec_returns, aligned_close, rank=False, dates=ic_dates
    )
    spearman = cross_section_corr(
        exec_returns, aligned_close, rank=True, dates=ic_dates
    )
    usable = pearson.dropna()

    exits = exit_anchor_dates(exec_returns.index, horizon)
    store = IntradayParquetStore(cache_root)
    hand_rows, worst = _hand_check_rows(
        frame, exec_returns, exits, store, params, n_hand_checks
    )
    ex_rows = _ex_date_rows(frame, exec_returns, exits, n_ex_date_checks)

    sanity = ExecBasisSanity(
        corr_median=float(usable.median()) if len(usable) else float("nan"),
        corr_p10=float(usable.quantile(0.10)) if len(usable) else float("nan"),
        corr_p90=float(usable.quantile(0.90)) if len(usable) else float("nan"),
        corr_dates=int(len(usable)),
        corr_median_spearman=(
            float(spearman.dropna().median()) if spearman.notna().any() else float("nan")
        ),
        exec_stats=_stats(exec_returns),
        close_stats=_stats(aligned_close),
        hand_checks=tuple(hand_rows),
        hand_check_max_abs_diff=worst,
        ex_date_checks=tuple(ex_rows),
        corr_floor=corr_floor,
    )
    if enforce and not sanity.corr_ok:
        raise ValueError(
            f"exec_to_exec sanity check FAILED: the median per-date cross-sectional "
            f"correlation with close_to_close is {sanity.corr_median:.4f} over "
            f"{sanity.corr_dates} dates, below the {corr_floor:.2f} floor. Two "
            f"one-day return series over the same names cannot disagree this much "
            f"unless one of them is wrong — most likely the execution bar, the VWAP, "
            f"the adjustment identity or the h-step alignment. STOP and investigate; "
            f"do NOT explain this away and do not score factors against it."
        )
    if enforce and np.isfinite(worst) and worst > HAND_CHECK_TOL and hand_rows:
        raise ValueError(
            f"exec_to_exec sanity check FAILED: a hand recomputation from the raw "
            f"cached 1min bars disagrees with the generated return by {worst:.3e} "
            f"(tolerance {HAND_CHECK_TOL:.0e}). The generated series does not equal "
            f"(raw VWAP * adj_factor) exit / entry - 1 for at least one pair."
        )
    return sanity


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return "n/a" if value is None else str(value)


def render_sanity_report(
    sanity: ExecBasisSanity,
    params: ExecBasisParams,
    coverage: dict[str, object],
    *,
    key: str,
    artifact_path: str,
    horizon: int,
    minute_live_calls: int,
) -> str:
    """Deterministic Markdown for the four checks — the audit trail for §3."""
    lines: list[str] = []
    lines.append("# exec-to-exec return basis — sanity checks")
    lines.append("")
    lines.append(
        "The eleven minute-derived factors are scored against the execution-anchored "
        "return below. These checks exist because a wrong return series still "
        "produces confident-looking verdicts."
    )
    lines.append("")
    lines.append("## Basis and execution parameters")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append("| return basis | `exec_to_exec` |")
    lines.append(f"| forward return horizon | {horizon} evaluation period(s) |")
    lines.append(f"| signal cutoff (decision) | {params.decision_cutoff} |")
    lines.append(f"| data lag | {params.data_lag} |")
    lines.append(f"| session open | {params.session_open} |")
    lines.append(f"| execution model | {params.execution_model} |")
    lines.append(
        f"| execution window | [{params.execution_window[0]}, "
        f"{params.execution_window[1]}] |"
    )
    lines.append(f"| price basis | {params.execution_price_basis} (bar amount/volume) |")
    lines.append("| adjustment | `(raw*af)_exit / (raw*af)_entry - 1` (applied) |")
    lines.append(f"| parameter source | {params.source} |")
    lines.append(f"| artifact | `{artifact_path}` (key `{key}`) |")
    lines.append(f"| stk_mins live calls | {minute_live_calls} |")
    lines.append("")

    lines.append("## 1. Agreement with close_to_close")
    lines.append("")
    lines.append(
        f"Per-date cross-sectional correlation over {sanity.corr_dates} dates — "
        f"median **{_fmt(sanity.corr_median, 4)}** "
        f"(p10 {_fmt(sanity.corr_p10, 4)}, p90 {_fmt(sanity.corr_p90, 4)}); "
        f"Spearman median {_fmt(sanity.corr_median_spearman, 4)}. "
        f"Floor {sanity.corr_floor:.2f} — "
        f"{'PASS' if sanity.corr_ok else 'FAIL'}."
    )
    lines.append("")

    lines.append("## 2. Magnitude")
    lines.append("")
    lines.append("| series | mean | std | p1 | p99 | n |")
    lines.append("|---|---|---|---|---|---|")
    for name, stats in (("exec_to_exec", sanity.exec_stats), ("close_to_close", sanity.close_stats)):
        lines.append(
            f"| {name} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
            f"{_fmt(stats['p1'])} | {_fmt(stats['p99'])} | {_fmt(stats['n'], 0)} |"
        )
    lines.append("")

    lines.append("## 3. Hand recomputation from the raw cached bars")
    lines.append("")
    lines.append(
        f"{len(sanity.hand_checks)} pair(s), recomputed with plain arithmetic on the "
        f"stored 1min rows (no execution-layer helper involved). Worst absolute "
        f"difference **{_fmt(sanity.hand_check_max_abs_diff, 12)}**."
    )
    lines.append("")
    if sanity.hand_checks:
        lines.append(
            "| symbol | entry date | entry bar | amount | volume | vwap | af | "
            "exit date | exit bar | vwap | af | hand return | generated | diff |"
        )
        lines.append("|" + "---|" * 14)
        for row in sanity.hand_checks:
            lines.append(
                f"| {row['symbol']} | {row['entry_date']} | {row['entry_bar']} | "
                f"{_fmt(row['entry_amount'], 2)} | {_fmt(row['entry_volume'], 2)} | "
                f"{_fmt(row['entry_vwap'], 6)} | {_fmt(row['entry_adj_factor'], 6)} | "
                f"{row['exit_date']} | {row['exit_bar']} | "
                f"{_fmt(row['exit_vwap'], 6)} | {_fmt(row['exit_adj_factor'], 6)} | "
                f"{_fmt(row['hand_return'], 10)} | {_fmt(row['generated_return'], 10)} | "
                f"{_fmt(row['abs_diff'], 12)} |"
            )
        lines.append("")

    lines.append("## 4. Ex-date holding period")
    lines.append("")
    if not sanity.ex_date_checks:
        lines.append(
            "No holding period in this window straddles an `adj_factor` change, so "
            "the adjustment cannot be demonstrated on real data here."
        )
    else:
        lines.append(
            "`adj_factor` moves inside the holding period, so the raw VWAP ratio "
            "carries a mechanical drop the adjusted return removes."
        )
        lines.append("")
        lines.append(
            "| symbol | entry | exit | raw vwap in | raw vwap out | af in | af out | "
            "unadjusted | adjusted | correction |"
        )
        lines.append("|" + "---|" * 10)
        for row in sanity.ex_date_checks:
            lines.append(
                f"| {row['symbol']} | {row['entry_date']} | {row['exit_date']} | "
                f"{_fmt(row['entry_raw_vwap'])} | {_fmt(row['exit_raw_vwap'])} | "
                f"{_fmt(row['entry_adj_factor'])} | {_fmt(row['exit_adj_factor'])} | "
                f"{_fmt(row['unadjusted_return'])} | {_fmt(row['adjusted_return'])} | "
                f"{_fmt(row['correction'])} |"
            )
    lines.append("")

    lines.append("## Coverage loss vs close_to_close")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    for label, value in coverage.items():
        if isinstance(value, dict):
            for cause, count in value.items():
                lines.append(f"| {label}.{cause} | {count} |")
        else:
            lines.append(f"| {label} | {_fmt(value, 4)} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CLOSE_CORR_FLOOR",
    "HAND_CHECK_ROWS",
    "HAND_CHECK_SEED",
    "HAND_CHECK_TOL",
    "ExecBasisSanity",
    "check_exec_basis",
    "exit_anchor_dates",
    "render_sanity_report",
]
