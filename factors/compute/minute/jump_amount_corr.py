"""Price-jump turnover-correlation factor (PR-C): math + surface (D2).

Reproduces the Kaiyuan report §6 factor (``价格跳跃成交额相关性``, full-A RankIC
-10.23%): the trailing-``lookback_days``-trading-day lagged Pearson correlation
between the traded ``amount`` at price-JUMP minutes and the amount at the
STRICTLY-next minute. The daily value at ``(date d, symbol s)`` uses ONLY bars
at dates ``<= d`` (the trailing window ending at d), so a factor value never
sees a future bar (invariant #1); it is meant to trade close-to-close from d+1.

Definition (LOCKED, per bar of one (symbol, day) session):
  * ``amplitude = (high - low) / open``           (guard open > 0, amount finite);
  * ``jump``     = within-(symbol, day) amplitude z-score (ddof=1) ``> jump_z``;
  * pair each jump minute ``t`` with the STRICTLY-next minute (same session,
    ``bar_end`` gap exactly 60s — this excludes the lunch break AND the close);
  * ``factor(s, d)`` = Pearson corr(amount[jump t], amount[t+1]) over ALL
    jump-pairs whose date is in the trailing ``lookback_days`` TRADING DAYS
    (the symbol's own minute-trading days) ending at d; NaN when fewer than
    ``min_pairs`` pairs fall in the window (or the correlation is undefined).

Vectorized (no per-rebalance-date python loop): the trailing-window correlation
is a rolling sum of per-day sufficient statistics (n, sum x, sum y, sum x^2,
sum y^2, sum xy) over the trading-day axis, then Pearson's closed form. Rolling
over ROWS of the per-day-sorted stats == a trailing window of trading days,
because consecutive rows are consecutive trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_schema import (
    DATE_LEVEL,
    SYMBOL_LEVEL,
    validate_intraday_bars,
)
from factors.base import Factor
from factors.compute.minute.primitives import (
    ONE_MINUTE_SECONDS,
    empty_factor_series,
)
from factors.spec import FactorSpec, PanelField

# Factor DEFINITION constants (reproduced from the report; NOT tuned knobs). The
# daily value is a trailing-``JUMP_LOOKBACK_DAYS``-trading-day lagged correlation
# between the traded ``amount`` at price-JUMP minutes and the amount at the
# STRICTLY-next minute; a jump minute is one whose within-(symbol, day) amplitude
# z-score exceeds ``JUMP_Z``. Requires at least ``JUMP_MIN_PAIRS`` jump-pairs.
JUMP_LOOKBACK_DAYS = 20
JUMP_MIN_PAIRS = 10
JUMP_Z = 1.0


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def compute_jump_amount_corr(
    bars: pd.DataFrame,
    *,
    lookback_days: int = JUMP_LOOKBACK_DAYS,
    min_pairs: int = JUMP_MIN_PAIRS,
    jump_z: float = JUMP_Z,
    name: str = "jump_amount_corr",
) -> pd.Series:
    """PIT-safe daily "price-jump turnover correlation" factor from 1min ``bars``.

    See the module docstring for the LOCKED definition.

    Args:
        bars: normalized 1min bars (:mod:`data.clean.intraday_schema`),
            ``MultiIndex(time, symbol)``. May carry one or many symbols; the
            grouping is strictly per symbol (no cross-symbol leakage).
        lookback_days: trailing trading-day window length (part of the definition).
        min_pairs: minimum jump-pairs in the window for a finite value.
        jump_z: within-day amplitude z-score threshold defining a jump minute.
        name: the returned Series name (the factor-panel column name).

    Returns:
        ``MultiIndex(date, symbol)`` Series (midnight-normalized dates) of the daily
        factor value, sorted, named ``name``. Pure: never mutates ``bars``.
    """
    validate_intraday_bars(bars)
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1; got {lookback_days!r}.")
    if min_pairs < 2:
        # Pearson correlation needs at least 2 points; below that it is undefined.
        raise ValueError(f"min_pairs must be >= 2; got {min_pairs!r}.")
    if len(bars) == 0:
        return empty_factor_series(name)

    work = bars.reset_index()[
        [SYMBOL_LEVEL, "bar_end", "open", "high", "low", "amount"]
    ].copy()
    # Guard bad rows BEFORE anything else: a non-positive open makes the amplitude
    # meaningless and a non-finite amount would poison the correlation.
    work = work[(work["open"] > 0.0) & np.isfinite(work["amount"].to_numpy(dtype=float))]
    if work.empty:
        return empty_factor_series(name)
    work[DATE_LEVEL] = work["bar_end"].dt.normalize()
    # Sort so the "strictly next minute" shift and the per-symbol trading-day
    # rolling both see bars in chronological order within each (symbol, day).
    work = work.sort_values([SYMBOL_LEVEL, "bar_end"], kind="mergesort")

    work["amp"] = (work["high"] - work["low"]) / work["open"]
    by_session = work.groupby([SYMBOL_LEVEL, DATE_LEVEL], sort=False)
    mean_amp = by_session["amp"].transform("mean")
    std_amp = by_session["amp"].transform("std")  # ddof=1 (pandas default)
    zscore = (work["amp"] - mean_amp) / std_amp
    next_bar_end = by_session["bar_end"].shift(-1)
    amt_next = by_session["amount"].shift(-1)
    gap = (next_bar_end - work["bar_end"]).dt.total_seconds()
    is_jump = (zscore > jump_z) & (gap == ONE_MINUTE_SECONDS)

    pairs = pd.DataFrame(
        {
            SYMBOL_LEVEL: work[SYMBOL_LEVEL].to_numpy(),
            DATE_LEVEL: work[DATE_LEVEL].to_numpy(),
            "x": work["amount"].to_numpy(dtype=float),
            "y": amt_next.to_numpy(dtype=float),
        }
    ).loc[is_jump.to_numpy()]

    # Per-(symbol, day) sufficient statistics of the jump-pairs.
    if pairs.empty:
        stats = pd.DataFrame(
            columns=["cnt", "sx", "sy", "sxx", "syy", "sxy"], dtype=float
        )
        stats.index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
            names=[SYMBOL_LEVEL, DATE_LEVEL],
        )
    else:
        pairs = pairs.assign(
            xx=pairs["x"] * pairs["x"],
            yy=pairs["y"] * pairs["y"],
            xy=pairs["x"] * pairs["y"],
        )
        stats = pairs.groupby([SYMBOL_LEVEL, DATE_LEVEL], sort=True).agg(
            cnt=("x", "size"),
            sx=("x", "sum"),
            sy=("y", "sum"),
            sxx=("xx", "sum"),
            syy=("yy", "sum"),
            sxy=("xy", "sum"),
        )

    # Trading-day axis = every (symbol, day) the symbol has minute bars on, so the
    # rolling window counts TRADING DAYS (days with no jump still occupy a row, as
    # zeros — they consume one of the trailing ``lookback_days`` slots).
    axis = (
        work[[SYMBOL_LEVEL, DATE_LEVEL]]
        .drop_duplicates()
        .sort_values([SYMBOL_LEVEL, DATE_LEVEL], kind="mergesort")
    )
    full_index = pd.MultiIndex.from_arrays(
        [axis[SYMBOL_LEVEL].to_numpy(), axis[DATE_LEVEL].to_numpy()],
        names=[SYMBOL_LEVEL, DATE_LEVEL],
    )
    dense = stats.reindex(full_index).fillna(0.0)
    rolled = (
        dense.groupby(level=SYMBOL_LEVEL, sort=False)
        .rolling(lookback_days, min_periods=1)
        .sum()
    )
    # groupby.rolling prepends the group key -> drop it, keep (symbol, date).
    rolled.index = rolled.index.droplevel(0)

    n = rolled["cnt"].to_numpy(dtype=float)
    sx = rolled["sx"].to_numpy(dtype=float)
    sy = rolled["sy"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n * rolled["sxy"].to_numpy(dtype=float) - sx * sy
        var_x = n * rolled["sxx"].to_numpy(dtype=float) - sx * sx
        var_y = n * rolled["syy"].to_numpy(dtype=float) - sy * sy
        den = np.sqrt(var_x * var_y)
        corr = np.where((n >= min_pairs) & (den > 0.0), cov / den, np.nan)
    corr = np.clip(corr, -1.0, 1.0)

    out = pd.Series(corr, index=rolled.index, name=name)
    out.index = out.index.set_names([SYMBOL_LEVEL, DATE_LEVEL])
    return out.reorder_levels([DATE_LEVEL, SYMBOL_LEVEL]).sort_index()


class JumpAmountCorrFactor(Factor):
    """Price-jump turnover-correlation factor (daily signal, minute-derived).

    ``compute`` reads the pre-aggregated daily column the runner placed on the
    panel (produced by :func:`compute_jump_amount_corr`); it does NO minute work
    of its own, mirroring the value / financial factors that surface an enriched
    column.

    Args:
        lookback_days: trailing trading-day window; part of the factor DEFINITION
            (reproduced from the report), not a tuned knob. It only names the
            column so a non-default window cannot silently mislabel it.
    """

    name: str = f"jump_amount_corr_{JUMP_LOOKBACK_DAYS}"

    def __init__(self, lookback_days: int = JUMP_LOOKBACK_DAYS) -> None:
        if not isinstance(lookback_days, int) or lookback_days < 1:
            raise ValueError(
                f"jump-amount-corr lookback_days must be a positive integer; got "
                f"{lookback_days!r}."
            )
        self._lookback_days = lookback_days
        self.name = f"jump_amount_corr_{lookback_days}"

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract; a property so ``factor_id`` tracks the window.

        expected_ic_sign=-1: the report's RankIC mean is -10.23% (full A, market-cap
        + industry neutral) — high jump-amount-correlation predicts LOWER forward
        returns. The sign is fixed BEFORE the run; a validated prototype reproduced
        it (mean RankIC -0.074 on 2022-2024 sampled names). is_intraday=False by the
        module docstring's reasoning (daily signal traded close-to-close).
        min_history_bars=0: the warm-up is DATA-dependent (a value appears once
        >= ``JUMP_MIN_PAIRS`` jump-pairs accumulate in the trailing window), not a
        fixed leading count — the honest NaN rate is reported by data_coverage
        rather than hidden behind a fabricated warm-up window.

        D1 declarations (D0 pre-assignment table row 1): adjustment=
        returns_invariant — the jump is a within-(symbol, day) amplitude
        z-score of the SAME-DAY ratio (high-low)/open (this module's
        ``compute_jump_amount_corr``) and the correlated quantity is
        ``amount`` (anchor-free), so the adjustment anchor cancels.
        overnight_boundary=none — jump/amount pairs are strictly same-session
        adjacent minutes (the exact-60s gap test); no raw-price comparison
        crosses the overnight boundary.
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Price-jump turnover correlation (Kaiyuan report §6): trailing "
                f"{self._lookback_days}-trading-day lagged Pearson corr between the "
                f"traded amount at price-JUMP minutes (within-day amplitude z-score "
                f">1) and the amount at the strictly-next minute. Derived from 1min "
                f"bars but a DAILY signal traded close-to-close; >= {JUMP_MIN_PAIRS} "
                f"jump-pairs required else NaN."
            ),
            expected_ic_sign=-1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            # The 1min bar fields the upstream aggregation is derived from. These
            # are declared for honest provenance disclosure (data_coverage lists
            # them); the daily panel surfaces the pre-aggregated column itself.
            input_fields=("high", "low", "open", "amount"),
            requires=_minute_requires("high", "low", "open", "amount"),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="microstructure",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily jump-amount-corr column off ``panel``.

        The runner runs ``compute_jump_amount_corr`` on the minute cache upstream
        and joins the result as ``self.name``; here we only surface it, so this
        factor does no temporal logic and cannot introduce lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"JumpAmountCorrFactor needs the pre-aggregated '{self.name}' column "
                f"on the panel (produced upstream by compute_jump_amount_corr and "
                f"joined by the runner); panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "JUMP_LOOKBACK_DAYS",
    "JUMP_MIN_PAIRS",
    "JUMP_Z",
    "JumpAmountCorrFactor",
    "compute_jump_amount_corr",
]
