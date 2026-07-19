"""``build_eval_ir``: the four VECTORIZED intermediates every metric reduces from.

Design ``tmp/design/factor_eval_contract_v0.1.md`` §8. The user's actual pain is
"分钟信号生成的因子的 ICIR 分层回测太慢", and the root cause is a per-rebalance
Python loop: for every date, build a frame, dropna, correlate, bucket, average.
This module computes the same four objects with **groupby/bincount passes over
the whole panel** — the number of Python-level iterations no longer grows with
the number of dates:

    1. ``factor``            F, MultiIndex(date, symbol)  — the processed input
    2. ``forward_returns``   R_h at the spec's horizon, same index
    3. ``ic``                Series[date]                 — ONE grouped reduction
    4. ``quantile_returns``  DataFrame[date, quantile]    — ONE grouped reduction

Everything a mandatory section needs is then a cheap reduction of these four
(plus the metadata carried alongside), which is what makes the report interface
stable while the engine underneath is free to change (design §10: 先钉接口,后换
引擎).

PIT BOUNDARY (invariant #1). Forward returns are computed HERE, at the analytics
boundary — ``analytics/factor.py`` is the only place allowed to look into the
future, and this module reuses it rather than re-deriving it. ``factors/`` never
sees an ``R_h``, and this module is downstream of ``factors`` (it imports
``factors.spec``, never the reverse).

TWO CONVENTIONS WORTH KNOWING BEFORE READING A REPORT
    * **One row of F = one evaluation period.** The IR does NOT resample: it
      evaluates the grid it is handed. But ``EvalConfig.rebalance`` is not merely
      a label either — a daily panel declared "monthly" would report ~250 settled
      rebalances instead of ~12 and walk straight through
      ``VerdictThresholds.min_rebalances``, the gate whose entire job is deciding
      INSUFFICIENT-DATA. So :func:`_check_rebalance_grid` compares the observed
      spacing against the declared frequency and **raises** on a mismatch (it does
      not resample — see that function for why).
      One row = one period is also why ``FactorSpec.forward_return_horizon``
      (documented as "the horizon IN EVALUATION PERIODS") is resolved on F's OWN
      grid: a monthly factor over a daily price panel is scored on its
      next-PERIOD return, not its next-TRADING-DAY return. See
      :func:`_resolve_forward_returns`.
    * **Buckets are formed from the FACTOR cross-section alone**, never from the
      (factor, forward-return) intersection. Dropping a name because its future
      return is missing would let tomorrow's data decide today's bucket. The
      per-bucket missing-return rate is disclosed instead. This is a deliberate
      difference from ``analytics.factor.quantile_returns`` (which buckets the
      intersection); see :func:`assign_quantile_buckets`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.eval.config import EvalConfig
from analytics.factor import forward_returns as _forward_returns
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.spec import FactorSpec

#: Staleness lags (in evaluation periods) the IC-decay / factor-autocorrelation
#: curves are evaluated at. A loop over THESE is a loop over 4 constants — each
#: iteration is still one vectorized pass over the whole panel — which is not the
#: per-period loop design §8 is about.
DECAY_LAGS: tuple[int, ...] = (1, 2, 3, 5)

#: Fallback annualization when ``EvalConfig.rebalance`` is not a label we know.
#: Disclosed in the report rather than silently assumed.
PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
    "yearly": 1,
}
DEFAULT_PERIODS_PER_YEAR = 252

#: Expected spacing, in CALENDAR days, between consecutive evaluation periods for
#: each recognized rebalance label. A trading-day grid has a median gap of 1.0
#: (weekends sit in the tail, not the median); a month averages 365.25/12.
REBALANCE_SPACING_DAYS: dict[str, float] = {
    "daily": 1.0,
    "weekly": 7.0,
    "biweekly": 14.0,
    "monthly": 365.25 / 12.0,
    "quarterly": 365.25 / 4.0,
    "annual": 365.25,
    "yearly": 365.25,
}

#: A declared grid is accepted when the observed median spacing lands inside
#: ``[expected / TOL, expected * TOL]``. Deliberately COARSE: this exists to catch
#: the ORDER-OF-MAGNITUDE lie (a daily panel declared "monthly" is off by ~30x),
#: not a fine distinction (weekly vs biweekly is 2x and passes). The gate it
#: protects — ``min_rebalances`` — only cares about order of magnitude.
REBALANCE_SPACING_TOLERANCE = 2.0


@dataclass(frozen=True)
class EvalContext:
    """Everything :func:`build_eval_ir` needs BEYOND the factor panel.

    The evaluator contract leaves ``ctx`` to PR-B (``evaluator.py``: "``ctx``
    carries whatever a section needs beyond the factor panel; PR-B defines its
    shape"). This is that shape. Every field is optional, and an absent field
    makes the section that needed it say so — it never makes a section invent a
    number.

    Attributes
    ----------
    price_panel : the CANONICAL market panel (MultiIndex(date, symbol),
        front-adjusted per ``FactorSpec.price_adjust``), used to derive ``R_h`` for
        a ``close_to_close`` factor. Only ``close`` is read, but the whole core
        column set is required: ``R_h`` comes from
        ``analytics.factor.forward_returns``, which validates the panel through
        the shared schema. Reusing the project's ONE forward-return boundary is
        worth carrying its contract.
    forward_returns : a PRE-COMPUTED ``R_h``, for bases this module cannot derive
        itself (``exec_to_exec``: the holding period runs exec(T) -> exec(T_next)
        and only the minute tail machinery knows the execution anchors).
    known_factors : MultiIndex(date, symbol) frame of ALREADY-PROCESSED anchor
        factors for the purity section. Absent -> purity is Skipped, never faked.
    universe_symbols : the universe the factor was supposed to cover, so
        data_coverage can report what was DROPPED instead of guessing.
    fee_rate : the base one-way fee rate the cost scenarios multiply.
        ``EvalConfig.cost_scenarios`` are MULTIPLIERS and the contract carries no
        base rate, so it lives here. 0.001 is the project's standing base (I5d).
    execution_capacity : execution facts MEASURED ELSEWHERE (I5b fill feasibility
        / I5f capacity) — this evaluator does not run the backtest engine, so it
        reports what it is handed and Skips when handed nothing. Recognized keys:
        ``tradable`` (bool), ``capacity_sufficient`` (bool), plus any diagnostics
        to render. The provenance is stamped into the section: these numbers are
        never produced here.
    """

    price_panel: pd.DataFrame | None = None
    forward_returns: pd.Series | None = None
    known_factors: pd.DataFrame | None = None
    universe_symbols: tuple[str, ...] = ()
    fee_rate: float = 0.001
    execution_capacity: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        rate = self.fee_rate
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
            raise ValueError(
                f"EvalContext.fee_rate must be a non-negative number (the base "
                f"one-way fee the EvalConfig.cost_scenarios multiply); got {rate!r}."
            )
        object.__setattr__(self, "universe_symbols", tuple(self.universe_symbols))


@dataclass(frozen=True)
class StandardEvalIR:
    """The design §8 four objects + the metadata the sections and verdict need.

    Satisfies the frozen ``analytics.eval.evaluator.EvalIR`` Protocol (``factor``
    / ``forward_returns`` / ``ic`` / ``quantile_returns``) and carries the rest as
    plain, already-reduced facts so no section has to touch the panel again.
    """

    # -- the Protocol's four ------------------------------------------------
    factor: pd.Series
    forward_returns: pd.Series
    ic: pd.Series
    quantile_returns: pd.DataFrame

    # -- provenance ---------------------------------------------------------
    spec: FactorSpec
    cfg: EvalConfig
    ctx: EvalContext

    # -- derived, computed once --------------------------------------------
    #: linear (Pearson) IC — secondary to the rank IC (design §A).
    ic_pearson: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: 1..n_quantiles bucket label per (date, symbol) with a valid factor value.
    quantile_labels: pd.Series = field(default_factory=lambda: pd.Series(dtype="int64"))
    #: per-(date, bucket) turnover, sum_i |w_i(t) - w_i(t-1)|. Computed with the
    #: rest of the IR so return_risk and stability_cost reduce it instead of each
    #: rebuilding it (turnover x fee_rate = the period's cost).
    quantile_turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: per-date count of (factor, forward-return) pairs the IC actually used.
    cross_section_size: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: per-date count of non-NaN factor values (the bucketed cross-section).
    factor_count: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: date -> the date the period's forward return is REALIZED on (t+h), NaT at
    #: the tail. The OOS split reads THIS, never the signal date (P3-3's fix).
    realized_date: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    #: sorted unique dates of F (one per evaluation period).
    dates: pd.Index = field(default_factory=lambda: pd.Index([], name=DATE_LEVEL))
    #: how R_h was obtained — disclosed, never guessed.
    forward_return_source: str = ""
    #: observed median spacing between evaluation periods, in calendar days.
    median_period_gap_days: float = float("nan")
    #: outcome of checking the supplied grid against ``cfg.rebalance`` (a mismatch
    #: raises in the builder; this records OK / NOT CHECKED and why).
    rebalance_grid_check: str = ""
    #: rows within a symbol's declared ``min_history_bars`` warm-up.
    warmup_mask: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))

    @property
    def settled_rebalances(self) -> int:
        """Periods that produced usable evidence = a FINITE IC.

        A period with no realized forward return (the last h rows) or with a
        degenerate cross-section yields NaN and is NOT counted: the verdict's
        sample-size gate must not be inflated by periods that never settled.
        """
        return int(self.ic.notna().sum())

    @property
    def n_rebalances(self) -> int:
        """Evaluation periods on the supplied grid (settled or not)."""
        return int(len(self.dates))

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR.get(self.cfg.rebalance.strip().lower(), DEFAULT_PERIODS_PER_YEAR)

    @property
    def periods_per_year_is_default(self) -> bool:
        """True when ``rebalance`` was not a recognized label (-> disclose it)."""
        return self.cfg.rebalance.strip().lower() not in PERIODS_PER_YEAR


# -- vectorized primitives -------------------------------------------------


def _unique_dates(index: pd.Index) -> pd.Index:
    return pd.Index(
        pd.unique(index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    ).sort_values()


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a NaN or an infinity, like ``analytics.factor`` does.

    +/-inf is treated as missing rather than as an extreme value: a polluted
    return would otherwise silently dominate a correlation or a bucket mean.
    """
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def cross_section_corr(
    left: pd.Series,
    right: pd.Series,
    *,
    rank: bool = True,
    dates: pd.Index | None = None,
) -> pd.Series:
    """Per-date cross-sectional correlation of two aligned panels — NO date loop.

    Semantically identical to calling ``analytics.factor.compute_ic`` (drop
    non-finite pairs within the date; NaN when fewer than 2 pairs survive or
    either side has zero variance; Spearman = Pearson of the average-tie ranks),
    but computed as a handful of whole-panel passes: ONE grouped rank, then
    ``np.bincount`` reductions over the date codes.

    Numerically it demeans first (two-pass) instead of using the
    sum-of-squares shortcut, so a large-mean cross-section cannot lose the
    correlation to catastrophic cancellation.

    Parameters
    ----------
    left, right : aligned MultiIndex(date, symbol) Series.
    rank : True -> rank IC (Spearman, the primary; design §A); False -> Pearson.
    dates : the full date index to report on. Dates with no usable pair still
        appear, as NaN — an absent period must not silently vanish from the
        series the ICIR and the Newey-West t are computed over.

    Returns
    -------
    Series indexed by ``dates`` (or the union index's dates), one value per date.
    """
    frame = pd.DataFrame({"l": left, "r": right})
    all_dates = _unique_dates(frame.index) if dates is None else dates
    empty = pd.Series(np.nan, index=all_dates, dtype=float, name="ic")
    valid = _finite(frame)
    if valid.empty:
        return empty

    if rank:
        by_date = valid.groupby(level=DATE_LEVEL, sort=False)
        x = by_date["l"].rank().to_numpy(dtype=float)
        y = by_date["r"].rank().to_numpy(dtype=float)
    else:
        x = valid["l"].to_numpy(dtype=float)
        y = valid["r"].to_numpy(dtype=float)

    codes, uniques = pd.factorize(valid.index.get_level_values(DATE_LEVEL), sort=True)
    k = len(uniques)
    n = np.bincount(codes, minlength=k).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.bincount(codes, weights=x, minlength=k) / n
        my = np.bincount(codes, weights=y, minlength=k) / n
        xc = x - mx[codes]
        yc = y - my[codes]
        sxy = np.bincount(codes, weights=xc * yc, minlength=k)
        sxx = np.bincount(codes, weights=xc * xc, minlength=k)
        syy = np.bincount(codes, weights=yc * yc, minlength=k)
        den = np.sqrt(sxx * syy)
        # den > 0 already implies n >= 2 (a single point demeans to zero), but the
        # count check is kept explicit: it is the documented contract, not a
        # side effect of the algebra.
        usable = (den > 0) & (n >= 2)
        corr = np.where(usable, sxy / np.where(usable, den, 1.0), np.nan)
    out = pd.Series(
        np.clip(corr, -1.0, 1.0), index=pd.Index(uniques, name=DATE_LEVEL), name="ic"
    )
    return out.reindex(all_dates)


def assign_quantile_buckets(factor: pd.Series, n_quantiles: int) -> pd.Series:
    """EQUAL-COUNT rank buckets (1 = lowest factor) per date — NO date loop.

    Reproduces the project's canonical rule
    (:func:`qt.intraday_groups.assign_quantile_buckets`, the I5d/I5e grouped
    backtest) exactly, with the per-date ``sorted() + np.array_split`` replaced by
    integer arithmetic on grouped ranks:

      * names are ordered by ``(factor ascending, symbol ascending)`` — the
        ``rank(method="first")`` tie-break IS the symbol order because
        :func:`build_eval_ir` sorts the panel first — and split BY POSITION, so
        ties and degenerate cross-sections still assign deterministically where a
        value cut could not;
      * chunk sizes differ by at most one and the extra names go to the LOW
        buckets (the ``np.array_split`` convention);
      * fewer names than buckets simply leaves the high buckets empty.

    ⚠️ This deliberately differs from ``analytics.factor.quantile_returns``, which
    ``pd.qcut``s the ranks (equivalent when ``n_quantiles`` divides the
    cross-section, off by a name or two otherwise) AND buckets the (factor,
    forward-return) intersection. Buckets here are formed from the FACTOR
    cross-section ALONE: a name whose future return is missing must not be able to
    change which bucket its neighbours land in.
    """
    valid = _finite(factor.to_frame("f"))["f"]
    if valid.empty:
        return pd.Series([], index=factor.index[:0], dtype="int64", name="quantile")

    by_date = valid.groupby(level=DATE_LEVEL, sort=False)
    ranks = by_date.rank(method="first").to_numpy(dtype=np.int64)  # 1..n within date
    sizes = by_date.transform("size").to_numpy(dtype=np.int64)

    # np.array_split(n, q): the first (n % q) chunks hold (n // q + 1) names, the
    # rest hold (n // q). Invert that to rank -> chunk without materializing it.
    base = sizes // n_quantiles
    remainder = sizes % n_quantiles
    head = remainder * (base + 1)  # ranks 1..head live in the bigger chunks
    in_head = ranks <= head
    safe_base = np.where(base > 0, base, 1)  # base == 0 <=> every rank is in head
    labels = np.where(
        in_head,
        (ranks - 1) // (base + 1) + 1,
        remainder + (ranks - 1 - head) // safe_base + 1,
    )
    return pd.Series(
        np.clip(labels, 1, n_quantiles), index=valid.index, name="quantile"
    )


def quantile_return_matrix(
    fwd: pd.Series, labels: pd.Series, n_quantiles: int, dates: pd.Index
) -> pd.DataFrame:
    """Mean forward return per (date, bucket) — ONE ``groupby([date, q]).mean()``.

    Returns a ``dates x 1..n_quantiles`` frame. A bucket with no realized return
    on a date is NaN (an empty bucket, or every member's future missing) — never
    a silent zero.
    """
    columns = pd.Index(range(1, n_quantiles + 1), name="quantile")
    pair = _finite(pd.DataFrame({"r": fwd.reindex(labels.index), "q": labels}))
    if pair.empty:
        return pd.DataFrame(np.nan, index=dates, columns=columns)
    grouped = pair.groupby(
        [pair.index.get_level_values(DATE_LEVEL), pair["q"].astype("int64")],
        sort=True,
    )["r"].mean()
    out = grouped.unstack(level=-1)
    out.index.name = DATE_LEVEL
    return out.reindex(index=dates, columns=columns)


def bucket_membership_weights(labels_wide: pd.DataFrame, bucket: int) -> pd.DataFrame:
    """Equal weights of ONE bucket as a date x symbol frame (0.0 outside it).

    ``labels_wide`` is the bucket-label frame already unstacked ONCE (unstacking
    per bucket would rebuild a 6M-cell frame five times over at all-A scale). A
    missing label compares False, which is what we want: not a member.
    """
    member = (labels_wide == bucket).astype(float)
    counts = member.sum(axis=1)
    return member.div(counts.where(counts > 0), axis=0).fillna(0.0)


def quantile_turnover(
    labels: pd.Series, n_quantiles: int, dates: pd.Index
) -> pd.DataFrame:
    """Per-(date, bucket) turnover ``sum_i |w_i(t) - w_i(t-1)|`` — NO date loop.

    Matches the project's turnover convention, where ``turnover x fee_rate`` is
    the period's cost (the phase-2 baseline reports turnover 1.0818 -> cost
    0.108%/period at fee_rate 0.001), i.e. it counts BOTH sides of the trade.

    The FIRST period's turnover is the cost of establishing the book
    (``sum_i |w_i(0)|`` = 1.0 for a non-empty bucket), not NaN: entering the
    position is a real trade and must be charged for.

    The loop is over the ``n_quantiles`` buckets — a handful of whole-panel
    vectorized passes, not one pass per date.
    """
    columns = pd.Index(range(1, n_quantiles + 1), name="quantile")
    if labels.empty:
        return pd.DataFrame(0.0, index=dates, columns=columns)
    labels_wide = labels.unstack(level=SYMBOL_LEVEL).reindex(index=dates)
    out: dict[int, pd.Series] = {}
    for bucket in columns:
        weights = bucket_membership_weights(labels_wide, bucket)
        delta = weights.diff()
        if len(delta):
            delta.iloc[0] = weights.iloc[0]
        out[bucket] = delta.abs().sum(axis=1)
    frame = pd.DataFrame(out, index=dates)
    frame.columns = columns
    frame.index.name = DATE_LEVEL
    return frame


# -- the builder -----------------------------------------------------------


def _as_factor_series(
    factor_panel: pd.Series | pd.DataFrame, spec: FactorSpec
) -> pd.Series:
    """Coerce the supplied panel to ONE float factor Series, or say why not."""
    if isinstance(factor_panel, pd.DataFrame):
        if spec.factor_id in factor_panel.columns:
            series = factor_panel[spec.factor_id]
        elif factor_panel.shape[1] == 1:
            series = factor_panel.iloc[:, 0]
        else:
            raise ValueError(
                f"build_eval_ir: the factor panel has columns "
                f"{list(factor_panel.columns)!r} and none is named "
                f"{spec.factor_id!r}. Pass the factor's own column — guessing which "
                f"column IS the factor would silently evaluate the wrong one."
            )
    elif isinstance(factor_panel, pd.Series):
        series = factor_panel
    else:
        raise TypeError(
            f"build_eval_ir: factor_panel must be a Series or a DataFrame; got "
            f"{type(factor_panel).__name__}."
        )
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError(
            f"build_eval_ir: the factor panel must carry a MultiIndex"
            f"({DATE_LEVEL}, {SYMBOL_LEVEL}); got index names "
            f"{list(series.index.names)!r}."
        )
    if list(series.index.names) != [DATE_LEVEL, SYMBOL_LEVEL]:
        raise ValueError(
            f"build_eval_ir: the factor panel index levels must be named "
            f"({DATE_LEVEL!r}, {SYMBOL_LEVEL!r}); got {list(series.index.names)!r}."
        )
    if series.index.has_duplicates:
        raise ValueError(
            "build_eval_ir: the factor panel has duplicate (date, symbol) rows; "
            "one of them would silently win every alignment."
        )
    # Sorting is load-bearing, not cosmetic: it makes the bucket tie-break
    # (rank method='first') mean "symbol ascending", which is what the project's
    # canonical assign_quantile_buckets does.
    return series.astype(float).sort_index().rename(spec.factor_id)


def _resolve_forward_returns(
    spec: FactorSpec, ctx: EvalContext, dates: pd.Index
) -> tuple[pd.Series, str]:
    """``R_h`` + how it was obtained (never guessed).

    ``h`` IS A HORIZON IN EVALUATION PERIODS — that is what the frozen
    ``FactorSpec.forward_return_horizon`` says it is ("the horizon (in evaluation
    periods) the factor claims to predict"), and one row of F is one evaluation
    period. So the shift must happen on the FACTOR's grid. Handing the whole price
    panel to ``forward_returns(periods=(h,))`` would shift h rows of the PRICE
    panel instead: identical when the two grids coincide (the common case), but a
    monthly factor over a daily price panel would silently be scored on its
    next-TRADING-DAY return rather than its next-PERIOD return. The panel is
    therefore restricted to F's own dates first.
    """
    horizon = spec.forward_return_horizon
    if ctx.forward_returns is not None:
        fwd = pd.Series(ctx.forward_returns, dtype=float)
        if not isinstance(fwd.index, pd.MultiIndex) or fwd.index.nlevels != 2:
            raise ValueError(
                "build_eval_ir: EvalContext.forward_returns must carry a "
                f"MultiIndex({DATE_LEVEL}, {SYMBOL_LEVEL})."
            )
        source = (
            f"supplied via EvalContext.forward_returns (basis={spec.return_basis!r}, "
            f"h={horizon} evaluation periods); computed OUTSIDE this evaluator"
        )
        return fwd, source
    if spec.return_basis != "close_to_close":
        raise ValueError(
            f"build_eval_ir: FactorSpec.return_basis={spec.return_basis!r} cannot be "
            f"derived from a close panel — its holding period is execution-anchored "
            f"(exec(T) -> exec(T_next)) and only the minute-tail machinery knows the "
            f"execution anchors. Supply EvalContext.forward_returns computed there."
        )
    if ctx.price_panel is None:
        raise ValueError(
            "build_eval_ir: no forward returns available. Supply either "
            "EvalContext.price_panel (a close panel, for a close_to_close factor) "
            "or EvalContext.forward_returns. Forward returns are computed at the "
            "analytics boundary and never handed to the factor layer."
        )
    panel_dates = _unique_dates(ctx.price_panel.index)
    absent = dates.difference(panel_dates)
    if len(absent):
        raise ValueError(
            f"build_eval_ir: {len(absent)} factor date(s) are absent from the price "
            f"panel (e.g. {[str(d) for d in absent[:3]]}), so their forward return "
            f"cannot be measured at all. Align the factor panel to the price panel "
            f"before evaluating."
        )
    if dates.equals(panel_dates):
        on_grid = ctx.price_panel  # the common case: no copy of a 6M-row panel
        restriction = ""
    else:
        on_grid = ctx.price_panel[
            ctx.price_panel.index.get_level_values(DATE_LEVEL).isin(dates)
        ]
        restriction = (
            f" (price panel restricted from {len(panel_dates)} to the factor's own "
            f"{len(dates)} evaluation dates FIRST, so h counts evaluation periods "
            f"and not price-panel rows)"
        )
    column = f"forward_return_{horizon}d"
    fwd = _forward_returns(on_grid, periods=(horizon,))[column]
    source = (
        f"analytics.factor.forward_returns(price_panel, periods=({horizon},)) — "
        f"close[t+{horizon} evaluation periods]/close[t] - 1{restriction}"
    )
    return fwd.astype(float), source


def _realized_dates(dates: pd.Index, horizon: int) -> pd.Series:
    """date -> the date its forward return is realized on: ``h`` PERIODS later.

    Positional on F's own grid, because h is a horizon in evaluation periods and
    one row of F is one period.

    Every OOS split in this project slices by the REALIZED date, never the signal
    date: a period signalled at t is only out-of-sample once t+h has happened
    (P3-3 fixed exactly this bug, and the fix moved the numbers materially).
    """
    ahead = np.arange(len(dates)) + horizon
    within = ahead < len(dates)
    realized = np.full(len(dates), pd.NaT, dtype=object)
    realized[within] = dates[ahead[within]]
    return pd.Series(realized, index=dates, name="realized_date")


def _check_rebalance_grid(cfg: EvalConfig, dates: pd.Index, median_gap: float) -> str:
    """Reject a factor grid that contradicts the DECLARED rebalance frequency.

    Returns a disclosure string when the grid is consistent (or unverifiable);
    raises when it is provably not.

    WHY THIS EXISTS. One row of F is one evaluation period and the IR does not
    resample, so a DAILY panel declared ``rebalance: monthly`` reports ~250 settled
    rebalances instead of ~12 — and sails past ``VerdictThresholds.min_rebalances``
    (default 24), the gate whose ENTIRE JOB is deciding INSUFFICIENT-DATA. Merely
    disclosing that is not enough: a sample-adequacy gate that can be satisfied by a
    sample which does not exist at the declared frequency is not a gate, and every
    annualized number in the report is scaled by that same declared frequency.

    WHY REJECT RATHER THAN RESAMPLE — the alternative was to snap F onto the
    declared grid ourselves. Rejected, deliberately:

      1. The project already has exactly ONE monthly convention (last trading day of
         the month, ``runtime.backtest.events.monthly_rebalance_dates``, which calls
         itself "the single source of truth for which dates"). Resampling here would
         either duplicate it — a second, silently divergent calendar, the very
         problem the two coexisting bucketing rules already cause — or extend it to
         "weekly"/"quarterly", for which the project has NO convention at all and the
         eval layer would simply be inventing one.
      2. It would make the EVALUATOR choose which ~12 of ~250 dates constitute the
         sample, and that choice moves every number in the report. The caller knows
         their intent; a heuristic here would silently overrule it (month-end vs
         month-start alone flips the answer).
      3. It would silently discard ~95% of the supplied rows.
      4. A rejection cannot quietly produce a wrong number. A wrong resample can.

    So: the caller resamples to the grid they mean, or declares the frequency they
    actually supplied. The contract ("one row = one period") is ENFORCED here rather
    than papered over.
    """
    label = cfg.rebalance.strip().lower()
    expected = REBALANCE_SPACING_DAYS.get(label)
    if expected is None:
        # Not a recognized label (e.g. a minute-frequency study): there is no
        # expected spacing to compare against. Say so — an unverified declaration
        # must not read as a verified one.
        return (
            f"NOT CHECKED — rebalance={cfg.rebalance!r} is not one of "
            f"{tuple(sorted(REBALANCE_SPACING_DAYS))}, so the evaluator has no "
            f"expected spacing to check it against. The declared frequency is "
            f"UNVERIFIED: settled_rebalances counts the rows supplied, and "
            f"min_rebalances gates on that count."
        )
    if not math.isfinite(median_gap):
        return (
            f"NOT CHECKED — fewer than 2 evaluation periods, so no spacing is "
            f"observable (declared {label!r})."
        )
    low = expected / REBALANCE_SPACING_TOLERANCE
    high = expected * REBALANCE_SPACING_TOLERANCE
    if not low <= median_gap <= high:
        raise ValueError(
            f"build_eval_ir: EvalConfig.rebalance={cfg.rebalance!r} declares "
            f"evaluation periods ~{expected:.2f} calendar days apart, but the "
            f"supplied factor panel's {len(dates)} periods are a median "
            f"{median_gap:.2f} days apart (accepted band {low:.2f}..{high:.2f} days).\n"
            f"\n"
            f"ONE ROW OF THE FACTOR PANEL IS ONE EVALUATION PERIOD — this evaluator "
            f"does not resample. It would therefore report {len(dates)} settled "
            f"rebalances for a grid you called {label!r}, and "
            f"VerdictThresholds.min_rebalances (the gate that decides "
            f"INSUFFICIENT-DATA) would pass on a sample size that does not exist at "
            f"the declared frequency. Every annualized number would be scaled by the "
            f"declared frequency too.\n"
            f"\n"
            f"Fix: resample the factor panel to the grid you actually mean to "
            f"evaluate BEFORE calling the evaluator (the project's monthly "
            f"convention is the last trading day of each month — "
            f"runtime.backtest.events.monthly_rebalance_dates), or declare the "
            f"frequency you actually supplied."
        )
    return (
        f"OK — declared {label!r} (~{expected:.2f} calendar days between periods); "
        f"observed median spacing {median_gap:.2f} days over {len(dates)} periods."
    )


def _median_gap_days(dates: pd.Index) -> float:
    if len(dates) < 2 or not isinstance(dates, pd.DatetimeIndex):
        return float("nan")
    gaps = pd.Series(dates).diff().dropna()
    if gaps.empty:
        return float("nan")
    return float(gaps.dt.total_seconds().median() / 86400.0)


def build_eval_ir(
    factor_panel: pd.Series | pd.DataFrame,
    spec: FactorSpec,
    cfg: EvalConfig,
    ctx: EvalContext | None = None,
) -> StandardEvalIR:
    """Compute the design §8 four objects ONCE, vectorized (no per-period loop).

    Parameters
    ----------
    factor_panel : the PROCESSED factor, MultiIndex(date, symbol). A DataFrame is
        accepted when the factor's own column can be identified unambiguously.
    spec, cfg : the frozen provenance contract (horizon, hypothesis, quantiles).
    ctx : :class:`EvalContext` — the price panel / pre-computed returns / anchors.

    Returns
    -------
    :class:`StandardEvalIR`, satisfying the frozen ``EvalIR`` Protocol.
    """
    if not isinstance(spec, FactorSpec):
        raise TypeError(f"build_eval_ir needs a FactorSpec; got {type(spec).__name__}.")
    if not isinstance(cfg, EvalConfig):
        raise TypeError(f"build_eval_ir needs an EvalConfig; got {type(cfg).__name__}.")
    context = ctx if ctx is not None else EvalContext()
    if not isinstance(context, EvalContext):
        raise TypeError(
            f"build_eval_ir: ctx must be an EvalContext; got {type(context).__name__}."
        )

    factor = _as_factor_series(factor_panel, spec)
    dates = _unique_dates(factor.index)
    # Fail fast, BEFORE any expensive work: a grid that contradicts the declared
    # rebalance frequency makes the sample-adequacy gate meaningless downstream.
    median_gap = _median_gap_days(dates)
    grid_check = _check_rebalance_grid(cfg, dates, median_gap)
    fwd_full, fwd_source = _resolve_forward_returns(spec, context, dates)
    # R_h is aligned ONTO F: the IR's two panels always share one index, so every
    # downstream reduction is a plain column-wise operation.
    fwd = fwd_full.reindex(factor.index).astype(float).rename("forward_return")

    ic = cross_section_corr(factor, fwd, rank=True, dates=dates)
    ic_pearson = cross_section_corr(factor, fwd, rank=False, dates=dates)
    labels = assign_quantile_buckets(factor, cfg.n_quantiles)
    quantiles = quantile_return_matrix(fwd, labels, cfg.n_quantiles, dates)

    usable_pair = _finite(pd.DataFrame({"f": factor, "r": fwd}))
    cross_section_size = (
        usable_pair.groupby(level=DATE_LEVEL, sort=True).size().reindex(dates).fillna(0)
    )
    factor_count = (
        _finite(factor.to_frame("f"))
        .groupby(level=DATE_LEVEL, sort=True)
        .size()
        .reindex(dates)
        .fillna(0)
    )
    warmup_mask = (
        factor.groupby(level=SYMBOL_LEVEL, sort=False).cumcount() < spec.min_history_bars
    )

    return StandardEvalIR(
        factor=factor,
        forward_returns=fwd,
        ic=ic,
        quantile_returns=quantiles,
        spec=spec,
        cfg=cfg,
        ctx=context,
        ic_pearson=ic_pearson,
        quantile_labels=labels,
        quantile_turnover=quantile_turnover(labels, cfg.n_quantiles, dates),
        cross_section_size=cross_section_size.astype(float),
        factor_count=factor_count.astype(float),
        realized_date=_realized_dates(dates, spec.forward_return_horizon),
        dates=dates,
        forward_return_source=fwd_source,
        median_period_gap_days=median_gap,
        rebalance_grid_check=grid_check,
        warmup_mask=warmup_mask.rename("warmup"),
    )


__all__ = [
    "DECAY_LAGS",
    "DEFAULT_PERIODS_PER_YEAR",
    "PERIODS_PER_YEAR",
    "REBALANCE_SPACING_DAYS",
    "REBALANCE_SPACING_TOLERANCE",
    "EvalContext",
    "StandardEvalIR",
    "assign_quantile_buckets",
    "bucket_membership_weights",
    "build_eval_ir",
    "cross_section_corr",
    "quantile_return_matrix",
    "quantile_turnover",
]
