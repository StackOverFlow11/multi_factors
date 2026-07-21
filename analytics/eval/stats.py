"""Statistics the eval sections reduce the IR with — pure, small, testable.

Only what the rest of ``analytics`` does not already provide:
``analytics/performance.py`` supplies annual return / max drawdown / volatility /
Sharpe from a nav, and ``analytics/factor.py`` supplies ``ic_summary``; this
module adds the pieces specific to judging a FACTOR rather than a portfolio.

The headline one is :func:`newey_west_t`. Design §A: "IC t 必须自相关校正
(NW/block)". An IC series is autocorrelated (today's cross-sectional ranking
overlaps heavily with yesterday's), and the textbook IID t divides by a standard
error that pretends the periods are independent — so it OVERSTATES significance,
sometimes by a lot. Both are reported side by side so the gap is visible instead
of implied.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) automatic bandwidth: ``floor(4 * (n/100)^(2/9))``.

    The usual default. Clamped to ``[0, n-1]``: a bandwidth at or beyond the
    sample length has no lagged pairs left to estimate.
    """
    if n < 2:
        return 0
    return max(0, min(int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), n - 1))


def newey_west_t(series: pd.Series, lags: int | None = None) -> dict[str, float]:
    """Autocorrelation-corrected t of the MEAN of ``series`` (e.g. an IC series).

    Uses the Bartlett-kernel HAC variance

        var = gamma_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) * gamma_j

    with ``gamma_j`` the sample autocovariance at lag j, and ``se = sqrt(var/n)``.
    The loop is over the ~4 BANDWIDTH LAGS, not over the periods.

    GAPS ARE HANDLED ON THE TIME GRID, NOT BY DROPPING FIRST. An IC series has
    holes: a degenerate cross-section here and there yields NaN. Dropping those
    and *then* lagging would pair observations that are j SURVIVING ROWS apart
    while calling them j PERIODS apart — silently bridging every hole and
    reporting an autocovariance that does not exist. Since the whole reason this
    statistic exists is that the naive IID t overstates significance on an
    autocorrelated series, a quietly wrong autocovariance here would defeat its
    purpose. Instead the series is kept on its own time grid and the deviations
    are ZERO-FILLED at the holes, so any lagged product touching a hole vanishes:
    the pair is DROPPED rather than bridged. Each ``gamma_j`` is divided by ``n``
    (the count of finite observations), the standard amplitude-modulation
    convention, which keeps the Bartlett kernel's variance non-negative.

    Returns
    -------
    ``{"t": <NW t>, "t_iid": <naive t>, "mean", "se", "se_iid", "lags", "n",
    "n_dropped"}``. Non-estimable pieces are NaN, never a fabricated 0.
    """
    raw = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan)
    values = raw.to_numpy(dtype=float)
    valid = np.isfinite(values)
    n = int(valid.sum())
    nan = float("nan")
    out = {
        "t": nan, "t_iid": nan, "mean": nan, "se": nan, "se_iid": nan,
        "lags": float(0), "n": float(n), "n_dropped": float(len(values) - n),
    }
    if n < 2:
        return out

    observed = values[valid]
    mean = float(observed.mean())
    out["mean"] = mean

    std = float(observed.std(ddof=1))
    if math.isfinite(std) and std > 0:
        out["se_iid"] = std / math.sqrt(n)
        out["t_iid"] = mean / out["se_iid"]

    bandwidth = newey_west_lag(n) if lags is None else max(0, min(int(lags), n - 1))
    out["lags"] = float(bandwidth)

    # Zero-filled deviations ON THE TIME GRID: dev[t] is 0 wherever the period is
    # missing, so dev[t] * dev[t-j] is 0 unless BOTH periods are observed -- a hole
    # breaks the lag pair instead of bridging it.
    deviations = np.where(valid, values - mean, 0.0)
    variance = float(deviations @ deviations) / n  # gamma_0
    for j in range(1, bandwidth + 1):
        gamma_j = float(deviations[j:] @ deviations[:-j]) / n
        variance += 2.0 * (1.0 - j / (bandwidth + 1.0)) * gamma_j
    # A Bartlett-kernel HAC variance is guaranteed non-negative, but it CAN come
    # out at (numerically) zero — a constant series, or one whose autocovariances
    # cancel. Report NaN rather than an infinite t.
    if not math.isfinite(variance) or variance <= 0:
        return out
    se = math.sqrt(variance / n)
    out["se"] = se
    out["t"] = mean / se
    return out


def effective_sample_size(
    series: pd.Series, lags: int | None = None
) -> dict[str, float | str]:
    """Effective sample size of an autocorrelated series: ``N / (1 + 2*sum_k rho_k)``.

    Gate part A of design §6 (v0.3). A RAW COUNT IS NOT A SAMPLE SIZE: consecutive
    daily IC observations overlap heavily, so 500 raw points can carry only a few
    dozen independent ones. This is block averaging — correlated MD frames are not
    independent samples either — applied to the IC series.

    LAG TRUNCATION: ``newey_west_lag(n)`` is the FLOOR, then the window is
    EXTENDED while rho-hat stays positive (Geyer's initial positive sequence),
    capped at ``n-1``.

        Why not the NW bandwidth alone, as the HAC t uses? Because it is the wrong
        tool for THIS quantity and fails in the PERMISSIVE direction. NW(1994)'s
        ``4*(n/100)^(2/9)`` is 5 lags at n=500 — fine for a HAC variance, but an
        AR(1) with rho=0.95 has an integrated autocorrelation time of
        (1+rho)/(1-rho) ~ 39, so truncating at 5 measures N_eff ~ 53 when the truth
        is ~13. A gate fed a 4x-too-large N_eff is not a gate. Sharing the
        bandwidth as a FLOOR keeps the two statistics from disagreeing about the
        autocorrelation structure in the direction that matters (the ESS window is
        never SHORTER than the NW-t's view of it, and both read the same rho-hats
        off the same grid with the same gap rule); it simply does not stop looking
        while the series is still visibly correlated. An explicit ``lags`` disables
        the extension — the caller chose the truncation.

    GAPS: identical convention to :func:`newey_west_t` — deviations are ZERO-FILLED
    on the time grid, so a lagged product touching a hole vanishes and the PAIR IS
    DROPPED rather than bridged. Never resampled, never dropped-then-lagged.

    GUARDS — the result is ALWAYS a finite, non-negative number, never NaN/inf/negative:
      * no finite observation -> 0.0 (an empty sample is 0 effective samples).
      * constant series (gamma_0 = 0) -> 1.0. It carries exactly one distinct
        value, i.e. one sample's worth of information: the perfectly-correlated
        limit, and the 0/0 the ratio would otherwise be.
      * ``1 + 2*sum_k rho_k <= 0`` (a net anti-correlated or merely noisy rho-hat)
        -> clamped to N, the same answer as the tiny-positive-denominator case it
        is the continuation of. The estimator is saying "at least as informative as
        i.i.d."; we credit exactly i.i.d. and no more.
      * ``N_eff`` is CLAMPED TO [1, N]. You cannot hold more independent
        observations than observations, and crediting anti-correlation with
        N_eff > N is precisely how a noisy rho-hat would buy evidence a run does
        not have. Clamping DOWN is the conservative direction for a gate that
        requires N_eff to be LARGE.

    Returns
    -------
    ``{"n_eff", "n", "lags", "lags_nw", "sum_rho", "denominator", "status"}``.
    ``status`` is "" unless a guard fired, and says which.
    """
    raw = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan)
    values = raw.to_numpy(dtype=float)
    valid = np.isfinite(values)
    n = int(valid.sum())
    nan = float("nan")
    out: dict[str, float | str] = {
        "n_eff": float(n), "n": float(n), "lags": 0.0, "lags_nw": 0.0,
        "sum_rho": nan, "denominator": nan, "status": "",
    }
    if n == 0:
        out["n_eff"] = 0.0
        out["status"] = "no finite observation: 0 effective samples."
        return out
    if n == 1:
        out["n_eff"] = 1.0
        out["status"] = "a single finite observation."
        return out

    mean = float(values[valid].mean())
    # Zero-filled deviations ON THE TIME GRID (see newey_west_t): a hole breaks
    # the lag pair instead of bridging it.
    deviations = np.where(valid, values - mean, 0.0)
    # All autocovariances in ONE vectorized pass (lags 0..len-1); the only loop
    # anywhere here is numpy's, and never over rebalance periods.
    gammas = np.correlate(deviations, deviations, mode="full")[len(deviations) - 1 :] / n
    gamma_0 = float(gammas[0])
    if not math.isfinite(gamma_0) or gamma_0 <= 0:
        out["n_eff"] = 1.0
        out["status"] = (
            "constant series (zero variance): one distinct value carries one "
            "sample's worth of information."
        )
        return out

    rho = gammas / gamma_0
    # Cap lags at n-1 exactly like newey_west_t does.
    max_lag = min(len(deviations) - 1, n - 1)
    bandwidth_nw = (
        newey_west_lag(n) if lags is None else max(0, min(int(lags), n - 1))
    )
    out["lags_nw"] = float(bandwidth_nw)
    window = bandwidth_nw
    if lags is None and bandwidth_nw < max_lag:
        # extend past the NW floor while the autocorrelation is still positive
        tail = rho[bandwidth_nw + 1 : max_lag + 1]
        non_positive = np.flatnonzero(tail <= 0)
        window += int(non_positive[0]) if non_positive.size else int(tail.size)
    out["lags"] = float(window)

    sum_rho = float(rho[1 : window + 1].sum()) if window >= 1 else 0.0
    out["sum_rho"] = sum_rho
    denominator = 1.0 + 2.0 * sum_rho
    out["denominator"] = denominator
    if not math.isfinite(denominator) or denominator <= 0:
        out["n_eff"] = float(n)
        out["status"] = (
            f"1 + 2*sum(rho) = {denominator:.4f} <= 0 (net anti-correlated or a "
            f"noisy rho-hat): clamped to N={n}. Anti-correlation is never credited "
            f"with more independent samples than observations."
        )
        return out

    n_eff = n / denominator
    clamped = min(max(n_eff, 1.0), float(n))
    out["n_eff"] = float(clamped)
    if clamped != n_eff:
        out["status"] = f"N_eff {n_eff:.2f} clamped into [1, {n}]."
    return out


#: Two-sided normal z-scores for the confidence levels the CI helpers support.
#: A small table rather than a dependency on scipy — the eval layer is pure
#: numpy/pandas, and these three levels cover every use.
_Z_TWO_SIDED: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}
#: The project's standing confidence level for the verdict CIs (design §6, v0.6).
DEFAULT_CONFIDENCE = 0.95


def _z_for(confidence: float) -> float:
    """The two-sided normal z for ``confidence``, or a readable error."""
    z = _Z_TWO_SIDED.get(round(float(confidence), 4))
    if z is None:
        raise ValueError(
            f"confidence must be one of {sorted(_Z_TWO_SIDED)} (a normal-approx CI "
            f"table, no scipy dependency); got {confidence!r}."
        )
    return z


def mean_ci(
    series: pd.Series, confidence: float = DEFAULT_CONFIDENCE
) -> dict[str, float]:
    """CI of the MEAN of an autocorrelated series, using N_eff (design §6, v0.6).

    ``SE(mean) = std / sqrt(N_eff)`` — the standard error DEFLATED by the effective
    sample size (:func:`effective_sample_size`), not the raw count, so an
    autocorrelated IC / spread series gets the WIDER interval its overlapping
    observations deserve. CI = mean +/- z * SE at the stated confidence.

    Returns ``{point, se, ci_low, ci_high, n_eff, z, confidence}``; every numeric
    piece is NaN when fewer than two finite observations exist or the std is zero.
    """
    z = _z_for(confidence)
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    nan = float("nan")
    out = {
        "point": nan, "se": nan, "ci_low": nan, "ci_high": nan,
        "n_eff": nan, "z": z, "confidence": float(confidence),
    }
    if len(clean) < 2:
        out["point"] = float(clean.mean()) if len(clean) else nan
        return out
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    n_eff = float(effective_sample_size(clean)["n_eff"])
    out["point"] = mean
    out["n_eff"] = n_eff
    if not math.isfinite(std) or std <= 0 or not math.isfinite(n_eff) or n_eff <= 0:
        return out
    se = std / math.sqrt(n_eff)
    out["se"] = se
    out["ci_low"] = mean - z * se
    out["ci_high"] = mean + z * se
    return out


def information_ratio_ci(
    series: pd.Series, confidence: float = DEFAULT_CONFIDENCE
) -> dict[str, float]:
    """CI of the INFORMATION RATIO (ICIR = mean/std) of an IC series, using N_eff.

    Design §6, v0.6. The ICIR is a Sharpe ratio of the IC series, so its standard
    error is the Lo (2002) iid-Sharpe SE

        SE(IR) = sqrt((1 + 0.5 * IR^2) / N_eff)

    with the sample count N replaced by the EFFECTIVE count N_eff
    (:func:`effective_sample_size`) to absorb the IC series' autocorrelation — the
    ``0.5 * IR^2`` term is the extra uncertainty from estimating the denominator
    (std) as well as the mean. CI = IR +/- z * SE. This is the LOWER bound the
    verdict gates on (:mod:`analytics.eval.verdict`).

    Returns ``{point, se, ci_low, ci_high, n_eff, z, confidence}``; NaN pieces when
    the IR itself is undefined (< 2 finite obs, or a zero-variance IC series).
    """
    z = _z_for(confidence)
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    nan = float("nan")
    out = {
        "point": nan, "se": nan, "ci_low": nan, "ci_high": nan,
        "n_eff": nan, "z": z, "confidence": float(confidence),
    }
    if len(clean) < 2:
        return out
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    n_eff = float(effective_sample_size(clean)["n_eff"])
    out["n_eff"] = n_eff
    if not math.isfinite(std) or std <= 0:
        return out  # a zero-variance IC series has no ICIR (and no CI)
    ir = mean / std
    out["point"] = ir
    if not math.isfinite(ir) or not math.isfinite(n_eff) or n_eff <= 0:
        return out
    se = math.sqrt((1.0 + 0.5 * ir * ir) / n_eff)
    out["se"] = se
    out["ci_low"] = ir - z * se
    out["ci_high"] = ir + z * se
    return out


def hypothesis_win_rate(series: pd.Series, expected_sign: int) -> float:
    """Share of finite periods whose value carries the EXPECTED sign.

    ``VERDICT_KEYS`` documents ``ic_win_rate`` as hypothesis-relative BY
    DEFINITION, so the sign is applied here and not by the verdict. Exactly zero
    is not the expected sign (it is not evidence for the hypothesis). NaN when no
    finite period exists — never a 0.0 that would read as "always wrong".
    """
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float((expected_sign * clean > 0).mean())


def half_life(autocorr: float) -> float:
    """Periods for a signal with lag-1 autocorrelation ``autocorr`` to halve.

    Assumes the AR(1) decay ``rho^k``: ``log(0.5) / log(rho)``. Defined only for
    ``0 < rho < 1``; a non-positive rho (the signal flips rather than decays) or a
    rho >= 1 (no decay) has no half-life and returns NaN rather than a number that
    would be read as one.
    """
    if not isinstance(autocorr, (int, float)) or not math.isfinite(autocorr):
        return float("nan")
    if not 0.0 < autocorr < 1.0:
        return float("nan")
    return float(math.log(0.5) / math.log(autocorr))


def sortino(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized Sortino: mean / downside deviation * sqrt(ppy).

    Downside deviation uses the returns BELOW zero, with the sum divided by the
    FULL sample size (the standard definition — a strategy that is rarely
    negative should be rewarded for it, not have its few bad periods averaged
    among themselves). NaN when there is no downside at all: the ratio is
    genuinely undefined, and +inf would render as a spectacular fake.
    """
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return float("nan")
    downside = clean.where(clean < 0, 0.0)
    dd = math.sqrt(float((downside**2).sum()) / len(clean))
    if not math.isfinite(dd) or dd <= 0:
        return float("nan")
    return float(clean.mean() / dd * math.sqrt(periods_per_year))


def spearman(x: pd.Series | list, y: pd.Series | list) -> float:
    """Spearman correlation of two short aligned vectors (NaN pairs dropped).

    Used for the bucket-index vs bucket-return monotonicity. RAW: the hypothesis
    is applied by the verdict, never here.
    """
    pair = (
        pd.DataFrame({"x": pd.Series(list(x), dtype=float), "y": pd.Series(list(y), dtype=float)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return float(pair["x"].corr(pair["y"], method="spearman"))


def spearman_series_by_date(
    quantile_returns: pd.DataFrame, min_buckets: int = 3
) -> pd.Series:
    """PER-DATE Spearman(bucket index, bucket return), one value per QUALIFYING date.

    The series :func:`spearman_by_date` averages, exposed in its own right (design
    §6, v0.9) so the verdict can put a DISPERSION estimate on that average. A bare
    cross-date mean of per-date rank correlations is heavily attenuated by daily
    noise — an empirically perfect quantile ladder scores ~0.05-0.11 in this
    project's real runs — so gating a direction claim on whether that point happens
    to land above 0.0 reads noise as evidence. With the series in hand the caller
    can hand it to :func:`mean_ci` and gate on the N_eff-based interval instead.

    A date is SKIPPED — ABSENT FROM THE SERIES, never a zero and never a NaN row —
    when it carries fewer than ``min_buckets`` finite bucket returns (too few points
    for a meaningful rank correlation), or when its surviving buckets are all tied
    (Spearman undefined there). NO date qualifying yields an EMPTY series, not a
    series of NaN: "no observation" and "an unknown observation" are different
    facts, and only the first is true here.

    Because skipped dates are absent rather than held as holes, the index is the
    SURVIVING date grid. Downstream N_eff (via :func:`mean_ci`) is therefore read
    off that compacted grid — the same convention :func:`mean_ci` already applies to
    the IC series, whose degenerate cross-sections it drops before measuring
    autocorrelation. Noted rather than hidden (design §11).

    RAW, like :func:`spearman`: the hypothesis sign is applied by the verdict.
    """
    dates: list[object] = []
    daily: list[float] = []
    if not (quantile_returns.empty or quantile_returns.shape[1] < min_buckets):
        index = [float(q) for q in quantile_returns.columns]
        for date, row in quantile_returns.iterrows():
            values = [float(v) for v in row.to_numpy(dtype=float)]
            if sum(1 for v in values if math.isfinite(v)) < min_buckets:
                continue
            # spearman() drops the non-finite pairs itself, keeping each surviving
            # bucket paired with its OWN index (not a renumbered 1..k).
            rho = spearman(index, values)
            if math.isfinite(rho):
                dates.append(date)
                daily.append(rho)
    name = quantile_returns.index.name if isinstance(quantile_returns, pd.DataFrame) else None
    return pd.Series(daily, index=pd.Index(dates, name=name), dtype=float)


def mean_by_date_spearman(daily: pd.Series) -> float:
    """The cross-date average :func:`spearman_by_date` reduces its series with.

    Exposed so a caller that already built the per-date series (the standard layer
    does, to attach a CI to it) gets the POINT from the very same expression rather
    than paying for a second pass over the panel — the point and the interval can
    then never disagree about which observations they describe.

    Deliberately the sequential Python ``sum(...) / len(...)`` of the pre-v0.9
    implementation, not ``Series.mean()``: the two can differ in the last bit, and
    this figure is a published, cross-run-comparable report field.
    """
    values = daily.tolist()
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def spearman_by_date(
    quantile_returns: pd.DataFrame, min_buckets: int = 3
) -> float:
    """Mean over dates of the PER-DATE Spearman(bucket index, bucket return).

    Structurally parallel to the rank IC (design §6, v0.8): each date's statistic
    is BOUNDED in [-1, 1] before the cross-date average, so a handful of extreme
    return days cannot dominate it. The pooled :func:`spearman` version correlates
    against cross-date ARITHMETIC MEANS, which are unbounded and magnitude
    sensitive — one outlier bucket-day can flip it while the daily-capped rank IC
    barely moves.

    As of v0.9 this is the mean of :func:`spearman_series_by_date` (see there for
    the skip rules) — a pure refactor: the returned float is BIT-IDENTICAL to the
    v0.8 implementation, which is asserted against the v0.8 fixtures.

    NaN when NO date qualifies: unknown, not 0.0.

    RAW, like :func:`spearman`: the hypothesis sign is applied by the verdict.
    """
    return mean_by_date_spearman(
        spearman_series_by_date(quantile_returns, min_buckets=min_buckets)
    )


def as_float(value: object) -> float:
    """Coerce to a plain float for a payload; anything unusable becomes NaN."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


__all__ = [
    "DEFAULT_CONFIDENCE",
    "as_float",
    "effective_sample_size",
    "half_life",
    "hypothesis_win_rate",
    "information_ratio_ci",
    "mean_by_date_spearman",
    "mean_ci",
    "newey_west_lag",
    "newey_west_t",
    "sortino",
    "spearman",
    "spearman_by_date",
    "spearman_series_by_date",
]
