"""D2 stratified hand-computed anchors (R12): plain arithmetic vs the new engine.

Every migrated factor gets >= 5 anchor rows recomputed BY HAND from the raw
cache parquets — stdlib + numpy + pandas arithmetic only. THIS MODULE MUST
NEVER IMPORT ``factors.*`` OR ``data.clean.*`` (the #79 rule: a hand check
that imports the engine inherits the engine's bugs); a runtime guard at the
bottom of the module raises if any such module is loaded. The engine side of
each comparison is read FROM DISK (the ``panels_d2`` parquet files the D2
reconciliation wrote) for the 14 frozen factors; the four ops-rewritten daily
factors without a frozen panel (momentum/reversal/liquidity/overnight_mom) are
compared by the companion ``qt.hand_anchors_engine_values`` (which may import
the engine — the INDEPENDENCE requirement binds the hand computation, not the
comparer).

Stratified sampling (R12): per factor, one row from each APPLICABLE boundary
class — (a) the warm-up END (the symbol's first finite value; verified by hand
to sit exactly at the accumulation floor), (b) a GUARD boundary row (a day
whose gated count sits exactly at its ``min_*`` floor where the data attains
it; otherwise the nearest attainable is used and DISCLOSED), (c) an EX-DATE
row for price-dependent factors (the trailing window spans a true
``af(t) != af(t_prev)`` event) — plus >= 2 seeded interior rows. Uniform
sampling would hit a boundary row with ~0.4% probability; boundaries are where
migration bugs live.

Hand reimplementation notes (all plain arithmetic, no engine helpers):
* PIT visibility: the runners normalize with ``data_lag='1min'``, so a bar is
  visible iff ``bar_end + 1min <= trade_date + 14:50``.
* peak taxonomy: same-slot strictly-prior baselines via EXPLICIT position
  windows (numpy mean/std ddof=1 over the trailing 20 day-rows, >= 10 obs,
  excluding the day itself); eruptive = vol > mu + sigma; a peak needs both
  60s-adjacent neighbours mild; ridge = eruptive & ~peak; valley = mild.
* qfq for the daily factors: raw close x af / af(symbol's last panel day) —
  the ``front_adjust`` anchor convention, re-derived here from its definition.

Run AFTER ``python -m qt.panel_reconcile`` produced ``panels_d2``:

    python -m qt.hand_anchors_d2            # select + hand-compute + compare (14)
    python -m qt.hand_anchors_engine_values # compare the 4 daily factors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "artifacts/cache/tushare/v1"
D2_PANELS = REPO / "artifacts/refactor_baseline/panels_d2"
OUT_JSON = REPO / "artifacts/refactor_baseline/hand_anchors_d2.json"
WINDOW_LO = pd.Timestamp("2021-07-01")
WINDOW_HI = pd.Timestamp("2026-06-30")
TOL = 1e-12
SEED = 20260724

# ---- pinned definition constants, restated as PLAIN NUMBERS (the point of a
# hand check is to restate the definition, not to import it) -----------------
BASELINE_DAYS, BASELINE_MIN_OBS, SIGMA_K = 20, 10, 1.0
MIN_VALID_DAYS, MIN_CLASSIFIABLE = 10, 100
JUMP_LOOKBACK, JUMP_MIN_PAIRS, JUMP_Z = 20, 10, 1.0
IDEAL_LOOKBACK, IDEAL_LAM, IDEAL_MIN_MINUTES = 10, 0.25, 1150
ANOM_LOOKBACK, ANOM_MIN_POOL, ANOM_MIN_SELECTED, ANOM_K = 20, 460, 20, 1.0
CUT_LOOKBACK, CUT_LAM, CUT_MIN_DAY, CUT_MIN_VALID, CUT_MIN_XS = 10, 0.20, 100, 6, 10
VPC_LOOKBACK = 20
PIK_LOOKBACK, PIK_MIN_INTERVALS = 20, 20
VVW_LOOKBACK, VVW_MIN_VALLEY = 20, 20
VRV_LOOKBACK, VRV_MIN_VALLEY, VRV_MIN_RIDGE = 20, 20, 10
RMR_LOOKBACK, RMR_MIN_RIDGE = 20, 10
VPQ_LOOKBACK, VPQ_MIN_VALLEY, VPQ_REV_DAYS, VPQ_MIN_XS = 20, 20, 20, 10
PRA_LOOKBACK, PRA_MIN_PEAK, PRA_MIN_RIDGE = 20, 5, 10


# --------------------------------------------------------------------------- #
# raw cache readers (plain parquet reads)
# --------------------------------------------------------------------------- #
def read_minutes(
    symbol: str, lo: pd.Timestamp, hi: pd.Timestamp, *, pit: bool = True
) -> pd.DataFrame:
    """1min bars for one symbol over [lo, hi] trade dates.

    ``pit=True`` keeps only the 14:50-visible bars (``bar_end + 1min <=
    trade_date + 14:50`` — the runners' ``data_lag='1min'``). ``pit=False``
    returns the FULL day: two factors need it — PR-C (jump) predates the
    cutoff convention and its engine computes on full-day bars (day-level PIT
    only), and PR-E (amp anomaly) resamples the FULL day to 5min FIRST and
    PIT-filters the DERIVED bars by their own available_time.
    """
    # Clip to the evaluation plane's start: the cache holds BACKFILLED bars
    # from before 2021-07-01, but the frozen panels were computed on
    # store.read_range(cfg.data.start, ...) — reading earlier months would
    # hand the check a richer history than the engine ever saw (the first
    # G run's early-window failures were exactly this).
    lo = max(lo, WINDOW_LO)
    months = pd.period_range(lo, hi, freq="M")
    parts = []
    base = (
        CACHE / "stk_mins_1min" / "freq=1min"
        / f"symbol_prefix={symbol[:3]}" / f"symbol={symbol}"
    )
    for p in months:
        f = base / f"year={p.year}" / f"month={p.month:02d}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "bar_end"], keep="last")
    df["trade_date"] = df["bar_end"].dt.normalize()
    df = df[(df["trade_date"] >= lo) & (df["trade_date"] <= hi)]
    if pit:
        visible = (df["bar_end"] + pd.Timedelta("1min")) <= (
            df["trade_date"] + pd.Timedelta("14:50:00")
        )
        df = df[visible]
    return df.sort_values("bar_end").reset_index(drop=True)


def read_daily(symbol: str, endpoint: str) -> pd.DataFrame:
    f = CACHE / endpoint / f"symbol_prefix={symbol[:3]}" / f"{symbol}.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
    return df[(df["date"] >= WINDOW_LO) & (df["date"] <= WINDOW_HI)].sort_values("date")


def af_series(symbol: str) -> pd.Series:
    df = read_daily(symbol, "adj_factor")
    return df.set_index("date")["adj_factor"] if len(df) else pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# the peak taxonomy BY HAND (explicit windows; no engine import)
# --------------------------------------------------------------------------- #
def classify(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar classifiable/valley/peak/ridge for ONE symbol's visible bars."""
    work = bars.copy()
    vol = work["volume"].to_numpy(float)
    work = work[np.isfinite(vol) & (vol >= 0.0)].copy()
    work["slot"] = (
        (work["bar_end"] - work["trade_date"]) // pd.Timedelta(minutes=1)
    ).astype(int)
    days = sorted(work["trade_date"].unique())
    day_ix = {d: i for i, d in enumerate(days)}
    slots = sorted(work["slot"].unique())
    slot_ix = {s: i for i, s in enumerate(slots)}
    mat = np.full((len(days), len(slots)), np.nan)
    di = work["trade_date"].map(day_ix).to_numpy()
    si = work["slot"].map(slot_ix).to_numpy()
    mat[di, si] = work["volume"].to_numpy(float)

    thr = np.full_like(mat, np.nan)
    for i in range(len(days)):
        lo = max(0, i - BASELINE_DAYS)
        win = mat[lo:i, :]  # strictly prior positions
        if win.shape[0] == 0:
            continue
        n_obs = np.sum(~np.isnan(win), axis=0)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(win, axis=0)
            sd = np.nanstd(win, axis=0, ddof=1)
        ok = n_obs >= BASELINE_MIN_OBS
        thr[i, ok] = mu[ok] + SIGMA_K * sd[ok]

    t = thr[di, si]
    v = work["volume"].to_numpy(float)
    classifiable = np.isfinite(t)
    eruptive = classifiable & (v > t)
    mild = classifiable & ~eruptive
    work["classifiable"] = classifiable
    work["valley"] = mild
    work = work.sort_values(["trade_date", "bar_end"], kind="mergesort").reset_index(
        drop=True
    )
    # neighbour test per day: exactly 60s away and mild on both sides
    peak = np.zeros(len(work), dtype=bool)
    er = work["classifiable"].to_numpy() & (
        work["volume"].to_numpy(float)
        > thr[work["trade_date"].map(day_ix).to_numpy(),
              work["slot"].map(slot_ix).to_numpy()]
    )
    ml = work["valley"].to_numpy(bool)
    be = work["bar_end"].to_numpy("datetime64[ns]").astype("int64")
    td = work["trade_date"].to_numpy()
    for i in range(len(work)):
        if not er[i]:
            continue
        prev_ok = (
            i > 0 and td[i - 1] == td[i] and be[i] - be[i - 1] == 60_000_000_000
            and ml[i - 1]
        )
        next_ok = (
            i + 1 < len(work) and td[i + 1] == td[i]
            and be[i + 1] - be[i] == 60_000_000_000 and ml[i + 1]
        )
        peak[i] = prev_ok and next_ok
    work["eruptive"] = er
    work["peak"] = peak
    work["ridge"] = er & ~peak
    return work


def day_stats(work: pd.DataFrame) -> pd.DataFrame:
    """Per-day taxonomy counts + guarded sums used by the peak family."""
    vol = work["volume"].to_numpy(float)
    amt = work["amount"].to_numpy(float)
    tradable = np.isfinite(vol) & (vol > 0) & np.isfinite(amt) & (amt > 0)
    amt_ok = np.isfinite(amt) & (amt > 0)
    g = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            "classifiable": work["classifiable"].to_numpy(bool).astype(int),
            "peak_n": (work["peak"].to_numpy(bool) & amt_ok).astype(int),
            "ridge_n_amt": (work["ridge"].to_numpy(bool) & amt_ok).astype(int),
            "valley_n": (work["valley"].to_numpy(bool) & tradable).astype(int),
            "ridge_n": (work["ridge"].to_numpy(bool) & tradable).astype(int),
            "peak_any": work["peak"].to_numpy(bool).astype(int),
            "valley_amt": np.where(work["valley"].to_numpy(bool) & tradable, amt, 0.0),
            "valley_vol": np.where(work["valley"].to_numpy(bool) & tradable, vol, 0.0),
            "ridge_amt": np.where(work["ridge"].to_numpy(bool) & tradable, amt, 0.0),
            "ridge_vol": np.where(work["ridge"].to_numpy(bool) & tradable, vol, 0.0),
            "day_amt": np.where(tradable, amt, 0.0),
            "day_vol": np.where(tradable, vol, 0.0),
            "peak_amt": np.where(work["peak"].to_numpy(bool) & amt_ok, amt, 0.0),
            "ridge_amt2": np.where(work["ridge"].to_numpy(bool) & amt_ok, amt, 0.0),
        }
    )
    return g.groupby("trade_date", sort=True).sum()


# --------------------------------------------------------------------------- #
# hand factor values (one (d, symbol) at a time; each restates the LOCKED
# definition in plain arithmetic)
# --------------------------------------------------------------------------- #
def _lookback_slice(days: list, d: pd.Timestamp, n: int) -> list:
    j = days.index(d)
    return days[max(0, j - n + 1) : j + 1]


def hand_volume_peak_count(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    work = classify(bars)
    st = day_stats(work)
    valid = st.index[st["classifiable"] >= MIN_CLASSIFIABLE].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, VPC_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    return float(st.loc[win, "peak_any"].sum())


def hand_peak_interval_kurtosis(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    work = classify(bars)
    st = day_stats(work)
    valid = st.index[st["classifiable"] >= MIN_CLASSIFIABLE].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, PIK_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    pool = []
    for day in win:
        sub = work[work["trade_date"] == day].reset_index(drop=True)
        pos = np.arange(len(sub))
        pk = pos[sub["peak"].to_numpy(bool)]
        pool.extend(np.diff(pk).tolist())
    x = np.array(pool, dtype=float)
    if x.size < PIK_MIN_INTERVALS or x.size < 4:
        return float("nan")
    dctr = x - x.mean()
    m2 = float(np.dot(dctr, dctr))
    if not m2 > 0:
        return float("nan")
    m4 = float(np.dot(dctr * dctr, dctr * dctr))
    n = x.size
    return n * (n + 1.0) * (n - 1.0) * m4 / ((n - 2.0) * (n - 3.0) * m2 * m2) - 3.0 * (
        n - 1.0
    ) ** 2 / ((n - 2.0) * (n - 3.0))


def hand_valley_relative_vwap(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    st = day_stats(classify(bars))
    valid_mask = (
        (st["classifiable"] >= MIN_CLASSIFIABLE)
        & (st["valley_n"] >= VVW_MIN_VALLEY)
        & (st["day_vol"] > 0)
        & (st["valley_vol"] > 0)
    )
    valid = st.index[valid_mask].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, VVW_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    ratios = [
        (st.loc[t, "valley_amt"] / st.loc[t, "valley_vol"])
        / (st.loc[t, "day_amt"] / st.loc[t, "day_vol"])
        for t in win
    ]
    return float(np.mean(ratios))


def hand_valley_ridge_vwap_ratio(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    st = day_stats(classify(bars))
    valid_mask = (
        (st["classifiable"] >= MIN_CLASSIFIABLE)
        & (st["valley_n"] >= VRV_MIN_VALLEY)
        & (st["ridge_n"] >= VRV_MIN_RIDGE)
        & (st["valley_vol"] > 0)
        & (st["ridge_vol"] > 0)
    )
    valid = st.index[valid_mask].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, VRV_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    ratios = [
        (st.loc[t, "valley_amt"] / st.loc[t, "valley_vol"])
        / (st.loc[t, "ridge_amt"] / st.loc[t, "ridge_vol"])
        for t in win
    ]
    return float(np.mean(ratios))


def hand_ridge_minute_return(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    work = classify(bars)
    close = work["close"].to_numpy(float)
    td = work["trade_date"].to_numpy()
    prev = np.full(len(work), np.nan)
    prev[1:] = np.where(td[1:] == td[:-1], close[:-1], np.nan)
    has_ret = np.isfinite(close) & (close > 0) & np.isfinite(prev) & (prev > 0)
    ret = np.where(has_ret, close / np.where(has_ret, prev, 1.0) - 1.0, 0.0)
    ridge_ret = work["ridge"].to_numpy(bool) & has_ret
    per = pd.DataFrame(
        {
            "trade_date": td,
            "s": np.where(ridge_ret, ret, 0.0),
            "n": ridge_ret.astype(int),
            "c": work["classifiable"].to_numpy(bool).astype(int),
        }
    ).groupby("trade_date", sort=True).sum()
    valid = per.index[(per["c"] >= MIN_CLASSIFIABLE) & (per["n"] >= RMR_MIN_RIDGE)].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, RMR_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    return float(per.loc[win, "s"].sum())


def hand_peak_ridge_amount_ratio(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=200), d)
    st = day_stats(classify(bars))
    valid_mask = (
        (st["classifiable"] >= MIN_CLASSIFIABLE)
        & (st["peak_n"] >= PRA_MIN_PEAK)
        & (st["ridge_n_amt"] >= PRA_MIN_RIDGE)
        & (st["peak_amt"] > 0)
        & (st["ridge_amt2"] > 0)
    )
    valid = st.index[valid_mask].tolist()
    if d not in valid:
        return float("nan")
    win = _lookback_slice(valid, d, PRA_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    return float(st.loc[win, "peak_amt"].sum() / st.loc[win, "ridge_amt2"].sum())


def hand_jump_amount_corr(symbol: str, d: pd.Timestamp) -> float:
    # FULL-day bars: the engine's compute has NO 14:50 cutoff (PR-C predates
    # the cutoff convention; its PIT contract is day-level, dates <= d only).
    bars = read_minutes(symbol, d - pd.Timedelta(days=90), d, pit=False)
    ok = (bars["open"].to_numpy(float) > 0) & np.isfinite(bars["amount"].to_numpy(float))
    work = bars[ok].sort_values("bar_end").reset_index(drop=True)
    days = sorted(work["trade_date"].unique())
    if d not in days:
        return float("nan")
    win = set(_lookback_slice(days, d, JUMP_LOOKBACK))
    xs, ys = [], []
    for day, sub in work.groupby("trade_date", sort=True):
        if day not in win:
            continue
        amp = (sub["high"].to_numpy(float) - sub["low"].to_numpy(float)) / sub[
            "open"
        ].to_numpy(float)
        mu, sd = amp.mean(), amp.std(ddof=1)
        z = (amp - mu) / sd if sd > 0 else np.full_like(amp, np.nan)
        be = sub["bar_end"].to_numpy("datetime64[ns]").astype("int64")
        amt = sub["amount"].to_numpy(float)
        for i in range(len(sub) - 1):
            if np.isfinite(z[i]) and z[i] > JUMP_Z and be[i + 1] - be[i] == 60_000_000_000:
                xs.append(amt[i])
                ys.append(amt[i + 1])
    n = len(xs)
    if n < JUMP_MIN_PAIRS:
        return float("nan")
    x, y = np.array(xs), np.array(ys)
    cov = n * float((x * y).sum()) - float(x.sum()) * float(y.sum())
    vx = n * float((x * x).sum()) - float(x.sum()) ** 2
    vy = n * float((y * y).sum()) - float(y.sum()) ** 2
    den = np.sqrt(vx * vy)
    if not den > 0:
        return float("nan")
    return float(np.clip(cov / den, -1.0, 1.0))


def hand_minute_ideal_amp(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=60), d)
    low = bars["low"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    work = bars[(low > 0) & (high >= low)].reset_index(drop=True)
    days = sorted(work["trade_date"].unique())
    if d not in days:
        return float("nan")
    win = set(_lookback_slice(days, d, IDEAL_LOOKBACK))
    pool = work[work["trade_date"].isin(win)]
    closes = pool["close"].to_numpy(float)
    amps = pool["high"].to_numpy(float) / pool["low"].to_numpy(float) - 1.0
    bar_ns = pool["bar_end"].to_numpy("datetime64[ns]").astype("int64")
    n = closes.size
    if n < IDEAL_MIN_MINUTES:
        return float("nan")
    k = int(np.floor(IDEAL_LAM * n))
    if k < 1:
        return float("nan")
    order = np.lexsort((bar_ns, closes))
    a = amps[order]
    return float(a[-k:].mean() - a[:k].mean())


def hand_amp_anomaly(symbol: str, d: pd.Timestamp) -> float:
    """5min resample by hand (ceil bar_end to 5min; OHLC first/max/min/last).

    ORDER MATTERS and mirrors the engine's definition: the FULL day is
    resampled to 5min FIRST, then the DERIVED bar is PIT-filtered by its own
    ``available_time = max(source 1min bar_end) + 1min <= trade_date + 14:50``
    — so a bucket containing any post-14:49 constituent is dropped WHOLE (a
    partial bucket must never masquerade as a visible 5min bar).
    """
    bars = read_minutes(symbol, d - pd.Timedelta(days=90), d, pit=False)
    if bars.empty:
        return float("nan")
    b = bars.sort_values("bar_end").copy()
    b["bucket"] = b["bar_end"].dt.ceil("5min")
    coarse = b.groupby("bucket", sort=True).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        bar_end=("bar_end", "max"),
    ).reset_index(drop=True)
    coarse["trade_date"] = coarse["bar_end"].dt.normalize()
    # PIT on the DERIVED bar: available = max(source bar_end) + 1min.
    visible = (coarse["bar_end"] + pd.Timedelta("1min")) <= (
        coarse["trade_date"] + pd.Timedelta("14:50:00")
    )
    coarse = coarse[visible].reset_index(drop=True)
    low = coarse["low"].to_numpy(float)
    high = coarse["high"].to_numpy(float)
    coarse = coarse[(low > 0) & (high >= low)].reset_index(drop=True)
    days = sorted(coarse["trade_date"].unique())
    if d not in days:
        return float("nan")
    win = set(_lookback_slice(days, d, ANOM_LOOKBACK))
    dabs_all, ret_all = [], []
    for day, sub in coarse.groupby("trade_date", sort=True):
        if day not in win:
            continue
        amp = sub["high"].to_numpy(float) / sub["low"].to_numpy(float) - 1.0
        close = sub["close"].to_numpy(float)
        dab = np.abs(np.diff(amp))
        ret = close[1:] / close[:-1] - 1.0
        keep = np.isfinite(dab) & np.isfinite(ret)
        dabs_all.extend(dab[keep].tolist())
        ret_all.extend(ret[keep].tolist())
    dabs = np.array(dabs_all)
    rets = np.array(ret_all)
    if dabs.size < ANOM_MIN_POOL:
        return float("nan")
    thr = dabs.mean() + ANOM_K * dabs.std(ddof=1)
    sel = dabs > thr
    if int(sel.sum()) < ANOM_MIN_SELECTED:
        return float("nan")
    return float(rets[sel].std(ddof=1))


def _amp_cut_stats_one(symbol: str, d: pd.Timestamp) -> tuple[float, float]:
    bars = read_minutes(symbol, d - pd.Timedelta(days=60), d)
    if bars.empty:
        return float("nan"), float("nan")
    low = bars["low"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    work = bars[(low > 0) & (high >= low)].sort_values("bar_end").reset_index(drop=True)
    vdays, vvals = [], []
    for day, sub in work.groupby("trade_date", sort=True):
        amp = sub["high"].to_numpy(float) / sub["low"].to_numpy(float) - 1.0
        close = sub["close"].to_numpy(float)
        ret = np.full(len(sub), np.nan)
        ret[1:] = close[1:] / close[:-1] - 1.0
        be = sub["bar_end"].to_numpy("datetime64[ns]").astype("int64")
        valid = np.isfinite(amp) & np.isfinite(ret)
        a, r, b_ns = amp[valid], ret[valid], be[valid]
        if a.size < CUT_MIN_DAY:
            continue
        k = int(np.floor(CUT_LAM * a.size))
        if k < 1:
            continue
        order = np.lexsort((b_ns, r))
        sa = a[order]
        vdays.append(day)
        vvals.append(float(sa[-k:].mean() - sa[:k].mean()))
    if d not in vdays:
        return float("nan"), float("nan")
    win_vals = [v for t, v in zip(vdays, vvals) if t in set(_lookback_slice(vdays, d, CUT_LOOKBACK))]
    if len(win_vals) < CUT_MIN_VALID:
        return float("nan"), float("nan")
    arr = np.array(win_vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def hand_intraday_amp_cut(symbols: list[str], target: str, d: pd.Timestamp) -> float:
    """Full cross-sectional z-combine by hand: needs EVERY covered symbol's stats."""
    vm, vs, names = [], [], []
    for s in symbols:
        m, sd = _amp_cut_stats_one(s, d)
        if np.isfinite(m) and np.isfinite(sd):
            vm.append(m)
            vs.append(sd)
            names.append(s)
    if target not in names or len(names) < CUT_MIN_XS:
        return float("nan")
    vm_a, vs_a = np.array(vm), np.array(vs)
    sm, ss = vm_a.std(ddof=1), vs_a.std(ddof=1)
    if not (np.isfinite(sm) and np.isfinite(ss)) or sm == 0 or ss == 0:
        return float("nan")
    zm = (vm_a - vm_a.mean()) / sm
    zs = (vs_a - vs_a.mean()) / ss
    return float(((zm + zs) / 2.0)[names.index(target)])


def _vpq_qbar_one(symbol: str, d: pd.Timestamp) -> float:
    bars = read_minutes(symbol, d - pd.Timedelta(days=130), d)
    if bars.empty:
        return float("nan")
    work = classify(bars)
    vol = work["volume"].to_numpy(float)
    amt = work["amount"].to_numpy(float)
    high = work["high"].to_numpy(float)
    low = work["low"].to_numpy(float)
    close = work["close"].to_numpy(float)
    tradable = np.isfinite(vol) & (vol > 0) & np.isfinite(amt) & (amt > 0)
    priced = np.isfinite(high) & np.isfinite(low) & (low > 0) & (high >= low)
    valley = work["valley"].to_numpy(bool) & tradable
    per = pd.DataFrame(
        {
            "trade_date": work["trade_date"].to_numpy(),
            "v_amt": np.where(valley, amt, 0.0),
            "v_vol": np.where(valley, vol, 0.0),
            "v_n": valley.astype(int),
            "c_n": work["classifiable"].to_numpy(bool).astype(int),
            "hi": np.where(priced, high, -np.inf),
            "lo": np.where(priced, low, np.inf),
        }
    )
    g = per.groupby("trade_date", sort=True)
    agg = g[["v_amt", "v_vol", "v_n", "c_n"]].sum()
    agg["hi"] = g["hi"].max()
    agg["lo"] = g["lo"].min()
    usable = np.isfinite(close) & (close > 0)
    lc = pd.DataFrame(
        {"trade_date": work["trade_date"].to_numpy()[usable], "close": close[usable]}
    ).groupby("trade_date", sort=True)["close"].last()
    pc = lc.reindex(agg.index).shift(1)
    hi = np.maximum(agg["hi"].to_numpy(float), pc.to_numpy(float))
    lo = np.minimum(agg["lo"].to_numpy(float), pc.to_numpy(float))
    valid = (
        (agg["c_n"].to_numpy() >= MIN_CLASSIFIABLE)
        & (agg["v_n"].to_numpy() >= VPQ_MIN_VALLEY)
        & (agg["v_vol"].to_numpy(float) > 0)
        & np.isfinite(pc.to_numpy(float))
        & np.isfinite(hi) & np.isfinite(lo) & (hi > lo)
    )
    vdays = list(agg.index[valid])
    if d not in vdays:
        return float("nan")
    q = (
        agg["v_amt"].to_numpy(float)[valid] / agg["v_vol"].to_numpy(float)[valid] - lo[valid]
    ) / (hi[valid] - lo[valid])
    qs = pd.Series(q, index=agg.index[valid])
    win = _lookback_slice(vdays, d, VPQ_LOOKBACK)
    if len(win) < MIN_VALID_DAYS:
        return float("nan")
    return float(qs.loc[win].mean())


def _qfq_close(symbol: str) -> pd.Series:
    md = read_daily(symbol, "market_daily")
    if md.empty:
        return pd.Series(dtype=float)
    af = af_series(symbol)
    close = md.set_index("date")["close"]
    afx = af.reindex(close.index)
    anchor = afx.dropna().iloc[-1] if afx.notna().any() else np.nan
    return close * afx / anchor


def hand_valley_price_quantile(symbols: list[str], target: str, d: pd.Timestamp) -> float:
    """Cross-sectional OLS residual of qbar on rev20 (T-1, qfq) BY HAND."""
    qbars, revs, names = [], [], []
    for s in symbols:
        qb = _vpq_qbar_one(s, d)
        if not np.isfinite(qb):
            continue
        qfq = _qfq_close(s)
        qfq = qfq[np.isfinite(qfq.to_numpy(float)) & (qfq.to_numpy(float) > 0)]
        if d not in qfq.index:
            continue
        j = qfq.index.get_loc(d)
        if j < VPQ_REV_DAYS + 1:
            continue
        rev = -(qfq.iloc[j - 1] / qfq.iloc[j - (VPQ_REV_DAYS + 1)] - 1.0)
        if not np.isfinite(rev):
            continue
        qbars.append(qb)
        revs.append(float(rev))
        names.append(s)
    if target not in names or len(names) < VPQ_MIN_XS:
        return float("nan")
    x, y = np.array(revs), np.array(qbars)
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    if not sxx > 0:
        return float("nan")
    ym = y.mean()
    slope = float(((x - xm) * (y - ym)).sum()) / sxx
    resid = y - (ym + slope * (x - xm))
    return float(resid[names.index(target)])


def hand_value_ratio(symbol: str, d: pd.Timestamp, field: str) -> float:
    db = read_daily(symbol, "daily_basic")
    if db.empty:
        return float("nan")
    row = db[db["date"] == d]
    if row.empty:
        return float("nan")
    v = float(row.iloc[0][field])
    return 1.0 / v if np.isfinite(v) and v > 0 else float("nan")


def hand_volatility_20(symbol: str, d: pd.Timestamp) -> float:
    qfq = _qfq_close(symbol)
    if d not in qfq.index:
        return float("nan")
    j = qfq.index.get_loc(d)
    if j < 20:
        return float("nan")
    window_prices = qfq.iloc[j - 20 : j + 1].to_numpy(float)
    rets = window_prices[1:] / window_prices[:-1] - 1.0
    if np.isnan(rets).any():
        return float("nan")
    return float(np.std(rets, ddof=1))


# NOTE on the qfq anchor: front_adjust anchors on the symbol's LAST adj_factor
# row in the loaded window; _qfq_close reproduces that from the raw caches.

# --------------------------------------------------------------------------- #
# runtime import guard: the hand computation must not have loaded the engine
# --------------------------------------------------------------------------- #
def assert_no_engine_imports() -> None:
    loaded = [
        m for m in sys.modules
        if m.startswith("factors.") or m.startswith("data.clean")
        or m in ("factors", )
    ]
    if loaded:
        raise RuntimeError(
            f"hand-anchor purity violated: engine modules imported: {loaded}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", default=str(OUT_JSON))
    args = parser.parse_args(argv)
    assert_no_engine_imports()
    from qt.hand_anchor_rows import run_hand_anchors  # heavy driver, engine-free

    return run_hand_anchors(Path(args.selection))


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
