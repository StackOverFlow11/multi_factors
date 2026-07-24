"""D2 hand-anchor row selection + orchestration (engine-free; see hand_anchors_d2).

Selects the stratified rows (R12), calls the plain-arithmetic hand computations
of :mod:`qt.hand_anchors_d2`, compares each against the NEW-ENGINE value read
from the ``panels_d2`` parquet files (file reads — never engine imports), and
writes the selection + results JSON. The four ops-rewritten daily factors that
have no frozen panel are SELECTED and hand-computed here too; their engine-side
values are filled in by ``qt.hand_anchors_engine_values`` (the only module of
the trio allowed to import the engine).

Selection sources (all engine-free): the ``panels_d2`` files give the finite /
NaN pattern (warm-up ends, interior rows); the ``adj_factor`` cache gives the
ex-date rows; guard-boundary rows are found by HAND-scanning seeded symbols'
day statistics and taking a day whose gated count sits exactly at its floor —
when the floor is not attained in the scan budget the NEAREST attainable day is
used and the miss is DISCLOSED in the output (never silently substituted).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qt import hand_anchors_d2 as H

TOL = H.TOL
SEED = H.SEED
D2 = H.D2_PANELS
SCAN_SYMBOLS = 30  # guard-boundary scan budget (seeded symbols per factor)


def _panel(factor_id: str) -> pd.Series:
    frame = pd.read_parquet(D2 / f"{factor_id}.parquet")
    return frame.set_index(["date", "symbol"])[factor_id].sort_index()


def _symbols(series: pd.Series) -> list[str]:
    return sorted(set(series.index.get_level_values("symbol")))


def _finite(series: pd.Series) -> pd.Series:
    return series[np.isfinite(series.to_numpy(float))]


def _pick_warmup(series: pd.Series, rng: np.random.RandomState) -> tuple:
    """A symbol's FIRST finite row (its warm-up end on the factor's own axis).

    Prefers a symbol with a real leading-NaN ramp. Some factors emit rows only
    once every accumulation floor is already crossed (amp_cut's combine drops
    non-finite stat pairs), so no ramp exists on the panel — then the symbol's
    FIRST EMITTED row IS the warm-up end and is used as the anchor.
    """
    syms = rng.permutation(_symbols(series))
    for sym in syms:
        sub = series.xs(sym, level="symbol").sort_index()
        fin = np.isfinite(sub.to_numpy(float))
        if fin.any() and not fin[0]:  # a real warm-up ramp exists
            j = int(np.argmax(fin))
            return (sub.index[j], str(sym))
    for sym in syms:  # fallback: first emitted (already finite) row
        sub = series.xs(sym, level="symbol").sort_index()
        fin = np.isfinite(sub.to_numpy(float))
        if fin.any():
            j = int(np.argmax(fin))
            return (sub.index[j], str(sym))
    raise RuntimeError("no warm-up row found")


def _pick_random(series: pd.Series, rng: np.random.RandomState, k: int = 2) -> list[tuple]:
    fin = _finite(series)
    idx = rng.choice(len(fin), size=k, replace=False)
    return [(fin.index[int(i)][0], str(fin.index[int(i)][1])) for i in idx]


def _pick_exdate(series: pd.Series, rng: np.random.RandomState, window: int) -> tuple:
    """A finite row whose trailing ``window`` panel days span a true ex-date."""
    for sym in rng.permutation(_symbols(series)):
        af = H.af_series(str(sym))
        if af.empty:
            continue
        changes = af.index[af.ne(af.shift(1)) & af.shift(1).notna()]
        if len(changes) == 0:
            continue
        sub = series.xs(sym, level="symbol").sort_index()
        dates = list(sub.index)
        fin = np.isfinite(sub.to_numpy(float))
        for ch in changes:
            after = [j for j, dt in enumerate(dates) if dt >= ch]
            for j in after[: window - 1]:
                lo = max(0, j - window + 1)
                if fin[j] and any(dates[i] >= ch for i in range(lo + 1, j + 1)):
                    return (dates[j], str(sym))
    raise RuntimeError("no ex-date row found")


def _guard_scan(
    series: pd.Series,
    rng: np.random.RandomState,
    count_of_day,
    floor: int,
) -> tuple[tuple, int]:
    """Find a finite row whose OWN day count sits exactly at ``floor``.

    ``count_of_day(symbol) -> pd.Series(trade_date -> gated count)`` is a HAND
    computation. Returns ((date, symbol), realized_count); if the exact floor is
    not attained within the scan budget the nearest attainable is returned
    (caller discloses).
    """
    best: tuple | None = None
    best_count = None
    syms = list(rng.permutation(_symbols(series)))[:SCAN_SYMBOLS]
    for sym in syms:
        counts = count_of_day(str(sym))
        if counts is None or counts.empty:
            continue
        sub = series.xs(str(sym), level="symbol")
        fin_dates = set(sub.index[np.isfinite(sub.to_numpy(float))])
        counts = counts[counts.index.isin(fin_dates)]
        if counts.empty:
            continue
        at = counts[counts == floor]
        if len(at):
            return ((at.index[0], str(sym)), floor)
        cand = counts[counts >= floor]
        if len(cand):
            c = int(cand.min())
            if best_count is None or c < best_count:
                best_count = c
                best = (cand.idxmin(), str(sym))
    if best is None:
        raise RuntimeError("guard scan found no candidate row")
    return (best, int(best_count))


# ---- per-factor day-count providers for the guard scan (hand computations) --
def _peak_family_counts(column: str):
    def provider(sym: str) -> pd.Series | None:
        bars = H.read_minutes(sym, H.WINDOW_LO, H.WINDOW_LO + pd.Timedelta(days=240))
        if bars.empty:
            return None
        st = H.day_stats(H.classify(bars))
        return st[column].astype(int)

    return provider


def run_hand_anchors(out_path: Path) -> int:
    started = time.monotonic()
    rng = np.random.RandomState(SEED)
    results: list[dict] = []

    def compare(factor_id, cls, key, hand, note=""):
        d, sym = key
        engine = float("nan")
        if (D2 / f"{factor_id}.parquet").exists():
            series = _panel(factor_id)
            if (d, sym) in series.index:
                engine = float(series.loc[(d, sym)])
        rel = (
            abs(hand - engine) / max(abs(hand), abs(engine))
            if np.isfinite(hand) and np.isfinite(engine) and max(abs(hand), abs(engine)) > 0
            else (0.0 if hand == engine or (np.isnan(hand) and np.isnan(engine)) else float("inf"))
        )
        ok = rel <= TOL
        results.append(
            {
                "factor_id": factor_id, "class": cls, "date": str(pd.Timestamp(d).date()),
                "symbol": sym, "hand": hand, "engine": engine,
                "rel_diff": rel, "ok": bool(ok), "note": note,
            }
        )
        print(f"{'OK  ' if ok else 'FAIL'} {factor_id:28s} {cls:12s} "
              f"{pd.Timestamp(d).date()} {sym} hand={hand!r} engine={engine!r} rel={rel:.2e} {note}")

    # ---------------- minute factors, peak family ---------------------------
    peak_specs = [
        ("volume_peak_count_20", H.hand_volume_peak_count, "classifiable",
         H.MIN_CLASSIFIABLE, H.VPC_LOOKBACK),
        ("peak_interval_kurtosis_20", H.hand_peak_interval_kurtosis, "peak_any",
         2, H.PIK_LOOKBACK),  # >=2 peaks -> >=1 interval on the day
        ("valley_relative_vwap_20", H.hand_valley_relative_vwap, "valley_n",
         H.VVW_MIN_VALLEY, H.VVW_LOOKBACK),
        ("valley_ridge_vwap_ratio_20", H.hand_valley_ridge_vwap_ratio, "ridge_n",
         H.VRV_MIN_RIDGE, H.VRV_LOOKBACK),
        ("ridge_minute_return_20", H.hand_ridge_minute_return, "ridge_n",
         H.RMR_MIN_RIDGE, H.RMR_LOOKBACK),
        ("peak_ridge_amount_ratio_20", H.hand_peak_ridge_amount_ratio, "peak_n",
         H.PRA_MIN_PEAK, H.PRA_LOOKBACK),
    ]
    for factor_id, hand_fn, guard_col, guard_floor, window in peak_specs:
        series = _panel(factor_id)
        w = _pick_warmup(series, rng)
        compare(factor_id, "warmup_end", w, hand_fn(w[1], w[0]))
        (g, realized) = _guard_scan(series, rng, _peak_family_counts(guard_col), guard_floor)
        note = "" if realized == guard_floor else f"nearest>{guard_floor}: {realized} (floor not attained in scan)"
        compare(factor_id, "guard_boundary", g, hand_fn(g[1], g[0]), note)
        e = _pick_exdate(series, rng, window)
        compare(factor_id, "ex_date_window", e, hand_fn(e[1], e[0]))
        for r in _pick_random(series, rng):
            compare(factor_id, "random", r, hand_fn(r[1], r[0]))

    # ---------------- jump ---------------------------------------------------
    series = _panel("jump_amount_corr_20")
    w = _pick_warmup(series, rng)
    compare("jump_amount_corr_20", "warmup_end", w, H.hand_jump_amount_corr(w[1], w[0]))
    e = _pick_exdate(series, rng, H.JUMP_LOOKBACK)
    compare("jump_amount_corr_20", "ex_date_window", e, H.hand_jump_amount_corr(e[1], e[0]))
    for r in _pick_random(series, rng, 3):
        compare("jump_amount_corr_20", "random", r, H.hand_jump_amount_corr(r[1], r[0]))

    # ---------------- minute ideal amplitude --------------------------------
    series = _panel("minute_ideal_amp_10")
    w = _pick_warmup(series, rng)
    compare("minute_ideal_amp_10", "warmup_end", w, H.hand_minute_ideal_amp(w[1], w[0]))
    e = _pick_exdate(series, rng, H.IDEAL_LOOKBACK)
    compare("minute_ideal_amp_10", "ex_date_window", e, H.hand_minute_ideal_amp(e[1], e[0]))
    for r in _pick_random(series, rng, 3):
        compare("minute_ideal_amp_10", "random", r, H.hand_minute_ideal_amp(r[1], r[0]))

    # ---------------- amp anomaly (5min derived) ----------------------------
    series = _panel("amp_marginal_anomaly_vol_20")
    w = _pick_warmup(series, rng)
    compare("amp_marginal_anomaly_vol_20", "warmup_end", w, H.hand_amp_anomaly(w[1], w[0]))
    e = _pick_exdate(series, rng, H.ANOM_LOOKBACK)
    compare("amp_marginal_anomaly_vol_20", "ex_date_window", e, H.hand_amp_anomaly(e[1], e[0]))
    for r in _pick_random(series, rng, 3):
        compare("amp_marginal_anomaly_vol_20", "random", r, H.hand_amp_anomaly(r[1], r[0]))

    # ---------------- amp cut (full cross-section; amortized on 2 dates) ----
    series = _panel("intraday_amp_cut_10")
    all_syms = _symbols(series)
    w = _pick_warmup(series, rng)
    r1, r2 = _pick_random(series, rng, 2)
    e = _pick_exdate(series, rng, H.CUT_LOOKBACK)
    picks = [("warmup_end", w), ("ex_date_window", e), ("random", r1), ("random", r2)]
    # add a second symbol on r1's date (amortizes the heavy cross-section)
    same_date = _finite(series).xs(r1[0], level="date", drop_level=False)
    extra_sym = str(same_date.index[rng.randint(len(same_date))][1])
    picks.append(("random", (r1[0], extra_sym)))
    xs_cache: dict = {}
    for cls, (d, sym) in picks:
        if d not in xs_cache:
            xs_cache[d] = {
                s: H._amp_cut_stats_one(s, d) for s in all_syms
            }
        stats = xs_cache[d]
        vm = np.array([m for m, s in stats.values()])
        vs = np.array([s for m, s in stats.values()])
        names = list(stats.keys())
        fin = np.isfinite(vm) & np.isfinite(vs)
        vm_f, vs_f = vm[fin], vs[fin]
        names_f = [n for n, f in zip(names, fin) if f]
        if sym not in names_f or len(names_f) < H.CUT_MIN_XS:
            hand = float("nan")
        else:
            sm, ss = vm_f.std(ddof=1), vs_f.std(ddof=1)
            if not (np.isfinite(sm) and np.isfinite(ss)) or sm == 0 or ss == 0:
                hand = float("nan")
            else:
                zm = (vm_f - vm_f.mean()) / sm
                zs = (vs_f - vs_f.mean()) / ss
                hand = float(((zm + zs) / 2.0)[names_f.index(sym)])
        compare("intraday_amp_cut_10", cls, (d, sym), hand,
                note=f"cross_section={int(fin.sum())}")

    # ---------------- valley price quantile (cross-section; 3 dates) --------
    # 5 rows amortized over 3 distinct dates: each date costs a full-universe
    # hand cross-section (qbar for every covered symbol), so the two extra
    # interior rows share the first random date (disclosed amortization).
    series = _panel("valley_price_quantile_20")
    w = _pick_warmup(series, rng)
    (r1,) = _pick_random(series, rng, 1)
    e = _pick_exdate(series, rng, H.VPQ_LOOKBACK)
    picks = [("warmup_end", w), ("ex_date_window", e), ("random", r1)]
    same_date = _finite(series).xs(r1[0], level="date", drop_level=False)
    for _ in range(2):
        extra_sym = str(same_date.index[rng.randint(len(same_date))][1])
        picks.append(("random", (r1[0], extra_sym)))
    vpq_cache: dict = {}
    for cls, (d, sym) in picks:
        if d not in vpq_cache:
            qb = {s: H._vpq_qbar_one(s, d) for s in all_syms}
            vpq_cache[d] = qb
        qb = vpq_cache[d]
        qbars, revs, names = [], [], []
        for s, val in qb.items():
            if not np.isfinite(val):
                continue
            qfq = H._qfq_close(s)
            qfq = qfq[np.isfinite(qfq.to_numpy(float)) & (qfq.to_numpy(float) > 0)]
            if d not in qfq.index:
                continue
            j = qfq.index.get_loc(d)
            if j < H.VPQ_REV_DAYS + 1:
                continue
            rev = -(qfq.iloc[j - 1] / qfq.iloc[j - (H.VPQ_REV_DAYS + 1)] - 1.0)
            if not np.isfinite(rev):
                continue
            qbars.append(val)
            revs.append(float(rev))
            names.append(s)
        if sym not in names or len(names) < H.VPQ_MIN_XS:
            hand = float("nan")
        else:
            x, y = np.array(revs), np.array(qbars)
            xm = x.mean()
            sxx = float(((x - xm) ** 2).sum())
            ym = y.mean()
            slope = float(((x - xm) * (y - ym)).sum()) / sxx
            hand = float((y - (ym + slope * (x - xm)))[names.index(sym)])
        compare("valley_price_quantile_20", cls, (d, sym), hand,
                note=f"cross_section={len(names)}")

    # ---------------- book factors ------------------------------------------
    for factor_id, field in (("value_ep", "pe"), ("value_bp", "pb")):
        series = _panel(factor_id)
        # guard boundary: a NaN row caused by a non-positive published ratio
        found = None
        for sym in rng.permutation(_symbols(series))[:SCAN_SYMBOLS]:
            db = H.read_daily(str(sym), "daily_basic")
            if db.empty:
                continue
            neg = db[db[field] <= 0]
            if len(neg):
                found = (neg.iloc[0]["date"], str(sym))
                break
        if found is not None:
            compare(factor_id, "guard_boundary",
                    found, H.hand_value_ratio(found[1], found[0], field),
                    note=f"{field}<=0 -> NaN")
        else:
            # Disclose "searched but not found" so the JSON is self-describing:
            # a missing guard_boundary row must be distinguishable from a
            # forgotten stratification class (review D2, LOW).
            results.append(
                {
                    "factor_id": factor_id, "class": "guard_boundary_skipped",
                    "date": "", "symbol": "", "hand": None, "engine": None,
                    "rel_diff": None, "ok": True,
                    "note": (f"scanned {SCAN_SYMBOLS} random symbols; no "
                             f"{field}<=0 row found — class searched, not omitted"),
                }
            )
            print(f"SKIP {factor_id:28s} guard_boundary: no {field}<=0 row "
                  f"in {SCAN_SYMBOLS} scanned symbols (disclosed)")
        first = series.xs(_symbols(series)[0], level="symbol").index[0]
        k0 = (first, _symbols(series)[0])
        compare(factor_id, "first_row", k0, H.hand_value_ratio(k0[1], k0[0], field),
                note="warm-up not applicable: same-day published ratio (disclosed)")
        for r in _pick_random(series, rng, 3):
            compare(factor_id, "random", r, H.hand_value_ratio(r[1], r[0], field))

    series = _panel("volatility_20")
    w = _pick_warmup(series, rng)
    compare("volatility_20", "warmup_end", w, H.hand_volatility_20(w[1], w[0]),
            note="warmup == min_periods guard boundary (same floor; disclosed)")
    e = _pick_exdate(series, rng, 21)
    compare("volatility_20", "ex_date_window", e, H.hand_volatility_20(e[1], e[0]))
    for r in _pick_random(series, rng, 3):
        compare("volatility_20", "random", r, H.hand_volatility_20(r[1], r[0]))

    # ---------------- daily ops-rewritten factors (engine side filled later) -
    daily_rows: list[dict] = []
    md_syms = _symbols(series)  # volatility panel symbols == universe symbols
    for factor_id, window, kind in (
        ("momentum_20", 20, "momentum"),
        ("reversal_20", 20, "reversal"),
        ("liquidity_20", 20, "liquidity"),
        ("overnight_mom_20", 20, "overnight"),
    ):
        rows: list[tuple[str, tuple, str]] = []
        sym = str(md_syms[int(rng.randint(len(md_syms)))])
        md = H.read_daily(sym, "market_daily")
        dates = list(md["date"])
        need = window + 1
        if len(dates) <= need + 2:
            raise RuntimeError(f"{factor_id}: symbol {sym} history too short")
        rows.append(("warmup_end", (dates[need - 1] if kind != "liquidity" else dates[window - 1], sym), ""))
        # ex-date row (price factors only; liquidity reads the amount channel)
        if kind != "liquidity":
            af = H.af_series(sym)
            ch = af.index[af.ne(af.shift(1)) & af.shift(1).notna()]
            ex_row = None
            for c in ch:
                later = [dt for dt in dates if dt >= c]
                if later:
                    j = dates.index(later[0])
                    if j + 3 < len(dates):
                        ex_row = dates[j + 3]
                        break
            if ex_row is not None:
                rows.append(("ex_date_window", (ex_row, sym), ""))
        # >= 5 rows per factor (R12): warmup + ex-date + 3 random for the
        # price factors, warmup + 4 random for liquidity (amount channel — the
        # ex-date class is not applicable and is disclosed, not faked).
        for _ in range(3 if kind != "liquidity" else 4):
            j = int(rng.randint(need + 1, len(dates)))
            rows.append(("random", (dates[j], sym), ""))
        for cls, (d, s), note in rows:
            if kind in ("momentum", "reversal"):
                qfq = H._qfq_close(s)
                hand = float("nan")
                if d in qfq.index:
                    j = qfq.index.get_loc(d)
                    if j >= window:
                        hand = float(qfq.iloc[j] / qfq.iloc[j - window] - 1.0)
                        if kind == "reversal":
                            hand = -hand
            elif kind == "liquidity":
                amt = H.read_daily(s, "market_daily").set_index("date")["amount"]
                hand = float("nan")
                if d in amt.index:
                    j = amt.index.get_loc(d)
                    if j >= window - 1:
                        m = float(amt.iloc[j - window + 1 : j + 1].mean())
                        hand = float(np.log(m)) if m > 0 else float("nan")
            else:  # overnight
                md_f = H.read_daily(s, "market_daily").set_index("date")
                af = H.af_series(s).reindex(md_f.index)
                anchor = af.dropna().iloc[-1]
                open_q = md_f["open"] * af / anchor
                close_q = md_f["close"] * af / anchor
                hand = float("nan")
                if d in md_f.index:
                    j = md_f.index.get_loc(d)
                    if j >= window:
                        terms = []
                        okall = True
                        for t in range(j - window + 1, j + 1):
                            o, c = float(open_q.iloc[t]), float(close_q.iloc[t - 1])
                            if not (o > 0 and c > 0):
                                okall = False
                                break
                            terms.append(np.log(o / c))
                        hand = float(np.sum(terms)) if okall else float("nan")
            daily_rows.append(
                {"factor_id": factor_id, "class": cls, "date": str(pd.Timestamp(d).date()),
                 "symbol": s, "hand": hand, "note": note}
            )
            print(f"HAND {factor_id:28s} {cls:12s} {pd.Timestamp(d).date()} {s} hand={hand!r}")

    H.assert_no_engine_imports()
    payload = {
        "seed": SEED, "tolerance": TOL,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "frozen14": results,
        "daily_pending_engine": daily_rows,
        "all_ok_frozen14": all(r["ok"] for r in results),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n_bad = sum(1 for r in results if not r["ok"])
    print(f"hand anchors (frozen 14): {len(results)} rows, {n_bad} mismatches -> {out_path}")
    print(f"daily rows pending engine comparison: {len(daily_rows)} "
          f"(run python -m qt.hand_anchors_engine_values)")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(run_hand_anchors(H.OUT_JSON))
