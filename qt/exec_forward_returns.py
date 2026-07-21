"""``exec_to_exec`` forward returns — the 14:51 execution-anchored evaluation basis.

WHY THIS EXISTS. A minute-derived factor is fixed at the 14:50 signal cutoff, and
the only fill this framework can actually MODEL is the 1min bar in the execution
window that opens at 14:51. Scoring such a factor on ``close(t+h)/close(t)``
credits it with a closing auction this project cannot simulate (the
``closing_call_proxy`` execution model needs ``stk_auction_*``, which the data
plan does not carry). This module builds the honest alternative::

    R_exec[d, s] = (vwap(d+h, s) * af(d+h, s)) / (vwap(d, s) * af(d, s)) - 1

  * ``vwap(d, s)`` is the RAW ``amount / volume`` of the EARLIEST 1min bar whose
    ``bar_end`` falls in the execution window — the bar
    :func:`runtime.intraday_execution.resolve_fill` selects, priced by
    :func:`runtime.intraday_execution.bar_execution_price`. Both are REUSED, never
    reimplemented: two copies of an execution rule drift, and this project has
    paid for that lesson.
  * ``af`` is the daily panel's raw cumulative ``adj_factor``. The per-symbol
    anchor ``data/clean/adjust.py`` divides by cancels in the ratio, so a holding
    period spanning an ex-date is corporate-action free WITHOUT re-deriving a qfq
    series — the identity PR #75 established for
    :meth:`runtime.backtest.event_models.IntradayTailEventModel.holding_returns`,
    copied here rather than re-derived.
  * ``d+h`` is ``h`` steps on the FACTOR'S OWN EVALUATION GRID (h is "a horizon in
    evaluation periods", per ``FactorSpec.forward_return_horizon``), never the
    next calendar day and never the next day the minute cache happens to hold.

UNDEFINED IS MISSING — NEVER A FALLBACK. No bar in the window, a bar whose VWAP
is undefined (missing / non-finite / non-positive volume or amount), or a missing
/ non-positive ``adj_factor`` all yield NaN for that (date, symbol), counted under
their OWN cause (:data:`MISS_REASONS`). The bar close is never substituted for the
VWAP, the daily close is never substituted for the bar, and ``adj_factor`` is
never assumed to be 1.0 — each of those would silently manufacture a return.

CACHE-ONLY. Minute bars are read straight from
:class:`data.cache.intraday_parquet_store.IntradayParquetStore`, which has no
fetch closure: a miss yields no rows. ``stk_mins`` live calls are therefore
provably zero, and that zero is reported rather than asserted.

SHARED ARTIFACT. The expensive part — one adjusted execution price per
(date, symbol) — is written once to ``artifacts/data`` and reused by every factor
evaluated on the same universe / window / execution parameters, so the whole
factor family is scored against ONE return series and cross-factor comparisons
stay meaningful. The cache key encodes the universe, the window, the evaluation
grid and every execution parameter, so a changed parameter can never hit a stale
artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache.intraday_cache import ENDPOINT as MINUTE_ENDPOINT
from data.cache.intraday_cache import READ_COLUMNS
from data.cache.intraday_parquet_store import IntradayParquetStore
from data.clean.intraday_schema import RAW_INTRADAY_FREQ, normalize_intraday_bars
from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from factors.spec import INTRADAY_RETURN_BASIS, FactorSpec
from qt.config import IntradayCfg, RootConfig
from runtime.intraday_execution import (
    REASON_NO_BAR,
    IntradayExecutionConfig,
    build_execution_prices,
)

#: Bumped whenever the artifact's columns or their meaning change, so an old file
#: can never be read as a new one (it simply misses the key and is rebuilt).
ARTIFACT_SCHEMA_VERSION = "exec_prices_v1"

#: Per-(date, symbol) outcome of pricing the execution bar.
STATUS_OK = "ok"
#: No 1min bar in the execution window (suspended, delisted, uncached day, ...).
MISS_NO_BAR = "no_bar"
#: A bar exists but its configured price basis is undefined (for ``bar_vwap``:
#: missing / non-finite / non-positive ``volume`` or ``amount``).
MISS_BAD_VWAP = "bad_vwap"
#: The bar priced fine, but the daily panel's ``adj_factor`` is absent, NaN or
#: non-positive, so no corporate-action-free price can be formed.
MISS_BAD_ADJ_FACTOR = "bad_adj_factor"
MISS_REASONS: tuple[str, ...] = (MISS_NO_BAR, MISS_BAD_VWAP, MISS_BAD_ADJ_FACTOR)

#: Columns of the persisted artifact (stable; see ARTIFACT_SCHEMA_VERSION).
ARTIFACT_COLUMNS: tuple[str, ...] = (
    "raw_exec_price",
    "adj_factor",
    "adj_exec_price",
    "exec_time",
    "status",
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324 - not crypto


# --------------------------------------------------------------------------- #
# The ONE source of the execution parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecBasisParams:
    """Every execution parameter the ``exec_to_exec`` basis depends on.

    ONE object feeds all three consumers, so they cannot disagree:

      * :meth:`exec_config` — what actually selects and prices the bar;
      * :meth:`spec_fields` — the five ``FactorSpec`` minute-block fields the
        report publishes (a spec that DESCRIBED different parameters than the
        returns were computed with would be a lie the contract cannot catch);
      * :meth:`as_dict` — the artifact cache key, so changing a parameter forces a
        rebuild instead of hitting a stale file.

    ``from_config`` reads the config's ``intraday`` block when it declares one and
    otherwise falls back to :class:`qt.config.IntradayCfg`'s own defaults — the
    project's declared minute conventions. Which of the two applied is recorded in
    ``source`` and disclosed, never left to be assumed.
    """

    decision_cutoff: str
    data_lag: str
    session_open: str
    execution_model: str
    execution_window: tuple[str, str]
    execution_price_basis: str
    source: str

    @classmethod
    def from_config(cls, cfg: RootConfig) -> ExecBasisParams:
        declared = cfg.intraday is not None
        ic = cfg.intraday if declared else IntradayCfg()
        return cls(
            decision_cutoff=ic.decision_time,
            data_lag=ic.data_lag,
            session_open=ic.session_open,
            execution_model=ic.execution_model,
            execution_window=(ic.execution_window[0], ic.execution_window[1]),
            execution_price_basis=ic.execution_price_basis,
            source=(
                "config intraday block"
                if declared
                else "qt.config.IntradayCfg defaults (config declares no intraday block)"
            ),
        )

    def exec_config(self) -> IntradayExecutionConfig:
        """The runtime execution config — validated by ITS OWN ``__post_init__``.

        Building it here means the window/model/basis are checked by the same code
        the intraday backtest uses; there is no second validation path to drift.
        """
        return IntradayExecutionConfig(
            decision_time=self.decision_cutoff,
            data_lag=self.data_lag,
            execution_model=self.execution_model,
            execution_window=self.execution_window,
            execution_price_basis=self.execution_price_basis,
        )

    def spec_fields(self) -> dict[str, str]:
        """The five ``FactorSpec`` minute-block fields, from THESE parameters."""
        return {
            "decision_cutoff": self.decision_cutoff,
            "data_lag": self.data_lag,
            "session_open": self.session_open,
            "execution_model": self.execution_model,
            "execution_window": (
                f"[{self.execution_window[0]},{self.execution_window[1]}]"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        """Deterministic, JSON-safe view for the cache key and the disclosure."""
        return {
            "decision_cutoff": self.decision_cutoff,
            "data_lag": self.data_lag,
            "session_open": self.session_open,
            "execution_model": self.execution_model,
            "execution_window": list(self.execution_window),
            "execution_price_basis": self.execution_price_basis,
        }


def intraday_spec_variant(spec: FactorSpec, params: ExecBasisParams) -> FactorSpec:
    """Derive the ``exec_to_exec`` twin of a daily ``spec`` — no literal editing.

    The factor itself is untouched (same ``factor_id`` / ``version`` / hypothesis /
    horizon / inputs: the VALUES are identical, only what they are scored against
    changes). ``FactorSpec.__post_init__`` then enforces the minute block's
    completeness; that check is deliberately left to run rather than worked around.
    """
    if spec.is_intraday:
        raise ValueError(
            f"intraday_spec_variant expects the DAILY spec of {spec.factor_id!r} "
            f"(is_intraday=False); it is already an intraday variant. Deriving a "
            f"variant of a variant would hide which basis the report describes."
        )
    return replace(
        spec,
        is_intraday=True,
        return_basis=INTRADAY_RETURN_BASIS,
        **params.spec_fields(),
    )


# --------------------------------------------------------------------------- #
# The adjusted execution-price panel (the shared, cached artifact)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecPricePanel:
    """One adjusted execution price per (date, symbol), plus how it was obtained."""

    frame: pd.DataFrame  # MultiIndex(date, symbol), ARTIFACT_COLUMNS
    params: ExecBasisParams
    key: str
    path: Path
    reused: bool
    minute_live_calls: int
    symbols_requested: int
    symbols_with_bars: int
    raw_minute_rows: int

    @property
    def adjusted_price(self) -> pd.Series:
        """``raw VWAP * adj_factor`` — NaN wherever the pair is not ``ok``."""
        return self.frame["adj_exec_price"]

    def status_counts(self) -> dict[str, int]:
        """Row count per status over the whole (date, symbol) grid."""
        counts = self.frame["status"].value_counts()
        keys = (STATUS_OK, *MISS_REASONS)
        return {k: int(counts.get(k, 0)) for k in keys}


def _restrict_to_execution_window(
    stored: pd.DataFrame, params: ExecBasisParams
) -> pd.DataFrame:
    """Keep only bars whose ``bar_end`` time-of-day lies in the execution window.

    A PERFORMANCE pre-filter, not a semantic one: :func:`resolve_fill` applies the
    very same window to whatever it is handed, so dropping the other ~230 bars per
    day cannot change which bar is selected or how it is priced (locked by
    ``test_window_prefilter_matches_full_day_build``). It is what makes a five-year
    all-symbol build affordable — the alternative is normalizing ~40x more rows.
    """
    if stored.empty:
        return stored
    bar_end = pd.to_datetime(stored["bar_end"])
    tod = bar_end - bar_end.dt.normalize()
    lo = pd.Timedelta(params.execution_window[0])
    hi = pd.Timedelta(params.execution_window[1])
    return stored.loc[(tod >= lo) & (tod <= hi)]


def _symbol_exec_fills(
    store: IntradayParquetStore,
    symbol: str,
    dates: list[pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: ExecBasisParams,
    exec_cfg: IntradayExecutionConfig,
) -> tuple[list, int]:
    """``(fills, raw_minute_rows)`` for ONE symbol over ``dates`` — cache-only.

    ``IntradayParquetStore.read_range`` carries no fetch closure, so a cache miss
    yields an empty frame and NEVER a live ``stk_mins`` call.
    """
    stored = store.read_range(MINUTE_ENDPOINT, symbol, RAW_INTRADAY_FREQ, start, end)
    raw_rows = int(len(stored))
    window = _restrict_to_execution_window(stored, params)
    if window.empty:
        return [], raw_rows
    bars = normalize_intraday_bars(
        window.rename(columns={"bar_end": "time"})[READ_COLUMNS],
        freq=RAW_INTRADAY_FREQ,
        data_lag=params.data_lag,
    )
    _, fills = build_execution_prices(bars, dates, [symbol], exec_cfg)
    return fills, raw_rows


def _fills_frame(fills: list, symbol: str) -> pd.DataFrame:
    """Fills of one symbol -> frame indexed by (date, symbol)."""
    if not fills:
        return pd.DataFrame(
            columns=["raw_exec_price", "exec_time", "miss_reason"],
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([]), pd.Index([], dtype=object)],
                names=[DATE_LEVEL, SYMBOL_LEVEL],
            ),
        )
    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([pd.Timestamp(f.date).normalize() for f in fills]),
            pd.Index([symbol] * len(fills), dtype=object),
        ],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )
    return pd.DataFrame(
        {
            "raw_exec_price": [
                np.nan if f.exec_price is None else float(f.exec_price) for f in fills
            ],
            "exec_time": [
                pd.NaT if f.exec_time is None else pd.Timestamp(f.exec_time)
                for f in fills
            ],
            "miss_reason": [
                None
                if not f.blocked
                else (MISS_NO_BAR if f.reason == REASON_NO_BAR else MISS_BAD_VWAP)
                for f in fills
            ],
        },
        index=index,
    )


def _assemble(panel: pd.DataFrame, collected: pd.DataFrame) -> pd.DataFrame:
    """Join the per-symbol fills onto the panel grid and classify every row."""
    frame = pd.DataFrame(index=panel.index)
    aligned = collected.reindex(panel.index)
    frame["raw_exec_price"] = aligned["raw_exec_price"].astype(float)
    frame["adj_factor"] = pd.to_numeric(panel["adj_factor"], errors="coerce")
    frame["exec_time"] = pd.to_datetime(aligned["exec_time"])

    price = frame["raw_exec_price"].to_numpy(dtype=float)
    factor = frame["adj_factor"].to_numpy(dtype=float)
    price_ok = np.isfinite(price)
    # Mirrors IntradayTailEventModel._index_adj_factors: only a finite, STRICTLY
    # POSITIVE factor is usable. Treating a missing one as 1.0 is precisely the
    # defect PR #75 removed from the intraday holding return.
    factor_ok = np.isfinite(factor) & (factor > 0.0)

    frame["adj_exec_price"] = np.where(price_ok & factor_ok, price * factor, np.nan)

    reason = aligned["miss_reason"].to_numpy(dtype=object)
    # A symbol/date the loop never produced a fill for (no cached minute at all)
    # is "no bar" — the same cause, reached without a fill record.
    reason = np.where(pd.isna(reason), MISS_NO_BAR, reason)
    status = np.where(
        price_ok,
        np.where(factor_ok, STATUS_OK, MISS_BAD_ADJ_FACTOR),
        reason,
    )
    frame["status"] = pd.Series(status, index=frame.index, dtype=object)
    return frame[list(ARTIFACT_COLUMNS)]


def artifact_key(
    cfg: RootConfig,
    symbols: list[str],
    dates: pd.Index,
    params: ExecBasisParams,
) -> tuple[str, dict]:
    """``(key, payload)`` identifying one adjusted-execution-price artifact.

    Everything the numbers depend on is in the payload: the universe, the window,
    the cache root, the exact symbol set and evaluation grid, and every execution
    parameter. Change any of them and the key changes, so a stale artifact cannot
    be mistaken for a current one — the failure mode that would silently score a
    factor family against the wrong returns.
    """
    payload = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "universe_type": cfg.universe.type,
        "index_code": cfg.universe.index_code,
        "start": str(cfg.data.start),
        "end": str(cfg.data.end),
        "cache_root": str(cfg.data.cache.root_dir),
        "price_adjust": "qfq",
        "params": params.as_dict(),
        "n_symbols": len(symbols),
        "n_dates": int(len(dates)),
        "symbols_sha1": _sha1("|".join(sorted(str(s) for s in symbols))),
        "dates_sha1": _sha1(
            "|".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates)
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha1(blob)[:16], payload


def _artifact_paths(cfg: RootConfig, key: str) -> tuple[Path, Path]:
    stem = f"exec_forward_returns_{key}"
    directory = Path(cfg.output.data_dir)
    return directory / f"{stem}.parquet", directory / f"{stem}.json"


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def build_exec_price_panel(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    params: ExecBasisParams,
    logger,
    *,
    force_rebuild: bool = False,
) -> ExecPricePanel:
    """Adjusted execution prices for the whole (date, symbol) grid — build or reuse.

    ``panel`` is the front-adjusted daily panel: it supplies BOTH the evaluation
    grid (its index) and the raw ``adj_factor`` the adjustment identity needs.
    Prices come only from the minute cache; nothing is fetched.
    """
    if "adj_factor" not in panel.columns:
        raise ValueError(
            "build_exec_price_panel needs the daily panel's raw 'adj_factor' column "
            "(front_adjust preserves it). Without it the exec-to-exec return cannot "
            "be made corporate-action free, and assuming 1.0 is not an option."
        )
    dates = pd.Index(
        pd.unique(panel.index.get_level_values(DATE_LEVEL)), name=DATE_LEVEL
    ).sort_values()
    key, key_payload = artifact_key(cfg, symbols, dates, params)
    parquet_path, meta_path = _artifact_paths(cfg, key)

    if parquet_path.exists() and not force_rebuild:
        frame = pd.read_parquet(parquet_path)
        logger.info(
            "exec price artifact: REUSED %s (%d rows, key=%s)",
            parquet_path, len(frame), key,
        )
        priced = frame["raw_exec_price"].notna().groupby(level=SYMBOL_LEVEL).any()
        return ExecPricePanel(
            frame=frame,
            params=params,
            key=key,
            path=parquet_path,
            reused=True,
            minute_live_calls=0,
            symbols_requested=len(symbols),
            symbols_with_bars=int(priced.sum()),
            raw_minute_rows=0,
        )

    exec_cfg = params.exec_config()
    store = IntradayParquetStore(cfg.data.cache.root_dir)
    start = pd.Timestamp(dates[0]).normalize()
    end = pd.Timestamp(dates[-1]).normalize() + pd.Timedelta("23:59:59")

    grid = pd.DataFrame(
        {
            "date": panel.index.get_level_values(DATE_LEVEL),
            "symbol": panel.index.get_level_values(SYMBOL_LEVEL).astype(str),
        }
    )
    by_symbol: dict[str, list[pd.Timestamp]] = {
        str(symbol): list(part["date"])
        for symbol, part in grid.groupby("symbol", sort=True)
    }

    parts: list[pd.DataFrame] = []
    raw_rows = 0
    with_bars = 0
    ordered = sorted(by_symbol)
    for i, symbol in enumerate(ordered):
        fills, rows = _symbol_exec_fills(
            store, symbol, by_symbol[symbol], start, end, params, exec_cfg
        )
        raw_rows += rows
        if fills:
            part = _fills_frame(fills, symbol)
            if part["raw_exec_price"].notna().any():
                with_bars += 1
            parts.append(part)
        if (i + 1) % 100 == 0:
            logger.info(
                "exec price build: %d/%d symbols (%d with a priced bar)",
                i + 1, len(ordered), with_bars,
            )
    collected = (
        pd.concat(parts)
        if parts
        else _fills_frame([], "")
    )
    if collected.index.has_duplicates:
        raise ValueError(
            "exec price build produced duplicate (date, symbol) rows; one would "
            "silently win the alignment. This means a symbol was processed twice."
        )
    frame = _assemble(panel, collected)

    _write_atomic(frame, parquet_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"key": key, **key_payload}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "exec price artifact: BUILT %s (%d rows, %d symbols with a priced bar, "
        "%d raw 1min rows read, stk_mins_live_calls=0, key=%s)",
        parquet_path, len(frame), with_bars, raw_rows, key,
    )
    return ExecPricePanel(
        frame=frame,
        params=params,
        key=key,
        path=parquet_path,
        reused=False,
        minute_live_calls=0,
        symbols_requested=len(symbols),
        symbols_with_bars=with_bars,
        raw_minute_rows=raw_rows,
    )


# --------------------------------------------------------------------------- #
# Forward returns on the FACTOR's own evaluation grid
# --------------------------------------------------------------------------- #
def exec_forward_returns(
    adjusted_price: pd.Series, dates: pd.Index, horizon: int
) -> pd.Series:
    """``R_h`` from adjusted execution prices, shifted on ``dates`` — h PERIODS.

    Deliberately the same shape of computation as
    :func:`analytics.factor.forward_returns` (restrict to the evaluation grid, mask
    a non-positive denominator to NaN, then ``groupby(symbol).shift(-h)``), so the
    ``exec_to_exec`` series and the ``close_to_close`` series it replaces are
    aligned by exactly the same rule and differ ONLY in the price they are built
    from. ``h`` counts EVALUATION PERIODS: a date the grid does not contain is not
    a next period, no matter what the minute cache holds for it.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError(
            f"exec_forward_returns: horizon must be a positive int (evaluation "
            f"periods); got {horizon!r}."
        )
    on_grid = adjusted_price[
        adjusted_price.index.get_level_values(DATE_LEVEL).isin(dates)
    ].sort_index()
    safe = on_grid.where(on_grid > 0.0)
    grouped = safe.groupby(level=SYMBOL_LEVEL, sort=False, group_keys=False)
    forward = grouped.shift(-horizon)
    return (forward / safe - 1.0).rename(f"forward_return_{horizon}d")


# --------------------------------------------------------------------------- #
# Coverage loss vs the close_to_close basis (the price of the new basis)
# --------------------------------------------------------------------------- #
def coverage_loss(
    exec_returns: pd.Series,
    close_returns: pd.Series,
    status: pd.Series,
    horizon: int,
) -> dict[str, object]:
    """What the exec basis costs, in (date, symbol) pairs, BY CAUSE.

    A pair is "lost" when ``close_to_close`` produced a finite return and
    ``exec_to_exec`` did not. The cause is read off the pair's ENTRY anchor status
    first and its EXIT anchor status otherwise — the entry is what a trade would
    have hit first, and attributing to one cause keeps the counts a partition
    rather than an overlapping tally.
    """
    close = close_returns.reindex(exec_returns.index)
    lost_mask = np.isfinite(close.to_numpy(dtype=float)) & ~np.isfinite(
        exec_returns.to_numpy(dtype=float)
    )
    measurable = int(np.isfinite(close.to_numpy(dtype=float)).sum())
    lost_index = exec_returns.index[lost_mask]

    entry_status = status.reindex(exec_returns.index)
    # The exit anchor is h evaluation periods later WITHIN the symbol — the same
    # positional shift the returns used, so causes line up with the returns.
    exit_status = entry_status.groupby(
        level=SYMBOL_LEVEL, sort=False, group_keys=False
    ).shift(-horizon)
    # Entry first, exit otherwise: one cause per lost pair, so the counts partition
    # the loss instead of double-counting a pair blocked at both ends.
    cause = entry_status.where(entry_status != STATUS_OK).fillna(exit_status)
    lost_cause = cause[lost_mask]
    counts = lost_cause.value_counts()
    by_cause = {r: int(counts.get(r, 0)) for r in MISS_REASONS}
    # Neither anchor names a cause (both read "ok"): only reachable via a
    # non-positive price. Counted apart rather than folded into a cause it does
    # not belong to.
    unattributed = int(len(lost_cause) - sum(by_cause.values()))
    lost_symbols = set(lost_index.get_level_values(SYMBOL_LEVEL).astype(str))
    return {
        "close_to_close_measurable_pairs": measurable,
        "exec_to_exec_measurable_pairs": int(
            np.isfinite(exec_returns.to_numpy(dtype=float)).sum()
        ),
        "lost_pairs": int(lost_mask.sum()),
        "lost_pairs_pct_of_close_to_close": (
            float(lost_mask.sum()) / measurable * 100.0 if measurable else float("nan")
        ),
        "lost_pairs_by_cause": by_cause,
        "lost_pairs_unattributed": unattributed,
        "distinct_symbols_affected": len(lost_symbols),
    }


__all__ = [
    "ARTIFACT_COLUMNS",
    "ARTIFACT_SCHEMA_VERSION",
    "MISS_BAD_ADJ_FACTOR",
    "MISS_BAD_VWAP",
    "MISS_NO_BAR",
    "MISS_REASONS",
    "STATUS_OK",
    "ExecBasisParams",
    "ExecPricePanel",
    "artifact_key",
    "build_exec_price_panel",
    "coverage_loss",
    "exec_forward_returns",
    "intraday_spec_variant",
]
