"""Event models for the shared backtest engine (I5a).

Each model bundles a *schedule* with a *pricing* and *feasibility* time-basis:

  * :class:`DailyCloseEventModel` — the accepted monthly close-to-close model.
    Pricing and feasibility both read the daily panel; reproduces the legacy
    ``BacktestDriver`` ledger byte-for-byte.
  * :class:`IntradayTailEventModel` — a 14:50-decision / 14:51-execution tail
    rebalance. Pricing is the 1min execution-bar price (reusing
    :func:`runtime.intraday_execution.build_execution_prices`); returns are
    exec-to-exec; a missing/NaN execution bar is an explicit block, NEVER a
    daily-close fallback.

Both schedules come from :func:`runtime.backtest.events.monthly_anchor_pairs`, so
daily and intraday rebalance on the same calendar; only the time basis differs.
The intraday model never prices off the daily panel.
"""

from __future__ import annotations

import pandas as pd

from runtime.backtest.events import (
    HoldingPeriod,
    monthly_anchor_pairs,
    trading_calendar,
)
from runtime.fills import feasibility_from_cross
from runtime.intraday_execution import (
    ExecutionFill,
    IntradayExecutionConfig,
    build_execution_prices,
)


class DailyCloseEventModel:
    """Monthly close-to-close model — the accepted daily behaviour.

    decision = execution = entry = the rebalance date (close of T); exit = the
    next rebalance date (close of T_next). Holding returns and execution
    feasibility both read the daily ``prices`` panel, exactly as the legacy
    driver did.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        if "close" not in prices.columns:
            raise ValueError("price panel must have a 'close' column")
        self._prices = prices

    def holding_periods(self) -> list[HoldingPeriod]:
        pairs = monthly_anchor_pairs(trading_calendar(self._prices))
        return [
            HoldingPeriod(
                date=date,
                entry_date=date,
                exit_date=exit_date,
                decision_ts=date,
                execution_ts=date,
                exit_execution_ts=exit_date,  # priced at the exit date's close
            )
            for date, exit_date in pairs
        ]

    def _close_at(self, date: pd.Timestamp, symbol: str) -> float:
        try:
            return float(self._prices.loc[(date, symbol), "close"])
        except KeyError:
            return float("nan")

    def holding_returns(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> pd.Series:
        """Close-to-close gross return per symbol over (entry, exit].

        Symbols with a missing/zero start price get a flat (0.0) return rather
        than NaN, so the book stays well-defined (matches the legacy driver).
        """
        start, end = period.entry_date, period.exit_date
        out: dict[str, float] = {}
        for sym in symbols:
            start_px = self._close_at(start, sym)
            end_px = self._close_at(end, sym)
            if (
                pd.isna(start_px)
                or pd.isna(end_px)
                or start_px == 0.0
            ):
                out[sym] = 0.0
            else:
                out[sym] = end_px / start_px - 1.0
        return pd.Series(out, dtype=float)

    def feasibility(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> tuple[dict, dict]:
        """Per-symbol (can_buy, can_sell) from the rebalance date's cross-section."""
        if not symbols:
            return {}, {}
        try:
            cross = self._prices.xs(
                pd.Timestamp(period.date).normalize(), level="date"
            )
        except KeyError:
            cross = self._prices.iloc[0:0]
        return feasibility_from_cross(cross, symbols)


class IntradayTailEventModel:
    """14:50-decision / 14:51-execution tail rebalance over minute bars (I5a/I5b).

    The rebalance schedule is the same monthly calendar as the daily model, but
    each period's decision is timestamped at ``decision_time`` and its execution
    at the start of the execution window. Per-symbol entry/exit prices are the
    earliest valid 1min close in the execution window (``next_minute_close``);
    holding returns are ``exec_price(exit) / exec_price(entry) - 1``.

    Base (I5a) feasibility rule: a symbol can be traded at the rebalance ONLY if
    it has a valid (non-NaN) execution bar at the entry anchor; a missing/NaN
    execution bar blocks BOTH directions. A suspended stock has no minute bars and
    is blocked by this rule.

    I5b adds OPT-IN execution-time price-limit feasibility (``price_limit_check``):
    on top of the bar-exists rule, the selected execution-minute RAW close is
    compared to that symbol/date's raw ``stk_limit`` band — a buy is blocked at the
    upper limit, a sell is blocked at the lower limit, directionally. The
    comparison is RAW-vs-RAW only: the raw 1min close (the intraday cache stores
    unadjusted bars) against raw ``stk_limit``; it never reads a qfq / daily close
    or a daily-close-derived limit flag. A missing/NaN limit row never silently
    counts as a passed check: in strict mode (``require_price_limit_coverage``) a
    missing REQUIRED row fails at construction (before any result); in lenient mode
    it is counted/disclosed and the name falls back to the bar-exists rule (the
    limit is recorded as unchecked, not as cleared).
    """

    def __init__(
        self,
        *,
        calendar_panel: pd.DataFrame,
        bars: pd.DataFrame,
        cfg: IntradayExecutionConfig | None = None,
        price_limits: pd.DataFrame | None = None,
        price_limit_check: bool = False,
        limit_tolerance: float = 1e-6,
        require_price_limit_coverage: bool = True,
    ) -> None:
        self._calendar_panel = calendar_panel
        self._cfg = cfg or IntradayExecutionConfig()
        pairs = monthly_anchor_pairs(trading_calendar(calendar_panel))
        self._pairs = pairs
        # Anchor dates we must price: every rebalance date AND every exit date.
        anchor_dates = sorted(
            {d for pair in pairs for d in pair},
            key=lambda t: pd.Timestamp(t),
        )
        symbols = sorted({str(s) for s in bars.index.get_level_values("symbol")})
        self._symbols = symbols
        if anchor_dates and symbols:
            prices, fills = build_execution_prices(
                bars, anchor_dates, symbols, self._cfg
            )
        else:
            prices = pd.DataFrame()
            fills = []
        self._exec_prices = prices
        self._fills = fills

        # -- I5b price-limit feasibility state ----------------------------- #
        self._price_limit_check = bool(price_limit_check)
        self._limit_tol = float(limit_tolerance)
        self._require_limit_coverage = bool(require_price_limit_coverage)
        # raw limit band keyed by (normalized date, symbol); only rows with BOTH a
        # non-NaN up_limit and down_limit count as a usable (covered) limit. Built
        # only when the check is on (off -> the map is never consulted).
        self._limits: dict[tuple[pd.Timestamp, str], tuple[float, float]] = (
            self._index_limits(price_limits) if self._price_limit_check else {}
        )
        # entry/rebalance anchor dates (the first element of each pair); the exit
        # date of the final pair is a pure pricing anchor and never a decision, so
        # it needs no limit coverage.
        self._entry_dates = sorted(
            {pd.Timestamp(p[0]).normalize() for p in pairs},
            key=lambda t: t,
        )
        # idempotent, reason-attributed limit diagnostics keyed by (date, symbol),
        # so repeated feasibility() calls (e.g. a second engine.run) never double
        # count. Populated lazily as feasibility() is evaluated per rebalance.
        self._up_blocked_buys: dict[tuple[pd.Timestamp, str], float] = {}
        self._down_blocked_sells: dict[tuple[pd.Timestamp, str], float] = {}
        self._unchecked_limits: set[tuple[pd.Timestamp, str]] = set()
        if self._price_limit_check and self._require_limit_coverage:
            self._assert_limit_coverage()

    @staticmethod
    def _index_limits(
        price_limits: pd.DataFrame | None,
    ) -> dict[tuple[pd.Timestamp, str], tuple[float, float]]:
        """Index raw ``stk_limit`` rows by (date, symbol) -> (up_limit, down_limit).

        Only rows with BOTH limits present (non-NaN) are usable; a NaN on either
        side leaves the pair uncovered (so the limit check cannot silently pass).
        """
        out: dict[tuple[pd.Timestamp, str], tuple[float, float]] = {}
        if price_limits is None or len(price_limits) == 0:
            return out
        for _, row in price_limits.iterrows():
            up = row.get("up_limit", float("nan"))
            down = row.get("down_limit", float("nan"))
            if pd.isna(up) or pd.isna(down):
                continue
            key = (pd.Timestamp(row["date"]).normalize(), str(row["symbol"]))
            out[key] = (float(up), float(down))
        return out

    def _required_limit_pairs(self) -> list[tuple[pd.Timestamp, str]]:
        """(entry anchor date, symbol) pairs that need a raw limit row.

        Only names with a valid execution bar at a REBALANCE date can reach the
        limit gate (a missing bar already blocks both directions before it), so
        those are exactly the pairs that require coverage — deterministic and
        independent of which names selection happens to pick.
        """
        pairs: list[tuple[pd.Timestamp, str]] = []
        for d in self._entry_dates:
            for s in self._symbols:
                if pd.notna(self._exec_price(d, s)):
                    pairs.append((d, s))
        return pairs

    def limit_coverage(self) -> dict[str, int]:
        """{required, present, missing} raw-limit coverage over rebalance anchors."""
        required = self._required_limit_pairs()
        present = [p for p in required if p in self._limits]
        return {
            "required": len(required),
            "present": len(present),
            "missing": len(required) - len(present),
        }

    def _assert_limit_coverage(self) -> None:
        """Strict mode: fail before any result if a required limit row is missing."""
        missing = [p for p in self._required_limit_pairs() if p not in self._limits]
        if not missing:
            return
        shown = ", ".join(f"{d.date()}|{s}" for d, s in missing[:10])
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise ValueError(
            "intraday price-limit feasibility blocked: "
            f"{len(missing)} required (rebalance date, symbol) pairs have no raw "
            f"stk_limit row (e.g. {shown}{more}). With "
            "intraday.require_price_limit_coverage=true a missing limit row must "
            "NOT be treated as a passed check. Extend the stk_limit cache/window, "
            "or set require_price_limit_coverage=false to drop to lenient mode "
            "(missing rows are counted/disclosed and fall back to the bar-exists "
            "rule, never a silent limit pass)."
        )

    def holding_periods(self) -> list[HoldingPeriod]:
        decision = pd.Timedelta(self._cfg.decision_time)
        execution = pd.Timedelta(self._cfg.execution_window[0])
        out: list[HoldingPeriod] = []
        for date, exit_date in self._pairs:
            day = pd.Timestamp(date).normalize()
            exit_day = pd.Timestamp(exit_date).normalize()
            out.append(
                HoldingPeriod(
                    date=date,
                    entry_date=date,
                    exit_date=exit_date,
                    decision_ts=day + decision,
                    execution_ts=day + execution,
                    # the book is priced out at the exit date's execution-window
                    # start (the planned exit fill time), NOT close-to-close.
                    exit_execution_ts=exit_day + execution,
                )
            )
        return out

    def _exec_price(self, date: pd.Timestamp, symbol: str) -> float:
        try:
            return float(self._exec_prices.loc[pd.Timestamp(date).normalize(), str(symbol)])
        except (KeyError, TypeError):
            return float("nan")

    def holding_returns(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> pd.Series:
        """exec-to-exec gross return per symbol over (entry, exit].

        A symbol missing an execution price at EITHER anchor is omitted (so the
        engine's ``settle`` treats it as flat — it earns nothing for the period).
        Never substitutes a daily close.
        """
        entry, exit_ = period.entry_date, period.exit_date
        out: dict[str, float] = {}
        for sym in symbols:
            a = self._exec_price(entry, sym)
            b = self._exec_price(exit_, sym)
            if pd.notna(a) and pd.notna(b) and a != 0.0:
                out[str(sym)] = b / a - 1.0
            # else: blocked at an anchor -> omitted -> flat via settle.
        return pd.Series(out, dtype=float)

    def feasibility(
        self, period: HoldingPeriod, symbols: list[str]
    ) -> tuple[dict, dict]:
        """Per-symbol (can_buy, can_sell) at the rebalance's execution minute.

        Base rule (I5a): tradable in a direction iff a valid execution bar exists
        at the entry anchor; a missing bar blocks both directions BEFORE any limit
        logic. I5b layer (when ``price_limit_check``): a buy is additionally blocked
        if the selected execution-minute raw close sits at/above the raw upper
        limit, and a sell if it sits at/below the raw lower limit (raw-vs-raw, with
        ``limit_tolerance`` as the equality band). Limit-up blocks BUY only;
        limit-down blocks SELL only — the other direction still executes if the bar
        exists.
        """
        entry = pd.Timestamp(period.entry_date).normalize()
        can_buy: dict[str, bool] = {}
        can_sell: dict[str, bool] = {}
        for sym in symbols:
            s = str(sym)
            price = self._exec_price(entry, s)
            has_bar = bool(pd.notna(price))
            buy_ok = has_bar
            sell_ok = has_bar
            if has_bar and self._price_limit_check:
                key = (entry, s)
                band = self._limits.get(key)
                if band is not None:
                    up, down = band
                    if float(price) >= up - self._limit_tol:
                        buy_ok = False
                        self._up_blocked_buys[key] = up
                    if float(price) <= down + self._limit_tol:
                        sell_ok = False
                        self._down_blocked_sells[key] = down
                else:
                    # No usable raw limit row: in lenient mode we must not pretend
                    # the limit was checked. Record it as unchecked and fall back
                    # to the bar-exists rule (strict mode already failed at build).
                    self._unchecked_limits.add(key)
            can_buy[s] = buy_ok
            can_sell[s] = sell_ok
        return can_buy, can_sell

    # -- diagnostics ------------------------------------------------------ #
    def execution_prices(self) -> pd.DataFrame:
        """The (date x symbol) execution-price matrix (NaN = blocked)."""
        return self._exec_prices

    def fills(self) -> list[ExecutionFill]:
        """Per-(date, symbol) execution fill log (incl. blocked reasons)."""
        return list(self._fills)

    def blocked_fills(self) -> list[ExecutionFill]:
        """Only the blocked fills (no execution bar / NaN price)."""
        return [f for f in self._fills if f.blocked]

    # -- I5b price-limit diagnostics (idempotent over feasibility() calls) - #
    def price_limit_check_enabled(self) -> bool:
        """True if execution-time price-limit feasibility is active."""
        return self._price_limit_check

    def up_limit_blocked_buys(self) -> int:
        """Count of (rebalance date, symbol) buys blocked by a raw up-limit."""
        return len(self._up_blocked_buys)

    def down_limit_blocked_sells(self) -> int:
        """Count of (rebalance date, symbol) sells blocked by a raw down-limit."""
        return len(self._down_blocked_sells)

    def missing_limit_rows(self) -> int:
        """Count of evaluated (date, symbol) pairs with no usable raw limit row."""
        return len(self._unchecked_limits)
