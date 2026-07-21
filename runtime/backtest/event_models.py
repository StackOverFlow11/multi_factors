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
    at the start of the execution window. Per-symbol entry/exit prices come from
    the earliest valid 1min bar in the execution window (``next_minute_close``),
    priced on the config's ``execution_price_basis`` (default that bar's VWAP =
    ``amount / volume``); holding returns are
    ``exec_price(exit) / exec_price(entry) - 1``.

    Base (I5a) feasibility rule: a symbol can be traded at the rebalance ONLY if
    it has a valid (non-NaN) execution bar at the entry anchor; a missing/NaN
    execution bar blocks BOTH directions. A suspended stock has no minute bars and
    is blocked by this rule.

    I5b adds OPT-IN execution-time price-limit feasibility (``price_limit_check``):
    on top of the bar-exists rule, the RAW price the trade actually pays — the
    ``execution_price_basis``, a bar VWAP by default — is compared to that
    symbol/date's raw ``stk_limit`` band. A buy is blocked at the upper limit, a
    sell at the lower limit, directionally.

    Why the gate reads the VWAP and not the bar close. A limit-up execution minute
    has exactly two shapes, and they are the feasibility question:

      * LOCKED (封死涨停) — every trade in the minute prints at the limit, so the
        VWAP equals the limit price up to rounding. There is no seller except at
        the limit and the queue is hopeless: the buy must be blocked.
      * OPENED (盘中打开) — some trades printed below the limit, so the VWAP sits
        below it. Those prints are direct evidence that a fill WAS achievable:
        the buy must go through.

    The bar close cannot make that distinction and misclassifies both edges — a
    minute that closed at the limit but opened during it gets over-blocked, and a
    minute that closed below but was locked most of the way gets under-blocked.
    The VWAP encodes exactly "was there volume available below the limit".

    Calibration follows from that, and it is the part that is easy to get wrong:
    ``limit_tolerance`` must stay at ROUNDING scale and must NOT be widened to
    "catch near-limit bars". Widening it re-blocks fills that demonstrably were
    achievable, which is the opposite of the point. Measured on real cached 14:51
    bars joined to raw ``stk_limit`` (13,699 bars; 149 closing at the up limit):
    the 146 LOCKED ones sit a median 0.000000% and at most 0.0021% below the
    limit — pure ``amount`` rounding — while the 3 OPENED ones sit 0.013%-0.017%
    below it. The populations separate by an order of magnitude. At the default
    ``limit_tolerance=1e-6`` every OPENED minute is correctly allowed and 2.1% of
    LOCKED minutes are misclassified as opened by sub-tick rounding; widening to
    0.01 RMB would instead misclassify 100% of OPENED minutes as locked.

    An "opened but with tiny volume below the limit" bar is a CAPACITY question,
    not a feasibility one: it belongs to the I5f capacity layer, which sizes the
    trade against the execution minute's traded ``amount``. This gate answers only
    whether a fill was possible at all.

    The comparison is RAW-vs-RAW only: the raw 1min execution price (the intraday
    cache stores unadjusted bars, and a bar VWAP = raw amount / raw volume is raw
    too) against raw ``stk_limit``; it never reads a qfq / daily close or a
    daily-close-derived limit flag. A missing/NaN limit row never silently
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
        precomputed_prices: tuple[pd.DataFrame, list[ExecutionFill]] | None = None,
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
        # The execution-price matrix + fills are a PURE function of
        # (bars, anchor_dates, symbols, cfg). When several models share the SAME
        # bars/cfg (e.g. one fresh model per quantile group in I5d), the caller may
        # pass ``precomputed_prices`` — the exact ``(prices, fills)`` returned by
        # ``build_execution_prices`` over those same inputs — so the heavy matrix is
        # built ONCE and reused, while each model keeps its OWN fresh mutable
        # feasibility diagnostics. Default None reproduces the original build
        # in-place (byte-identical I5a/I5b behaviour, locked by tests).
        if precomputed_prices is not None:
            prices, fills = precomputed_prices
        elif anchor_dates and symbols:
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
        # The RAW CLOSE of each selected execution bar, keyed by (date, symbol).
        # NOT the gate's input (the gate compares the executed price); used only to
        # label OPENED limit minutes for diagnostics. Carried on the fills, so it
        # is available on the ``precomputed_prices`` path too.
        self._limit_refs: dict[tuple[pd.Timestamp, str], float] = (
            self._index_limit_refs(self._fills) if self._price_limit_check else {}
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
        # OPENED limit minutes: the bar closed at a limit but traded through it, so
        # the VWAP gate lets the fill stand where a close-based gate would block.
        # Purely diagnostic — recorded so that divergence is reportable, not silent.
        self._opened_limit_ups: dict[tuple[pd.Timestamp, str], float] = {}
        self._opened_limit_downs: dict[tuple[pd.Timestamp, str], float] = {}
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

    @staticmethod
    def _index_limit_refs(
        fills: list[ExecutionFill],
    ) -> dict[tuple[pd.Timestamp, str], float]:
        """Index the selected execution bars' RAW closes by (date, symbol).

        A fill with no usable raw close (no bar, or a NaN close) contributes no
        entry, so the gate reports that pair as unchecked rather than passing it.
        """
        out: dict[tuple[pd.Timestamp, str], float] = {}
        for f in fills:
            ref = f.limit_reference_price
            if ref is None or pd.isna(ref):
                continue
            out[(pd.Timestamp(f.date).normalize(), str(f.symbol))] = float(ref)
        return out

    def _closed_at(self, key: tuple[pd.Timestamp, str], limit: float) -> bool:
        """Did the selected bar's RAW CLOSE sit at ``limit``? (diagnostic only.)

        Used solely to label an OPENED limit minute — one that ended at the limit
        but traded through it. It never affects ``can_buy``/``can_sell``.
        """
        ref = self._limit_refs.get(key)
        if ref is None:
            return False
        return abs(ref - float(limit)) <= self._limit_tol

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
        if the RAW execution price sits at/above the raw upper limit, and a sell if
        it sits at/below the raw lower limit (raw-vs-raw, with ``limit_tolerance``
        as the equality band). Limit-up blocks BUY only; limit-down blocks SELL
        only — the other direction still executes if the bar exists.

        The gate's input is the price the trade PAYS. A locked minute's VWAP is the
        limit price (every print is there); an opened minute's VWAP is below it
        precisely because volume traded below the limit, which is what makes the
        fill achievable. See the class docstring for the measured separation and
        for why ``limit_tolerance`` must stay at rounding scale.
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
                    ref = float(price)
                    if ref >= up - self._limit_tol:
                        buy_ok = False
                        self._up_blocked_buys[key] = up
                    elif self._closed_at(key, up):
                        # Opened limit-up: the minute ENDED at the limit but traded
                        # below it, so the fill stands. Counted so the divergence
                        # from a close-based gate is auditable, never silent.
                        self._opened_limit_ups[key] = up
                    if ref <= down + self._limit_tol:
                        sell_ok = False
                        self._down_blocked_sells[key] = down
                    elif self._closed_at(key, down):
                        self._opened_limit_downs[key] = down
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

    def opened_limit_up_minutes(self) -> int:
        """Buys ALLOWED at a bar that closed at the up limit but traded below it.

        The exact set where the VWAP gate diverges from a close-based gate: the
        minute ended locked, yet volume printed under the limit, so the fill was
        achievable. Reported so the divergence is auditable rather than silent.
        """
        return len(self._opened_limit_ups)

    def opened_limit_down_minutes(self) -> int:
        """Sells ALLOWED at a bar that closed at the down limit but traded above it."""
        return len(self._opened_limit_downs)

    def up_limit_blocked_buy_keys(self) -> set[tuple[pd.Timestamp, str]]:
        """(rebalance date, symbol) keys whose BUY was blocked by a raw up-limit.

        Read-only view of the idempotent diagnostic state populated during
        ``feasibility()``. Exposed so a report-only liquidity diagnostic (I5f) can
        skip trades the existing feasibility already blocked, without re-deriving
        or reclassifying the block reason.
        """
        return set(self._up_blocked_buys.keys())

    def down_limit_blocked_sell_keys(self) -> set[tuple[pd.Timestamp, str]]:
        """(rebalance date, symbol) keys whose SELL was blocked by a raw down-limit."""
        return set(self._down_blocked_sells.keys())
