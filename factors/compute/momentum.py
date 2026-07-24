"""The ``momentum_20`` cross-sectional factor (FAC-001..004).

Definition (FIXED in CONTRACTS.md s6, fixed event order):

    momentum_20[t] = close[t] / close[t - window] - 1      # window default = 20

    compute factor at the CLOSE of date t
    rebalance AFTER the close of date t
    hold from the NEXT trading day

Using ``close[t]`` is NOT lookahead: the position formed from the factor at the
close of t is only held starting t+1, so it earns the t+1 forward return. A value
at date t therefore depends only on bars at dates <= t (INV-001 / CLAUDE.md
invariant #1). Early dates without a full ``window`` of history yield NaN.

Computation is strictly per-symbol (grouped on the ``symbol`` index level) so one
symbol's prices can never leak into another's factor value.
"""

from __future__ import annotations

import pandas as pd

from data.availability_policy import MARKET_DAILY
from factors.base import Factor
from factors.ops import ts_window_return
from factors.spec import FactorSpec, PanelField


class MomentumFactor(Factor):
    """Trailing price-momentum over a fixed lookback window.

    Args:
        window: Number of trading bars between the numerator and denominator
            close. Default 20 (the canonical ``momentum_20``).
        price_col: Panel column to use as the price. Default ``"close"``.
    """

    name: str = "momentum_20"

    def __init__(self, window: int = 20, price_col: str = "close") -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"momentum window must be a positive integer, got {window!r}."
            )
        self._window = window
        self._price_col = price_col
        # Instance name tracks the actual window so a non-default window does NOT
        # mislabel the factor column (the class attr stays the canonical default).
        self.name = f"momentum_{window}"

    @property
    def window(self) -> int:
        return self._window

    @property
    def price_col(self) -> str:
        return self._price_col

    @property
    def spec(self) -> FactorSpec:
        """Evaluation contract. A property, not a class attribute: ``factor_id``
        must track the ACTUAL window (mirrors the ``self.name`` idiom above).

        expected_ic_sign=+1: the classic cross-sectional momentum prior (winners
        keep winning). NOTE the project's own evidence disagrees in magnitude —
        P3-3/P3-4 found momentum_20 IC ~= 0 and sign-flipping across cells — but
        the prior stays the STATED hypothesis, which is the point: the sign is
        fixed before the run and the verdict then checks it factually.

        D1 declarations (derived, evidence): adjustment=returns_invariant —
        ``compute`` is the ratio of two front-adjusted (qfq) closes
        (``price / prev - 1``, this file), so the adjustment anchor cancels
        exactly (data-layer lock:
        tests/test_adjust.py::test_front_adjust_returns_invariant_to_anchor).
        overnight_boundary=none — both legs sit on the SAME qfq basis, so no
        RAW-price comparison ever crosses the ex-date basis break
        (``front_adjust`` removed it upstream, data/clean/adjust.py).
        """
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description=(
                f"Trailing {self._window}-bar price momentum: "
                f"{self._price_col}[t] / {self._price_col}[t-{self._window}] - 1."
            ),
            expected_ic_sign=+1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=(self._price_col,),
            requires=(PanelField(self._price_col, source=MARKET_DAILY),),
            adjustment="returns_invariant",
            overnight_boundary="none",
            family="momentum",
            # value at t needs bars t-window..t -> the leading ``window`` rows
            # of every symbol are NaN by construction.
            min_history_bars=self._window,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute per-symbol momentum aligned to the panel index.

        Returns a MultiIndex(date, symbol) Series named ``momentum_20``. The
        input is never mutated.
        """
        if self._price_col not in panel.columns:
            raise ValueError(
                f"momentum factor needs a '{self._price_col}' column; panel has "
                f"{list(panel.columns)}."
            )
        if not isinstance(panel.index, pd.MultiIndex) or list(
            panel.index.names
        ) != ["date", "symbol"]:
            raise ValueError(
                "momentum factor expects a MultiIndex(date, symbol) panel; got "
                f"index names {list(panel.index.names)}."
            )

        price = panel[self._price_col]
        # D2: the per-symbol strictly-backward ratio lives in factors.ops —
        # ``ts_window_return`` groups on the symbol level and lags ``window``
        # rows, so the value at t uses only bars at t and t-window (both <= t):
        # no lookahead, no cross-symbol leakage (ops convention #1).
        momentum = ts_window_return(price, self._window)

        # Preserve the exact panel index/order; name it for the factor panel.
        return momentum.reindex(panel.index).rename(self.name)
