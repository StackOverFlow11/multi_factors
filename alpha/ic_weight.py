"""RollingICWeightAlpha: walk-forward IC-weighted multi-factor combination (P3-2).

The alpha layer is the ONLY layer allowed to see forward returns, and only for
weight fitting (CLAUDE.md invariant #1; the factor layer never receives them).
This model makes that boundary auditable:

  * ``fit`` computes the per-date cross-sectional rank IC of every factor column
    against the supplied forward returns (the full series is STORED, never
    consumed whole);
  * ``predict`` at date ``d`` uses ONLY realized observations: a (factor[t],
    fwd_h[t]) pair is admissible iff ``t + h <= d`` in TRADING-DAY positions —
    the h-day forward return of factor date t realizes at t+h, so anything later
    is invisible at d. Perturbing unrealized forward returns cannot change the
    weights (locked by tests).

Weights = mean realized IC per factor over the training window (``rolling`` =
the trailing ``window`` trading days, the conservative default; ``expanding`` =
all realized history), L1-normalized and SIGN-PRESERVING (a negative-IC factor
gets a negative weight — standard IC weighting). Fallback contract: if any
factor has fewer than ``min_periods`` valid realized ICs in the window, or the
ICs are degenerate (all zero/NaN), the date falls back to the EQUAL-WEIGHT mean
(identical to ``EqualWeightAlpha``) and the fallback is logged for disclosure.

Per-date effective weights + fallback flags are recorded (``weights_log`` /
``fallback_log``) so the report can disclose them. Inputs are never mutated.
"""

from __future__ import annotations

import math

import pandas as pd

from alpha.base import AlphaModel
from analytics.factor import compute_ic

_SYMBOL_LEVEL = "symbol"
_DATE_LEVEL = "date"
_VALID_MODES = ("rolling", "expanding")


class RollingICWeightAlpha(AlphaModel):
    """Walk-forward rolling/expanding IC-weighted factor combiner."""

    # pipeline hint: this model needs historical forward returns in fit().
    requires_forward_returns: bool = True

    def __init__(
        self,
        window: int = 60,
        min_periods: int = 20,
        horizon: int = 1,
        mode: str = "rolling",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"RollingICWeightAlpha mode must be one of {_VALID_MODES}; got {mode!r}."
            )
        if window < 1 or min_periods < 1 or horizon < 1:
            raise ValueError(
                "RollingICWeightAlpha window / min_periods / horizon must be >= 1; "
                f"got window={window}, min_periods={min_periods}, horizon={horizon}."
            )
        self._window = int(window)
        self._min_periods = int(min_periods)
        self._horizon = int(horizon)
        self._mode = mode
        self._ic: pd.DataFrame | None = None
        self._calendar: pd.DatetimeIndex | None = None
        self._log: dict[pd.Timestamp, dict] = {}

    # ------------------------------------------------------------------ #
    # AlphaModel interface
    # ------------------------------------------------------------------ #
    def fit(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series | None = None,
    ) -> "RollingICWeightAlpha":
        """Compute and store the per-date per-factor rank-IC series.

        ``forward_returns`` is REQUIRED here (this model learns weights from
        realized history); the series is stored as per-date ICs and only ever
        consumed through the realized-at-d slice in :meth:`predict`.
        """
        if not isinstance(factors, pd.DataFrame):
            raise TypeError(
                "RollingICWeightAlpha expects a pandas DataFrame of factors; "
                f"got {type(factors).__name__}."
            )
        if forward_returns is None:
            raise ValueError(
                "RollingICWeightAlpha.fit requires forward_returns (it learns "
                "factor weights from realized history). Use EqualWeightAlpha for "
                "a no-future-data baseline."
            )
        ic_cols = {
            name: compute_ic(factors[name], forward_returns)
            for name in factors.columns
        }
        ic = pd.DataFrame(ic_cols).sort_index()
        self._ic = ic
        self._calendar = pd.DatetimeIndex(ic.index)
        self._log = {}
        return self

    def predict(self, factors_today: pd.DataFrame) -> pd.Series:
        """Score one DATED cross-section with weights trained walk-forward.

        The cross-section must carry its date (MultiIndex with a ``date``
        level): without it the realized-history cutoff ``t + h <= d`` cannot be
        enforced, so a plain symbol-indexed frame is a readable error.
        """
        self._require_fitted()
        date = self._extract_date(factors_today)
        weights = self.weights_for(date)
        n_factors = factors_today.shape[1]
        if weights is None:
            # EQUAL-WEIGHT fallback — identical to EqualWeightAlpha's row mean.
            scores = factors_today.mean(axis=1)
            effective = pd.Series(
                1.0 / n_factors if n_factors else float("nan"),
                index=list(factors_today.columns),
            )
            self._log[date] = {
                "fallback": True,
                "reason": (
                    f"insufficient realized IC history (< {self._min_periods} valid "
                    f"realized ICs for some factor in the {self._mode} window) or "
                    "degenerate ICs"
                ),
                "weights": effective,
            }
        else:
            # NaN rows propagate to NaN scores (skipna=False): a name missing a
            # factor is not scored on a partial weighted sum.
            scores = (factors_today[weights.index] * weights).sum(axis=1, skipna=False)
            self._log[date] = {"fallback": False, "reason": None, "weights": weights}
        return self._to_symbol_index(scores).rename("score")

    # ------------------------------------------------------------------ #
    # walk-forward training slice (the lookahead boundary lives HERE)
    # ------------------------------------------------------------------ #
    def _realized_window(self, date: pd.Timestamp) -> pd.DataFrame:
        """IC rows realized at ``date``, truncated to the training window.

        A factor date t's h-day forward return realizes at trading position
        pos(t) + h, so the admissible rows are pos(t) <= pos(date) - h. Rolling
        mode then keeps the trailing ``window`` TRADING DAYS (not the trailing
        ``window`` valid ICs — conservative and auditable); expanding keeps all.
        """
        assert self._ic is not None and self._calendar is not None
        pos_d = int(self._calendar.searchsorted(pd.Timestamp(date), side="right")) - 1
        cutoff = pos_d - self._horizon
        if cutoff < 0:
            return self._ic.iloc[0:0]
        realized = self._ic.iloc[: cutoff + 1]
        if self._mode == "rolling":
            realized = realized.iloc[-self._window:]
        return realized

    def weights_for(self, date: pd.Timestamp) -> pd.Series | None:
        """L1-normalized sign-preserving weights at ``date`` (None = fallback)."""
        self._require_fitted()
        realized = self._realized_window(pd.Timestamp(date))
        if realized.empty:
            return None
        counts = realized.notna().sum()
        if bool((counts < self._min_periods).any()):
            return None
        raw = realized.mean()  # skipna mean IC per factor
        l1 = float(raw.abs().sum())
        if not math.isfinite(l1) or l1 == 0.0:
            return None  # degenerate (all-zero / NaN ICs)
        return raw / l1

    def train_size_for(self, date: pd.Timestamp) -> int:
        """Number of realized IC rows in the training window at ``date``."""
        self._require_fitted()
        return int(len(self._realized_window(pd.Timestamp(date))))

    # ------------------------------------------------------------------ #
    # disclosure logs (consumed by the report)
    # ------------------------------------------------------------------ #
    def weights_log(self) -> pd.DataFrame:
        """Per-predicted-date EFFECTIVE weights + a ``fallback`` flag column."""
        if not self._log:
            cols = list(self._ic.columns) if self._ic is not None else []
            return pd.DataFrame(columns=[*cols, "fallback"])
        rows = {}
        for date, entry in sorted(self._log.items()):
            row = dict(entry["weights"])
            row["fallback"] = entry["fallback"]
            rows[date] = row
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = _DATE_LEVEL
        return out

    def fallback_log(self) -> dict[pd.Timestamp, str | None]:
        """date -> fallback reason (None when the trained weights were used)."""
        return {date: entry["reason"] for date, entry in sorted(self._log.items())}

    def params(self) -> dict:
        """Echoable hyper-parameters (for the report; no secrets)."""
        return {
            "window": self._window,
            "min_periods": self._min_periods,
            "horizon": self._horizon,
            "mode": self._mode,
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _require_fitted(self) -> None:
        if self._ic is None:
            raise ValueError(
                "RollingICWeightAlpha is not fitted; call fit(factors, "
                "forward_returns) first."
            )

    @staticmethod
    def _extract_date(factors_today: pd.DataFrame) -> pd.Timestamp:
        idx = factors_today.index
        if not (isinstance(idx, pd.MultiIndex) and _DATE_LEVEL in (idx.names or [])):
            raise ValueError(
                "RollingICWeightAlpha.predict needs a DATED cross-section "
                "(MultiIndex with a 'date' level): the walk-forward cutoff "
                "t + h <= d cannot be enforced without the prediction date."
            )
        dates = idx.get_level_values(_DATE_LEVEL).unique()
        if len(dates) != 1:
            raise ValueError(
                "RollingICWeightAlpha.predict expects a single-date cross-section; "
                f"got {len(dates)} distinct dates."
            )
        return pd.Timestamp(dates[0])

    @staticmethod
    def _to_symbol_index(scores: pd.Series) -> pd.Series:
        """Collapse a (date, symbol) cross-section to a plain symbol index."""
        index = scores.index
        if isinstance(index, pd.MultiIndex) and _SYMBOL_LEVEL in (index.names or []):
            symbols = index.get_level_values(_SYMBOL_LEVEL)
            out = pd.Series(scores.to_numpy(), index=symbols)
            out.index.name = _SYMBOL_LEVEL
            return out
        return scores
