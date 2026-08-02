"""The I3 session-return feature as a first-class factor (D6c).

``intraday_ret_0930_1450`` — the close/open return from session open to the
14:50 decision cutoff — was an I3 aggregate feature consumed by the I5a/I5b
intraday tail framework, never a registered Factor. D6c makes it one so the
two intraday runners can be switched onto FactorService (PR-2).

The ret MATH stays in ``data.clean.intraday_aggregate.asof_daily_features``
(the R14 generic core, its single definition point): the raw compute here
DELEGATES to it and surfaces the one column under the factor's name. This
module may import the aggregate module (the code-hash allowlist lists it and
the one-hop rule folds it into this factor's code hash); the reverse
direction is what the cycle rule forbids, and the aggregate module never
imports this one.
"""

from __future__ import annotations

import pandas as pd

from data.availability_policy import STK_MINS_1MIN
from data.clean.intraday_aggregate import asof_daily_features
from factors.base import Factor
from factors.spec import FactorSpec, PanelField

#: The factor id == the legacy I3 aggregate column name (every shipped I5a/I5b
#: report is keyed on it verbatim). Pinned equal by test.
SESSION_RET_FACTOR_NAME = "intraday_ret_0930_1450"


def _minute_requires(*fields: str) -> tuple[PanelField, ...]:
    """The stk_mins_1min requires tuple of a minute-derived factor (D1)."""
    return tuple(PanelField(f, source=STK_MINS_1MIN) for f in fields)


def compute_intraday_session_ret(
    bars: pd.DataFrame,
    *,
    name: str = SESSION_RET_FACTOR_NAME,
) -> pd.Series:
    """The I3 ``ret`` session feature as a factor raw daily Series (D6c).

    Pure delegation: :func:`data.clean.intraday_aggregate.asof_daily_features`
    with ``features=["ret"]`` applies the per-bar PIT cutoff
    (``available_time <= trade_date + 14:50``) and computes
    ``last_visible_close / first_visible_open - 1`` per ``(date, symbol)``;
    this function only takes the sole column out under ``name``. Keeping the
    math at its single definition point means the factor can never drift from
    the feature the I5a/I5b reports describe.

    Returns:
        ``MultiIndex(date, symbol)`` float Series (midnight-normalized dates),
        sorted, named ``name``. Pure: never mutates ``bars``.
    """
    frame = asof_daily_features(bars, features=["ret"])
    if len(frame.columns) != 1:  # pragma: no cover - features=["ret"] is one key
        raise RuntimeError(
            f"asof_daily_features(features=['ret']) returned "
            f"{list(frame.columns)}; expected exactly the one ret column."
        )
    return frame[frame.columns[0]].rename(name)


class IntradaySessionRetFactor(Factor):
    """The I3 session-return score as a first-class factor (D6c).

    The factor id IS the legacy aggregate column name
    (``intraday_ret_0930_1450``), so every report already keyed on that column
    keeps its text byte-for-byte. ``compute`` surfaces the pre-aggregated
    daily column the runner placed on the panel; the raw bars-based compute
    lives in :func:`compute_intraday_session_ret` (bound in
    ``factors.compute.minute.binding``).

    expected_ic_sign=-1: a same-day session return reverts (short-horizon
    intraday reversal) — a pre-registered hypothesis, not a validated
    conclusion: the I5a/I5b runs used this feature as a framework smoke score,
    never as research alpha.
    """

    name: str = SESSION_RET_FACTOR_NAME

    spec = FactorSpec(
        factor_id=SESSION_RET_FACTOR_NAME,
        version="1.0",
        description=(
            "Intraday session return: 1min bars PIT-truncated at 14:50 per "
            "bar, then last-visible-close / first-visible-open - 1 per "
            "(date, symbol) — the I3 'ret' feature (I5a/I5b framework smoke "
            "score; EXPLORATORY, not a research alpha)."
        ),
        expected_ic_sign=-1,
        is_intraday=True,
        forward_return_horizon=1,
        return_basis="exec_to_exec",
        input_fields=("open", "close"),
        requires=_minute_requires("open", "close"),
        adjustment="returns_invariant",
        overnight_boundary="none",
        family="microstructure",
        min_history_bars=0,
        # The value at d reads ONLY d's own visible bars, so the transitive
        # depth is the materializer's floor: the signal day itself.
        lookback_depth=1,
        decision_cutoff="14:50:00",
        data_lag="1min",
        session_open="09:30:00",
        execution_model="next_minute_close",
        execution_window="[14:51:00,14:56:59]",
    )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Select the pre-aggregated daily session-return column off ``panel``.

        The runner runs :func:`compute_intraday_session_ret` on the minute
        cache upstream and joins the result as ``self.name``; here we only
        surface it, so this factor does no temporal logic and cannot introduce
        lookahead.
        """
        if self.name not in panel.columns:
            raise ValueError(
                f"IntradaySessionRetFactor needs the pre-aggregated "
                f"'{self.name}' column on the panel (produced upstream by "
                f"compute_intraday_session_ret and joined by the runner); "
                f"panel has {list(panel.columns)}."
            )
        return panel[self.name].rename(self.name)


__all__ = [
    "SESSION_RET_FACTOR_NAME",
    "IntradaySessionRetFactor",
    "compute_intraday_session_ret",
]
