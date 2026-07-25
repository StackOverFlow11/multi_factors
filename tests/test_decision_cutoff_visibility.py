"""Does a minute factor's day-d value depend on bars AFTER the 14:50 cutoff?

THE PROPERTY. Under the ``exec_to_exec`` basis the entry anchor for date ``d`` is
``d``'s 14:51 execution bar (``qt.exec_forward_returns``). A factor value at ``d``
may therefore only use information visible at 14:50; anything later is information
from after its own entry. The D0 policy table states the same rule for the minute
endpoint (``available_time <= d 14:50``), and the D4 materializer enforces it in the
DECISION view — but that enforcement arrived with the materializer, and the eleven
factors' published values were produced by the old runners, which hand the compute
functions FULL-DAY bars and rely on each compute truncating for itself.

THE TEST. Perturb ONLY the bars strictly after the cutoff, assert every pre-cutoff
row is byte-identical (so a failure cannot be blamed on the fixture), and ask
whether the value moves. This is the "perturb the future -> value unchanged" shape
the project already uses, aimed at the WITHIN-DAY boundary rather than the
across-day one.

RESULT, ENCODED RATHER THAN DESCRIBED. Nine of the ten bars-bound minute factors are
clean. ``jump_amount_corr_20`` is not: its compute is the ONLY one of the eleven that
applies no decision-time truncation of its own (grep: zero ``decision_time`` /
``prepare_visible_minute_bars`` references in its module), so under the old runners
it saw 09:30-15:00 of day d.

That is recorded here as a KNOWN, NAMED exception rather than fixed. Truncating it
would change a published factor's values and could move its verdict — a definition
-affecting research decision, not a refactor's business (design v3.2 §〇). The
exception is asserted POSITIVELY, so the day someone changes it, this test goes red
and the change has to be made deliberately instead of noticed later.

WHY THE EXISTING SUITE DID NOT CATCH IT — and why that is not a contradiction.
``tests/test_jump_amount_corr_factor.py::test_pit_perturbing_future_bars_does_not_
change_factor_at_d`` is correct and passes: it perturbs day ``d+1`` and checks day
``d``, i.e. the ACROSS-DAY boundary. Its fixture sessions are six bars long, so a
14:50 boundary does not exist in them at all. The WITHIN-DAY boundary is a different
property, it only became load-bearing when PR #79 moved the entry anchor from
``close(d)`` to ``14:51(d)``, and nothing tested it. Two boundaries, two tests.

Measured on the real cache the same way (12 CSI500 names, 2021-07..2021-12, 361,500
bars, 4.6% of them post-cutoff): nine factors moved 0 cells; ``jump_amount_corr_20``
moved 1,477 of 1,477 with max |diff| = 1.276 on a correlation bounded in [-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors.compute.minute.binding import _MINUTE_STREAM_BINDINGS, minute_raw_from_bars

CUTOFF = "14:50:00"

#: The one factor whose value is NOT decision-cutoff-safe on its own. Named, not
#: inferred — see the module docstring for why it is recorded instead of fixed.
KNOWN_POST_CUTOFF_DEPENDENT = "jump_amount_corr_20"

SYMBOLS = [f"6000{i:02d}.SH" for i in range(12)]
DATES = pd.bdate_range("2021-01-04", periods=40)


def _session_times(day: pd.Timestamp) -> list[pd.Timestamp]:
    """A realistic A-share session: 09:31-11:30 and 13:01-15:00 (240 one-minute bars).

    The full afternoon matters: a session that stopped before 14:50 would have no
    post-cutoff bars at all and the perturbation below would be a no-op — the
    unfailable-test shape this repo keeps catching. The pre-assertion in
    :func:`_poison` refuses that fixture outright.
    """
    morning = pd.date_range(day + pd.Timedelta("09:31:00"), periods=120, freq="1min")
    afternoon = pd.date_range(day + pd.Timedelta("13:01:00"), periods=120, freq="1min")
    return list(morning) + list(afternoon)


def _bars() -> pd.DataFrame:
    rng = np.random.RandomState(5)
    rows: list[tuple] = []
    for s in SYMBOLS:
        for d in DATES:
            price = 100.0 + rng.normal(0, 2)
            for i, t in enumerate(_session_times(d)):
                price += rng.normal(0, 0.05)
                slot = 1e4 * (1.0 + 0.3 * np.sin(i / 12.0))
                erupt = 6.0 if (rng.rand() < 0.06) else 1.0
                vol = slot * erupt * (1.0 + 0.1 * rng.rand())
                w = 0.15 * price * (1.0 + (2.0 if erupt > 1 else 0.0)) * (0.5 + rng.rand())
                hi, lo = price + abs(w) * rng.rand(), price - abs(w) * rng.rand()
                cl = lo + (hi - lo) * rng.rand()
                rows.append((t, s, price, hi, lo, cl, vol, cl * vol))
    cols = ["time", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return normalize_intraday_bars(pd.DataFrame(rows, columns=cols), freq="1min")


BARS = _bars()


def _poison(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale prices/volumes of the POST-cutoff bars only; leave the rest untouched.

    Returns the perturbed frame after asserting BOTH halves of what makes the test
    meaningful: the post-cutoff rows really changed, and the pre-cutoff rows did not.
    """
    times = pd.DatetimeIndex(bars.index.get_level_values("time"))
    after = bars["available_time"].to_numpy() > (
        times.normalize() + pd.Timedelta(CUTOFF)
    ).to_numpy()
    assert after.any(), "fixture has no post-cutoff bars — the perturbation is a no-op"
    rng = np.random.RandomState(11)
    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        out.loc[after, col] = out.loc[after, col] * (1.0 + 0.5 * rng.rand(int(after.sum())))
    for col in ("volume", "amount"):
        out.loc[after, col] = out.loc[after, col] * 7.0
    assert not out.loc[after].equals(bars.loc[after]), "perturbation changed nothing"
    pd.testing.assert_frame_equal(out.loc[~after], bars.loc[~after], check_exact=True)
    return out


def _moved_cells(factor_id: str) -> tuple[int, int, float]:
    factor = factor_registry.build(factor_id)
    before = minute_raw_from_bars(factor, BARS)
    after = minute_raw_from_bars(factor, _poison(BARS))
    joined = pd.DataFrame({"a": before, "b": after}).dropna()
    if joined.empty:
        return 0, 0, 0.0
    diff = (joined["a"] - joined["b"]).abs()
    return len(joined), int((diff > 1e-12).sum()), float(diff.max())


BOUND_FACTOR_IDS = tuple(sorted(cls().name for cls in _MINUTE_STREAM_BINDINGS))
CLEAN_FACTOR_IDS = tuple(f for f in BOUND_FACTOR_IDS if f != KNOWN_POST_CUTOFF_DEPENDENT)


@pytest.mark.parametrize("factor_id", CLEAN_FACTOR_IDS)
def test_value_does_not_depend_on_bars_after_the_decision_cutoff(factor_id):
    """The nine that truncate for themselves: perturbing the future moves nothing."""
    cells, moved, worst = _moved_cells(factor_id)
    assert cells > 0, f"{factor_id}: no comparable cells — the test would be vacuous"
    assert moved == 0, (
        f"{factor_id} moved {moved}/{cells} values (max |diff| {worst:.3e}) when ONLY "
        f"post-{CUTOFF} bars changed: its day-d value uses information from after its "
        f"own 14:51 entry anchor."
    )


def test_the_known_exception_is_still_exactly_one_factor_and_still_leaks():
    """jump_amount_corr_20: recorded as a defect, asserted positively.

    Asserted in the POSITIVE direction on purpose. A test that merely tolerated the
    exception would stay green whether it was fixed, worsened, or spread to a second
    factor. This one goes red on any of those, which is what forces the decision to
    be taken deliberately.
    """
    assert KNOWN_POST_CUTOFF_DEPENDENT in BOUND_FACTOR_IDS
    cells, moved, worst = _moved_cells(KNOWN_POST_CUTOFF_DEPENDENT)
    assert cells > 0
    assert moved > 0, (
        f"{KNOWN_POST_CUTOFF_DEPENDENT} no longer depends on post-{CUTOFF} bars. If "
        f"that was deliberate, this factor's PUBLISHED values changed: update the "
        f"exception list here, and re-state its verdict rather than letting the "
        f"artifacts drift silently."
    )
    assert worst > 0.0
