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

RESULT, ENCODED RATHER THAN DESCRIBED. All ten bars-bound minute factors are now
clean. ``jump_amount_corr_20`` was NOT when this file was written: its compute was
the ONLY one of the eleven that applied no decision-time truncation of its own
(grep: zero ``decision_time`` / ``prepare_visible_minute_bars`` references in its
module), so under the old runners it saw 09:30-15:00 of day d.

It was recorded here as a KNOWN, NAMED exception rather than fixed, because
truncating it changes a published factor's values and could move its verdict — a
definition-affecting research decision, not a refactor's business (design v3.2 §〇).
The exception was asserted POSITIVELY, and that is precisely how the fix was forced
to be deliberate: the separate correctness-fix PR truncated the factor, THIS test
went red with the instruction to drop the entry and re-state the verdict, and the
entry was dropped in the same PR that re-stated it. The factor has moved into the
clean parametrization below, so a regression that re-introduces the leak is red
again — the positive assertion did its job and is not needed a second time.

THE LIST IS PRODUCTION STATE, NOT TEST STATE. It lives in
``factors.compute.minute.binding.NOT_DECISION_CUTOFF_SAFE`` because the exec path
consults it to decide whether an evaluation may declare ``view=decision``
(``qt.exec_basis_eval.subject_view``); this file is what MEASURES it. A copy here
would be a second thing to keep in step, and the failure mode of drifting apart is
an artifact declaring an information set its values do not have.

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
After the truncation fix it moves 0 of its cells here, like the other nine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from factors import registry as factor_registry
from factors.compute.minute.binding import (
    _MINUTE_STREAM_BINDINGS,
    NOT_DECISION_CUTOFF_SAFE,
    minute_raw_from_bars,
)

CUTOFF = "14:50:00"

#: DERIVED from the production deny list, not spelled again here. The exec path has
#: to consult the same fact to decide whether it may declare ``view=decision``
#: (``qt.exec_basis_eval.subject_view``), so the list lives with the factors and
#: this file MEASURES it. Two copies would be two things to keep in step.
KNOWN_POST_CUTOFF_DEPENDENT_IDS = frozenset(c().name for c in NOT_DECISION_CUTOFF_SAFE)

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
CLEAN_FACTOR_IDS = tuple(
    f for f in BOUND_FACTOR_IDS if f not in KNOWN_POST_CUTOFF_DEPENDENT_IDS
)


@pytest.mark.parametrize("factor_id", CLEAN_FACTOR_IDS)
def test_value_does_not_depend_on_bars_after_the_decision_cutoff(factor_id):
    """Every bars-bound minute factor: perturbing the future moves nothing.

    ``jump_amount_corr_20`` joined this parametrization when its truncation was
    fixed; before the fix it moved 1,477/1,477 cells here.
    """
    cells, moved, worst = _moved_cells(factor_id)
    assert cells > 0, f"{factor_id}: no comparable cells — the test would be vacuous"
    assert moved == 0, (
        f"{factor_id} moved {moved}/{cells} values (max |diff| {worst:.3e}) when ONLY "
        f"post-{CUTOFF} bars changed: its day-d value uses information from after its "
        f"own 14:51 entry anchor."
    )


def test_the_deny_list_is_empty_and_that_emptiness_is_a_measurement():
    """No known exception remains — and the claim is backed, not merely absent.

    "Empty deny list" and "every factor measured clean" are DIFFERENT statements:
    a list can be empty because nothing was ever measured. So this asserts both,
    and that the measured set is exactly the bound set — the emptiness is only
    worth anything while the parametrization above covers every bound factor.

    The old positive assertion (jump_amount_corr_20 is on the list AND still
    leaks) is gone because its subject is gone, not because it was inconvenient:
    it fired, red, on the branch that fixed the factor, and its message is what
    told that branch to drop the entry and re-state the verdict. A new offender
    is still caught — by the clean parametrization, which is where it belongs.
    """
    assert KNOWN_POST_CUTOFF_DEPENDENT_IDS == frozenset()
    assert set(CLEAN_FACTOR_IDS) == set(BOUND_FACTOR_IDS)
    assert len(BOUND_FACTOR_IDS) == 10, (
        "the bound minute-factor set changed; the emptiness above only covers "
        "what this file measures, so re-check the new one before trusting it"
    )


def test_every_clean_factor_is_absent_from_the_production_deny_list():
    """The measurement and the list the exec path trusts cannot disagree.

    Without this, a factor could be measured clean here while still sitting on the
    deny list (its exec evaluation blocked for no reason) or, worse, be measured
    dirty and be missing from the list (its exec artifact declaring view=decision).
    """
    for factor_id in CLEAN_FACTOR_IDS:
        assert factor_id not in KNOWN_POST_CUTOFF_DEPENDENT_IDS
    assert set(CLEAN_FACTOR_IDS) | KNOWN_POST_CUTOFF_DEPENDENT_IDS == set(
        BOUND_FACTOR_IDS
    )
