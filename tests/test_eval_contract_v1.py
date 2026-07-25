"""Factor-evaluation contract v1.0 (D5 C3): identity fields + basis-column guard.

Three groups, matching what the upgrade is allowed and forbidden to do:

1. IDENTITY — ``EvalConfig`` carries ``view`` / ``return_basis`` and enforces the
   legal pairing at construction time (design §1.4 mechanism 1, D0's single
   source), and the report states them.
2. NON-OMITTABLE BASIS COLUMN (R24) — a cross-basis summary that omits ``basis`` /
   ``view`` REFUSES to render.
3. THE ADJUDICATION CORE DID NOT MOVE — every ``VerdictThresholds`` default is
   pinned to its frozen close-era value. "Upgrade, do not rewrite" is only a claim
   until a test says a moved threshold is a failure; §十 R24 forbids re-calibrating
   a bar inside the refactor, so the bar is nailed down here rather than described.
"""

from __future__ import annotations

import pytest

from analytics.eval import (
    EVAL_CONTRACT_VERSION,
    EvalConfig,
    VerdictThresholds,
    basis_identity_phrase,
    render_verdict_summary,
    require_basis_columns,
)
from analytics.eval.render import _requires_row
from data.availability_policy import ReturnBasis, View


def _cfg(**overrides) -> EvalConfig:
    base = dict(
        universe="000905.SH",
        universe_is_pit=True,
        start="2021-07-01",
        end="2026-06-30",
        is_exploratory=True,
        post_hoc_selected=False,
    )
    base.update(overrides)
    return EvalConfig(**base)


# --------------------------------------------------------------------------- #
# 1. identity fields
# --------------------------------------------------------------------------- #
def test_the_default_describes_the_callers_that_do_not_pass_it():
    """The default is the LEGACY pairing, and that is the point.

    A default's only job is to be TRUE for the callers that omit the field. Today
    those are the eleven close-basis runners, each of which builds ONE EvalConfig
    and hands it to both its close_to_close reports AND (via qt.exec_basis_eval) its
    exec ones. Defaulting to decision/exec_to_exec would have made every close
    artifact state a basis it was not scored on — the exact describe-the-check drift
    the field was added to prevent, introduced by the field itself. The exec path
    sets the identity explicitly instead (see the test below).
    """
    cfg = _cfg()
    assert cfg.view == View.CLOSE.value
    assert cfg.return_basis == ReturnBasis.CLOSE_TO_CLOSE.value


def test_the_exec_pairing_is_constructible_and_is_what_the_exec_path_declares():
    cfg = _cfg(view="decision", return_basis="exec_to_exec")
    assert (cfg.view, cfg.return_basis) == ("decision", "exec_to_exec")


def test_the_exec_basis_module_restates_the_identity_it_scores_on():
    """qt.exec_basis_eval must not inherit the caller's close-basis identity.

    Structural rather than descriptive: the module derives the exec twin of the
    SPEC one line earlier, and the config's identity has to travel the same way or
    the exec artifact would carry the close label.
    """
    import inspect

    from qt import exec_basis_eval

    source = inspect.getsource(exec_basis_eval.run_exec_basis_evaluation)
    assert "replace(" in source and "EXEC_TO_EXEC" in source and "DECISION" in source


@pytest.mark.parametrize(
    ("view", "basis"),
    [("close", "exec_to_exec"), ("decision", "close_to_close")],
)
def test_illegal_pairing_is_a_construction_error(view, basis):
    """The mixed pairings raise — the failure the D0 pairing rule exists to cause."""
    with pytest.raises(ValueError, match="illegal view/basis pairing"):
        _cfg(view=view, return_basis=basis)


def test_unknown_view_is_rejected_not_passed_through():
    with pytest.raises(ValueError):
        _cfg(view="intraday_9am", return_basis="exec_to_exec")


def test_enum_members_normalize_to_canonical_strings():
    """An enum member in, canonical VALUE out — the exported record has one spelling."""
    cfg = _cfg(view=View.DECISION, return_basis=ReturnBasis.EXEC_TO_EXEC)
    assert cfg.view == "decision" and cfg.return_basis == "exec_to_exec"
    assert isinstance(cfg.view, str) and type(cfg.view) is str


def test_identity_phrase_is_authored_once():
    """The provenance row COMPOSES the phrase rather than spelling it again."""
    cfg = _cfg(view="decision", return_basis="exec_to_exec")
    assert (
        basis_identity_phrase(cfg.view, cfg.return_basis)
        == "view=decision x return_basis=exec_to_exec"
    )
    legacy = _cfg()
    assert (
        basis_identity_phrase(legacy.view, legacy.return_basis)
        == "view=close x return_basis=close_to_close"
    )


# --------------------------------------------------------------------------- #
# 2. the non-omittable basis column (R24)
# --------------------------------------------------------------------------- #
_ROWS = [
    {"factor_id": "a_20", "verdict": "Watch", "return_basis": "exec_to_exec", "view": "decision"},
    {
        "factor_id": "b_20",
        "verdict": "Watch",
        "return_basis": "close_to_close",
        "view": "close",
    },
]


def test_summary_renders_when_basis_and_view_are_present():
    out = render_verdict_summary(
        _ROWS, columns=("factor_id", "verdict", "return_basis", "view")
    )
    assert out == (
        "| factor_id | verdict | return_basis | view |\n"
        "|---|---|---|---|\n"
        "| a_20 | Watch | exec_to_exec | decision |\n"
        "| b_20 | Watch | close_to_close | close |\n"
    )


@pytest.mark.parametrize(
    "columns",
    [
        ("factor_id", "verdict"),                       # both omitted
        ("factor_id", "verdict", "return_basis"),       # view omitted
        ("factor_id", "verdict", "view"),               # basis omitted
    ],
)
def test_summary_refuses_to_render_without_the_identity_columns(columns):
    with pytest.raises(ValueError, match="identity column"):
        render_verdict_summary(_ROWS, columns=columns)


def test_declaring_the_column_and_leaving_it_blank_is_the_same_omission():
    rows = [dict(_ROWS[0]), {**_ROWS[1], "return_basis": ""}]
    with pytest.raises(ValueError, match="leaves the identity column"):
        render_verdict_summary(rows, columns=("factor_id", "return_basis", "view"))


def test_require_basis_columns_is_reusable_on_its_own():
    assert require_basis_columns(["view", "return_basis", "x"]) == (
        "view",
        "return_basis",
        "x",
    )
    with pytest.raises(ValueError):
        require_basis_columns(["x"])


# --------------------------------------------------------------------------- #
# 3. the adjudication core did NOT move (§十 R24: no re-calibration in a refactor)
# --------------------------------------------------------------------------- #
def test_every_verdict_threshold_default_is_the_frozen_close_era_value():
    """Pinned VALUES, not "unchanged" prose.

    These are the UNVALIDATED close-era defaults (v0.7-v0.9). Carrying them over
    byte-for-byte is what makes this cycle's verdicts comparable to the eleven-factor
    loop's; moving one inside a refactor would make every verdict in this cycle
    uninterpretable, and would do it silently.
    """
    t = VerdictThresholds()
    assert t.min_rebalances == 12
    assert t.min_effective_samples == 24.0
    assert t.min_span_days == 365
    assert t.min_abs_icir == 0.30
    assert t.min_incremental_abs_icir == 0.15
    assert t.min_ic_win_rate == 0.55
    assert t.min_abs_nw_t == 2.0
    assert t.min_monotonicity_spearman == 0.0


def test_contract_version_is_stated():
    assert EVAL_CONTRACT_VERSION == "1.0"


# --------------------------------------------------------------------------- #
# provenance rendering of the factor's declarations
# --------------------------------------------------------------------------- #
class _FakeField:
    def __init__(self, field: str, source: str) -> None:
        self.field, self.source = field, source


class _FakeSpec:
    def __init__(self, requires) -> None:
        self.requires = requires


def test_requires_row_renders_source_dot_field_sorted():
    spec = _FakeSpec(
        (_FakeField("close", "market_daily"), _FakeField("volume", "stk_mins_1min"))
    )
    assert _requires_row(spec) == "market_daily.close, stk_mins_1min.volume"


def test_empty_requires_is_marked_not_blank():
    """An empty declaration is stated, never rendered as an empty row."""
    assert _requires_row(_FakeSpec(())) == "(none declared)"
    assert _requires_row(_FakeSpec(None)) == "(none declared)"
