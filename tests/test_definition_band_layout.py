"""The FACTOR DEFINITION band must not draw its text on top of itself.

WHAT THIS CATCHES, AND WHY A HUMAN LOOKING AT THE PNG DOES NOT COUNT
--------------------------------------------------------------------
The band anchors its pieces at FIXED axes fractions: the description at y=0.62
(va="top") and the metadata row at y=0.20/0.10 (va="bottom") — 0.20 when the
spec is intraday (the minute block then also draws at 0.04), 0.10 otherwise.
Nothing bounds the description, so a long one simply keeps wrapping downward
and is DRAWN OVER the metadata — both texts become unreadable, and the
artifact silently loses content it appears to show. Measured with the real
renderer before the bound existed, nine of the eleven shipped minute factors
overlapped (``valley_price_quantile_20`` by 296 px daily / 314 px intraday,
at 25 wrapped lines against the slot).

WHY BOTH FORMS ARE MEASURED, NOT ONE
------------------------------------
The two spec forms have DIFFERENT capacities (5 lines daily, 4 intraday —
the metadata row sits 0.10 higher in the intraday branch). The dashboard a
minute factor's exec-basis report actually ships renders the
``intraday_spec_variant`` (``is_intraday=True``), while ``build(factor_id).spec``
is the daily form — so a guard that only measures the close form protects the
half that was never broken and misses the 9-of-11 exec-form overlap. Every
assertion in this file therefore runs against BOTH the close spec and the
exec spec derived by the same ``intraday_spec_variant`` the exec path uses.

That the underlying defect is standing rather than a regression:
``analytics/eval/figures.py`` had exactly one commit in its history when the
bound was added, so every dashboard ever produced was drawn by that code.

The check is geometric, not visual: render the band on the REAL dashboard
geometry, ask matplotlib for each Text's window extent, and assert the boxes
do not intersect. A person squinting at a PNG is not a guard, and neither is
a substring match on anything — the failure mode is pixels landing on pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from analytics.eval.figures import (  # noqa: E402
    DEFINITION_MAX_LINES_DAILY,
    DEFINITION_MAX_LINES_INTRADAY,
    DEFINITION_WRAP_WIDTH,
    _definition_band,
    definition_description_lines,
    definition_max_lines,
)
from factors.compute.minute import binding as binding_module  # noqa: E402
from factors.registry import build  # noqa: E402
from qt.config import IntradayCfg  # noqa: E402
from qt.exec_forward_returns import (  # noqa: E402
    ExecBasisParams,
    intraday_spec_variant,
)

#: Every minute factor that reaches a dashboard. Derived from the binding tables
#: (plus the deferred one) rather than listed, so a new factor is covered without
#: anyone remembering to add it here.
MINUTE_FACTOR_IDS = tuple(
    sorted(
        {cls().name for cls in binding_module._MINUTE_STREAM_BINDINGS}
        | {"valley_price_quantile_20"}
    )
)

#: The two dashboard forms a minute factor ships in: the close-basis one
#: (daily spec) and the exec-basis one (``intraday_spec_variant``).
FORMS = ("close", "exec")


def _exec_params() -> ExecBasisParams:
    """The exec block, sourced from the SAME defaults the exec path falls back to.

    Built field-by-field from ``IntradayCfg()`` exactly as
    ``ExecBasisParams.from_config`` does for a config with no intraday block,
    so a changed project default moves this test instead of silently making it
    measure a geometry the shipped dashboards no longer use.
    """
    ic = IntradayCfg()
    return ExecBasisParams(
        decision_cutoff=ic.decision_time,
        data_lag=ic.data_lag,
        session_open=ic.session_open,
        execution_model=ic.execution_model,
        execution_window=(ic.execution_window[0], ic.execution_window[1]),
        execution_price_basis=ic.execution_price_basis,
        source="test: qt.config.IntradayCfg defaults",
    )


def _spec_for(factor_id: str, form: str):
    """The spec the dashboard of the given form actually renders."""
    spec = build(factor_id).spec
    if form == "close":
        return spec
    return intraday_spec_variant(spec, _exec_params())


class _Data:
    """Minimal stand-in for DashboardData: the band reads only ``spec``."""

    def __init__(self, spec) -> None:
        self.spec = spec


def _band_boxes(spec):
    """(description bbox, lowest metadata bbox) on the REAL dashboard geometry.

    The figsize and GridSpec are copied from ``figures._render`` on purpose: a
    layout assertion measured on some other canvas size would prove nothing about
    the artifact that actually ships.
    """
    fig = plt.figure(figsize=(15, 21.5))
    gs = GridSpec(
        7, 3, figure=fig,
        height_ratios=[0.30, 0.38, 0.80, 1.5, 1.1, 1.0, 1.0],
        hspace=0.62, wspace=0.28,
        left=0.06, right=0.945, top=0.975, bottom=0.04,
    )
    ax = fig.add_subplot(gs[2, :])
    _definition_band(ax, _Data(spec))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    description = None
    metadata = []
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        if text.get_va() == "top" and "\n" in text.get_text() or (
            text.get_va() == "top" and text.get_text() == spec.description
        ):
            description = box
        elif text.get_va() == "bottom":
            metadata.append(box)
    if description is None:  # single-line description (no newline to key on)
        candidates = [
            t for t in ax.texts
            if t.get_va() == "top" and t.get_text() not in ("FACTOR DEFINITION",)
        ]
        description = max(candidates, key=lambda t: t.get_window_extent(renderer).width
                          ).get_window_extent(renderer)
    top_metadata = max(metadata, key=lambda b: b.y1)
    plt.close(fig)
    return description, top_metadata


@pytest.mark.parametrize("factor_id", MINUTE_FACTOR_IDS)
@pytest.mark.parametrize("form", FORMS)
def test_the_description_never_overlaps_the_metadata_row(factor_id, form):
    """The load-bearing assertion: the two boxes must not intersect.

    The capacities are set to the MEASURED per-branch maxima (5 daily / 4
    intraday), which leave only +3.8 / +1.5 px of slack — so this test is what
    makes those numbers safe to use. A font or matplotlib change that eats the
    slack fails HERE, loudly, instead of silently overprinting two texts in
    every dashboard.
    """
    spec = _spec_for(factor_id, form)
    description, metadata = _band_boxes(spec)
    gap = description.y0 - metadata.y1
    assert gap >= 0.0, (
        f"{factor_id} ({form} form): the description block overruns the "
        f"metadata row by {-gap:.1f} px — both texts are drawn on top of each "
        f"other and the dashboard silently loses content it appears to show. "
        f"If the layout itself changed, re-measure DEFINITION_MAX_LINES_DAILY "
        f"/ DEFINITION_MAX_LINES_INTRADAY rather than relaxing this assertion."
    )


@pytest.mark.parametrize("factor_id", MINUTE_FACTOR_IDS)
@pytest.mark.parametrize("form", FORMS)
def test_the_drawn_description_stays_within_the_line_budget(factor_id, form):
    """The mechanism behind the geometric assertion, pinned separately.

    Two assertions rather than one because they fail for different reasons: this
    one goes red if the bounding stops being applied, the geometric one goes red
    if the budget itself becomes wrong (a font or layout change).
    """
    spec = _spec_for(factor_id, form)
    lines = definition_description_lines(spec)
    assert len(lines) <= definition_max_lines(spec)


def test_the_budget_is_keyed_on_the_spec_form():
    """5 daily / 4 intraday — the split this file exists to keep honest.

    Pinned as numbers, not just as "different": the measured capacities are
    what the geometric test re-checks, and a silent re-unification of the two
    constants would re-open the exec-form overlap.
    """
    assert DEFINITION_MAX_LINES_DAILY == 5
    assert DEFINITION_MAX_LINES_INTRADAY == 4
    spec = build("volume_peak_count_20").spec
    assert definition_max_lines(spec) == DEFINITION_MAX_LINES_DAILY
    assert definition_max_lines(_spec_for("volume_peak_count_20", "exec")) == (
        DEFINITION_MAX_LINES_INTRADAY
    )


def test_a_five_line_description_fits_close_but_is_elided_in_the_exec_form():
    """The teeth of the per-form split, in BOTH directions.

    ``volume_peak_count_20`` wraps to exactly 5 lines: the close form (budget
    5) must render it VERBATIM, the exec form (budget 4) must elide it to
    4 lines with a marked tail. If the budget were measured on the daily
    branch only — the defect being guarded — the first assertion would pass
    and the second would be the 9-of-11 overlap all over again.
    """
    import textwrap

    close_spec = _spec_for("volume_peak_count_20", "close")
    full = textwrap.wrap(close_spec.description, width=DEFINITION_WRAP_WIDTH)
    assert len(full) == DEFINITION_MAX_LINES_DAILY, (
        "the fixture factor no longer wraps to exactly 5 lines — pick another "
        "one for this boundary test rather than weakening it"
    )

    close_lines = definition_description_lines(close_spec)
    assert close_lines == full  # verbatim, no marker

    exec_lines = definition_description_lines(_spec_for("volume_peak_count_20", "exec"))
    assert len(exec_lines) == DEFINITION_MAX_LINES_INTRADAY
    assert "elided" in exec_lines[-1]
    assert str(len(full) - (DEFINITION_MAX_LINES_INTRADAY - 1)) in exec_lines[-1]


@pytest.mark.parametrize("form", FORMS)
def test_an_over_long_description_is_marked_not_silently_dropped(form):
    """Elision must announce itself and say where the full text is.

    Silently showing the first few lines would be the same class of defect as the
    JSON's silent truncation: a document not saying what it left out.
    """
    spec = _spec_for("valley_price_quantile_20", form)
    cap = definition_max_lines(spec)
    lines = definition_description_lines(spec)
    assert len(lines) == cap
    assert "elided" in lines[-1]
    assert "Markdown" in lines[-1]
    # the elided count is real, not decorative
    import textwrap

    full = textwrap.wrap(spec.description, width=DEFINITION_WRAP_WIDTH)
    assert str(len(full) - (cap - 1)) in lines[-1]


@pytest.mark.parametrize("form", FORMS)
def test_a_short_description_is_untouched(form):
    """A factor that fits must be rendered verbatim — no marker, no elision."""
    import textwrap

    spec = _spec_for("jump_amount_corr_20", form)
    lines = definition_description_lines(spec)
    assert lines == textwrap.wrap(spec.description, width=DEFINITION_WRAP_WIDTH)
    assert "elided" not in " ".join(lines)


def test_the_correction_marker_shares_the_header_row_and_does_not_wrap():
    """The correction marker is a short header-row badge, not a wrapping block.

    If it ever grows into prose it would start colliding the same way the
    description did, so its shape is pinned here rather than left to taste.
    """
    from analytics.eval.figures import _correction_marker

    marker = _correction_marker(build("jump_amount_corr_20").spec)
    assert marker and "\n" not in marker
    assert len(marker) < 60
