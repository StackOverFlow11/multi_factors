"""The FACTOR DEFINITION band must not draw its text on top of itself.

WHAT THIS CATCHES, AND WHY A HUMAN LOOKING AT THE PNG DOES NOT COUNT
--------------------------------------------------------------------
The band anchors its pieces at FIXED axes fractions: the description at y=0.62
(va="top") and the metadata row at y=0.20/0.10 (va="bottom"). Nothing bounds the
description, so a long one simply keeps wrapping downward and is DRAWN OVER the
metadata — both texts become unreadable, and the artifact silently loses content
it appears to show. Measured with the real renderer before the fix, six of the
eleven shipped minute factors overlapped (``valley_price_quantile_20`` by 291 px
at 25 wrapped lines against the slot).

That is a standing defect rather than a regression: ``analytics/eval/figures.py``
has exactly one commit in its history, so every dashboard ever produced was drawn
by this code.

The check is geometric, not visual: render the band on the REAL dashboard
geometry, ask matplotlib for each Text's window extent, and assert the boxes do
not intersect. A person squinting at a PNG is not a guard, and neither is a
substring match on anything — the failure mode is pixels landing on pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from analytics.eval.figures import (  # noqa: E402
    DEFINITION_MAX_LINES,
    DEFINITION_WRAP_WIDTH,
    _definition_band,
    definition_description_lines,
)
from factors.compute.minute import binding as binding_module  # noqa: E402
from factors.registry import build  # noqa: E402

#: Every minute factor that reaches a dashboard. Derived from the binding tables
#: (plus the deferred one) rather than listed, so a new factor is covered without
#: anyone remembering to add it here.
MINUTE_FACTOR_IDS = tuple(
    sorted(
        {cls().name for cls in binding_module._MINUTE_STREAM_BINDINGS}
        | {"valley_price_quantile_20"}
    )
)


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
def test_the_description_never_overlaps_the_metadata_row(factor_id):
    """The load-bearing assertion: the two boxes must not intersect.

    ``DEFINITION_MAX_LINES`` is set to the MEASURED capacity (5), which leaves
    only ~4 px of slack — so this test is what makes that number safe to use. A
    font or matplotlib change that eats the slack fails HERE, loudly, instead of
    silently overprinting two texts in every dashboard.
    """
    spec = build(factor_id).spec
    description, metadata = _band_boxes(spec)
    gap = description.y0 - metadata.y1
    assert gap >= 0.0, (
        f"{factor_id}: the description block overruns the metadata row by "
        f"{-gap:.1f} px — both texts are drawn on top of each other and the "
        f"dashboard silently loses content it appears to show. If the layout "
        f"itself changed, re-measure DEFINITION_MAX_LINES rather than relaxing "
        f"this assertion."
    )


@pytest.mark.parametrize("factor_id", MINUTE_FACTOR_IDS)
def test_the_drawn_description_stays_within_the_line_budget(factor_id):
    """The mechanism behind the geometric assertion, pinned separately.

    Two assertions rather than one because they fail for different reasons: this
    one goes red if the bounding stops being applied, the geometric one goes red
    if the budget itself becomes wrong (a font or layout change).
    """
    lines = definition_description_lines(build(factor_id).spec)
    assert len(lines) <= DEFINITION_MAX_LINES


def test_an_over_long_description_is_marked_not_silently_dropped():
    """Elision must announce itself and say where the full text is.

    Silently showing the first few lines would be the same class of defect as the
    JSON's silent truncation: a document not saying what it left out.
    """
    spec = build("valley_price_quantile_20").spec
    lines = definition_description_lines(spec)
    assert len(lines) == DEFINITION_MAX_LINES
    assert "elided" in lines[-1]
    assert "Markdown" in lines[-1]
    # the elided count is real, not decorative
    import textwrap

    full = textwrap.wrap(spec.description, width=DEFINITION_WRAP_WIDTH)
    assert str(len(full) - (DEFINITION_MAX_LINES - 1)) in lines[-1]


def test_a_short_description_is_untouched():
    """A factor that fits must be rendered verbatim — no marker, no elision."""
    import textwrap

    spec = build("jump_amount_corr_20").spec
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
