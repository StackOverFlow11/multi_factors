"""P3-4: robustness matrix over the P3-3 OOS cell core (network-free).

Locks the matrix contract:
  * the matrix config validates (universes × windows, unique labels, skips must
    reference declared cells, at least one cell must remain);
  * per-cell configs are derived from the base verbatim except universe /
    window / split / output_name — every cell reuses the SAME P3-3 cell core
    (holding-window slicing etc.) and the matrix runner shares the single-run
    guards (incl. the ic_weighted fake-comparison guard);
  * multi-cell aggregation never mixes cells' numbers;
  * the report carries universe/window/fold labels, skipped-cell disclosure,
    boundary/fallback/sign-consistency columns, the caveat, and no secret;
  * the single-run P3-3 behaviour is unchanged (its own test file still passes).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from qt.config import ConfigError, load_config
from qt.robustness import (
    cell_label,
    derive_cell_config,
    iter_cells,
    run_phase3_robustness,
    summarize_matrix,
)

_MATRIX_CONFIG = str(
    Path(__file__).resolve().parents[1]
    / "config" / "phase3_real_robustness_matrix.yaml"
)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_matrix_config_validates():
    cfg = load_config(_MATRIX_CONFIG)
    assert cfg.robustness is not None
    assert "000016.SH" in cfg.robustness.universes
    assert "000300.SH" in cfg.robustness.universes
    labels = [w.label for w in cfg.robustness.windows]
    assert len(labels) == len(set(labels))
    assert cfg.alpha.model == "ic_weighted"


def _mutate(tmp_path, **patch):
    raw = yaml.safe_load(Path(_MATRIX_CONFIG).read_text(encoding="utf-8"))
    for key, value in patch.items():
        raw["robustness"][key] = value
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def test_matrix_config_rejects_split_outside_window(tmp_path):
    bad = [{"label": "w", "start": "2022-07-01", "end": "2024-06-30",
            "split": "2024-07-01"}]
    with pytest.raises(ConfigError, match="STRICTLY inside"):
        load_config(_mutate(tmp_path, windows=bad))


def test_matrix_config_rejects_unknown_skip(tmp_path):
    with pytest.raises(ConfigError, match="unknown"):
        load_config(_mutate(
            tmp_path, skip_cells=[{"universe": "999999.SH", "window": "2022-2024"}]
        ))


def test_matrix_config_rejects_all_cells_skipped(tmp_path):
    raw = yaml.safe_load(Path(_MATRIX_CONFIG).read_text(encoding="utf-8"))
    skips = [
        {"universe": u, "window": w["label"]}
        for u in raw["robustness"]["universes"]
        for w in raw["robustness"]["windows"]
    ]
    with pytest.raises(ConfigError, match="at least one cell"):
        load_config(_mutate(tmp_path, skip_cells=skips))


# --------------------------------------------------------------------------- #
# cell derivation + enumeration
# --------------------------------------------------------------------------- #
def test_iter_cells_excludes_skips_and_keeps_order():
    cfg = load_config(_MATRIX_CONFIG)
    cells = list(iter_cells(cfg))
    labels = [cell_label(u, w) for u, w in cells]
    # config: 2 universes x 2 windows - 1 skip (CSI300/2020-2022) = 3 cells
    assert labels == [
        "000016.SH|2020-2022", "000016.SH|2022-2024", "000300.SH|2022-2024",
    ]


def test_derive_cell_config_swaps_only_cell_fields():
    cfg = load_config(_MATRIX_CONFIG)
    window = cfg.robustness.windows[0]  # 2020-2022
    cell = derive_cell_config(cfg, "000300.SH", window)
    # swapped-in cell identity
    assert cell.universe.index_code == "000300.SH"
    assert cell.data.start == window.start and cell.data.end == window.end
    assert cell.oos.split_date == window.split
    assert cell.data.output_name != cfg.data.output_name  # no parquet collision
    # everything else identical to the base (no drift into the cells)
    assert cell.factors == cfg.factors
    assert cell.alpha == cfg.alpha
    assert cell.processing == cfg.processing
    assert cell.portfolio == cfg.portfolio
    assert cell.cost == cfg.cost
    # derived cell still passes the OOS preconditions (ic_weighted guard etc.)
    from qt.oos_stability import check_oos_preconditions

    check_oos_preconditions(cell, runner="run-phase3-robustness")


def test_derive_cell_config_output_names_unique_per_cell():
    cfg = load_config(_MATRIX_CONFIG)
    names = {
        derive_cell_config(cfg, u, w).data.output_name
        for u in cfg.robustness.universes
        for w in cfg.robustness.windows
    }
    assert len(names) == 4  # every (universe, window) gets its own panel file


# --------------------------------------------------------------------------- #
# aggregation — no cross-cell mixing
# --------------------------------------------------------------------------- #
def _fake_cell(ic_test_mean: float, consistent: bool, ic_beats_eq: bool):
    perf = lambda a: {"annual_return": a, "volatility": 0.15, "sharpe": -0.2,  # noqa: E731
                      "max_drawdown": -0.1, "avg_turnover": 0.8, "n_rebalances": 11}
    eq_test, ic_test = (-0.05, -0.02) if ic_beats_eq else (-0.02, -0.05)
    stats = {
        "train": {"ic_mean": 0.01 if consistent else -0.01,
                  "ic_ir": 0.1, "hit_rate": 0.5, "n": 100},
        "test": {"ic_mean": ic_test_mean, "ic_ir": 0.1, "hit_rate": 0.5, "n": 100},
    }
    return {
        "performance": {
            "equal_weight": {"train": perf(-0.06), "test": perf(eq_test)},
            "ic_weighted": {"train": perf(-0.03), "test": perf(ic_test)},
        },
        "ic_stats": {"momentum_20": stats,
                     "combo_ic_weighted": stats},
        "sign_consistency": {"momentum_20": consistent,
                             "combo_ic_weighted": consistent},
        "sign_flips": {"momentum_20": 3},
        "n_scored": 200, "n_fallback": 20,
    }


def test_summarize_matrix_attributes_cells_correctly():
    cells = {
        "A|w1": _fake_cell(ic_test_mean=0.02, consistent=True, ic_beats_eq=True),
        "B|w1": _fake_cell(ic_test_mean=-0.03, consistent=False, ic_beats_eq=False),
    }
    summary = summarize_matrix(cells)
    mom = summary["series"]["momentum_20"]
    assert mom["n_cells"] == 2
    assert mom["test_ic_positive"] == 1      # only cell A
    assert mom["sign_consistent"] == 1       # only cell A
    assert summary["ic_beats_eq_test"] == 1  # only cell A
    assert summary["n_cells"] == 2


def test_summarize_matrix_single_cell_not_diluted():
    summary = summarize_matrix(
        {"A|w1": _fake_cell(ic_test_mean=0.02, consistent=True, ic_beats_eq=True)}
    )
    mom = summary["series"]["momentum_20"]
    assert mom["n_cells"] == 1 and mom["test_ic_positive"] == 1


# --------------------------------------------------------------------------- #
# runner guards (no network)
# --------------------------------------------------------------------------- #
def test_matrix_runner_rejects_demo_source(example_config_path):
    with pytest.raises(ValueError, match="tushare|REAL"):
        run_phase3_robustness(example_config_path)


def test_matrix_runner_requires_robustness_section(tmp_path):
    raw = yaml.safe_load(Path(_MATRIX_CONFIG).read_text(encoding="utf-8"))
    raw.pop("robustness")
    p = tmp_path / "no_matrix.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="robustness"):
        run_phase3_robustness(str(p))


def test_matrix_runner_rejects_non_ic_weighted_alpha(tmp_path):
    raw = yaml.safe_load(Path(_MATRIX_CONFIG).read_text(encoding="utf-8"))
    raw["alpha"] = {"model": "equal_weight", "params": {}}
    p = tmp_path / "eq.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="ic_weighted"):
        run_phase3_robustness(str(p))


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #
def _synthetic_matrix_result():
    from qt.robustness import RobustnessResult
    from tests.test_oos_stability import _synthetic_oos_result

    base = _synthetic_oos_result()
    cell = dataclasses.replace(
        base,
        downgrades=(
            "DATA PATH = REAL tushare: PIT index membership (000016.SH); ...",
            "shared-disclosure line",
        ),
    )
    cells = {
        "000016.SH|2022-2024": cell,
        "000300.SH|2022-2024": dataclasses.replace(
            cell, n_fallback=33, sign_flips={"a": 5, "b": 1},
            downgrades=(
                "DATA PATH = REAL tushare: PIT index membership (000300.SH); ...",
                "shared-disclosure line",
            ),
        ),
    }
    plain = {
        label: {
            "performance": c.performance,
            "ic_stats": c.ic_stats,
            "sign_consistency": c.sign_consistency,
            "sign_flips": c.sign_flips,
            "n_scored": c.n_scored,
            "n_fallback": c.n_fallback,
        }
        for label, c in cells.items()
    }
    return RobustnessResult(
        config=load_config(_MATRIX_CONFIG),
        elapsed_seconds=1000.0,
        cells=cells,
        cell_runtimes={"000016.SH|2022-2024": 900.0, "000300.SH|2022-2024": 100.0},
        skipped_cells=("000300.SH|2020-2022",),
        summary=summarize_matrix(plain),
        report_path=Path("artifacts/reports/phase3_robustness_matrix.md"),
        log_path=Path("artifacts/logs/run_phase3_robustness.log"),
    )


def test_render_matrix_report_covers_cells_summary_and_caveat():
    from qt.robustness import render_robustness_matrix

    md = render_robustness_matrix(_synthetic_matrix_result())
    # universe x window cell labels + skipped-cell disclosure
    assert "000016.SH|2022-2024" in md and "000300.SH|2022-2024" in md
    assert "000300.SH|2020-2022" in md and "skipped" in md.lower()
    # per-cell content: performance models, IC stability, weights diagnostics
    assert "equal_weight" in md and "ic_weighted" in md
    assert "sign consistency" in md.lower() or "sign-consistency" in md.lower()
    assert "sign flip" in md.lower() and "fallback" in md.lower()
    # boundary disclosure survives into the matrix report
    assert "boundary" in md.lower()
    # cross-cell summary matrix
    assert "cells" in md.lower()
    # caveat
    assert "not a" in md.lower() and "claim" in md.lower()


def test_render_matrix_downgrades_cover_all_universes():
    """The DOWNGRADES/caveats section must disclose EVERY run universe — never
    only the first cell's universe-specific lines (MEDIUM review finding)."""
    from qt.robustness import render_robustness_matrix

    md = render_robustness_matrix(_synthetic_matrix_result())
    caveats = md.split("## DOWNGRADES")[1]
    # matrix-level scope line with run cells / skipped cells / universes / windows
    assert "MATRIX SCOPE" in caveats
    assert "000016.SH|2022-2024" in caveats and "000300.SH|2022-2024" in caveats
    assert "000300.SH|2020-2022" in caveats  # the skipped cell, in the scope line
    # BOTH universes' membership disclosures present (union, not first-cell-only)
    assert "PIT index membership (000016.SH)" in caveats
    assert "PIT index membership (000300.SH)" in caveats
    # the shared line is deduplicated (union, not concatenation)
    assert caveats.count("shared-disclosure line") == 1
    # the per-cell caveat no longer reads as a single-universe claim
    assert "one index, two years" not in md
    assert "multiple" in caveats.lower()


def test_render_matrix_report_leaks_no_secret():
    from qt.robustness import render_robustness_matrix

    result = _synthetic_matrix_result()
    md = render_robustness_matrix(result)
    assert result.config.data.external_secret_file not in md
    assert "token" not in md.lower()
