"""D6b: the phase3 cell cores get their factor values from the factor service.

``_run_oos_cell`` and ``_run_subset_cell`` (with ``qt.robustness`` inheriting
through the former) used to call the pre-D6a ``_compute_factor_panel``; they now
go through ``_serve_factor_panel`` behind ``open_factor_value_store``, exactly
like phase0/phase2 since D6a. Three properties are pinned here, all on the demo
path (the cell cores carry no tushare gate — that lives in
``check_oos_preconditions``, one level up):

* WIRING, behaviorally: the cell actually reaches ``_serve_factor_panel`` and
  the run log carries the service's own line. No AST — "pin the function, miss
  the wiring" is a failure shape this project has been bitten by.
* EQUIVALENCE: the served factor panel equals, BITWISE, what the legacy path
  computes on the same panel. The comparison inlines ``factor.compute`` rather
  than calling ``_compute_factor_panel`` by name — a named call would make this
  file a legacy CALLER and trip the census in
  ``tests/test_legacy_factor_panel_callers.py``; that file's
  ``test_the_legacy_entry_point_still_computes`` already pins the legacy
  function itself equal to this same comprehension, so the triangle closes.
* B1: a demo (synthetic-source) cell leaves NO ``factor_store/`` under
  ``output.root_dir`` — the ephemeral store leaves no durable artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from qt.config import load_config
from qt.factor_source import FACTOR_STORE_DIRNAME
from qt.pipeline import _make_logger, _serve_factor_panel


def _cell_cfg(tmp_path: Path, example_config_path: str, extra: dict):
    """The example config, demo source, short window, outputs under tmp_path."""
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    raw["data"]["source"] = "demo"
    raw["data"]["start"] = "2024-01-01"
    raw["data"]["end"] = "2024-06-30"
    raw["output"] = {
        "root_dir": str(out),
        "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"),
        "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"),
        "overwrite": True,
    }
    raw.update(extra)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(path))


_OOS_EXTRA = {
    "oos": {"split_date": "2024-04-01"},
    # small walk-forward window so the short demo window still fits weights
    "alpha": {
        "model": "ic_weighted",
        "params": {"window": 20, "min_periods": 5, "horizon": 1},
    },
}

_SUBSET_EXTRA = {
    **_OOS_EXTRA,
    "subset_validation": {
        "groups": [{"label": "all", "factors": ["momentum_20"]}],
        "cost_scenarios": [{"label": "base", "fee_multiplier": 1.0}],
    },
}


def _equivalence_spy(calls: list):
    """Wrap the REAL ``_serve_factor_panel``: delegate, then pin equivalence.

    The delegation is what makes this a behavioral test — the service path is
    genuinely walked (and its log line is asserted separately). The wrapper
    fires only if the cell is wired to the service entry point; a cell reverted
    to the legacy path never calls it, which the ``calls`` assertion turns red.
    """

    def spy(cfg, panel, factors, symbols, logger, *, store):
        served = _serve_factor_panel(cfg, panel, factors, symbols, logger, store=store)
        calls.append((cfg, panel, factors))
        # The legacy computation, inlined (module docstring: why not by name).
        legacy = pd.concat(
            [factor.compute(panel).rename(factor.name) for factor in factors], axis=1
        )
        pd.testing.assert_frame_equal(served, legacy, check_exact=True)
        return served

    return spy


def _assert_service_walked(log_path: Path) -> None:
    log = log_path.read_text(encoding="utf-8")
    assert "served by the factor service" in log
    assert "factor store: EPHEMERAL" in log


def test_oos_cell_sources_factor_values_from_the_service(
    tmp_path, example_config_path, monkeypatch
):
    import qt.oos_stability as oos

    cfg = _cell_cfg(tmp_path, example_config_path, _OOS_EXTRA)
    calls: list = []
    monkeypatch.setattr(oos, "_serve_factor_panel", _equivalence_spy(calls))
    log_path = Path(cfg.output.log_dir) / "oos_cell.log"
    logger = _make_logger(log_path, name="test.phase3_wiring.oos")

    result = oos._run_oos_cell(cfg, logger, log_path)

    assert calls, "the OOS cell never reached _serve_factor_panel — wiring lost"
    assert result.factor_names == ("momentum_20",)
    _assert_service_walked(log_path)
    assert not (Path(cfg.output.root_dir) / FACTOR_STORE_DIRNAME).exists()


def test_subset_cell_sources_factor_values_from_the_service(
    tmp_path, example_config_path, monkeypatch
):
    import qt.subset_validation as subset

    cfg = _cell_cfg(tmp_path, example_config_path, _SUBSET_EXTRA)
    calls: list = []
    monkeypatch.setattr(subset, "_serve_factor_panel", _equivalence_spy(calls))
    log_path = Path(cfg.output.log_dir) / "subset_cell.log"
    logger = _make_logger(log_path, name="test.phase3_wiring.subset")

    result = subset._run_subset_cell(cfg, logger)

    assert calls, "the subset cell never reached _serve_factor_panel — wiring lost"
    assert result.factor_names == ("momentum_20",)
    _assert_service_walked(log_path)
    assert not (Path(cfg.output.root_dir) / FACTOR_STORE_DIRNAME).exists()
