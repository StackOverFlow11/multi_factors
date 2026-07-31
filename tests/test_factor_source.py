"""D6a: the factor-value source wiring (store policy + served-panel assembly).

Two properties carry this step:

* a SYNTHETIC data source can never reach a durable factor-store artifact —
  structurally, not by convention (``qt/factor_source.py`` explains why such
  values are not storable at all); and
* routing a daily runner through the service changes the PATH and not the
  VALUES: what the service returns is reduced back to the panel's own grid, and
  the rows that reduction drops are counted rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from factors.compute.momentum import MomentumFactor
from factors.store import FactorValueStore
from qt import factor_source
from qt.config import load_config
from qt.factor_source import (
    DailyEvalPanelProvider,
    config_declared_sources,
    factor_store_root,
    factor_values,
    is_synthetic_source,
    known_sources,
    open_factor_value_store,
)
from tests.fixtures.panel_factory import make_demo_panel


def _cfg(tmp_path: Path, example_config_path: str, *, source: str = "demo"):
    """The example config with ``source`` and every output dir under tmp_path."""
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    raw["data"]["source"] = source
    raw["output"] = {
        "root_dir": str(out),
        "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"),
        "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"),
        "overwrite": True,
    }
    path = tmp_path / f"cfg_{source}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(path))


# --------------------------------------------------------------------------- #
# storability classification
# --------------------------------------------------------------------------- #
def test_every_config_source_has_a_storability_classification():
    """A new ``data.source`` must declare whether its values are storable.

    The drift guard, not a restatement: the reference side is read off the
    ``DataCfg.source`` Literal, so adding a source without classifying it fails
    here instead of silently inheriting a durable store.
    """
    assert known_sources() == config_declared_sources()


def test_the_two_classes_are_disjoint():
    assert not (factor_source.SYNTHETIC_SOURCES & factor_source.PERSISTABLE_SOURCES)


def test_demo_is_synthetic_and_tushare_is_not():
    assert is_synthetic_source("demo") is True
    assert is_synthetic_source("tushare") is False


def test_an_unclassified_source_is_refused_not_assumed_real():
    with pytest.raises(ValueError, match="storability classification"):
        is_synthetic_source("some_new_vendor")


# --------------------------------------------------------------------------- #
# store root
# --------------------------------------------------------------------------- #
def test_real_store_root_follows_the_run_output_root(tmp_path, example_config_path):
    cfg = _cfg(tmp_path, example_config_path, source="tushare")
    assert factor_store_root(cfg) == str(tmp_path / "artifacts" / "factor_store")


def test_a_synthetic_source_has_no_persistent_store_root(tmp_path, example_config_path):
    """Not a fallback: asking is the error, and the message says why.

    The demo close for a (date, symbol) is built from a counter that starts at
    ``data.start``, so two windows produce different values under one key.
    """
    cfg = _cfg(tmp_path, example_config_path, source="demo")
    with pytest.raises(ValueError, match="data.start"):
        factor_store_root(cfg)


def test_a_synthetic_run_gets_an_ephemeral_store_that_leaves_nothing(
    tmp_path, example_config_path
):
    cfg = _cfg(tmp_path, example_config_path, source="demo")
    with open_factor_value_store(cfg) as store:
        seen = _store_root_of(store)
        assert seen.exists()
        # not under the run's own output tree, so no later run can be served it
        assert tmp_path not in seen.parents
    assert not seen.exists()
    assert not (tmp_path / "artifacts" / "factor_store").exists()


def test_a_real_run_gets_the_shared_store_at_the_derived_root(
    tmp_path, example_config_path
):
    cfg = _cfg(tmp_path, example_config_path, source="tushare")
    with open_factor_value_store(cfg) as store:
        assert _store_root_of(store) == Path(factor_store_root(cfg))


def _store_root_of(store: FactorValueStore) -> Path:
    """The store's on-disk root, via its public ``values_root``."""
    return store.values_root.parent


# --------------------------------------------------------------------------- #
# served panel: reindexed to the panel grid, footprint rows counted
# --------------------------------------------------------------------------- #
def _served(panel, symbols, tmp_path):
    return factor_values(
        [MomentumFactor(window=5)],
        panel,
        symbols,
        store=FactorValueStore(tmp_path / "store"),
        params_by_id={"momentum_5": {"window": 5}},
    )


def test_a_dense_panel_drops_no_footprint_rows(tmp_path):
    panel = make_demo_panel()
    symbols = sorted(set(panel.index.get_level_values("symbol")))
    out = _served(panel, symbols, tmp_path)

    assert out.footprint_rows_dropped == 0
    assert out.frame.index.equals(panel.index)


def test_a_sparse_panel_reindexes_back_and_counts_what_that_dropped(tmp_path):
    """The index-universe geometry: the service answers over the grid it is asked
    about, the panel is smaller, and the reduction is reported."""
    panel = make_demo_panel()
    symbols = sorted(set(panel.index.get_level_values("symbol")))
    dates = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    late = symbols[-1]
    d = panel.index.get_level_values("date")
    s = panel.index.get_level_values("symbol")
    sparse = panel[~((s == late) & (d < dates[10]))]

    out = _served(sparse, symbols, tmp_path)
    factor = MomentumFactor(window=5)

    assert out.footprint_rows_dropped == 10
    assert out.served_rows == len(dates) * len(symbols)
    assert out.frame.index.equals(sparse.index)
    # ... and the SURVIVING VALUES are right, not just the surviving labels.
    # Asserting only the index would accept a reduction that lands the correct
    # index carrying the wrong rows -- and that defect is invisible on a dense
    # panel, where the reduction is a no-op, so no other test in this file can
    # see it (mutation: reducing positionally instead of by label keeps the
    # index assertion and the dense value test green and turns this red).
    pd.testing.assert_frame_equal(
        out.frame,
        factor.compute(sparse).rename(factor.name).to_frame(),
        check_exact=True,
    )


def test_the_served_values_do_not_depend_on_which_store_they_were_filled_into(
    tmp_path,
):
    """Two cold stores, one request: the same values.

    NAMED FOR WHAT IT MEASURES. It was called
    ``test_the_reindex_preserves_every_value_the_service_served``, which it never
    tested: both sides came from the same already-reindexed ``factor_values``
    call, so the comparison reduced to ``frame == frame`` for that property and
    could only ever have caught a difference between the two STORES.

    That misnaming did real damage before it was caught: it was read (by its
    author, then by a reviewer taking the author's word) as evidence that
    reindex value-preservation was pinned, and on that basis nobody added the
    assertion that actually pins it. It is now in
    ``test_a_sparse_panel_reindexes_back_and_counts_what_that_dropped`` -- which
    is where it belongs, next to the reduction it constrains.

    What is left here is worth keeping on its own: a factor value must be a
    function of (factor, panel, universe) alone, so filling store A and filling
    store B must agree. That is the assumption the read-through rests on.
    """
    panel = make_demo_panel()
    symbols = sorted(set(panel.index.get_level_values("symbol")))
    dates = pd.DatetimeIndex(sorted(set(panel.index.get_level_values("date"))))
    d = panel.index.get_level_values("date")
    s = panel.index.get_level_values("symbol")
    sparse = panel[~((s == symbols[-1]) & (d < dates[10]))]

    into_a = _served(sparse, symbols, tmp_path).frame
    into_b = factor_values(
        [MomentumFactor(window=5)],
        sparse,
        symbols,
        store=FactorValueStore(tmp_path / "store2"),
        params_by_id={"momentum_5": {"window": 5}},
    ).frame

    pd.testing.assert_frame_equal(into_a, into_b, check_exact=True)


def test_the_service_values_equal_a_direct_compute_on_the_same_panel(tmp_path):
    """The step's whole claim, on the close view: same values, different path."""
    panel = make_demo_panel()
    symbols = sorted(set(panel.index.get_level_values("symbol")))
    factor = MomentumFactor(window=5)

    legacy = factor.compute(panel).rename(factor.name).to_frame()
    served = _served(panel, symbols, tmp_path).frame

    pd.testing.assert_frame_equal(served, legacy, check_exact=True)


def test_a_panel_cell_the_service_cannot_serve_is_loud(tmp_path):
    """A short cross-section is the sample bias the I5a red line forbids."""
    panel = make_demo_panel()
    symbols = sorted(set(panel.index.get_level_values("symbol")))

    with pytest.raises(ValueError, match="returned no row for"):
        _served(panel, symbols[:-1], tmp_path)  # one panel symbol not requested


# --------------------------------------------------------------------------- #
# end to end: a demo run persists no factor value, and says so
# --------------------------------------------------------------------------- #
def test_a_demo_run_leaves_no_factor_store_behind(tmp_path, example_config_path):
    """The B1 guard at the runner level, not just at the helper.

    ``config/example.yaml``'s demo universe is spelled with REAL A-share tickers
    over real 2024 dates, so a persisted demo value would be addressed by a key
    a later real run over the same cells would hit.
    """
    from qt.pipeline import run_phase0

    cfg_path = _write_cfg_file(tmp_path, example_config_path, source="demo")
    result = run_phase0(str(cfg_path))

    assert not (tmp_path / "artifacts" / "factor_store").exists()
    assert result.factor_path.exists()  # the per-run panel artifact still lands


def test_a_demo_run_discloses_that_its_store_is_ephemeral(tmp_path, example_config_path):
    from qt.pipeline import run_phase0

    cfg_path = _write_cfg_file(tmp_path, example_config_path, source="demo")
    result = run_phase0(str(cfg_path))

    log = result.log_path.read_text(encoding="utf-8")
    assert "factor store: EPHEMERAL" in log


def test_the_run_log_discloses_the_footprint_reduction(tmp_path, example_config_path):
    """Restoring the panel's grid is allowed to be silent about nothing."""
    from qt.pipeline import run_phase0

    cfg_path = _write_cfg_file(tmp_path, example_config_path, source="demo")
    result = run_phase0(str(cfg_path))

    log = result.log_path.read_text(encoding="utf-8")
    assert "all-NaN footprint row(s) outside the market panel" in log


def test_the_persisted_factor_panel_keeps_the_market_panel_grid(
    tmp_path, example_config_path
):
    from qt.pipeline import run_phase0

    cfg_path = _write_cfg_file(tmp_path, example_config_path, source="demo")
    result = run_phase0(str(cfg_path))

    stored = pd.read_parquet(result.factor_path)
    market = pd.read_parquet(result.data_path)
    assert len(stored) == len(market)


def _write_cfg_file(tmp_path: Path, example_config_path: str, *, source: str) -> Path:
    raw = yaml.safe_load(Path(example_config_path).read_text(encoding="utf-8"))
    out = tmp_path / "artifacts"
    raw["data"]["source"] = source
    raw["data"]["end"] = "2024-06-30"  # keep the run small
    raw["output"] = {
        "root_dir": str(out),
        "data_dir": str(out / "data"),
        "factor_dir": str(out / "factors"),
        "report_dir": str(out / "reports"),
        "log_dir": str(out / "logs"),
        "overwrite": True,
    }
    path = tmp_path / "run_cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# mid-migration: the runners D6a does not move must still be able to call
# --------------------------------------------------------------------------- #
#: Runners still on the pre-D6a ``_compute_factor_panel`` path. D6b empties this
#: list; D6d deletes the function. It is spelled out here because those runners
#: need real tushare data, so NOTHING in the suite executes their call — the
#: first D6a draft changed the shared signature under them and every test stayed
#: green (the break was found by reading the call sites, not by a failure).
UNMIGRATED_RUNNERS = ("qt/oos_stability.py", "qt/subset_validation.py")


@pytest.mark.parametrize("module_path", UNMIGRATED_RUNNERS)
def test_an_unmigrated_runners_call_still_matches_the_legacy_signature(module_path):
    """Their call is parsed from source and bound against the LIVE signature."""
    import ast
    import inspect

    from qt import pipeline

    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / module_path).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compute_factor_panel"
    ]
    assert calls, f"{module_path} no longer calls _compute_factor_panel"

    signature = inspect.signature(pipeline._compute_factor_panel)
    for call in calls:
        signature.bind(
            *["<arg>"] * len(call.args),
            **{kw.arg: "<arg>" for kw in call.keywords if kw.arg},
        )


def test_the_legacy_entry_point_still_computes(tmp_path, example_config_path):
    """Real coverage for the path those runners are still on."""
    from qt.pipeline import _compute_factor_panel

    cfg = _cfg(tmp_path, example_config_path, source="demo")
    panel = make_demo_panel()
    factor = MomentumFactor(window=5)
    logger = __import__("logging").getLogger("test.legacy_entry")

    out = _compute_factor_panel(cfg, panel, [factor], logger)

    pd.testing.assert_series_equal(out[factor.name], factor.compute(panel).rename(factor.name))


# --------------------------------------------------------------------------- #
# author-once: the eval wiring re-exports this provider, it does not copy it
# --------------------------------------------------------------------------- #
def test_the_eval_provider_is_this_provider():
    from qt import factor_eval_providers

    assert factor_eval_providers.DailyEvalPanelProvider is DailyEvalPanelProvider
