"""D6c: ``_serve_score_panel`` — the intraday runners' factor-service score source.

Network-free: the factor service is SPIED (monkeypatched
``factors.service.panel`` / ``cross_section``) and the value store is a dummy
object, so no cache, feed or token is touched. Covers:

* the config/spec guard — ``decision_time`` / ``session_open`` / ``data_lag``
  must equal the registered factor's spec (the store key carries no cutoff
  dimension, so a mismatch would be served the spec-default values silently);
* the ``score_feature`` -> factor id mapping, incl. the three Literal values
  D6c PR-1 catalogued as deliberately NOT registered (readable error);
* the fill-geometry split — ONE ``panel`` call for the tail runner's
  near-continuous window vs PER-DAY ``cross_section`` for the group runner's
  sparse anchors (a range fill would read the 5-year convex hull);
* the shape restoration — the served grid is reindexed to EXACTLY the legacy
  visible-cell row set (all-NaN footprint rows dropped and logged), a visible
  cell the service did not answer is a loud error, and a fully post-cutoff
  input yields the schema-shaped empty score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from data.clean.intraday_schema import normalize_intraday_bars
from factors import service as factor_service
from qt.config import load_config
from qt.intraday_tail_framework import _serve_score_panel, _visible_score_cells

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_I5C_CONFIG = _CONFIG_DIR / "phase_i5c_mmp_minute_factor.yaml"

_MMP_ID = "intraday_mmp20_ew_0930_1450"
_RET_ID = "intraday_ret_0930_1450"
_DAY = "2024-01-02"
_SYMS = ["000001.SZ", "000002.SZ"]
_LOGGER = logging.getLogger("test.serve_score_panel")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _mbars(specs: list[tuple]) -> pd.DataFrame:
    """specs = [(time_str, symbol, open, high, low, close, volume), ...]."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([s[0] for s in specs]),
            "symbol": [s[1] for s in specs],
            "open": [s[2] for s in specs],
            "high": [s[3] for s in specs],
            "low": [s[4] for s in specs],
            "close": [s[5] for s in specs],
            "volume": [s[6] for s in specs],
            "amount": [s[5] * s[6] for s in specs],
        }
    )
    return normalize_intraday_bars(df, freq="1min", data_lag="1min")


def _session_specs(symbol: str, n: int = 5, day: str = _DAY) -> list[tuple]:
    base = pd.Timestamp(f"{day} 09:31:00")
    return [
        (str(base + pd.Timedelta(minutes=i)), symbol, 10.0, 10.5, 9.5,
         10.1 + 0.01 * i, 100.0)
        for i in range(n)
    ]


def _bars_two_symbols() -> pd.DataFrame:
    """Visible bars for both symbols + one POST-cutoff bar (must not add a cell)."""
    specs = _session_specs("000001.SZ") + _session_specs("000002.SZ")
    specs.append((f"{_DAY} 14:55:00", "000001.SZ", 10.0, 10.5, 9.5, 10.2, 100.0))
    return _mbars(specs)


def _cfg(**intraday_overrides):
    cfg = load_config(str(_I5C_CONFIG))
    assert cfg.intraday is not None
    for key, value in intraday_overrides.items():
        setattr(cfg.intraday, key, value)
    return cfg


def _frame(rows: list[tuple], factor_id: str) -> pd.DataFrame:
    """rows = [(date_str, symbol, value), ...] as a one-column service frame."""
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for d, s, _ in rows], names=["date", "symbol"]
    )
    return pd.DataFrame({factor_id: [v for _, _, v in rows]}, index=index)


def _spy(monkeypatch, served: pd.DataFrame, calls: list) -> None:
    """Route the runner's service calls to ``served`` and record them."""
    def fake_panel(factor_ids, universe, decisions, **kwargs):
        calls.append(("panel", tuple(factor_ids), len(decisions)))
        return served

    def fake_cross_section(factor_ids, universe, decision, **kwargs):
        calls.append(("cross_section", tuple(factor_ids), 1))
        d = pd.Timestamp(decision.date).normalize()
        dates = served.index.get_level_values("date")
        return served[dates == d]

    monkeypatch.setattr(factor_service, "panel", fake_panel)
    monkeypatch.setattr(factor_service, "cross_section", fake_cross_section)


# --------------------------------------------------------------------------- #
# guard: config must match the registered factor's spec
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field,value",
    [
        ("decision_time", "14:45:00"),
        ("session_open", "09:31:00"),
        ("data_lag", "2min"),
    ],
)
def test_guard_rejects_config_that_mismatches_the_factor_spec(field, value):
    cfg = _cfg(**{field: value})
    with pytest.raises(ValueError, match="does not match the registered factor"):
        _serve_score_panel(cfg, _bars_two_symbols(), _SYMS, object(), _LOGGER)


def test_guard_message_names_the_mismatched_fields():
    cfg = _cfg(decision_time="14:45:00", data_lag="2min")
    with pytest.raises(ValueError, match="decision_time") as exc_info:
        _serve_score_panel(cfg, _bars_two_symbols(), _SYMS, object(), _LOGGER)
    assert "data_lag" in str(exc_info.value)
    assert "session_open" not in str(exc_info.value)  # the matching field is silent


# --------------------------------------------------------------------------- #
# mapping: score_feature -> factor id
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,factor_id", [("ret", _RET_ID), ("mmp_ew", _MMP_ID)]
)
def test_score_feature_maps_to_the_registered_factor(monkeypatch, key, factor_id):
    cfg = _cfg(score_feature=key)
    bars = _bars_two_symbols()
    calls: list = []
    _spy(monkeypatch, _frame([(_DAY, "000001.SZ", 0.01), (_DAY, "000002.SZ", 0.02)],
                             factor_id), calls)
    score, col = _serve_score_panel(cfg, bars, _SYMS, object(), _LOGGER)
    assert col == factor_id
    assert calls[0][1] == (factor_id,)  # exactly one id, no prefix matching
    assert score.name == "score"


@pytest.mark.parametrize("key", ["realized_vol", "vwap", "last30m_ret"])
def test_unregistered_score_feature_is_a_readable_error(key):
    cfg = _cfg(score_feature=key)
    with pytest.raises(ValueError, match="no D6c-registered factor"):
        _serve_score_panel(cfg, _bars_two_symbols(), _SYMS, object(), _LOGGER)


# --------------------------------------------------------------------------- #
# fill geometry: one panel call vs per-day cross_section
# --------------------------------------------------------------------------- #
def test_tail_geometry_serves_one_panel_call_over_all_decisions(monkeypatch):
    bars = _mbars(
        _session_specs("000001.SZ", day="2024-01-02")
        + _session_specs("000001.SZ", day="2024-01-03")
        + _session_specs("000001.SZ", day="2024-01-04")
    )
    calls: list = []
    _spy(monkeypatch, _frame(
        [("2024-01-02", "000001.SZ", 0.01), ("2024-01-03", "000001.SZ", 0.02),
         ("2024-01-04", "000001.SZ", 0.03)], _MMP_ID), calls)
    score, _ = _serve_score_panel(_cfg(), bars, ["000001.SZ"], object(), _LOGGER)
    assert calls == [("panel", (_MMP_ID,), 3)]  # ALL decision dates in ONE call
    assert len(score) == 3


def test_sparse_geometry_serves_one_cross_section_per_anchor(monkeypatch):
    bars = _mbars(
        _session_specs("000001.SZ", day="2024-01-31")
        + _session_specs("000001.SZ", day="2024-02-29")
        + _session_specs("000001.SZ", day="2024-03-29")
    )
    calls: list = []
    _spy(monkeypatch, _frame(
        [("2024-01-31", "000001.SZ", 0.01), ("2024-02-29", "000001.SZ", 0.02),
         ("2024-03-29", "000001.SZ", 0.03)], _MMP_ID), calls)
    score, _ = _serve_score_panel(
        _cfg(), bars, ["000001.SZ"], object(), _LOGGER, sparse_anchors=True
    )
    assert calls == [("cross_section", (_MMP_ID,), 1)] * 3  # per day, never a range
    assert len(score) == 3


# --------------------------------------------------------------------------- #
# shape restoration to the legacy visible-cell row set
# --------------------------------------------------------------------------- #
def test_shape_restored_to_visible_cells_and_footprint_dropped(monkeypatch, caplog):
    bars = _bars_two_symbols()
    served = _frame(
        [
            (_DAY, "000001.SZ", 0.011),
            (_DAY, "000002.SZ", float("nan")),  # a computed-empty cell stays NaN
            (_DAY, "000003.SZ", float("nan")),  # footprint: covered, no bars
            ("2024-01-03", "000001.SZ", float("nan")),  # footprint: barless date
        ],
        _MMP_ID,
    )
    calls: list = []
    _spy(monkeypatch, served, calls)
    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        score, col = _serve_score_panel(cfg=_cfg(), bars=bars,
                                        symbols_covered=_SYMS + ["000003.SZ"],
                                        store=object(), logger=_LOGGER)
    expected = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(_DAY), "000001.SZ"), (pd.Timestamp(_DAY), "000002.SZ")],
        names=["date", "symbol"],
    )
    assert score.index.equals(expected)  # EXACTLY the legacy row set, sorted
    assert score.loc[(pd.Timestamp(_DAY), "000001.SZ")] == pytest.approx(0.011)
    assert pd.isna(score.loc[(pd.Timestamp(_DAY), "000002.SZ")])
    assert col == _MMP_ID
    assert "2 footprint row(s)" in caplog.text  # the drop is logged, not invisible


def test_post_cutoff_only_symbol_has_no_cell():
    bars = _mbars(
        _session_specs("000001.SZ")
        + [(f"{_DAY} 14:55:00", "000004.SZ", 10.0, 10.5, 9.5, 10.2, 100.0)]
    )
    cells = _visible_score_cells(bars, "14:50:00")
    assert list(cells) == [(pd.Timestamp(_DAY), "000001.SZ")]


def test_unserved_visible_cell_is_a_loud_error(monkeypatch):
    bars = _bars_two_symbols()
    calls: list = []
    # The spy answers only ONE of the two visible cells — a silently short
    # score panel would drop a name from the cross-section.
    _spy(monkeypatch, _frame([(_DAY, "000001.SZ", 0.011)], _MMP_ID), calls)
    with pytest.raises(ValueError, match="no row for 1"):
        _serve_score_panel(_cfg(), bars, _SYMS, object(), _LOGGER)


def test_no_visible_bars_returns_empty_score_without_calling_the_service(monkeypatch):
    bars = _mbars(
        [(f"{_DAY} 14:55:00", "000001.SZ", 10.0, 10.5, 9.5, 10.2, 100.0)]
    )

    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the service must not be called for an empty request")

    monkeypatch.setattr(factor_service, "panel", fail)
    monkeypatch.setattr(factor_service, "cross_section", fail)
    score, col = _serve_score_panel(_cfg(), bars, _SYMS, object(), _LOGGER)
    assert score.empty and score.name == "score"
    assert list(score.index.names) == ["date", "symbol"]
    assert col == _MMP_ID
