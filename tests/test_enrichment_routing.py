"""D6a-2: panel enrichment is routed off the factors' DECLARED requires.

The retired dispatch tested ``isinstance(f, FinancialFactor)`` /
``isinstance(f, ValueFactor)``; the current one reads ``spec.requires`` via
``factors.registry.requirements_of`` and routes source endpoint -> enricher
(``fina_indicator`` field -> as-of financial column; ``daily_basic`` pe/pb ->
derived ``value_ep``/``value_bp`` column). Three evidence directions, all
network-free (the tushare feeds are fakes):

1. BEHAVIOUR EQUIVALENCE — the trigger set the isinstance rule selected
   (roe/netprofit_yoy/grossprofit_margin -> fina; value_ep/value_bp -> value)
   produces a panel IDENTICAL to an independent oracle below that re-selects
   with the LEGACY rule and re-applies the enrichment arithmetic from the
   ``data.clean`` primitives (assert_frame_equal + a byte-level csv compare).
2. DECLARATION-DRIVEN — a brand-new Factor subclass that is NEITHER a
   FinancialFactor NOR a ValueFactor but DECLARES the endpoint requirement is
   enriched with zero pipeline changes. Under the isinstance rule this was
   impossible; it is the thing declarative routing buys.
3. MUTATION — editing the declaration flips the behaviour (the feed is not
   even called), in both directions, proving the DECLARATION drives the
   routing rather than a second hardcoded table.
"""

from __future__ import annotations

import logging

import pandas as pd
import yaml

from data.availability_policy import DAILY_BASIC, FINA_INDICATOR, MARKET_DAILY
from data.clean.pit_financials import asof_financials
from factors.base import Factor
from factors.compute.candidates import ValueFactor
from factors.compute.financial import FinancialFactor
from factors.compute.momentum import MomentumFactor
from factors.requires import PanelField
from factors.spec import FactorSpec
from qt.config import load_config
from qt.pipeline import _maybe_enrich_financials, _maybe_enrich_value

_LOGGER = logging.getLogger("test.enrichment_routing")


# --------------------------------------------------------------------------- #
# fixtures: config / panel / fake feeds
# --------------------------------------------------------------------------- #
def _cfg(tmp_path, example_config_path, source="tushare"):
    base = yaml.safe_load(open(example_config_path, encoding="utf-8"))
    base["factors"] = [{"name": "momentum_20", "enabled": True, "params": {}}]
    base["data"]["source"] = source
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return load_config(str(path))


def _panel():
    dates = pd.bdate_range("2024-01-08", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    close = [100.0 + i + 50.0 * j for i in range(10) for j in range(2)]
    return pd.DataFrame({"close": close}, index=idx)


_FINA_ROWS = pd.DataFrame(
    {
        "symbol": ["A", "A", "B"],
        "ann_date": ["20240105", "20240115", "20240110"],
        "end_date": ["20231231", "20240331", "20231231"],
        "roe": [10.0, 12.0, 8.0],
        "netprofit_yoy": [5.0, 7.0, 3.0],
        "grossprofit_margin": [30.0, 32.0, 25.0],
    }
)

_RATIO_ROWS = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-08", "2024-01-08", "2024-01-09", "2024-01-09"]
        ),
        "symbol": ["A", "B", "A", "B"],
        "pe": [20.0, 10.0, -5.0, 25.0],   # the negative pe must become NaN
        "pb": [2.0, 0.0, 4.0, 5.0],       # the zero pb must become NaN
    }
)


class _FakeFinaFeed:
    calls: list[list[str]] = []

    def __init__(self, *args, **kwargs):
        pass

    def get_fina_indicator(self, symbols, start, end, fields=None):
        type(self).calls.append(list(fields or []))
        return _FINA_ROWS.copy()


class _FakeCovariatesFeed:
    calls: int = 0

    def __init__(self, *args, **kwargs):
        pass

    def value_ratios(self, symbols, start, end):
        type(self).calls += 1
        return _RATIO_ROWS.copy()


# --------------------------------------------------------------------------- #
# independent oracles: the LEGACY isinstance selection + the enrichment math
# --------------------------------------------------------------------------- #
def _legacy_fina_oracle(panel, factors):
    fields = [f.name for f in factors if isinstance(f, FinancialFactor)]
    expected = panel.copy()
    aligned = asof_financials(panel.index, _FINA_ROWS, fields)
    for field in fields:
        expected[field] = aligned[field]
    return expected


def _legacy_value_oracle(panel, factors):
    fields = [f.name for f in factors if isinstance(f, ValueFactor)]
    expected = panel.copy()
    r = _RATIO_ROWS.copy()
    r["symbol"] = r["symbol"].astype(str)
    r = r.set_index(["date", "symbol"]).sort_index()
    inverted = {
        "value_ep": 1.0 / r["pe"].where(r["pe"] > 0),
        "value_bp": 1.0 / r["pb"].where(r["pb"] > 0),
    }
    for field in fields:
        expected[field] = inverted[field].reindex(expected.index)
    return expected


# --------------------------------------------------------------------------- #
# 1. behaviour equivalence on the legacy trigger set
# --------------------------------------------------------------------------- #
def test_fina_enrichment_matches_legacy_selection_byte_for_byte(
    tmp_path, example_config_path, monkeypatch
):
    _FakeFinaFeed.calls = []
    monkeypatch.setattr("qt.pipeline.TushareFinancialFeed", _FakeFinaFeed)
    cfg = _cfg(tmp_path, example_config_path)
    panel = _panel()
    factors = [
        MomentumFactor(20),  # non-trigger: must not be picked up
        FinancialFactor("roe"),
        FinancialFactor("netprofit_yoy"),
        FinancialFactor("grossprofit_margin"),
    ]
    enriched = _maybe_enrich_financials(cfg, panel, ["A", "B"], factors, _LOGGER)
    # ONE batched fetch of exactly the declared fields, in factor order.
    assert _FakeFinaFeed.calls == [["roe", "netprofit_yoy", "grossprofit_margin"]]
    expected = _legacy_fina_oracle(panel, factors)
    pd.testing.assert_frame_equal(enriched, expected, check_exact=True)
    assert enriched.to_csv() == expected.to_csv()  # byte-level equality
    assert "roe" not in panel.columns  # input panel untouched


def test_value_enrichment_matches_legacy_selection_byte_for_byte(
    tmp_path, example_config_path, monkeypatch
):
    _FakeCovariatesFeed.calls = 0
    monkeypatch.setattr("qt.pipeline.TushareCovariatesFeed", _FakeCovariatesFeed)
    cfg = _cfg(tmp_path, example_config_path)
    panel = _panel()
    factors = [
        ValueFactor("value_ep"),
        MomentumFactor(20),  # non-trigger
        ValueFactor("value_bp"),
    ]
    enriched = _maybe_enrich_value(cfg, panel, ["A", "B"], factors, _LOGGER)
    assert _FakeCovariatesFeed.calls == 1  # ONE fetch covers both fields
    expected = _legacy_value_oracle(panel, factors)
    pd.testing.assert_frame_equal(enriched, expected, check_exact=True)
    assert enriched.to_csv() == expected.to_csv()  # byte-level equality
    # the guards survived the re-route: pe<=0 / pb<=0 -> NaN
    assert pd.isna(enriched.loc[(pd.Timestamp("2024-01-09"), "A"), "value_ep"])
    assert pd.isna(enriched.loc[(pd.Timestamp("2024-01-08"), "B"), "value_bp"])


# --------------------------------------------------------------------------- #
# 2. declaration-driven: a NON-FinancialFactor / NON-ValueFactor is enriched
# --------------------------------------------------------------------------- #
class _DeclaredFinaFactor(Factor):
    """NOT a FinancialFactor — yet declares a fina_indicator requirement."""

    name = "declared_fina"

    def __init__(self, requires):
        self._requires = tuple(requires)

    @property
    def spec(self) -> FactorSpec:
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description="test double for declaration-driven fina routing",
            expected_ic_sign=+1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=("roe",),
            requires=self._requires,
            adjustment="none",
            overnight_boundary="none",
            family="quality",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return panel["roe"].rename(self.name)


class _DeclaredPeFactor(Factor):
    """NOT a ValueFactor — yet declares the daily_basic pe requirement."""

    name = "declared_pe"

    def __init__(self, requires):
        self._requires = tuple(requires)

    @property
    def spec(self) -> FactorSpec:
        return FactorSpec(
            factor_id=self.name,
            version="1.0",
            description="test double for declaration-driven value routing",
            expected_ic_sign=+1,
            is_intraday=False,
            forward_return_horizon=1,
            return_basis="close_to_close",
            input_fields=("value_ep",),
            requires=self._requires,
            adjustment="none",
            overnight_boundary="none",
            family="value",
            min_history_bars=0,
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return panel["value_ep"].rename(self.name)


def test_fina_enrichment_is_declaration_driven(
    tmp_path, example_config_path, monkeypatch
):
    _FakeFinaFeed.calls = []
    monkeypatch.setattr("qt.pipeline.TushareFinancialFeed", _FakeFinaFeed)
    cfg = _cfg(tmp_path, example_config_path)
    factor = _DeclaredFinaFactor((PanelField("roe", source=FINA_INDICATOR),))
    assert not isinstance(factor, FinancialFactor)  # the premise of the proof
    enriched = _maybe_enrich_financials(cfg, _panel(), ["A", "B"], [factor], _LOGGER)
    assert _FakeFinaFeed.calls == [["roe"]]
    assert "roe" in enriched.columns  # enriched WITHOUT any pipeline change


def test_value_enrichment_is_declaration_driven(
    tmp_path, example_config_path, monkeypatch
):
    _FakeCovariatesFeed.calls = 0
    monkeypatch.setattr("qt.pipeline.TushareCovariatesFeed", _FakeCovariatesFeed)
    cfg = _cfg(tmp_path, example_config_path)
    factor = _DeclaredPeFactor((PanelField("pe", source=DAILY_BASIC),))
    assert not isinstance(factor, ValueFactor)  # the premise of the proof
    enriched = _maybe_enrich_value(cfg, _panel(), ["A", "B"], [factor], _LOGGER)
    assert _FakeCovariatesFeed.calls == 1
    # the declared pe requirement routed to the DERIVED value_ep column
    assert "value_ep" in enriched.columns
    assert enriched.loc[(pd.Timestamp("2024-01-08"), "A"), "value_ep"] == 1.0 / 20.0


# --------------------------------------------------------------------------- #
# 3. mutation: the DECLARATION drives the routing, in both directions
# --------------------------------------------------------------------------- #
def test_fina_enrichment_follows_a_mutated_declaration(
    tmp_path, example_config_path, monkeypatch
):
    _FakeFinaFeed.calls = []
    monkeypatch.setattr("qt.pipeline.TushareFinancialFeed", _FakeFinaFeed)
    cfg = _cfg(tmp_path, example_config_path)
    panel = _panel()
    # start WITHOUT a fina requirement: no enrichment, no fetch
    factor = _DeclaredFinaFactor((PanelField("volume", source=MARKET_DAILY),))
    out = _maybe_enrich_financials(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert out is panel
    assert _FakeFinaFeed.calls == []
    # MUTATION: same object, declaration edited -> the enrichment switches ON
    factor._requires = (PanelField("netprofit_yoy", source=FINA_INDICATOR),)
    out = _maybe_enrich_financials(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert _FakeFinaFeed.calls == [["netprofit_yoy"]]
    assert "netprofit_yoy" in out.columns
    # and back OFF again
    factor._requires = (PanelField("volume", source=MARKET_DAILY),)
    out = _maybe_enrich_financials(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert out is panel
    assert _FakeFinaFeed.calls == [["netprofit_yoy"]]  # no second fetch


def test_value_enrichment_follows_a_mutated_declaration(
    tmp_path, example_config_path, monkeypatch
):
    _FakeCovariatesFeed.calls = 0
    monkeypatch.setattr("qt.pipeline.TushareCovariatesFeed", _FakeCovariatesFeed)
    cfg = _cfg(tmp_path, example_config_path)
    panel = _panel()
    factor = _DeclaredPeFactor((PanelField("pe", source=DAILY_BASIC),))
    out = _maybe_enrich_value(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert _FakeCovariatesFeed.calls == 1
    assert "value_ep" in out.columns
    # MUTATION: pb instead of pe -> the DERIVED column follows the declaration
    factor._requires = (PanelField("pb", source=DAILY_BASIC),)
    out = _maybe_enrich_value(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert "value_bp" in out.columns and "value_ep" not in out.columns
    # a daily_basic field with no derived value column routes NOWHERE
    factor._requires = (PanelField("total_mv", source=DAILY_BASIC),)
    out = _maybe_enrich_value(cfg, panel, ["A", "B"], [factor], _LOGGER)
    assert out is panel
    assert _FakeCovariatesFeed.calls == 2  # the third call never fetched
