"""The unified exec-only factor-eval runner (D5 C4) — network-free.

The heavy seams are faked: ``build_eval_service`` returns a bundle wired to
synthetic providers + a tmp value store, and ``run_exec_basis_evaluation`` is a
stub that captures its kwargs (the exec tail is qt.exec_basis_eval's own tested
code). What THESE tests pin is the runner's own logic: the config gates
(catalogue C1/C2 collapse), the BUG 5 config-book closure, the exec identity
on the EvalConfig, the two book modes' ``book_view`` derivation, the
add-Section passthrough (mechanism A sink AND mechanism B neutralization),
and the artifact stem isolation (incl. the ``_bookclose`` suffix).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics.eval.sections import Section
from data.clean.intraday_schema import normalize_intraday_bars
from factors.materialize import MaterializeSources
from factors.store import FactorValueStore
from qt.config import (
    AlphaCfg,
    BacktestCfg,
    CacheCfg,
    CostCfg,
    DataCfg,
    FactorCfg,
    NeutralizeCfg,
    OOSCfg,
    OutputCfg,
    PortfolioCfg,
    ProcessingCfg,
    RootConfig,
    UniverseCfg,
)
from qt.exec_basis_eval import ExecBasisEvaluation
from qt.factor_eval_disclosures import RidgeCoverage
from qt.factor_eval_providers import EvalServiceBundle
from qt.factor_eval_runner import (
    _build_eval_config,
    _check_config_book,
    run_factor_eval,
)

SYMS = ["000001.SZ", "000002.SZ"]
DATES = pd.bdate_range("2021-07-01", periods=30)

_METRICS = {"deployment": "Watch", "predictive": "PASS", "incremental": "NOT_ASSESSED"}


def _min_config(tmp_path, **overrides) -> RootConfig:
    kwargs = dict(
        data=DataCfg(
            source="tushare",
            start="2021-07-01",
            end="2021-08-11",
            external_secret_file="/nonexistent.json",
            cache=CacheCfg(enabled=True, root_dir=str(tmp_path / "cache")),
        ),
        universe=UniverseCfg(type="index", index_code="000905.SH", symbols=[]),
        factors=[],
        processing=ProcessingCfg(neutralize=NeutralizeCfg(enabled=True)),
        alpha=AlphaCfg(),
        portfolio=PortfolioCfg(top_n=1),
        backtest=BacktestCfg(),
        cost=CostCfg(),
        output=OutputCfg(
            root_dir=str(tmp_path / "artifacts"),
            data_dir=str(tmp_path / "artifacts" / "data"),
            factor_dir=str(tmp_path / "artifacts" / "factors"),
            report_dir=str(tmp_path / "artifacts" / "reports"),
            log_dir=str(tmp_path / "artifacts" / "logs"),
        ),
        oos=OOSCfg(split_date="2021-07-20"),
    )
    kwargs.update(overrides)
    return RootConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Config gates (the collapsed C1/C2 bodies + BUG 5 closure)
# --------------------------------------------------------------------------- #
def test_missing_oos_is_a_readable_error(tmp_path):
    cfg = _min_config(tmp_path, oos=None)
    with pytest.raises(ValueError, match="requires an 'oos' section"):
        _build_eval_config(cfg)


def _cfg_demo_source(tmp_path) -> RootConfig:
    return _min_config(
        tmp_path,
        data=DataCfg(
            source="demo",
            start="2021-07-01",
            end="2021-08-11",
            cache=CacheCfg(enabled=True, root_dir=str(tmp_path / "cache")),
        ),
    )


def _cfg_cache_disabled(tmp_path) -> RootConfig:
    return _min_config(
        tmp_path,
        data=DataCfg(
            source="tushare",
            start="2021-07-01",
            end="2021-08-11",
            external_secret_file="/nonexistent.json",
            cache=CacheCfg(enabled=False, root_dir=str(tmp_path / "cache")),
        ),
    )


def _cfg_static_universe(tmp_path) -> RootConfig:
    return _min_config(tmp_path, universe=UniverseCfg(type="static", symbols=["A.SZ"]))


def _cfg_neutralize_disabled(tmp_path) -> RootConfig:
    return _min_config(
        tmp_path,
        processing=ProcessingCfg(neutralize=NeutralizeCfg(enabled=False)),
    )


@pytest.mark.parametrize(
    "cfg_factory, match",
    [
        (_cfg_demo_source, "data.source='tushare'"),
        (_cfg_cache_disabled, "data.cache.enabled=true"),
        (_cfg_static_universe, "universe.type='index'"),
        (_cfg_neutralize_disabled, "neutralize.enabled=true"),
    ],
)
def test_preconditions_fail_readably(tmp_path, monkeypatch, cfg_factory, match):
    monkeypatch.setattr(
        "qt.factor_eval_runner.load_config", lambda path: cfg_factory(tmp_path)
    )
    monkeypatch.setattr(
        "qt.factor_eval_runner.build_eval_service",
        lambda *a, **k: pytest.fail("service must not be built on a failed gate"),
    )
    with pytest.raises(ValueError, match=match):
        run_factor_eval("ignored.yaml", "jump_amount_corr_20")


def test_config_book_empty_or_exact_is_accepted(tmp_path):
    _check_config_book(_min_config(tmp_path))  # factors: [] — the honest declaration
    exact = _min_config(
        tmp_path,
        factors=[
            FactorCfg(name="value_ep"),
            FactorCfg(name="value_bp"),
            FactorCfg(name="volatility_20", params={"window": 20, "price_col": "close"}),
        ],
    )
    _check_config_book(exact)  # the legacy eval configs' declaration still passes


def test_config_book_mismatch_is_a_readable_error(tmp_path):
    cfg = _min_config(tmp_path, factors=[FactorCfg(name="momentum_20")])
    with pytest.raises(ValueError, match="does not read config 'factors:'"):
        _check_config_book(cfg)
    wrong_params = _min_config(
        tmp_path,
        factors=[
            FactorCfg(name="value_ep"),
            FactorCfg(name="value_bp"),
            FactorCfg(name="volatility_20", params={"window": 10}),
        ],
    )
    with pytest.raises(ValueError, match="does not read config 'factors:'"):
        _check_config_book(wrong_params)


def test_build_eval_config_declares_the_exec_identity_and_shared_kwargs(tmp_path):
    eval_cfg = _build_eval_config(_min_config(tmp_path))
    # exec-only: the identity is explicit (the EvalConfig default is the close pairing)
    assert eval_cfg.view == "decision"
    assert eval_cfg.return_basis == "exec_to_exec"
    assert eval_cfg.book_view is None  # per-run: no-book None / with-book book_mode
    # the catalogue C1 shared kwargs, character-for-character with the legacy bodies
    assert eval_cfg.universe == "000905.SH"
    assert eval_cfg.universe_is_pit is True
    assert (eval_cfg.start, eval_cfg.end) == ("2021-07-01", "2021-08-11")
    assert eval_cfg.is_exploratory is True
    assert eval_cfg.post_hoc_selected is False
    assert eval_cfg.rebalance == "daily"
    assert eval_cfg.n_quantiles == 5
    assert eval_cfg.cost_scenarios == (1.0, 2.0, 4.0)
    assert eval_cfg.oos_split == "2021-07-20"
    assert eval_cfg.winsorize is None
    assert eval_cfg.standardize == "zscore"
    assert eval_cfg.neutralization == ("industry", "size")
    assert eval_cfg.industry_level == "L1"
    assert eval_cfg.tuned is False
    assert eval_cfg.n_factors_screened == 1


def test_invalid_book_mode_is_rejected_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="--book-mode"):
        run_factor_eval("ignored.yaml", "jump_amount_corr_20", book_mode="bogus")


def test_valley_price_quantile_runs_end_to_end_with_neutralization_section(
    monkeypatch, tmp_path
):
    """vpq is SERVED (PR-C4b): mechanism-B disclosure -> exactly one add-Section.

    The service seams are faked (``panel`` returns a finite grid;
    ``stored_payload`` returns a synthetic raw_qbar intermediate); the expected
    coverage is recomputed INDEPENDENTLY from the same fakes via the real
    ``summarize_neutralization`` + ``reversal_20``, so a wrong wiring (wrong
    raw, wrong residual, wrong floor) fails the equality, not a tautology.
    """
    from dataclasses import asdict

    import factors.service as service_mod
    from factors.compute.minute.binding import RAW_QBAR_COL
    from factors.compute.minute.valley_price_quantile import (
        VALLEY_QUANTILE_MIN_CROSS_SECTION,
        VALLEY_QUANTILE_REVERSAL_DAYS,
        reversal_20,
    )
    from qt.factor_eval_disclosures import (
        NEUTRALIZATION_SECTION_NAME,
        NeutralizationCoverage,
        summarize_neutralization,
    )

    captured: dict = {}
    bundle = _wire(monkeypatch, tmp_path, captured)

    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(DATES), SYMS], names=["date", "symbol"]
    )

    def fake_panel(factor_ids, universe, decisions, **kwargs):
        return pd.DataFrame(
            {fid: np.arange(len(idx), dtype=float) + 1.0 for fid in factor_ids},
            index=idx,
        )

    payload_frame = pd.DataFrame(
        {RAW_QBAR_COL: np.linspace(-0.5, 0.5, len(idx))}, index=idx
    )
    payload_calls: dict = {}

    def fake_stored_payload(factor_id, universe, decisions, **kwargs):
        payload_calls.update(
            factor_id=factor_id, universe=list(universe), decisions=decisions,
            kwargs=kwargs,
        )
        return payload_frame

    monkeypatch.setattr(service_mod, "panel", fake_panel)
    monkeypatch.setattr(service_mod, "stored_payload", fake_stored_payload)

    result = run_factor_eval("ignored.yaml", "valley_price_quantile_20")

    # the mechanism-B inputs were requested for THIS factor / universe / decisions
    assert payload_calls["factor_id"] == "valley_price_quantile_20"
    assert payload_calls["universe"] == SYMS
    assert payload_calls["kwargs"]["store"] is bundle.store
    assert payload_calls["kwargs"]["sources"] is bundle.sources

    # the coverage equals the INDEPENDENTLY recomputed one
    residual = fake_panel(["valley_price_quantile_20"], SYMS, None)[
        "valley_price_quantile_20"
    ]
    rev = reversal_20(
        bundle.panel[["close"]], days=VALLEY_QUANTILE_REVERSAL_DAYS
    )
    expected = summarize_neutralization(
        payload_frame[RAW_QBAR_COL],
        rev,
        residual,
        min_cross_section=VALLEY_QUANTILE_MIN_CROSS_SECTION,
    )
    assert isinstance(result.coverage, NeutralizationCoverage)
    # field-wise with NaN tolerance: the 2-symbol fixture is below
    # min_cross_section, so raw_rev_spearman_mean is legitimately NaN on BOTH
    # sides (and NaN != NaN); every count field is an exact check.
    got, want = asdict(result.coverage), asdict(expected)
    assert got.keys() == want.keys()
    for key in got:
        g, w = got[key], want[key]
        if isinstance(g, float) and isinstance(w, float) and math.isnan(g) and math.isnan(w):
            continue
        assert g == w, key

    # ...and it reached the exec tail as exactly one add-Section
    extras = captured["extra_sections"]
    assert extras is not None and len(extras) == 1
    section = extras[0]
    assert isinstance(section, Section)
    assert section.name == NEUTRALIZATION_SECTION_NAME
    assert section.payload.keys() == got.keys()  # no derived props on this coverage
    for key in section.payload:
        g, w = section.payload[key], got[key]
        if isinstance(g, float) and isinstance(w, float) and math.isnan(g) and math.isnan(w):
            continue
        assert g == w, key
    assert section.note == expected.render()
    assert captured["stem"] == "factor_eval_valley_price_quantile_20"


# --------------------------------------------------------------------------- #
# End-to-end wiring with fake providers + a stubbed exec tail
# --------------------------------------------------------------------------- #
def _daily_panel() -> pd.DataFrame:
    # The bundle's panel is the ENRICHED close-view panel: value_ep / value_bp
    # columns are what _maybe_enrich_value would have added from daily_basic
    # pe/pb, and industry / market_cap are the neutralization covariates.
    rows = []
    for si, s in enumerate(SYMS):
        px = 100.0 + si * 20 + np.cumsum(
            np.random.RandomState(si).normal(0, 1.0, len(DATES))
        )
        for d, p in zip(DATES, px):
            rows.append(
                (d, s, p - 0.3, p + 0.5, p - 0.5, p, 1e5, p * 1e5,
                 0.05 + 0.01 * si, 0.5 + 0.1 * si, 1e9 * (si + 1), f"industry_{si}")
            )
    return (
        pd.DataFrame(
            rows,
            columns=[
                "date", "symbol", "open", "high", "low", "close", "volume",
                "amount", "value_ep", "value_bp", "market_cap", "industry",
            ],
        )
        .set_index(["date", "symbol"])
        .sort_index()
    )


def _minute_bars() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    rows = []
    for si, s in enumerate(SYMS):
        for d in DATES:
            base = pd.Timestamp(d) + pd.Timedelta("09:31:00")
            price = 100.0 + si * 5 + rng.normal(0, 2)
            for i in range(100):
                t = base + pd.Timedelta(minutes=i)
                price += rng.normal(0, 0.05)
                vol = 1e4 * (1.0 + rng.rand())
                rows.append((t, s, price, price + 0.1, price - 0.1, price, vol, price * vol))
    frame = pd.DataFrame(
        rows,
        columns=["time", "symbol", "open", "high", "low", "close", "volume", "amount"],
    )
    return normalize_intraday_bars(frame, freq="1min")


class _DailyProv:
    def __init__(self, panel):
        self._panel = panel

    def daily_panel(self, symbols, start, end):
        m = self._panel.index.get_level_values("date")
        return self._panel[(m >= pd.Timestamp(start)) & (m <= pd.Timestamp(end))]


class _MinuteProv:
    live_calls = 0

    def __init__(self, bars):
        self._bars = bars

    def minute_bars(self, symbols, start, end):
        t = self._bars.index.get_level_values("time")
        keep = self._bars[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]
        syms = keep.index.get_level_values("symbol")
        return keep[syms.isin(list(symbols))]

    def earliest_available(self, symbols):
        return DATES[0]


def _stub_exec_basis(report_dir, stem, with_book_suffix="", book_view=""):
    """Stand-in for the write-out layer, honouring its ``with_book_suffix``.

    The suffix is applied HERE, at write time, exactly as
    ``qt.exec_basis_eval.run_exec_basis_evaluation`` applies it — so a runner
    that passes the wrong suffix writes the wrong file names and the callers'
    assertions fail. The written content names the book view, so a test can
    tell whose artifact a given file is.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind in ("no_book", "with_book"):
        stem_for_kind = (
            f"{stem}_exec_with_book{with_book_suffix}"
            if kind == "with_book"
            else f"{stem}_exec_no_book"
        )
        for suffix, key in ((".md", "md"), (".json", "json"), ("_dashboard.png", "dashboard")):
            p = report_dir / f"{stem_for_kind}{suffix}"
            p.write_text(f"stub {kind} book_view={book_view}", encoding="utf-8")
            paths[f"{kind}_{key}"] = p
    return ExecBasisEvaluation(
        spec=None, params=None, artifact_path=report_dir / "a.parquet",
        artifact_key="k", artifact_reused=False, minute_live_calls=0,
        coverage={}, sanity=None, sanity_report_path=report_dir / "sanity.md",
        no_book=None, with_book=None,
        no_book_md=paths["no_book_md"], no_book_json=paths["no_book_json"],
        with_book_md=paths["with_book_md"], with_book_json=paths["with_book_json"],
        no_book_dashboard=paths["no_book_dashboard"],
        with_book_dashboard=paths["with_book_dashboard"],
        no_book_metrics=dict(_METRICS), with_book_metrics=dict(_METRICS), elapsed=0.0,
    )


def _wire(monkeypatch, tmp_path, captured):
    cfg = _min_config(tmp_path)
    monkeypatch.setattr("qt.factor_eval_runner.load_config", lambda path: cfg)
    daily = _daily_panel()
    bundle = EvalServiceBundle(
        store=FactorValueStore(str(tmp_path / "store")),
        sources=MaterializeSources(
            daily=_DailyProv(daily), minute=_MinuteProv(_minute_bars())
        ),
        panel=daily,
        symbols=list(SYMS),
        cache=None,
    )
    monkeypatch.setattr(
        "qt.factor_eval_runner.build_eval_service", lambda *a, **k: bundle
    )

    def fake_exec_eval(factor_panel, spec, eval_cfg, book, **kwargs):
        captured.update(
            factor_panel=factor_panel, spec=spec, eval_cfg=eval_cfg, book=book, **kwargs
        )
        return _stub_exec_basis(
            kwargs["report_dir"], kwargs["stem"],
            with_book_suffix=kwargs.get("with_book_suffix", ""),
            book_view=kwargs.get("book_view", ""),
        )

    monkeypatch.setattr(
        "qt.factor_eval_runner.run_exec_basis_evaluation", fake_exec_eval
    )
    return bundle


@pytest.mark.parametrize("book_mode", ["decision", "close"])
def test_end_to_end_wiring_both_book_modes(monkeypatch, tmp_path, book_mode):
    captured: dict = {}
    _wire(monkeypatch, tmp_path, captured)
    result = run_factor_eval("ignored.yaml", "jump_amount_corr_20", book_mode=book_mode)

    # the service produced the subject values; processing ran; the exec tail got them
    assert result.requested_symbols == 2
    assert captured["eval_cfg"].view == "decision"
    assert captured["eval_cfg"].return_basis == "exec_to_exec"
    # book_view IS the book mode (no-book's None is derived inside exec_basis_eval)
    assert captured["book_view"] == book_mode
    # the book carries exactly the frozen trio
    assert sorted(captured["book"].columns) == ["value_bp", "value_ep", "volatility_20"]
    # artifact stem isolation: never the legacy eval_{name} stem
    assert captured["stem"] == "factor_eval_jump_amount_corr_20"
    assert not captured["stem"].startswith("eval_")
    # jump publishes no per-day disclosure -> no extra sections, no coverage
    assert captured["extra_sections"] is None
    assert result.coverage is None
    assert result.minute_live_calls == 0
    nb = result.exec_basis.no_book_md
    assert nb.name == "factor_eval_jump_amount_corr_20_exec_no_book.md"
    wb = result.exec_basis.with_book_md
    if book_mode == "close":
        # the with-book artifacts are suffix-isolated from the decision-mode ones
        assert wb.name == "factor_eval_jump_amount_corr_20_exec_with_book_bookclose.md"
        assert wb.exists()
        assert result.exec_basis.with_book_json.exists()
        assert result.exec_basis.with_book_dashboard.exists()
    else:
        assert wb.name == "factor_eval_jump_amount_corr_20_exec_with_book.md"


def test_disclosure_rides_the_diagnostics_sink_into_an_extra_section(
    monkeypatch, tmp_path
):
    """valley_ridge_vwap_ratio: sink -> summarizer -> add-Section passthrough."""
    captured: dict = {}
    _wire(monkeypatch, tmp_path, captured)

    diag = pd.DataFrame(
        {
            "classifiable_bars": [240, 240, 240, 240],
            "valley_bars": [200, 200, 200, 200],
            "ridge_bars": [4, 12, 25, 30],
            "valid": [False, True, True, True],
        },
        index=pd.DatetimeIndex(
            pd.bdate_range("2022-01-03", periods=4), name="trade_date"
        ),
    )
    def fake_panel(factor_ids, universe, decisions, **kwargs):
        diagnostics = kwargs.get("diagnostics")
        if diagnostics is not None:
            diagnostics.append(diag)
        idx = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(DATES), list(universe)], names=["date", "symbol"]
        )
        return pd.DataFrame(
            {fid: np.arange(len(idx), dtype=float) for fid in factor_ids}, index=idx
        )

    import factors.service as service_mod

    monkeypatch.setattr(service_mod, "panel", fake_panel)
    result = run_factor_eval("ignored.yaml", "valley_ridge_vwap_ratio_20")

    # the coverage was summarized from the sink and rendered into the run record
    assert isinstance(result.coverage, RidgeCoverage)
    assert result.coverage.symbol_days == 4
    assert result.coverage.valid_days == 3
    # ...and it reached the exec tail as exactly one add-Section
    extras = captured["extra_sections"]
    assert extras is not None and len(extras) == 1
    section = extras[0]
    assert isinstance(section, Section)
    assert section.name == "ridge_scarcity_coverage"
    assert section.payload["valid_days"] == 3
    assert section.note == result.coverage.render()
    # every requested symbol had a finite value in the fake panel
    assert result.covered_symbols == 2
    assert result.empty_symbols == 0


def test_running_decision_then_close_leaves_the_decision_artifacts_intact(
    monkeypatch, tmp_path
):
    """D5 C5 F5: the run ORDER must not decide which artifacts survive.

    The close-book run used to write the shared ``_exec_with_book`` names and
    rename them afterwards, so this exact order destroyed the decision run's
    three with-book artifacts — every one of the eleven factors lost them in
    the C5 full run, and the reconcile's reports leg then died on a bare
    FileNotFoundError. The suffix now travels into the write-out layer, so
    each book mode only ever writes its own names.
    """
    captured: dict = {}
    _wire(monkeypatch, tmp_path, captured)

    decision = run_factor_eval(
        "ignored.yaml", "jump_amount_corr_20", book_mode="decision"
    )
    assert captured["with_book_suffix"] == ""
    kept = {
        p: p.read_bytes()
        for p in (
            decision.exec_basis.with_book_md,
            decision.exec_basis.with_book_json,
            decision.exec_basis.with_book_dashboard,
        )
    }
    assert all(b"book_view=decision" in blob for blob in kept.values())

    close = run_factor_eval("ignored.yaml", "jump_amount_corr_20", book_mode="close")
    assert captured["with_book_suffix"] == "_bookclose"

    for path, blob in kept.items():
        assert path.exists(), f"the close run destroyed {path.name}"
        assert path.read_bytes() == blob
    assert close.exec_basis.with_book_md.name.endswith("_exec_with_book_bookclose.md")
    assert b"book_view=close" in close.exec_basis.with_book_md.read_bytes()
