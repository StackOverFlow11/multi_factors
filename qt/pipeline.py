"""Phase 0 end-to-end orchestration (the spine that wires every slice).

This module owns the *only* place where the slices are composed into a single
reproducible run:

    demo feed -> normalize_panel -> PanelStore.write/read
      -> StaticUniverse -> MomentumFactor.compute
      -> ProcessingPipeline.transform
      -> EqualWeightAlpha.fit(None).predict (per rebalance date)
      -> TopNEqualWeight.build
      -> BacktestDriver.run (SimExecution)
      -> analytics (IC via forward_returns, performance summary)
      -> markdown report writers.

It is deliberately thin glue: every layer keeps its own boundary. The CLI
(:mod:`qt.cli`) only parses args and calls :func:`run_phase0`.

Design invariants honoured here (CLAUDE.md / CONTRACTS.md):
  * factors never see forward returns (analytics is the forward-return boundary);
  * portfolio never touches a data source or places orders;
  * backtest/live share the ``Execution`` port (we use ``SimExecution``);
  * the event order is fixed: factor at close[t], hold from t+1 (the driver
    settles each rebalance against the NEXT holding period's return);
  * all writes land under the configured ``output`` dirs (SEC-003);
  * the run is re-entrant: re-running over existing files must not fail (INV-006).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha.base import AlphaModel
from alpha.equal_weight import EqualWeightAlpha
from alpha.ic_weight import RollingICWeightAlpha
from analytics.alphalens_adapter import alphalens_factor_metrics
from analytics.factor import compute_ic, forward_returns, ic_summary, quantile_returns
from analytics.performance import performance_summary
from analytics.quantstats_adapter import quantstats_performance
from data.clean.adjust import front_adjust
from data.clean.covariates import enrich_covariates, enrich_listing, enrich_pit_industry
from data.clean.pit_industry import asof_industry
from data.clean.pit_financials import asof_financials
from data.clean.tradability import enrich_tradability
from data.feed.base import DataFeed
from data.feed.demo_feed import DemoFeed
from data.feed.index_feed import IndexConstituentsFeed
from data.feed.tushare_covariates import TushareCovariatesFeed
from data.feed.tushare_feed import TushareFeed
from data.feed.tushare_fina import TushareFinancialFeed
from data.feed.tushare_flags import TushareFlagsFeed
from data.store.panel_store import PanelStore
from factors.compute.candidates import (
    VALUE_FIELDS,
    LiquidityFactor,
    OvernightMomentumFactor,
    ReversalFactor,
    ValueFactor,
    VolatilityFactor,
)
from factors.compute.financial import SUPPORTED_FIELDS as SUPPORTED_FINANCIAL_FIELDS
from factors.compute.financial import FinancialFactor
from factors.compute.momentum import MomentumFactor
from factors.process.pipeline import ProcessingPipeline
from portfolio.construct import TopNEqualWeight
from qt.config import RootConfig, load_config
from qt.reports import write_phase0_summary
from runtime.backtest.driver import BacktestDriver
from runtime.backtest.sim_execution import SimExecution
from universe.base import Universe
from universe.index_universe import PITIndexUniverse
from universe.static import StaticUniverse

_LOGGER_NAME = "qt.run_phase0"

# Periods per year per rebalance cadence. The nav table has one row per rebalance
# (monthly -> 12 rows/year), so performance metrics must annualize against that
# cadence, NOT the daily 252 that ``performance_summary`` defaults to.
_PERIODS_PER_YEAR: dict[str, int] = {"monthly": 12}
_DEFAULT_PERIODS_PER_YEAR: int = 12
_INDEX_UNIVERSE_LOOKBACK_DAYS: int = 370
# Financials are fetched from before the backtest start so the latest report
# DISCLOSED before start (its period end is months earlier than its ann_date) is
# in the set and can be as-of carried forward onto early trade dates. ~16 months
# covers an annual report disclosed up to a year-plus before start.
_FINANCIAL_LOOKBACK_DAYS: int = 500


def _periods_per_year(rebalance: str) -> int:
    """Annualization factor for the configured rebalance cadence (P0: monthly=12)."""
    return _PERIODS_PER_YEAR.get(rebalance, _DEFAULT_PERIODS_PER_YEAR)


@dataclass(frozen=True)
class Phase0Result:
    """Immutable summary of one phase0 run (what the report/tests consume)."""

    config: RootConfig
    panel_rows: int
    panel_symbols: int
    # primary factor = first enabled (drives the legacy single-factor fields and
    # the standard-analytics factor cross-check); factor_names = ALL enabled.
    factor_name: str
    factor_names: tuple[str, ...]
    # P3-1 per-factor + combo-score analytics (simple, authoritative).
    # per_factor[name] = {ic_mean, ic_ir, quantile_returns, coverage};
    # combo_analytics  = {ic_mean, ic_ir, quantile_returns} on the traded score.
    per_factor: dict[str, dict]
    combo_analytics: dict
    # P3-2 alpha disclosure: summary = {model, [hyper-params, n_dates,
    # n_fallback, trained_coverage]}; weights = full per-date EFFECTIVE weights
    # log (+ fallback flag) for walk-forward models, None for equal_weight.
    alpha_summary: dict
    alpha_weights: pd.DataFrame | None
    # legacy top-level metrics == the PRIMARY factor's (unchanged for
    # single-factor configs).
    ic_mean: float
    ic_ir: float
    quantile_returns: pd.DataFrame
    nav_table: pd.DataFrame
    performance: dict[str, float]
    avg_turnover: float
    cost_drag: float
    # P2-4 standard-analytics cross-check (report-only; never alters the above).
    std_performance: dict
    std_factor: dict
    downgrades: tuple[str, ...]
    data_path: Path
    factor_path: Path
    report_path: Path
    log_path: Path


class _FrameScores:
    """Scores source backed by a precomputed (date, symbol) score panel.

    Bridges the processed factor panel + alpha model to the ``ScoresSource``
    port the :class:`BacktestDriver` depends on. It only *reads* scores already
    computed from past/current factors — it never sees forward returns, so the
    no-lookahead boundary is preserved.
    """

    def __init__(self, score_panel: pd.Series) -> None:
        # score_panel: MultiIndex(date, symbol) -> score (one column collapsed).
        self._scores = score_panel

    def get(self, date: pd.Timestamp, symbols: list[str]) -> pd.Series:
        """Return symbol-indexed scores for ``date`` restricted to ``symbols``."""
        norm = pd.Timestamp(date).normalize()
        date_level = self._scores.index.get_level_values("date")
        cross = self._scores.loc[date_level == norm]
        if cross.empty:
            return pd.Series(index=list(symbols), dtype=float)
        cross = pd.Series(cross.to_numpy(), index=cross.index.get_level_values("symbol"))
        return cross.reindex(list(symbols))


def _make_logger(log_path: Path, name: str = _LOGGER_NAME) -> logging.Logger:
    """A run-scoped logger that writes to ``log_path`` (and never logs secrets)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Re-entrant: drop handlers from a previous run so we don't double-write.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _build_alpha(cfg: RootConfig) -> AlphaModel:
    """Instantiate the configured alpha model (P3-2).

    ``equal_weight`` -> :class:`EqualWeightAlpha` (P0 baseline, no future data).
    ``ic_weighted``  -> :class:`RollingICWeightAlpha` (walk-forward rolling-IC
    weights; the IC horizon is tied to the FIRST configured forward-return
    period so training and the reported IC share one definition).
    """
    params = dict(cfg.alpha.params)
    if cfg.alpha.model == "equal_weight":
        return EqualWeightAlpha()
    if cfg.alpha.model == "ic_weighted":
        return RollingICWeightAlpha(
            window=int(params.get("window", 60)),
            min_periods=int(params.get("min_periods", 20)),
            horizon=int(cfg.analytics.forward_return_periods[0]),
            mode=str(params.get("mode", "rolling")),
        )
    raise ValueError(
        f"Unknown alpha.model {cfg.alpha.model!r}; expected 'equal_weight' or "
        "'ic_weighted'."
    )


def _alpha_forward_returns(
    cfg: RootConfig, panel: pd.DataFrame, alpha: AlphaModel
) -> pd.Series | None:
    """Forward returns for ALPHA FITTING only (None for no-future-data models).

    Computed here — at the alpha boundary — and handed ONLY to ``alpha.fit``:
    the factor layer never sees them (CLAUDE.md invariant #1). The model itself
    enforces the walk-forward cutoff (a pair is used at date d only once
    realized, t + h <= d), locked by tests.
    """
    if not getattr(alpha, "requires_forward_returns", False):
        return None
    horizon = int(cfg.analytics.forward_return_periods[0])
    fwd = forward_returns(panel, periods=(horizon,))
    return fwd[f"forward_return_{horizon}d"]


def _alpha_disclosure(
    cfg: RootConfig, alpha: AlphaModel
) -> tuple[dict, pd.DataFrame | None]:
    """(alpha_summary, full per-date weights log) for the result/report.

    For ``ic_weighted``: echoes the hyper-parameters and counts the fallback
    dates (insufficient realized history -> equal weight) so the report can
    disclose training coverage. For ``equal_weight``: just the model name.
    """
    if isinstance(alpha, RollingICWeightAlpha):
        log = alpha.weights_log()
        n_dates = int(len(log))
        n_fallback = int(log["fallback"].sum()) if n_dates else 0
        summary = {
            "model": cfg.alpha.model,
            **alpha.params(),
            "n_dates": n_dates,
            "n_fallback": n_fallback,
            "trained_coverage": (
                (n_dates - n_fallback) / n_dates if n_dates else float("nan")
            ),
        }
        return summary, log
    return {"model": cfg.alpha.model}, None


def _build_scores(
    processed: pd.DataFrame,
    alpha: AlphaModel,
    forward_returns_for_fit: pd.Series | None = None,
) -> pd.Series:
    """Predict one score per (date, symbol) from the processed factor panel.

    ``forward_returns_for_fit`` is passed ONLY to ``alpha.fit`` (the alpha-layer
    boundary; None for equal-weight). Prediction is per-date cross-sections, so
    a single (date, symbol) score panel is produced for the backtest scores
    source; a walk-forward model derives each date's weights from realized
    history only.
    """
    alpha.fit(processed, forward_returns_for_fit)
    dates = processed.index.get_level_values("date")
    blocks: list[pd.Series] = []
    for date, block in processed.groupby(dates, sort=True):
        scores = alpha.predict(block)
        idx = pd.MultiIndex.from_product([[date], list(scores.index)], names=["date", "symbol"])
        blocks.append(pd.Series(scores.to_numpy(), index=idx))
    if not blocks:
        return pd.Series(dtype=float, index=processed.index[:0])
    return pd.concat(blocks).rename("score")


def _collect_downgrades(cfg: RootConfig) -> tuple[str, ...]:
    """Enumerate the downgrades that MUST be disclosed (INV-007), path-aware.

    The first item states the active DATA PATH explicitly so a reader can never
    mistake a demo run for a real PIT / financial validation, or vice versa.
    """
    real = cfg.data.source == "tushare"
    enabled_names = [f.name for f in cfg.factors if f.enabled] or ["?"]
    financial_names = [n for n in enabled_names if n in SUPPORTED_FINANCIAL_FIELDS]
    price_names = [n for n in enabled_names if n not in SUPPORTED_FINANCIAL_FIELDS]

    if real:
        parts = ["front-adjusted (qfq) prices"]
        parts.append(
            f"PIT index membership ({cfg.universe.index_code})"
            if cfg.universe.type == "index"
            else "STATIC universe (still a PIT downgrade — see below)"
        )
        factor_bits = []
        if price_names:
            factor_bits.append(f"price factor(s) {price_names}")
        if financial_names:
            factor_bits.append(f"ann_date-aligned financial factor(s) {financial_names}")
        parts.append(" + ".join(factor_bits) or "no factor (?)")
        if cfg.processing.neutralize.enabled:
            parts.append("industry+size neutralized")
        path = "DATA PATH = REAL tushare: " + "; ".join(parts) + "."
    else:
        path = (
            "DATA PATH = DEMO / offline (DemoFeed, deterministic, network-free): "
            "this is NOT real market data; results must NOT be read as a real PIT, "
            "ann_date, or financial validation (CFG-005)."
        )

    # membership
    if cfg.universe.type == "index":
        membership = (
            f"Index universe (PITIndexUniverse, {cfg.universe.index_code}): "
            "point-in-time membership from tushare index_weight snapshots (latest "
            "snapshot on-or-before each date; survivorship-safe). RESOLVES the "
            "static-universe PIT downgrade (UNI-003/UNI-009)."
        )
    else:
        membership = (
            "Static universe (StaticUniverse): membership is date-independent, NOT "
            "point-in-time index constituents (UNI-003 PIT downgrade). Survivorship / "
            "look-ahead membership bias is present."
        )

    # financials / ann_date
    if financial_names:
        financials = (
            f"Financial factor(s) {financial_names} are ann_date PIT-aligned "
            "(DATA-012): a figure is used only after its disclosure date, never by "
            "report period. All financial fields are fetched in ONE pass and "
            "as-of aligned together (P3-1, no per-factor refetch)."
        )
    else:
        financials = (
            "No financial factor in this run; ann_date alignment is implemented and "
            "available but unused here (price factor only)."
        )

    # factor combination (P3-1 equal-weight / P3-2 walk-forward IC weights)
    if cfg.alpha.model == "ic_weighted":
        combo = (
            f"Alpha = WALK-FORWARD rolling-IC weights (P3-2) over {enabled_names}: "
            "at each date d the factor weights are the mean per-factor rank IC over "
            "the trailing window of REALIZED observations only — a (factor[t], "
            "fwd_h[t]) pair is admitted only once realized (t + h <= d in trading "
            "days), so no full-sample or future information enters any date's weights "
            "(locked by a perturb-the-future test). The alpha layer sees historical "
            "realized forward returns ONLY for this fitting (invariant #1: factors "
            "never see them). Weights are L1-normalized and sign-preserving (a "
            "negative-IC factor gets a negative weight). Dates with insufficient "
            "realized history fall back to EQUAL WEIGHT — the fallback count is "
            "disclosed in the report. NOT a tuned-performance claim."
        )
    elif len(enabled_names) > 1:
        combo = (
            f"Multi-factor combination (P3-1): {enabled_names} are combined as the "
            "EQUAL-WEIGHT mean of the per-date processed (z-scored / neutralized) "
            "columns — no learned weights, no forward-return fitting, no parameter "
            "search (the alpha layer never sees future returns). drop_missing "
            "requires ALL enabled factors for a name on a date: a name missing any "
            "factor value is dropped from that cross-section (disclosed, not "
            "silently scored on partial data)."
        )
    else:
        combo = (
            "Single-factor run: the combined score equals the processed factor "
            "(equal-weight mean of one column)."
        )

    # neutralization
    if cfg.processing.neutralize.enabled and real:
        level = cfg.processing.neutralize.industry_level
        neutral = (
            f"Factor is industry + market-cap neutralized; industry is **point-in-time** "
            f"SW-{level} (UNI-010): as-of the trade date via index_member_all in/out dates, "
            f"NOT the current stock_basic tag. The SW level is configurable "
            f"(processing.neutralize.industry_level, default L1 = 31 broad sectors, the "
            f"standard for neutralization). Going PIT switches the taxonomy from the old "
            f"(non-PIT-able) stock_basic.industry tag to SW, so the result changes vs the "
            f"old tag regardless of level (L1 ≈ L2 in tests). Names with no SW history get "
            f"NaN (a disclosed coverage gap the neutralizer drops) — never a silent "
            f"current-tag fallback. Market cap is per-date (daily_basic.total_mv)."
        )
    elif cfg.processing.neutralize.enabled:
        neutral = (
            "Factor is industry + market-cap neutralized (real path only; this is not a "
            "tushare run, so the covariates are unavailable)."
        )
    else:
        neutral = "No neutralization in this run (raw cross-sectional factor)."

    if real and cfg.universe.min_listing_days > 0:
        listing = (
            f"universe.min_listing_days={cfg.universe.min_listing_days} is ENFORCED "
            "(real path): names younger than that as of each date are excluded from "
            "selection (UNI-008); a missing list_date is a disclosed data gap (kept, "
            "never silently dropped)."
        )
    else:
        listing = (
            f"universe.min_listing_days={cfg.universe.min_listing_days} is NOT enforced "
            "on this path (no listing dates on the demo source); disclosed no-op (UNI-008)."
        )
    execution = (
        "Execution feasibility is direction-aware (UNI-007 / P2-2): at-up-limit blocks "
        "buys, at-down-limit blocks sells, suspended/missing-close blocks both; blocked "
        "trades carry forward and turnover/cost count only executed trades (no forced "
        "impossible trades; idle cash from blocked buys earns cash_return). The demo "
        "panel carries no flags, so every trade is feasible there (P0/P1 unchanged)."
    )

    items = [
        path,
        membership,
        financials,
        combo,
        neutral,
        execution,
        f"Daily ({cfg.data.freq}) bars only; minute-level link is deferred.",
        "Analytics (P2-4): the AUTHORITATIVE backtest metrics are the simple "
        "numpy/pandas implementation (deterministic, audit-light); "
        "alphalens-reloaded (IC / quantiles) and quantstats (CAGR / Sharpe / maxDD "
        "/ vol) are computed ALONGSIDE as a standard cross-check, report-only — they "
        "never alter selection / portfolio / execution. The 'Standard analytics' "
        "report section names the backend actually used and discloses it when a "
        "library is unavailable or errors (no silent fake, INV-007).",
        listing,
    ]
    return tuple(items)


def run_phase0(config_path: str) -> Phase0Result:
    """Run the full Phase 0 pipeline from a YAML config and write the reports.

    Returns a :class:`Phase0Result`. Raises ``ConfigError`` (readable) on a bad
    config and ``ValueError`` (readable) on an empty/degenerate run. All file
    writes are confined to the configured ``output`` directories (SEC-003).
    """
    cfg = load_config(config_path)

    log_path = Path(cfg.output.log_dir) / "run_phase0.log"
    logger = _make_logger(log_path)
    logger.info("phase0 start: project=%s source=%s", cfg.project.name, cfg.data.source)

    cache = _build_cache(cfg)
    universe, symbols = _build_universe(cfg, logger, cache)
    panel = _load_panel(cfg, symbols, logger, cache)
    factors = _build_factors(cfg)
    primary = factors[0]
    panel = _maybe_enrich_financials(cfg, panel, symbols, factors, logger, cache)
    panel = _maybe_enrich_value(cfg, panel, symbols, factors, logger, cache)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger, cache)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger, cache)
    # P4-1/P4-2: one concise cache-stats line after every cached endpoint has run
    # (market bars + universe/tradability). A warm historical rerun shows all 0s.
    _log_run_cache_stats(cache, logger)
    factor_panel = _compute_factor_panel(cfg, panel, factors, logger)

    processed = _process_factors(cfg, factor_panel, panel)
    alpha = _build_alpha(cfg)
    scores = _build_scores(processed, alpha, _alpha_forward_returns(cfg, panel, alpha))
    alpha_summary, alpha_weights = _alpha_disclosure(cfg, alpha)
    logger.info("alpha: %s", alpha_summary)

    nav_table = _run_backtest(cfg, panel, scores, primary.name, universe, logger)
    per_factor, combo = _factor_analytics(cfg, panel, factor_panel, scores)
    first = per_factor[primary.name]
    ic_mean, ic_ir, q_returns = first["ic_mean"], first["ic_ir"], first["quantile_returns"]
    perf = (
        performance_summary(
            nav_table["nav"],
            periods_per_year=_periods_per_year(cfg.backtest.rebalance),
        )
        if not nav_table.empty
        else {
            "annual_return": float("nan"),
            "max_drawdown": float("nan"),
            "volatility": float("nan"),
            "sharpe": float("nan"),
        }
    )
    avg_turnover = float(nav_table["turnover"].mean()) if not nav_table.empty else 0.0
    cost_drag = float(nav_table["cost"].sum()) if not nav_table.empty else 0.0
    std_performance, std_factor = _standard_analytics(
        cfg, panel, factor_panel, primary.name, nav_table, ic_mean, ic_ir, perf, logger
    )

    downgrades = _collect_downgrades(cfg)
    result = Phase0Result(
        config=cfg,
        panel_rows=len(panel),
        panel_symbols=panel.index.get_level_values("symbol").nunique(),
        factor_name=primary.name,
        factor_names=tuple(f.name for f in factors),
        per_factor=per_factor,
        combo_analytics=combo,
        alpha_summary=alpha_summary,
        alpha_weights=alpha_weights,
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        quantile_returns=q_returns,
        nav_table=nav_table,
        performance=perf,
        avg_turnover=avg_turnover,
        cost_drag=cost_drag,
        std_performance=std_performance,
        std_factor=std_factor,
        downgrades=downgrades,
        data_path=Path(cfg.output.data_dir) / f"{cfg.data.output_name}.parquet",
        factor_path=Path(cfg.output.factor_dir) / "factors.parquet",
        report_path=Path(cfg.output.report_dir) / "phase0_summary.md",
        log_path=log_path,
    )
    write_phase0_summary(result)
    logger.info(
        "phase0 done: ic_mean=%.4f ic_ir=%.4f annual_return=%.4f report=%s",
        ic_mean,
        ic_ir,
        perf.get("annual_return", float("nan")),
        result.report_path,
    )
    return result


# --------------------------------------------------------------------------- #
# Step helpers (each small + single-purpose; orchestration stays linear above).
# --------------------------------------------------------------------------- #
def _build_feed(cfg: RootConfig, cache=None) -> DataFeed:
    """Construct the DataFeed for the configured source (``demo`` | ``tushare``).

    Dispatches on ``cfg.data.source`` so a ``tushare`` config is NOT silently
    served demo data. The tushare branch requires ``external_secret_file`` (the
    token never lives in the repo); constructing the feed performs no network /
    token read — that is deferred to first ``get_bars`` (SEC-001/004). The shared
    read-through ``cache`` (or None) is injected by the runner.
    """
    if cfg.data.source == "demo":
        return DemoFeed(calendar_start=cfg.data.start)
    if cfg.data.source == "tushare":
        secret_file = cfg.data.external_secret_file
        if not secret_file:
            raise ValueError(
                "data.source is 'tushare' but data.external_secret_file is not set. "
                "Point it at your .config.json (the token is read from there, never "
                "hardcoded)."
            )
        return TushareFeed(
            secret_file=secret_file,
            token_key=cfg.data.tushare_token_key,
            cache=cache,
        )
    raise ValueError(
        f"Unsupported data.source {cfg.data.source!r}; expected 'demo' or 'tushare'."
    )


def _build_cache(cfg: RootConfig):
    """Build the ONE shared read-through cache when ``data.cache.enabled``.

    Returns ``None`` when caching is off — every feed then behaves EXACTLY as
    before (backward compatible). A single instance is threaded through every
    tushare-backed feed in a run so coverage/store stay one consistent
    source-of-truth and the per-endpoint gap-fetch counts aggregate into one
    run-log line. The cache stores RAW endpoint facts only (P4-1 market bars +
    P4-2 index_weight / suspend_d / namechange / stk_limit / stock_basic);
    front_adjust, raw price-limit checks, and PIT as-of logic stay downstream.
    """
    cache_cfg = cfg.data.cache
    if not cache_cfg.enabled:
        return None
    from data.cache import CacheParquetStore, CoverageLedger, TushareCache

    # D-series schema drift guard: built ONLY when opted in (default-off => None =>
    # every cache parse site stays a byte-identical passthrough).
    schema_guard = None
    if cache_cfg.schema_guard.enabled:
        from data.cache.schema_registry import SchemaGuard

        schema_guard = SchemaGuard(mode=cache_cfg.schema_guard.mode)

    root = cache_cfg.root_dir
    return TushareCache(
        CacheParquetStore(root),
        CoverageLedger(root),
        refresh_recent_days=cache_cfg.refresh_recent_days,
        refresh_dimension_days=cache_cfg.refresh_dimension_days,
        force_refresh=tuple(cache_cfg.force_refresh),
        schema_guard=schema_guard,
    )


def _build_universe(
    cfg: RootConfig, logger: logging.Logger, cache=None
) -> tuple[Universe, list[str]]:
    """Construct the universe and the symbol set whose market data must be loaded.

    ``static`` -> :class:`StaticUniverse` over the configured symbols (PIT downgrade).
    ``index``  -> :class:`PITIndexUniverse` from real tushare ``index_weight``
    snapshots; the symbol set is the union of all historical constituents so the
    backtest can settle names that later left the index (no survivorship bias).
    """
    filters = cfg.universe.filters.model_dump()
    # min_listing_days is a buy/selection-eligibility filter (UNI-008); thread it
    # into the shared filter dict so both static and index universes enforce it
    # when a list_date column is present (real path) and no-op otherwise (demo).
    filters["min_listing_days"] = int(cfg.universe.min_listing_days or 0)
    if cfg.universe.type == "static":
        symbols = list(cfg.universe.symbols)
        if not symbols:
            raise ValueError(
                "universe.symbols is empty; configure at least one symbol (static)."
            )
        return StaticUniverse(symbols, filters), symbols
    if cfg.universe.type == "index":
        if cfg.data.source != "tushare":
            raise ValueError(
                "universe.type='index' requires data.source='tushare' "
                "(constituents come from tushare index_weight; demo has no index)."
            )
        if not cfg.data.external_secret_file:
            raise ValueError(
                "universe.type='index' requires data.external_secret_file "
                "(token for index_weight)."
            )
        feed = IndexConstituentsFeed(
            cfg.data.external_secret_file,
            token_key=cfg.data.tushare_token_key,
            cache=cache,
        )
        # As-of membership at cfg.data.start needs the latest snapshot *before*
        # the backtest window too. Pulling only [start, end] can leave early
        # rebalance dates empty when the run starts between two index snapshots.
        cons_start = (
            pd.Timestamp(cfg.data.start) - pd.Timedelta(days=_INDEX_UNIVERSE_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")
        cons = feed.get_constituents(cfg.universe.index_code, cons_start, cfg.data.end)
        if cons.empty:
            raise ValueError(
                f"No constituents for index {cfg.universe.index_code} over "
                f"[{cfg.data.start}, {cfg.data.end}]."
            )
        symbols = sorted(cons["symbol"].unique().tolist())
        logger.info(
            "universe: index %s, %d snapshots, %d distinct constituents",
            cfg.universe.index_code,
            cons["date"].nunique(),
            len(symbols),
        )
        return PITIndexUniverse(cons, filters), symbols
    raise ValueError(f"Unsupported universe.type {cfg.universe.type!r}.")


# Endpoints surfaced in the run-log cache-stats line, in a fixed order. Market
# bars (P4-1) come first so the historical "data cache: market_daily_gap_fetches=
# N adj_factor_gap_fetches=M ..." prefix is preserved; P4-2 endpoints follow.
_CACHED_ENDPOINTS: tuple[str, ...] = (
    "market_daily", "adj_factor", "index_weight", "suspend_d",
    "namechange", "stk_limit", "stock_basic",
    "daily_basic", "fina_indicator", "index_member_all",
)


def _format_cache_stats(stats: dict) -> str:
    """One concise line of per-endpoint gap-fetch counts (endpoint counts only)."""
    return "data cache: " + " ".join(
        f"{ep}_gap_fetches={int(stats.get(ep, 0))}" for ep in _CACHED_ENDPOINTS
    )


def _log_cache_stats(feed, logger: logging.Logger) -> None:
    """Log a feed's per-endpoint cache gap-fetch counts (P4-1/P4-2), if any.

    No-op unless ``feed`` exposes ``cache_stats()`` returning a dict (i.e. the
    persistent cache is enabled). Emits one concise line through the run-scoped
    logger — endpoint counts only, never a token, secret path, or symbol dump.
    """
    getter = getattr(feed, "cache_stats", None)
    if getter is None:
        return
    stats = getter()
    if not stats:
        return
    logger.info("%s", _format_cache_stats(stats))


def _log_run_cache_stats(cache, logger: logging.Logger) -> None:
    """Log the SHARED run cache's aggregated gap-fetch counts (P4-1/P4-2).

    Called once after all tushare-backed feeds in a run have used the shared
    cache, so the counts cover every cached endpoint (market bars + universe /
    tradability). No-op when caching is disabled (``cache`` is None). Endpoint
    counts only — never a token, secret path, or per-symbol detail.
    """
    if cache is None:
        return
    stats = cache.stats()
    if not stats:
        return
    logger.info("%s", _format_cache_stats(stats))
    # D-series schema drift guard: one secret-free count line, only when a guard
    # is attached (default-off => schema_summary() is None => nothing logged).
    summary_getter = getattr(cache, "schema_summary", None)
    summary = summary_getter() if summary_getter is not None else None
    if summary is not None:
        logger.info(
            "schema guard: hard=%d warning=%d total=%d",
            int(summary["hard"]), int(summary["warning"]), int(summary["total"]),
        )


def _load_panel(
    cfg: RootConfig, symbols: list[str], logger: logging.Logger, cache=None
) -> pd.DataFrame:
    """Fetch the market panel for ``symbols``, persist it, and read it back."""
    if not symbols:
        raise ValueError("No symbols to load; the universe produced an empty set.")
    feed = _build_feed(cfg, cache)
    panel = feed.get_bars(symbols, cfg.data.start, cfg.data.end, freq=cfg.data.freq)
    if panel.empty:
        raise ValueError(
            "No market data returned for the configured window "
            f"[{cfg.data.start}, {cfg.data.end}] and symbols {symbols}."
        )

    store = PanelStore(cfg.output.data_dir)
    store.write(cfg.data.output_name, panel, overwrite=cfg.output.overwrite)
    panel = store.read(cfg.data.output_name)
    # Tradability flags FIRST, on RAW prices: the price-limit flags must compare
    # the UNADJUSTED close against the raw stk_limit (limits are quoted in raw
    # price terms). front-adjust below only scales OHLC and leaves the boolean
    # flags intact, so factors/backtest see qfq prices while limit flags stay right.
    panel = _enrich_tradability(cfg, panel, symbols, logger, cache)
    # Store stays RAW (+ adj_factor, incremental-safe); front-adjust in memory so
    # factors / backtest see continuous qfq prices. Identity for demo (adj=1.0).
    panel = front_adjust(panel)
    logger.info(
        "data: %d rows, %d symbols (front-adjusted)",
        len(panel),
        panel.index.get_level_values("symbol").nunique(),
    )
    return panel


def _enrich_tradability(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    logger: logging.Logger,
    cache=None,
) -> pd.DataFrame:
    """Add suspended / ST / price-limit flags when filters need them (tushare only).

    No-op for the offline demo source (no flag data) and when no flag-driven
    filter is enabled, so the demo path is unchanged. Flags are joined as boolean
    columns that :func:`universe.filters.apply_tradable_filters` consults.
    """
    flt = cfg.universe.filters
    if cfg.data.source != "tushare" or not (flt.suspended or flt.st or flt.limit_up_down):
        return panel
    feed = TushareFlagsFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    suspended = feed.suspended(symbols, cfg.data.start, cfg.data.end) if flt.suspended else None
    st = feed.st_intervals(symbols) if flt.st else None
    limits = feed.limits(symbols, cfg.data.start, cfg.data.end) if flt.limit_up_down else None
    panel = enrich_tradability(panel, suspended=suspended, st_intervals=st, limits=limits)
    logger.info(
        "tradability: flags enriched (suspended=%s st=%s limit_up_down=%s)",
        flt.suspended, flt.st, flt.limit_up_down,
    )
    return panel


def _build_factors(cfg: RootConfig) -> list:
    """Instantiate EVERY enabled factor from config, in config order (P3-1).

    ``momentum*`` -> :class:`MomentumFactor` (price-based, P0). A financial field
    name (e.g. ``roe``, ``netprofit_yoy``) -> :class:`FinancialFactor`, which needs
    the tushare data path (ann_date alignment) — not available for demo data.

    Multiple enabled factors each become their own factor-panel column; the
    combined score stays the alpha layer's equal-weight mean (no learned weights).
    Duplicate factor names are a config error: names become panel columns and a
    silent collision would overwrite one factor with another.
    """
    enabled = [f for f in cfg.factors if f.enabled]
    if not enabled:
        raise ValueError(
            "No enabled factor in config.factors; the pipeline needs at least one."
        )
    factors: list = []
    for spec in enabled:
        params = dict(spec.params)
        window = int(params.get("window", 20))
        if spec.name in SUPPORTED_FINANCIAL_FIELDS:
            factor = FinancialFactor(field=spec.name)
        elif spec.name in VALUE_FIELDS:
            factor = ValueFactor(spec.name)
        elif spec.name.startswith("overnight_mom"):
            factor = OvernightMomentumFactor(
                window=window,
                open_col=str(params.get("open_col", "open")),
                close_col=str(params.get("close_col", "close")),
            )
        elif spec.name.startswith("momentum"):
            factor = MomentumFactor(
                window=window, price_col=str(params.get("price_col", "close"))
            )
        elif spec.name.startswith("reversal"):
            factor = ReversalFactor(
                window=window, price_col=str(params.get("price_col", "close"))
            )
        elif spec.name.startswith("volatility"):
            factor = VolatilityFactor(
                window=window, price_col=str(params.get("price_col", "close"))
            )
        elif spec.name.startswith("liquidity"):
            factor = LiquidityFactor(
                window=window, amount_col=str(params.get("amount_col", "amount"))
            )
        else:
            raise ValueError(
                f"Unknown factor {spec.name!r}; expected 'momentum*', 'reversal*', "
                f"'volatility*', 'liquidity*', 'overnight_mom*', one of "
                f"{VALUE_FIELDS} or one of {SUPPORTED_FINANCIAL_FIELDS}."
            )
        # window-named factors derive their name from params: a spec named
        # reversal_5 with params.window=10 would silently mislabel the column.
        if factor.name != spec.name:
            raise ValueError(
                f"Factor name/params mismatch: config names {spec.name!r} but the "
                f"params resolve to {factor.name!r} (window-named factors must "
                "agree with params.window)."
            )
        factors.append(factor)
    names = [f.name for f in factors]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"Duplicate enabled factor name(s) {dupes}: factor names become factor-"
            "panel columns and must be unique (e.g. two momentum entries with the "
            "same window resolve to the same name)."
        )
    return factors


def _maybe_enrich_financials(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    factors: list,
    logger: logging.Logger,
    cache=None,
) -> pd.DataFrame:
    """Attach ann_date-aligned columns for ALL financial factors (single fetch).

    Every :class:`FinancialFactor` field among the enabled ``factors`` is fetched
    in ONE ``fina_indicator`` pass and as-of aligned in ONE
    :func:`asof_financials` call (P3-1) — no per-factor refetch. Financial factors
    require the tushare data path: a demo run has no disclosure dates, so we fail
    with a readable error rather than fabricate financials.
    """
    fields = [f.name for f in factors if isinstance(f, FinancialFactor)]
    if not fields:
        return panel
    if cfg.data.source != "tushare":
        raise ValueError(
            f"Factor(s) {fields} are financial factors and need real financial "
            f"data (data.source='tushare'); they cannot run on demo data."
        )
    feed = TushareFinancialFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    # Look back before start so the prior already-disclosed report is fetched and
    # can be as-of carried forward onto the early trade dates (no NaN gap).
    fetch_start = (
        pd.Timestamp(cfg.data.start) - pd.Timedelta(days=_FINANCIAL_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    fina = feed.get_fina_indicator(symbols, fetch_start, cfg.data.end, fields=fields)
    enriched = panel.copy()
    aligned = asof_financials(panel.index, fina, fields)
    for field in fields:
        enriched[field] = aligned[field]
    logger.info(
        "financials: as-of aligned %s by ann_date (%d disclosed rows, single fetch)",
        fields, len(fina),
    )
    return enriched


def _maybe_enrich_value(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    factors: list,
    logger: logging.Logger,
    cache=None,
) -> pd.DataFrame:
    """Attach value_ep / value_bp columns when value factors are enabled (P3-5).

    ONE daily_basic fetch covers both fields. The published pe/pb are same-day
    ratios (PIT-safe by construction); the inversion guards non-positive ratios
    to NaN (a negative or zero pe/pb would otherwise flip the value ranking).
    Demo has no pe/pb — a readable error, never fabricated ratios.
    """
    fields = [f.name for f in factors if isinstance(f, ValueFactor)]
    if not fields:
        return panel
    if cfg.data.source != "tushare":
        raise ValueError(
            f"Factor(s) {fields} need daily_basic pe/pb (data.source='tushare'); "
            "they cannot run on demo data."
        )
    feed = TushareCovariatesFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    ratios = feed.value_ratios(symbols, cfg.data.start, cfg.data.end)
    enriched = panel.copy()
    if ratios.empty:
        for field in fields:
            enriched[field] = float("nan")
        logger.info("value: daily_basic returned no rows; %s all-NaN", fields)
        return enriched
    r = ratios.copy()
    r["symbol"] = r["symbol"].astype(str)
    r = r.set_index(["date", "symbol"]).sort_index()
    inverted = {
        "value_ep": 1.0 / r["pe"].where(r["pe"] > 0),
        "value_bp": 1.0 / r["pb"].where(r["pb"] > 0),
    }
    for field in fields:
        enriched[field] = inverted[field].reindex(enriched.index)
    logger.info(
        "value: daily_basic pe/pb enriched -> %s (%d ratio rows, single fetch)",
        fields, len(r),
    )
    return enriched


def _maybe_enrich_listing(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    logger: logging.Logger,
    cache=None,
) -> pd.DataFrame:
    """Attach a per-symbol ``list_date`` column when min_listing_days is enforced.

    Real (tushare) path only: a demo run has no listing dates, so the
    ``min_listing_days`` selection filter stays a DISCLOSED no-op there rather
    than fabricating ages. A missing list_date is carried as NaT (data gap; the
    filter keeps such names and the report discloses the count).
    """
    if int(cfg.universe.min_listing_days or 0) <= 0:
        return panel
    if cfg.data.source != "tushare":
        return panel  # demo: no listing dates -> filter is a disclosed no-op
    feed = TushareCovariatesFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    listing = feed.listing_dates(symbols)
    panel = enrich_listing(panel, listing)
    n_known = sum(1 for v in listing.values() if pd.notna(v))
    logger.info(
        "listing: list_date enriched (%d/%d known) for min_listing_days=%d",
        n_known, len(symbols), cfg.universe.min_listing_days,
    )
    return panel


def _compute_factor_panel(
    cfg: RootConfig,
    panel: pd.DataFrame,
    factors: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute every factor as its own column and persist the multi-column panel.

    One column per enabled factor (P3-1); a single-factor config produces the
    same one-column frame as before. All columns share the panel's
    MultiIndex(date, symbol), so downstream processing stays per-date and
    per-column.
    """
    columns = [factor.compute(panel).rename(factor.name) for factor in factors]
    factor_panel = pd.concat(columns, axis=1)
    _write_factor_panel(cfg, factor_panel)
    logger.info(
        "factors: %s computed (%d rows x %d columns)",
        [f.name for f in factors], len(factor_panel), factor_panel.shape[1],
    )
    return factor_panel


def _write_factor_panel(cfg: RootConfig, factor_panel: pd.DataFrame) -> None:
    """Persist the factor panel to ``factors/factors.parquet`` (re-entrant)."""
    target = Path(cfg.output.factor_dir) / "factors.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    flat = factor_panel.reset_index()
    tmp = target.with_suffix(".parquet.tmp")
    flat.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(target)


def _process_factors(
    cfg: RootConfig, factor_panel: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Run drop_missing + (winsorize) + neutralize + z-score from config toggles.

    Neutralization pulls the ``industry`` / ``market_cap`` covariates off the panel
    (placed there by :func:`_maybe_enrich_covariates`). When neutralize is enabled
    but they are absent, ``ProcessingPipeline`` raises a readable error.
    """
    industry = panel["industry"] if "industry" in panel.columns else None
    market_cap = panel["market_cap"] if "market_cap" in panel.columns else None
    pipeline = ProcessingPipeline(
        drop_missing=cfg.processing.drop_missing,
        standardize=cfg.processing.standardize.enabled,
        winsorize=cfg.processing.winsorize.enabled,
        neutralize=cfg.processing.neutralize.enabled,
        industry=industry,
        market_cap=market_cap,
    )
    return pipeline.transform(factor_panel)


def _maybe_enrich_covariates(
    cfg: RootConfig, panel: pd.DataFrame, symbols: list[str],
    logger: logging.Logger, cache=None,
) -> pd.DataFrame:
    """Attach industry + market_cap when neutralization is enabled (tushare only)."""
    if not cfg.processing.neutralize.enabled:
        return panel
    if cfg.data.source != "tushare":
        raise ValueError(
            "processing.neutralize is enabled but data.source is not 'tushare'; "
            "industry + market_cap need real data (demo has neither)."
        )
    feed = TushareCovariatesFeed(
        cfg.data.external_secret_file,
        token_key=cfg.data.tushare_token_key,
        cache=cache,
    )
    market_cap = feed.market_cap(symbols, cfg.data.start, cfg.data.end)
    panel = enrich_covariates(panel, market_cap=market_cap)
    # PIT (as-of) SW industry replaces the current-tag broadcast (UNI-010): each
    # name's industry varies by trade_date via index_member_all in/out dates at the
    # configured SW level; a name with no SW history gets NaN — a disclosed gap the
    # neutralizer drops — never a silent fallback to the current stock_basic tag.
    level = cfg.processing.neutralize.industry_level
    intervals = feed.pit_sw_intervals(symbols, level=level)
    industry_series = asof_industry(panel.index, intervals)
    panel = enrich_pit_industry(panel, industry_series)
    coverage = float(industry_series.notna().mean()) if len(industry_series) else 0.0
    logger.info(
        "covariates: PIT SW-%s industry (coverage %.1f%%, %d/%d symbols with history) "
        "+ market_cap(%d rows) for neutralization",
        level, coverage * 100.0, len(intervals), len(symbols), len(market_cap),
    )
    return panel


def _run_backtest(
    cfg: RootConfig,
    panel: pd.DataFrame,
    scores: pd.Series,
    factor_name: str,
    universe: Universe,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Wire the universe + scores + constructor + execution into the driver."""
    constructor = TopNEqualWeight(cfg.portfolio.top_n, long_only=cfg.portfolio.long_only)
    execution = SimExecution(fee_rate=cfg.cost.fee_rate)
    driver = BacktestDriver(
        universe=universe,
        scores=_FrameScores(scores),
        constructor=constructor,
        execution=execution,
        prices=panel,
        rebalance=cfg.backtest.rebalance,
        fee_rate=cfg.cost.fee_rate,
        initial_nav=cfg.backtest.initial_nav,
        cash_return=cfg.backtest.cash_return,
    )
    nav_table = driver.run()
    logger.info("backtest: %d rebalance rows", len(nav_table))
    return nav_table


def _factor_analytics(
    cfg: RootConfig,
    panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    score_panel: pd.Series,
) -> tuple[dict[str, dict], dict]:
    """Per-factor + combo-score IC / quantile analytics (simple, authoritative).

    Analytics is the only place forward returns are computed (INV-001); they are
    computed ONCE here and shared. Each RAW factor column gets its own IC summary
    / quantile returns / coverage (non-NaN fraction); the COMBO score — the
    processed equal-weight mean the backtest actually trades — gets the same
    treatment (P3-1). The first configured forward-return period is the holding
    horizon proxy. Returns ``(per_factor, combo)``.
    """
    periods = tuple(int(p) for p in cfg.analytics.forward_return_periods)
    fwd = forward_returns(panel, periods=periods)
    horizon = periods[0]
    fwd_col = fwd[f"forward_return_{horizon}d"]
    per_factor: dict[str, dict] = {}
    for name in factor_panel.columns:
        series = factor_panel[name]
        summary = ic_summary(compute_ic(series, fwd_col))
        per_factor[name] = {
            "ic_mean": summary["ic_mean"],
            "ic_ir": summary["ic_ir"],
            "quantile_returns": quantile_returns(
                series, fwd_col, quantiles=cfg.analytics.quantiles
            ),
            "coverage": float(series.notna().mean()) if len(series) else float("nan"),
        }
    combo_summary = ic_summary(compute_ic(score_panel, fwd_col))
    combo = {
        "ic_mean": combo_summary["ic_mean"],
        "ic_ir": combo_summary["ic_ir"],
        "quantile_returns": quantile_returns(
            score_panel, fwd_col, quantiles=cfg.analytics.quantiles
        ),
    }
    return per_factor, combo


def _standard_analytics(
    cfg: RootConfig,
    panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factor_name: str,
    nav_table: pd.DataFrame,
    ic_mean: float,
    ic_ir: float,
    perf: dict,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Report-only standard-library cross-check (quantstats + alphalens, P2-4).

    Reads the already-computed nav / factor and produces standard-tool metrics for
    the report; it NEVER feeds back into selection / portfolio / execution, so the
    backtest numbers are unchanged. Unavailable/erroring backends are disclosed and
    keep the authoritative simple fallback (no silent fake). The alphalens factor
    cross-check runs on the PRIMARY (first-enabled) raw factor column — the same
    series the legacy simple IC is reported on — so single-factor reports are
    byte-stable; per-factor / combo diagnostics come from the simple
    implementation (P3-1).
    """
    if not nav_table.empty:
        std_performance = quantstats_performance(
            nav_table["net_return"],
            periods_per_year=_periods_per_year(cfg.backtest.rebalance),
            simple_fallback=perf,
        )
    else:
        std_performance = {"backend": "skipped", **perf}
    prices_wide = panel["close"].unstack("symbol")
    horizon = int(cfg.analytics.forward_return_periods[0])
    std_factor = alphalens_factor_metrics(
        factor_panel[factor_name],
        prices_wide,
        quantiles=int(cfg.analytics.quantiles),
        period=horizon,
        simple_fallback={"ic_mean": ic_mean, "ic_ir": ic_ir},
    )
    logger.info(
        "standard analytics (report-only): quantstats=%s alphalens=%s",
        std_performance.get("backend"), std_factor.get("backend"),
    )
    return std_performance, std_factor
