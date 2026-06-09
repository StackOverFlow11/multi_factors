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

from alpha.equal_weight import EqualWeightAlpha
from analytics.factor import compute_ic, forward_returns, ic_summary, quantile_returns
from analytics.performance import performance_summary
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
    factor_name: str
    ic_mean: float
    ic_ir: float
    quantile_returns: pd.DataFrame
    nav_table: pd.DataFrame
    performance: dict[str, float]
    avg_turnover: float
    cost_drag: float
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


def _build_scores(
    processed: pd.DataFrame, alpha: EqualWeightAlpha
) -> pd.Series:
    """Predict one score per (date, symbol) from the processed factor panel.

    The alpha is fit with ``forward_returns=None`` (equal-weight needs no future
    data) and predicts per-date cross-sections, so a single (date, symbol) score
    panel is produced for the backtest scores source.
    """
    alpha.fit(processed, None)
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
    factor_name = cfg.factors[0].name if cfg.factors else "?"
    is_financial = factor_name in SUPPORTED_FINANCIAL_FIELDS

    if real:
        parts = ["front-adjusted (qfq) prices"]
        parts.append(
            f"PIT index membership ({cfg.universe.index_code})"
            if cfg.universe.type == "index"
            else "STATIC universe (still a PIT downgrade — see below)"
        )
        parts.append(
            f"ann_date-aligned financial factor '{factor_name}'"
            if is_financial
            else f"price factor '{factor_name}'"
        )
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
    if is_financial:
        financials = (
            f"Financial factor '{factor_name}' is ann_date PIT-aligned (DATA-012): "
            "a figure is used only after its disclosure date, never by report period."
        )
    else:
        financials = (
            "No financial factor in this run; ann_date alignment is implemented and "
            "available but unused here (price factor only)."
        )

    # neutralization
    if cfg.processing.neutralize.enabled and real:
        neutral = (
            "Factor is industry + market-cap neutralized; industry is **point-in-time** "
            "SW-L1 (UNI-010): as-of the trade date via index_member_all in/out dates, NOT "
            "the current stock_basic tag. Names with no SW history get NaN (a disclosed "
            "coverage gap the neutralizer drops) — never a silent current-tag fallback. "
            "Market cap is per-date (daily_basic.total_mv)."
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
        neutral,
        execution,
        f"Daily ({cfg.data.freq}) bars only; minute-level link is deferred.",
        "IC / quantile returns use a simple numpy/pandas implementation, NOT "
        "alphalens-reloaded (simple-vs-alphalens fallback, INV-007).",
        "Performance metrics use a simple numpy/pandas implementation, NOT "
        "quantstats (simple-vs-quantstats fallback, INV-007).",
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

    universe, symbols = _build_universe(cfg, logger)
    panel = _load_panel(cfg, symbols, logger)
    factor = _build_factor(cfg)
    panel = _maybe_enrich_financials(cfg, panel, symbols, factor, logger)
    panel = _maybe_enrich_covariates(cfg, panel, symbols, logger)
    panel = _maybe_enrich_listing(cfg, panel, symbols, logger)
    factor_panel = _compute_factor_panel(cfg, panel, factor, logger)

    processed = _process_factors(cfg, factor_panel, panel)
    scores = _build_scores(processed, EqualWeightAlpha())

    nav_table = _run_backtest(cfg, panel, scores, factor.name, universe, logger)
    ic_mean, ic_ir, q_returns = _factor_analytics(cfg, panel, factor_panel, factor.name)
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

    downgrades = _collect_downgrades(cfg)
    result = Phase0Result(
        config=cfg,
        panel_rows=len(panel),
        panel_symbols=panel.index.get_level_values("symbol").nunique(),
        factor_name=factor.name,
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        quantile_returns=q_returns,
        nav_table=nav_table,
        performance=perf,
        avg_turnover=avg_turnover,
        cost_drag=cost_drag,
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
def _build_feed(cfg: RootConfig) -> DataFeed:
    """Construct the DataFeed for the configured source (``demo`` | ``tushare``).

    Dispatches on ``cfg.data.source`` so a ``tushare`` config is NOT silently
    served demo data. The tushare branch requires ``external_secret_file`` (the
    token never lives in the repo); constructing the feed performs no network /
    token read — that is deferred to first ``get_bars`` (SEC-001/004).
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
        )
    raise ValueError(
        f"Unsupported data.source {cfg.data.source!r}; expected 'demo' or 'tushare'."
    )


def _build_universe(
    cfg: RootConfig, logger: logging.Logger
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
            cfg.data.external_secret_file, token_key=cfg.data.tushare_token_key
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


def _load_panel(
    cfg: RootConfig, symbols: list[str], logger: logging.Logger
) -> pd.DataFrame:
    """Fetch the market panel for ``symbols``, persist it, and read it back."""
    if not symbols:
        raise ValueError("No symbols to load; the universe produced an empty set.")
    feed = _build_feed(cfg)
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
    panel = _enrich_tradability(cfg, panel, symbols, logger)
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
    cfg: RootConfig, panel: pd.DataFrame, symbols: list[str], logger: logging.Logger
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
        cfg.data.external_secret_file, token_key=cfg.data.tushare_token_key
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


def _build_factor(cfg: RootConfig):
    """Instantiate the (single, first-enabled) factor from config by name.

    ``momentum*`` -> :class:`MomentumFactor` (price-based, P0). A financial field
    name (e.g. ``roe``, ``netprofit_yoy``) -> :class:`FinancialFactor`, which needs
    the tushare data path (ann_date alignment) — not available for demo data.
    """
    enabled = [f for f in cfg.factors if f.enabled]
    if not enabled:
        raise ValueError("No enabled factor in config.factors; phase0 needs a factor.")
    spec = enabled[0]
    params = dict(spec.params)
    if spec.name in SUPPORTED_FINANCIAL_FIELDS:
        return FinancialFactor(field=spec.name)
    if spec.name.startswith("momentum"):
        return MomentumFactor(
            window=int(params.get("window", 20)),
            price_col=str(params.get("price_col", "close")),
        )
    raise ValueError(
        f"Unknown factor {spec.name!r}; expected 'momentum*' or one of "
        f"{SUPPORTED_FINANCIAL_FIELDS}."
    )


def _maybe_enrich_financials(
    cfg: RootConfig,
    panel: pd.DataFrame,
    symbols: list[str],
    factor,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Attach the ann_date-aligned financial column when a financial factor is used.

    Financial factors require the tushare data path: a demo run has no disclosure
    dates, so we fail with a readable error rather than fabricate financials.
    """
    if not isinstance(factor, FinancialFactor):
        return panel
    if cfg.data.source != "tushare":
        raise ValueError(
            f"Factor '{factor.name}' is a financial factor and needs real financial "
            f"data (data.source='tushare'); it cannot run on demo data."
        )
    feed = TushareFinancialFeed(
        cfg.data.external_secret_file, token_key=cfg.data.tushare_token_key
    )
    # Look back before start so the prior already-disclosed report is fetched and
    # can be as-of carried forward onto the early trade dates (no NaN gap).
    fetch_start = (
        pd.Timestamp(cfg.data.start) - pd.Timedelta(days=_FINANCIAL_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    fina = feed.get_fina_indicator(symbols, fetch_start, cfg.data.end, fields=[factor.name])
    enriched = panel.copy()
    aligned = asof_financials(panel.index, fina, [factor.name])
    enriched[factor.name] = aligned[factor.name]
    logger.info(
        "financials: as-of aligned '%s' by ann_date (%d disclosed rows)",
        factor.name, len(fina),
    )
    return enriched


def _maybe_enrich_listing(
    cfg: RootConfig, panel: pd.DataFrame, symbols: list[str], logger: logging.Logger
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
        cfg.data.external_secret_file, token_key=cfg.data.tushare_token_key
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
    factor: MomentumFactor,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute the factor series, frame it, and persist it via PanelStore-style write."""
    series = factor.compute(panel)
    factor_panel = series.to_frame(name=factor.name)
    _write_factor_panel(cfg, factor_panel)
    logger.info("factor: %s computed (%d rows)", factor.name, len(factor_panel))
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
    cfg: RootConfig, panel: pd.DataFrame, symbols: list[str], logger: logging.Logger
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
        cfg.data.external_secret_file, token_key=cfg.data.tushare_token_key
    )
    market_cap = feed.market_cap(symbols, cfg.data.start, cfg.data.end)
    panel = enrich_covariates(panel, market_cap=market_cap)
    # PIT (as-of) SW-L1 industry replaces the current-tag broadcast (UNI-010): each
    # name's industry varies by trade_date via index_member_all in/out dates; a name
    # with no SW history gets NaN — a disclosed gap the neutralizer drops — never a
    # silent fallback to the current stock_basic tag.
    intervals = feed.pit_sw_l1_intervals(symbols)
    industry_series = asof_industry(panel.index, intervals)
    panel = enrich_pit_industry(panel, industry_series)
    coverage = float(industry_series.notna().mean()) if len(industry_series) else 0.0
    logger.info(
        "covariates: PIT SW-L1 industry (coverage %.1f%%, %d/%d symbols with history) "
        "+ market_cap(%d rows) for neutralization",
        coverage * 100.0, len(intervals), len(symbols), len(market_cap),
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
    factor_name: str,
) -> tuple[float, float, pd.DataFrame]:
    """IC summary + quantile returns from the RAW factor + forward returns.

    Analytics is the only place forward returns are computed (INV-001). We use
    the first configured forward-return period for IC (the holding horizon proxy).
    """
    periods = tuple(int(p) for p in cfg.analytics.forward_return_periods)
    fwd = forward_returns(panel, periods=periods)
    horizon = periods[0]
    fwd_col = fwd[f"forward_return_{horizon}d"]
    factor_series = factor_panel[factor_name]
    ic = compute_ic(factor_series, fwd_col)
    summary = ic_summary(ic)
    q_returns = quantile_returns(factor_series, fwd_col, quantiles=cfg.analytics.quantiles)
    return summary["ic_mean"], summary["ic_ir"], q_returns
