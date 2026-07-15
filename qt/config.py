"""Pydantic v2 config models for the Phase 0 framework.

These mirror ``config/example.yaml`` (a.k.a. example_config_v1.yaml) exactly.
``load_config`` reads the YAML, validates it, and turns any pydantic validation
error into a user-readable message (CLI-003) — non-CS users must understand
what is wrong without reading a raw traceback.

Design note: this is the single source of truth for config field names.
Downstream agents read fields off ``RootConfig`` and its sub-models; they do not
re-parse the YAML.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _Strict(BaseModel):
    """Base model: forbid unknown keys so config typos surface early."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Sub-models (mirror the YAML block-by-block)
# --------------------------------------------------------------------------- #
class ProjectCfg(_Strict):
    name: str
    timezone: str = "Asia/Shanghai"


class SchemaGuardCfg(_Strict):
    """Default-off endpoint schema drift guard for the tushare cache (D-series).

    Disabled by DEFAULT — ``enabled=False`` keeps every existing config byte/
    behaviour identical (no guard is constructed, every cache parse site stays a
    passthrough). When ``enabled``, the cache runs report-only (or ``strict``,
    which raises on a HARD drift) checks on each endpoint's RAW source columns,
    parsed canonical columns, and stored-schema hash vs the ledger history. The
    guard sees only column + endpoint names — never a token or a data value.
    """

    enabled: bool = False
    mode: Literal["report_only", "strict"] = "report_only"


class CacheCfg(_Strict):
    """Persistent endpoint-level raw cache (P4-1 market bars + P4-2 universe/tradability).

    Disabled by DEFAULT for backward compatibility — an existing real config
    runs exactly as before until it opts in. When enabled the tushare-backed
    feeds read through the cache (read-through: only uncovered date ranges /
    stale snapshots hit the API). The cache stores RAW endpoint facts only
    (unadjusted OHLCV/amount, raw adj_factor, raw index_weight / suspend_d /
    stk_limit / namechange / stock_basic rows) — never qfq prices, never a
    derived tradability flag as source of truth, never any secret. The
    downstream transforms (``front_adjust``, raw price-limit checks, PIT as-of
    membership / industry / financials) still run in memory, unchanged.

    Cached: market bars (P4-1); index_weight / suspend_d / namechange / stk_limit
    / stock_basic (P4-2); daily_basic / fina_indicator / index_member_all (P4-3).
    The 21:00 ``data-update`` job warms these incrementally (see DataUpdateCfg).
    """

    enabled: bool = False
    root_dir: str = "artifacts/cache/tushare/v1"
    # Any requested range whose end is within this many days of "today" has its
    # recent tail refetched (recent rows can be corrected/delayed upstream).
    # Applies to the dense date-range endpoints (market bars, suspend_d,
    # stk_limit, index_weight).
    refresh_recent_days: int = 14
    # Snapshot / dimension endpoints (stock_basic, namechange) carry no date
    # range; they are refetched once their last successful fetch is older than
    # this many days (a slow-moving freshness policy). 0 disables staleness
    # refresh (only force_refresh re-pulls them).
    refresh_dimension_days: int = 30
    # Endpoint names (e.g. "market_daily", "index_weight") to always refetch in
    # full, ignoring coverage — for forcing a clean re-pull of one endpoint.
    force_refresh: list[str] = Field(default_factory=list)
    # D-series schema drift guard (default-off; see SchemaGuardCfg).
    schema_guard: SchemaGuardCfg = Field(default_factory=SchemaGuardCfg)

    @field_validator("refresh_recent_days")
    @classmethod
    def _check_recent_days(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"data.cache.refresh_recent_days must be >= 0; got {v}."
            )
        return v

    @field_validator("refresh_dimension_days")
    @classmethod
    def _check_dimension_days(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"data.cache.refresh_dimension_days must be >= 0; got {v}."
            )
        return v

    @field_validator("root_dir")
    @classmethod
    def _check_root_dir(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("data.cache.root_dir must be a non-empty path.")
        return v


class DataCfg(_Strict):
    source: Literal["demo", "tushare"] = "demo"
    freq: str = "D"
    start: str
    end: str
    external_secret_file: str | None = None
    tushare_token_key: str = "tushare.token"
    output_name: str = "daily"
    cache: CacheCfg = Field(default_factory=CacheCfg)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_date_to_str(cls, v: Any) -> Any:
        # YAML may parse unquoted dates as date objects; keep them as ISO strings.
        if isinstance(v, (_date, datetime)):
            return v.strftime("%Y-%m-%d")
        return v

    @model_validator(mode="after")
    def _check_date_order(self) -> "DataCfg":
        try:
            start = datetime.strptime(self.start, "%Y-%m-%d")
            end = datetime.strptime(self.end, "%Y-%m-%d")
        except ValueError as exc:  # pragma: no cover - exercised via load_config
            raise ValueError(
                f"data.start / data.end must be 'YYYY-MM-DD' dates; got "
                f"start={self.start!r}, end={self.end!r} ({exc})."
            ) from exc
        if start > end:
            raise ValueError(
                f"data.start ({self.start}) must be on or before data.end ({self.end})."
            )
        return self


class UniverseFilters(_Strict):
    missing_close: bool = True
    suspended: bool = False
    st: bool = False
    limit_up_down: bool = False


class UniverseCfg(_Strict):
    type: Literal["static", "index"] = "static"
    symbols: list[str] = Field(default_factory=list)
    index_code: str | None = None  # required when type == "index" (PIT membership)
    min_listing_days: int = 60
    filters: UniverseFilters = Field(default_factory=UniverseFilters)

    @model_validator(mode="after")
    def _check_type_requirements(self) -> "UniverseCfg":
        if self.type == "index" and not self.index_code:
            raise ValueError(
                "universe.type is 'index' but universe.index_code is not set "
                "(e.g. '000300.SH' for CSI300)."
            )
        return self


class FactorCfg(_Strict):
    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class StandardizeCfg(_Strict):
    enabled: bool = True
    method: Literal["zscore"] = "zscore"


class WinsorizeCfg(_Strict):
    enabled: bool = False
    method: str = "mad"
    n: float = 3.0


class NeutralizeCfg(_Strict):
    enabled: bool = False
    industry_col: str = "industry"
    size_col: str = "market_cap"
    # SW industry level for the PIT industry covariate (P2-3). Default L1 = the 31
    # broad SW sectors, the standard granularity for industry neutralization and the
    # safest on small cross-sections (more residual DOF than ~130 L2 sub-industries).
    # NOTE: going PIT necessarily switches the taxonomy from the old (non-PIT-able)
    # stock_basic.industry tag to SW — only SW carries in/out-date history — so the
    # backtest result changes vs the old tag regardless of level (L1 ≈ L2 in tests).
    industry_level: Literal["L1", "L2", "L3"] = "L1"


class ProcessingCfg(_Strict):
    drop_missing: bool = True
    standardize: StandardizeCfg = Field(default_factory=StandardizeCfg)
    winsorize: WinsorizeCfg = Field(default_factory=WinsorizeCfg)
    neutralize: NeutralizeCfg = Field(default_factory=NeutralizeCfg)


class AlphaCfg(_Strict):
    # equal_weight = P0 baseline (no future data); ic_weighted = P3-2
    # walk-forward rolling-IC weights (alpha layer only sees REALIZED history).
    model: Literal["equal_weight", "ic_weighted"] = "equal_weight"
    params: dict[str, Any] = Field(default_factory=dict)


class PortfolioCfg(_Strict):
    constructor: str = "topn_equal_weight"
    top_n: int
    long_only: bool = True
    max_weight: float | None = None
    turnover_cap: float | None = None

    @field_validator("top_n")
    @classmethod
    def _check_top_n(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"portfolio.top_n must be a positive integer; got {v}.")
        return v


class BacktestCfg(_Strict):
    initial_nav: float = 1.0
    rebalance: Literal["monthly"] = "monthly"
    # ``close_to_next_period`` = the daily close-to-close model (default, unchanged).
    # ``intraday_tail_rebalance`` = the I5a 14:50-decision / 14:51-execution model,
    # which requires ``intraday.enabled=true`` (enforced on RootConfig).
    event_order: Literal["close_to_next_period", "intraday_tail_rebalance"] = (
        "close_to_next_period"
    )
    cash_return: float = 0.0


# Execution models the intraday tail event model can price (I5a: just the
# conservative first one). Coarser/auction proxies are future work and rejected
# readably so a config can never silently fall back to a wrong model.
_SUPPORTED_INTRADAY_EXECUTION_MODELS: tuple[str, ...] = ("next_minute_close",)


def _parse_hms(value: str) -> int:
    """Parse an ``HH:MM:SS`` clock string to seconds-since-midnight (validation only)."""
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    h, m, s = (int(p) for p in parts)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"out-of-range clock value {value!r}")
    return h * 3600 + m * 60 + s


class LiquidityDiagnosticsCfg(_Strict):
    """Opt-in, report-only intraday execution liquidity diagnostics (I5f).

    OFF by default, so every existing config validates and behaves unchanged.
    When enabled, the intraday-tail report estimates — per desired rebalance trade
    — whether the SELECTED execution-minute 1min bar's traded ``amount`` (RMB) can
    absorb the trade at ``max_participation_rate``. This is REPORT-ONLY: it never
    changes fills, can_buy/can_sell, blocked reasons, target weights, achieved
    holdings, turnover, cost, NAV, factor scores, MMP grouping, alpha, or portfolio
    construction. Only ``mode='report_only'`` is supported; an enforcement mode
    fails readably (this layer must not move a real trade).
    """

    enabled: bool = False
    portfolio_notional: float | None = None
    max_participation_rate: float = 0.05
    mode: str = "report_only"

    @model_validator(mode="after")
    def _check(self) -> "LiquidityDiagnosticsCfg":
        if not self.enabled:
            return self
        if self.mode != "report_only":
            raise ValueError(
                "intraday.liquidity_diagnostics.mode only supports 'report_only' "
                "(report-only diagnostics never change fills/NAV; no enforcement "
                f"mode is implemented); got {self.mode!r}."
            )
        if self.portfolio_notional is None or float(self.portfolio_notional) <= 0.0:
            raise ValueError(
                "intraday.liquidity_diagnostics.portfolio_notional must be a "
                "positive RMB number when enabled (it scales the desired trade "
                f"notional); got {self.portfolio_notional!r}."
            )
        if not (0.0 < float(self.max_participation_rate) <= 1.0):
            raise ValueError(
                "intraday.liquidity_diagnostics.max_participation_rate must be in "
                f"(0, 1]; got {self.max_participation_rate!r}."
            )
        return self


class IntradayCfg(_Strict):
    """Opt-in intraday tail-rebalance event model declaration (I5a).

    Off by default; all daily configs validate unchanged. When the backtest's
    ``event_order`` is ``intraday_tail_rebalance`` this section must have
    ``enabled=true`` (enforced on RootConfig). ``decision_time`` is the signal
    cutoff (features must satisfy ``available_time <= decision_time``);
    ``execution_window`` is where the fill bar is taken; the window must start
    strictly after the decision and be non-empty.
    """

    enabled: bool = False
    decision_time: str = "14:50:00"
    data_lag: str = "1min"
    session_open: str = "09:30:00"
    execution_model: str = "next_minute_close"
    execution_window: tuple[str, str] = ("14:51:00", "14:56:59")
    require_cache_coverage: bool = True
    missing_execution: Literal["block"] = "block"
    # I5c: which PIT-safe daily intraday feature the tail-rebalance score uses.
    # Default "ret" reproduces the I5a/I5b smoke (intraday_ret_0930_1450); "mmp_ew"
    # selects the exploratory Minute Microstructure Pressure factor. The allowed
    # set mirrors data.clean.intraday_aggregate.INTRADAY_FEATURE_KEYS (a drift test
    # locks them equal); an unknown key fails readably at validation.
    score_feature: Literal[
        "ret", "realized_vol", "vwap", "last30m_ret", "mmp_ew"
    ] = "ret"
    # I5b execution-time price-limit feasibility. OFF by default, so every I5a /
    # daily config validates and behaves unchanged. When enabled, the intraday
    # tail model gates buys at the raw upper limit and sells at the raw lower
    # limit, comparing the selected execution-minute RAW close to raw ``stk_limit``
    # (never qfq / daily close). ``require_price_limit_coverage`` makes a missing
    # required (anchor date, symbol) limit row a hard, pre-result failure instead
    # of a silent "checked" pass; ``limit_tolerance`` is the raw-price equality
    # band for the at-limit test.
    price_limit_check: bool = False
    require_price_limit_coverage: bool = True
    limit_tolerance: float = 1e-6
    # I5f: opt-in, report-only execution liquidity diagnostics. Default-off nested
    # block; with the default it changes nothing (existing configs validate and
    # behave byte-identically).
    liquidity_diagnostics: LiquidityDiagnosticsCfg = Field(
        default_factory=LiquidityDiagnosticsCfg
    )

    @field_validator("limit_tolerance")
    @classmethod
    def _check_limit_tolerance(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"intraday.limit_tolerance must be >= 0; got {v!r}."
            )
        return v

    @model_validator(mode="after")
    def _check_execution(self) -> "IntradayCfg":
        if self.execution_model not in _SUPPORTED_INTRADAY_EXECUTION_MODELS:
            raise ValueError(
                f"intraday.execution_model {self.execution_model!r} is not "
                f"supported; choose one of {_SUPPORTED_INTRADAY_EXECUTION_MODELS}."
            )
        try:
            decision = _parse_hms(self.decision_time)
            start = _parse_hms(self.execution_window[0])
            end = _parse_hms(self.execution_window[1])
        except ValueError as exc:
            raise ValueError(
                f"intraday time fields must be HH:MM:SS clock strings: {exc}"
            ) from exc
        if not (start > decision and start <= end):
            raise ValueError(
                "intraday.execution_window must satisfy "
                "decision_time < window_start <= window_end (got "
                f"decision={self.decision_time}, window={list(self.execution_window)})."
            )
        return self


class CostCfg(_Strict):
    fee_rate: float = 0.001
    slippage_rate: float = 0.0
    turnover_formula: Literal["l1"] = "l1"


class AnalyticsCfg(_Strict):
    forward_return_periods: list[int] = Field(default_factory=lambda: [1, 5, 20])
    quantiles: int = 5
    benchmark: str | None = None


class OutputCfg(_Strict):
    root_dir: str = "artifacts"
    data_dir: str = "artifacts/data"
    factor_dir: str = "artifacts/factors"
    report_dir: str = "artifacts/reports"
    log_dir: str = "artifacts/logs"
    overwrite: bool = True
    # Filename for the real-baseline report (run-phase2-baseline). None keeps the
    # historical default 'phase2_real_baseline.md'; a multi-factor baseline config
    # sets its own name so it never overwrites the phase2 report (P3-1).
    baseline_report_name: str | None = None
    # Filename for the subset-validation report (run-phase3-subset). None keeps
    # the historical default 'phase3_subset_validation.md'; a config sets its
    # own name so different studies sharing the run mode never overwrite each
    # other's report (P3-8; the same precedent as baseline_report_name).
    subset_report_name: str | None = None
    # H1 title for the subset-validation report (run-phase3-subset). None keeps
    # the renderer's sample-aware default (P3-7 independent / P3-6 post-hoc); a
    # config that reuses the run mode for a DIFFERENT study (e.g. the P3-8 CSI500
    # generalization check) sets its own title so the report header names the
    # actual study instead of the machinery's default phase label (P3-8).
    subset_report_title: str | None = None
    # Report/log basename for the intraday tail smoke (run-phase-i5a-intraday).
    # None keeps the historical 'phase_i5a_intraday_tail_framework'; the I5b
    # execution-feasibility config sets its own so it never overwrites the
    # accepted I5a artifact (same precedent as baseline_report_name).
    intraday_report_name: str | None = None
    # H1 title for the intraday tail report. None keeps the renderer's
    # study-aware default (I5c for the MMP factor / I5b when price-limit on / else
    # I5a); a config that reuses the runner for a DIFFERENT study sets its own so
    # the heading names the actual study, not a stale phase label (I5c precedent,
    # same as subset_report_title).
    intraday_report_title: str | None = None


class OOSCfg(_Strict):
    """P3-3 out-of-sample split: train = [data.start, split_date), test =
    [split_date, data.end]. Evaluation is walk-forward (rolling subperiod):
    weights at any date use only observations realized by that date, so no
    test-period forward return can reach a train-period computation."""

    split_date: str

    @field_validator("split_date", mode="before")
    @classmethod
    def _coerce_date_to_str(cls, v: Any) -> Any:
        if isinstance(v, (_date, datetime)):
            return v.strftime("%Y-%m-%d")
        return v


class RobustnessWindowCfg(_Strict):
    """One time fold of the P3-4 robustness matrix: [start, end] split at split."""

    label: str
    start: str
    end: str
    split: str

    @field_validator("start", "end", "split", mode="before")
    @classmethod
    def _coerce_date_to_str(cls, v: Any) -> Any:
        if isinstance(v, (_date, datetime)):
            return v.strftime("%Y-%m-%d")
        return v

    @model_validator(mode="after")
    def _check_window(self) -> "RobustnessWindowCfg":
        try:
            start = datetime.strptime(self.start, "%Y-%m-%d")
            end = datetime.strptime(self.end, "%Y-%m-%d")
            split = datetime.strptime(self.split, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"robustness window {self.label!r}: start/end/split must be "
                f"'YYYY-MM-DD' dates ({exc})."
            ) from exc
        if not (start < split < end):
            raise ValueError(
                f"robustness window {self.label!r}: split ({self.split}) must lie "
                f"STRICTLY inside [{self.start}, {self.end}] so both subperiods "
                "are non-empty."
            )
        return self


class RobustnessSkipCfg(_Strict):
    """One EXPLICITLY skipped matrix cell (runtime budget; disclosed in report)."""

    universe: str
    window: str  # a windows[].label


class RobustnessCfg(_Strict):
    """P3-4 robustness matrix: every universe × window pair is one OOS cell.

    ``skip_cells`` removes named cells from the run (e.g. a wide-universe long
    fold whose rate-limited pull would blow the runtime budget); skipped cells
    are DISCLOSED in the report — coverage is never silently reduced.
    """

    universes: list[str]
    windows: list[RobustnessWindowCfg]
    skip_cells: list[RobustnessSkipCfg] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_non_empty(self) -> "RobustnessCfg":
        if not self.universes:
            raise ValueError("robustness.universes must list at least one index code.")
        if not self.windows:
            raise ValueError("robustness.windows must list at least one window.")
        labels = [w.label for w in self.windows]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(
                f"robustness.windows labels must be unique; duplicate(s): {dupes}."
            )
        for skip in self.skip_cells:
            if skip.universe not in self.universes:
                raise ValueError(
                    f"robustness.skip_cells references unknown universe "
                    f"{skip.universe!r} (declared: {self.universes})."
                )
            if skip.window not in labels:
                raise ValueError(
                    f"robustness.skip_cells references unknown window label "
                    f"{skip.window!r} (declared: {labels})."
                )
        run_cells = len(self.universes) * len(self.windows) - len(self.skip_cells)
        if run_cells < 1:
            raise ValueError(
                "robustness.skip_cells removes every cell; at least one cell "
                "must remain to run."
            )
        return self


class FactorGroupCfg(_Strict):
    """One named factor group of the P3-6 subset validation.

    ``factors`` must reference ENABLED entries of the top-level ``factors``
    list (checked at the RootConfig level — a disabled or unknown name has no
    raw factor-panel column to subset). Each group is re-processed
    INDEPENDENTLY from the shared raw factor panel, so ``drop_missing``
    applies per group, exactly as if the group were the configured factor set.
    """

    label: str
    factors: list[str]

    @model_validator(mode="after")
    def _check_factors(self) -> "FactorGroupCfg":
        if not self.factors:
            raise ValueError(
                f"subset_validation group {self.label!r} must list at least one factor."
            )
        dupes = sorted({f for f in self.factors if self.factors.count(f) > 1})
        if dupes:
            raise ValueError(
                f"subset_validation group {self.label!r} lists duplicate factor(s) "
                f"{dupes}; each factor may appear once per group."
            )
        return self


class CostScenarioCfg(_Strict):
    """One trading-cost scenario: ``cost.fee_rate`` × ``fee_multiplier``.

    Scenarios scale the fee ONLY — scores and fills never see the fee, so the
    trades (and turnover) are identical across scenarios; only the cost line
    (and thus net return) changes.
    """

    label: str
    fee_multiplier: float

    @field_validator("fee_multiplier")
    @classmethod
    def _check_multiplier(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"cost scenario fee_multiplier must be positive; got {v!r}."
            )
        return v


class IndependentCellCfg(_Strict):
    """One matrix cell declared a GENUINELY INDEPENDENT holdout (P3-7).

    The declaration is explicit and human-made — the machine cannot know which
    data took part in factor screening. Cells not listed are labeled
    screened/post-hoc in the report; the independent verdict reads ONLY the
    declared cells.
    """

    universe: str
    window: str  # a robustness.windows[].label


class SubsetValidationCfg(_Strict):
    """P3-6 subset validation: factor groups × cost scenarios over the matrix.

    A multiplier-1.0 scenario is REQUIRED: it anchors the cost-drag comparison
    (every other scenario reads as "the same trades at k× the fee").

    P3-7 adds the independent-sample dimension: ``independent_cells`` labels
    holdout cells (everything else is screened/post-hoc), ``hypotheses`` fixes
    the expected IC sign per factor BEFORE the run (a factual sign check, not a
    return claim), and ``min_rebalances`` gates sample sufficiency (too few
    settled rebalances → an INSUFFICIENT-DATA verdict, never a silent pass).
    """

    groups: list[FactorGroupCfg]
    cost_scenarios: list[CostScenarioCfg] = Field(
        default_factory=lambda: [CostScenarioCfg(label="base", fee_multiplier=1.0)]
    )
    independent_cells: list[IndependentCellCfg] = Field(default_factory=list)
    hypotheses: dict[str, Literal["positive", "negative"]] = Field(
        default_factory=dict
    )
    min_rebalances: int = 8

    @field_validator("min_rebalances")
    @classmethod
    def _check_min_rebalances(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"subset_validation.min_rebalances must be a positive integer; "
                f"got {v!r}."
            )
        return v

    @model_validator(mode="after")
    def _check_sections(self) -> "SubsetValidationCfg":
        if not self.groups:
            raise ValueError(
                "subset_validation.groups must list at least one group."
            )
        labels = [g.label for g in self.groups]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(
                f"subset_validation group labels must be unique; duplicate(s): {dupes}."
            )
        scn_labels = [s.label for s in self.cost_scenarios]
        scn_dupes = sorted({x for x in scn_labels if scn_labels.count(x) > 1})
        if scn_dupes:
            raise ValueError(
                f"subset_validation cost scenario labels must be unique; "
                f"duplicate(s): {scn_dupes}."
            )
        if not any(s.fee_multiplier == 1.0 for s in self.cost_scenarios):
            raise ValueError(
                "subset_validation.cost_scenarios must include one scenario with "
                "fee_multiplier == 1.0 (the base anchor for the cost-drag "
                "comparison)."
            )
        return self


_DATA_UPDATE_ENDPOINTS = frozenset({
    "market_daily", "adj_factor", "index_weight", "suspend_d", "namechange",
    "stk_limit", "stock_basic", "daily_basic", "fina_indicator",
    "index_member_all", "stk_mins_1min",
})

# D3b quality can only check the STRUCTURAL endpoints the updater already loads as
# in-memory frames (market bars + 1min minutes). Universe / financial / dimension
# endpoints are not re-materialized as frames here, so they are out of scope.
_DATA_UPDATE_QUALITY_ENDPOINTS = frozenset({
    "market_daily", "adj_factor", "stk_mins_1min",
})


class DataUpdateQualityCfg(_Strict):
    """D3b report-only data-quality hook for ``data-update`` (default OFF).

    When ``enabled`` the updater runs the accepted D3 ``data/quality`` STRUCTURAL
    checks on the frames it ALREADY warmed (no extra API call) and writes a
    deterministic Markdown report under ``output.report_dir``. It is report-only:
    it never filters / repairs / mutates data, never fails the job, never changes
    cache coverage or the per-endpoint request summary. With ``enabled=false``
    (the default) every existing config behaves exactly as before.
    """

    enabled: bool = False
    # Which warmed surfaces to quality-check (only the structural ones above).
    endpoints: list[str] = Field(
        default_factory=lambda: ["market_daily", "adj_factor", "stk_mins_1min"]
    )
    # A BARE filename written under output.report_dir (never absolute, never a
    # path with separators or '..').
    report_name: str = "data_update_quality_report.md"

    @field_validator("endpoints")
    @classmethod
    def _check_quality_endpoints(cls, v: list[str]) -> list[str]:
        unknown = [e for e in v if e not in _DATA_UPDATE_QUALITY_ENDPOINTS]
        if unknown:
            raise ValueError(
                f"data_update.quality.endpoints {unknown} unknown; must be in "
                f"{sorted(_DATA_UPDATE_QUALITY_ENDPOINTS)}."
            )
        return v

    @field_validator("report_name")
    @classmethod
    def _check_report_name(cls, v: str) -> str:
        name = str(v)
        if not name.strip():
            raise ValueError(
                "data_update.quality.report_name must be a non-empty filename."
            )
        if (
            "/" in name
            or "\\" in name
            or ".." in name
            or name in (".", "..")
        ):
            raise ValueError(
                "data_update.quality.report_name must be a bare filename under "
                "output.report_dir (no path separators, no '..', not absolute); "
                f"got {name!r}."
            )
        return name


class DataUpdateConcurrencyCfg(_Strict):
    """D5 opt-in bounded concurrency for ``data-update`` cache warms (default serial).

    ``max_workers=1`` (the default) keeps the warm fully serial — byte/behavior
    compatible with every existing config. ``max_workers>1`` fans the per-symbol gap
    FETCH stage onto a bounded thread pool, all funneled through ONE shared global
    rate limiter (reusing ``data_update.rate_limit_per_min``) so the Tushare quota is
    never multiplied per thread; store upserts and ledger writes still happen in
    deterministic main-thread order.
    """

    max_workers: int = 1

    @field_validator("max_workers")
    @classmethod
    def _check_max_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"data_update.concurrency.max_workers must be >= 1 (1 == serial); "
                f"got {v}."
            )
        return v


class DataUpdateCfg(_Strict):
    """Standalone data-updater section (P4-3) — consumed ONLY by ``data-update``.

    Declares the daily 21:00 (Asia/Shanghai) incremental warm: which endpoints to
    refresh, the lookback/tail policy, the not-ready pending window, and a
    conservative rate limit. Scheduling is external (systemd timer / cron calls
    the CLI); this is purely the job's parameters. Optional on RootConfig, so
    every existing config still validates unchanged.
    """

    timezone: str = "Asia/Shanghai"
    scheduled_start: str = "21:00:00"
    # Which universe the warm resolves. "config" (default) = the existing
    # behavior (static symbols / union of index constituents via ``universe``),
    # byte-identical. "all_a" = the WHOLE listed A-share market from the
    # stock_basic snapshot (post-market all-A auto-fetch). Only ``data-update``
    # reads this; it never leaks into the backtest universe (UniverseCfg).
    universe_scope: Literal["config", "all_a"] = "config"
    endpoints: list[str] = Field(default_factory=list)
    index_codes: list[str] = Field(default_factory=list)
    lookback_days: int = 400
    tail_refresh_days: int = 14
    not_ready_days: int = 1
    fina_tail_days: int = 400
    fina_fields: list[str] = Field(default_factory=lambda: ["roe", "netprofit_yoy"])
    rate_limit_per_min: int = 450
    force_refresh: list[str] = Field(default_factory=list)
    # D3b report-only quality hook (default OFF; see DataUpdateQualityCfg).
    quality: DataUpdateQualityCfg = Field(default_factory=DataUpdateQualityCfg)
    # D5 opt-in bounded concurrency (default max_workers=1 == serial).
    concurrency: DataUpdateConcurrencyCfg = Field(
        default_factory=DataUpdateConcurrencyCfg
    )

    @field_validator("endpoints", "force_refresh")
    @classmethod
    def _check_endpoints(cls, v: list[str]) -> list[str]:
        unknown = [e for e in v if e not in _DATA_UPDATE_ENDPOINTS]
        if unknown:
            raise ValueError(
                f"data_update endpoint(s) {unknown} unknown; "
                f"must be in {sorted(_DATA_UPDATE_ENDPOINTS)}."
            )
        return v

    @field_validator(
        "lookback_days", "tail_refresh_days", "not_ready_days",
        "fina_tail_days", "rate_limit_per_min",
    )
    @classmethod
    def _check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"data_update integer fields must be >= 0; got {v}.")
        return v

    @field_validator("rate_limit_per_min")
    @classmethod
    def _check_rate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"data_update.rate_limit_per_min must be > 0; got {v}.")
        return v


class RootConfig(_Strict):
    """Top-level config composing every section.

    Required top-level sections (CFG-002): data, universe, factors, alpha,
    portfolio, backtest, cost, output. ``project``, ``processing`` and
    ``analytics`` have sensible defaults but are present in the template.
    ``oos`` is optional and only consumed by ``run-phase3-oos``.
    """

    project: ProjectCfg = Field(default_factory=lambda: ProjectCfg(name="quantitative_trading"))
    data: DataCfg
    universe: UniverseCfg
    factors: list[FactorCfg]
    processing: ProcessingCfg = Field(default_factory=ProcessingCfg)
    alpha: AlphaCfg
    portfolio: PortfolioCfg
    backtest: BacktestCfg
    cost: CostCfg
    analytics: AnalyticsCfg = Field(default_factory=AnalyticsCfg)
    output: OutputCfg
    # P-I5a intraday tail event model (consumed only by run-phase-i5a-intraday and
    # any backtest with event_order='intraday_tail_rebalance'). Off by default.
    intraday: IntradayCfg | None = None
    oos: OOSCfg | None = None
    # P3-4 robustness matrix (consumed only by run-phase3-robustness).
    robustness: RobustnessCfg | None = None
    # P3-6 subset validation (consumed only by run-phase3-subset).
    subset_validation: SubsetValidationCfg | None = None
    # P4-3 data updater (consumed only by data-update).
    data_update: DataUpdateCfg | None = None

    @model_validator(mode="after")
    def _check_intraday_event_order(self) -> "RootConfig":
        """``intraday_tail_rebalance`` requires an enabled ``intraday`` section.

        The daily default (``close_to_next_period``) needs no intraday section, so
        every existing config validates unchanged. Selecting the intraday event
        model without ``intraday.enabled=true`` is a configuration error (it would
        otherwise silently run the daily model).
        """
        if self.backtest.event_order == "intraday_tail_rebalance":
            if self.intraday is None or not self.intraday.enabled:
                raise ValueError(
                    "backtest.event_order='intraday_tail_rebalance' requires an "
                    "'intraday' section with enabled=true; got "
                    f"intraday={'None' if self.intraday is None else 'enabled=false'}."
                )
        return self

    @model_validator(mode="after")
    def _check_subset_groups_reference_enabled_factors(self) -> "RootConfig":
        if self.subset_validation is None:
            return self
        enabled = {f.name for f in self.factors if f.enabled}
        for group in self.subset_validation.groups:
            unknown = [f for f in group.factors if f not in enabled]
            if unknown:
                raise ValueError(
                    f"subset_validation group {group.label!r} references factor(s) "
                    f"{unknown} that are not ENABLED entries of config.factors "
                    f"(enabled: {sorted(enabled)}). A disabled or unknown factor "
                    "has no raw factor-panel column to subset."
                )
        bad_hyp = [f for f in self.subset_validation.hypotheses if f not in enabled]
        if bad_hyp:
            raise ValueError(
                f"subset_validation.hypotheses references factor(s) {bad_hyp} that "
                f"are not ENABLED entries of config.factors (enabled: "
                f"{sorted(enabled)}); a hypothesis needs a raw factor-panel column "
                "to check."
            )
        return self

    @model_validator(mode="after")
    def _check_independent_cells(self) -> "RootConfig":
        if self.subset_validation is None or not self.subset_validation.independent_cells:
            return self
        if self.robustness is None:
            raise ValueError(
                "subset_validation.independent_cells references matrix cells, but "
                "there is no 'robustness' section declaring universes/windows."
            )
        labels = [w.label for w in self.robustness.windows]
        skipped = {(s.universe, s.window) for s in self.robustness.skip_cells}
        for cell in self.subset_validation.independent_cells:
            if cell.universe not in self.robustness.universes:
                raise ValueError(
                    f"subset_validation.independent_cells references unknown "
                    f"universe {cell.universe!r} (declared: {self.robustness.universes})."
                )
            if cell.window not in labels:
                raise ValueError(
                    f"subset_validation.independent_cells references unknown "
                    f"window label {cell.window!r} (declared: {labels})."
                )
            if (cell.universe, cell.window) in skipped:
                raise ValueError(
                    f"subset_validation.independent_cells declares "
                    f"{cell.universe!r}|{cell.window!r} an independent holdout, but "
                    "that cell is skip_cells-listed and never runs — an independent "
                    "validation that does not run is a contradiction."
                )
        return self

    @model_validator(mode="after")
    def _check_oos_split_inside_window(self) -> "RootConfig":
        if self.oos is None:
            return self
        try:
            split = datetime.strptime(self.oos.split_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"oos.split_date must be a 'YYYY-MM-DD' date; got "
                f"{self.oos.split_date!r} ({exc})."
            ) from exc
        start = datetime.strptime(self.data.start, "%Y-%m-%d")
        end = datetime.strptime(self.data.end, "%Y-%m-%d")
        if not (start < split < end):
            raise ValueError(
                f"oos.split_date ({self.oos.split_date}) must lie STRICTLY inside "
                f"the data window ({self.data.start}, {self.data.end}) so both the "
                "train and test subperiods are non-empty."
            )
        return self


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """User-readable configuration error (CLI-003)."""


# Map machine field names to a friendly hint for required-field errors.
_REQUIRED_HINTS = {
    "data": "the 'data' section (source/start/end)",
    "universe": "the 'universe' section (type/symbols)",
    "factors": "the 'factors' list (e.g. [{name: momentum_20}])",
    "alpha": "the 'alpha' section (model)",
    "portfolio": "the 'portfolio' section (constructor/top_n)",
    "backtest": "the 'backtest' section (rebalance)",
    "cost": "the 'cost' section (fee_rate)",
    "output": "the 'output' section (root_dir)",
    "start": "data.start (a 'YYYY-MM-DD' date)",
    "end": "data.end (a 'YYYY-MM-DD' date)",
    "top_n": "portfolio.top_n (a positive integer)",
}


def _format_validation_error(err: ValidationError) -> str:
    """Turn a pydantic ValidationError into a readable, multi-line message."""
    lines: list[str] = ["Invalid configuration:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        leaf = str(e["loc"][-1]) if e["loc"] else ""
        msg = e["msg"]
        if e["type"] == "missing":
            hint = _REQUIRED_HINTS.get(leaf, f"'{loc}'")
            lines.append(f"  - missing required field: {hint}")
        else:
            lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def load_config(path: str) -> RootConfig:
    """Read a YAML config file and return a validated ``RootConfig``.

    Raises ``ConfigError`` with a user-readable message (never a raw pydantic
    traceback) if the file is missing, unparseable, or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML ({path}): {exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config root must be a mapping of sections; got {type(raw).__name__} in {path}."
        )

    try:
        return RootConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
