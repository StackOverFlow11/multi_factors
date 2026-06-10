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


class DataCfg(_Strict):
    source: Literal["demo", "tushare"] = "demo"
    freq: str = "D"
    start: str
    end: str
    external_secret_file: str | None = None
    tushare_token_key: str = "tushare.token"
    output_name: str = "daily"

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
    event_order: str = "close_to_next_period"
    cash_return: float = 0.0


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
    oos: OOSCfg | None = None

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
