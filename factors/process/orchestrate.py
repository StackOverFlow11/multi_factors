"""The single home of the factor-processing ORCHESTRATION (design §3.5 R9, D4).

``drop_missing + winsorize + neutralize + zscore`` is a cross-sectional,
covariate-dependent transform whose ORCHESTRATION previously lived in
``qt.pipeline._process_factors`` and was consumed by 10+ modules. The D5 unified
eval runner will build in the analytics layer, where red line #10 forbids
``analytics -> qt`` — so the orchestration cannot stay in ``qt``. Left there, the
eval and backtest paths would each copy it (the P3-6 ``drop_missing``-across-
columns lesson) or break layering.

It is homed HERE, in ``factors/process`` (where the ``neutralize_by_date``
primitive already lives), so the eval and backtest paths share ONE processing
truth. The covariates (industry / market_cap) are INJECTED by the caller —
``factors`` never touches a feed (red line #3) — and the store stays raw-only
(red line #7): processing is a consumer-side decision applied to the RAW factor
panel the service returns, never persisted as a source of truth.

Behavior is preserved verbatim: this constructs the SAME
:class:`~factors.process.pipeline.ProcessingPipeline` with the SAME arguments the
retired ``_process_factors`` did and calls ``transform`` — so the phase0 anchor
and every existing processing number are byte-for-byte unchanged.
"""

from __future__ import annotations

import pandas as pd

from factors.process.pipeline import ProcessingPipeline


def covariates_from_panel(
    panel: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Pull the ``industry`` / ``market_cap`` neutralization covariates off a panel.

    Returns ``(industry, market_cap)``, each ``None`` when the column is absent
    (the retired ``_process_factors`` did exactly this). The caller placed the
    covariates on the panel (e.g. ``_maybe_enrich_covariates``); ``factors`` never
    fetches them itself.
    """
    industry = panel["industry"] if "industry" in panel.columns else None
    market_cap = panel["market_cap"] if "market_cap" in panel.columns else None
    return industry, market_cap


def process_factor_panel(
    factor_panel: pd.DataFrame,
    *,
    drop_missing: bool = True,
    standardize: bool = True,
    winsorize: bool = False,
    neutralize: bool = False,
    industry: pd.Series | None = None,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the cross-sectional processing pipeline over a RAW factor panel.

    The ONE orchestration point (design §3.5 R9): eval and backtest both call
    this, so ``drop_missing`` (cross-column, per date) and neutralization can
    never drift between the two paths. Covariates are injected; when
    ``neutralize`` is enabled but they are absent, ``ProcessingPipeline`` raises a
    readable error rather than silently skipping.
    """
    pipeline = ProcessingPipeline(
        drop_missing=drop_missing,
        standardize=standardize,
        winsorize=winsorize,
        neutralize=neutralize,
        industry=industry,
        market_cap=market_cap,
    )
    return pipeline.transform(factor_panel)


__all__ = ["covariates_from_panel", "process_factor_panel"]
