"""Join industry + market-cap covariates onto the panel (for neutralization).

Adds two optional columns the neutralizer consumes:

  * ``industry``   — per-symbol industry tag, broadcast to every date.
  * ``market_cap`` — per (date, symbol) total market value.

Only the provided covariates are added; pure (never mutates the input panel).
"""

from __future__ import annotations

import pandas as pd

from data.clean.schema import validate_panel


def enrich_covariates(
    panel: pd.DataFrame,
    *,
    industry: dict[str, str] | None = None,
    market_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a NEW panel with ``industry`` / ``market_cap`` columns added."""
    validate_panel(panel)
    out = panel.copy()
    symbols = out.index.get_level_values("symbol")

    if industry is not None:
        out["industry"] = [industry.get(str(s)) for s in symbols]

    if market_cap is not None and not market_cap.empty:
        mc = market_cap.copy()
        mc["date"] = pd.to_datetime(mc["date"]).dt.normalize()
        mc["symbol"] = mc["symbol"].astype(str)
        mc = mc.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
        out["market_cap"] = mc["market_cap"].reindex(out.index)

    return out
