"""Internal pure frame helpers shared by the D3 quality checks.

No mutation of inputs, no I/O, no secrets — just index normalization and bounded
deterministic example extraction.
"""

from __future__ import annotations

import pandas as pd


def reset_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Promote a MultiIndex (e.g. (date, symbol) / (time, symbol)) to columns.

    Returns a NEW frame (``reset_index`` copies); a frame whose keys are already
    plain columns is returned unchanged. The input is never mutated.
    """
    if isinstance(df.index, pd.MultiIndex):
        return df.reset_index()
    return df


def row_examples(
    d: pd.DataFrame,
    mask,
    key_cols: list[str],
    *,
    extra: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Bounded, deterministic example rows for a boolean ``mask`` over ``d``.

    Sorted by ``key_cols`` so the sample is stable; values are returned raw
    (``make_finding`` cleans them). At most ``limit`` rows.
    """
    cols = list(key_cols) + ([extra] if extra else [])
    sub = d.loc[mask, cols].sort_values(list(key_cols)).head(limit)
    return [{c: row[c] for c in cols} for _, row in sub.iterrows()]
