"""Parquet-backed store for canonical (date, symbol) panels (DATA-008/009/010).

A :class:`PanelStore` persists a normalized panel to a single parquet file and
reads it back, optionally filtering by a **closed** date interval ``[start, end]``
and/or a symbol subset. The round-trip is loss-free with respect to the panel
contract in :mod:`data.clean.schema`: the MultiIndex(date, symbol), column set and
order, sort order, dtypes, and values (NaN cells included) all survive.

Design notes
------------
- P0 layout: one file per logical panel name -> ``<root>/<name>.parquet``.
- Parquet cannot store a MultiIndex directly, so we ``reset_index`` before
  writing and ``set_index`` + re-normalize after reading. Re-normalizing through
  :func:`normalize_panel` guarantees the read panel obeys the same contract as a
  freshly built one (sorted, datetime date level, str symbols).
- The store never mutates its inputs and never returns a view onto internal
  state — every read produces a fresh frame.
- This layer is pure storage: it does not fetch data, compute factors, or touch
  forward returns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.clean.schema import (
    DATE_LEVEL,
    INDEX_NAMES,
    SYMBOL_LEVEL,
    normalize_panel,
    validate_panel,
)


class PanelStore:
    """Persist and load canonical panels as parquet files under ``root``."""

    def __init__(self, root: str) -> None:
        """Create a store rooted at ``root`` (the base directory for parquet files).

        The directory is created lazily on the first :meth:`write`; constructing a
        store never touches the filesystem, so it is safe in dry runs.
        """
        self._root = Path(root)

    def path_for(self, name: str) -> Path:
        """Return the parquet path for a logical panel ``name`` (``<root>/<name>.parquet``)."""
        if not name or "/" in name or "\\" in name:
            raise ValueError(
                f"Panel name must be a simple non-empty token without path separators; got {name!r}."
            )
        return self._root / f"{name}.parquet"

    def write(self, name: str, panel: pd.DataFrame, overwrite: bool = True) -> None:
        """Persist ``panel`` (a canonical (date, symbol) panel) to parquet.

        ``panel`` is validated against the schema contract first, so a malformed
        frame fails fast with a readable error instead of writing junk. The
        MultiIndex is flattened to ``date``/``symbol`` columns for storage.

        ``overwrite`` mirrors ``output.overwrite``: when ``False`` and the target
        file already exists, raise rather than clobber it.
        """
        validate_panel(panel)

        target = self.path_for(name)
        if target.exists() and not overwrite:
            raise ValueError(
                f"A stored panel '{name}' already exists at {target} and overwrite=False. "
                "Pass overwrite=True to replace it, or choose a different name."
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        # Flatten the MultiIndex into columns so parquet can round-trip it.
        flat = panel.reset_index()
        # Atomic-ish replace: write to a temp file then move into place, so a
        # crash mid-write never leaves a half-written panel at the real path.
        tmp = target.with_suffix(".parquet.tmp")
        flat.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(target)

    def read(
        self,
        name: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load the panel stored under ``name`` and return a normalized panel.

        Optional filters (applied before re-normalizing):
          - ``start`` / ``end``: keep only rows whose date falls in the **closed**
            interval ``[start, end]``. Either bound may be a ``"YYYY-MM-DD"`` string
            or a ``pd.Timestamp``; either may be omitted to leave that side open.
          - ``symbols``: keep only rows whose symbol is in this subset.

        The result is always a fresh, schema-valid panel (sorted, MultiIndex,
        correct dtypes). Filtering to an empty selection is allowed and yields an
        empty panel with the correct columns/index names.
        """
        target = self.path_for(name)
        if not target.exists():
            raise FileNotFoundError(
                f"No stored panel '{name}' at {target}. Write it first with PanelStore.write()."
            )

        flat = pd.read_parquet(target, engine="pyarrow")
        flat = self._apply_date_filter(flat, start, end)
        flat = self._apply_symbol_filter(flat, symbols)

        # Rebuild the canonical shape. normalize_panel re-sorts, re-types the date
        # level to datetime and the symbol level to str, and revalidates uniqueness.
        return normalize_panel(flat)

    @staticmethod
    def _apply_date_filter(
        flat: pd.DataFrame,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Return rows of ``flat`` whose ``date`` is in the closed [start, end] interval."""
        if start is None and end is None:
            return flat
        date_col = pd.to_datetime(flat[DATE_LEVEL]).dt.normalize()
        mask = pd.Series(True, index=flat.index)
        if start is not None:
            mask &= date_col >= pd.Timestamp(start).normalize()
        if end is not None:
            mask &= date_col <= pd.Timestamp(end).normalize()
        return flat.loc[mask]

    @staticmethod
    def _apply_symbol_filter(
        flat: pd.DataFrame,
        symbols: list[str] | None,
    ) -> pd.DataFrame:
        """Return rows of ``flat`` whose ``symbol`` is in ``symbols`` (no-op if None)."""
        if symbols is None:
            return flat
        wanted = {str(s) for s in symbols}
        mask = flat[SYMBOL_LEVEL].astype(str).isin(wanted)
        return flat.loc[mask]


__all__ = ["PanelStore", "INDEX_NAMES"]
