"""Small pure planning helpers for the tushare read-through cache (D2 split from
``tushare_cache.py``).

A stable field-set hash and a compact date formatter. No I/O, no token, no
endpoint dispatch — just the two leaf helpers the gap planner uses. Behaviour is
identical to the pre-split functions.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _fields_hash(columns: list[str]) -> str:
    """Stable short hash of a field set (order-independent)."""
    return hashlib.sha1(",".join(sorted(columns)).encode("utf-8")).hexdigest()[:16]


def _compact(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")
