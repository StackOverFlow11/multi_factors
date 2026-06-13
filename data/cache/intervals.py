"""Closed-interval day algebra for the cache's missing-range planning (P4-1).

Coverage is tracked as a set of CLOSED calendar-date intervals ``[start, end]``
(day resolution). The read-through planner needs two operations:

  * ``merge_intervals`` — collapse overlapping / day-adjacent intervals so the
    covered set is canonical;
  * ``subtract_intervals`` — given a requested ``[start, end]`` and the covered
    set, return the uncovered sub-intervals (the gaps to fetch).

Calendar (not trading-day) granularity is deliberate: a fetched ``[s, e]`` range
COVERS that whole calendar span even on days the endpoint legitimately returned
no row (not listed / suspended / holiday). Tracking coverage by calendar range —
not by row presence — is exactly what lets the cache distinguish "missing because
not fetched" from "missing because the source has no row" (the coverage-ledger
rationale).

Pure functions; one day == ``pd.Timedelta(days=1)``.
"""

from __future__ import annotations

import pandas as pd

_ONE_DAY = pd.Timedelta(days=1)

Interval = tuple[pd.Timestamp, pd.Timestamp]


def _norm(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).normalize()


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Sort and merge overlapping or day-adjacent closed intervals."""
    cleaned = [(_norm(s), _norm(e)) for s, e in intervals if _norm(s) <= _norm(e)]
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[Interval] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        # adjacent (gap of exactly one day) intervals merge into one.
        if start <= last_end + _ONE_DAY:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(
    start, end, covered: list[Interval]
) -> list[Interval]:
    """Return the closed sub-intervals of ``[start, end]`` NOT in ``covered``.

    Both endpoints are inclusive. An empty result means the request is fully
    covered (a repeated run fetches nothing).
    """
    req_start, req_end = _norm(start), _norm(end)
    if req_start > req_end:
        return []
    gaps: list[Interval] = []
    cursor = req_start
    for cov_start, cov_end in merge_intervals(covered):
        if cov_end < cursor:
            continue
        if cov_start > req_end:
            break
        if cov_start > cursor:
            gaps.append((cursor, min(cov_start - _ONE_DAY, req_end)))
        cursor = max(cursor, cov_end + _ONE_DAY)
        if cursor > req_end:
            break
    if cursor <= req_end:
        gaps.append((cursor, req_end))
    return gaps
