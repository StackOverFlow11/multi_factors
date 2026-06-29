"""Declarative tushare endpoint schema registry + a default-off drift guard.

THE CORE INSIGHT. Each ``_parse_*`` in :mod:`data.cache.tushare_parsers` returns
a FIXED canonical column set and defensively NaN-fills any canonical column whose
source column is absent (``if col not in df.columns: df[col] = nan``). So the
``fields_hash`` of the STORED canonical columns is IMMUNE to tushare changing its
source columns — the real corruption risk is: tushare removes/renames a SOURCE
column a parser depends on -> the parser silently fills NaN -> the cache stores
NaN with ZERO signal. The highest-value guard therefore checks the RAW INPUT
boundary, BEFORE parsing.

This module:
  * derives canonical info (``canonical_columns`` / ``natural_key`` /
    ``expected_canonical_hash``) from :mod:`data.cache.tushare_specs` (SINGLE
    SOURCE OF TRUTH — no duplicated canonical column lists), and
  * declares, per endpoint, the RAW SOURCE columns each parser depends on, split
    into ``required_source_columns`` (drift here is HARD), ``optional_source_
    columns`` (tushare may legitimately omit; absent => WARNING-free, parser
    defaults), and ``known_extra_columns`` (documented-but-ignored tushare
    response columns, so check #2 does not warn on them). The declarations are
    LOCKED both ways (see ``tests/test_schema_registry``): a forward lock (feed
    exactly the required set through the REAL parser => no NaN in the required-
    derived canonical columns) and an inverse lock (drop any optional column =>
    the parser does NOT raise and still yields the canonical schema).

Per-endpoint REQUIRED vs OPTIONAL rule (the HIGH-1 fix). A source column is
REQUIRED only when tushare GUARANTEES it back, i.e. one of:
  (a) the parser reads it directly so its absence RAISES (the axis columns that
      become ``date`` / ``symbol`` / a natural-key part, plus ``adj_factor`` /
      ``start_date`` / ``name`` / ``in_date``); or
  (b) it is explicitly requested via a ``fields=`` selector (daily_basic /
      fina_indicator request their fields, so tushare returns those columns); or
  (c) it is a core column of the endpoint's documented STANDARD response for a
      no-``fields=`` endpoint (daily OHLCV+vol+amount, stk_limit up/down_limit,
      index_weight weight) — always present, so a missing column is true drift.
A source column is OPTIONAL when the parser DEFENSIVELY fills it AND there is
evidence tushare legitimately omits it / it is not axis-critical: e.g.
index_member_all's ``l1/l2/l3_name`` (the DIRECT feed path guards their absence
at ``data/feed/tushare_covariates.py:155`` — proof tushare omits them),
``out_date`` / ``end_date`` (open-interval ends), ``ts_code`` (parser falls back
to the symbol arg), ``suspend_type`` (query-filter default 'S'), ``list_date``
(the feed keeps + discloses a stock with no list_date). Optional drift is NOT a
HARD finding — it is benign, so it produces neither a HARD nor (being known) a
check-#2 WARNING.

The :class:`SchemaGuard` is REPORT-ONLY by default and DEFAULT-OFF in the cache
(``TushareCache(schema_guard=None)`` is byte-identical to before). It never sees
a token or a data value — only endpoint names + column names — so a finding,
log, or summary can never carry a secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.cache.tushare_planning import _fields_hash
from data.cache.tushare_specs import (
    ADJ_FACTOR,
    ALL_ENDPOINTS,
    DAILY_BASIC,
    FINA_FIELDS,
    FINA_INDICATOR,
    INDEX_MEMBER_ALL,
    INDEX_WEIGHT,
    MARKET_DAILY,
    NAMECHANGE,
    STK_LIMIT,
    STOCK_BASIC,
    SUSPEND_D,
    _ADJ_COLUMNS,
    _DAILY_BASIC_COLUMNS,
    _DAILY_BASIC_KEY,
    _DAILY_COLUMNS,
    _FINA_COLUMNS,
    _FINA_KEY,
    _INDEX_MEMBER_COLUMNS,
    _INDEX_MEMBER_KEY,
    _INDEX_WEIGHT_COLUMNS,
    _INDEX_WEIGHT_KEY,
    _KEY_COLS,
    _NAMECHANGE_COLUMNS,
    _NAMECHANGE_KEY,
    _STK_LIMIT_COLUMNS,
    _STK_LIMIT_KEY,
    _STOCK_BASIC_COLUMNS,
    _STOCK_BASIC_KEY,
    _SUSPEND_COLUMNS,
    _SUSPEND_KEY,
)

# guard modes.
REPORT_ONLY = "report_only"
STRICT = "strict"

# finding severities.
HARD = "hard"
WARNING = "warning"

# check ids (stable, secret-free identifiers used in findings + tests).
CHECK_MISSING_REQUIRED = "missing_required_source_columns"
CHECK_UNKNOWN_EXTRA = "unknown_extra_source_columns"
CHECK_CANONICAL_MISMATCH = "canonical_columns_mismatch"
CHECK_HASH_CHANGED = "stored_schema_hash_changed"


@dataclass(frozen=True)
class EndpointSchema:
    """The raw-input + canonical-output contract for one cached tushare endpoint.

    ``required_source_columns`` are the RAW columns tushare GUARANTEES and the
    parser depends on (their absence is true drift / the silent-NaN risk).
    ``optional_source_columns`` are columns the parser reads but tolerates as
    absent (defaults / arg fallbacks) and tushare may legitimately omit — so
    their absence is neither HARD nor a check-#2 warning. ``known_extra_columns``
    is a best-effort catalogue of documented-but-IGNORED tushare response columns
    (the no-``fields=`` endpoints return more than the parser keeps); check #2
    warns only on a column OUTSIDE required ∪ optional ∪ known_extra, i.e. a
    genuinely new/uncatalogued column. ``canonical_columns`` / ``natural_key``
    come straight from the spec constants (no duplication). ``expected_canonical_
    hash`` uses the SAME hashing as the coverage ledger's ``fields_hash`` so
    check #4 can compare them.
    """

    endpoint: str
    required_source_columns: frozenset[str]
    canonical_columns: tuple[str, ...]
    natural_key: tuple[str, ...]
    optional_source_columns: frozenset[str] = field(default_factory=frozenset)
    known_extra_columns: frozenset[str] = field(default_factory=frozenset)

    @property
    def known_source_columns(self) -> frozenset[str]:
        """Required + optional + catalogued extras — columns NOT to flag as #2."""
        return (
            self.required_source_columns
            | self.optional_source_columns
            | self.known_extra_columns
        )

    @property
    def expected_canonical_hash(self) -> str:
        """Stable hash of the canonical columns (matches the ledger fields_hash)."""
        return _fields_hash(list(self.canonical_columns))


@dataclass(frozen=True)
class SchemaDriftFinding:
    """One schema-drift finding. Carries ONLY endpoint + column names (no secret).

    ``columns`` is the sorted tuple of offending column names (missing required /
    unknown extra / canonical diff); empty for the hash-change check. ``detail`` is
    a human-readable, secret-free message (endpoint + column names + the short
    column-set hashes only — never a token, path, or data value).
    """

    endpoint: str
    check: str
    severity: str
    detail: str
    columns: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Registry. canonical_columns / natural_key are the spec constants verbatim.
# required / optional / known_extra are declared per the REQUIRED-vs-OPTIONAL rule
# in the module docstring (LOCKED both ways by tests). known_extra is a best-effort
# catalogue of documented tushare response columns the parser ignores — a brand-new
# tushare column simply yields one WARNING until it is catalogued here.
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, EndpointSchema] = {
    # _parse_daily (no fields=): ts_code->symbol, trade_date->date (axis, raise);
    # open/high/low/close/vol/amount are tushare daily's guaranteed core (NaN-filled
    # if a column vanished = the silent-NaN drift we want HARD).
    MARKET_DAILY: EndpointSchema(
        endpoint=MARKET_DAILY,
        required_source_columns=frozenset(
            {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
        ),
        canonical_columns=tuple(_DAILY_COLUMNS),
        natural_key=tuple(_KEY_COLS),
        known_extra_columns=frozenset({"pre_close", "change", "pct_chg"}),
    ),
    # _parse_adj (no fields=): adj_factor selected directly with NO defensive fill —
    # absence raises; ts_code/trade_date are the axis.
    ADJ_FACTOR: EndpointSchema(
        endpoint=ADJ_FACTOR,
        required_source_columns=frozenset({"ts_code", "trade_date", "adj_factor"}),
        canonical_columns=tuple(_ADJ_COLUMNS),
        natural_key=tuple(_KEY_COLS),
    ),
    # _parse_index_weight (no fields=): con_code->symbol, trade_date->date (axis);
    # weight is the endpoint's core; index_code is INJECTED from the arg (raw's is
    # overwritten / ignored => catalogued extra, never read).
    INDEX_WEIGHT: EndpointSchema(
        endpoint=INDEX_WEIGHT,
        required_source_columns=frozenset({"con_code", "trade_date", "weight"}),
        canonical_columns=tuple(_INDEX_WEIGHT_COLUMNS),
        natural_key=tuple(_INDEX_WEIGHT_KEY),
        known_extra_columns=frozenset({"index_code"}),
    ),
    # _parse_suspend (filter suspend_type='S', no fields=): ts_code->symbol,
    # trade_date->date (axis). suspend_type is defensively defaulted to 'S' (matches
    # the query filter) => optional; suspend_timing is ignored.
    SUSPEND_D: EndpointSchema(
        endpoint=SUSPEND_D,
        required_source_columns=frozenset({"ts_code", "trade_date"}),
        canonical_columns=tuple(_SUSPEND_COLUMNS),
        natural_key=tuple(_SUSPEND_KEY),
        optional_source_columns=frozenset({"suspend_type"}),
        known_extra_columns=frozenset({"suspend_timing"}),
    ),
    # _parse_namechange (no fields=): start_date + name read directly (absence
    # raises). ts_code falls back to the symbol arg, end_date defaults to NaT (open
    # interval) => both optional; ann_date / change_reason are ignored.
    NAMECHANGE: EndpointSchema(
        endpoint=NAMECHANGE,
        required_source_columns=frozenset({"start_date", "name"}),
        canonical_columns=tuple(_NAMECHANGE_COLUMNS),
        natural_key=tuple(_NAMECHANGE_KEY),
        optional_source_columns=frozenset({"ts_code", "end_date"}),
        known_extra_columns=frozenset({"ann_date", "change_reason"}),
    ),
    # _parse_stk_limit (no fields=): ts_code->symbol, trade_date->date (axis);
    # up_limit/down_limit are the endpoint's core; pre_close is ignored.
    STK_LIMIT: EndpointSchema(
        endpoint=STK_LIMIT,
        required_source_columns=frozenset(
            {"ts_code", "trade_date", "up_limit", "down_limit"}
        ),
        canonical_columns=tuple(_STK_LIMIT_COLUMNS),
        natural_key=tuple(_STK_LIMIT_KEY),
        known_extra_columns=frozenset({"pre_close"}),
    ),
    # _parse_stock_basic (fields="ts_code,list_date"): ts_code->symbol read directly
    # (raise). list_date is defensively defaulted to None AND the feed keeps +
    # discloses a stock with no list_date (proof it is legitimately omitted) =>
    # optional, not HARD.
    STOCK_BASIC: EndpointSchema(
        endpoint=STOCK_BASIC,
        required_source_columns=frozenset({"ts_code"}),
        canonical_columns=tuple(_STOCK_BASIC_COLUMNS),
        natural_key=tuple(_STOCK_BASIC_KEY),
        optional_source_columns=frozenset({"list_date"}),
    ),
    # _parse_daily_basic (fields="ts_code,trade_date,pe,pb,total_mv"): every kept
    # column is explicitly requested, so tushare guarantees the columns back =>
    # required (a missing COLUMN is drift; a row-level NaN pe is downstream's job).
    DAILY_BASIC: EndpointSchema(
        endpoint=DAILY_BASIC,
        required_source_columns=frozenset(
            {"ts_code", "trade_date", "pe", "pb", "total_mv"}
        ),
        canonical_columns=tuple(_DAILY_BASIC_COLUMNS),
        natural_key=tuple(_DAILY_BASIC_KEY),
    ),
    # _parse_fina (fields="ts_code,ann_date,end_date,*FINA_FIELDS" superset):
    # ts_code->symbol; the coverage axis ``date`` is DERIVED from end_date, and
    # ann_date drives the downstream as-of — both are requested so guaranteed back
    # => required. All FINA_FIELDS are requested too => required.
    FINA_INDICATOR: EndpointSchema(
        endpoint=FINA_INDICATOR,
        required_source_columns=frozenset(
            {"ts_code", "ann_date", "end_date", *FINA_FIELDS}
        ),
        canonical_columns=tuple(_FINA_COLUMNS),
        natural_key=tuple(_FINA_KEY),
    ),
    # _parse_index_member (no fields=): in_date read directly (absence raises) — the
    # ONLY required column. l1/l2/l3_name are defensively None-filled AND the direct
    # feed path (tushare_covariates.py:155) guards their absence (proof tushare omits
    # level columns) => optional, NEVER HARD. ts_code falls back to the symbol arg,
    # out_date defaults to NaT (open membership) => optional. l*_code / name / is_new
    # are ignored.
    INDEX_MEMBER_ALL: EndpointSchema(
        endpoint=INDEX_MEMBER_ALL,
        required_source_columns=frozenset({"in_date"}),
        canonical_columns=tuple(_INDEX_MEMBER_COLUMNS),
        natural_key=tuple(_INDEX_MEMBER_KEY),
        optional_source_columns=frozenset(
            {"ts_code", "l1_name", "l2_name", "l3_name", "out_date"}
        ),
        known_extra_columns=frozenset(
            {"l1_code", "l2_code", "l3_code", "name", "is_new"}
        ),
    ),
}

# every endpoint the cache knows must have a registry entry. Use an explicit raise
# (NOT assert, which `python -O` strips) so a future endpoint can never slip in
# uncovered.
if set(REGISTRY) != set(ALL_ENDPOINTS):
    raise RuntimeError(
        "schema REGISTRY is out of sync with tushare_specs.ALL_ENDPOINTS: "
        f"missing={sorted(set(ALL_ENDPOINTS) - set(REGISTRY))}, "
        f"extra={sorted(set(REGISTRY) - set(ALL_ENDPOINTS))}."
    )


class SchemaGuard:
    """Stateful, secret-free schema-drift guard for the tushare read-through cache.

    DEFAULT-OFF: a cache built with ``schema_guard=None`` never constructs one. In
    ``report_only`` mode every check appends a :class:`SchemaDriftFinding`; in
    ``strict`` mode a HARD finding RAISES a secret-free ``RuntimeError`` (warnings
    still only append). Findings carry only endpoint + column names.
    """

    def __init__(self, mode: str = REPORT_ONLY) -> None:
        if mode not in (REPORT_ONLY, STRICT):
            raise ValueError(
                f"SchemaGuard mode must be {REPORT_ONLY!r} or {STRICT!r}; got {mode!r}."
            )
        self._mode = mode
        self._findings: list[SchemaDriftFinding] = []
        # endpoints whose stored-schema hash has already been checked this run
        # (check #4 fires at most once per endpoint per run).
        self._hash_checked: set[str] = set()

    @property
    def mode(self) -> str:
        return self._mode

    # -- raw-input checks (#1 missing required, #2 unknown extra) ------------ #
    def inspect_raw(self, endpoint: str, raw: "pd.DataFrame | None") -> None:
        """Check a NON-EMPTY raw frame's source columns; skip None/empty returns.

        An empty / None return is legitimate coverage (a stock not listed /
        suspended / out of index on a range), NOT drift — so it is skipped.
        """
        if raw is None or len(raw) == 0:
            return
        schema = REGISTRY.get(endpoint)
        if schema is None:
            return
        cols = {str(c) for c in raw.columns}
        missing = schema.required_source_columns - cols
        if missing:
            names = tuple(sorted(missing))
            self._emit(
                endpoint,
                CHECK_MISSING_REQUIRED,
                HARD,
                f"endpoint {endpoint} raw frame is missing required source "
                f"column(s) {list(names)}; the parser will silently NaN-fill the "
                f"canonical data",
                names,
            )
        extra = cols - schema.known_source_columns
        if extra:
            names = tuple(sorted(extra))
            self._emit(
                endpoint,
                CHECK_UNKNOWN_EXTRA,
                WARNING,
                f"endpoint {endpoint} raw frame has unknown extra source "
                f"column(s) {list(names)} the parser ignores",
                names,
            )

    # -- canonical-output check (#3 parsed columns vs registry) -------------- #
    def inspect_canonical(self, endpoint: str, parsed_columns) -> None:
        """Check the parsed canonical columns equal the registry (order-independent).

        A mismatch means the parser / spec / registry disagree — a code bug, not
        upstream drift — so it is HARD.
        """
        schema = REGISTRY.get(endpoint)
        if schema is None:
            return
        expected = set(schema.canonical_columns)
        got = {str(c) for c in parsed_columns}
        if expected != got:
            diff = tuple(sorted((expected - got) | (got - expected)))
            self._emit(
                endpoint,
                CHECK_CANONICAL_MISMATCH,
                HARD,
                f"endpoint {endpoint} parsed canonical columns "
                f"{sorted(got)} != registry {sorted(expected)}",
                diff,
            )

    # -- stored-schema hash check (#4 vs cached ledger history) -------------- #
    def check_fields_hash(
        self, endpoint: str, current_hash: str, prior_hash: "str | None"
    ) -> None:
        """Compare the registry canonical hash to the ledger's last fields_hash.

        Fires at most once per endpoint per run. A mismatch means the STORED
        schema changed vs the cached ledger history (a canonical-column migration)
        — HARD. This is a MIGRATION DETECTOR, not an init validator:
        ``prior_hash is None`` (a fresh cache / an endpoint never recorded) is
        ALWAYS clean, and an equal hash is clean.
        """
        if endpoint in self._hash_checked:
            return
        self._hash_checked.add(endpoint)
        if prior_hash is None or prior_hash == current_hash:
            return
        self._emit(
            endpoint,
            CHECK_HASH_CHANGED,
            HARD,
            f"endpoint {endpoint} stored schema changed vs cached ledger history "
            f"(ledger fields_hash={prior_hash}, current={current_hash})",
            (),
        )

    # -- accumulation / reporting ------------------------------------------- #
    def _emit(
        self,
        endpoint: str,
        check: str,
        severity: str,
        detail: str,
        columns: tuple[str, ...],
    ) -> None:
        finding = SchemaDriftFinding(
            endpoint=endpoint,
            check=check,
            severity=severity,
            detail=detail,
            columns=columns,
        )
        if self._mode == STRICT and severity == HARD:
            raise RuntimeError(detail)
        self._findings.append(finding)

    def findings(self) -> tuple[SchemaDriftFinding, ...]:
        """Accumulated findings (insertion order — stable + deterministic)."""
        return tuple(self._findings)

    def summary(self) -> dict:
        """One-line-friendly counts: total / hard / warning / by_endpoint."""
        hard = sum(1 for f in self._findings if f.severity == HARD)
        warning = sum(1 for f in self._findings if f.severity == WARNING)
        by_endpoint: dict[str, int] = {}
        for f in self._findings:
            by_endpoint[f.endpoint] = by_endpoint.get(f.endpoint, 0) + 1
        return {
            "total": len(self._findings),
            "hard": hard,
            "warning": warning,
            "by_endpoint": by_endpoint,
        }
