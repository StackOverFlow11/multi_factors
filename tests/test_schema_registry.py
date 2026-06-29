"""D-series schema registry + default-off drift guard (network-free, fakes only).

Locks:
  * the registry declarations are consistent with the spec column lists AND the
    REAL parsers (feeding exactly ``required_source_columns`` introduces no NaN);
  * ``expected_canonical_hash`` matches the ledger's ``fields_hash`` hashing;
  * check #1 (missing required) -> HARD / strict raises (endpoint+column, no secret);
  * check #2 (unknown extra) -> WARNING, never hard;
  * empty / None raw -> no findings (legitimate coverage, not drift);
  * check #3 (parsed canonical mismatch) -> HARD;
  * check #4 (stored-schema hash changed) -> HARD once per endpoint, deduped;
  * default-off ``_guarded_parse`` is a byte-identical passthrough;
  * report_only records coverage + a finding; strict raises BEFORE upsert/coverage;
  * ``CoverageLedger.last_fields_hash`` returns latest-by-fetched_at / None;
  * findings + summary carry no token / secret-file content.

No test hits the network or reads the real token: fake raw frames + a fake fetch
closure drive everything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.cache import tushare_parsers as P
from data.cache.coverage import CoverageLedger
from data.cache.parquet_store import CacheParquetStore
from data.cache.schema_registry import (
    CHECK_CANONICAL_MISMATCH,
    CHECK_HASH_CHANGED,
    CHECK_MISSING_REQUIRED,
    CHECK_UNKNOWN_EXTRA,
    REGISTRY,
    EndpointSchema,
    SchemaDriftFinding,
    SchemaGuard,
)
from data.cache.tushare_cache import TushareCache
from data.cache.tushare_planning import _fields_hash
from data.cache.tushare_specs import (
    ALL_ENDPOINTS,
    INDEX_MEMBER_ALL,
    MARKET_DAILY,
    _DAILY_COLUMNS,
)

# bench of one valid value per known source column (drives the lock test).
_SAMPLE: dict[str, object] = {
    "ts_code": "000001.SZ", "con_code": "000001.SZ", "trade_date": "20240102",
    "start_date": "20240102", "in_date": "20240102", "out_date": "20240630",
    "ann_date": "20240102",
    "end_date": "20240331", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
    "vol": 1000.0, "amount": 1.0e6, "adj_factor": 1.0, "up_limit": 2.2,
    "down_limit": 1.8, "weight": 5.0, "pe": 10.0, "pb": 1.2, "total_mv": 1.0e9,
    "name": "PINGAN", "l1_name": "Bank", "l2_name": "Bank2", "l3_name": "Bank3",
    "roe": 12.0, "netprofit_yoy": 8.0, "grossprofit_margin": 30.0,
}

# the real parser per endpoint (index_weight/namechange/index_member take an arg).
_PARSERS = {
    "market_daily": P._parse_daily,
    "adj_factor": P._parse_adj,
    "suspend_d": P._parse_suspend,
    "stk_limit": P._parse_stk_limit,
    "daily_basic": P._parse_daily_basic,
    "fina_indicator": P._parse_fina,
    "index_weight": lambda raw: P._parse_index_weight(raw, "000300.SH"),
    "namechange": lambda raw: P._parse_namechange(raw, "000001.SZ"),
    "stock_basic": P._parse_stock_basic,
    "index_member_all": lambda raw: P._parse_index_member(raw, "000001.SZ"),
}

# canonical columns derived from OPTIONAL source columns: feeding only the required
# source columns legitimately leaves these None/NaT (open-interval ends, omittable
# SW level names), so they are excluded from the forward no-NaN lock.
_NULLABLE = {
    "namechange": {"end_date"},
    "index_member_all": {"l1_name", "l2_name", "l3_name", "out_date"},
}


def _raw_with(cols) -> pd.DataFrame:
    return pd.DataFrame({c: [_SAMPLE[c]] for c in cols})


# --------------------------------------------------------------------------- #
# 1. registry <-> spec/parser consistency (THE LOCK)
# --------------------------------------------------------------------------- #
def test_registry_covers_every_endpoint():
    assert set(REGISTRY) == set(ALL_ENDPOINTS)


@pytest.mark.parametrize("endpoint", list(ALL_ENDPOINTS))
def test_required_source_columns_are_sufficient(endpoint):
    """Feeding EXACTLY required_source_columns through the REAL parser yields the
    registry canonical columns with no NaN in the data columns."""
    schema = REGISTRY[endpoint]
    raw = _raw_with(sorted(schema.required_source_columns))
    out = _PARSERS[endpoint](raw)
    assert list(out.columns) == list(schema.canonical_columns)
    data_cols = [c for c in out.columns if c not in _NULLABLE.get(endpoint, set())]
    assert not out[data_cols].isna().any().any(), (
        endpoint, out[data_cols].isna().any().to_dict()
    )


@pytest.mark.parametrize("endpoint", list(ALL_ENDPOINTS))
def test_canonical_columns_match_real_parser_shape(endpoint):
    """The registry canonical columns equal the parser's full (empty) shape."""
    empty = _PARSERS[endpoint](None)
    assert list(REGISTRY[endpoint].canonical_columns) == list(empty.columns)


# --------------------------------------------------------------------------- #
# 2. expected_canonical_hash == ledger fields_hash hashing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("endpoint", list(ALL_ENDPOINTS))
def test_expected_canonical_hash_matches_fields_hash(endpoint):
    schema = REGISTRY[endpoint]
    assert schema.expected_canonical_hash == _fields_hash(list(schema.canonical_columns))


# --------------------------------------------------------------------------- #
# 3. check #1 (missing required) -> HARD / strict raises, no secret
# --------------------------------------------------------------------------- #
def test_missing_required_is_hard_in_report_only():
    guard = SchemaGuard(mode="report_only")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns - {"amount"}))
    guard.inspect_raw(MARKET_DAILY, raw)
    findings = guard.findings()
    hard = [f for f in findings if f.severity == "hard"]
    assert len(hard) == 1
    f = hard[0]
    assert f.check == CHECK_MISSING_REQUIRED
    assert f.endpoint == MARKET_DAILY
    assert "amount" in f.columns


def test_missing_required_raises_in_strict_with_endpoint_and_column():
    guard = SchemaGuard(mode="strict")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns - {"amount"}))
    with pytest.raises(RuntimeError) as exc:
        guard.inspect_raw(MARKET_DAILY, raw)
    msg = str(exc.value)
    assert MARKET_DAILY in msg
    assert "amount" in msg
    # strict raise carries no secret.
    assert ".config.json" not in msg and "token" not in msg.lower()
    assert guard.findings() == ()  # nothing accumulated on a raise


# --------------------------------------------------------------------------- #
# 4. check #2 (unknown extra) -> WARNING, never hard
# --------------------------------------------------------------------------- #
def test_unknown_extra_source_column_is_warning():
    guard = SchemaGuard(mode="report_only")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns))
    raw["surprise_col"] = [123.0]
    guard.inspect_raw(MARKET_DAILY, raw)
    findings = guard.findings()
    assert all(f.severity != "hard" for f in findings)
    warn = [f for f in findings if f.check == CHECK_UNKNOWN_EXTRA]
    assert len(warn) == 1
    assert warn[0].severity == "warning"
    assert "surprise_col" in warn[0].columns


def test_unknown_extra_never_raises_in_strict():
    guard = SchemaGuard(mode="strict")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns))
    raw["surprise_col"] = [123.0]
    guard.inspect_raw(MARKET_DAILY, raw)  # must NOT raise (warning only)
    assert any(f.check == CHECK_UNKNOWN_EXTRA for f in guard.findings())


# HIGH-2: a catalogued known-extra column must NOT warn; only a truly-new column does.
def test_known_extra_column_is_not_flagged():
    guard = SchemaGuard(mode="report_only")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns))
    for extra in sorted(schema.known_extra_columns):  # pre_close/change/pct_chg
        raw[extra] = [0.0]
    guard.inspect_raw(MARKET_DAILY, raw)
    assert guard.findings() == ()  # all extras catalogued => silent


def test_only_uncatalogued_extra_warns_once():
    guard = SchemaGuard(mode="report_only")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns))
    for extra in sorted(schema.known_extra_columns):
        raw[extra] = [0.0]
    raw["brand_new_tushare_col"] = [1.0]
    guard.inspect_raw(MARKET_DAILY, raw)
    warn = [f for f in guard.findings() if f.check == CHECK_UNKNOWN_EXTRA]
    assert len(warn) == 1
    assert warn[0].columns == ("brand_new_tushare_col",)  # only the uncatalogued one


def test_no_fields_endpoints_do_not_warn_on_standard_extras():
    """The documented tushare response columns the parser ignores are catalogued,
    so a normal fetch raises no check-#2 noise (the HIGH-2 fix)."""
    guard = SchemaGuard(mode="report_only")
    standard_response = {
        "market_daily": ["ts_code", "trade_date", "open", "high", "low", "close",
                         "pre_close", "change", "pct_chg", "vol", "amount"],
        "stk_limit": ["ts_code", "trade_date", "pre_close", "up_limit", "down_limit"],
        "suspend_d": ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
        "namechange": ["ts_code", "name", "start_date", "end_date",
                       "ann_date", "change_reason"],
        "index_member_all": ["l1_code", "l1_name", "l2_code", "l2_name", "l3_code",
                             "l3_name", "ts_code", "name", "in_date", "out_date",
                             "is_new"],
    }
    for ep, cols in standard_response.items():
        raw = pd.DataFrame({c: [_SAMPLE.get(c, "x")] for c in cols})
        guard.inspect_raw(ep, raw)
    extra = [f for f in guard.findings() if f.check == CHECK_UNKNOWN_EXTRA]
    assert extra == [], [f.endpoint + ":" + str(f.columns) for f in extra]


# --------------------------------------------------------------------------- #
# MED-3: inverse lock — every optional column is legitimately droppable, and the
# index_member_all required/optional split is pinned (prevents HIGH-1 regression).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("endpoint", list(ALL_ENDPOINTS))
def test_optional_columns_are_truly_droppable(endpoint):
    """Dropping ANY single optional source column from a full raw frame must NOT
    raise and must still yield the registry canonical schema."""
    schema = REGISTRY[endpoint]
    full_cols = sorted(schema.required_source_columns | schema.optional_source_columns)
    for drop in sorted(schema.optional_source_columns):
        raw = _raw_with([c for c in full_cols if c != drop])
        out = _PARSERS[endpoint](raw)  # must not raise
        assert list(out.columns) == list(schema.canonical_columns), (endpoint, drop)


def test_index_member_required_is_only_in_date():
    """HIGH-1 regression lock: the defensively-filled level names are OPTIONAL,
    never REQUIRED (else strict mode loops on a stock missing a level column)."""
    schema = REGISTRY[INDEX_MEMBER_ALL]
    assert schema.required_source_columns == frozenset({"in_date"})
    assert {"l1_name", "l2_name", "l3_name", "out_date"} <= schema.optional_source_columns
    # and a real raw frame missing every level name must NOT raise in strict mode.
    guard = SchemaGuard(mode="strict")
    raw = pd.DataFrame({"in_date": ["20240102"], "ts_code": ["000001.SZ"]})
    guard.inspect_raw(INDEX_MEMBER_ALL, raw)  # no level cols -> must not raise
    assert all(f.severity != "hard" for f in guard.findings())


# --------------------------------------------------------------------------- #
# 5. empty / None raw -> no findings (legitimate coverage, not drift)
# --------------------------------------------------------------------------- #
def test_empty_or_none_raw_is_not_drift():
    guard = SchemaGuard(mode="strict")  # strict: would raise if it inspected
    guard.inspect_raw(MARKET_DAILY, None)
    guard.inspect_raw(MARKET_DAILY, pd.DataFrame(columns=["ts_code"]))
    assert guard.findings() == ()


# --------------------------------------------------------------------------- #
# 6. check #3 (parsed canonical mismatch) -> HARD
# --------------------------------------------------------------------------- #
def test_canonical_mismatch_is_hard():
    guard = SchemaGuard(mode="report_only")
    wrong = list(_DAILY_COLUMNS) + ["unexpected_col"]
    guard.inspect_canonical(MARKET_DAILY, wrong)
    hard = [f for f in guard.findings() if f.severity == "hard"]
    assert len(hard) == 1
    assert hard[0].check == CHECK_CANONICAL_MISMATCH
    assert "unexpected_col" in hard[0].columns


def test_canonical_match_has_no_finding():
    guard = SchemaGuard(mode="report_only")
    guard.inspect_canonical(MARKET_DAILY, list(_DAILY_COLUMNS))
    assert guard.findings() == ()


# --------------------------------------------------------------------------- #
# 7. check #4 (stored-schema hash changed) -> HARD once per endpoint, deduped
# --------------------------------------------------------------------------- #
def test_fields_hash_change_is_hard_once_then_deduped():
    guard = SchemaGuard(mode="report_only")
    cur = REGISTRY[MARKET_DAILY].expected_canonical_hash
    guard.check_fields_hash(MARKET_DAILY, cur, prior_hash="STALEHASH")
    guard.check_fields_hash(MARKET_DAILY, cur, prior_hash="STALEHASH")  # deduped
    hard = [f for f in guard.findings() if f.check == CHECK_HASH_CHANGED]
    assert len(hard) == 1
    assert hard[0].severity == "hard"
    assert hard[0].endpoint == MARKET_DAILY


def test_fields_hash_none_prior_no_finding():
    guard = SchemaGuard(mode="report_only")
    cur = REGISTRY[MARKET_DAILY].expected_canonical_hash
    guard.check_fields_hash(MARKET_DAILY, cur, prior_hash=None)
    assert guard.findings() == ()


def test_fields_hash_equal_no_finding():
    guard = SchemaGuard(mode="report_only")
    cur = REGISTRY[MARKET_DAILY].expected_canonical_hash
    guard.check_fields_hash(MARKET_DAILY, cur, prior_hash=cur)
    assert guard.findings() == ()


def test_fields_hash_change_strict_raises():
    guard = SchemaGuard(mode="strict")
    cur = REGISTRY[MARKET_DAILY].expected_canonical_hash
    with pytest.raises(RuntimeError) as exc:
        guard.check_fields_hash(MARKET_DAILY, cur, prior_hash="STALEHASH")
    msg = str(exc.value)
    assert MARKET_DAILY in msg
    assert ".config.json" not in msg and "token" not in msg.lower()


# --------------------------------------------------------------------------- #
# 8. default-off passthrough is byte-identical
# --------------------------------------------------------------------------- #
def _daily_raw(ts_code, dates, drop=()):
    rows = [
        {"ts_code": ts_code, "trade_date": d, "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "vol": 1000.0, "amount": 1.0e6}
        for d in dates
    ]
    return pd.DataFrame(rows).drop(columns=list(drop))


def test_guarded_parse_passthrough_when_no_guard(tmp_path):
    cache = TushareCache(
        CacheParquetStore(str(tmp_path)), CoverageLedger(str(tmp_path)),
    )
    raw = _daily_raw("000001.SZ", ["20240102", "20240103"])
    out = cache._guarded_parse(MARKET_DAILY, raw, P._parse_daily)
    pd.testing.assert_frame_equal(out, P._parse_daily(raw))


# --------------------------------------------------------------------------- #
# 9. report_only records coverage + finding; strict raises before upsert/coverage
# --------------------------------------------------------------------------- #
def _drift_fetch(symbol, s_compact, e_compact):
    # a non-empty frame MISSING the required 'amount' column (parser NaN-fills it).
    return _daily_raw(symbol, ["20240102", "20240103"], drop=("amount",))


def test_report_only_still_upserts_and_records_coverage(tmp_path):
    cache = TushareCache(
        CacheParquetStore(str(tmp_path)), CoverageLedger(str(tmp_path)),
        schema_guard=SchemaGuard(mode="report_only"),
    )
    out = cache.daily_bars(["000001.SZ"], "2024-01-02", "2024-01-03", _drift_fetch)
    # row is upserted (NaN amount) + coverage recorded — report-only changes nothing.
    assert not out.empty
    assert not cache._store.read_symbol(MARKET_DAILY, "000001.SZ").empty
    assert cache._ledger.covered_intervals(MARKET_DAILY, "000001.SZ")
    # and a HARD missing-required finding is recorded.
    hard = [f for f in cache.schema_findings() if f.severity == "hard"]
    assert any(f.check == CHECK_MISSING_REQUIRED and "amount" in f.columns for f in hard)


def test_strict_raises_before_upsert_and_coverage(tmp_path):
    cache = TushareCache(
        CacheParquetStore(str(tmp_path)), CoverageLedger(str(tmp_path)),
        schema_guard=SchemaGuard(mode="strict"),
    )
    with pytest.raises(RuntimeError):
        cache.daily_bars(["000001.SZ"], "2024-01-02", "2024-01-03", _drift_fetch)
    # the failing gap recorded NO coverage and upserted NO rows (retryable).
    assert cache._store.read_symbol(MARKET_DAILY, "000001.SZ").empty
    assert cache._ledger.covered_intervals(MARKET_DAILY, "000001.SZ") == []


# --------------------------------------------------------------------------- #
# 10. CoverageLedger.last_fields_hash returns latest-by-fetched_at / None
# --------------------------------------------------------------------------- #
def test_last_fields_hash_latest_and_absent(tmp_path):
    led = CoverageLedger(str(tmp_path))
    assert led.last_fields_hash(MARKET_DAILY) is None  # never recorded
    base = {
        "endpoint": MARKET_DAILY, "key_type": "symbol", "key": "000001.SZ",
        "start_date": pd.Timestamp("2024-01-02"), "end_date": pd.Timestamp("2024-01-03"),
        "row_count": 1, "status": "ok", "source_version": None,
    }
    led.record_many([
        {**base, "fields_hash": "OLDHASH", "fetched_at": pd.Timestamp("2026-06-01")},
        {**base, "fields_hash": "NEWHASH", "fetched_at": pd.Timestamp("2026-06-10")},
    ])
    assert led.last_fields_hash(MARKET_DAILY) == "NEWHASH"
    assert led.last_fields_hash("adj_factor") is None  # other endpoint absent


# --------------------------------------------------------------------------- #
# 11. findings / summary carry no secret
# --------------------------------------------------------------------------- #
def test_findings_and_summary_have_no_secret():
    guard = SchemaGuard(mode="report_only")
    schema = REGISTRY[MARKET_DAILY]
    raw = _raw_with(sorted(schema.required_source_columns - {"amount"}))
    raw["surprise_col"] = [1.0]
    guard.inspect_raw(MARKET_DAILY, raw)
    guard.check_fields_hash(MARKET_DAILY, schema.expected_canonical_hash, "STALE")
    # also exercise check #3's detail string (canonical mismatch).
    guard.inspect_canonical(MARKET_DAILY, list(_DAILY_COLUMNS) + ["unexpected_col"])
    rendered = " ".join(
        f"{f.endpoint} {f.check} {f.severity} {f.detail} {f.columns}"
        for f in guard.findings()
    )
    rendered += " " + str(guard.summary())
    for needle in (".config.json", "tushare.token", "token="):
        assert needle not in rendered
    summary = guard.summary()
    assert summary["total"] == len(guard.findings())
    assert summary["hard"] >= 1 and summary["warning"] >= 1


def test_endpoint_schema_and_finding_are_frozen():
    schema = REGISTRY[MARKET_DAILY]
    with pytest.raises(Exception):
        schema.endpoint = "x"  # type: ignore[misc]
    finding = SchemaDriftFinding(
        endpoint=MARKET_DAILY, check=CHECK_MISSING_REQUIRED, severity="hard",
        detail="d", columns=("amount",),
    )
    with pytest.raises(Exception):
        finding.severity = "warning"  # type: ignore[misc]
    # known_source_columns is required + optional.
    custom = EndpointSchema(
        endpoint="x", required_source_columns=frozenset({"a"}),
        canonical_columns=("a",), natural_key=("a",),
        optional_source_columns=frozenset({"b"}),
    )
    assert custom.known_source_columns == frozenset({"a", "b"})
