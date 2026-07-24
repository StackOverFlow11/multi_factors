"""Run registry (refactor D3, commit 3, R22/R25): append-only, no-secret, startup check.

Network-free. Covers JSONL append-only behavior, the string status field + its
validation, the no-secret contract (a stray secret in a note is redacted), and the
startup consistency assertion (code registry must cover every run-registry id).
"""

from __future__ import annotations

import pytest

import factors.registry.builtin  # noqa: F401 - populate DEFAULT_REGISTRY
from factors.registry.registry import build
from factors.store import RunRecord, RunRegistry, assert_registry_consistent, store_key
from factors.store.fingerprint import data_fingerprint

_TOKEN = "abcd1234deadbeefabcd1234deadbeef"


def _record(status="watch", note=None, factor_id="volatility_20"):
    return RunRecord(
        factor_id=factor_id,
        params_hash="p0",
        code_hash="c0",
        view="decision",
        status=status,
        endpoints=("market_daily",),
        adjustment="returns_invariant",
        schema_version="deadbeef",
        created_utc="2026-07-24 00:00:00",
        note=note,
    )


def test_append_then_read_roundtrips(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.append(_record())
    rows = reg.read_all()
    assert len(rows) == 1
    assert rows[0]["factor_id"] == "volatility_20"
    assert rows[0]["status"] == "watch"


def test_registry_is_append_only(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.append(_record(factor_id="volatility_20"))
    reg.append(_record(factor_id="value_ep"))
    rows = reg.read_all()
    # two appends => two lines; the first is never overwritten (verdicts append-only)
    assert [r["factor_id"] for r in rows] == ["volatility_20", "value_ep"]


def test_unknown_status_is_a_readable_error(tmp_path):
    reg = RunRegistry(tmp_path)
    with pytest.raises(ValueError, match="unknown run-registry status"):
        reg.append(_record(status="promoted"))


def test_note_secret_is_redacted_in_the_written_line(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.append(_record(note=f"see /home/x/.config.json token={_TOKEN}"))
    text = reg.path.read_text()
    assert ".config.json" not in text
    assert _TOKEN not in text
    assert "[REDACTED]" in text


def test_no_secret_anywhere_in_a_real_appended_record(tmp_path):
    reg = RunRegistry(tmp_path)
    vol = build("volatility_20", {"window": 20})
    key = store_key(vol, view="decision", params={"window": 20})
    fp = data_fingerprint(adjustment=vol.spec.adjustment)
    reg.append_run(key=key, factor=vol, status="watch", fingerprint=fp)
    text = reg.path.read_text()
    assert "token" not in text.lower()
    assert ".config.json" not in text


def test_append_run_builds_endpoints_from_spec_requires(tmp_path):
    reg = RunRegistry(tmp_path)
    ridge = build("ridge_minute_return_20", {"lookback_days": 20})
    key = store_key(ridge, view="decision", params={"lookback_days": 20})
    fp = data_fingerprint(adjustment=ridge.spec.adjustment)
    rec = reg.append_run(key=key, factor=ridge, status="exploratory", fingerprint=fp)
    assert rec.endpoints == ("stk_mins_1min",)  # ridge requires stk_mins_1min only
    assert reg.read_all()[0]["endpoints"] == ["stk_mins_1min"]


# --------------------------------------------------------------------------- #
# startup consistency (R25)
# --------------------------------------------------------------------------- #
def test_startup_consistency_passes_for_known_factors(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.append(_record(factor_id="volatility_20"))
    reg.append(_record(factor_id="ridge_minute_return_20"))
    assert_registry_consistent(reg)  # both resolve in the code registry -> no raise


def test_startup_consistency_flags_an_unknown_factor(tmp_path):
    reg = RunRegistry(tmp_path)
    reg.append(_record(factor_id="a_deleted_factor"))
    with pytest.raises(ValueError, match="does not know"):
        assert_registry_consistent(reg)
