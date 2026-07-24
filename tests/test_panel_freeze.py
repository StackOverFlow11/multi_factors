"""Network-free tests for qt.panel_freeze (the D1 raw-panel freeze tool).

Synthetic panels only — the real-data obligations (determinism double-run,
artifact reconciliation, live-call zero) are RUN-time verifications recorded in
the freeze manifest, not unit tests. Here we pin the pure machinery:

* canonical content hash: sensitive to any value / index change, row-order
  independent, NaN-payload independent, +0/-0 distinct, loud on malformed input;
* atomic parquet write: readers never see a partial file, failed writes leave
  no tmp residue and never touch an existing target;
* manifest row / renderer: field-complete, full-precision, deterministic;
* eval-artifact reconciliation: exact-match discipline — a mismatch raises.

Every invariance claim here is paired with a sensitivity assertion on the same
axis (the shuffle test alone could be satisfied by a constant hash; the
value/index sensitivity tests kill that degenerate implementation).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from qt.panel_freeze import (
    DETERMINISM_FACTORS,
    MANIFEST_ROW_FIELDS,
    atomic_write_parquet,
    canonical_content_hash,
    file_sha256,
    manifest_row,
    read_frozen_panel,
    reconcile_with_eval_artifact,
    render_manifest_markdown,
)


def _panel(values, keys, name="factor_x") -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), s) for d, s in keys], names=[DATE_LEVEL, SYMBOL_LEVEL]
    )
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


KEYS = [
    ("2024-01-02", "000001.SZ"),
    ("2024-01-02", "600000.SH"),
    ("2024-01-03", "000001.SZ"),
    ("2024-01-03", "600000.SH"),
]


# --------------------------------------------------------------------------- #
# canonical_content_hash
# --------------------------------------------------------------------------- #
def test_hash_row_order_independent():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    shuffled = base.iloc[[2, 0, 3, 1]]
    assert canonical_content_hash(shuffled) == canonical_content_hash(base)


def test_hash_value_sensitive():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    changed = _panel([1.0, 2.0, 3.0, 4.0 + 1e-12], KEYS)
    assert canonical_content_hash(changed) != canonical_content_hash(base)


def test_hash_symbol_sensitive():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    keys = list(KEYS)
    keys[3] = ("2024-01-03", "600001.SH")
    assert canonical_content_hash(_panel([1.0, 2.0, 3.0, 4.0], keys)) != (
        canonical_content_hash(base)
    )


def test_hash_date_sensitive():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    keys = list(KEYS)
    keys[3] = ("2024-01-04", "600000.SH")
    assert canonical_content_hash(_panel([1.0, 2.0, 3.0, 4.0], keys)) != (
        canonical_content_hash(base)
    )


def test_hash_nan_payload_bits_collapse():
    # Two DIFFERENT IEEE NaN bit patterns must hash identically (the canonical
    # hash rewrites every NaN to one bit pattern). Build the arrays in numpy so
    # no pandas construction path can normalize the payload behind our back.
    alt_nan = struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_0123))[0]
    std = np.array([1.0, float("nan"), 3.0, 4.0], dtype="<f8")
    alt = np.array([1.0, alt_nan, 3.0, 4.0], dtype="<f8")
    # precondition: the payloads really differ at the bit level (else this test
    # would be the impossible-to-fail kind)
    assert std[1:2].tobytes() != alt[1:2].tobytes()
    assert canonical_content_hash(_panel(std, KEYS)) == canonical_content_hash(
        _panel(alt, KEYS)
    )


def test_hash_keeps_signed_zero_distinct():
    plus = _panel([0.0, 2.0, 3.0, 4.0], KEYS)
    minus = _panel([-0.0, 2.0, 3.0, 4.0], KEYS)
    assert canonical_content_hash(plus) != canonical_content_hash(minus)


def test_hash_accepts_single_column_frame_rejects_multi():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    assert canonical_content_hash(base.to_frame()) == canonical_content_hash(base)
    two = pd.concat([base.rename("a"), base.rename("b")], axis=1)
    with pytest.raises(ValueError, match="single-column"):
        canonical_content_hash(two)


def test_hash_column_name_is_not_content():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS, name="a")
    renamed = base.rename("b")
    assert canonical_content_hash(base) == canonical_content_hash(renamed)


def test_hash_rejects_duplicate_keys():
    keys = list(KEYS)
    keys[1] = keys[0]
    with pytest.raises(ValueError, match="duplicate"):
        canonical_content_hash(_panel([1.0, 2.0, 3.0, 4.0], keys))


def test_hash_rejects_wrong_level_names():
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    wrong = base.copy()
    wrong.index = wrong.index.set_names(["d", SYMBOL_LEVEL])
    with pytest.raises(ValueError, match="named"):
        canonical_content_hash(wrong)


def test_hash_rejects_non_datetime_dates():
    index = pd.MultiIndex.from_tuples(
        [("2024-01-02", "000001.SZ"), ("2024-01-03", "000001.SZ")],
        names=[DATE_LEVEL, SYMBOL_LEVEL],
    )
    series = pd.Series([1.0, 2.0], index=index)
    with pytest.raises(ValueError, match="datetime64"):
        canonical_content_hash(series)


def test_hash_rejects_flat_index_and_non_numeric():
    flat = pd.Series([1.0], index=pd.Index([0]))
    with pytest.raises(ValueError, match="MultiIndex"):
        canonical_content_hash(flat)
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001.SZ")], names=[DATE_LEVEL, SYMBOL_LEVEL]
    )
    with pytest.raises(ValueError, match="numeric"):
        canonical_content_hash(pd.Series(["x"], index=index))


# --------------------------------------------------------------------------- #
# atomic_write_parquet
# --------------------------------------------------------------------------- #
def test_atomic_write_round_trips_and_leaves_no_tmp(tmp_path: Path):
    base = _panel([1.0, np.nan, 3.0, 4.0], KEYS)
    target = tmp_path / "factor_x.parquet"
    sha = atomic_write_parquet(base, target)
    assert target.exists()
    assert not target.with_name(target.name + ".tmp").exists()
    assert sha == file_sha256(target)
    read = (
        pd.read_parquet(target)
        .set_index([DATE_LEVEL, SYMBOL_LEVEL])["factor_x"]
        .sort_index(kind="mergesort")
    )
    pd.testing.assert_series_equal(read, base.sort_index(kind="mergesort"))
    assert canonical_content_hash(read) == canonical_content_hash(base)


def test_read_frozen_panel_round_trips_and_is_loud_on_wrong_column(tmp_path: Path):
    base = _panel([1.0, np.nan, 3.0, 4.0], KEYS)
    target = tmp_path / "factor_x.parquet"
    atomic_write_parquet(base, target)
    read = read_frozen_panel(target, "factor_x")
    assert canonical_content_hash(read) == canonical_content_hash(base)
    with pytest.raises(ValueError, match="no column"):
        read_frozen_panel(target, "factor_y")


def test_atomic_write_failure_leaves_no_tmp_and_keeps_target(
    tmp_path: Path, monkeypatch
):
    base = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    target = tmp_path / "factor_x.parquet"
    atomic_write_parquet(base, target)
    before = target.read_bytes()

    def broken_to_parquet(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial garbage")
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken_to_parquet)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_write_parquet(base, target)
    assert not target.with_name(target.name + ".tmp").exists()
    assert target.read_bytes() == before  # the existing artifact is untouched


# --------------------------------------------------------------------------- #
# manifest rows + renderer
# --------------------------------------------------------------------------- #
def test_manifest_row_fields_complete_and_correct():
    base = _panel([1.0, np.nan, 3.0, 5.0], KEYS)
    row = manifest_row("factor_x", "minute", base, "c" * 64, "f" * 64, "factor_x.parquet")
    assert tuple(row.keys()) == MANIFEST_ROW_FIELDS
    assert row["rows"] == 4
    assert row["n_nan"] == 1
    assert row["n_symbols"] == 2
    assert row["date_min"] == "2024-01-02"
    assert row["date_max"] == "2024-01-03"
    assert row["mean"] == pytest.approx(float(base.mean()))
    assert row["std"] == pytest.approx(float(base.std()))  # pandas ddof=1, NaN skipped
    assert row["canonical_sha256"] == "c" * 64
    assert row["file_sha256"] == "f" * 64
    assert row["kind"] == "minute"
    assert row["file"] == "factor_x.parquet"


def test_render_manifest_markdown_deterministic_and_full_precision():
    base = _panel([1.0, 2.0, 3.0, 4.000000000000123], KEYS)
    row = manifest_row("factor_x", "book", base, "c" * 64, "f" * 64, "factor_x.parquet")
    header = {"producing_git_sha": "abc123", "window": "2021-07-01..2026-06-30"}
    text = render_manifest_markdown(header, [row])
    assert text == render_manifest_markdown(header, [row])  # deterministic
    assert "abc123" in text and "factor_x" in text and "c" * 64 in text
    # floats are rendered via repr -> full precision survives the markdown
    assert repr(row["mean"]) in text


def test_determinism_subjects_cover_two_minute_and_one_book():
    assert "value_ep" in DETERMINISM_FACTORS  # the book subject
    minute = [f for f in DETERMINISM_FACTORS if f.endswith("_20")]
    assert len(minute) >= 2  # two minute subjects (incl. the panel-consuming loader)


# --------------------------------------------------------------------------- #
# eval-artifact reconciliation
# --------------------------------------------------------------------------- #
def _write_eval_json(reports_dir: Path, stem: str, payload: dict) -> None:
    document = {"sections": [{"name": "data_coverage", "payload": payload}]}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{stem}_no_book.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def _matching_payload(processed: pd.Series, declared: list[str]) -> dict:
    values = processed.to_numpy(dtype=float)
    evaluated = set(processed.index.get_level_values(SYMBOL_LEVEL))
    return {
        "panel_rows": int(len(processed)),
        "evaluation_periods": int(
            pd.unique(processed.index.get_level_values(DATE_LEVEL)).size
        ),
        "symbols_evaluated": len(evaluated),
        "universe_symbols_declared": len(declared),
        "dropped_symbols_count": len(set(declared) - evaluated),
        "factor_nan_rate": round(1.0 - np.isfinite(values).sum() / len(values), 6),
    }


def test_reconcile_passes_on_exact_match(tmp_path: Path):
    processed = _panel([1.0, np.nan, 3.0, 4.0], KEYS)
    declared = ["000001.SZ", "600000.SH", "999999.SZ"]
    _write_eval_json(tmp_path, "eval_x", _matching_payload(processed, declared))
    out = reconcile_with_eval_artifact("eval_x", processed, declared, tmp_path)
    assert out["artifact"] == "eval_x_no_book.json"
    assert all(check["ok"] for check in out["checks"].values())


def test_reconcile_raises_on_mismatch(tmp_path: Path):
    processed = _panel([1.0, np.nan, 3.0, 4.0], KEYS)
    declared = ["000001.SZ", "600000.SH"]
    payload = _matching_payload(processed, declared)
    payload["panel_rows"] += 1  # the artifact claims one more row than we froze
    _write_eval_json(tmp_path, "eval_x", payload)
    with pytest.raises(ValueError, match="panel_rows"):
        reconcile_with_eval_artifact("eval_x", processed, declared, tmp_path)


def test_reconcile_raises_on_nan_rate_mismatch(tmp_path: Path):
    processed = _panel([1.0, np.nan, 3.0, 4.0], KEYS)
    declared = ["000001.SZ", "600000.SH"]
    payload = _matching_payload(processed, declared)
    payload["factor_nan_rate"] = round(payload["factor_nan_rate"] + 1e-6, 6)
    _write_eval_json(tmp_path, "eval_x", payload)
    with pytest.raises(ValueError, match="factor_nan_rate"):
        reconcile_with_eval_artifact("eval_x", processed, declared, tmp_path)


def test_reconcile_missing_artifact_is_loud(tmp_path: Path):
    processed = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    with pytest.raises(FileNotFoundError, match="no_book.json"):
        reconcile_with_eval_artifact("eval_x", processed, ["000001.SZ"], tmp_path)


def test_reconcile_missing_payload_is_loud(tmp_path: Path):
    (tmp_path / "eval_x_no_book.json").write_text(
        json.dumps({"sections": [{"name": "other", "payload": {"a": 1}}]}),
        encoding="utf-8",
    )
    processed = _panel([1.0, 2.0, 3.0, 4.0], KEYS)
    with pytest.raises(ValueError, match="data_coverage"):
        reconcile_with_eval_artifact("eval_x", processed, ["000001.SZ"], tmp_path)
