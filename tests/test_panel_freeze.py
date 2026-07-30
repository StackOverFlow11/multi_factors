"""Network-free tests for qt.panel_freeze (VERIFY-ONLY since D5 C6).

Synthetic panels only. Two halves:

* the pure machinery that DESCRIBES the frozen artifacts — canonical content
  hash (sensitive to any value / index change, row-order independent,
  NaN-payload independent, +0/-0 distinct, loud on malformed input), the atomic
  parquet write it was laid down with, the manifest row and its renderer;
* verification — every way a frozen tree can be wrong must come back red, and
  the retired regeneration entry point must fail loudly rather than no-op.

Every invariance claim here is paired with a sensitivity assertion on the same
axis (the shuffle test alone could be satisfied by a constant hash; the
value/index sensitivity tests kill that degenerate implementation). The same
rule applies to the verification tests: each starts from a tree that verifies
GREEN, so a red result is attributable to the tampering and not to a fixture
that never verified in the first place.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.clean.schema import DATE_LEVEL, SYMBOL_LEVEL
from qt.panel_freeze import (
    MANIFEST_ROW_FIELDS,
    RegenerationRetiredError,
    atomic_write_parquet,
    canonical_content_hash,
    file_sha256,
    main,
    manifest_row,
    read_frozen_panel,
    render_manifest_markdown,
    retirement_message,
    run_panel_freeze,
    verify_frozen_panels,
)
from tests.fixtures.frozen_baseline import (
    build_frozen_tree,
    make_panel,
    patch_manifest,
    rewrite_panel,
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




# --------------------------------------------------------------------------- #
# Retired regeneration entry point
# --------------------------------------------------------------------------- #
def test_regeneration_raises_instead_of_quietly_doing_nothing():
    with pytest.raises(RegenerationRetiredError, match="RETIRED"):
        run_panel_freeze()


def test_the_retirement_message_names_the_ruling_the_tool_and_the_way_forward():
    with pytest.raises(RegenerationRetiredError) as caught:
        run_panel_freeze("config/whatever.yaml")
    text = str(caught.value)
    assert "owner ruling 2026-07-28, D5 C6" in text
    assert "python -m qt.panel_freeze --verify" in text
    assert "no longer rebuilds" in text


def test_the_retirement_core_is_composed_not_restated():
    """The shared sentence is authored once and every tool composes it.

    A regex cannot assert that no other sentence says the same thing; "there is
    no other sentence" can (CLAUDE.md methodology #2). So: the core must come
    out of ``retirement_message``, and the per-tool clauses must be the ONLY
    difference between the three messages.
    """
    composed = retirement_message("tool-x", "product-y", "WHY.", "VERIFIES.")
    assert composed.startswith("tool-x: regeneration is RETIRED")
    assert "WHY." in composed and "VERIFIES." in composed
    source = Path(__file__).resolve().parents[1] / "qt" / "panel_freeze.py"
    body = source.read_text(encoding="utf-8")
    assert body.count("regeneration is RETIRED (") == 1


def test_the_old_command_line_still_parses_so_it_gets_the_explanation(capsys):
    """``--resume`` / ``--only`` were the documented regeneration flags. Dropping
    them from the parser would answer the old command with ``unrecognized
    arguments``, which reads like a broken install rather than a retirement."""
    assert main(["--resume", "--only", "value_ep"]) == 1
    assert "RETIRED" in capsys.readouterr().err


def test_a_bare_invocation_is_non_zero_and_explains_itself(capsys):
    assert main([]) == 1
    captured = capsys.readouterr()
    assert "RETIRED" in captured.err and "--verify" in captured.err


# --------------------------------------------------------------------------- #
# Verification: the green control, then one tampering per failure mode
# --------------------------------------------------------------------------- #
def test_an_untouched_frozen_tree_verifies_green(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    result = verify_frozen_panels(tmp_path, doc)
    assert result.ok and result.n_ok == 2 and not result.problems


def test_a_changed_cell_is_convicted_and_attributed_to_its_factor(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    assert verify_frozen_panels(tmp_path, doc).ok  # green control

    panel = make_panel("alpha_20")
    panel.iloc[0] = float(panel.iloc[0]) + 1e-9
    rewrite_panel(tmp_path, "alpha_20", panel)

    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    bad = {panel.factor_id for panel in result.panels if not panel.ok}
    assert bad == {"alpha_20"}
    assert any("canonical content hash" in p for p in _problems(result, "alpha_20"))


def test_moving_the_machine_manifest_too_does_not_buy_a_pass(tmp_path: Path):
    """The failure mode the git/gitignore split exists for: a tree regenerated
    together with its own manifest. Both local witnesses now agree with each
    other; the git-tracked document must still convict."""
    doc = build_frozen_tree(tmp_path)
    panel = make_panel("alpha_20")
    panel.iloc[0] = float(panel.iloc[0]) + 1e-9
    rewrite_panel(tmp_path, "alpha_20", panel)
    new_hash = canonical_content_hash(panel)
    new_file_hash = file_sha256(tmp_path / "panels" / "alpha_20.parquet")

    def _move(document):
        for row in document["rows"]:
            if row["factor_id"] == "alpha_20":
                row.update(
                    canonical_sha256=new_hash,
                    file_sha256=new_file_hash,
                    mean=float(panel.mean()),
                    std=float(panel.std()),
                )

    patch_manifest(tmp_path, "manifest.json", _move)
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("git-tracked" in p for p in _problems(result, "alpha_20"))


def test_a_missing_panel_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    (tmp_path / "panels" / "book_x.parquet").unlink()
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("missing from disk" in p for p in _problems(result, "book_x"))


def test_an_extra_panel_is_convicted_rather_than_ignored(tmp_path: Path):
    """An unregistered panel is a problem, not a bonus: the frozen inventory is
    defined by the document, so an extra file means someone wrote into the
    frozen tree."""
    doc = build_frozen_tree(tmp_path)
    atomic_write_parquet(make_panel("intruder_20"), tmp_path / "panels" / "intruder_20.parquet")
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("unregistered panel" in p for p in result.problems)


def test_a_missing_machine_manifest_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    (tmp_path / "manifest.json").unlink()
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("machine manifest missing" in p for p in result.problems)


def test_a_producing_sha_that_disagrees_with_the_document_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, producing_sha="a" * 40)
    patch_manifest(
        tmp_path, "manifest.json",
        lambda d: d["header"].update(producing_git_sha="b" * 40),
    )
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("producing SHA mismatch" in p for p in result.problems)


def test_a_reconciliation_recorded_as_not_ok_is_convicted(tmp_path: Path):
    """A divergent panel must never have been frozen; a manifest edited to say
    otherwise is exactly what this check is for."""
    doc = build_frozen_tree(tmp_path)

    def _fail_one(document):
        document["reconciliation"]["alpha_20"]["checks"]["panel_rows"]["ok"] = False

    patch_manifest(tmp_path, "manifest.json", _fail_one)
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("recorded as NOT ok" in p for p in result.problems)


def test_a_dropped_reconciliation_check_is_convicted(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    patch_manifest(
        tmp_path, "manifest.json",
        lambda d: d["reconciliation"]["alpha_20"]["checks"].pop("factor_nan_rate"),
    )
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    assert any("missing the 'factor_nan_rate' check" in p for p in result.problems)


def test_a_rewritten_but_content_identical_file_is_still_reported(tmp_path: Path):
    """Same values, new bytes. A frozen-forever file has no legitimate reason to
    be rewritten, so this is reported — and the message says which of the two it
    is, so nobody reads it as a value change."""
    doc = build_frozen_tree(tmp_path)
    target = tmp_path / "panels" / "alpha_20.parquet"
    before = file_sha256(target)
    frame = pd.read_parquet(target)
    frame.to_parquet(target, engine="pyarrow", index=False, compression="gzip")
    assert file_sha256(target) != before  # the mutation landed

    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    problems = _problems(result, "alpha_20")
    assert any("file sha256" in p and "CONTENT still matches" in p for p in problems)
    assert not any("canonical content hash" in p for p in problems)


def test_verify_writes_nothing_into_the_tree_it_checks(tmp_path: Path):
    doc = build_frozen_tree(tmp_path, with_d2=True)
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert verify_frozen_panels(tmp_path, doc, show_manifest=True).ok
    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    assert after == before


def test_show_manifest_renders_the_table_from_the_panels_on_disk(tmp_path: Path):
    doc = build_frozen_tree(tmp_path)
    result = verify_frozen_panels(tmp_path, doc, show_manifest=True)
    assert result.rendered_manifest is not None
    assert "alpha_20" in result.rendered_manifest
    assert verify_frozen_panels(tmp_path, doc).rendered_manifest is None


def _problems(result, factor_id: str) -> list[str]:
    return [
        problem
        for panel in result.panels
        if panel.factor_id == factor_id
        for problem in panel.problems
    ]


def test_a_machine_manifest_row_that_disagrees_with_its_panel_is_convicted(tmp_path: Path):
    """The document carries the hashes; the machine manifest carries the fuller
    statistical row. This case is the one only the SECOND witness can see: the
    document still agrees with the panel, and only the recorded row does not."""
    doc = build_frozen_tree(tmp_path)
    assert verify_frozen_panels(tmp_path, doc).ok  # green control

    def _lie(document):
        for row in document["rows"]:
            if row["factor_id"] == "alpha_20":
                row["n_nan"] = row["n_nan"] + 7

    patch_manifest(tmp_path, "manifest.json", _lie)
    result = verify_frozen_panels(tmp_path, doc)
    assert not result.ok
    problems = _problems(result, "alpha_20")
    assert any("manifest.json n_nan" in p for p in problems)
    # nothing else fired: the git-tracked side is untouched and still agrees
    assert not any("git-tracked" in p for p in problems)
