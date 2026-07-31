"""Tests for the git-tracked frozen-baseline manifest document parser.

The parser is the load-bearing half of "the hashes live in git": if it silently
returned an empty or partial expectation set, ``--verify`` would pass a tampered
tree while looking busy. So every failure mode it is supposed to be loud about
gets a test, and the two REAL documents are parsed here as a coupling check —
a reformat that breaks the table stops the suite, not a verification run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qt.frozen_manifest_doc import parse_doc_manifest, parse_doc_producing_sha
from qt.panel_freeze import D1_MANIFEST_DOC, PR_C_MANIFEST_DOC

REPO = Path(__file__).resolve().parents[1]

HEADER = "| factor_id | rows | canonical_sha256 |"
SEPARATOR = "|---|---|---|"
HASH_A = "a" * 64
HASH_B = "b" * 64


def _doc(tmp_path: Path, body: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_parses_the_table_and_reads_columns_by_name(tmp_path: Path):
    path = _doc(
        tmp_path,
        "\n".join(
            [
                "# heading",
                "",
                "| item | value |",  # decoy table without the required columns
                "|---|---|",
                "| universe | CSI500 |",
                "",
                HEADER,
                SEPARATOR,
                f"| alpha_20 | 10 | `{HASH_A}` |",
                f"| beta_20 | **20** | {HASH_B} |",
            ]
        ),
    )
    rows = parse_doc_manifest(path)
    assert set(rows) == {"alpha_20", "beta_20"}
    assert rows["alpha_20"].canonical_sha256 == HASH_A
    assert rows["beta_20"].rows == 20  # bold markers stripped, read by column NAME


def test_absent_columns_are_none_rather_than_invented(tmp_path: Path):
    """The PR-C table carries no ``kind`` / ``file_sha256``; a verifier that made
    values up for them would be checking its own invention."""
    path = _doc(
        tmp_path,
        "\n".join([HEADER, SEPARATOR, f"| alpha_20 | 10 | {HASH_A} |"]),
    )
    row = parse_doc_manifest(path)["alpha_20"]
    assert row.kind is None and row.file_sha256 is None and row.n_nan is None
    assert row.rows == 10


def test_two_matching_tables_is_a_readable_error(tmp_path: Path):
    path = _doc(
        tmp_path,
        "\n".join(
            [HEADER, SEPARATOR, f"| alpha_20 | 10 | {HASH_A} |", "",
             HEADER, SEPARATOR, f"| alpha_20 | 10 | {HASH_B} |"]
        ),
    )
    with pytest.raises(ValueError, match="exactly ONE table"):
        parse_doc_manifest(path)


def test_no_matching_table_is_a_readable_error(tmp_path: Path):
    path = _doc(tmp_path, "# nothing here\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    with pytest.raises(ValueError, match="exactly ONE table"):
        parse_doc_manifest(path)


def test_duplicate_factor_id_is_rejected(tmp_path: Path):
    path = _doc(
        tmp_path,
        "\n".join(
            [HEADER, SEPARATOR, f"| alpha_20 | 10 | {HASH_A} |",
             f"| alpha_20 | 10 | {HASH_B} |"]
        ),
    )
    with pytest.raises(ValueError, match="duplicate factor id"):
        parse_doc_manifest(path)


@pytest.mark.parametrize(
    "bad", ["not-a-hash", "A" * 64, "a" * 63, ""],
)
def test_a_malformed_hash_is_rejected(tmp_path: Path, bad: str):
    path = _doc(tmp_path, "\n".join([HEADER, SEPARATOR, f"| alpha_20 | 10 | {bad} |"]))
    with pytest.raises(ValueError, match="64 lowercase hex"):
        parse_doc_manifest(path)


def test_a_ragged_row_is_rejected(tmp_path: Path):
    path = _doc(tmp_path, "\n".join([HEADER, SEPARATOR, f"| alpha_20 | {HASH_A} |"]))
    with pytest.raises(ValueError, match="cells but"):
        parse_doc_manifest(path)


def test_producing_sha_is_read_from_the_document(tmp_path: Path):
    sha = "3" * 40
    path = _doc(tmp_path, f"- **producing SHA**: `{sha}` (some note)\n")
    assert parse_doc_producing_sha(path) == sha


def test_producing_sha_absent_reads_as_none(tmp_path: Path):
    assert parse_doc_producing_sha(_doc(tmp_path, "# no provenance line\n")) is None


# --------------------------------------------------------------------------- #
# Coupling to the REAL git-tracked documents
# --------------------------------------------------------------------------- #
def test_the_real_d1_document_lists_the_fourteen_frozen_factors():
    rows = parse_doc_manifest(REPO / D1_MANIFEST_DOC)
    assert len(rows) == 14
    assert {"value_ep", "value_bp", "volatility_20"} <= set(rows)  # the 3 book factors
    assert sum(1 for row in rows.values() if row.kind == "minute") == 11
    assert all(row.file_sha256 is not None for row in rows.values())


def test_the_real_pr_c_document_lists_the_one_corrected_panel():
    rows = parse_doc_manifest(REPO / PR_C_MANIFEST_DOC)
    assert set(rows) == {"jump_amount_corr_20"}
    # The corrected panel is a DIFFERENT factor definition from the D1 row of the
    # same name; if these ever hashed equal, one of the two documents is wrong.
    d1 = parse_doc_manifest(REPO / D1_MANIFEST_DOC)["jump_amount_corr_20"]
    assert rows["jump_amount_corr_20"].canonical_sha256 != d1.canonical_sha256


def test_both_real_documents_pin_a_producing_sha():
    for doc in (D1_MANIFEST_DOC, PR_C_MANIFEST_DOC):
        sha = parse_doc_producing_sha(REPO / doc)
        assert sha is not None and len(sha) == 40, doc
