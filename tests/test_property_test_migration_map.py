"""R16: the property-test migration map must stay true to the suite.

``docs/factors/d5_property_test_migration_map.md`` is the admission ticket for
deleting the eleven legacy eval runners and their tests: it claims that each of
the 59 legacy runner tests has a mapping into the new suite, so coverage does
not fall. A claim like that is worth exactly as much as its citations, and a
document cannot notice when a test it names is renamed away.

So these tests read the map and check it against the tree:

* every new-suite test the map cites exists, resolves to ONE file, and family
  wildcards have the size the document claims;
* while the legacy files still exist, none of their tests is absent from the
  map.

Guard range, stated here because a guard that does not say what it cannot see
invites being trusted further than it should be: this checks that citations
RESOLVE, not that the cited test still asserts the property the map claims for
it. A test renamed away is caught; a test gutted in place is not.

The second check goes vacuous the day the legacy files are deleted — by design.
Its job is to make an UNMAPPED legacy test impossible while they exist, not to
keep them alive; a guard that failed on their deletion would be a booby trap for
the very PR this map exists to authorise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAP = REPO / "docs/factors/d5_property_test_migration_map.md"
TESTS = REPO / "tests"

QUALIFIED = re.compile(
    r"tests/test_[A-Za-z0-9_]+\.py::(test_[A-Za-z0-9_]+)(?![A-Za-z0-9_*])"
)
# ``(?![A-Za-z0-9_*])`` on both patterns: a bare ``(?!\*)`` is defeated by
# backtracking (the name simply gives back its trailing underscore and matches a
# shorter, still-wrong name), which reports three phantom missing tests.
BARE = re.compile(r"(?<!\w)::(test_[A-Za-z0-9_]+)(?![A-Za-z0-9_*])")
FAMILY = re.compile(r"::(test_[A-Za-z0-9_]+)_\*`?（(\d+) 条）")


def _suite_test_names() -> dict[str, set[str]]:
    """{test function name: {files defining it}} across the whole suite."""
    names: dict[str, set[str]] = {}
    for path in sorted(TESTS.rglob("test_*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("def test"):
                names.setdefault(line[4:].split("(")[0], set()).add(path.name)
    return names


def _map_text() -> str:
    return MAP.read_text(encoding="utf-8")


def _legacy_runner_tests() -> dict[str, list[str]]:
    return {
        path.name: [
            line[4:].split("(")[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("def test")
        ]
        for path in sorted(TESTS.glob("test_eval_*_runner.py"))
    }


def test_the_map_document_exists_and_carries_citations():
    """Anti-vacuity: every check below is a loop over parsed citations, and an
    empty parse would make all of them pass."""
    text = _map_text()
    cited = set(QUALIFIED.findall(text)) | set(BARE.findall(text))
    assert len(cited) >= 40, f"only {len(cited)} citations parsed — parser or map broke"


@pytest.mark.parametrize("pattern", [QUALIFIED, BARE], ids=["qualified", "bare"])
def test_every_cited_new_suite_test_exists_and_is_unambiguous(pattern: re.Pattern):
    suite = _suite_test_names()
    missing, ambiguous = [], []
    for name in sorted(set(pattern.findall(_map_text()))):
        files = suite.get(name)
        if not files:
            missing.append(name)
        elif len(files) > 1:
            ambiguous.append(f"{name} -> {sorted(files)}")
    assert not missing, f"cited by the migration map but not defined anywhere: {missing}"
    # A duplicated test-function name silently drops one of the two from a full
    # run in this repo (no __init__.py in tests/), so an ambiguous citation is a
    # finding in its own right.
    assert not ambiguous, f"cited name defined in more than one file: {ambiguous}"


def test_every_family_wildcard_has_the_size_the_document_claims():
    suite = _suite_test_names()
    families = FAMILY.findall(_map_text())
    assert families, "no family wildcards parsed — the pattern or the map changed"
    for stem, claimed in families:
        prefix = f"{stem}_"
        found = sorted(name for name in suite if name.startswith(prefix))
        assert len(found) == int(claimed), (
            f"{prefix}*: document claims {claimed}, suite has {len(found)}: {found}"
        )


def test_no_legacy_runner_test_is_missing_from_the_map():
    """Vacuous once the legacy files are deleted — see the module docstring."""
    text = _map_text()
    legacy = _legacy_runner_tests()
    unmapped = {
        file_name: [name for name in names if f"`{name}`" not in text]
        for file_name, names in legacy.items()
    }
    unmapped = {k: v for k, v in unmapped.items() if v}
    assert not unmapped, f"legacy tests with no row in the migration map: {unmapped}"


def test_the_counting_table_is_internally_consistent():
    """7 migrated + 50 covered + 2 carrier-retired must equal the 59 claimed."""
    text = _map_text()
    numbers = {
        label: int(re.search(pattern, text).group(1))
        for label, pattern in (
            ("total", r"旧套件总条数（10 文件） \| \*\*(\d+)\*\*"),
            ("migrated", r"已搬迁（逐字移入新文件） \| (\d+)"),
            ("covered", r"已覆盖（性质在新套件有钉） \| (\d+)"),
            ("retired", r"载体退役（[^|]*） \| (\d+)"),
        )
    }
    assert (
        numbers["migrated"] + numbers["covered"] + numbers["retired"] == numbers["total"]
    ), numbers
    assert "59 / 59（100%）" in text
    assert "完全无映射、无理由退役 | **0**" in text


def test_the_claimed_total_matches_the_legacy_files_while_they_exist():
    legacy = _legacy_runner_tests()
    if not legacy:  # deleted: the claim is history, not a live count
        pytest.skip("legacy runner test files are gone; the count is historical")
    assert len(legacy) == 10
    assert sum(len(names) for names in legacy.values()) == 59
