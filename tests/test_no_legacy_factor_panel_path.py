"""D6d successor census: the deleted legacy factor-panel path STAYS deleted.

D6d removed ``qt.pipeline._compute_factor_panel`` — the pre-D6a second
factor-sourcing path (design decision 3: one path, the factor service) —
together with its caller census (``tests/test_legacy_factor_panel_callers.py``,
whose own docstring said it goes away with the function). This guard is what
keeps the deletion from quietly reverting, on the same derived-not-listed
mechanism as the deleted census:

  1. the symbol ``_compute_factor_panel`` does not reappear anywhere in the
     repo's python files — no definition, no call, no import; and
  2. no production code assembles a factor panel by calling ``.compute(...)``
     on the loop variable of a comprehension — the
     ``[factor.compute(panel) ... for factor in factors]`` shape the deleted
     path was built on.

SCOPE OF THE DERIVATION — stated because a guard that does not say what it
cannot see is worth less than no guard:

* FILES: ``git ls-files`` UNION ``git ls-files --others --exclude-standard``,
  i.e. tracked files plus untracked-but-not-ignored ones (same derivation as
  the deleted census: tracked-only would be blind to a file added but not yet
  staged; a bare directory walk would drag in gitignored artifacts).
* (1) reads AST name/attribute/import/getattr nodes. Prose is not a symbol:
  docstrings that TALK about the deleted function in past tense
  (``qt/phase3_capture.py`` keeps several) do not trip it, by construction.
* (2) is scoped to production files (``tests/`` excluded — a test may
  legitimately inline ``factor.compute`` to build an expected value), and
  matches comprehension shapes only. Two production sites are SANCTIONED, each
  with its reason inline at :data:`SANCTIONED_COMPREHENSION_SITES`; the pin is
  exact in both directions so a NEW site anywhere — including elsewhere in
  those two files — turns this red, and a sanctioned site going missing turns
  it red too (that is how the census tells "no violations" apart from "the
  detector went blind").
* WHAT IT STILL CANNOT SEE: a panel assembled by calling ``.compute`` outside
  a comprehension (a plain for-loop appending columns), a ``.compute`` call
  hidden in a comprehension's ``if`` clause rather than its element
  expression(s), a call through a
  dynamically built name (``getattr(obj, "_compute_" + "factor_panel")``), and
  anything in a file git ignores. No test here claims otherwise.

ANTI-VACUITY: with the expected violation set empty, a blind guard reads as
green. Three floors make "the guard cannot see" red instead: the file census
must cover the repo (``_MIN_FILES_SCANNED``), the symbol predicate must prove
itself on a synthetic snippet (flag the symbol, pass a near-miss), and the
comprehension census must keep finding the two SANCTIONED sites.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

#: The deleted pre-D6a entry point this census keeps deleted.
SYMBOL = "_compute_factor_panel"

#: Production sites that call ``.compute(...)`` on a comprehension's loop
#: variable ON PURPOSE, as (repo-relative file, enclosing function):
#:
#: * ``qt/phase3_capture.py::legacy_factor_panel`` — the D6b capture harness
#:   keeps the legacy math verbatim as its reconciliation reference; D6d
#:   deleted the pipeline helper it mirrored, so this copy IS the reference.
#: * ``qt/factor_eval_runner.py::_load_book_raw`` — the close-book mode is
#:   documented to "replicate the legacy runners exactly" (direct compute on
#:   the close-view enriched panel), as the control leg against the
#:   decision-view service panel.
SANCTIONED_COMPREHENSION_SITES: frozenset[tuple[str, str]] = frozenset(
    {
        ("qt/phase3_capture.py", "legacy_factor_panel"),
        ("qt/factor_eval_runner.py", "_load_book_raw"),
    }
)

#: A census that scans nothing passes vacuously. The repo has ~330 python
#: files; this floor only has to be high enough that an empty or broken file
#: listing cannot read as "nothing found".
_MIN_FILES_SCANNED = 100


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_files() -> list[Path]:
    root = _repo_root()
    seen: dict[str, Path] = {}
    for args in (
        ["git", "ls-files", "--", "*.py"],
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
    ):
        out = subprocess.run(
            args, cwd=root, capture_output=True, text=True, check=True
        ).stdout
        for line in out.splitlines():
            rel = line.strip()
            if rel:
                seen.setdefault(rel, root / rel)
    return [path for _rel, path in sorted(seen.items())]


def _symbol_references(tree: ast.AST) -> list[int]:
    """Line numbers where ``SYMBOL`` is defined, referenced, or imported."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == SYMBOL
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == SYMBOL:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == SYMBOL:
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(
            SYMBOL in (alias.name, alias.asname) for alias in node.names
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and any(
                isinstance(a, ast.Constant) and a.value == SYMBOL for a in node.args
            )
        ):
            lines.append(node.lineno)
    return lines


def _compute_comprehension_sites(tree: ast.AST) -> list[tuple[str, int]]:
    """(enclosing function, line) of comprehensions calling ``.compute`` on
    their own loop variable."""

    sites: list[tuple[str, int]] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                targets = {
                    t.id
                    for gen in child.generators
                    for t in ast.walk(gen.target)
                    if isinstance(t, ast.Name)
                }
                _elts = (child.elt,) if not isinstance(child, ast.DictComp) else (child.key, child.value)
                if any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "compute"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in targets
                    for elt in _elts
                    for n in ast.walk(elt)
                ):
                    sites.append((function, child.lineno))
            visit(child, function)

    visit(tree, "<module>")
    return sites


def _census() -> tuple[dict[str, list[int]], dict[str, list[tuple[str, int]]], int]:
    root = _repo_root()
    symbol_hits: dict[str, list[int]] = {}
    comprehension_hits: dict[str, list[tuple[str, int]]] = {}
    files = _python_files()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # not our business to compile the repo
            continue
        rel = path.relative_to(root).as_posix()
        refs = _symbol_references(tree)
        if refs:
            symbol_hits[rel] = refs
        if not rel.startswith("tests/"):
            sites = _compute_comprehension_sites(tree)
            if sites:
                comprehension_hits[rel] = sites
    return symbol_hits, comprehension_hits, len(files)


def test_the_file_census_actually_scanned_the_repo():
    """Distinguishes 'nothing found' from 'the file listing came back empty'."""
    files = _python_files()
    assert len(files) >= _MIN_FILES_SCANNED, (
        f"the python-file derivation returned only {len(files)} file(s); every "
        f"census below would pass vacuously. Check the git listing, not the "
        f"findings."
    )


def test_the_symbol_predicate_is_not_vacuous():
    """The detector must flag the symbol and pass a near-miss.

    With the expected set empty, an unproven predicate reads as green while
    seeing nothing; this synthetic pair is the proof it can still see.
    """
    dirty = ast.parse("pipeline._compute_factor_panel(cfg, panel, factors, logger)\n")
    clean = ast.parse("pipeline._serve_factor_panel(cfg, panel, factors, logger)\n")
    assert _symbol_references(dirty), "the predicate missed an attribute reference"
    assert not _symbol_references(clean), "the predicate flags the live entry point"


def test_no_legacy_factor_panel_symbol_anywhere():
    """The deleted entry point does not reappear — in ``qt/`` or anywhere else."""
    symbol_hits, _comprehension_hits, _n = _census()
    assert not symbol_hits, (
        f"{SYMBOL} reappeared: {symbol_hits}. D6d deleted the legacy "
        f"factor-panel path; route through ``_serve_factor_panel`` (the "
        f"factor service), do not resurrect the second path."
    )


def test_direct_compute_panel_assembly_is_only_the_sanctioned_sites():
    """The load-bearing one: a NEW direct-compute panel site turns this red.

    Red here is STOP-AND-REPORT, and which report depends on which way it moved:

    * an ADDED site — someone built a second factor-sourcing path next to the
      service. Route it through ``_serve_factor_panel`` /
      ``qt.factor_source.factor_values``, or sanction it here with its reason
      in the same commit.
    * a REMOVED site — a sanctioned reference was retired (then shrink the
      sanctioned set deliberately, in the same commit) or the detector broke
      (check ``test_the_file_census_actually_scanned_the_repo`` first).
    """
    _symbol_hits, comprehension_hits, _n = _census()
    found = frozenset(
        (rel, function)
        for rel, sites in comprehension_hits.items()
        for function, _line in sites
    )
    assert found == SANCTIONED_COMPREHENSION_SITES, (
        f"the set of comprehension-based direct-compute sites changed.\n"
        f"  added:   {sorted(found - SANCTIONED_COMPREHENSION_SITES)}\n"
        f"  removed: {sorted(SANCTIONED_COMPREHENSION_SITES - found)}\n"
        f"STOP AND REPORT — see this test's docstring for which of the two "
        f"situations you are in."
    )
