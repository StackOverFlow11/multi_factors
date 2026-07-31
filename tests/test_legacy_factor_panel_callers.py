"""Who still calls the PRE-D6a factor path, derived from the repo — not listed.

D6a moved phase0 and phase2_baseline onto ``_serve_factor_panel`` (the factor
service) and left ``qt.pipeline._compute_factor_panel`` in place, unchanged, for
the runners D6b migrates. Two things then need guarding, and NOTHING in the test
suite executes those runners: they need real tushare data, so a break in them is
invisible to every other test here. That is not hypothetical — the first draft of
D6a changed the shared signature under them and the whole suite stayed green.

  1. every call to the legacy entry point still matches its live signature; and
  2. the set of files that call it is EXACTLY the set we think it is.

(2) is the one that matters, and it is why this census is DERIVED rather than
listed. A hand-kept list of "runners still on the old path" only ever confirms
what its author already knew: add a third caller in a file the list does not
name, change the signature, and dutifully update everything the list DOES name,
and a listed guard passes in silence. Measured on the previous, hand-kept
version of this guard: exactly that scenario left all 21 tests green.

SCOPE OF THE DERIVATION — stated because a guard that does not say what it
cannot see is worth less than no guard:

* FILES: ``git ls-files`` UNION ``git ls-files --others --exclude-standard``,
  i.e. tracked files plus untracked-but-not-ignored ones. Tracked-only would be
  blind to a caller added but not yet staged; a bare directory walk would drag in
  gitignored artifacts (and, in a worktree, follow the ``artifacts`` symlink into
  another checkout entirely).
* CALL FORMS: a direct call (``_compute_factor_panel(...)``), an attribute call
  on any object (``pipeline._compute_factor_panel(...)``), and a call through a
  local import alias (``from qt.pipeline import _compute_factor_panel as X``;
  ``X(...)``). ``getattr(obj, "_compute_factor_panel")`` cannot be bound, so it
  is REFUSED by name rather than skipped.
* WHAT IT STILL CANNOT SEE: a call assembled from a dynamically built name
  (``getattr(obj, "_compute_" + "factor_panel")``). No test here claims
  otherwise.

⚠️ THIS FILE IS SUPPOSED TO GO RED IN D6b, AND THEN BE DELETED. D6b moves
``oos_stability`` / ``subset_validation`` onto the service and D6d deletes
``_compute_factor_panel`` entirely; at that point ``EXPECTED_CALLERS`` no longer
matches and the symbol no longer exists. That is the guard reporting the
migration finished, not a stale test. Delete it with the function; do not "fix"
it by emptying the expected set while the function still has callers.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from qt import pipeline

#: The legacy entry point this census tracks.
SYMBOL = "_compute_factor_panel"

#: The module that DEFINES it — excluded from the caller census because a
#: definition is not a call. Derived below by finding the ``FunctionDef``, so a
#: move to another module does not silently drop the exclusion.
DEFINING_MODULE = Path(inspect.getsourcefile(pipeline)).name

#: Every file that calls it today, repo-relative.
#:
#: ``qt/oos_stability.py`` and ``qt/subset_validation.py`` are the two runners
#: D6b migrates. ``tests/test_factor_source.py`` calls it too, deliberately: it
#: is the only execution coverage the legacy path has, since the two runners
#: cannot be run without real data.
EXPECTED_CALLERS: frozenset[str] = frozenset(
    {
        "qt/oos_stability.py",
        "qt/subset_validation.py",
        "tests/test_factor_source.py",
    }
)

#: A census that scans nothing passes vacuously. The repo has ~300 python files;
#: this floor only has to be high enough that an empty or broken file listing
#: cannot read as "no callers found".
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


def _local_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to ``SYMBOL`` by an import in this module."""
    names = {SYMBOL}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == SYMBOL and alias.asname:
                    names.add(alias.asname)
    return names


def _calls_in(path: Path) -> tuple[list[ast.Call], list[str]]:
    """(bindable call nodes, unbindable references) for one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # not our business to compile the repo
        return [], []
    aliases = _local_aliases(tree)
    calls: list[ast.Call] = []
    unbindable: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in aliases:
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == SYMBOL:
            calls.append(node)
        elif (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and any(
                isinstance(a, ast.Constant) and a.value == SYMBOL for a in node.args
            )
        ):
            unbindable.append(f"{path.name}:{node.lineno} getattr(...)")
    return calls, unbindable


def _census() -> tuple[dict[str, list[ast.Call]], list[str], int]:
    root = _repo_root()
    callers: dict[str, list[ast.Call]] = {}
    unbindable: list[str] = []
    files = _python_files()
    for path in files:
        if path.name == DEFINING_MODULE:
            continue
        calls, bad = _calls_in(path)
        unbindable.extend(bad)
        if calls:
            callers[path.relative_to(root).as_posix()] = calls
    return callers, unbindable, len(files)


def test_the_file_census_actually_scanned_the_repo():
    """Distinguishes 'no callers' from 'the file listing came back empty'."""
    files = _python_files()
    assert len(files) >= _MIN_FILES_SCANNED, (
        f"the python-file derivation returned only {len(files)} file(s); every "
        f"census below would pass vacuously. Check the git listing, not the "
        f"callers."
    )


def test_the_set_of_legacy_callers_is_exactly_what_we_think_it_is():
    """The load-bearing one: a NEW caller anywhere in the repo turns this red.

    Red here is STOP-AND-REPORT, and which report depends on which way it moved:

    * an ADDED file — someone put a new caller on the pre-D6a path. Route it
      through ``_serve_factor_panel`` instead, or say why it must not be.
    * a REMOVED file — expected exactly once, when D6b migrates the two runners.
      Then this whole module goes away with ``_compute_factor_panel`` (D6d); do
      not keep it alive against an empty set.
    """
    callers, _unbindable, _n = _census()
    found = frozenset(callers)
    assert found, (
        f"no caller of {SYMBOL} found anywhere in the repo. Either the function "
        f"is already dead (then delete it and this module together) or the "
        f"derivation broke — check "
        f"test_the_file_census_actually_scanned_the_repo first."
    )
    assert found == EXPECTED_CALLERS, (
        f"the set of files calling {SYMBOL} changed.\n"
        f"  added:   {sorted(found - EXPECTED_CALLERS)}\n"
        f"  removed: {sorted(EXPECTED_CALLERS - found)}\n"
        f"STOP AND REPORT — see this test's docstring for which of the two "
        f"situations you are in."
    )


def test_every_legacy_call_matches_the_live_signature():
    """Parsed from source, bound against the function object as it exists now."""
    callers, _unbindable, _n = _census()
    signature = inspect.signature(getattr(pipeline, SYMBOL))
    for rel, calls in sorted(callers.items()):
        for call in calls:
            try:
                signature.bind(
                    *["<arg>"] * len(call.args),
                    **{kw.arg: "<arg>" for kw in call.keywords if kw.arg},
                )
            except TypeError as exc:  # noqa: PERF203 - one message per bad call site
                pytest.fail(
                    f"{rel}:{call.lineno} calls {SYMBOL} with a shape its current "
                    f"signature {signature} does not accept ({exc}). Nothing in "
                    f"the suite executes that call, so this guard is the only "
                    f"thing between the change and a runtime TypeError."
                )


def test_no_caller_reaches_the_symbol_in_a_form_this_guard_cannot_bind():
    """``getattr`` by name is refused, not silently skipped."""
    _callers, unbindable, _n = _census()
    assert not unbindable, (
        f"{SYMBOL} is referenced through getattr at {unbindable}, a form this "
        f"census can find but cannot bind against the signature. Make it a "
        f"direct call, or extend the guard deliberately — do not leave a "
        f"reference it can only half-see."
    )
