"""Factor code identity: enumerated shared set + AST import allowlist (D3, §3.4/R3).

A stored factor value is only valid while the CODE that produced it is unchanged.
``code_hash(factor)`` = content hash of THREE parts:

1. the factor's own module file;
2. the ENUMERATED shared set every factor leans on:
   ``{factors.compute.minute.primitives, factors.ops.*, factors.base, factors.spec}``;
3. the factor module's DIRECT project-internal imports that are NOT already in the
   shared set — folded ONE HOP, module-granular, derived from the AST (never a
   manual list, red line #6).

Part 3 closes the momentum->reversal stale-reuse hole (review MEDIUM): candidates.py
composes ``MomentumFactor``, so momentum.py's value surface (which ops function it
calls, ``price_col``, the reindex) folds into the code hash of every factor DEFINED
in candidates.py (reversal / value / volatility). A change to momentum.py therefore
invalidates those factors' keys — but NOT factors in other modules (e.g.
financial.py never imports momentum), so it is not a global over-invalidation.

ONE HOP only, no transitive-closure walk (design §3.4 "明确不做：闭包扩到全 data/"):
momentum's own rolling core lives in ``factors.ops``, which IS the shared set, so one
hop suffices. Granularity is the module FILE — folding momentum.py also invalidates
value_ep (also defined in candidates.py), the SAME granularity as "editing
candidates.py itself invalidates all its factors"; the direction is safe
(over-invalidation costs a recompute, under-invalidation serves a stale value).

Two consequences, both over-invalidate-safe and disclosed: (a) declaration-only
leaves reached this way (e.g. ``data.availability_policy``, imported by many factors)
fold into their importers' keys — harmless, since their content does not change a
computed value; (b) a PIT/schema module a factor imports (``data.clean.
intraday_schema``) folds here AND drives the DATA fingerprint's schema-version
dimension (``factors.store.fingerprint``) — the overlap only ever over-invalidates.

The AST import allowlist (:func:`module_import_violations`) still guards WHAT a
factor may import (stdlib + numpy/pandas + an enumerated set of first-party leaves);
a new first-party import goes red, forcing review. Dynamic imports (``importlib`` /
``__import__``) are refused (they would hide a dependency from the static walk).

Layering: imports the stdlib, ``factors.base`` (type only) and the sibling
``hashing`` leaf. Never qt / feeds / analytics.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

from factors.base import Factor
from factors.store.hashing import content_hash_of_labeled_files

# --------------------------------------------------------------------------- #
# The enumerated shared set (folded into every factor's code hash).
# --------------------------------------------------------------------------- #
#: Single-module members (design §3.4). ``factors.ops`` is a PACKAGE, expanded to
#: every ``*.py`` under it (so a new operator file joins the set automatically).
_SHARED_SET_SINGLE_MODULES: tuple[str, ...] = (
    "factors.compute.minute.primitives",
    "factors.base",
    "factors.spec",
)
_SHARED_SET_PACKAGES: tuple[str, ...] = ("factors.ops",)


def _module_file(module_name: str) -> Path:
    """Resolve an importable module to its source file path."""
    import importlib

    module = importlib.import_module(module_name)
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"cannot resolve a source file for module {module_name!r}.")
    return Path(source)


def _package_py_files(package_name: str) -> list[tuple[str, Path]]:
    """Every ``<package>.<stem>`` .py file under a package, as (label, path)."""
    import importlib

    package = importlib.import_module(package_name)
    pkg_file = inspect.getsourcefile(package)
    if pkg_file is None:
        raise RuntimeError(f"cannot resolve a source dir for package {package_name!r}.")
    pkg_dir = Path(pkg_file).parent
    out: list[tuple[str, Path]] = []
    for path in sorted(pkg_dir.glob("*.py")):
        out.append((f"{package_name}.{path.stem}", path))
    return out


def shared_set_labeled_files() -> tuple[tuple[str, Path], ...]:
    """The enumerated shared set as ``(module_label, path)`` pairs (sorted)."""
    items: list[tuple[str, Path]] = [
        (name, _module_file(name)) for name in _SHARED_SET_SINGLE_MODULES
    ]
    for package in _SHARED_SET_PACKAGES:
        items.extend(_package_py_files(package))
    return tuple(sorted(items, key=lambda t: t[0]))


def factor_module_labeled_file(factor: Factor | type[Factor]) -> tuple[str, Path]:
    """The factor's OWN module as ``(module_dotted_name, path)``."""
    cls = factor if isinstance(factor, type) else type(factor)
    module_name = cls.__module__
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError(f"cannot resolve a source file for {cls!r}.")
    return (module_name, Path(source))


def _in_shared_set(module_name: str) -> bool:
    """Whether ``module_name`` is already a shared-set member (folded globally)."""
    if module_name in _SHARED_SET_SINGLE_MODULES:
        return True
    return any(
        module_name == pkg or module_name.startswith(pkg + ".")
        for pkg in _SHARED_SET_PACKAGES
    )


def _direct_project_imports(source: str) -> set[str]:
    """The project-internal modules a source file imports DIRECTLY (AST, one hop).

    Absolute imports only (relative imports are refused by the allowlist anyway);
    filtered to first-party roots. No transitive walk — this is the ``import`` line
    set of exactly one file.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a factor module always parses
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module:
                modules.add(node.module)
    return {m for m in modules if _root(m) in _FIRST_PARTY_ROOTS}


def _onehop_dependency_files(
    own_module_name: str, own_path: Path
) -> list[tuple[str, Path]]:
    """One-hop code deps: the factor module's direct project imports NOT in the
    shared set, as ``(module_dotted_name, path)`` (module-granular, from the AST).
    """
    out: list[tuple[str, Path]] = []
    for module_name in sorted(_direct_project_imports(own_path.read_text())):
        if module_name == own_module_name or _in_shared_set(module_name):
            continue
        try:
            out.append((module_name, _module_file(module_name)))
        except Exception:  # noqa: BLE001 - an unresolvable import is skipped, not fatal
            continue
    return out


def code_hash(factor: Factor | type[Factor]) -> str:
    """Full sha256 hex of the factor's module + shared set + one-hop deps.

    Deterministic and checkout-independent (content + module labels only, never
    paths). Two factors defined in the SAME module (e.g. value_ep / volatility_20
    both in candidates.py) share this hash — their store keys still differ via
    factor_id + params_hash, but a change to that shared module invalidates both,
    which is correct. The ONE-HOP deps (module docstring) fold a composed module
    (candidates.py -> momentum.py) into the importer's hash, so a change there
    invalidates the factors that USE it without a transitive-closure walk.
    """
    own = factor_module_labeled_file(factor)
    shared = list(shared_set_labeled_files())
    onehop = _onehop_dependency_files(own[0], own[1])
    # Dedup by label: the own module + shared set take precedence; a one-hop dep
    # that is also (somehow) a shared-set label is already folded.
    labels = {own[0]} | {label for label, _ in shared}
    items = [own] + shared + [pair for pair in onehop if pair[0] not in labels]
    return content_hash_of_labeled_files(items)


# --------------------------------------------------------------------------- #
# AST import allowlist.
# --------------------------------------------------------------------------- #
#: First-party top-level packages of THIS repo. An import whose root is one of
#: these is "internal" and must match the allowlist below; anything else is
#: judged against the stdlib + numpy/pandas rule.
_FIRST_PARTY_ROOTS: frozenset[str] = frozenset(
    {"factors", "data", "analytics", "qt", "runtime", "alpha", "portfolio", "universe"}
)

#: The ONLY internal modules a factor module may import (dotted prefixes). A new
#: entry here is a deliberate, reviewed act — the point of the allowlist is that
#: adding one forces the question "should this be in the code-hash shared set?".
#: DELIBERATELY NARROW (review LOW-3): only modules a shipped factor actually
#: imports are listed. ``data.clean.schema`` / ``factors.requires`` were removed
#: as dead entries — if a future factor imports one, this test goes red and forces
#: an explicit decision (the intended behaviour). Every allowlisted internal import
#: a factor makes ALSO folds into that factor's code hash via the one-hop rule
#: (module docstring), so the allowlist and the code hash cannot silently drift.
ALLOWED_INTERNAL_IMPORTS: frozenset[str] = frozenset(
    {
        # shared compute set (folded into code_hash directly)
        "factors.base",
        "factors.spec",
        "factors.compute.minute.primitives",
        "factors.ops",
        # daily-factor composition (candidates.py -> momentum); folded into the
        # importer's code hash by the one-hop rule
        "factors.compute.momentum",
        # PIT / schema / declaration leaves (data layer). One-hop folds these into
        # the importer's code hash too (over-invalidate-safe); the schema module
        # ALSO drives the DATA fingerprint's schema-version dimension.
        "data.availability_policy",
        "data.clean.intraday_schema",
        "data.clean.intraday_aggregate",
    }
)

#: Third-party (non-stdlib) packages a factor module may import.
_ALLOWED_THIRD_PARTY: frozenset[str] = frozenset({"numpy", "pandas"})

#: Dynamic-import machinery that would hide a dependency from the static walk.
_FORBIDDEN_DYNAMIC = frozenset({"importlib", "__import__"})


def _root(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _import_allowed(dotted: str) -> bool:
    root = _root(dotted)
    if root in _FIRST_PARTY_ROOTS:
        return any(
            dotted == allowed or dotted.startswith(allowed + ".")
            for allowed in ALLOWED_INTERNAL_IMPORTS
        )
    if root in _ALLOWED_THIRD_PARTY:
        return True
    # stdlib is always allowed; anything else (a new third-party dep) is not.
    return root in sys.stdlib_module_names


def module_import_violations(source: str, *, module_name: str = "<factor module>") -> list[str]:
    """Return the import statements in ``source`` that break the factor allowlist.

    A factor module may import ONLY: the stdlib, numpy/pandas, and the enumerated
    :data:`ALLOWED_INTERNAL_IMPORTS`. Relative imports and dynamic imports
    (``importlib`` / ``__import__``) are refused (both hide dependencies from the
    static shared-set closure). An empty list means the module is clean.
    """
    try:
        tree = ast.parse(source, filename=module_name)
    except SyntaxError as exc:  # pragma: no cover - a factor module always parses
        return [f"{module_name}: could not parse ({exc})."]
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in _FORBIDDEN_DYNAMIC:
                    violations.append(f"dynamic import forbidden: import {alias.name}")
                elif not _import_allowed(alias.name):
                    violations.append(f"disallowed import: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                violations.append(
                    f"relative import forbidden (level {node.level}): "
                    f"from {'.' * node.level}{node.module or ''} import ..."
                )
                continue
            mod = node.module or ""
            if _root(mod) in _FORBIDDEN_DYNAMIC:
                violations.append(f"dynamic import forbidden: from {mod} import ...")
            elif not _import_allowed(mod):
                violations.append(f"disallowed import: from {mod} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "__import__":
                violations.append("dynamic import forbidden: __import__(...) call")
    return violations


def factor_source_files(registry=None) -> dict[str, Path]:
    """``{module_dotted_name: path}`` for every registered factor class.

    Deduplicated by module (value_ep / volatility_20 share candidates.py). Used by
    the D3 allowlist test to check exactly the modules whose code hash matters —
    which naturally excludes the legacy ``intraday_derived`` surface (no factor
    class is DEFINED there; the classes live in ``factors.compute.minute.*``).
    """
    if registry is None:
        from factors.registry.registry import DEFAULT_REGISTRY

        registry = DEFAULT_REGISTRY
    out: dict[str, Path] = {}
    entries = list(registry._exact.values()) + list(registry._prefixes)
    for entry in entries:
        cls = entry.factor_cls
        source = inspect.getsourcefile(cls)
        if source is None:  # pragma: no cover - factor classes have source
            continue
        out[cls.__module__] = Path(source)
    return out


__all__ = [
    "ALLOWED_INTERNAL_IMPORTS",
    "code_hash",
    "factor_module_labeled_file",
    "factor_source_files",
    "module_import_violations",
    "shared_set_labeled_files",
]
