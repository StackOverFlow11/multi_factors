"""Factor code identity: enumerated shared set + AST import allowlist (D3, §3.4/R3).

A stored factor value is only valid while the CODE that produced it is unchanged.
``code_hash(factor)`` = content hash of the factor's own module file PLUS a small,
ENUMERATED shared set of machinery every factor leans on:

    {factors.compute.minute.primitives, factors.ops.*, factors.base, factors.spec}

Design decision R3 (``tmp/design/factor_refactor_design_v3.md`` §3.4): we do NOT
build a transitive-import-graph walker — the §3.2 migration already collapsed the
factor closure into this fixed set. Instead TWO guards keep the set honest:

1. **AST import allowlist** (:func:`module_import_violations`): a factor module may
   import only the stdlib + numpy/pandas + an enumerated set of first-party leaves.
   A new first-party import that is not on the list makes the D3 allowlist test go
   red, FORCING a human to decide whether the new dependency belongs in the code
   hash — equivalent to an auto-closure, with a far smaller implementation. Dynamic
   imports (``importlib`` / ``__import__``) are refused for the same reason (they
   would hide a dependency from the static walk).
2. **A drift/mutation test**: changing the content of a shared-set member changes
   ``code_hash`` (proved in the D3 store-keys test).

WHAT IS DELIBERATELY NOT HASHED (documented limitation, out of the closing-14
scope): the PIT/schema-bearing ``data.clean`` modules go into the DATA fingerprint
(``factors.store.fingerprint``, the schema-version dimension), not here — folding
them into ``code_hash`` would over-invalidate on data-layer churn (design §3.4
"明确不做：闭包扩到全 data/"). ``data.availability_policy`` / ``factors.requires``
are pure declaration leaves (enum values / a metadata dataclass) whose content does
not change a computed factor value. And ``factors.compute.momentum`` is allowlisted
(candidates.py composes it for ReversalFactor) but not in the shared set: none of
the closing 14 factors is factor-on-factor, and momentum's rolling math lives in
``factors.ops`` (which IS in the shared set). When the first residual/composed
factor lands, add its composed module to the shared set (design §11).

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


def code_hash(factor: Factor | type[Factor]) -> str:
    """Full sha256 hex of the factor's module + the enumerated shared set.

    Deterministic and checkout-independent (content + module labels only, never
    paths). Two factors defined in the SAME module (e.g. value_ep / volatility_20
    both in candidates.py) share this hash — their store keys still differ via
    factor_id + params_hash, but a change to that shared module invalidates both,
    which is correct.
    """
    own = factor_module_labeled_file(factor)
    shared = list(shared_set_labeled_files())
    # If the factor's own module IS a shared-set member (none today), dedup so the
    # fold never sees a duplicate label.
    labels = {own[0]}
    items = [own] + [pair for pair in shared if pair[0] not in labels]
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
ALLOWED_INTERNAL_IMPORTS: frozenset[str] = frozenset(
    {
        # shared compute set (folded into code_hash)
        "factors.base",
        "factors.spec",
        "factors.requires",
        "factors.compute.minute.primitives",
        "factors.ops",
        # daily-factor composition (candidates.py -> momentum); allowlisted but
        # not in the shared set (documented in the module docstring)
        "factors.compute.momentum",
        # PIT / schema / declaration leaves (data layer): permitted, NOT in the
        # code hash — the schema modules feed the DATA fingerprint instead
        "data.availability_policy",
        "data.clean.intraday_schema",
        "data.clean.intraday_aggregate",
        "data.clean.schema",
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
