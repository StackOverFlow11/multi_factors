"""AST import allowlist for factor modules (refactor D3, commit 2, R3).

A factor module may import ONLY the stdlib + numpy/pandas + the enumerated
``ALLOWED_INTERNAL_IMPORTS``. This is the guard that keeps the code-hash shared
set honest WITHOUT a transitive-import-graph walker (design §3.4): a new
first-party import that is not on the list makes this test go red, forcing a
human to decide whether it belongs in the code hash.

Two halves: every SHIPPED factor module is clean (the real guard), and synthetic
bad modules are flagged (the teeth — the check is not vacuous).

Network-free.
"""

from __future__ import annotations

import factors.registry.builtin  # noqa: F401 - populate DEFAULT_REGISTRY
from factors.store.code_hash import factor_source_files, module_import_violations


# --------------------------------------------------------------------------- #
# the real guard: every registered factor module is clean
# --------------------------------------------------------------------------- #
def test_every_shipped_factor_module_passes_the_allowlist():
    sources = factor_source_files()
    assert sources, "no factor modules resolved from the registry"
    offenders: dict[str, list[str]] = {}
    for module_name, path in sources.items():
        violations = module_import_violations(path.read_text(), module_name=module_name)
        if violations:
            offenders[module_name] = violations
    assert offenders == {}, f"factor modules broke the import allowlist: {offenders}"


def test_the_registry_covers_the_full_factor_surface():
    # The 11 minute factors + the daily families are all reachable, so the guard
    # actually checks the modules whose code hash matters (and NOT the legacy
    # intraday_derived surface, which defines no factor class).
    sources = factor_source_files()
    assert any("candidates" in m for m in sources)
    assert any("minute.ridge_minute_return" in m for m in sources)
    assert not any(m.endswith("intraday_derived") for m in sources)


# --------------------------------------------------------------------------- #
# teeth: synthetic modules that MUST be flagged
# --------------------------------------------------------------------------- #
def test_allowed_imports_pass():
    src = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import os\n"
        "from collections.abc import Sequence\n"
        "from factors.base import Factor\n"
        "from factors.spec import FactorSpec, PanelField\n"
        "from factors.compute.minute.primitives import peak_mask_for_symbol\n"
        "from factors.ops import ts_std\n"
        "from data.availability_policy import MARKET_DAILY\n"
        "from data.clean.intraday_schema import DAILY_INDEX_NAMES\n"
    )
    assert module_import_violations(src) == []


def test_disallowed_first_party_import_is_flagged():
    v = module_import_violations("from qt.pipeline import run\n")
    assert any("disallowed import" in s for s in v)


def test_disallowed_first_party_bare_import_is_flagged():
    v = module_import_violations("import runtime.backtest\n")
    assert any("disallowed import" in s for s in v)


def test_new_third_party_import_is_flagged():
    v = module_import_violations("import scipy.stats\n")
    assert any("disallowed import" in s for s in v)


def test_importlib_dynamic_import_is_flagged():
    assert any("dynamic import" in s for s in module_import_violations("import importlib\n"))
    assert any(
        "dynamic import" in s
        for s in module_import_violations("from importlib import import_module\n")
    )


def test_dunder_import_call_is_flagged():
    v = module_import_violations("mod = __import__('os')\n")
    assert any("dynamic import" in s for s in v)


def test_relative_import_is_flagged():
    v = module_import_violations("from . import sibling\n")
    assert any("relative import" in s for s in v)
