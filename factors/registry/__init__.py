"""Factor registry package (D1): the single name -> class dispatch.

Importing this package loads the explicit builtin registration list
(``factors.registry.builtin`` — design §6 pit #3: an explicit import list,
never pkgutil auto-discovery), so ``resolve`` / ``build`` are ready to use
from any access path.
"""

from factors.registry.registry import (
    DEFAULT_REGISTRY,
    Builder,
    FactorRegistry,
    RegistryEntry,
    build,
    register,
    requirements,
    require_legal_pairing,
    resolve,
)
from factors.registry import builtin as _builtin  # noqa: F401  (registers the builtins)

__all__ = [
    "Builder",
    "DEFAULT_REGISTRY",
    "FactorRegistry",
    "RegistryEntry",
    "build",
    "register",
    "requirements",
    "require_legal_pairing",
    "resolve",
]
