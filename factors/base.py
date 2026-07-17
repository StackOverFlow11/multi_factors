"""Factor port: compute a cross-sectional feature from market bars.

NO-LOOKAHEAD RULE (CLAUDE.md invariant #1, INV-001):
    A factor value at date t may use ONLY bars at dates <= t. It must never
    read future prices and must never receive forward returns. Per the fixed
    event order (see CONTRACTS.md), ``momentum_20[t] = close[t]/close[t-20] - 1``
    is acceptable because rebalancing happens AFTER the close of t and holding
    starts the next trading day.

MANDATORY SPEC (factor-evaluation contract, enforcement layer #1):
    every CONCRETE subclass MUST declare a :class:`~factors.spec.FactorSpec` in
    its own class body, as a plain class attribute or as a ``property``. Writing
    a factor therefore MEANS declaring its hypothesis (``expected_ic_sign``), its
    PIT contract and its inputs — there is no "add the spec later" path.

    "Concrete" is meant literally: a class that still has unimplemented abstract
    methods (a shared factor-family base that leaves ``compute`` to its children)
    is NOT yet a factor and is not asked for a spec — it would have to invent a
    bogus one. The requirement fires on the first class that could actually be
    instantiated and evaluated.
"""

from __future__ import annotations

import functools
import inspect
from abc import ABC, ABCMeta, abstractmethod

import pandas as pd

from factors.spec import FactorSpec

# Sentinel: ``None`` is a plausible (wrong) declared value, so it cannot mark
# "nothing was declared".
_UNDECLARED = object()

# Declaration forms a subclass may use for ``spec``:
#   * a FactorSpec instance   -> validated right here, at class definition;
#   * a property/cached_property -> needs an instance, so _FactorMeta.__call__
#     validates whatever it returns at construction time.
# An explicit allow-list on purpose: ``hasattr(type(x), "__get__")`` accepts a
# plain function, a staticmethod and a classmethod too (all are descriptors),
# which is exactly the "author forgot @property" bug this must reject.
_DEFERRED_SPEC_TYPES = (property, functools.cached_property)


def _validate_instance_spec(factor: object) -> FactorSpec:
    """Return ``factor.spec``, asserting it really is a :class:`FactorSpec`."""
    spec = factor.spec  # a property is evaluated here; errors propagate as-is
    if not isinstance(spec, FactorSpec):
        raise TypeError(
            f"{type(factor).__name__}.spec must be a FactorSpec instance; got "
            f"{type(spec).__name__}. Declare it as a class attribute, or as a "
            f"property returning a FactorSpec when the id depends on "
            f"constructor params (e.g. momentum_{{window}})."
        )
    return spec


def _validate_declared_spec(cls: type) -> None:
    """Class-definition-time check: this class body declares a usable ``spec``.

    Skipped while the class is still ABSTRACT (``compute`` not implemented yet):
    such a class is a shared family base, not a factor, and demanding a spec
    would force it to invent one. The moment a subclass becomes concrete, the
    requirement applies in full.
    """
    declared = cls.__dict__.get("spec", _UNDECLARED)
    if declared is _UNDECLARED:
        # ABCMeta.__new__ has already computed this by the time we run.
        if getattr(cls, "__abstractmethods__", frozenset()):
            return  # still abstract: not a factor yet, so not asked for a spec
        raise TypeError(
            f"Factor subclass {cls.__name__!r} must declare a FactorSpec as "
            f"'spec' in its own class body — a class attribute, or a property "
            f"returning one when the id depends on constructor params. Writing "
            f"a factor means declaring its hypothesis (expected_ic_sign), its "
            f"PIT contract and its input fields (see factors/spec.py)."
        )
    if isinstance(declared, FactorSpec):
        return  # static class attribute: already a validated spec
    if isinstance(declared, _DEFERRED_SPEC_TYPES):
        return  # needs an instance: _FactorMeta.__call__ validates the result
    if isinstance(declared, (staticmethod, classmethod)) or inspect.isfunction(declared):
        raise TypeError(
            f"Factor subclass {cls.__name__!r} declares 'spec' as a plain "
            f"{type(declared).__name__} — did you forget @property? Accessing "
            f"'spec' would then yield the {type(declared).__name__} itself, not "
            f"a FactorSpec, so the mandatory-spec contract would go unenforced "
            f"until something tried to read it. Declare 'spec' as a FactorSpec "
            f"class attribute, or decorate it with @property."
        )
    raise TypeError(
        f"Factor subclass {cls.__name__!r} declares spec={declared!r}, which is "
        f"neither a FactorSpec nor a property returning one."
    )


class _FactorMeta(ABCMeta):
    """Metaclass validating the ``spec`` at class definition AND at construction.

    WHY A METACLASS FOR THE DECLARATION CHECK (and not ``__init_subclass__``):
    the check must skip classes that are still abstract, and
    ``__abstractmethods__`` is computed by ``ABCMeta.__new__`` — which runs
    ``__init_subclass__`` on the way, i.e. TOO EARLY: the hook would read the
    PARENT's abstract set and mis-judge every intermediate base. ``__new__``
    after ``super().__new__`` is the first point where abstractness is known.

    WHY A METACLASS FOR THE INSTANCE CHECK (and not ``Factor.__init__``):
    concrete factors define their own ``__init__`` and do not call
    ``super().__init__()``, so a base-class ``__init__`` would simply never run —
    the check would be silently dead. And a ``property`` spec cannot be evaluated
    at class-definition time, since its ``factor_id`` depends on constructor
    params that do not exist yet. ``ABCMeta.__call__`` is the one hook that sees
    a FULLY built instance without touching a single factor constructor, so the
    spec is checked at construction: an invalid spec cannot even be instantiated,
    let alone reach an evaluator.
    """

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict,
        **kwargs: object,
    ) -> type:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        _validate_declared_spec(cls)
        return cls

    def __call__(cls, *args: object, **kwargs: object) -> object:
        instance = super().__call__(*args, **kwargs)
        _validate_instance_spec(instance)
        return instance


class Factor(ABC, metaclass=_FactorMeta):
    """Abstract cross-sectional factor.

    Subclasses set the class attribute ``name`` (used as the factor-panel column),
    declare a :class:`~factors.spec.FactorSpec` as ``spec``, and implement
    ``compute``.
    """

    name: str
    # Annotation only: it must NOT land in ``Factor.__dict__``, or every subclass
    # would inherit a "declared" spec and the check could never fire. (``Factor``
    # itself is abstract, so _validate_declared_spec skips it either way.)
    spec: FactorSpec

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the factor over a canonical market panel.

        Args:
            panel: MultiIndex(date, symbol) market panel with CORE_COLUMNS.

        Returns:
            A pd.Series indexed by MultiIndex(date, symbol), aligned to ``panel``,
            with ``.name == self.name``. Early dates with an insufficient window
            yield NaN. Computation must be per-symbol (no cross-symbol leakage)
            and must use only current/past bars (no lookahead).
        """
        raise NotImplementedError
