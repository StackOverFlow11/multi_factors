"""Factor registry: THE one name -> class dispatch (design v3.2 §3.1, D1).

Red line #4: the mapping from a config factor name to a Factor class exists in
exactly ONE place — here. ``qt.pipeline._build_factors``'s if/elif chain is
retired; a second dispatch chain must never grow back.

Five-name / three-timepoint consistency (design §6 pit #1 — the five names are
the class name, ``Factor.name``, ``spec.factor_id``, the config name, and the
factor-panel column, which IS ``factor.name`` by ``compute``'s rename):

  1. REGISTRATION time (class-definition time for the explicit builtin list):
     the class must be a concrete Factor whose declared spec passes the
     metaclass validation (re-run here — red line #1: ``register`` reuses the
     existing FactorSpec checks, it does not weaken or reimplement them), and
     a STATIC class-attribute spec must agree with the class-level ``name``.
     A property spec cannot be evaluated without an instance (pit #2 — window
     parameterized ids), so its name check is DEFERRED to build time; do not
     "fix" that by demanding class attributes.
  2. BUILD time: the config name must resolve in the registry; the built
     instance's ``name`` must equal the config name (the pre-existing
     name/params-mismatch semantics of ``_build_factors``, kept verbatim);
     the instance spec's ``factor_id`` must equal ``factor.name``.
  3. CONFIG-PARSE time: ``qt.config`` calls :func:`resolve` on every enabled
     factor entry, so an unknown name fails at ``validate-config`` instead of
     minutes into a run.

Layering (red line #10): this module imports ``factors.*`` and the pure
declaration leaf ``data.availability_policy`` only — never qt, never feeds,
never an orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from data import availability_policy as _policy
from factors.base import Factor, _validate_declared_spec
from factors.requires import PanelField
from factors.spec import FactorSpec

# A builder maps (config_name, params) -> a Factor instance. It performs the
# SAME params coercion the retired if/elif chain performed for its family
# (e.g. ``int(params.get("window", 20))``) — behavior-preserving by design.
Builder = Callable[[str, Mapping[str, object]], Factor]


@dataclass(frozen=True)
class RegistryEntry:
    """One dispatch rule: an exact config name or a name-family prefix."""

    key: str
    kind: str  # "exact" | "prefix"
    factor_cls: type[Factor]
    builder: Builder


class FactorRegistry:
    """Name -> class dispatch with the three-timepoint naming checks.

    A plain class (not module state) so tests can exercise collision /
    validation behavior on a fresh instance without polluting the default
    registry that ``factors.registry.builtin`` populates.
    """

    def __init__(self) -> None:
        self._exact: dict[str, RegistryEntry] = {}
        self._prefixes: list[RegistryEntry] = []  # kept in registration order

    # -- registration (timepoint 1) ---------------------------------------

    def register(
        self,
        cls: type[Factor],
        *,
        exact: Iterable[str] = (),
        prefix: str | None = None,
        builder: Builder,
    ) -> type[Factor]:
        """Register ``cls`` under exact name(s) OR a name-family prefix.

        Duplicate keys and ambiguous (mutually-prefixing) prefixes are
        readable errors — a silent overwrite would be a second dispatch truth.
        """
        exact_names = tuple(exact)
        if bool(exact_names) == (prefix is not None):
            raise ValueError(
                f"register({cls!r}) needs EITHER exact names or a prefix "
                f"(got exact={exact_names!r}, prefix={prefix!r})."
            )
        self._validate_class(cls)
        entry_names = exact_names if exact_names else (prefix,)
        for name in entry_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"register({cls.__name__}) got a blank/non-string key "
                    f"{name!r}; registry keys must be non-empty strings."
                )
        if exact_names:
            for name in exact_names:
                if name in self._exact:
                    raise ValueError(
                        f"duplicate factor registration for exact name "
                        f"{name!r}: already registered to "
                        f"{self._exact[name].factor_cls.__name__}, refusing "
                        f"{cls.__name__}. The name -> class mapping must have "
                        f"exactly one truth (red line #4)."
                    )
                self._exact[name] = RegistryEntry(name, "exact", cls, builder)
            return cls
        if prefix is None:  # unreachable (XOR-guarded above); no bare assert (-O safe)
            raise ValueError("register() reached the prefix path without a prefix.")
        for other in self._prefixes:
            if other.key == prefix:
                raise ValueError(
                    f"duplicate factor registration for prefix {prefix!r}: "
                    f"already registered to {other.factor_cls.__name__}, "
                    f"refusing {cls.__name__}."
                )
            if other.key.startswith(prefix) or prefix.startswith(other.key):
                raise ValueError(
                    f"ambiguous factor prefixes: {prefix!r} ({cls.__name__}) "
                    f"vs already-registered {other.key!r} "
                    f"({other.factor_cls.__name__}) — one is a prefix of the "
                    f"other, so dispatch would depend on registration order. "
                    f"Pick non-overlapping family prefixes."
                )
        self._prefixes.append(RegistryEntry(prefix, "prefix", cls, builder))
        return cls

    @staticmethod
    def _validate_class(cls: object) -> None:
        """Timepoint-1 checks; first step REUSES the existing spec validation."""
        if not isinstance(cls, type) or not issubclass(cls, Factor):
            raise TypeError(
                f"register() needs a Factor subclass; got {cls!r}."
            )
        if getattr(cls, "__abstractmethods__", frozenset()):
            raise TypeError(
                f"register({cls.__name__}) refused: the class is still "
                f"abstract (a factor-family base, not a factor)."
            )
        # Red line #1: reuse the mandatory-spec validation the metaclass runs
        # at class definition (re-running is idempotent and keeps register()
        # from ever accepting less than the base class demands).
        _validate_declared_spec(cls)
        # A CLASS-level name is optional: exact-name classes (Financial/Value)
        # derive ``self.name`` purely from the constructor field, so their
        # naming consistency is carried by the build-time checks. When a
        # class-level default IS declared it must be sane, and a STATIC
        # class-attribute spec must agree with it at definition time.
        name = getattr(cls, "name", None)
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise TypeError(
                f"register({cls.__name__}) refused: the ``name`` class "
                f"attribute must be a non-empty string when declared (it is "
                f"the factor-panel column default); got {name!r}."
            )
        declared = cls.__dict__.get("spec")
        if isinstance(declared, FactorSpec):
            if name is None:
                raise ValueError(
                    f"register({cls.__name__}) refused: a class-attribute "
                    f"spec (factor_id={declared.factor_id!r}) needs a class "
                    f"``name`` attribute to agree with — declare one."
                )
            if declared.factor_id != name:
                raise ValueError(
                    f"register({cls.__name__}) refused: class attribute spec "
                    f"declares factor_id={declared.factor_id!r} but the class "
                    f"name attribute is {name!r} — the five-name consistency "
                    f"starts at definition time. (Property specs are checked "
                    f"at build time instead, once constructor params exist.)"
                )

    # -- lookup / build (timepoints 2 and 3) ------------------------------

    def resolve(self, name: str) -> RegistryEntry:
        """Find the entry for a config factor name; unknown -> readable error.

        Exact names win over prefixes (same precedence the retired chain had:
        membership tests came before ``startswith`` families); prefixes are
        unambiguous by the registration-time mutual-prefix guard.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"factor name must be a non-empty string; got {name!r}."
            )
        entry = self._exact.get(name)
        if entry is not None:
            return entry
        for candidate in self._prefixes:
            if name.startswith(candidate.key):
                return candidate
        exacts = sorted(self._exact)
        prefixes = [f"{e.key}*" for e in self._prefixes]
        raise ValueError(
            f"Unknown factor {name!r}; the factor registry knows the name "
            f"families {prefixes} and the exact names {exacts}. New factors "
            f"register in factors/registry/builtin.py (the ONE name -> class "
            f"mapping)."
        )

    def build(self, name: str, params: Mapping[str, object] | None = None) -> Factor:
        """Instantiate the factor for a config (name, params) pair.

        Runs the timepoint-2 naming checks: config-name/instance-name equality
        (the ``_build_factors`` name/params-mismatch semantics, kept verbatim)
        and instance-name/spec-id equality (property specs get their deferred
        check here — pit #2).
        """
        entry = self.resolve(name)
        factor = entry.builder(name, dict(params or {}))
        if not isinstance(factor, entry.factor_cls):
            raise TypeError(
                f"registry builder for {entry.key!r} returned a "
                f"{type(factor).__name__}, not the registered "
                f"{entry.factor_cls.__name__} — the builder and the "
                f"registration disagree about the class."
            )
        # window-named factors derive their name from params: a spec named
        # reversal_5 with params.window=10 would silently mislabel the column.
        if factor.name != name:
            raise ValueError(
                f"Factor name/params mismatch: config names {name!r} but the "
                f"params resolve to {factor.name!r} (window-named factors must "
                "agree with params.window)."
            )
        spec = factor.spec
        if spec.factor_id != factor.name:
            raise ValueError(
                f"Factor name/spec mismatch for {type(factor).__name__}: "
                f"instance name {factor.name!r} but spec.factor_id "
                f"{spec.factor_id!r} — the panel column and the evaluation "
                f"identity would disagree."
            )
        return factor

    def requirements(
        self,
        names: Iterable[str],
        params_by_name: Mapping[str, Mapping[str, object]] | None = None,
    ) -> tuple[PanelField, ...]:
        """Aggregate the deduplicated endpoint requirements of ``names``.

        Builds each factor (so every naming check applies — a window-named
        factor with a non-default window needs its params here exactly like
        in ``build``) and unions ``spec.requires`` preserving first-seen
        order. This is the D4 materializer's one-stop shopping list; in D1 it
        only needs to be callable and correct.
        """
        params_by_name = params_by_name or {}
        seen: dict[PanelField, None] = {}
        for name in names:
            factor = self.build(name, params_by_name.get(name))
            for requirement in factor.spec.requires or ():
                seen.setdefault(requirement, None)
        return tuple(seen)


#: The process-wide registry ``factors/registry/builtin.py`` populates.
DEFAULT_REGISTRY = FactorRegistry()


def register(
    cls: type[Factor],
    *,
    exact: Iterable[str] = (),
    prefix: str | None = None,
    builder: Builder,
) -> type[Factor]:
    """Register into the default registry (see :class:`FactorRegistry`)."""
    return DEFAULT_REGISTRY.register(cls, exact=exact, prefix=prefix, builder=builder)


def resolve(name: str) -> RegistryEntry:
    """Resolve a config factor name in the default registry."""
    return DEFAULT_REGISTRY.resolve(name)


def build(name: str, params: Mapping[str, object] | None = None) -> Factor:
    """Build a factor from the default registry."""
    return DEFAULT_REGISTRY.build(name, params)


def requirements(
    names: Iterable[str],
    params_by_name: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[PanelField, ...]:
    """Aggregate deduplicated requirements from the default registry."""
    return DEFAULT_REGISTRY.requirements(names, params_by_name)


def require_legal_pairing(view: object, basis: object):
    """Forwarding call point for the view x return-basis legality check (D1).

    The single source stays ``data.availability_policy.require_legal_pairing``
    (design §1.4 mechanism 1); the registry only guarantees the check is
    CALLABLE from the factor layer so the D4 materializer / store wiring has
    its hook. Deep wiring (view in store keys, per-run enforcement) is D4 —
    deliberately NOT done here.
    """
    return _policy.require_legal_pairing(view, basis)


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
