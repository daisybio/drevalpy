"""Shared assertions for the ``test_init.py`` package-surface tests.

Every re-export barrel in ``drevalpy`` is pinned the same four ways: ``__all__``
is sorted and duplicate-free, it matches a surface recorded by hand in the test
file, every promised name resolves, and each re-export ``is`` the very object its
defining module holds. Seven ``test_init.py`` files spelled that idiom out one
assertion at a time; it lives here once instead, and each barrel test contributes
only its data plus whatever is genuinely specific to it.

What deliberately stays in the barrel test files is the *recorded surface* - the
``name -> defining module`` table. Deriving it from ``__all__`` at runtime would
make :meth:`DeclaredSurface.test_all_matches_the_recorded_surface` unfalsifiable,
and the reason for recording it by hand is that a reviewer reading a diff sees
exactly which public name changed.

Origins belong against the module the barrel does *not* import from - typically
the leaf that defines the object, sometimes a second re-export of it. Comparing a
re-export with the module it was imported from directly cannot fail.

Each assertion reports every offending name rather than stopping at the first,
which is what the parametrised originals bought with one test item per name.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import ClassVar

#: Tells an absent attribute apart from one that is legitimately ``None``.
_MISSING = object()


def _resolve(module: ModuleType, name: str) -> object:
    """Return the attribute, or :data:`_MISSING` when the module does not have it."""
    return getattr(module, name, _MISSING)


def _is_defining_object(barrel: ModuleType, name: str, origin: str) -> bool:
    """Return whether ``barrel.name`` is the object ``origin`` holds under that name."""
    exported = _resolve(barrel, name)
    return exported is not _MISSING and exported is _resolve(importlib.import_module(origin), name)


class ReExportSurface:
    """Pin a barrel against the surface its subclass records.

    Subclasses set :attr:`barrel` and whichever tables apply. This base makes no
    assumption that the barrel publishes ``__all__`` - the top-level ``drevalpy``
    package marks its re-exports with ``import x as x`` instead.
    """

    #: The already-imported package whose surface is under test.
    barrel: ClassVar[ModuleType]

    #: ``exported name -> import path of the module that defines it``.
    origins: ClassVar[Mapping[str, str]] = {}

    #: Surface names with no recorded origin, either because the barrel defines
    #: them itself or because the defining module is deliberately not pinned.
    unpinned_names: ClassVar[tuple[str, ...]] = ()

    #: Surface names the barrel promises are callable.
    callable_names: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def recorded_surface(cls) -> list[str]:
        """Return the recorded names: those with an origin plus the unpinned ones."""
        return sorted({*cls.origins, *cls.unpinned_names})

    @classmethod
    def _names_that_must_resolve(cls) -> list[str]:
        return cls.recorded_surface()

    def test_every_promised_name_resolves(self) -> None:
        unresolved = [name for name in self._names_that_must_resolve() if _resolve(self.barrel, name) is _MISSING]
        assert not unresolved, f"{self.barrel.__name__} promises names that do not resolve: {unresolved}"

    def test_export_is_the_object_its_defining_module_holds(self) -> None:
        drifted = [
            name for name, origin in sorted(self.origins.items()) if not _is_defining_object(self.barrel, name, origin)
        ]
        assert not drifted, f"{self.barrel.__name__} re-exports that are not the defining object: {drifted}"

    def test_promised_callables_are_callable(self) -> None:
        uncallable = [name for name in self.callable_names if not callable(_resolve(self.barrel, name))]
        assert not uncallable, f"{self.barrel.__name__} promises callables that are not callable: {uncallable}"


class DeclaredSurface(ReExportSurface):
    """A barrel that publishes ``__all__``, so the list itself is pinned too."""

    @classmethod
    def _names_that_must_resolve(cls) -> list[str]:
        """Include ``__all__``: an entry left behind after its symbol moved is a broken import."""
        return sorted({*super()._names_that_must_resolve(), *cls.barrel.__all__})

    def test_all_is_sorted_and_unique(self) -> None:
        declared = list(self.barrel.__all__)
        assert declared == sorted(set(declared))

    def test_all_matches_the_recorded_surface(self) -> None:
        """Catch drift in both directions: a stale entry and an unrecorded addition."""
        assert sorted(self.barrel.__all__) == self.recorded_surface()


class SingletonFacadeSurface(DeclaredSurface):
    """A barrel whose module-level ``list``/``get``/``metadata`` wrap one registry singleton.

    The forwarding check is read-only: it never calls ``register``, so the
    process-global registry is left exactly as it was found.
    """

    #: The process-global registry the facade functions delegate to.
    singleton: ClassVar[object]

    #: Singleton attribute listing the registered keys - ``modes`` or ``names``.
    keys_attribute: ClassVar[str]

    def test_facade_reads_forward_to_the_singleton(self) -> None:
        """A facade that stopped delegating would silently serve an empty registry."""
        keys = self.barrel.list()
        assert keys == getattr(self.singleton, self.keys_attribute)
        assert keys, "the built-in components are registered before every test"
        for key in keys:
            assert self.barrel.get(key) is self.singleton.get(key)
            assert self.barrel.metadata(key) == self.singleton.get_metadata(key)
