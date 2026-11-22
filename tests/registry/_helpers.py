"""Shared registry-state helpers for tests that need empty or built-in-only component registries.

``tests/conftest.py`` restores every registry after each test, so a test may register
names freely. These helpers are for tests that need a known starting state instead:
:func:`empty_component_registries` empties the component registries (call it from a
module-local autouse fixture; ``tests/conftest.py`` restores them afterwards), and
:func:`restore_builtin_components` puts back exactly the built-ins. The built-in state is
captured when this module is first imported, which ``tests/conftest.py`` does right after
``import drevalpy.registry`` and before any test module registers anything.
"""

from __future__ import annotations

from typing import Any

from drevalpy.registry._registry import Registry, Snapshot
from drevalpy.registry.featurizer import featurizer_registries
from drevalpy.registry.predictor import predictor_registry

_COMPONENT_REGISTRIES: tuple[Registry[Any], ...] = (*featurizer_registries.values(), predictor_registry)
_BUILTIN_SNAPSHOTS: tuple[Snapshot[Any], ...] = tuple(registry.snapshot() for registry in _COMPONENT_REGISTRIES)


def empty_component_registries() -> None:
    """Empty the featurizer and predictor registries."""
    for registry in _COMPONENT_REGISTRIES:
        registry.restore(Snapshot(registry.kind, ()))


def restore_builtin_components() -> None:
    """Put the component registries back to the built-ins, dropping every other registration."""
    for registry, snapshot in zip(_COMPONENT_REGISTRIES, _BUILTIN_SNAPSHOTS, strict=True):
        registry.restore(snapshot)
