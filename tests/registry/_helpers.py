"""Shared registry-state helpers for the ``tests/registry`` tree.

The component registries are process-global singletons and ``tests/conftest.py``
repopulates the built-ins before every test, so a test that registers a name must
remove it again. ``register_builtin_components`` only *adds* - it does not evict -
so restoring state means clearing first and repopulating second. Doing only the
second half leaves the test's own names behind, which surfaces much later as a
duplicate-name ``ValueError`` in an unrelated test.
"""

from __future__ import annotations

from collections.abc import Iterator

from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry


def clear_component_registries() -> None:
    """Empty the cell-line featurizer, drug featurizer and predictor registries."""
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()


def restore_component_registries() -> None:
    """Drop every registration and re-register only the built-in components."""
    clear_component_registries()
    register_builtin_components()


def isolated_component_registries() -> Iterator[None]:
    """Yield with empty component registries, restoring the built-ins afterwards.

    Intended to back a module-local ``@pytest.fixture(autouse=True)``.
    """
    clear_component_registries()
    yield
    restore_component_registries()
