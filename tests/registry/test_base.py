"""Tests for the shared :class:`~drevalpy.registry._base.Registry` ABC.

Every test operates on a locally constructed registry, never on a module
singleton, so nothing here can leak an entry into the global stores that the
autouse ``_ensure_registries_populated`` fixture repopulates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from drevalpy.registry._base import Registry


class _ExampleRegistry(Registry):
    """Minimal concrete registry exposing an ``add`` seam for tests."""

    def __init__(self) -> None:
        super().__init__("example", "Example component", "examples")

    def add(self, name: str, cls: type[Any]) -> None:
        """Insert *cls* under *name* without any registration validation."""
        self._store[name] = cls

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return {
            "registry": self._display_name,
            "name": name,
            "description": getattr(cls, "description", ""),
            "tags": getattr(cls, "tags", frozenset()),
        }


class _Alpha:
    description = "first component"
    tags = frozenset({"baseline"})


class _Beta:
    description = "second component"
    tags = frozenset({"omics", "baseline"})


@pytest.fixture
def registry() -> _ExampleRegistry:
    populated = _ExampleRegistry()
    populated.add("alpha", _Alpha)
    populated.add("beta", _Beta)
    return populated


def test_registry_is_abstract() -> None:
    with pytest.raises(TypeError, match="_component_metadata"):
        Registry("example", "Example", "examples")  # type: ignore[abstract]


def test_constructor_stores_identity() -> None:
    registry = _ExampleRegistry()

    assert registry._registry_id == "example"
    assert registry._label == "Example component"
    assert registry._display_name == "examples"
    assert registry.list_names() == []


def test_get_returns_registered_class(registry: _ExampleRegistry) -> None:
    assert registry.get("alpha") is _Alpha


def test_get_unknown_name_lists_available(registry: _ExampleRegistry) -> None:
    with pytest.raises(ValueError, match=r"Unknown Example component: 'missing'\. Available: \['alpha', 'beta'\]"):
        registry.get("missing")


def test_list_names_reflects_insertion_order(registry: _ExampleRegistry) -> None:
    assert registry.list_names() == ["alpha", "beta"]


def test_get_metadata_delegates_to_component_metadata(registry: _ExampleRegistry) -> None:
    assert registry.get_metadata("alpha") == {
        "registry": "examples",
        "name": "alpha",
        "description": "first component",
        "tags": frozenset({"baseline"}),
    }


def test_list_metadata_returns_every_component(registry: _ExampleRegistry) -> None:
    rows = registry.list_metadata()

    assert [row["name"] for row in rows] == ["alpha", "beta"]


def test_list_metadata_filters_by_tag(registry: _ExampleRegistry) -> None:
    rows = registry.list_metadata(tag="omics")

    assert [row["name"] for row in rows] == ["beta"]


def test_list_metadata_strips_whitespace_from_tag(registry: _ExampleRegistry) -> None:
    rows = registry.list_metadata(tag="  omics  ")

    assert [row["name"] for row in rows] == ["beta"]


def test_clear_empties_the_store(registry: _ExampleRegistry) -> None:
    registry.clear()

    assert registry.list_names() == []


def test_retain_only_drops_unlisted_names(registry: _ExampleRegistry) -> None:
    registry.retain_only(frozenset({"beta"}))

    assert registry.list_names() == ["beta"]


def test_retain_only_keeps_names_it_does_not_know(registry: _ExampleRegistry) -> None:
    registry.retain_only(frozenset({"alpha", "beta", "never-registered"}))

    assert registry.list_names() == ["alpha", "beta"]


def test_to_dataframe_renders_name_description_and_sorted_tags(registry: _ExampleRegistry) -> None:
    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Tags"]
    pd.testing.assert_series_equal(
        frame["Tags"],
        pd.Series(["baseline", "baseline, omics"], name="Tags"),
    )


def test_to_dataframe_of_empty_registry_has_no_rows() -> None:
    assert _ExampleRegistry().to_dataframe().empty


def test_repr_is_the_dataframe_without_the_index(registry: _ExampleRegistry) -> None:
    rendered = repr(registry)

    assert "alpha" in rendered
    assert not rendered.startswith("0")


def test_repr_html_emits_a_table(registry: _ExampleRegistry) -> None:
    assert "<table" in registry._repr_html_()
