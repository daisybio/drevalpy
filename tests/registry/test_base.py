"""Tests for the shared :class:`~drevalpy.registry._base.Registry` ABC (on locally constructed registries only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.registry._base import Registry, TableReprMixin
from drevalpy.registry.dataset import dataset_registry
from drevalpy.registry.splitter import splitter_registry
from drevalpy.registry.visualization import visualization_registry


class _ExampleRegistry(Registry):
    def __init__(self) -> None:
        super().__init__("example", "Example component", "examples")

    def add(self, name: str, cls: type[Any]) -> None:
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


def test_get_and_list_names(registry: _ExampleRegistry) -> None:
    assert _ExampleRegistry().list_names() == []
    assert registry.list_names() == ["alpha", "beta"]
    assert registry.get("alpha") is _Alpha
    with pytest.raises(
        UnknownNameError, match=r"Unknown Example component: 'missing'\. Available: \['alpha', 'beta'\]"
    ):
        registry.get("missing")


def test_metadata_listing_and_tag_filter(registry: _ExampleRegistry) -> None:
    assert registry.get_metadata("alpha") == {
        "registry": "examples",
        "name": "alpha",
        "description": "first component",
        "tags": frozenset({"baseline"}),
    }
    assert [row["name"] for row in registry.list_metadata()] == ["alpha", "beta"]
    assert [row["name"] for row in registry.list_metadata(tag="  omics  ")] == ["beta"]


def test_clear_and_retain_only(registry: _ExampleRegistry) -> None:
    registry.retain_only(frozenset({"beta", "never-registered"}))
    assert registry.list_names() == ["beta"]

    registry.clear()
    assert registry.list_names() == []


def test_dataframe_repr_and_html(registry: _ExampleRegistry) -> None:
    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Tags"]
    pd.testing.assert_series_equal(frame["Tags"], pd.Series(["baseline", "baseline, omics"], name="Tags"))
    assert _ExampleRegistry().to_dataframe().empty
    assert "alpha" in repr(registry)
    assert not repr(registry).startswith("0")
    assert "<table" in registry._repr_html_()


@pytest.mark.parametrize(
    "registry",
    [dataset_registry, splitter_registry, visualization_registry],
    ids=["dataset", "splitter", "visualization"],
)
def test_registries_outside_the_component_hierarchy_share_the_table_repr(registry: TableReprMixin) -> None:
    assert isinstance(registry, TableReprMixin)
    assert "<table" in registry._repr_html_()
