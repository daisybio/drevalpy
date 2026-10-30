"""Tests for the generic ``Registry[T]``."""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.registry._registry import Registry


@pytest.fixture
def registry() -> Registry[int]:
    reg: Registry[int] = Registry("example")
    reg.register("one", 1, metadata={"description": "First", "tags": frozenset({"b", "a"})})
    reg.register("two", 2)
    return reg


def test_get_names_items_contains_and_len(registry: Registry[int]) -> None:
    assert registry.get("one") == 1
    assert registry.names() == ("one", "two")
    assert registry.items() == (("one", 1), ("two", 2))
    assert "one" in registry
    assert "three" not in registry
    assert len(registry) == 2


def test_names_keep_registration_order() -> None:
    reg: Registry[str] = Registry("example")
    for name in ("zeta", "alpha", "mid"):
        reg.register(name, name)
    assert reg.names() == ("zeta", "alpha", "mid")


def test_unknown_name_raises_and_lists_the_available_names(registry: Registry[int]) -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown example 'three'\. Available: one, two"):
        registry.get("three")
    with pytest.raises(UnknownNameError, match="three"):
        registry.metadata("three")
    with pytest.raises(UnknownNameError, match="<none>"):
        Registry[int]("empty").get("x")


def test_duplicate_raises_unless_replace(registry: Registry[int]) -> None:
    with pytest.raises(ValueError, match="'one' already registered"):
        registry.register("one", 10)
    assert registry.get("one") == 1

    assert registry.register("one", 10, replace=True) == 10
    assert registry.get("one") == 10
    assert registry.names() == ("one", "two")
    assert registry.metadata("one") == {}


@pytest.mark.parametrize("name", ["", "  "])
def test_empty_name_is_rejected(registry: Registry[int], name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(name, 3)


def test_metadata_is_a_read_only_copy(registry: Registry[int]) -> None:
    given = {"description": "Three"}
    registry.register("three", 3, metadata=given)
    given["description"] = "changed"

    stored = registry.metadata("three")
    assert stored == {"description": "Three"}
    with pytest.raises(TypeError):
        stored["description"] = "x"  # type: ignore[index]


def test_metadata_keys_must_be_strings(registry: Registry[int]) -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        registry.register("three", 3, metadata={1: "x"})  # type: ignore[dict-item]


def test_decorator_registers_and_returns_the_object() -> None:
    reg: Registry[type] = Registry("component")

    @reg.decorator("thing", description="A thing")
    class Thing:
        pass

    assert reg.get("thing") is Thing
    assert reg.metadata("thing") == {"description": "A thing"}
    with pytest.raises(ValueError, match="already registered"):
        reg.decorator("thing")(Thing)


def test_snapshot_and_restore_undo_registrations(registry: Registry[int]) -> None:
    snapshot = registry.snapshot()
    registry.register("three", 3)
    registry.register("one", 100, replace=True)

    registry.restore(snapshot)

    assert registry.items() == (("one", 1), ("two", 2))
    assert registry.metadata("one")["description"] == "First"


def test_restore_rejects_a_snapshot_of_another_kind(registry: Registry[int]) -> None:
    with pytest.raises(ValueError, match="cannot restore a other snapshot"):
        registry.restore(Registry[int]("other").snapshot())


def test_to_dataframe_has_a_column_per_metadata_key(registry: Registry[int]) -> None:
    registry.register("three", 3, metadata={"result_type": None})

    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Tags", "Result type"]
    assert frame["Name"].tolist() == ["one", "two", "three"]
    assert frame.loc[0, "Tags"] == "a, b"
    assert frame.loc[2, "Result type"] == ""
    assert "one" in repr(registry)
    assert "<table" in registry._repr_html_()


def test_to_dataframe_of_an_empty_registry_has_a_name_column() -> None:
    assert list(Registry[int]("empty").to_dataframe().columns) == ["Name"]
