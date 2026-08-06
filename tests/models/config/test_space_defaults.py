"""Tests for drevalpy.models.config._space_defaults."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config._space_defaults import split_space_and_options


class _Component:
    """Stand-in component declaring a mixed hyperparameter space."""

    @staticmethod
    def get_hyperparameter_space() -> dict[str, Any]:
        """Declare one tunable entry and one non-mapping entry.

        :returns: The declared space.
        """
        return {
            "alpha": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
            "not_a_spec": "opaque",
        }


def test_declared_tunable_moves_its_default() -> None:
    space, options = split_space_and_options(_Component, {"alpha": 0.9})
    assert space["alpha"]["default"] == 0.9
    assert options == {}


def test_undeclared_value_becomes_an_option() -> None:
    space, options = split_space_and_options(_Component, {"device": "cpu"})
    assert options == {"device": "cpu"}
    assert space["alpha"]["default"] == 0.5


def test_non_mapping_spec_is_treated_as_an_option_target() -> None:
    """A declared key whose spec is not a mapping has no ``default`` to move."""
    _, options = split_space_and_options(_Component, {"not_a_spec": 3})
    assert options == {"not_a_spec": 3}


def test_the_full_declared_space_is_returned() -> None:
    """The result records every declared entry, not only the touched ones."""
    space, _ = split_space_and_options(_Component, {"alpha": 0.1})
    assert set(space) == {"alpha", "not_a_spec"}


def test_the_declared_space_is_not_mutated() -> None:
    original = _Component.get_hyperparameter_space()
    split_space_and_options(_Component, {"alpha": 0.9})
    assert _Component.get_hyperparameter_space() == original


def test_empty_values_leave_the_defaults_alone() -> None:
    space, options = split_space_and_options(_Component, {})
    assert space["alpha"]["default"] == 0.5
    assert options == {}
