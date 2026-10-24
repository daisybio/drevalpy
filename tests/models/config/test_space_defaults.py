"""Tests for space defaults."""

from __future__ import annotations

from typing import Any

import pytest

from drevalpy.models.config._space_defaults import split_space_and_options


class _Component:
    @staticmethod
    def get_hyperparameter_space() -> dict[str, Any]:
        return {"alpha": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5}, "not_a_spec": "opaque"}


@pytest.mark.parametrize(
    ("values", "alpha_default", "options"),
    [
        ({"alpha": 0.9}, 0.9, {}),
        ({"device": "cpu"}, 0.5, {"device": "cpu"}),
        ({"not_a_spec": 3}, 0.5, {"not_a_spec": 3}),
        ({}, 0.5, {}),
    ],
)
def test_values_move_defaults_or_become_options(values: dict, alpha_default: float, options: dict) -> None:
    space, got_options = split_space_and_options(_Component, values)
    assert set(space) == {"alpha", "not_a_spec"}
    assert space["alpha"]["default"] == alpha_default
    assert got_options == options


def test_the_declared_space_is_not_mutated() -> None:
    original = _Component.get_hyperparameter_space()
    split_space_and_options(_Component, {"alpha": 0.9})
    assert _Component.get_hyperparameter_space() == original
