"""Tests for the naive predictors' shared state mixin (key naming, vector/matrix restore, fitted flag)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.naive._state_mixin import MeanEffectsStateMixin


class _Host:
    def __init__(self, hyperparameters: dict | None = None) -> None:
        """Store hyperparameters like the predictor base."""
        self.hyperparameters = hyperparameters


class _MeanOnly(MeanEffectsStateMixin, _Host):
    pass


class _OneVector(MeanEffectsStateMixin, _Host):
    state_effects: ClassVar[tuple[str, ...]] = ("effects",)


class _ThreeVectors(MeanEffectsStateMixin, _Host):
    state_effects: ClassVar[tuple[str, ...]] = ("tissue_effects", "sample_effects", "drug_effects")


class _OneMatrix(MeanEffectsStateMixin, _Host):
    state_effects: ClassVar[tuple[str, ...]] = ("effects",)
    state_effects_ndim: ClassVar[int] = 2


def test_initialisation_forwards_hyperparameters_and_starts_unfitted() -> None:
    model = _ThreeVectors({"a": 1})

    assert model.hyperparameters == {"a": 1}
    assert (model._tissue_effects, model._sample_effects, model._drug_effects) == (None, None, None)
    assert not model.is_fitted()
    assert not _MeanOnly().is_fitted()
    assert _OneVector().get_state() == {}


def test_is_fitted_needs_the_mean_and_every_effect_array() -> None:
    model = _OneVector()
    model._effects = np.zeros(2)
    assert not model.is_fitted()

    three = _ThreeVectors()
    three._dataset_mean = 0.5
    three._tissue_effects = np.zeros(1)
    three._sample_effects = np.zeros(1)
    assert not three.is_fitted()
    assert three.get_state() == {}
    three._drug_effects = np.zeros(1)
    assert three.is_fitted()
    assert list(three.get_state()) == ["dataset_mean", "tissue_effects", "sample_effects", "drug_effects"]

    mean_only = _MeanOnly()
    mean_only._dataset_mean = 0.25
    assert mean_only.is_fitted()
    assert mean_only.get_state() == {"dataset_mean": 0.25}


def test_an_empty_effect_array_still_counts_as_fitted_and_serialises_as_a_list() -> None:
    model = _OneVector()
    model._dataset_mean = 0.5
    model._effects = np.empty((0,))
    assert model.is_fitted()

    model._effects = np.array([1.0, 2.0])
    assert model.get_state()["effects"] == [1.0, 2.0]


def test_set_state_round_trips_vectors_and_matrices() -> None:
    three = _ThreeVectors()
    three._dataset_mean = -1.5
    three._tissue_effects = np.array([0.1])
    three._sample_effects = np.array([0.2, 0.3])
    three._drug_effects = np.array([0.4, 0.5, 0.6])
    restored = _ThreeVectors()
    restored.set_state(three.get_state())

    assert restored.is_fitted()
    np.testing.assert_allclose(restored._drug_effects, [0.4, 0.5, 0.6])

    matrix = _OneMatrix()
    matrix._dataset_mean = 0.0
    matrix._effects = np.array([[1.0, 2.0], [3.0, 4.0]])
    restored_matrix = _OneMatrix()
    restored_matrix.set_state(matrix.get_state())

    assert restored_matrix._effects.shape == (2, 2)
    np.testing.assert_allclose(restored_matrix._effects, matrix._effects)


def test_set_state_normalises_payload_shape_and_dtype() -> None:
    matrix = _OneMatrix()
    matrix.set_state({"dataset_mean": 0.0, "effects": [1.0, 2.0]})
    vector = _OneVector()
    vector.set_state({"dataset_mean": 0.0, "effects": [[1], [2]]})
    mean_only = _MeanOnly()
    mean_only.set_state({"dataset_mean": "0.75"})

    assert matrix._effects.shape == (2, 1)
    assert vector._effects.shape == (2,)
    assert vector._effects.dtype == np.float64
    assert mean_only._dataset_mean == 0.75


def test_set_state_with_partial_or_empty_payload_keeps_existing_values() -> None:
    model = _OneVector()
    model.set_state({})
    assert not model.is_fitted()

    model._dataset_mean = 0.5
    model._effects = np.array([1.0])
    model.set_state({"dataset_mean": 2.0})

    assert model._dataset_mean == 2.0
    np.testing.assert_allclose(model._effects, [1.0])
