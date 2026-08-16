"""Tests for :mod:`drevalpy.components.predictors.naive._state_mixin`.

Mirrors the private module with the underscore stripped. The five naive
predictors get their persistence from this mixin, and each asserts its own round
trip; what is pinned here is the mixin's own contract - key naming, the
vector/matrix restore switch and the ``is_fitted`` transitions - on throwaway
subclasses, so no predictor lifecycle is involved.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.naive._state_mixin import MeanEffectsStateMixin


class _Host:
    """Stand-in for the predictor base the mixin cooperates with."""

    def __init__(self, hyperparameters: dict | None = None) -> None:
        self.hyperparameters = hyperparameters


class _MeanOnly(MeanEffectsStateMixin, _Host):
    """Holds nothing but the dataset mean, like ``NaiveMeanPredictor``."""


class _OneVector(MeanEffectsStateMixin, _Host):
    """One 1-D effect array, like the per-entity and per-tissue means."""

    state_effects: ClassVar[tuple[str, ...]] = ("effects",)


class _ThreeVectors(MeanEffectsStateMixin, _Host):
    """Three 1-D effect arrays, like ``NaiveMeanEffectsPredictor``."""

    state_effects: ClassVar[tuple[str, ...]] = ("tissue_effects", "cell_line_effects", "drug_effects")


class _OneMatrix(MeanEffectsStateMixin, _Host):
    """One 2-D effect array, like ``NaiveTissueDrugMeanPredictor``."""

    state_effects: ClassVar[tuple[str, ...]] = ("effects",)
    state_effects_ndim: ClassVar[int] = 2


class TestInitialisation:
    def test_forwards_hyperparameters_to_the_host(self):
        assert _OneVector({"a": 1}).hyperparameters == {"a": 1}

    def test_starts_unfitted(self):
        assert _OneVector().is_fitted() is False

    def test_declares_every_effect_attribute_as_none(self):
        model = _ThreeVectors()

        assert (model._tissue_effects, model._cell_line_effects, model._drug_effects) == (None, None, None)

    def test_an_effect_free_host_is_unfitted_until_the_mean_is_set(self):
        assert _MeanOnly().is_fitted() is False


class TestIsFitted:
    def test_needs_the_mean(self):
        model = _OneVector()
        model._effects = np.zeros(2)

        assert model.is_fitted() is False

    def test_needs_every_effect_array(self):
        model = _ThreeVectors()
        model._dataset_mean = 0.5
        model._tissue_effects = np.zeros(1)
        model._cell_line_effects = np.zeros(1)

        assert model.is_fitted() is False

        model._drug_effects = np.zeros(1)

        assert model.is_fitted() is True

    def test_the_mean_alone_fits_an_effect_free_host(self):
        model = _MeanOnly()
        model._dataset_mean = 0.5

        assert model.is_fitted() is True

    def test_an_empty_effect_array_still_counts_as_fitted(self):
        """``NaiveMeanEffectsPredictor`` stores an empty tissue vector when no tissue block is present."""
        model = _OneVector()
        model._dataset_mean = 0.5
        model._effects = np.empty((0,))

        assert model.is_fitted() is True


class TestGetState:
    def test_is_empty_while_unfitted(self):
        assert _OneVector().get_state() == {}

    def test_is_empty_when_one_effect_array_is_missing(self):
        model = _ThreeVectors()
        model._dataset_mean = 0.5
        model._tissue_effects = np.zeros(1)

        assert model.get_state() == {}

    def test_keys_are_the_attribute_names_without_the_underscore(self):
        model = _ThreeVectors()
        model._dataset_mean = 0.5
        model._tissue_effects = np.zeros(1)
        model._cell_line_effects = np.zeros(1)
        model._drug_effects = np.zeros(1)

        assert list(model.get_state()) == [
            "dataset_mean",
            "tissue_effects",
            "cell_line_effects",
            "drug_effects",
        ]

    def test_arrays_are_serialized_as_plain_lists(self):
        model = _OneVector()
        model._dataset_mean = 0.5
        model._effects = np.array([1.0, 2.0])

        assert model.get_state()["effects"] == [1.0, 2.0]

    def test_effect_free_host_reports_only_the_mean(self):
        model = _MeanOnly()
        model._dataset_mean = 0.25

        assert model.get_state() == {"dataset_mean": 0.25}


class TestSetState:
    def test_round_trips_a_vector(self):
        model = _OneVector()
        model._dataset_mean = 0.5
        model._effects = np.array([1.0, -2.0, 3.0])

        restored = _OneVector()
        restored.set_state(model.get_state())

        assert restored._dataset_mean == 0.5
        np.testing.assert_allclose(restored._effects, model._effects)

    def test_round_trips_every_effect_array(self):
        model = _ThreeVectors()
        model._dataset_mean = -1.5
        model._tissue_effects = np.array([0.1])
        model._cell_line_effects = np.array([0.2, 0.3])
        model._drug_effects = np.array([0.4, 0.5, 0.6])

        restored = _ThreeVectors()
        restored.set_state(model.get_state())

        assert restored.is_fitted() is True
        np.testing.assert_allclose(restored._cell_line_effects, [0.2, 0.3])
        np.testing.assert_allclose(restored._drug_effects, [0.4, 0.5, 0.6])

    def test_restores_a_matrix_as_two_dimensional(self):
        model = _OneMatrix()
        model._dataset_mean = 0.0
        model._effects = np.array([[1.0, 2.0], [3.0, 4.0]])

        restored = _OneMatrix()
        restored.set_state(model.get_state())

        assert restored._effects.shape == (2, 2)
        np.testing.assert_allclose(restored._effects, model._effects)

    def test_a_matrix_host_keeps_a_flat_payload_two_dimensional(self):
        restored = _OneMatrix()

        restored.set_state({"dataset_mean": 0.0, "effects": [1.0, 2.0]})

        assert restored._effects.shape == (2, 1)

    def test_a_vector_host_flattens_its_payload(self):
        restored = _OneVector()

        restored.set_state({"dataset_mean": 0.0, "effects": [[1.0], [2.0]]})

        assert restored._effects.shape == (2,)

    def test_an_empty_state_leaves_the_model_unfitted(self):
        restored = _OneVector()

        restored.set_state({})

        assert restored.is_fitted() is False

    def test_a_partial_state_does_not_clobber_what_is_absent(self):
        model = _OneVector()
        model._dataset_mean = 0.5
        model._effects = np.array([1.0])

        model.set_state({"dataset_mean": 2.0})

        assert model._dataset_mean == 2.0
        np.testing.assert_allclose(model._effects, [1.0])

    def test_reads_a_stringified_mean(self):
        """``state_float`` accepts the string a JSON/YAML checkpoint may carry."""
        model = _MeanOnly()

        model.set_state({"dataset_mean": "0.75"})

        assert model._dataset_mean == 0.75

    def test_coerces_restored_arrays_to_float(self):
        model = _OneVector()

        model.set_state({"dataset_mean": 0.0, "effects": [1, 2]})

        assert model._effects.dtype == np.float64
