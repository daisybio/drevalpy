"""Tests for :mod:`drevalpy.data.splitters._quality_view`."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.data.splitters._quality_view import QualityMasked
from drevalpy.types import MuDataLike
from tests.data.splitters._helpers import MockMuDataset, first_measured_pairs


@pytest.fixture
def failing(mock_dataset: MockMuDataset) -> list[tuple[int, int]]:
    return first_measured_pairs(mock_dataset, 3)


@pytest.fixture
def dataset(failing: list[tuple[int, int]]) -> MockMuDataset:
    return MockMuDataset(failing_pairs=failing)


class TestResponseMatrix:
    def test_failed_pairs_are_nan(self, dataset: MockMuDataset, failing: list[tuple[int, int]]) -> None:
        view = QualityMasked(dataset, CurveQuality.default())

        assert all(np.isnan(view.response_matrix[row, column]) for row, column in failing)

    def test_other_pairs_keep_their_values(self, dataset: MockMuDataset, failing: list[tuple[int, int]]) -> None:
        view = QualityMasked(dataset, CurveQuality.default())

        keep = np.ones(dataset.response_matrix.shape, dtype=bool)
        for row, column in failing:
            keep[row, column] = False
        np.testing.assert_array_equal(view.response_matrix[keep], dataset.response_matrix[keep])

    def test_the_wrapped_dataset_is_not_modified(self, dataset: MockMuDataset) -> None:
        before = dataset.response_matrix.copy()

        QualityMasked(dataset, CurveQuality.default())

        np.testing.assert_array_equal(dataset.response_matrix, before)

    def test_off_blanks_nothing(self, dataset: MockMuDataset) -> None:
        view = QualityMasked(dataset, CurveQuality.off())

        np.testing.assert_array_equal(view.response_matrix, dataset.response_matrix)

    def test_a_rule_needing_a_missing_layer_raises(self, dataset: MockMuDataset) -> None:
        with pytest.raises(KeyError):
            QualityMasked(dataset, CurveQuality(min_r2=0.5))


class TestDelegation:
    @pytest.fixture
    def view(self, dataset: MockMuDataset) -> QualityMasked:
        return QualityMasked(dataset, CurveQuality.default())

    def test_satisfies_the_protocol(self, view: QualityMasked) -> None:
        assert isinstance(view, MuDataLike)

    def test_ids(self, view: QualityMasked, dataset: MockMuDataset) -> None:
        np.testing.assert_array_equal(view.sample_ids, dataset.sample_ids)
        np.testing.assert_array_equal(view.drug_ids, dataset.drug_ids)

    def test_sample_metadata(self, view: QualityMasked, dataset: MockMuDataset) -> None:
        ids = dataset.sample_ids[:4]

        np.testing.assert_array_equal(view.get_tissue(ids), dataset.get_tissue(ids))
        np.testing.assert_array_equal(view.sample_column("tissue", ids), dataset.get_tissue(ids))

    def test_response_layers_stay_unmasked(self, view: QualityMasked, dataset: MockMuDataset) -> None:
        assert view.response_layer_names() == dataset.response_layer_names()
        np.testing.assert_array_equal(
            view.get_response_layer("relevance_score"), dataset.get_response_layer("relevance_score")
        )

    def test_other_attributes_reach_the_dataset(self, view: QualityMasked, dataset: MockMuDataset) -> None:
        dataset.extra = "kept"

        assert view.extra == "kept"

    def test_unknown_attributes_raise(self, view: QualityMasked) -> None:
        with pytest.raises(AttributeError):
            view.nonexistent  # noqa: B018

    def test_copies_do_not_recurse(self, view: QualityMasked) -> None:
        np.testing.assert_array_equal(copy.deepcopy(view).response_matrix, view.response_matrix)
