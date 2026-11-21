"""The response triple handed to predictors."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.data import synthetic
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.pair_mask import PairMask


@pytest.fixture()
def batch() -> ResponseBatch:
    """Two measured pairs across two samples and two drugs."""
    return ResponseBatch(
        response=np.array([1.0, 2.0]),
        sample_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )


class TestConstruction:
    def test_fields_are_stored_verbatim(self, batch: ResponseBatch):
        np.testing.assert_array_equal(batch.response, np.array([1.0, 2.0]))
        assert batch.sample_ids.tolist() == ["cl1", "cl2"]
        assert batch.drug_ids.tolist() == ["d1", "d2"]

    def test_arrays_are_not_copied(self):
        response = np.array([1.0])

        batch = ResponseBatch(response=response, sample_ids=np.array(["cl1"]), drug_ids=np.array(["d1"]))

        assert batch.response is response

    def test_keyword_construction_requires_all_three_fields(self):
        with pytest.raises(TypeError):
            ResponseBatch(response=np.array([1.0]))  # type: ignore[call-arg]


class TestLength:
    def test_len_counts_response_pairs(self, batch: ResponseBatch):
        assert len(batch) == 2

    def test_an_empty_batch_has_length_zero(self):
        empty = ResponseBatch(
            response=np.array([]),
            sample_ids=np.array([]),
            drug_ids=np.array([]),
        )

        assert len(empty) == 0

    def test_nan_responses_still_count_as_pairs(self):
        """NaN marks an unmeasured pair; filtering happens in the predictors, not here."""
        with_nan = ResponseBatch(
            response=np.array([1.0, np.nan]),
            sample_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )

        assert len(with_nan) == 2


class TestSubset:
    def test_a_boolean_mask_keeps_the_selected_pairs_in_order(self, batch: ResponseBatch):
        subset = batch.subset(np.array([False, True]))

        assert subset.response.tolist() == [2.0]
        assert subset.sample_ids.tolist() == ["cl2"]
        assert subset.drug_ids.tolist() == ["d2"]

    def test_an_all_false_mask_yields_an_empty_batch(self, batch: ResponseBatch):
        assert len(batch.subset(np.zeros(2, dtype=bool))) == 0

    def test_the_original_is_unchanged(self, batch: ResponseBatch):
        batch.subset(np.array([True, False]))

        assert len(batch) == 2


class TestImmutability:
    def test_fields_cannot_be_reassigned(self, batch: ResponseBatch):
        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.response = np.array([9.0])  # type: ignore[misc]

    def test_slots_replace_the_instance_dict(self, batch: ResponseBatch):
        """``slots=True`` keeps these per-pair containers cheap to allocate."""
        assert ResponseBatch.__slots__ == ("response", "sample_ids", "drug_ids", "layers")
        assert not hasattr(batch, "__dict__")


class TestLayers:
    def test_a_batch_has_no_layers_by_default(self, batch: ResponseBatch):
        assert dict(batch.layers) == {}

    def test_a_layer_needs_one_value_per_pair(self):
        with pytest.raises(ValueError, match="regulation"):
            ResponseBatch(
                np.array([1.0, 2.0]), np.array(["s1", "s2"]), np.array(["d1", "d2"]), {"regulation": np.zeros(3)}
            )

    def test_subset_keeps_the_layers_aligned(self):
        batch = ResponseBatch(
            np.array([1.0, 2.0, 3.0]),
            np.array(["s1", "s2", "s3"]),
            np.array(["d1", "d1", "d1"]),
            {"regulation": np.array([-1.0, 0.0, 1.0])},
        )

        assert batch.subset(np.array([True, False, True])).layers["regulation"].tolist() == [-1.0, 1.0]

    def test_from_dataset_reads_a_layer_at_the_observed_pairs_of_a_shuffled_scope(self):
        dataset = synthetic(n_samples=6, n_drugs=3, seed=1)
        scope = PairMask(np.ones(dataset.response_matrix.shape, dtype=bool), pair_seed=3)

        batch = ResponseBatch.from_dataset(dataset, scope, observed_only=True, layers=("regulation",))

        pairs = scope.pairs[~np.isnan(dataset.response_matrix[scope.pairs[:, 0], scope.pairs[:, 1]])]
        expected = dataset.get_response_layer("regulation")[pairs[:, 0], pairs[:, 1]]
        np.testing.assert_array_equal(batch.layers["regulation"], expected)
        assert len(batch.layers["regulation"]) == len(batch)

    def test_from_dataset_leaves_out_a_layer_the_dataset_lacks(self):
        dataset = synthetic(n_samples=4, n_drugs=2, seed=1)
        scope = PairMask(np.ones(dataset.response_matrix.shape, dtype=bool))

        batch = ResponseBatch.from_dataset(dataset, scope, layers=("no_such_layer",))

        assert dict(batch.layers) == {}
