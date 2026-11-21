"""The response pairs handed to predictors."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.data import synthetic
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.data.selection_mask import SelectionMask


@pytest.fixture()
def pairs() -> ResponseTable:
    """Two measured pairs across two samples and two drugs."""
    return ResponseTable(
        response=np.array([1.0, 2.0]),
        sample_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )


class TestConstruction:
    def test_fields_are_stored_verbatim(self, pairs: ResponseTable):
        np.testing.assert_array_equal(pairs.response, np.array([1.0, 2.0]))
        assert pairs.sample_ids.tolist() == ["cl1", "cl2"]
        assert pairs.drug_ids.tolist() == ["d1", "d2"]

    def test_arrays_are_not_copied(self):
        response = np.array([1.0])

        pairs = ResponseTable(response=response, sample_ids=np.array(["cl1"]), drug_ids=np.array(["d1"]))

        assert pairs.response is response

    def test_keyword_construction_requires_all_three_fields(self):
        with pytest.raises(TypeError):
            ResponseTable(response=np.array([1.0]))  # type: ignore[call-arg]


class TestValidation:
    @pytest.mark.parametrize(
        ("sample_length", "drug_length", "message"),
        [(2, 3, "drug_ids must be one-dimensional"), (3, 2, "sample_ids must be one-dimensional")],
    )
    def test_identifiers_need_one_value_per_pair(self, sample_length: int, drug_length: int, message: str):
        with pytest.raises(ValueError, match=message):
            ResponseTable(np.ones(2), np.arange(sample_length).astype(str), np.arange(drug_length).astype(str))

    def test_identifiers_must_be_one_dimensional(self):
        with pytest.raises(ValueError, match="sample_ids must be one-dimensional"):
            ResponseTable(np.ones(1), np.array("cl1"), np.array(["d1"]))


class TestLength:
    def test_len_counts_response_pairs(self, pairs: ResponseTable):
        assert len(pairs) == 2

    def test_an_empty_container_has_length_zero(self):
        empty = ResponseTable(
            response=np.array([]),
            sample_ids=np.array([]),
            drug_ids=np.array([]),
        )

        assert len(empty) == 0

    def test_nan_responses_still_count_as_pairs(self):
        """NaN marks an unmeasured pair; filtering happens in the predictors, not here."""
        with_nan = ResponseTable(
            response=np.array([1.0, np.nan]),
            sample_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )

        assert len(with_nan) == 2


class TestSubset:
    def test_a_boolean_mask_keeps_the_selected_pairs_in_order(self, pairs: ResponseTable):
        subset = pairs[np.array([False, True])]

        assert subset.response.tolist() == [2.0]
        assert subset.sample_ids.tolist() == ["cl2"]
        assert subset.drug_ids.tolist() == ["d2"]

    def test_an_all_false_mask_yields_empty_pairs(self, pairs: ResponseTable):
        assert len(pairs[np.zeros(2, dtype=bool)]) == 0

    def test_the_original_is_unchanged(self, pairs: ResponseTable):
        _ = pairs[np.array([True, False])]

        assert len(pairs) == 2

    def test_slicing_selects_entries_in_order(self, pairs: ResponseTable):
        assert pairs[:1].sample_ids.tolist() == ["cl1"]


class TestImmutability:
    def test_fields_cannot_be_reassigned(self, pairs: ResponseTable):
        with pytest.raises(dataclasses.FrozenInstanceError):
            pairs.response = np.array([9.0])  # type: ignore[misc]

    def test_slots_replace_the_instance_dict(self, pairs: ResponseTable):
        """``slots=True`` keeps these per-pair containers cheap to allocate."""
        assert ResponseTable.__slots__ == ("response", "sample_ids", "drug_ids", "layers")
        assert not hasattr(pairs, "__dict__")


class TestLayers:
    def test_pairs_have_no_layers_by_default(self, pairs: ResponseTable):
        assert dict(pairs.layers) == {}

    def test_a_layer_needs_one_value_per_pair(self):
        with pytest.raises(ValueError, match="regulation"):
            ResponseTable(
                np.array([1.0, 2.0]), np.array(["s1", "s2"]), np.array(["d1", "d2"]), {"regulation": np.zeros(3)}
            )

    def test_subset_keeps_the_layers_aligned(self):
        pairs = ResponseTable(
            np.array([1.0, 2.0, 3.0]),
            np.array(["s1", "s2", "s3"]),
            np.array(["d1", "d1", "d1"]),
            {"regulation": np.array([-1.0, 0.0, 1.0])},
        )

        assert pairs[np.array([True, False, True])].layers["regulation"].tolist() == [-1.0, 1.0]

    def test_from_dataset_reads_a_layer_at_the_observed_pairs_of_a_shuffled_scope(self):
        dataset = synthetic(n_samples=6, n_drugs=3, seed=1)
        scope = SelectionMask(np.ones(dataset.response_matrix.shape, dtype=bool), pair_seed=3)

        pairs = ResponseTable.from_dataset(dataset, scope, observed_only=True, layers=("regulation",))

        positions = scope.pairs[~np.isnan(dataset.response_matrix[scope.pairs[:, 0], scope.pairs[:, 1]])]
        expected = dataset.get_response_layer("regulation")[positions[:, 0], positions[:, 1]]
        np.testing.assert_array_equal(pairs.layers["regulation"], expected)
        assert len(pairs.layers["regulation"]) == len(pairs)

    def test_from_dataset_leaves_out_a_layer_the_dataset_lacks(self):
        dataset = synthetic(n_samples=4, n_drugs=2, seed=1)
        scope = SelectionMask(np.ones(dataset.response_matrix.shape, dtype=bool))

        pairs = ResponseTable.from_dataset(dataset, scope, layers=("no_such_layer",))

        assert dict(pairs.layers) == {}
