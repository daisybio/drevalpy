"""Tests for the immutable ``ResponseBatch`` triple container."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.types.data.batch.response_batch import ResponseBatch


@pytest.fixture()
def batch() -> ResponseBatch:
    """Two measured pairs across two cell lines and two drugs."""
    return ResponseBatch(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )


class TestConstruction:
    def test_fields_are_stored_verbatim(self, batch: ResponseBatch):
        np.testing.assert_array_equal(batch.response, np.array([1.0, 2.0]))
        assert batch.cell_line_ids.tolist() == ["cl1", "cl2"]
        assert batch.drug_ids.tolist() == ["d1", "d2"]

    def test_arrays_are_not_copied(self):
        response = np.array([1.0])

        batch = ResponseBatch(response=response, cell_line_ids=np.array(["cl1"]), drug_ids=np.array(["d1"]))

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
            cell_line_ids=np.array([]),
            drug_ids=np.array([]),
        )

        assert len(empty) == 0

    def test_nan_responses_still_count_as_pairs(self):
        """NaN marks an unmeasured pair; filtering happens in the predictors, not here."""
        with_nan = ResponseBatch(
            response=np.array([1.0, np.nan]),
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )

        assert len(with_nan) == 2


class TestImmutability:
    def test_fields_cannot_be_reassigned(self, batch: ResponseBatch):
        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.response = np.array([9.0])  # type: ignore[misc]

    def test_slots_replace_the_instance_dict(self, batch: ResponseBatch):
        """``slots=True`` keeps these per-pair containers cheap to allocate."""
        assert ResponseBatch.__slots__ == ("response", "cell_line_ids", "drug_ids")
        assert not hasattr(batch, "__dict__")
