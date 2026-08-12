"""Tests for the shared densification helper.

``to_dense`` is on every matrix read on the dataset hot path, so the contract
worth pinning is that it duck-types on ``toarray`` and returns dense inputs
untouched rather than copying them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from drevalpy.types.data.dataset_utils._dense import to_dense


class TestSparseInput:
    @pytest.mark.parametrize(
        "make_sparse",
        [
            pytest.param(sparse.csr_matrix, id="csr"),
            pytest.param(sparse.csc_matrix, id="csc"),
            pytest.param(sparse.coo_matrix, id="coo"),
        ],
    )
    def test_sparse_matrices_are_densified(self, make_sparse):
        dense = np.array([[1.0, 0.0], [0.0, 2.0]])

        result = to_dense(make_sparse(dense))

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, dense)

    def test_densification_preserves_the_shape(self):
        result = to_dense(sparse.csr_matrix((3, 5), dtype=np.float32))

        assert result.shape == (3, 5)

    def test_implicit_zeros_materialize(self):
        matrix = sparse.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))

        assert to_dense(matrix).sum() == 1.0


class TestDenseInput:
    def test_a_dense_array_is_returned_unchanged(self):
        array = np.array([[1.0, 2.0]])

        assert to_dense(array) is array

    def test_a_dataframe_is_returned_unchanged(self):
        frame = pd.DataFrame({"a": [1.0]})

        assert to_dense(frame) is frame

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="none"),
            pytest.param([[1.0]], id="nested-list"),
            pytest.param(3.5, id="scalar"),
        ],
    )
    def test_objects_without_toarray_pass_through(self, value):
        assert to_dense(value) is value


class TestDuckTyping:
    def test_any_object_exposing_toarray_is_densified(self):
        class FakeSparse:
            def toarray(self) -> np.ndarray:
                return np.array([[7.0]])

        np.testing.assert_array_equal(to_dense(FakeSparse()), np.array([[7.0]]))

    def test_a_non_callable_toarray_attribute_is_ignored(self):
        class NotSparse:
            toarray = "not callable"

        instance = NotSparse()

        assert to_dense(instance) is instance
