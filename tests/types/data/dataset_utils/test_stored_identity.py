"""Tests for the helpers that read a dataset's stored identity back out of ``uns``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drevalpy.types.data.dataset_utils._stored_identity import plain_value, stored_randomization


class TestStoredRandomization:
    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param(["SVRC", "gene_expression"], id="list"),
            pytest.param(("SVRC", "gene_expression"), id="tuple"),
            pytest.param(np.array(["SVRC", "gene_expression"], dtype=object), id="ndarray"),
        ],
    )
    def test_a_pair_comes_back_as_a_tuple_of_strings(self, stored) -> None:
        assert stored_randomization({"randomization": stored}) == ("SVRC", "gene_expression")

    def test_absent_key_is_no_randomization(self) -> None:
        assert stored_randomization({}) is None

    @pytest.mark.parametrize("stored", [["SVRC"], "SVRC", ["a", "b", "c"], None], ids=["one", "str", "three", "none"])
    def test_anything_but_a_pair_is_no_randomization(self, stored) -> None:
        assert stored_randomization({"randomization": stored}) is None


class TestPlainValue:
    def test_arrays_become_lists_and_numpy_scalars_python_scalars(self) -> None:
        stored = {"views": np.array(["a", "b"], dtype=object), "seed": np.int64(3), "nested": {"x": np.float32(0.5)}}

        plain = plain_value(stored)

        assert plain == {"views": ["a", "b"], "seed": 3, "nested": {"x": 0.5}}
        json.dumps(plain)

    def test_an_empty_array_is_an_empty_list(self) -> None:
        assert plain_value(np.array([])) == []

    def test_plain_python_values_pass_through(self) -> None:
        assert plain_value({"type": "permutation", "views": ["a"], "n": 2}) == {
            "type": "permutation",
            "views": ["a"],
            "n": 2,
        }
