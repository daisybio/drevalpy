"""Tests for :mod:`drevalpy.data._concat_stack`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drevalpy.data._concat_stack import stack_response, stack_view

NAMES = ["A", "B"]


def _view(rows: list[str], features: list[str], values: np.ndarray | sp.spmatrix, **uns: object) -> ad.AnnData:
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=pd.Index(rows)),
        var=pd.DataFrame(index=pd.Index(features)),
        uns=dict(uns),
    )


class TestStackView:
    def test_sparse_inputs_stay_sparse(self) -> None:
        first = _view(["s1"], ["g1", "g2"], sp.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32)))
        second = _view(["s1"], ["g1", "g2"], np.array([[0.0, 2.0]], dtype=np.float32))

        stacked = stack_view("v", [first, second], NAMES, "exact")

        assert sp.issparse(stacked.X)
        np.testing.assert_array_equal(stacked.X.toarray(), [[1.0, 0.0], [0.0, 2.0]])
        assert list(stacked.obs_names) == ["A::s1", "B::s1"]

    def test_inner_join_reorders_every_input_to_the_first_inputs_order(self) -> None:
        first = _view(["s1"], ["g1", "g2", "g3"], np.array([[1, 2, 3]], dtype=np.float32))
        second = _view(["s1"], ["g3", "g1"], np.array([[30, 10]], dtype=np.float32))

        stacked = stack_view("v", [first, second], NAMES, "inner")

        assert list(stacked.var_names) == ["g1", "g3"]
        np.testing.assert_array_equal(stacked.X, [[1, 3], [10, 30]])

    def test_the_mismatch_message_lists_features_missing_and_extra(self) -> None:
        first = _view(["s1"], ["g1", "g2"], np.zeros((1, 2), dtype=np.float32))
        second = _view(["s1"], ["g1", "g9"], np.zeros((1, 2), dtype=np.float32))

        with pytest.raises(ValueError, match=r"1 features of A missing in B \(e\.g\. g2\); 1 only in B \(e\.g\. g9\)"):
            stack_view("v", [first, second], NAMES, "exact")

    def test_units_are_checked_for_unit_keys_only(self) -> None:
        first = _view(["s1"], ["g1"], np.zeros((1, 1), dtype=np.float32), expression_unit="tpm", note="a")
        second = _view(["s1"], ["g1"], np.zeros((1, 1), dtype=np.float32), expression_unit="tpm", note="b")

        stacked = stack_view("v", [first, second], NAMES, "exact")

        assert stacked.uns == {"expression_unit": "tpm"}

    def test_layer_units_are_checked_too(self) -> None:
        first = _view(["s1"], ["g1"], np.zeros((1, 1), dtype=np.float32), layer_units={"counts": "rsem"})
        second = _view(["s1"], ["g1"], np.zeros((1, 1), dtype=np.float32), layer_units={"counts": "raw"})

        with pytest.raises(ValueError, match="layer_units"):
            stack_view("v", [first, second], NAMES, "inner")


class TestStackResponse:
    @staticmethod
    def _response(rows: list[str], drugs: list[str], dtype: type = np.float32) -> ad.AnnData:
        values = np.arange(len(rows) * len(drugs), dtype=dtype).reshape(len(rows), len(drugs))
        response = ad.AnnData(
            X=values,
            obs=pd.DataFrame(index=pd.Index(rows)),
            var=pd.DataFrame({"canonical_smiles": ["C"] * len(drugs)}, index=pd.Index(drugs, name="pubchem_id")),
        )
        response.layers["AUC"] = values + 100
        return response

    def test_a_drug_in_both_inputs_lands_in_one_column(self) -> None:
        first = self._response(["s1"], ["d1", "d2"])
        second = self._response(["s2"], ["d2", "d3"])

        stacked = stack_response([first, second], NAMES)

        assert list(stacked.var_names) == ["d1", "d2", "d3"]
        np.testing.assert_array_equal(stacked.X, [[0, 1, np.nan], [np.nan, 0, 1]])
        np.testing.assert_array_equal(stacked.layers["AUC"], [[100, 101, np.nan], [np.nan, 100, 101]])

    def test_the_result_is_float_even_when_the_inputs_are_not(self) -> None:
        stacked = stack_response(
            [self._response(["s1"], ["d1"], np.int64), self._response(["s2"], ["d2"], np.int64)], NAMES
        )

        assert np.issubdtype(stacked.X.dtype, np.floating)

    def test_strict_uns_decides_which_keys_may_differ(self) -> None:
        first = self._response(["s1"], ["d1"])
        second = self._response(["s2"], ["d1"])
        first.uns.update({"x_column": "pEC50", "rr_seed": 1})
        second.uns.update({"x_column": "pEC50", "rr_seed": 2})

        lenient = stack_response([first, second], NAMES, strict_uns=lambda key: key == "x_column")

        assert lenient.uns == {"x_column": "pEC50"}
        with pytest.raises(ValueError, match="rr_seed"):
            stack_response([first, second], NAMES)
