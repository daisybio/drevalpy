"""Tests for the helpers behind ``concat``: ``_concat_uns``, ``_concat_tables`` and ``_concat_stack``."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drevalpy.data._concat_stack import stack_response, stack_view
from drevalpy.data._concat_tables import listed, prefixed, stack_obs, union_var
from drevalpy.data._concat_uns import merge_uns, same
from drevalpy.errors import InvalidInputError

NAMES = ["A", "B"]


class TestSame:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("tpm", "tpm"),
            (3, 3.0),
            ({"x": [1, 2], "y": {"z": "a"}}, {"y": {"z": "a"}, "x": [1, 2]}),
            (np.array([1, 2]), [1, 2]),
        ],
    )
    def test_equal_values(self, first: object, second: object) -> None:
        assert same(first, second)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("tpm", "fpkm"),
            ({"x": 1}, {"x": 1, "y": 2}),
            ({"x": [1, 2]}, {"x": [1, 3]}),
            (np.array([1, 2]), np.array([1, 2, 3])),
            ("a", [1]),
        ],
    )
    def test_different_values(self, first: object, second: object) -> None:
        assert not same(first, second)


class TestMergeUns:
    def test_agreed_keys_are_deep_copied(self) -> None:
        first = {"nested": {"k": [1]}}

        merged = merge_uns([first, {"nested": {"k": [1]}}], names=NAMES, where="x")
        merged["nested"]["k"].append(2)

        assert first == {"nested": {"k": [1]}}

    def test_disagreeing_keys_are_dropped_with_one_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            merged = merge_uns([{"a": 1, "b": 1}, {"a": 2, "b": 2, "c": 3}], names=NAMES, where="view 'v'")

        assert merged == {}
        assert caplog.text.count("concat dropped view 'v' uns keys") == 1
        assert "a, b, c" in caplog.text

    def test_a_key_recorded_by_one_input_is_not_agreed(self) -> None:
        assert merge_uns([{"a": 1}, {}], names=NAMES, where="x") == {}

    def test_skipped_keys_are_left_out_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            merged = merge_uns([{"a": 1}, {"a": 1}], names=NAMES, where="x", skip={"a"})

        assert merged == {}
        assert caplog.text == ""

    @pytest.mark.parametrize(
        ("uns", "error", "match"),
        [
            ([{"u": "tpm"}, {"u": "fpkm"}], ValueError, r"view 'v': uns\['u'\].*A='tpm', B='fpkm'"),
            ([{"u": "tpm"}, {}], InvalidInputError, "B=not recorded"),
            ([{"u": "x" * 200}, {"u": "y"}], ValueError, r"^.{0,199}$"),
        ],
        ids=["differing", "recorded-by-one", "long-values-shortened"],
    )
    def test_a_strict_key_that_differs_or_is_missing_raises(self, uns: list[dict], error: type, match: str) -> None:
        with pytest.raises(error, match=match):
            merge_uns(uns, names=NAMES, where="view 'v'", strict=lambda key: True)


def test_prefixed_keeps_the_index_name_and_listed_says_how_many_more() -> None:
    result = prefixed("S", pd.Index(["a", "b"], name="sample"))

    assert list(result) == ["S::a", "S::b"]
    assert result.name == "sample"
    assert listed(["a", "b"]) == "a, b"
    assert listed(list("abcdefghij")) == "a, b, c, d, e, f, g, h and 2 more"


class TestStackObs:
    def test_columns_are_unioned_with_nan_fill(self) -> None:
        first = pd.DataFrame({"tissue": ["Lung"], "only_first": [1.5]}, index=["a"])
        second = pd.DataFrame({"tissue": ["Skin"], "only_second": ["x"]}, index=["b"])

        stacked = stack_obs([first, second])

        assert list(stacked.columns) == ["tissue", "only_first", "only_second"]
        assert list(stacked.index) == ["a", "b"]
        assert np.isnan(stacked.loc["b", "only_first"])
        assert stacked.loc["a", "only_first"] == 1.5

    def test_categoricals_stay_categorical_only_when_every_table_has_them(self) -> None:
        lung = pd.DataFrame({"tissue": pd.Categorical(["Lung"])}, index=["a"])
        skin = pd.DataFrame({"tissue": pd.Categorical(["Skin"])}, index=["b"])
        plain = pd.DataFrame({"tissue": ["Skin"]}, index=["b"])

        both = stack_obs([lung, skin])["tissue"]

        assert isinstance(both.dtype, pd.CategoricalDtype)
        assert set(both.cat.categories) == {"Lung", "Skin"}
        assert not isinstance(stack_obs([lung, plain])["tissue"].dtype, pd.CategoricalDtype)

    def test_a_boolean_column_missing_in_one_table_stays_boolean(self) -> None:
        first = pd.DataFrame({"flag": [True]}, index=["a"])
        second = pd.DataFrame({"other": [1]}, index=["b"])

        stacked = stack_obs([first, second])

        assert str(stacked["flag"].dtype) == "boolean"
        assert stacked.loc["a", "flag"]
        assert pd.isna(stacked.loc["b", "flag"])


class TestUnionVar:
    @staticmethod
    def _frame(index: list[str], **columns: list[object]) -> pd.DataFrame:
        return pd.DataFrame(columns, index=pd.Index(index, name="pubchem_id"))

    def test_ids_are_first_seen_with_the_index_name_and_the_first_non_null_value_wins(self) -> None:
        first = self._frame(["3", "1"], canonical_smiles=["C", "N"], name=["three", np.nan])
        second = self._frame(["1", "2"], canonical_smiles=["N", "O"], name=["uno", "dos"])

        merged = union_var([first, second])

        assert list(merged.index) == ["3", "1", "2"]
        assert merged.index.name == "pubchem_id"
        assert list(merged["canonical_smiles"]) == ["C", "N", "O"]
        assert list(merged["name"]) == ["three", "uno", "dos"]

    def test_a_column_only_one_table_has_is_nan_for_the_others_drugs(self) -> None:
        merged = union_var([self._frame(["1"], extra=["x"]), self._frame(["2"], other=[3])])

        assert merged.loc["1", "extra"] == "x"
        assert pd.isna(merged.loc["2", "extra"])
        assert merged.loc["2", "other"] == 3

    def test_conflicting_smiles_raise_and_list_every_id_once_but_a_null_is_no_conflict(self) -> None:
        first = self._frame(["1", "2", "3"], canonical_smiles=["C", "N", "O"])
        second = self._frame(["1", "2", "3"], canonical_smiles=["CC", "NN", "O"])

        with pytest.raises(InvalidInputError, match=r"2 drug IDs: 1, 2"):
            union_var([first, second])
        nulled = union_var([self._frame(["1"], canonical_smiles=[np.nan]), self._frame(["1"], canonical_smiles=["C"])])
        assert nulled.loc["1", "canonical_smiles"] == "C"


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

        with pytest.raises(
            InvalidInputError, match=r"1 features of A missing in B \(e\.g\. g2\); 1 only in B \(e\.g\. g9\)"
        ):
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
