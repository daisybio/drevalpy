"""Tests for :mod:`drevalpy.data._concat_tables`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.data._concat_tables import listed, prefixed, stack_obs, union_var
from drevalpy.errors import InvalidInputError


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
