"""Tests for :mod:`drevalpy.data._concat_tables`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.data._concat_tables import listed, prefixed, stack_obs, union_var


class TestHelpers:
    def test_prefixed_keeps_the_index_name(self) -> None:
        result = prefixed("S", pd.Index(["a", "b"], name="sample"))

        assert list(result) == ["S::a", "S::b"]
        assert result.name == "sample"

    def test_listed_says_how_many_more(self) -> None:
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

    def test_categoricals_with_different_categories_stay_categorical(self) -> None:
        first = pd.DataFrame({"tissue": pd.Categorical(["Lung"])}, index=["a"])
        second = pd.DataFrame({"tissue": pd.Categorical(["Skin"])}, index=["b"])

        stacked = stack_obs([first, second])

        assert isinstance(stacked["tissue"].dtype, pd.CategoricalDtype)
        assert set(stacked["tissue"].cat.categories) == {"Lung", "Skin"}

    def test_a_column_that_is_categorical_in_only_one_table_is_not_forced_to_be(self) -> None:
        first = pd.DataFrame({"tissue": pd.Categorical(["Lung"])}, index=["a"])
        second = pd.DataFrame({"tissue": ["Skin"]}, index=["b"])

        assert not isinstance(stack_obs([first, second])["tissue"].dtype, pd.CategoricalDtype)

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

    def test_ids_are_first_seen_and_the_index_name_is_kept(self) -> None:
        first = self._frame(["3", "1"], canonical_smiles=["C", "N"])
        second = self._frame(["1", "2"], canonical_smiles=["N", "O"])

        merged = union_var([first, second])

        assert list(merged.index) == ["3", "1", "2"]
        assert merged.index.name == "pubchem_id"
        assert list(merged["canonical_smiles"]) == ["C", "N", "O"]

    def test_the_first_non_null_value_wins(self) -> None:
        first = self._frame(["1", "2"], name=["one", np.nan])
        second = self._frame(["1", "2"], name=["uno", "dos"])

        assert list(union_var([first, second])["name"]) == ["one", "dos"]

    def test_a_column_only_one_table_has_is_nan_for_the_others_drugs(self) -> None:
        first = self._frame(["1"], extra=["x"])
        second = self._frame(["2"], other=[3])

        merged = union_var([first, second])

        assert merged.loc["1", "extra"] == "x"
        assert pd.isna(merged.loc["2", "extra"])
        assert merged.loc["2", "other"] == 3

    def test_conflicting_smiles_raise_and_list_every_id_once(self) -> None:
        first = self._frame(["1", "2", "3"], canonical_smiles=["C", "N", "O"])
        second = self._frame(["1", "2", "3"], canonical_smiles=["CC", "NN", "O"])

        with pytest.raises(ValueError, match=r"2 drug IDs: 1, 2"):
            union_var([first, second])

    def test_a_null_smiles_is_not_a_conflict(self) -> None:
        first = self._frame(["1"], canonical_smiles=[np.nan])
        second = self._frame(["1"], canonical_smiles=["C"])

        assert union_var([first, second]).loc["1", "canonical_smiles"] == "C"
