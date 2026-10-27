"""Tests for :mod:`drevalpy.data._replicate_fits`, the ``response.uns["replicate_fits"]`` format."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drevalpy.data._replicate_fits import DRUG_COLUMN, SAMPLE_COLUMN, restrict, stack
from tests.data._replicate_fits_table import fits_table


class TestRestrict:
    def test_keeps_the_rows_of_the_given_samples_and_drugs(self) -> None:
        table = fits_table(["S1", "S2", "S3"], ["D1", "D2"])

        kept = restrict(table, ["S1", "S3", "unknown"], ["D2"])

        assert isinstance(kept.index, pd.RangeIndex)
        assert len(kept) == 4
        assert set(kept[SAMPLE_COLUMN]) == {"S1", "S3"}
        assert set(kept[DRUG_COLUMN]) == {"D2"}
        assert list(kept[SAMPLE_COLUMN].cat.categories) == ["S1", "S3"]
        assert list(kept[DRUG_COLUMN].cat.categories) == ["D2"]
        assert kept["replicate"].dtype == np.int32
        assert kept["pEC50"].dtype == np.float32
        assert len(table) == 12

    def test_drops_categories_no_kept_row_uses_in_every_column(self) -> None:
        table = fits_table(["S1"], ["D1"])

        kept = restrict(table, [], ["D1"])

        assert kept.empty
        assert list(kept["status"].cat.categories) == []


class TestStack:
    def test_prefixes_sample_ids_and_unions_columns(self) -> None:
        first = fits_table(["S1"], ["D1"], extra="cell_batch")
        second = fits_table(["S1", "S2"], ["D1", "D2"], extra="plate")

        stacked = stack([first, second], ["FIRST", "SECOND"])

        assert stacked is not None
        assert isinstance(stacked.index, pd.RangeIndex)
        assert len(stacked) == len(first) + len(second)
        assert list(stacked.columns) == [*first.columns, "plate"]
        assert list(stacked[SAMPLE_COLUMN][:2]) == ["FIRST::S1", "FIRST::S1"]
        assert stacked[SAMPLE_COLUMN].iloc[-1] == "SECOND::S2"
        assert list(stacked[DRUG_COLUMN].unique()) == ["D1", "D2"]
        assert stacked["cell_batch"][len(first) :].isna().all()
        assert stacked["plate"][: len(first)].isna().all()
        assert first[SAMPLE_COLUMN].iloc[0] == "S1"

    def test_keeps_dtypes_and_categoricals_over_the_union_of_categories(self) -> None:
        first = fits_table(["S1"], ["D1"], extra="cell_batch")
        second = fits_table(["S2"], ["D2"])

        stacked = stack([first, second], ["FIRST", "SECOND"])

        assert stacked is not None
        for column in (SAMPLE_COLUMN, DRUG_COLUMN, "status", "cell_batch"):
            assert isinstance(stacked[column].dtype, pd.CategoricalDtype), column
        assert list(stacked[DRUG_COLUMN].cat.categories) == ["D1", "D2"]
        assert list(stacked[SAMPLE_COLUMN].cat.categories) == ["FIRST::S1", "SECOND::S2"]
        assert list(stacked["status"].cat.categories) == ["fitted", "fit_failed", "too_few_doses"]
        assert stacked["replicate"].dtype == np.int32
        assert stacked["n_doses"].dtype == np.int16
        assert stacked["pEC50"].dtype == np.float32

    def test_returns_none_if_any_input_lacks_the_table(self) -> None:
        assert stack([fits_table(["S1"], ["D1"]), None], ["FIRST", "SECOND"]) is None
