"""Per-sample lookup of one ``obs`` column."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data import synthetic
from drevalpy.types.data.dataset import Dataset


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    ds = synthetic(n_samples=2, n_drugs=2)
    ds.mdata.obs["study"] = ["A", "B"]
    ds.mdata.obs["score"] = [0.5, 1.5]
    return ds


class TestSampleColumn:
    def test_returns_values_in_the_order_of_the_requested_ids(self, dataset: Dataset) -> None:
        ids = dataset.sample_ids[::-1]

        assert dataset.sample_column("study", ids).tolist() == ["B", "A"]

    def test_unknown_ids_give_nan(self, dataset: Dataset) -> None:
        values = dataset.sample_column("study", np.array([dataset.sample_ids[0], "NOPE"]))

        assert values[0] == "A"
        assert np.isnan(values[1])

    def test_a_numeric_column_stays_numeric_with_unknown_ids(self, dataset: Dataset) -> None:
        values = dataset.sample_column("score", np.array([dataset.sample_ids[1], "NOPE"]))

        assert values.dtype == np.float64
        assert values[0] == 1.5
        assert np.isnan(values[1])

    def test_missing_column_names_the_available_ones(self, dataset: Dataset) -> None:
        with pytest.raises(KeyError, match=r"'nope'.*Available:.*tissue.*study"):
            dataset.sample_column("nope", dataset.sample_ids)

    def test_get_tissue_delegates_to_the_tissue_column(self, dataset: Dataset) -> None:
        ids = np.array([*dataset.sample_ids, "NOPE"])

        expected = dataset.sample_column("tissue", ids)
        actual = dataset.get_tissue(ids)

        assert actual[:-1].tolist() == expected[:-1].tolist()
        assert np.isnan(actual[-1])
