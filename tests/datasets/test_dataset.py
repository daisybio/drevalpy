"""Tests for non-mutating DrugResponseDataset helpers and hashability."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset


def _toy_dataset(
    *,
    with_predictions: bool = False,
    with_tissues: bool = True,
) -> DrugResponseDataset:
    return DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        cell_line_ids=np.array(["CL-1", "CL-2", "CL-3", "CL-4"]),
        drug_ids=np.array(["D-1", "D-2", "D-1", "D-2"]),
        tissues=np.array(["T1", "T2", "T1", "T2"]) if with_tissues else None,
        predictions=np.array([1.1, 2.1, 3.1, 4.1]) if with_predictions else None,
        dataset_name="toy",
    )


def test_drug_response_dataset_is_unhashable() -> None:
    dataset = _toy_dataset()
    assert DrugResponseDataset.__hash__ is None
    with pytest.raises(TypeError):
        hash(dataset)
    with pytest.raises(TypeError):
        _ = {dataset: "value"}


def test_with_rows_added_returns_new_dataset() -> None:
    left = _toy_dataset()
    right = DrugResponseDataset(
        response=np.array([5.0]),
        cell_line_ids=np.array(["CL-5"]),
        drug_ids=np.array(["D-3"]),
        tissues=np.array(["T3"]),
        dataset_name="toy",
    )
    original_len = len(left)

    merged = left.with_rows_added(right)

    assert merged is not left
    assert len(left) == original_len
    assert len(merged) == original_len + 1
    assert np.array_equal(merged.response, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert merged.cell_line_ids[-1] == "CL-5"


def test_shuffled_returns_new_dataset_and_preserves_original() -> None:
    dataset = _toy_dataset()
    original_response = dataset.response.copy()

    shuffled = dataset.shuffled(random_state=42)

    assert shuffled is not dataset
    assert np.array_equal(dataset.response, original_response)
    assert len(shuffled) == len(dataset)
    assert not np.array_equal(shuffled.response, original_response)
    assert set(shuffled.response) == set(original_response)


def test_masked_returns_new_dataset() -> None:
    dataset = _toy_dataset()
    mask = np.array([True, False, True, False])

    masked = dataset.masked(mask)

    assert masked is not dataset
    assert len(dataset) == 4
    assert np.array_equal(masked.response, np.array([1.0, 3.0]))
    assert np.array_equal(masked.cell_line_ids, np.array(["CL-1", "CL-3"]))


def test_reduced_to_returns_new_dataset() -> None:
    dataset = _toy_dataset()

    reduced = dataset.reduced_to(cell_line_ids=np.array(["CL-1", "CL-3"]), drug_ids=np.array(["D-1"]))

    assert reduced is not dataset
    assert len(dataset) == 4
    assert len(reduced) == 2
    assert set(reduced.cell_line_ids) <= {"CL-1", "CL-3"}
    assert set(reduced.drug_ids) == {"D-1"}


def test_transformed_and_fit_transformed_return_new_datasets() -> None:
    dataset = _toy_dataset(with_predictions=True)
    original_response = dataset.response.copy()
    scaler = StandardScaler()

    fit_transformed = dataset.fit_transformed(scaler)

    assert fit_transformed is not dataset
    assert np.array_equal(dataset.response, original_response)
    assert np.isclose(fit_transformed.response.mean(), 0.0, atol=1e-8)

    transformed = dataset.transformed(scaler)
    assert transformed is not dataset
    assert np.allclose(transformed.response, fit_transformed.response)


def test_mutating_methods_still_work_for_compatibility() -> None:
    dataset = _toy_dataset()
    other = DrugResponseDataset(
        response=np.array([9.0]),
        cell_line_ids=np.array(["CL-9"]),
        drug_ids=np.array(["D-9"]),
        tissues=np.array(["T9"]),
    )

    dataset.add_rows(other)
    assert len(dataset) == 5

    dataset.shuffle(random_state=0)
    assert len(dataset) == 5

    dataset.reduce_to(drug_ids=np.array(["D-1", "D-2"]))
    assert set(dataset.drug_ids) <= {"D-1", "D-2"}

    dataset.mask(np.ones(len(dataset), dtype=bool))
    assert len(dataset) >= 1
