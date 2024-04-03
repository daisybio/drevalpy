import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from suite.dataset import DrugResponseDataset
import os


# Test if the dataset loads correctly from CSV files
def test_response_dataset_load():
    # Create a temporary CSV file with mock data
    data = {
        "cell_line_id": [1, 2, 3],
        "drug_id": ["A", "B", "C"],
        "response": [0.1, 0.2, 0.3],
    }
    dataset = DrugResponseDataset(cell_line_ids=data["cell_line_id"], drug_ids=data["drug_id"], response=data["response"])
    dataset.save("dataset.csv")
    del dataset
    # Load the dataset
    dataset = DrugResponseDataset()
    dataset.load("dataset.csv")

    os.remove("dataset.csv")

    # Check if the dataset loaded correctly
    assert np.array_equal(dataset.cell_line_ids, data["cell_line_id"])
    assert np.array_equal(dataset.drug_ids, data["drug_id"])
    assert np.allclose(dataset.response, data["response"])

def test_response_dataset_add_rows():
    dataset1 = DrugResponseDataset(
        response=np.array([1, 2, 3]),
        cell_line_ids=np.array([101, 102, 103]),
        drug_ids=np.array(["A", "B", "C"]),
    )
    dataset2 = DrugResponseDataset(
        response=np.array([4, 5, 6]),
        cell_line_ids=np.array([104, 105, 106]),
        drug_ids=np.array(["D", "E", "F"]),
    )
    dataset1.add_rows(dataset2)

    assert np.array_equal(dataset1.response, np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(
        dataset1.cell_line_ids, np.array([101, 102, 103, 104, 105, 106])
    )
    assert np.array_equal(
        dataset1.drug_ids, np.array(["A", "B", "C", "D", "E", "F"])
    )

def test_response_dataset_shuffle():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5, 6]),
        cell_line_ids=np.array([101, 102, 103, 104, 105, 106]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
    )

    # Shuffle the dataset
    dataset.shuffle(random_state=42)

    # Check if the length remains the same
    assert len(dataset.response) == 6
    assert len(dataset.cell_line_ids) == 6
    assert len(dataset.drug_ids) == 6

    # Check if the response, cell_line_ids, and drug_ids arrays are shuffled
    assert not np.array_equal(dataset.response, np.array([1, 2, 3, 4, 5]))
    assert not np.array_equal(
        dataset.cell_line_ids, np.array([101, 102, 103, 104, 105])
    )
    assert not np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "D", "E"]))

def test_response_data_remove_drugs_and_cell_lines():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )

    # Remove specific drugs and cell lines
    dataset.remove_drugs(["A", "C"])
    dataset.remove_cell_lines([101, 103])

    # Check if the removed drugs and cell lines are not present in the dataset
    assert "A" not in dataset.drug_ids
    assert "C" not in dataset.drug_ids
    assert 101 not in dataset.cell_line_ids
    assert 103 not in dataset.cell_line_ids

    # Check if the length of response, cell_line_ids, and drug_ids arrays is reduced accordingly
    assert len(dataset.response) == 3
    assert len(dataset.cell_line_ids) == 3
    assert len(dataset.drug_ids) == 3

def test_response_dataset_reduce_to():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )

    # Reduce the dataset to a subset of cell line IDs and drug IDs
    dataset.reduce_to(cell_line_ids=[102, 104], drug_ids=["B", "D"])

    # Check if only the rows corresponding to the specified cell line IDs and drug IDs remain
    assert all(cell_line_id in [102, 104] for cell_line_id in dataset.cell_line_ids)
    assert all(drug_id in ["B", "D"] for drug_id in dataset.drug_ids)

    # Check if the length of response, cell_line_ids, and drug_ids arrays is reduced accordingly
    assert len(dataset.response) == 2
    assert len(dataset.cell_line_ids) == 2
    assert len(dataset.drug_ids) == 2

@pytest.mark.parametrize("mode", ["LPO", "LCO", "LDO"])
@pytest.mark.parametrize("split_validation", [True, False])
def test_split_response_dataset(mode, split_validation):
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=[1, 2, 3, 4, 5],
        cell_line_ids=[101, 102, 103, 104, 105],
        drug_ids=["A", "B", "C", "D", "E"],
    )

    # Test splitting the dataset with the specified mode and validation split
    cv_splits = dataset.split_dataset(n_cv_splits=5, mode=mode, split_validation=split_validation, validation_ratio=0.1, random_state=42)
    assert isinstance(cv_splits, list)
    assert len(cv_splits) == 5  # Check if the correct number of splits is returned
    for split in cv_splits:
        assert isinstance(split["train"], DrugResponseDataset)
        assert isinstance(split["test"], DrugResponseDataset)

        # Check that drugs/cell lines in the training data are not present in the test data
        if mode == "LCO":
            train_cell_lines = set(split["train"].cell_line_ids)
            test_cell_lines = set(split["test"].cell_line_ids)

            assert train_cell_lines.isdisjoint(test_cell_lines)

            if split_validation:  # Only check if validation split is enabled
                validation_cell_lines = set(split["validation"].cell_line_ids)
                assert validation_cell_lines.isdisjoint(test_cell_lines)  # Check for disjointness between validation and test cell lines

        elif mode == "LDO":
            train_drugs = set(split["train"].drug_ids)
            test_drugs = set(split["test"].drug_ids)

            assert train_drugs.isdisjoint(test_drugs)

            if split_validation:  # Only check if validation split is enabled
                validation_drugs = set(split["validation"].drug_ids)
                assert validation_drugs.isdisjoint(test_drugs)  # Check for disjointness between validation and test drugs

        elif mode == "LPO":
            train_pairs = set(zip(split["train"].cell_line_ids, split["train"].drug_ids))
            test_pairs = set(zip(split["test"].cell_line_ids, split["test"].drug_ids))

            assert train_pairs.isdisjoint(test_pairs)

            if split_validation:  # Only check if validation split is enabled
                validation_pairs = set(zip(split["validation"].cell_line_ids, split["validation"].drug_ids))
                assert validation_pairs.isdisjoint(test_pairs)  # Check for disjointness between validation and test pairs

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
