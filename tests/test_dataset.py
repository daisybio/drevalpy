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
# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
