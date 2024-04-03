import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from suite.dataset import DrugResponseDataset
import os


# Test if the dataset loads correctly from CSV files
def test_drug_response_dataset_load():
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

def test_drug_response_dataset_add_rows():
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

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
