import numpy as np
import pytest
from suite.utils import leave_group_out_cv
from suite.dataset import DrugResponseDataset

# Mock dataset generation function
@pytest.fixture
def generate_mock_data():
    response = np.random.rand(100)
    cell_line_ids = np.random.randint(0, 10, size=100).astype(str)
    drug_ids = np.random.randint(20, 30, size=100).astype(str)
    return response, cell_line_ids, drug_ids

def test_leave_group_out_cv(mock_data):
    response, cell_line_ids, drug_ids = mock_data

    n_cv_splits = 5
    cv_sets = leave_group_out_cv(
        group="cell_line",
        n_cv_splits = n_cv_splits,
        response=response,
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        split_validation=True,
        validation_ratio=0.1,
        random_state=42,
    )

    # Check if the number of splits is correct
    assert len(cv_sets) == n_cv_splits

    for fold in cv_sets:
        train_dataset = fold["train"]
        test_dataset = fold["test"]
        validation_dataset = fold["validation"]

        # Check if train and test datasets are instances of DrugResponseDataset
        assert isinstance(train_dataset, DrugResponseDataset)
        assert isinstance(test_dataset, DrugResponseDataset)
        assert isinstance(validation_dataset, DrugResponseDataset)
        # Check if train and test datasets have the correct length
        assert len(train_dataset.response) + len(test_dataset.response) + len(validation_dataset.response) == len(response)

        # Check if train and test datasets have unique cell line/drug IDs
        assert len(np.intersect1d(train_dataset.cell_line_ids, test_dataset.cell_line_ids)) == 0
        assert len(np.intersect1d(validation_dataset.cell_line_ids, test_dataset.cell_line_ids)) == 0
        assert len(np.intersect1d(validation_dataset.cell_line_ids, train_dataset.cell_line_ids)) == 0

    cv_sets = leave_group_out_cv(
            group="cell_line",
            n_cv_splits = n_cv_splits,
            response=response,
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            split_validation=False,
            random_state=42,
        )
    for fold in cv_sets:
        train_dataset = fold["train"]
        test_dataset = fold["test"]

        # Check if train and test datasets are instances of DrugResponseDataset
        assert isinstance(train_dataset, DrugResponseDataset)
        assert isinstance(test_dataset, DrugResponseDataset)
        # Check if train and test datasets have the correct length
        assert len(train_dataset.response) + len(test_dataset.response) == len(response)

        # Check if train and test datasets have unique cell line/drug IDs
        assert len(np.intersect1d(train_dataset.cell_line_ids, test_dataset.cell_line_ids)) == 0
        assert "validation" not in fold
        
if __name__ == "__main__":
    pytest.main([__file__])
