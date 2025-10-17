"""Sample_dataset fixture for testing individual models."""

import os

import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_toyv1, load_toyv2


@pytest.fixture(scope="session")
def sample_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = os.path.join("..", "data")
    drug_response = load_toyv1(path_data)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session")
def cross_study_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = "../data"
    drug_response = load_toyv2(path_data)
    drug_response.remove_nan_responses()
    return drug_response
