"""Sample_dataset fixture for testing individual models."""

import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_ctrpv1, load_toy


@pytest.fixture(scope="session")
def sample_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = "../data"
    drug_response = load_toy(path_data)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session")
def ctrpv1_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = "../data"
    drug_response = load_ctrpv1(path_data)
    drug_response.remove_nan_responses()
    return drug_response
