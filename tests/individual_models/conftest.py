
import tempfile

import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.loader import load_toy
from drevalpy.models import SimpleNeuralNetwork

@pytest.fixture(scope="session")
def sample_dataset() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    path_data = "../data"
    drug_response = load_toy(path_data)
    cell_line_features = SimpleNeuralNetwork().load_cell_line_features(path_data, "Toy_Data")
    drug_features = SimpleNeuralNetwork().load_drug_features(path_data, "Toy_Data")
    return drug_response, cell_line_features, drug_features
