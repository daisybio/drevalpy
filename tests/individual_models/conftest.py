import os
import pickle
import tempfile
import zipfile

import pytest
import requests

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.loader import load_toy
from drevalpy.models import SimpleNeuralNetwork

@pytest.fixture(scope="session")
def sample_dataset() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    seconds = 30
    url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
    # Fetch the latest record

    tmpdir = tempfile.TemporaryDirectory()

    drug_response = load_toy(tmpdir.name)
    cell_line_features = SimpleNeuralNetwork().load_cell_line_features(tmpdir.name, "Toy_Data")
    drug_features = SimpleNeuralNetwork().load_drug_features(tmpdir.name, "Toy_Data")
   
    return drug_response, cell_line_features, drug_features
