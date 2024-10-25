import os
import pickle
import tempfile
import zipfile

import pytest
import requests
from typing import Tuple

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.toy import Toy
from drevalpy.models.utils import get_multiomics_feature_dataset, load_drug_fingerprint_features


@pytest.fixture(scope="session")
def sample_dataset() -> Tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
    # Fetch the latest record
    response = requests.get(url)
    latest_url = response.links["linkset"]["url"]
    response = requests.get(latest_url)
    data = response.json()
    name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
    tmpdir = tempfile.TemporaryDirectory()
    dir_name = tmpdir.name
    toy_data_url = name_to_url["Toy_Data.zip"]
    response = requests.get(toy_data_url)
    file_path = os.path.join(dir_name, "Toy_Data.zip")

    print(f"Loading Toy Dataset from Zenodo, from {data['created']}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dir_name)

    # Load the Toy dataset
    toy_dataset = Toy(path_data=dir_name)
    # Load the cell line features
    cell_line_input = get_multiomics_feature_dataset(
        data_path=dir_name, dataset_name="Toy_Data", gene_list=None
    )
    # Load the drug features
    drug_input = load_drug_fingerprint_features(data_path=dir_name, dataset_name="Toy_Data")
    return toy_dataset, cell_line_input, drug_input

