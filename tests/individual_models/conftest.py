import os
import pickle
import tempfile
import zipfile

import pytest
import requests

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


@pytest.fixture(scope="session")
def sample_dataset() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    seconds = 30
    url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
    # Fetch the latest record
    response = requests.get(url, timeout=seconds)
    latest_url = response.links["linkset"]["url"]
    response = requests.get(latest_url, timeout=seconds)
    data = response.json()
    name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
    tmpdir = tempfile.TemporaryDirectory()
    toy_data_url = name_to_url["Toy_Data.zip"]
    response = requests.get(toy_data_url, timeout=seconds)
    file_path = os.path.join(tmpdir.name, "Toy_Data.zip")

    print(f"Loading Toy Dataset from Zenodo, from {data['created']}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir.name)

    with open(os.path.join(tmpdir.name, "Toy_Data", "toy_data_drp_dataset.pkl"), "rb") as f:
        drug_response = pickle.load(f)
    with open(os.path.join(tmpdir.name, "Toy_Data", "toy_data_cl_features.pkl"), "rb") as f:
        cell_line_features = pickle.load(f)
    with open(
        os.path.join(tmpdir.name, "Toy_Data", "toy_data_drug_features.pkl"),
        "rb",
    ) as f:
        drug_features = pickle.load(f)
    return drug_response, cell_line_features, drug_features
