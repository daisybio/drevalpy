import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.loader import load_toy
from drevalpy.models.utils import (
    get_multiomics_feature_dataset,
    load_cl_ids_from_csv,
    load_drug_fingerprint_features,
    load_drug_ids_from_csv,
)


@pytest.fixture(scope="session")
def sample_dataset() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    path_data = "../data"
    drug_response = load_toy(path_data)
    cell_line_input = get_multiomics_feature_dataset(data_path=path_data, dataset_name="Toy_Data", gene_list=None)
    cell_line_ids = load_cl_ids_from_csv(path=path_data, dataset_name="Toy_Data")
    cell_line_input._add_features(cell_line_ids)
    # Load the drug features
    drug_ids = load_drug_ids_from_csv(data_path=path_data, dataset_name="Toy_Data")
    drug_input = load_drug_fingerprint_features(data_path=path_data, dataset_name="Toy_Data")
    drug_input._add_features(drug_ids)
    return drug_response, cell_line_input, drug_input
