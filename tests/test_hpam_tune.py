"""test hpam tune."""

import numpy as np

from drevalpy import experiment
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY


def test_hpam_tune(tmp_path):
    """
    Test hpam_tune with a toy dataset and ElasticNet model.

    :param tmp_path: pytest temporary path fixture
    """
    hpam_set = [
        {"alpha": 1.0, "l1_ratio": 0.0},
        {"alpha": 2.5, "l1_ratio": 0.5},
        {"alpha": 5.0, "l1_ratio": 1.0},
    ]

    model = MODEL_FACTORY["ElasticNet"]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")

    valid_cell_lines = list(cell_line_input.identifiers)[:2]
    valid_drugs = list(drug_input.identifiers)[:2]
    responses = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    cell_line_ids = np.array([valid_cell_lines[0], valid_cell_lines[0], valid_cell_lines[1], valid_cell_lines[1]])
    drug_ids = np.array([valid_drugs[0], valid_drugs[1], valid_drugs[0], valid_drugs[1]])
    train_dataset = DrugResponseDataset(
        response=responses,
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        dataset_name="TOYv1",
    )
    val_dataset = DrugResponseDataset(
        response=responses.copy(),
        cell_line_ids=cell_line_ids.copy(),
        drug_ids=drug_ids.copy(),
        dataset_name="TOYv1",
    )

    model = MODEL_FACTORY["ElasticNet"]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_val_before = len(val_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val dataset from {len_val_before} to {len(val_dataset)}")

    best = experiment.hpam_tune(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        early_stopping_dataset=None,
        hpam_set=hpam_set,
        response_transformation=None,
        metric="RMSE",
        path_data="../data",
        model_checkpoint_dir="TEMPORARY",
    )

    assert best in hpam_set
