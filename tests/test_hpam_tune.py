"""test hpam_tune with Ray Tune."""

import importlib.util

import numpy as np

from drevalpy import experiment
from drevalpy.components.tuning.config import HPOConfig
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import construct_model


def test_hpam_tune(tmp_path, data_dir):
    """Test hpam_tune with a toy dataset and ElasticNet model.

    :param tmp_path: pytest temporary path fixture
    :param data_dir: path to the data directory
    """
    if importlib.util.find_spec("ray") is None:
        print("Ray is not installed, skipping test_hpam_tune.")
        return
    defaults = {
        "alpha": 1.0,
        "l1_ratio": 0.0,
    }

    model_cls = construct_model("ElasticNet")
    model = model_cls(defaults)
    cell_line_input = model.load_cell_line_features(dataset_name="TOYv1")
    drug_input = model.load_drug_features(dataset_name="TOYv1")

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

    best = experiment.hpam_tune(
        model_class=model_cls,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        early_stopping_dataset=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2, storage_path=str(tmp_path)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
