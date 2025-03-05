"""Test the SimpleNeuralNetwork model."""

import tempfile
from typing import cast

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel


@pytest.mark.parametrize("test_mode", ["LPO"])
@pytest.mark.parametrize("model_name", ["SRMF", "SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"])
def test_simple_neural_network(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
) -> None:
    """
    Test the SimpleNeuralNetwork model.

    :param sample_dataset: from conftest.py
    :param model_name: either SRMF, SimpleNeuralNetwork, or MultiOmicsNeuralNetwork
    :param test_mode: LPO
    :param cross_study_dataset: from conftest.py
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    # smaller dataset for faster testing
    train_dataset.remove_rows(indices=np.array([list(range(len(train_dataset) - 1000))]))

    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]

    model = MODEL_FACTORY[model_name]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")
    if drug_input is None:
        raise ValueError("Drug input is None")
    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    model_class = cast(type[DRPModel], MODEL_FACTORY[model_name])
    model = model_class()
    hpams = model.get_hyperparameter_set()
    hpam_combi = hpams[0]
    hpam_combi["units_per_layer"] = [2, 2]
    hpam_combi["max_epochs"] = 1
    model.build_model(hyperparameters=hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(
            output=train_dataset,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=es_dataset,
            model_checkpoint_dir=tmpdirname,
        )

    val_es_dataset._predictions = model.predict(
        drug_ids=val_es_dataset.drug_ids,
        cell_line_ids=val_es_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )

    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    assert metrics["Pearson"] >= -1

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running cross-study prediction for {model_name}")
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data="../data",
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )
