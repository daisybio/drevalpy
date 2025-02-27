"""Test the SimpleNeuralNetwork model."""

import tempfile
from typing import cast

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import evaluate
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel


@pytest.mark.parametrize("test_mode", ["LPO"])
@pytest.mark.parametrize("model_name", ["SRMF", "SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"])
def test_simple_neural_network(
    sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset], model_name: str, test_mode: str
) -> None:
    """
    Test the SimpleNeuralNetwork model.

    :param sample_dataset: from conftest.py
    :param model_name: either SRMF, SimpleNeuralNetwork, or MultiOmicsNeuralNetwork
    :param test_mode: LPO
    """
    drug_response, cell_line_input, drug_input = sample_dataset
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
