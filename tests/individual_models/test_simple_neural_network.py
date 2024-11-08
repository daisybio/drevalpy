"""Test the SimpleNeuralNetwork model."""
import pytest

from drevalpy.evaluation import evaluate
from drevalpy.models import MODEL_FACTORY
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


@pytest.mark.parametrize("test_mode", ["LPO"])
@pytest.mark.parametrize("model_name", ["SRMF", "SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"])
def test_simple_neural_network(
        sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset],
        model_name: str,
        test_mode: str
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
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    train_dataset.remove_rows(indices=[list(range(len(train_dataset) - 1000))])  # smaller dataset for faster testing

    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    model = MODEL_FACTORY[model_name]()
    hpams = model.get_hyperparameter_set()
    hpam_combi = hpams[0]
    hpam_combi["units_per_layer"] = [2, 2]
    model.build_model(hyperparameters=hpam_combi)
    model.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        output_earlystopping=es_dataset,
    )

    val_es_dataset.predictions = model.predict(
        drug_ids=val_es_dataset.drug_ids,
        cell_line_ids=val_es_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )

    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    assert metrics["Pearson"] >= -1
