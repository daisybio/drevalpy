"""Test the MOLIR and SuperFELTR models."""

from typing import cast

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import evaluate, pearson
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel


@pytest.mark.parametrize("test_mode", ["LCO"])
@pytest.mark.parametrize("model_name", ["SuperFELTR", "MOLIR"])
def test_molir_superfeltr(
    sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset], model_name: str, test_mode: str
) -> None:
    """
    Test the MOLIR, SuperFELTR.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: LCO
    """
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    all_unique_drugs = np.unique(train_dataset.drug_ids)
    # randomly sample drugs to speed up testing
    np.random.seed(42)
    np.random.shuffle(all_unique_drugs)
    random_drug = all_unique_drugs[:1]
    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_pred_before = len(val_es_dataset)
    len_es_before = len(es_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val_es dataset from {len_pred_before} to {len(val_es_dataset)}")
    print(f"Reduced es dataset from {len_es_before} to {len(es_dataset)}")

    all_predictions = np.zeros_like(val_es_dataset.drug_ids, dtype=float)
    model_class = cast(type[DRPModel], MODEL_FACTORY[model_name])
    model = model_class()
    hpam_combi = model.get_hyperparameter_set()[0]
    hpam_combi["epochs"] = 1
    model.build_model(hpam_combi)

    output_mask = train_dataset.drug_ids == random_drug
    drug_train = train_dataset.copy()
    drug_train.mask(output_mask)
    es_mask = es_dataset.drug_ids == random_drug
    es_dataset_drug = es_dataset.copy()
    es_dataset_drug.mask(es_mask)
    # smaller dataset for faster testing
    drug_train.remove_rows(indices=np.array([list(range(len(drug_train) - 100))]))
    model.train(
        output=drug_train,
        cell_line_input=cell_line_input,
        drug_input=None,
        output_earlystopping=es_dataset_drug,
    )

    val_mask = val_es_dataset.drug_ids == random_drug
    all_predictions[val_mask] = model.predict(
        drug_ids=random_drug,
        cell_line_ids=val_es_dataset.cell_line_ids[val_mask],
        cell_line_input=cell_line_input,
    )
    pcc_drug = pearson(val_es_dataset.response[val_mask], all_predictions[val_mask])
    assert pcc_drug >= -1

    # subset the dataset to only the drugs that were used
    val_es_mask = np.isin(val_es_dataset.drug_ids, random_drug)
    val_es_dataset._cell_line_ids = val_es_dataset.cell_line_ids[val_es_mask]
    val_es_dataset._drug_ids = val_es_dataset.drug_ids[val_es_mask]
    val_es_dataset._response = val_es_dataset.response[val_es_mask]
    val_es_dataset._predictions = all_predictions[val_es_mask]
    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    print(f"{test_mode}: Collapsed performance of {model_name}: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1.0


@pytest.mark.parametrize("test_mode", ["LCO"])
@pytest.mark.parametrize("model_name", ["DIPK"])
def test_dipk(
    sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset], model_name: str, test_mode: str
) -> None:
    """Test the DIPK model.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: LCO
    """
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_es_dataset = split["validation_es"]
    model = MODEL_FACTORY[model_name]()
    hpam_combi = model.get_hyperparameter_set()[0]
    hpam_combi["epochs"] = 1
    hpam_combi["epochs_autoencoder"] = 1
    model.build_model(hpam_combi)
    drug_input = model.load_drug_features(data_path="../data", dataset_name="Toy_Data")  # type: ignore
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="Toy_Data")

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    model.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        output_earlystopping=val_es_dataset,
    )
    out = model.predict(
        cell_line_ids=val_es_dataset.cell_line_ids,
        drug_ids=val_es_dataset.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    val_es_dataset._predictions = out
    metrics = evaluate(val_es_dataset, metric=["Pearson"])
    assert metrics["Pearson"] >= -1.0
