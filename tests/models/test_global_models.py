"""Test the neural networks that are not single drug models."""

import os
import tempfile
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel


@pytest.mark.parametrize("test_mode", ["LTO"])
@pytest.mark.parametrize(
    "model_name",
    [
        "DrugGNN",
        "ChemBERTaNeuralNetwork",
        "SRMF",
        "DIPK",
        "SimpleNeuralNetwork",
        "MultiOmicsNeuralNetwork",
        "PharmaFormer",
    ],
)
def test_global_models(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
) -> None:
    """
    Test global drug response models.

    :param sample_dataset: from conftest.py
    :param model_name: e.g., DIPK, SRMF, SimpleNeuralNetwork, or MultiOmicsNeuralNetwork
    :param test_mode: LPO
    :param cross_study_dataset: from conftest.py
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(n_cv_splits=2, mode=test_mode, validation_ratio=0.4)
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]
    val_dataset = split["validation"]

    model = MODEL_FACTORY[model_name]()
    path_data = os.path.join("..", "data")
    cell_line_input = model.load_cell_line_features(data_path=path_data, dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=path_data, dataset_name="TOYv1")
    if drug_input is None:
        raise ValueError("Drug input is None")
    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    model_class = cast(type[DRPModel], MODEL_FACTORY[model_name])
    model = model_class()
    hpams = model.get_hyperparameter_set()
    hpam_combi = hpams[0]
    if model_name == "DIPK":
        hpam_combi["epochs"] = 1
        hpam_combi["epochs_autoencoder"] = 1
        hpam_combi["heads"] = 1
    elif model_name in ["SimpleNeuralNetwork", "MultiOmicsNeuralNetwork"]:
        hpam_combi["units_per_layer"] = [2, 2]
        hpam_combi["max_epochs"] = 1
    elif model_name == "PharmaFormer":
        hpam_combi["epochs"] = 1
        hpam_combi["patience"] = 2
    model.build_model(hyperparameters=hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        if model_name == "SRMF":
            # no early stopping
            model.train(
                output=train_dataset,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                output_earlystopping=None,
                model_checkpoint_dir=tmpdirname,
            )
        else:
            model.train(
                output=train_dataset,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                output_earlystopping=es_dataset,
                model_checkpoint_dir=tmpdirname,
            )
    if model_name == "DIPK":
        # test batch size = 1
        model.batch_size = 1  # type: ignore
    if model_name == "SRMF":
        # no early stopping
        prediction_dataset = val_dataset
    else:
        prediction_dataset = val_es_dataset
    prediction_dataset._predictions = model.predict(
        drug_ids=prediction_dataset.drug_ids,
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )
    # Save and load test (should either succeed or raise NotImplementedError)
    with tempfile.TemporaryDirectory() as model_dir:
        try:
            model.save(model_dir)
            loaded_model = model_class.load(model_dir)
            assert isinstance(loaded_model, DRPModel)

            preds_before = model.predict(
                drug_ids=prediction_dataset.drug_ids,
                cell_line_ids=prediction_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            preds_after = loaded_model.predict(
                drug_ids=prediction_dataset.drug_ids,
                cell_line_ids=prediction_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )

            assert preds_before.shape == preds_after.shape
            assert isinstance(preds_after, np.ndarray)
        except NotImplementedError:
            print(f"{model_name}: save/load not implemented")

    metrics = evaluate(prediction_dataset, metric=["Pearson"])
    print(f"Model: {model_name}, Pearson: {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running cross-study prediction for {model_name}")
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data=path_data,
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )


def test_pca_neural_network(
    sample_dataset: DrugResponseDataset,
    cross_study_dataset: DrugResponseDataset,
) -> None:
    """
    Test PCANeuralNetwork model.

    This test creates PCA features from gene expression data, then trains and evaluates
    the PCANeuralNetwork model.

    :param sample_dataset: from conftest.py
    :param cross_study_dataset: from conftest.py
    :raises ValueError: if drug input is None
    """
    test_mode = "LTO"
    model_name = "PCANeuralNetwork"
    n_components = 10  # Use small number for testing

    drug_response = sample_dataset
    drug_response.split_dataset(n_cv_splits=2, mode=test_mode, validation_ratio=0.4)
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_es_dataset = split["validation_es"]
    es_dataset = split["early_stopping"]

    path_data = os.path.join("..", "data")

    # Create PCA features from gene expression data
    ge_file = os.path.join(path_data, "TOYv1", "gene_expression.csv")
    ge_df = pd.read_csv(ge_file, index_col=CELL_LINE_IDENTIFIER)
    ge_df.index = ge_df.index.astype(str)
    if "cellosaurus_id" in ge_df.columns:
        ge_df = ge_df.drop(columns=["cellosaurus_id"])

    # Perform PCA
    scaler = StandardScaler()
    ge_scaled = scaler.fit_transform(ge_df.values)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(ge_scaled)

    # Save PCA features
    pca_df = pd.DataFrame(
        pca_features,
        index=ge_df.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    pca_df.index.name = CELL_LINE_IDENTIFIER
    pca_df = pca_df.reset_index()

    pca_output_file = os.path.join(path_data, "TOYv1", f"cell_line_gene_expression_pca_{n_components}.csv")
    pca_df.to_csv(pca_output_file, index=False)

    try:
        # Load model and features - need to build_model first to set n_components in hyperparameters
        model_class = cast(type[DRPModel], MODEL_FACTORY[model_name])
        model = model_class()
        hpams = model.get_hyperparameter_set()
        hpam_combi = hpams[0]
        hpam_combi["units_per_layer"] = [2, 2]
        hpam_combi["max_epochs"] = 1
        hpam_combi["n_components"] = n_components
        model.build_model(hyperparameters=hpam_combi)

        cell_line_input = model.load_cell_line_features(data_path=path_data, dataset_name="TOYv1")
        drug_input = model.load_drug_features(data_path=path_data, dataset_name="TOYv1")
        if drug_input is None:
            raise ValueError("Drug input is None")

        cell_lines_to_keep = cell_line_input.identifiers
        drugs_to_keep = drug_input.identifiers

        train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.train(
                output=train_dataset,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                output_earlystopping=es_dataset,
                model_checkpoint_dir=tmpdirname,
            )

        prediction_dataset = val_es_dataset
        prediction_dataset._predictions = model.predict(
            drug_ids=prediction_dataset.drug_ids,
            cell_line_ids=prediction_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )

        # Save and load test
        with tempfile.TemporaryDirectory() as model_dir:
            model.save(model_dir)
            loaded_model = model_class.load(model_dir)
            assert isinstance(loaded_model, DRPModel)

            preds_before = model.predict(
                drug_ids=prediction_dataset.drug_ids,
                cell_line_ids=prediction_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            preds_after = loaded_model.predict(
                drug_ids=prediction_dataset.drug_ids,
                cell_line_ids=prediction_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )

            assert preds_before.shape == preds_after.shape
            assert isinstance(preds_after, np.ndarray)

        metrics = evaluate(prediction_dataset, metric=["Pearson"])
        print(f"Model: {model_name}, Pearson: {metrics['Pearson']}")
        assert metrics["Pearson"] >= -1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running cross-study prediction for {model_name}")
            cross_study_prediction(
                dataset=cross_study_dataset,
                model=model,
                test_mode=test_mode,
                train_dataset=train_dataset,
                path_data=path_data,
                early_stopping_dataset=None,
                response_transformation=None,
                path_out=temp_dir,
                split_index=0,
                single_drug_id=None,
            )
    finally:
        # Clean up the generated PCA file
        if os.path.exists(pca_output_file):
            os.remove(pca_output_file)
