"""Test the neural networks that are not single drug models."""

import os
import tempfile
from typing import cast

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import construct_model
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import get_zoo_config


def _resolve_global_model_name(model_name: str) -> tuple[str, str]:
    whole_name = model_name
    if model_name.startswith("SimpleNeuralNetwork"):
        model_name = "SimpleNeuralNetwork"
    return whole_name, model_name


def _construct_global_model_class(whole_name: str, model_name: str) -> type[DRPModel]:
    if whole_name == "SimpleNeuralNetwork[chemberta]":
        config = get_zoo_config("SimpleNeuralNetwork").model_copy(
            update={
                "drug_featurizer": DrugFeaturizerConfig(
                    name="view",
                    hyperparameters={"view": "drug_chemberta_embeddings"},
                ),
            },
        )
        return cast(type[DRPModel], construct_model(model_name, config))
    return cast(type[DRPModel], construct_model(model_name))


def _apply_global_model_hpam_tweaks(model_name: str, hpam_combi: dict) -> None:
    if model_name == "DIPK":
        hpam_combi["batch_size"] = 1
        hpam_combi["epochs"] = 1
        hpam_combi["epochs_autoencoder"] = 1
        hpam_combi["heads"] = 1
    elif model_name in ["SimpleNeuralNetwork", "MultiViewNeuralNetwork"]:
        hpam_combi["units_per_layer"] = [2, 2]
        hpam_combi["max_epochs"] = 1
    elif model_name == "PharmaFormer":
        hpam_combi["epochs"] = 1
        hpam_combi["patience"] = 2
    elif model_name in {"Precily", "SparseGO"}:
        hpam_combi["epochs"] = 1
        hpam_combi["batch_size"] = 32
    elif model_name == "AdaBoostDecisionTree":
        hpam_combi["max_depth"] = 2
        hpam_combi["min_samples_split"] = 2
        hpam_combi["min_samples_leaf"] = 2
        hpam_combi["n_estimators"] = 2


def _reduce_global_model_datasets(
    train_dataset: DrugResponseDataset,
    val_es_dataset: DrugResponseDataset,
    es_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    cell_line_input,
    drug_input,
) -> None:
    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers
    for dataset in (train_dataset, val_es_dataset, es_dataset, val_dataset):
        dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)


def _train_global_model(
    model,
    model_name: str,
    train_dataset: DrugResponseDataset,
    cell_line_input,
    drug_input,
    es_dataset: DrugResponseDataset,
    tmpdirname: str,
) -> None:
    if model_name == "SRMF":
        model.train(
            output=train_dataset,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=None,
            model_checkpoint_dir=tmpdirname,
        )
        return
    model.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        output_earlystopping=es_dataset,
        model_checkpoint_dir=tmpdirname,
    )


def _assert_global_model_save_load_roundtrip(
    model,
    model_class: type[DRPModel],
    model_name: str,
    prediction_dataset: DrugResponseDataset,
    drug_input,
    cell_line_input,
    data_dir,
) -> None:
    with tempfile.TemporaryDirectory() as model_dir:
        try:
            checkpoint = f"{model_dir}/model"
            model.save(checkpoint)
            loaded_model = model_class.load(checkpoint)
            if model_name == "SparseGO":
                loaded_model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
            assert isinstance(loaded_model, DRPModel)

            preds_after = loaded_model.predict(
                drug_ids=prediction_dataset.drug_ids,
                cell_line_ids=prediction_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )

            assert prediction_dataset._predictions is not None
            assert prediction_dataset._predictions.shape == preds_after.shape
            assert isinstance(preds_after, np.ndarray)
        except NotImplementedError:
            print(f"{model_name}: save/load not implemented")


@pytest.mark.parametrize("test_mode", ["LTO"])
@pytest.mark.parametrize(
    "model_name",
    [
        "DrugGNN",
        "SRMF",
        "DIPK",
        "SimpleNeuralNetwork[fingerprints]",
        "SimpleNeuralNetwork[chemberta]",
        "MultiViewNeuralNetwork",
        "PharmaFormer",
        "Precily",
        "SparseGO",
    ],
)
def test_global_models(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
    data_dir,
) -> None:
    """Test global drug response models.

    :param sample_dataset: from conftest.py
    :param model_name: e.g., DIPK, SRMF, SimpleNeuralNetwork, or MultiViewNeuralNetwork
    :param test_mode: LPO
    :param cross_study_dataset: from conftest.py
    :param data_dir: path to the data directory
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

    whole_name, model_name = _resolve_global_model_name(model_name)

    model_class = _construct_global_model_class(whole_name, model_name)
    hpams = model_class.get_hyperparameter_set()
    hpam_combi = hpams[0]
    _apply_global_model_hpam_tweaks(model_name, hpam_combi)
    model = model_class(hpam_combi)

    cell_line_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")
    if drug_input is None:
        raise ValueError("Drug input is None")

    _reduce_global_model_datasets(train_dataset, val_es_dataset, es_dataset, val_dataset, cell_line_input, drug_input)

    with tempfile.TemporaryDirectory() as tmpdirname:
        _train_global_model(model, model_name, train_dataset, cell_line_input, drug_input, es_dataset, tmpdirname)
    prediction_dataset = val_dataset if model_name == "SRMF" else val_es_dataset
    prediction_dataset._predictions = model.predict(
        drug_ids=prediction_dataset.drug_ids,
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )
    _assert_global_model_save_load_roundtrip(
        model, model_class, model_name, prediction_dataset, drug_input, cell_line_input, data_dir
    )

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
            path_data=str(data_dir),
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )


@pytest.mark.parametrize("test_mode", ["LTO"])
def test_multi_view_neural_network_custom_views(sample_dataset: DrugResponseDataset, test_mode: str, data_dir) -> None:
    """Test MultiViewNeuralNetwork with a fully custom cell line view (not a built-in omic).

    Creates a fake CSV feature file and uses it via load_generic_csv to verify
    the flexible input pipeline works end-to-end including save/load without methylation.

    :param sample_dataset: from conftest.py
    :param test_mode: LTO
    :param data_dir: path to the data directory
    :raises ValueError: if drug input is None
    """
    import pandas as pd

    path_data = data_dir
    toy_dir = data_dir / "TOYv1"

    # Read existing cell line names from gene_expression.csv (cell_line_name is the index used by the loader)
    gex = pd.read_csv(toy_dir / "gene_expression.csv")
    cell_line_names = gex["cell_line_name"].values

    # Create a fake custom feature CSV with random data, matching the real CSV format
    rng = np.random.default_rng(42)
    n_features = 10
    custom_df = pd.DataFrame(
        rng.standard_normal((len(cell_line_names), n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    custom_df.insert(0, "cell_line_name", cell_line_names)
    custom_csv_path = toy_dir / "custom_test_view.csv"
    custom_df.to_csv(custom_csv_path, index=False)

    try:
        drug_response = sample_dataset
        drug_response.split_dataset(n_cv_splits=2, mode=test_mode, validation_ratio=0.4)
        assert drug_response.cv_splits is not None
        split = drug_response.cv_splits[0]
        train_dataset = split["train"]
        es_dataset = split["early_stopping"]
        val_es_dataset = split["validation_es"]

        model_class = cast(
            type[DRPModel],
            construct_model(
                "MultiViewNeuralNetwork",
                get_zoo_config("MultiViewNeuralNetwork").model_copy(
                    update={
                        "cell_line_featurizer": CellLineFeaturizerConfig(
                            name="raw",
                            view="custom_test_view",
                            hyperparameters={},
                        ),
                    },
                ),
            ),
        )

        hpam_combi = {
            "units_per_layer": [2, 2],
            "dropout_prob": 0.3,
            "max_epochs": 1,
        }
        model = model_class(hpam_combi)

        cell_line_input = model.load_cell_line_features(data_path=str(path_data), dataset_name="TOYv1")
        drug_input = model.load_drug_features(data_path=str(path_data), dataset_name="TOYv1")
        if drug_input is None:
            raise ValueError("Drug input is None")

        cell_lines_to_keep = cell_line_input.identifiers
        drugs_to_keep = drug_input.identifiers
        train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        val_es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.train(
                output=train_dataset,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                output_earlystopping=es_dataset,
                model_checkpoint_dir=tmpdirname,
            )

        preds = model.predict(
            drug_ids=val_es_dataset.drug_ids,
            cell_line_ids=val_es_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(val_es_dataset)

        # Save and load roundtrip — no methylation files should be required
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = f"{model_dir}/model"
            model.save(checkpoint)
            # Verify no methylation files were saved
            assert not os.path.exists(os.path.join(model_dir, "methylation_scaler.pkl"))
            assert not os.path.exists(os.path.join(model_dir, "methylation_pca.pkl"))

            loaded_model = model_class.load(checkpoint)
            assert isinstance(loaded_model, DRPModel)

            preds_after = loaded_model.predict(
                drug_ids=val_es_dataset.drug_ids,
                cell_line_ids=val_es_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            assert preds.shape == preds_after.shape
    finally:
        # Clean up the fake CSV
        if os.path.exists(custom_csv_path):
            os.remove(custom_csv_path)
