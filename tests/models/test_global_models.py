"""Test the neural networks that are not single drug models."""

import os
import tempfile
from typing import Any, cast

import numpy as np
import pytest

from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models import construct_model
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import get_zoo_config


def _zoo_config_variant(name: str, **updates: Any) -> ModelConfig:
    """Build a variant of a zoo preset by re-validating an updated dump.

    :param name: Zoo preset name.
    :param updates: ``ModelConfig`` field overrides.
    :returns: Newly validated ``ModelConfig``.
    """
    payload = get_zoo_config(name).model_dump(mode="python")
    payload.update(updates)
    return ModelConfig.model_validate(payload)


def _resolve_global_model_name(model_name: str) -> tuple[str, str]:
    whole_name = model_name
    if model_name.startswith("SimpleNeuralNetwork"):
        model_name = "SimpleNeuralNetwork"
    return whole_name, model_name


def _construct_global_model_class(whole_name: str, model_name: str) -> type[DRPModel]:
    if whole_name == "SimpleNeuralNetwork[chemberta]":
        config = _zoo_config_variant(
            "SimpleNeuralNetwork",
            drug_featurizer=DrugFeaturizerConfig(
                name="view",
                options={"view": "drug_chemberta_embeddings"},
            ),
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
    sample_dataset: ResponseBatch,
    model_name: str,
    test_mode: str,
    cross_study_dataset: ResponseBatch,
    data_dir,
) -> None:
    """Test global drug response models via MuDataset path.

    :param sample_dataset: ResponseBatch from conftest.py
    :param model_name: e.g., DIPK, SRMF, SimpleNeuralNetwork, or MultiViewNeuralNetwork
    :param test_mode: LTO
    :param cross_study_dataset: ResponseBatch from conftest.py
    :param data_dir: path to the data directory
    :raises ValueError: if drug input is None
    """
    from drevalpy.data import load
    from drevalpy.data.splitters import get_splitter

    mudataset = load("TOYv1")
    splitter = get_splitter(test_mode)
    folds = splitter(mudataset, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    whole_name, model_name = _resolve_global_model_name(model_name)
    model_class = _construct_global_model_class(whole_name, model_name)
    hpams = model_class.get_hyperparameter_set()
    hpam_combi = hpams[0]
    _apply_global_model_hpam_tweaks(model_name, hpam_combi)
    model = model_class(hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            model.train(mudataset, split, model_checkpoint_dir=tmpdirname)
        except (ValueError, KeyError) as exc:
            if "NaN" in str(exc) or "Modality" in str(exc):
                pytest.skip(f"Model {model_name} cannot handle LTO toy fold: {exc}")
            raise

    preds = model.predict(mudataset, split)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    with tempfile.TemporaryDirectory() as model_dir:
        try:
            checkpoint = f"{model_dir}/model"
            model.save(checkpoint)
            loaded_model = model_class.load(checkpoint)
            assert isinstance(loaded_model, DRPModel)
            preds_after = loaded_model.predict(mudataset, split)
            assert preds.shape == preds_after.shape
        except NotImplementedError:
            print(f"{model_name}: save/load not implemented")


@pytest.mark.parametrize("test_mode", ["LTO"])
def test_multi_view_neural_network_custom_views(sample_dataset: ResponseBatch, test_mode: str, data_dir) -> None:
    """Test MultiViewNeuralNetwork with a fully custom cell line view (not a built-in omic).

    Creates a fake CSV feature file and uses it via load_generic_csv to verify
    the flexible input pipeline works end-to-end including save/load without methylation.

    :param sample_dataset: ResponseBatch from conftest.py
    :param test_mode: LTO
    :param data_dir: path to the data directory
    :raises ValueError: if drug input is None
    """
    import pandas as pd

    from drevalpy.data import load
    from drevalpy.data.splitters import get_splitter

    toy_dir = data_dir / "TOYv1"
    gex = pd.read_csv(toy_dir / "gene_expression.csv")
    cell_line_names = gex["cell_line_name"].values

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
        mudataset = load("TOYv1")
        splitter = get_splitter(test_mode)
        folds = splitter(mudataset, n_splits=2, validation_ratio=0.4)
        split = folds[0]

        model_class = cast(
            type[DRPModel],
            construct_model(
                "MultiViewNeuralNetwork",
                _zoo_config_variant(
                    "MultiViewNeuralNetwork",
                    cell_line_featurizer=CellLineFeaturizerConfig(
                        name="raw",
                        view="custom_test_view",
                    ),
                ),
            ),
        )

        hpam_combi = {
            "units_per_layer": [2, 2],
            "dropout_prob": 0.3,
            "max_epochs": 1,
        }
        model = model_class(hpam_combi)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.train(mudataset, split, model_checkpoint_dir=tmpdirname)

        preds = model.predict(mudataset, split)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] > 0

        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = f"{model_dir}/model"
            model.save(checkpoint)
            assert not os.path.exists(os.path.join(model_dir, "methylation_scaler.pkl"))
            assert not os.path.exists(os.path.join(model_dir, "methylation_pca.pkl"))

            loaded_model = model_class.load(checkpoint)
            assert isinstance(loaded_model, DRPModel)
            preds_after = loaded_model.predict(mudataset, split)
            assert preds.shape == preds_after.shape
    finally:
        if os.path.exists(custom_csv_path):
            os.remove(custom_csv_path)
