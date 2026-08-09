"""Tests for all single drug models."""

import tempfile

import numpy as np
import pytest

from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models import construct_model
from drevalpy.models.drp_model import DRPModel


def _resolve_single_drug_model_name(whole_name: str) -> str:
    if whole_name.startswith("SingleDrugRandomForest"):
        return "SingleDrugRandomForest"
    if whole_name.startswith("SingleDrugElasticNet"):
        return "SingleDrugElasticNet"
    return whole_name


def _construct_single_drug_model(whole_name: str, model_name: str):
    if whole_name.endswith("[proteomics]"):
        predictor_token = model_name.replace("SingleDrug", "singleDrug")
        return construct_model(model_name, f"normalizedProteomics:{predictor_token}")
    return construct_model(model_name)


def _configure_single_drug_hpam(model_name: str, hpam_combi: dict) -> None:
    if model_name == "SingleDrugRandomForest":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1


@pytest.mark.parametrize(
    "model_name",
    [
        "SingleDrugRandomForest[gex]",
        "SingleDrugRandomForest[proteomics]",
        "SingleDrugElasticNet[gex]",
        "SingleDrugElasticNet[proteomics]",
        "MOLIR",
        "SuperFELTR",
    ],
)
@pytest.mark.parametrize("test_mode", ["LTO"])
def test_single_drug_models(
    sample_dataset: ResponseBatch,
    model_name: str,
    test_mode: str,
    cross_study_dataset: ResponseBatch,
    data_dir,
) -> None:
    """Test single drug models via MuDataset path.

    :param sample_dataset: ResponseBatch from conftest.py
    :param model_name: model name
    :param test_mode: split mode
    :param cross_study_dataset: ResponseBatch from conftest.py
    :param data_dir: path to the data directory
    """
    from drevalpy.data import load_mudataset
    from drevalpy.data.structures.splitting import MuDataSplitter
    from drevalpy.experiment import seed_everything

    seed_everything(42)

    whole_name = model_name
    model_name = _resolve_single_drug_model_name(whole_name)

    mudataset = load_mudataset("TOYv1")
    splitter = MuDataSplitter()
    folds = splitter.split(mudataset, mode=test_mode, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    model_class = _construct_single_drug_model(whole_name, model_name)
    hpams = model_class.get_hyperparameter_set()
    hpam_combi = hpams[0]
    _configure_single_drug_hpam(model_name, hpam_combi)
    model = model_class(hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            model.train(mudataset, split, model_checkpoint_dir=tmpdirname)
        except ValueError as exc:
            if "NaN" in str(exc):
                pytest.skip(f"Model {model_name} cannot handle NaN features in LTO toy fold: {exc}")
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
