"""Train/predict/round-trip gate for all single-drug models."""

from __future__ import annotations

import tempfile
from typing import cast

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.drp_model import DRPModel
from drevalpy.types.data.dataset import Dataset
from tests.synthetic.variants import SUPPORTED_SINGLE_DRUG_MODELS, model_param


def _resolve_single_drug_model_name(whole_name: str) -> str:
    if whole_name.startswith("SingleDrugRandomForest"):
        return "SingleDrugRandomForest"
    if whole_name.startswith("SingleDrugElasticNet"):
        return "SingleDrugElasticNet"
    return whole_name


def _construct_single_drug_model(whole_name: str, model_name: str) -> type[DRPModel]:
    if whole_name.endswith("[proteomics]"):
        predictor_token = model_name.replace("SingleDrug", "singleDrug")
        return cast(type[DRPModel], construct_model(model_name, f"normalizedProteomics:{predictor_token}"))
    return cast(type[DRPModel], construct_model(model_name))


def _configure_single_drug_hpam(model_name: str, hpam_combi: dict) -> None:
    """Shrink the model to the smallest configuration that still exercises it."""
    if model_name == "SingleDrugRandomForest":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1


@pytest.mark.parametrize("model_name", [model_param(name) for name in SUPPORTED_SINGLE_DRUG_MODELS])
def test_single_drug_models(synthetic_dataset: Dataset, model_name: str) -> None:
    """Each single-drug model trains, predicts and reloads on a Leave-Tissue-Out fold.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    :param model_name: Model name, possibly with a ``[view]`` suffix.
    """
    from drevalpy.registry.splitter import get as get_splitter
    from drevalpy.utils.seed import seed_everything

    seed_everything(42)

    whole_name = model_name
    model_name = _resolve_single_drug_model_name(whole_name)

    split = get_splitter("LTO")(synthetic_dataset, n_splits=2, validation_ratio=0.4)[0]

    model_class = _construct_single_drug_model(whole_name, model_name)
    hpam_combi = dict(model_class.get_hyperparameter_set()[0])
    _configure_single_drug_hpam(model_name, hpam_combi)
    model = model_class(hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(synthetic_dataset, split, model_checkpoint_dir=tmpdirname)

    preds = model.predict(synthetic_dataset, split)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded_model = model_class.load(checkpoint)
        assert isinstance(loaded_model, DRPModel)
        assert preds.shape == loaded_model.predict(synthetic_dataset, split).shape
