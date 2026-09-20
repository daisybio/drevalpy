"""Train/predict/round-trip gate for all single-drug models."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.types.data.dataset import Dataset
from tests.synthetic.variants import SUPPORTED_SINGLE_DRUG_MODELS, model_param

#: Extended tier: trains a separate model per drug, ~2.6s for six parametrizations.
pytestmark = pytest.mark.slow


def _resolve_single_drug_model_name(whole_name: str) -> str:
    if whole_name.startswith("SingleDrugRandomForest"):
        return "SingleDrugRandomForest"
    if whole_name.startswith("SingleDrugElasticNet"):
        return "SingleDrugElasticNet"
    return whole_name


def _construct_single_drug_spec(whole_name: str, model_name: str) -> ModelSpec:
    if whole_name.endswith("[proteomics]"):
        predictor_token = model_name.replace("SingleDrug", "singleDrug")
        return ModelSpec(model_name, from_spec(f"normalizedProteomics:{predictor_token}"))
    return ModelSpec.from_name(model_name)


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

    spec = _construct_single_drug_spec(whole_name, model_name)
    hpam_combi = dict(spec.default_hyperparameters)
    _configure_single_drug_hpam(model_name, hpam_combi)
    model = DRPModel(spec, hpam_combi)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.train(
            synthetic_dataset,
            split.train,
            early_stopping_scope=split.early_stopping_mask()[0] if spec.supports_early_stopping else None,
            model_checkpoint_dir=tmpdirname,
        )

    preds = model.predict(synthetic_dataset, split.test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] > 0

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = f"{model_dir}/model"
        model.save(checkpoint)
        loaded_model = DRPModel.load(checkpoint)
        assert isinstance(loaded_model, DRPModel)
        np.testing.assert_allclose(preds, loaded_model.predict(synthetic_dataset, split.test), rtol=1e-5, atol=1e-5)
