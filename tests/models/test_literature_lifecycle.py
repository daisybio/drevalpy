"""Smoke tests for literature models routed through the native facade."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset
from tests.models.synthetic_fixtures import lso_split_masks, small_data_spec, synthetic_mudataset


def _dataset(*, extra_views: tuple[str, ...] = ()) -> tuple[Dataset, SplitMasks]:
    dataset = synthetic_mudataset(n_features_per_view=4, fingerprint_width=4, extra_views=extra_views)
    return dataset, lso_split_masks()


_SMALL_HYPERPARAMETERS = {
    "DIPK": {"epochs": 1, "epochs_autoencoder": 1, "heads": 1},
    "SimpleNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    "MultiViewNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    "PharmaFormer": {"epochs": 1, "patience": 2},
    "Precily": {"epochs": 1, "batch_size": 32},
    "SparseGO": {"epochs": 1, "batch_size": 32},
}


@pytest.mark.parametrize(
    "model_name",
    [
        "DIPK",
        "DrugGNN",
        "MOLIR",
        "PharmaFormer",
        "Precily",
        "SRMF",
        "SimpleNeuralNetwork",
        "MultiViewNeuralNetwork",
        "SuperFELTR",
        "SparseGO",
    ],
)
def test_literature_models_build_with_defaults(model_name: str) -> None:
    spec = ModelSpec.from_name(model_name)
    DRPModel(spec, {**spec.default_hyperparameters, **_SMALL_HYPERPARAMETERS.get(model_name, {})})


@pytest.mark.parametrize(
    ("model_name", "hyperparameters", "extra_views"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, ()),
        ("SRMF", {"K": 2, "max_iter": 2, "n_features": 4}, ()),
        ("MultiViewNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, ("mutations",)),
        ("NaiveDrugMeanPredictor", {}, ()),
    ],
)
def test_literature_model_lifecycle(
    model_name: str,
    hyperparameters: dict,
    extra_views: tuple[str, ...],
) -> None:
    mudataset, split = _dataset(extra_views=extra_views)
    model = DRPModel(small_data_spec(model_name), hyperparameters)
    model.train(mudataset, split.train)
    preds = model.predict(mudataset, split.test)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = f"{directory}/model"
        model.save(checkpoint)
        loaded = DRPModel.load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split.test)
    assert np.allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_untrained_component_model_raises() -> None:
    model = DRPModel(ModelSpec("elasticNet", from_spec("raw[gene_expression]:fingerprints:elasticNet")), {})
    mudataset, split = _dataset()

    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(mudataset, split)
