"""Smoke tests for literature models routed through the native facade."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset
from tests.models.synthetic_fixtures import lco_split_masks, synthetic_mudataset

#: The multi-omics views ``MultiViewNeuralNetwork`` reads beyond gene expression.
MULTIVIEW_EXTRA_VIEWS = ("methylation", "mutations", "copy_number_variation_gistic")


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _dataset(*, extra_views: tuple[str, ...] = ()) -> tuple[Dataset, SplitMasks]:
    """Build a four-feature synthetic dataset and an LCO split over it.

    :param extra_views: Cell-line modalities beyond ``gene_expression``.
    :returns: ``(dataset, split)``.
    """
    dataset = synthetic_mudataset(n_features_per_view=4, fingerprint_width=4, extra_views=extra_views)
    return dataset, lco_split_masks()


LITERATURE_MODEL_NAMES = (
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
)


@pytest.mark.parametrize("model_name", LITERATURE_MODEL_NAMES)
def test_literature_models_build_with_defaults(model_name: str) -> None:
    model_cls = construct_model(model_name)
    hyperparameters = dict(model_cls.get_hyperparameter_set()[0])
    if model_name == "DIPK":
        hyperparameters.update({"epochs": 1, "epochs_autoencoder": 1, "heads": 1})
    elif model_name in {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}:
        hyperparameters.update({"units_per_layer": [2, 2], "max_epochs": 1})
    elif model_name == "PharmaFormer":
        hyperparameters.update({"epochs": 1, "patience": 2})
    elif model_name == "Precily":
        hyperparameters.update({"epochs": 1, "batch_size": 32})
    elif model_name == "SparseGO":
        hyperparameters.update({"epochs": 1, "batch_size": 32})
    model_cls(hyperparameters)


@pytest.mark.parametrize(
    ("model_name", "hyperparameters", "extra_views"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, ()),
        ("SRMF", {"K": 2, "max_iter": 2, "n_features": 4}, ()),
        (
            "MultiViewNeuralNetwork",
            {
                "units_per_layer": [2, 2],
                "max_epochs": 1,
                "methylation_pca_components": 2,
            },
            MULTIVIEW_EXTRA_VIEWS,
        ),
        ("NaiveDrugMeanPredictor", {}, ()),
    ],
)
def test_literature_model_lifecycle(
    model_name: str,
    hyperparameters: dict,
    extra_views: tuple[str, ...],
) -> None:
    mudataset, split = _dataset(extra_views=extra_views)
    model = construct_model(model_name)(hyperparameters)
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = f"{directory}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_untrained_component_model_raises() -> None:
    model_cls = construct_model("elasticNet", "raw[gene_expression]:fingerprints:elasticNet")
    model = model_cls({})
    mudataset, split = _dataset()

    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(mudataset, split)
