"""Smoke tests for literature models routed through the native facade."""

from __future__ import annotations

import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mudata as md
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.data.structures.mudataset import MuDataset
from drevalpy.data.structures import SplitMasks
from drevalpy.models import construct_model


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _make_mudataset_ge_fingerprints() -> tuple[MuDataset, SplitMasks]:
    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    ge_matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    gene_expression_ad = ad.AnnData(
        X=ge_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]),
    )
    response_ad.varm["fingerprints"] = np.array([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.3, 0.7]], dtype=np.float32)
    mdata = md.MuData({"response": response_ad, "gene_expression": gene_expression_ad})
    mudataset = MuDataset(mdata)
    split = SplitMasks(
        train_cell_lines=np.array([0]),
        test_cell_lines=np.array([1]),
        val_cell_lines=np.array([], dtype=np.intp),
    )
    return mudataset, split


def _make_mudataset_multiview() -> tuple[MuDataset, SplitMasks]:
    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    ge_matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    gene_expression_ad = ad.AnnData(
        X=ge_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]),
    )
    meth_matrix = np.array([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]], dtype=np.float32)
    methylation_ad = ad.AnnData(
        X=meth_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"cpg{i}" for i in range(4)]),
    )
    mut_matrix = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    mutations_ad = ad.AnnData(
        X=mut_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"mut{i}" for i in range(4)]),
    )
    cnv_matrix = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
    cnv_ad = ad.AnnData(
        X=cnv_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"cnv{i}" for i in range(4)]),
    )
    response_ad.varm["fingerprints"] = np.array([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.3, 0.7]], dtype=np.float32)
    mdata = md.MuData(
        {
            "response": response_ad,
            "gene_expression": gene_expression_ad,
            "methylation": methylation_ad,
            "mutations": mutations_ad,
            "copy_number_variation_gistic": cnv_ad,
        }
    )
    mudataset = MuDataset(mdata)
    split = SplitMasks(
        train_cell_lines=np.array([0]),
        test_cell_lines=np.array([1]),
        val_cell_lines=np.array([], dtype=np.intp),
    )
    return mudataset, split


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
    ("model_name", "hyperparameters", "data_factory"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, "_make_mudataset_ge_fingerprints"),
        ("SRMF", {"K": 2, "max_iter": 2, "n_features": 4}, "_make_mudataset_ge_fingerprints"),
        (
            "MultiViewNeuralNetwork",
            {
                "units_per_layer": [2, 2],
                "max_epochs": 1,
                "methylation_pca_components": 2,
            },
            "_make_mudataset_multiview",
        ),
        ("NaiveDrugMeanPredictor", {}, "_make_mudataset_ge_fingerprints"),
    ],
)
def test_literature_model_lifecycle(
    model_name: str,
    hyperparameters: dict,
    data_factory: str,
) -> None:
    mudataset, split = globals()[data_factory]()
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
    from drevalpy.models import construct_model

    model_cls = construct_model("elasticNet", "raw[expression]:fingerprints:elasticNet")
    model = model_cls({})
    mudataset, split = _make_mudataset_ge_fingerprints()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(mudataset, split)
