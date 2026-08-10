"""Tests for the public construct_model API."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models import DRPModel, construct_model
from drevalpy.models.config import from_spec


def test_construct_model_returns_drp_model_subclass() -> None:
    model_cls = construct_model("PcaIdentityRF", "pca[expression]:identity:randomForest")
    assert issubclass(model_cls, DRPModel)
    assert model_cls.get_model_name() == "PcaIdentityRF"


def test_construct_model_one_arg_zoo_name() -> None:
    model_cls = construct_model("ElasticNet")
    assert issubclass(model_cls, DRPModel)
    assert model_cls.get_model_name() == "ElasticNet"
    assert construct_model("ElasticNet") is model_cls


def test_construct_model_derives_early_stopping_from_predictor() -> None:
    model_cls = construct_model("DipkFacade", "DIPK")
    assert model_cls.supports_early_stopping() is True


def test_construct_model_accepts_model_config() -> None:
    config = from_spec("ElasticNet")
    model_cls = construct_model("ConfiguredElasticNet", config)
    assert model_cls.get_model_name() == "ConfiguredElasticNet"
    assert construct_model("ConfiguredElasticNet", config) is model_cls


def test_construct_model_invalid_spec_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        construct_model("BadModel", "not-a-valid-spec")


def test_construct_model_one_arg_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        construct_model("not-a-valid-spec")


def test_default_hyperparameters_for_constructed_pca_model() -> None:
    import drevalpy.components.registry.register_builtins as register_builtins
    from drevalpy.components.core.tuning.config_resolution import (
        assert_component_local_hyperparameters,
        default_config_for_drp_model,
    )

    register_builtins.register_builtin_components()

    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    hp = model_cls.get_default_hyperparameters()

    assert not any("." in key for key in hp)
    assert "cell_line_featurizer.pca[expression].n_components" not in hp
    assert "cell_line_featurizer.pca.0.n_components" not in hp
    assert hp["n_components"] == 128

    config = default_config_for_drp_model(model_cls)
    assert config is not None
    assert config.featurizer_values("cell_line", "pca[expression]")["n_components"] == 128
    assert_component_local_hyperparameters(config)

    model = model_cls(hp)
    assert model._resolved_model_config is not None
    assert_component_local_hyperparameters(model._resolved_model_config)


def test_construct_model_train_predict_smoke() -> None:
    import drevalpy.components.registry.register_builtins as register_builtins

    register_builtins.register_builtin_components()

    model_cls = construct_model("ComboRF", "raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    model = model_cls()

    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.data.structures import SplitMask, SplitMasks
    from drevalpy.types.data.dataset import Dataset

    cl_ids_unique = np.array(["cl1", "cl2"])
    drug_ids_all = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids_unique, "tissue": ["Lung", "Blood"]}, index=cl_ids_unique),
        var=pd.DataFrame(index=drug_ids_all),
    )
    ge_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    gene_expression_ad = ad.AnnData(
        X=ge_matrix,
        obs=pd.DataFrame(index=cl_ids_unique),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    mut_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mutations_ad = ad.AnnData(
        X=mut_matrix,
        obs=pd.DataFrame(index=cl_ids_unique),
        var=pd.DataFrame(index=["mut0", "mut1"]),
    )
    response_ad.varm["fingerprints"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mdata = md.MuData({"response": response_ad, "gene_expression": gene_expression_ad, "mutations": mutations_ad})
    mudataset = Dataset(mdata, name="test")
    split = SplitMasks(
        train=SplitMask(np.array([[True, True], [False, False]])),
        test=SplitMask(np.array([[False, False], [True, True]])),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
