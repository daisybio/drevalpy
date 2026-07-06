"""Tests for the public construct_model API."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import DRPModel, construct_model


def test_construct_model_returns_drp_model_subclass() -> None:
    model_cls = construct_model("PcaOneHotRF", "pca[expression]:oneHot:randomForest")
    assert issubclass(model_cls, DRPModel)
    assert model_cls.get_model_name() == "PcaOneHotRF"


def test_construct_model_invalid_spec_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        construct_model("BadModel", "not-a-valid-spec")


def test_default_hyperparameters_for_constructed_pca_model() -> None:
    import drevalpy.components.register_builtins as register_builtins
    from drevalpy.components.tuning.drp_hyperparameters import (
        assert_component_local_hyperparameters,
        default_config_for_drp_model,
    )

    register_builtins.register_builtin_components()

    model_cls = construct_model("PcaOneHotRF", "pca[expression]:identity:randomForest")
    hp = model_cls.get_default_hyperparameters()

    assert not any("." in key for key in hp)
    assert "featurizer.cell_line.pca.0.n_components" not in hp
    assert hp["n_components"] == 128

    config = default_config_for_drp_model(model_cls)
    assert config is not None
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert_component_local_hyperparameters(config)

    model = model_cls()
    model.build_model(hp)
    assert model._resolved_model_config is not None
    assert_component_local_hyperparameters(model._resolved_model_config)


def test_construct_model_train_predict_smoke() -> None:
    import drevalpy.components.register_builtins as register_builtins

    register_builtins.register_builtin_components()

    model_cls = construct_model("ComboRF", "geneExpression+mutations:fingerprints+oneHot:randomForest")
    model = model_cls()
    model.build_model(model.get_default_hyperparameters())

    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {
                "gene_expression": np.array([0.1, 0.2, 0.3]),
                "mutations": np.array([0.0, 1.0]),
            },
            "cl2": {
                "gene_expression": np.array([0.4, 0.5, 0.6]),
                "mutations": np.array([1.0, 0.0]),
            },
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0]), "one_hot": np.array([1.0, 0.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0]), "one_hot": np.array([0.0, 1.0, 0.0])},
        }
    )
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()
