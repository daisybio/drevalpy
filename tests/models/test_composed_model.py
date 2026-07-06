"""Integration tests for drevalpy.models.composed_model."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.factory import naive_model_config, sklearn_model_config


def test_sklearn_model_config_builds_composed_model() -> None:
    import drevalpy.components.register_builtins as register_builtins

    register_builtins.register_builtin_components()

    config = sklearn_model_config("elasticNet", {"alpha": 0.1, "l1_ratio": 0.5})
    config.validate()
    model = config.create_model()
    assert model is not None


def test_composed_model_train_predict_on_synthetic_data() -> None:
    import drevalpy.components.register_builtins as register_builtins

    register_builtins.register_builtin_components()

    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )

    model = sklearn_model_config("ridge", {"alpha": 1.0}).create_model()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()


def test_naive_model_config_train_predict_on_synthetic_data() -> None:
    import drevalpy.components.register_builtins as register_builtins

    register_builtins.register_builtin_components()

    response = DrugResponseDataset(
        response=np.array([1.0, 3.0, 5.0, 7.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    model = naive_model_config("naiveMean").create_model()
    model.train(response, FeatureDataset(features={}), None)
    preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        FeatureDataset(features={}),
        None,
    )
    assert np.allclose(preds, 4.0)
