"""Capability-contract tests for predictor state and lifecycle."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.feature_block import graph_feature_block, numeric_feature_block
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.predictors.sklearn_models import AdaBoostPredictor, RidgePredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.register_builtins import ensure_predictor_registered, register_builtin_components
from drevalpy.components.registry import get_predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig, from_spec
from tests.models.synthetic_fixtures import (
    cell_line_gene_expression,
    drug_fingerprints,
    multi_drug_response,
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _neural_batch(*, with_early_stopping: bool = False) -> ModelInputBatch:
    response = multi_drug_response()
    cell_line_features = np.vstack(
        [
            cell_line_gene_expression().features["cl1"]["gene_expression"],
            cell_line_gene_expression().features["cl2"]["gene_expression"],
        ]
    )
    drug_features = np.vstack(
        [
            drug_fingerprints().features["d1"]["fingerprints"],
            drug_fingerprints().features["d2"]["fingerprints"],
        ]
    )
    early_stopping = None
    if with_early_stopping:
        early_stopping = ResponseBatch(
            response=np.array([1.5, 2.5]),
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=cell_line_features,
        drug_features=drug_features,
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=early_stopping,
        training_context=TrainingContext(checkpoint_dir=Path(tempfile.mkdtemp())),
    )


def _drug_graph(*, num_features: int = 9) -> Data:
    return Data(
        x=torch.randn(4, num_features),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
    )


def _druggnn_batch(*, with_early_stopping: bool = False) -> ModelInputBatch:
    response = multi_drug_response()
    cell_line_blocks = {"gene_expression": numeric_feature_block(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))}
    graphs = np.empty(2, dtype=object)
    graphs[:] = [_drug_graph(), _drug_graph()]
    drug_blocks = {
        "drug_graph": graph_feature_block(graphs),
    }
    early_stopping = None
    if with_early_stopping:
        early_stopping = ResponseBatch(
            response=np.array([1.5, 2.5]),
            cell_line_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.zeros(4, dtype=np.int64),
        drug_pair_idx=None,
        cell_line_blocks=cell_line_blocks,
        drug_blocks=drug_blocks,
        early_stopping_response=early_stopping,
        training_context=TrainingContext(checkpoint_dir=Path(tempfile.mkdtemp())),
    )


def test_neural_network_configured_is_not_fitted() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]},
    )
    assert predictor._model is None
    assert predictor.is_fitted() is False


def test_neural_network_early_stopping_wires_validation_loader() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "batch_size": 2, "units_per_layer": [4, 2]},
    )
    batch = _neural_batch(with_early_stopping=True)
    captured: dict[str, object] = {}

    def _capture_fit(self, model, train_dataloaders, val_dataloaders=None):
        captured["val_loader"] = val_dataloaders
        return None

    with patch("pytorch_lightning.Trainer.fit", _capture_fit):
        predictor.fit(batch)
    assert captured["val_loader"] is not None


def test_neural_network_round_trip_state() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "batch_size": 2, "units_per_layer": [4, 2]},
    )
    predictor.fit(_neural_batch())
    preds = predictor.predict(_neural_batch())
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()

    restored = NeuralNetworkPredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted()
    restored_preds = restored.predict(_neural_batch())
    assert np.allclose(preds, restored_preds)


def test_neural_network_set_state_raises_on_invalid_payload() -> None:
    predictor = NeuralNetworkPredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"checkpoint": b"not-a-torch-checkpoint"})


def test_druggnn_delegates_training_to_lightning() -> None:
    predictor = DrugGNNPredictor(
        hyperparameters={"epochs": 1, "batch_size": 2, "num_workers": 0},
    )
    batch = _druggnn_batch(with_early_stopping=True)
    predictor.fit(batch)
    assert predictor.is_fitted()
    assert predictor._model is not None


def test_druggnn_supports_early_stopping_flag() -> None:
    assert DrugGNNPredictor.supports_early_stopping is True


def test_druggnn_round_trip_state() -> None:
    predictor = DrugGNNPredictor(
        hyperparameters={"epochs": 1, "batch_size": 2, "num_workers": 0},
    )
    batch = _druggnn_batch()
    predictor.fit(batch)
    assert predictor.is_fitted()
    assert predictor._model is not None
    original_weight = next(predictor._model.parameters()).detach().cpu()

    restored = DrugGNNPredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted()
    assert restored._model is not None
    restored_weight = next(restored._model.parameters()).detach().cpu()
    assert torch.allclose(original_weight, restored_weight)


def test_literature_predictor_lazy_package_import() -> None:
    ensure_predictor_registered("dipk")
    precily_module = "drevalpy.components.predictors.literature.precily.predictor"
    saved = sys.modules.pop(precily_module, None)
    try:
        cls = get_predictor("dipk")
        assert cls.__name__ == "DIPKPredictor"
        assert precily_module not in sys.modules
    finally:
        if saved is not None:
            sys.modules[precily_module] = saved


def test_structured_predictor_set_state_raises_on_invalid_blob() -> None:
    predictor = SRMFPredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"payload": b"invalid"})


def test_ridge_zoo_preset_exists() -> None:
    config = from_spec("Ridge")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "ridge"


def test_adaboost_default_depth_matches_space() -> None:
    predictor = AdaBoostPredictor()
    estimator = predictor._make_estimator()
    assert estimator.estimator.max_depth == 4


def test_sklearn_set_state_raises_when_estimator_missing() -> None:
    predictor = RidgePredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"hyperparameters": {"alpha": 1.0}, "mode": "regression"})


def test_naive_tissue_round_trip() -> None:
    import anndata as ad
    import pandas as pd

    import mudata as md
    from drevalpy.data.structures import SplitMasks
    from drevalpy.data.structures.mudataset import MuDataset

    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    mdata = md.MuData({"response": response_ad})
    mdata.obs["tissue"] = ["Lung", "Blood"]
    mudataset = MuDataset(mdata)
    split = SplitMasks(
        train=np.array([[0, 0], [0, 1]]),
        test=np.array([[1, 0], [1, 1]]),
        val=np.empty((0, 2), dtype=np.intp),
    )
    model = construct_model("NaiveTissueMeanPredictor")()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert np.isfinite(preds).all()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = f"{tmp}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(mudataset, split)
    assert np.allclose(preds, loaded_preds)


def test_xgboost_load_applies_thread_defaults_before_restore() -> None:
    pytest.importorskip("xgboost")
    from drevalpy.components.predictors.xgboost_pred import XGBoostPredictor, _set_xgboost_thread_defaults

    predictor = XGBoostPredictor(hyperparameters={"n_estimators": 5})
    predictor.fit(_neural_batch())
    state = predictor.get_state()

    with patch(
        "drevalpy.components.predictors.xgboost_pred._set_xgboost_thread_defaults",
        wraps=_set_xgboost_thread_defaults,
    ) as thread_defaults:
        restored = XGBoostPredictor()
        restored.set_state(state)
        thread_defaults.assert_called_once()
