"""Smoke test mirror for druggnn predictor package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.literature.druggnn.predictor import DrugGNNPredictor
from drevalpy.registry._builtins import ensure_predictor_registered, register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import graph_feature_block, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.components.predictors.literature._helpers import early_stopping_response, two_by_two_batch


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _drug_graph(*, num_features: int = 9) -> Data:
    return Data(
        x=torch.randn(4, num_features),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
    )


def _druggnn_batch(*, with_early_stopping: bool = False) -> ModelInputBatch:
    graphs = np.empty(2, dtype=object)
    graphs[:] = [_drug_graph(), _drug_graph()]
    return two_by_two_batch(
        cell_line_blocks={
            "gene_expression": numeric_feature_block(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])),
        },
        drug_blocks={"drug_graph": graph_feature_block(graphs)},
        cell_line_pair_idx=np.zeros(4, dtype=np.int64),
        drug_pair_idx=None,
        early_stopping_response=early_stopping_response() if with_early_stopping else None,
        checkpoint_dir=Path(tempfile.mkdtemp()),
    )


def test_druggnn_predictor_registry_name() -> None:
    ensure_predictor_registered("drugGNN")
    assert get_predictor("drugGNN") is DrugGNNPredictor


def test_druggnn_requires_graph_drug_contract() -> None:
    cls = get_predictor("drugGNN")
    assert cls.drug_contract.format == FeatureFormat.GRAPH
    assert cls.required_drug_blocks == ("drug_graph",)
    assert cls.supports_early_stopping is True


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
