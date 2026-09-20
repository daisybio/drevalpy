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
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    early_stopping_response,
    seed_literature,
    trainable_parameter_vector,
    two_by_two_batch,
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _drug_graph(*, num_features: int = 9, offset: float = 0.0) -> Data:
    return Data(
        x=torch.arange(4 * num_features, dtype=torch.float32).reshape(4, num_features) / 10 + offset,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
    )


def _druggnn_batch(*, with_early_stopping: bool = False) -> ModelInputBatch:
    graphs = np.empty(2, dtype=object)
    graphs[:] = [_drug_graph(), _drug_graph(offset=0.5)]
    return two_by_two_batch(
        cell_line_blocks={
            "gene_expression": numeric_feature_block(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])),
        },
        drug_blocks={"drug_graph": graph_feature_block(graphs)},
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
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


class TestTinyDrugGNNFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int) -> None:
        hyperparameters = {
            "epochs": 1,
            "batch_size": 2,
            "hidden_dim": 8,
            "dropout": 0.0,
            "learning_rate": 0.01,
        }
        batch = _druggnn_batch(with_early_stopping=True)
        seed_literature(seed)
        pristine = DrugGNNPredictor(hyperparameters)
        cell_line_matrix = batch.cell_line.blocks["gene_expression"].values
        drug_graphs = batch.drug.blocks["drug_graph"].values
        initial = trainable_parameter_vector(pristine._build_model(cell_line_matrix, drug_graphs))

        seed_literature(seed)
        predictor = DrugGNNPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)
        expected = np.asarray(predictor.predict(batch))
        assert np.isfinite(expected).all()
        assert np.isfinite(np.mean((expected - batch.response) ** 2))
        assert_training_predictions("druggnn", seed, expected)
        assert_frozen_reference("druggnn", seed)

        restored = DrugGNNPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(expected, restored.predict(batch), rtol=1e-5, atol=1e-5)

    def test_restores_best_validation_epoch_when_a_later_epoch_is_worse(self) -> None:
        import pytorch_lightning as pl

        class ValidationHistory(pl.Callback):
            def __init__(self) -> None:
                self.losses: list[float] = []

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                if not trainer.sanity_checking:
                    self.losses.append(float(trainer.callback_metrics["val_loss"]))

        hyperparameters = {
            "epochs": 4,
            "batch_size": 2,
            "hidden_dim": 8,
            "dropout": 0.0,
            "learning_rate": 0.5,
        }
        batch = _druggnn_batch(with_early_stopping=True)

        seed_literature(0)
        restored_best = DrugGNNPredictor(hyperparameters)
        restored_best.fit(batch)
        assert restored_best._model is not None

        seed_literature(0)
        final_epoch = DrugGNNPredictor(hyperparameters)
        cell_matrix = batch.cell_line.blocks["gene_expression"].values
        drug_graphs = batch.drug.blocks["drug_graph"].values
        final_epoch._model = final_epoch._build_model(cell_matrix, drug_graphs)
        train_loader = final_epoch._graph_loader(
            batch.cell_line_pair_idx,
            final_epoch._resolve_drug_pair_idx(batch),
            cell_matrix,
            drug_graphs,
            batch.response,
            batch_size=2,
            shuffle=True,
        )
        val_loader = final_epoch._build_val_loader(batch, cell_matrix, drug_graphs)
        assert val_loader is not None
        history = ValidationHistory()
        trainer = pl.Trainer(
            max_epochs=4,
            accelerator="cpu",
            devices=1,
            callbacks=[history],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(final_epoch._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        evaluator = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
        best_loss = evaluator.validate(restored_best._model, dataloaders=val_loader, verbose=False)[0]["val_loss"]
        final_loss = evaluator.validate(final_epoch._model, dataloaders=val_loader, verbose=False)[0]["val_loss"]
        assert history.losses[-1] > min(history.losses)
        assert best_loss == pytest.approx(min(history.losses), rel=1e-6, abs=1e-7)
        assert best_loss < final_loss
