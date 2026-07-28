"""Dense feed-forward neural network predictor."""

from __future__ import annotations

import io
import os
import secrets
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.literature._batch_dataset import PairMatrixDataset
from drevalpy.components.predictors.literature.impl.simple_neural_network.utils import FeedForwardNetwork
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "neuralNetwork",
    description="Dense feed-forward network on concatenated cell-line and drug features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NeuralNetworkPredictor(MatrixPredictor):
    """Neural network predictor component."""

    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._model: FeedForwardNetwork | None = None
        self._input_dim: int | None = None
        self._is_fitted = False

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "units_per_layer": [512, 256, 128],
            "dropout_prob": 0.2,
            "max_epochs": 50,
            "batch_size": 16,
            "patience": 5,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "dropout_prob": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "max_epochs": {"type": "int", "low": 10, "high": 100, "default": 50},
            "batch_size": {"type": "int", "low": 8, "high": 64, "default": 16},
        }

    def _materialize(self, input_dim: int) -> None:
        """Allocate the network once input dimensionality is known."""
        self._input_dim = input_dim
        self._model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": self._hyperparameters["units_per_layer"],
                "dropout_prob": self._hyperparameters["dropout_prob"],
            },
            input_dim=input_dim,
        )
        self._is_fitted = False

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Matrix predictors require response values during fit"
            raise ValueError(msg)
        x = batch.to_feature_matrix()
        y = np.asarray(batch.response, dtype=np.float64)
        validate_matrix_fit(x, y, n_pairs=batch.n_pairs)
        input_dim = int(x.shape[1]) if x.ndim == 2 else 0
        self._materialize(input_dim)
        if batch.n_pairs == 0:
            self._is_fitted = False
            return
        self._train_with_optional_early_stopping(batch, x, y)
        self._is_fitted = True

    def _train_with_optional_early_stopping(
        self,
        batch: ModelInputBatch,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if self._model is None:
            msg = "Neural network predictor must be materialized before training"
            raise RuntimeError(msg)
        batch_size = min(int(self._hyperparameters.get("batch_size", 16)), len(x))
        train_loader = DataLoader(
            PairMatrixDataset(x, y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=batch_size < len(x),
        )

        val_loader = None
        x_val = batch.early_stopping_feature_matrix()
        if x_val is not None and batch.early_stopping_response is not None:
            y_val = np.asarray(batch.early_stopping_response.response, dtype=np.float64)
            if len(x_val) > 0 and len(x_val) == len(y_val):
                val_loader = DataLoader(
                    PairMatrixDataset(x_val, y_val),
                    batch_size=batch_size,
                    shuffle=False,
                )

        monitor = "val_loss" if val_loader is not None else "train_loss"
        patience = int(self._hyperparameters.get("patience", 5))
        callbacks: list[pl.Callback] = [
            EarlyStopping(monitor=monitor, mode="min", patience=patience),
        ]

        checkpoint_dir = batch.training_context.checkpoint_dir
        unique_subfolder = os.path.join(checkpoint_dir, "run_" + secrets.token_hex(8))
        os.makedirs(unique_subfolder, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=unique_subfolder,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename="best",
        )
        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=int(self._hyperparameters.get("max_epochs", 50)),
            accelerator="cpu",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=False,
        )
        if val_loader is None:
            trainer.fit(self._model, train_dataloaders=train_loader)
        else:
            trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if checkpoint_callback.best_model_path:
            checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=True)  # noqa: S614
            self._model.load_state_dict(checkpoint["state_dict"])

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        _ = x, y

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._model is None or len(x) == 0:
            return np.full(len(x), np.nan, dtype=np.float64)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_state(self) -> dict[str, object]:
        if not self._is_fitted or self._model is None:
            return {}
        buffer = io.BytesIO()
        torch.save(
            {
                "hyperparameters": dict(self._hyperparameters),
                "state_dict": self._model.state_dict(),
                "input_dim": self._input_dim,
            },
            buffer,
        )
        return {"checkpoint": buffer.getvalue()}

    def set_state(self, state: dict[str, object]) -> None:
        checkpoint = state.get("checkpoint")
        if not isinstance(checkpoint, (bytes, bytearray)):
            msg = "NeuralNetworkPredictor state requires a checkpoint byte blob"
            raise PredictorStateError(msg)
        try:
            data = torch.load(io.BytesIO(checkpoint), weights_only=False)  # noqa: S614
        except Exception as exc:
            msg = "NeuralNetworkPredictor checkpoint could not be deserialized"
            raise PredictorStateError(msg) from exc
        if not isinstance(data, dict):
            msg = "NeuralNetworkPredictor checkpoint payload must be a mapping"
            raise PredictorStateError(msg)
        hyperparameters = data.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = "NeuralNetworkPredictor checkpoint is missing hyperparameters"
            raise PredictorStateError(msg)
        input_dim = data.get("input_dim")
        if input_dim is None:
            msg = "NeuralNetworkPredictor checkpoint is missing input_dim"
            raise PredictorStateError(msg)
        state_dict = data.get("state_dict")
        if state_dict is None:
            msg = "NeuralNetworkPredictor checkpoint is missing state_dict"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        self._input_dim = int(input_dim)
        self._materialize(self._input_dim)
        if self._model is None:
            msg = "NeuralNetworkPredictor failed to materialize from checkpoint"
            raise PredictorStateError(msg)
        self._model.load_state_dict(state_dict)
        self._is_fitted = True
