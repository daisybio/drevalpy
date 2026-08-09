"""Dense feed-forward neural network predictor."""

from __future__ import annotations

import io
import secrets
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from upath import UPath as Path

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._tensor_data import make_pair_loader
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode
from drevalpy.utils.torch_io import load_state_dict, load_trusted_mapping, save_torch_payload


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
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._model: FeedForwardNetwork | None = None
        self._input_dim: int | None = None
        self._is_fitted = False

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Get default hyperparameters.

        :returns: Result.
        """
        return {
            "units_per_layer": [512, 256, 128],
            "dropout_prob": 0.2,
            "max_epochs": 50,
            "batch_size": 16,
            "patience": 5,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "dropout_prob": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "max_epochs": {"type": "int", "low": 10, "high": 100, "default": 50},
            "batch_size": {"type": "int", "low": 8, "high": 64, "default": 16},
        }

    def _materialize(self, input_dim: int) -> None:
        """Allocate the network once input dimensionality is known.

        :param input_dim: Flattened feature width for one response pair.
        """
        self._input_dim = input_dim
        self._model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": self._hyperparameters["units_per_layer"],
                "dropout_prob": self._hyperparameters["dropout_prob"],
            },
            input_dim=input_dim,
        )
        self._is_fitted = False

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data using lazy pair-level lookup.

        :param batch: batch.
        """
        input_dim = self._compute_input_dim(batch)
        self._materialize(input_dim)
        if batch.n_pairs == 0:
            self._is_fitted = False
            return
        self._train_with_optional_early_stopping(batch)
        self._is_fitted = True

    @staticmethod
    def _compute_input_dim(batch: ModelInputBatch) -> int:
        """Determine the concatenated feature width from entity matrices.

        :param batch: Input batch.
        :returns: Total feature dimensionality per pair.
        """
        dim = 0
        if batch.cell_line_features.size > 0 and batch.cell_line_features.ndim == 2:
            dim += batch.cell_line_features.shape[1]
        if batch.drug_features is not None and batch.drug_features.size > 0 and batch.drug_features.ndim == 2:
            dim += batch.drug_features.shape[1]
        return dim

    def _build_pair_loader(
        self,
        batch: ModelInputBatch,
        cell_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray | None,
        response: np.ndarray | None,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        """Build a lazy pair loader from entity matrices and pair indices.

        :param batch: Batch providing entity-level feature matrices.
        :param cell_pair_idx: Cell-line pair indices.
        :param drug_pair_idx: Drug pair indices (or None).
        :param response: Optional response array.
        :param shuffle: Whether to shuffle.
        :param drop_last: Whether to drop last incomplete batch.
        :returns: DataLoader.
        """
        batch_size = min(int(self._hyperparameters.get("batch_size", 16)), len(cell_pair_idx))
        if batch_size < 1:
            batch_size = 1

        specs: list[tuple[np.ndarray, np.ndarray]] = []
        if batch.cell_line_features.size > 0:
            specs.append((batch.cell_line_features, cell_pair_idx))
        if batch.drug_features is not None and batch.drug_features.size > 0 and drug_pair_idx is not None:
            specs.append((batch.drug_features, drug_pair_idx))

        return make_pair_loader(
            *specs,
            response=response,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def _train_with_optional_early_stopping(self, batch: ModelInputBatch) -> None:
        if self._model is None:
            msg = "Neural network predictor must be materialized before training"
            raise RuntimeError(msg)

        y = np.asarray(batch.response, dtype=np.float32).reshape(-1)
        batch_size = min(int(self._hyperparameters.get("batch_size", 16)), batch.n_pairs)
        train_loader = self._build_pair_loader(
            batch,
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            y,
            shuffle=True,
            drop_last=batch_size < batch.n_pairs,
        )

        val_loader = None
        es_resp = batch.early_stopping_response
        if es_resp is not None and len(es_resp) > 0 and es_resp.response is not None:
            es_cell_idx, es_drug_idx = batch._pair_indices_for(es_resp)
            y_val = np.asarray(es_resp.response, dtype=np.float32).reshape(-1)
            if len(y_val) > 0:
                val_loader = self._build_pair_loader(
                    batch,
                    es_cell_idx,
                    es_drug_idx,
                    y_val,
                    shuffle=False,
                    drop_last=False,
                )

        monitor = "val_loss" if val_loader is not None else "train_loss"
        patience = int(self._hyperparameters.get("patience", 5))
        callbacks: list[pl.Callback] = [
            EarlyStopping(monitor=monitor, mode="min", patience=patience),
        ]

        checkpoint_dir = batch.training_context.checkpoint_dir
        unique_subfolder = Path(checkpoint_dir) / ("run_" + secrets.token_hex(8))
        unique_subfolder.mkdir(parents=True, exist_ok=True)
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
            checkpoint = load_state_dict(checkpoint_callback.best_model_path)
            self._model.load_state_dict(checkpoint["state_dict"])

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict using lazy pair-level lookup (no full feature expansion).

        :param batch: Featurized pairs to score.
        :returns: Predicted responses.
        """
        if not self._is_fitted or self._model is None or batch.n_pairs == 0:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)

        loader = self._build_pair_loader(
            batch,
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            response=None,
            shuffle=False,
            drop_last=False,
        )

        self._model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for tensors in loader:
                x = torch.cat(tensors, dim=1)
                preds = self._model(x).cpu().numpy()
                predictions.append(preds)

        return np.concatenate(predictions, axis=0).astype(np.float64).reshape(-1)

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
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._is_fitted

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if not self._is_fitted or self._model is None:
            return {}
        buffer = io.BytesIO()
        save_torch_payload(
            {
                "hyperparameters": dict(self._hyperparameters),
                "state_dict": self._model.state_dict(),
                "input_dim": self._input_dim,
            },
            buffer,
        )
        return {"checkpoint": buffer.getvalue()}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        :raises PredictorStateError: Raised on invalid input.
        """
        checkpoint = state.get("checkpoint")
        if not isinstance(checkpoint, (bytes, bytearray)):
            msg = "NeuralNetworkPredictor state requires a checkpoint byte blob"
            raise PredictorStateError(msg)
        try:
            data = load_trusted_mapping(checkpoint)
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
