"""Dense feed-forward neural network predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.literature._batch_dataset import PairMatrixDataset
from drevalpy.components.predictors.literature.impl.simple_neural_network.utils import FeedForwardNetwork
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


@register_predictor(
    "neuralNetwork",
    description="Dense feed-forward network on concatenated cell-line and drug features.",
    category="general_purpose",
)
class NeuralNetworkPredictor(Predictor):
    """Neural network predictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._model: FeedForwardNetwork | None = None
        self._input_dim: int | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "units_per_layer": [512, 256, 128],
            "dropout_prob": 0.2,
            "max_epochs": 50,
            "batch_size": 16,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "dropout_prob": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "max_epochs": {"type": "int", "low": 10, "high": 100, "default": 50},
            "batch_size": {"type": "int", "low": 8, "high": 64, "default": 16},
        }

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        merged = {**self.get_default_hyperparameters(), **hyperparameters}
        self._hyperparameters = merged
        input_dim = int(input_dims.get("cell_line", 0)) + int(input_dims.get("drug", 0))
        self._input_dim = input_dim
        self._model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": merged["units_per_layer"],
                "dropout_prob": merged["dropout_prob"],
            },
            input_dim=input_dim,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if self._model is None or len(x) == 0:
            return
        dataset = PairMatrixDataset(x, y)
        batch_size = min(int(self._hyperparameters.get("batch_size", 16)), len(x))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=batch_size < len(x),
        )
        trainer = pl.Trainer(
            max_epochs=int(self._hyperparameters.get("max_epochs", 50)),
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(self._model, train_dataloaders=loader)

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if self._model is None or len(x) == 0:
            return np.full(len(x), np.nan, dtype=np.float64)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    def is_fitted(self) -> bool:
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        if self._model is None:
            return {}
        import io

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
            return
        import io

        data = torch.load(io.BytesIO(checkpoint), weights_only=False)  # noqa: S614
        if not isinstance(data, dict):
            return
        hyperparameters = data.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)
        input_dim = data.get("input_dim")
        if input_dim is None:
            return
        merged = self._hyperparameters
        self._model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": merged.get("units_per_layer", [512, 256, 128]),
                "dropout_prob": merged.get("dropout_prob", 0.2),
            },
            input_dim=int(input_dim),
        )
        state_dict = data.get("state_dict")
        if state_dict is not None:
            self._model.load_state_dict(state_dict)
