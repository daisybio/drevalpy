"""Dense feed-forward neural network predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from drevalpy.components.config import PredictionMode
from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.pair_context import PairContext
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.predictors.literature._batch_dataset import PairMatrixDataset
from drevalpy.components.predictors.literature.impl.simple_neural_network.utils import FeedForwardNetwork
from drevalpy.components.registry import register_predictor


@register_predictor(
    "neuralNetwork",
    description="Dense feed-forward network on concatenated cell-line and drug features.",
    category="general_purpose",
)
class NeuralNetworkPredictor(Predictor):
    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._model: FeedForwardNetwork | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "units_per_layer": [512, 256, 128],
            "dropout_prob": 0.2,
            "max_epochs": 50,
            "batch_size": 16,
        }

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        merged = {**self.get_default_hyperparameters(), **hyperparameters}
        self._hyperparameters = merged
        input_dim = int(input_dims.get("cell_line", 0)) + int(input_dims.get("drug", 0))
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
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = pair_context
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
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = pair_context
        if self._model is None or len(x) == 0:
            return np.full(len(x), np.nan, dtype=np.float64)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    def is_fitted(self) -> bool:
        return self._model is not None
