"""Precily block literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._tensor_data import make_pair_loader
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature._torch_state import (
    load_object_mapping,
    load_state_dict,
    save_object_mapping,
    save_state_dict,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode

from .model_utils import PrecilyNetwork


@register_predictor(
    "precily",
    description="Precily pathway + SMILESVec model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PRECILY_REFERENCE,
)
class PrecilyPredictor(BlockPredictor):
    """Registered Precily predictor consuming ModelInputBatch directly."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("pathways",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("smilesvec",)
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the Precily predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: PrecilyNetwork | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters for Precily.

        :returns: Default hyperparameter mapping.
        """
        return {
            "learning_rate": 1.0e-3,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 128,
            "seed": 42,
        }

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the Precily model on the batch.

        :param batch: Training batch with pathways and smilesvec blocks.
        """
        pathway_entity = batch.cell_line_blocks["pathways"].values
        drug_entity = batch.drug_blocks["smilesvec"].values
        cell_pair_idx = batch.cell_line_pair_idx
        drug_pair_idx = batch.drug_pair_idx
        response = np.asarray(batch.response, dtype=np.float32)

        n_pathways = pathway_entity.shape[1]
        drug_dim = drug_entity.shape[1]
        input_dim = n_pathways + drug_dim

        self._model = PrecilyNetwork(
            input_dim=input_dim,
            dropout=self._hyperparameters.get("dropout", 0.1),
        ).to(self._device)

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self._hyperparameters["learning_rate"])

        train_loader = make_pair_loader(
            (pathway_entity, cell_pair_idx),
            (drug_entity, drug_pair_idx),
            response=response,
            batch_size=self._hyperparameters["batch_size"],
            shuffle=True,
        )

        for epoch in range(self._hyperparameters["epochs"]):
            self._model.train()
            epoch_loss = 0.0
            batch_count = 0
            for pathway_inputs, drug_inputs, targets in train_loader:
                pathway_inputs = pathway_inputs.to(self._device)
                drug_inputs = drug_inputs.to(self._device)
                targets = targets.to(self._device)

                x = torch.cat([pathway_inputs, drug_inputs], dim=1)
                outputs = self._model(x)
                loss = loss_func(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                batch_count += 1

            epoch_loss /= max(batch_count, 1)
            print(f"Precily: Epoch [{epoch + 1}/{self._hyperparameters['epochs']}] " f"Training Loss: {epoch_loss:.4f}")

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Run Precily inference on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "Precily model not initialized."
            raise ValueError(msg)

        pathway_entity = batch.cell_line_blocks["pathways"].values
        drug_entity = batch.drug_blocks["smilesvec"].values
        cell_pair_idx = batch.cell_line_pair_idx
        drug_pair_idx = batch.drug_pair_idx

        predict_loader = make_pair_loader(
            (pathway_entity, cell_pair_idx),
            (drug_entity, drug_pair_idx),
            batch_size=self._hyperparameters.get("batch_size", 128),
            shuffle=False,
        )

        self._model.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for pathway_inputs, drug_inputs in predict_loader:
                pathway_inputs = pathway_inputs.to(self._device)
                drug_inputs = drug_inputs.to(self._device)
                x = torch.cat([pathway_inputs, drug_inputs], dim=1)
                outputs = self._model(x)
                if outputs.numel() > 1:
                    predictions += outputs.cpu().tolist()
                else:
                    predictions += [outputs.item()]
        return np.asarray(predictions)

    def is_fitted(self) -> bool:
        """Return whether the model has been trained.

        :returns: True when the model is initialized.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob when fitted, else empty.
        :raises TypeError: If the network architecture is unexpected.
        """
        if self._model is None:
            return {}
        first_layer = self._model.net[0]
        if not isinstance(first_layer, nn.Linear):
            msg = "PrecilyNetwork must start with a Linear layer"
            raise TypeError(msg)
        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "input_dim": int(first_layer.in_features),
            "model_state": save_state_dict(self._model.state_dict()),
        }
        return {"payload": save_object_mapping(payload)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state containing a payload byte blob.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        blob = state.get("payload")
        if not isinstance(blob, (bytes, bytearray)):
            msg = "PrecilyPredictor state requires a payload byte blob"
            raise PredictorStateError(msg)
        try:
            payload = load_object_mapping(bytes(blob))
        except Exception as exc:
            msg = "PrecilyPredictor payload could not be deserialized"
            raise PredictorStateError(msg) from exc
        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = "PrecilyPredictor payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        input_dim = payload.get("input_dim")
        model_state = payload.get("model_state")
        if isinstance(input_dim, int) and isinstance(model_state, (bytes, bytearray)):
            self._model = PrecilyNetwork(
                input_dim=input_dim,
                dropout=float(hyperparameters.get("dropout", 0.1)),
            )
            self._model.load_state_dict(load_state_dict(bytes(model_state)))
            self._model.to(self._device)
