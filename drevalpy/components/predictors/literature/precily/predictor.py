"""Precily block literature predictor.

``torch`` and ``.model_utils`` are imported inside the methods that need them.
``drevalpy.registry`` imports this module to register the ``precily`` predictor on
``import drevalpy``, so a module-scope ``import torch`` put ~0.35s on the startup
path of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.tensor_data import make_pair_loader
from drevalpy.utils.torch_io import (
    load_state_dict,
    load_trusted_mapping,
    save_state_dict,
    save_trusted_mapping,
)

if TYPE_CHECKING:
    from torch import nn, optim
    from torch.utils.data import DataLoader

    from .model_utils import PrecilyNetwork


@register(
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
        import torch

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

    def _entity_blocks(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return the entity-level pathway and SMILESVec matrices.

        :param batch: Batch to read the blocks from.
        :returns: Tuple of ``(pathways, smilesvec)`` matrices.
        """
        return (
            batch.cell_line_blocks["pathways"].values,
            batch.drug_blocks["smilesvec"].values,
        )

    def _build_model(self, input_dim: int, dropout: float) -> PrecilyNetwork:
        """Construct the Precily network on the target device.

        :param input_dim: Concatenated pathway plus drug feature width.
        :param dropout: Dropout probability between hidden layers.
        :returns: Initialized network.
        """
        from .model_utils import PrecilyNetwork

        return PrecilyNetwork(input_dim=input_dim, dropout=dropout).to(self._device)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the Precily model on the batch.

        :param batch: Training batch with pathways and smilesvec blocks.
        """
        import torch.nn as nn
        import torch.optim as optim

        pathway_entity, drug_entity = self._entity_blocks(batch)
        model = self._build_model(
            pathway_entity.shape[1] + drug_entity.shape[1],
            self._hyperparameters.get("dropout", 0.1),
        )
        self._model = model

        train_loader = make_pair_loader(
            (pathway_entity, batch.cell_line_pair_idx),
            (drug_entity, require_drug_pair_idx(batch.drug_pair_idx)),
            response=np.asarray(batch.response, dtype=np.float32),
            batch_size=self._hyperparameters["batch_size"],
            shuffle=True,
        )

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self._hyperparameters["learning_rate"])
        epochs = self._hyperparameters["epochs"]
        for epoch in range(epochs):
            epoch_loss = self._run_epoch(model, train_loader, loss_func, optimizer)
            print(f"Precily: Epoch [{epoch + 1}/{epochs}] Training Loss: {epoch_loss:.4f}")

    def _run_epoch(
        self,
        model: PrecilyNetwork,
        loader: DataLoader,
        loss_func: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        """Run one training epoch over *loader*.

        :param model: Network being trained.
        :param loader: Training loader yielding pathway, drug and target tensors.
        :param loss_func: Loss function.
        :param optimizer: Optimizer to step.
        :returns: Mean epoch loss.
        """
        import torch

        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for pathway_inputs, drug_inputs, targets in loader:
            x = torch.cat([pathway_inputs.to(self._device), drug_inputs.to(self._device)], dim=1)
            loss = loss_func(model(x), targets.to(self._device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            batch_count += 1
        return epoch_loss / max(batch_count, 1)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Run Precily inference on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "Precily model not initialized."
            raise ValueError(msg)

        pathway_entity, drug_entity = self._entity_blocks(batch)
        return predict_pairs(
            self._model,
            batch,
            PairEvalSpec(
                cell_line_blocks=(pathway_entity,),
                drug_blocks=(drug_entity,),
                batch_size=self._hyperparameters.get("batch_size", 128),
                device=self._device,
            ),
            forward=concatenated_forward(self._model),
        )

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
        import torch.nn as nn

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
        return {"payload": save_trusted_mapping(payload)}

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
            payload = load_trusted_mapping(bytes(blob))
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
            self._model = self._build_model(input_dim, float(hyperparameters.get("dropout", 0.1)))
            self._model.load_state_dict(load_state_dict(bytes(model_state)))
