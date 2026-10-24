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
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._early_stopping import run_epoch
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.experiment._tracking import log_metrics
from drevalpy.log import get_logger
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.tensor_data import make_pair_loader
from drevalpy.utils.torch_io import load_state_dict, save_state_dict, save_trusted_mapping

if TYPE_CHECKING:
    import torch

    from .model_utils import PrecilyNetwork

logger = get_logger(__name__)


@register(
    "precily",
    description="Precily pathway + SMILESVec model.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PRECILY_REFERENCE,
)
class PrecilyPredictor(BlockPredictor):
    """Registered Precily predictor consuming ModelInputBatch directly."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("pathways",)
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
            batch.sample.blocks["pathways"].values,
            batch.drug.blocks["smilesvec"].values,
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
        import torch
        import torch.nn as nn
        import torch.optim as optim

        pathway_entity, drug_entity = self._entity_blocks(batch)
        model = self._build_model(
            pathway_entity.shape[1] + drug_entity.shape[1],
            self._hyperparameters["dropout"],
        )
        self._model = model

        train_loader = make_pair_loader(
            (pathway_entity, batch.sample_pair_idx),
            (drug_entity, require_drug_pair_idx(batch.drug_pair_idx)),
            response=np.asarray(batch.response, dtype=np.float32),
            batch_size=self._hyperparameters["batch_size"],
            shuffle=True,
        )

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self._hyperparameters["learning_rate"])

        def batch_loss(loader_batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            pathway_inputs, drug_inputs, targets = (tensor.to(self._device) for tensor in loader_batch)
            return loss_func(model(torch.cat([pathway_inputs, drug_inputs], dim=1)), targets)

        epochs = self._hyperparameters["epochs"]
        for epoch in range(epochs):
            epoch_loss = run_epoch(model, train_loader, batch_loss, optimizer)
            logger.info("Precily: Epoch [%d/%d] Training Loss: %.4f", epoch + 1, epochs, epoch_loss)
            log_metrics({"epoch": epoch + 1, "train_loss": epoch_loss})

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
                sample_blocks=(pathway_entity,),
                drug_blocks=(drug_entity,),
                batch_size=self._hyperparameters["batch_size"],
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
        payload = load_payload_state(state, "Precily")
        input_dim = payload.get("input_dim")
        model_state = payload.get("model_state")
        if not isinstance(input_dim, int) or not isinstance(model_state, (bytes, bytearray)):
            msg = "Precily payload is missing input_dim or model_state"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._model = self._build_model(input_dim, float(self._hyperparameters["dropout"]))
        self._model.load_state_dict(load_state_dict(bytes(model_state)))
