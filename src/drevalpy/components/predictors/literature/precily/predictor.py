"""Precily block literature predictor.

``torch`` and ``.model_utils`` are imported inside the methods that need them.
``drevalpy.registry`` imports this module to register the ``precily`` predictor on
``import drevalpy``, so a module-scope ``import torch`` put ~0.35s on the startup
path of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import fixed_entries
from drevalpy.components.predictors._pair_loader import make_pair_loader
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._early_stopping import run_epoch
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    declared_blocks,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.literature._torch_utils import DevicePicklingMixin
from drevalpy.experiment._tracking import log_metrics
from drevalpy.log import get_logger
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.accelerator import Accelerator

if TYPE_CHECKING:
    import torch

    from .model_utils import PrecilyNetwork

logger = get_logger(__name__)


@register(
    "precily",
    description="Precily pathway + SMILESVec model.",
    reference=PRECILY_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class PrecilyPredictor(DevicePicklingMixin, Predictor):
    """Registered Precily predictor consuming ModelInputBatch directly."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("pathways", FeatureFormat.NUMERIC_MATRIX))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("smilesvec", FeatureFormat.NUMERIC_MATRIX))
    supports_early_stopping: ClassVar[bool] = False

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the Precily predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: PrecilyNetwork | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space; Precily tunes nothing.

        :returns: Fixed specs for every Precily hyperparameter.
        """
        return fixed_entries(
            {
                "learning_rate": 1.0e-3,
                "dropout": 0.1,
                "epochs": 50,
                "batch_size": 128,
            }
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

        pathway_entity, drug_entity = declared_blocks(self, batch)
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

        pathway_entity, drug_entity = declared_blocks(self, batch)
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
