"""Precily block literature predictor.

``torch`` and ``.model_utils`` are imported inside the methods that need them.
``drevalpy.registry`` imports this module to register the ``precily`` predictor on
``import drevalpy``, so a module-scope ``import torch`` put ~0.35s on the startup
path of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors._pair_loader import (
    concatenated_forward,
    declared_blocks,
    pair_loader,
    predict_pairs,
)
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._torch_training import DevicePicklingMixin, run_epoch
from drevalpy.experiment._tracking import log_metrics
from drevalpy.log import get_logger
from drevalpy.registry._component import LiteratureReference
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

if TYPE_CHECKING:
    import torch

    from .model_utils import PrecilyNetwork

logger = get_logger(__name__)

_REFERENCE = LiteratureReference(
    repo_url="https://github.com/SmritiChawla/Precily",
    citation_doi="10.1038/s41467-022-33291-z",
    citation_text="Precily pathway and SMILESVec drug response model.",
    deviations=(
        "Hidden widths match the published pathways Model1 checkpoint. Use one regressor instead of the "
        "five-model ensemble, fold-specific GSVA and configurable drug features; PyTorch "
        "initialization/optimizer defaults differ from Keras."
    ),
)


@register(
    "precily",
    description="Precily pathway + SMILESVec model.",
    reference=_REFERENCE,
    license="NOASSERTION",
    accelerator=Accelerator.OPTIONAL,
)
class PrecilyPredictor(DevicePicklingMixin, Predictor):
    """Registered Precily predictor consuming PredictorInput directly."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("pathways"))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("smilesvec"))
    supports_early_stopping: ClassVar[bool] = False

    fixed_hyperparameters = {
        "learning_rate": 1.0e-3,
        "dropout": 0.1,
        "epochs": 50,
        "batch_size": 128,
    }

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the Precily predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: PrecilyNetwork | None = None

    def _build_model(self, input_dim: int, dropout: float) -> PrecilyNetwork:
        """Construct the Precily network on the target device.

        :param input_dim: Concatenated pathway plus drug feature width.
        :param dropout: Dropout probability between hidden layers.
        :returns: Initialized network.
        """
        from .model_utils import PrecilyNetwork

        return PrecilyNetwork(input_dim=input_dim, dropout=dropout).to(self._device)

    def _fit(self, data: PredictorInput) -> None:
        """Train the Precily model on the batch.

        :param data: Training batch with pathways and smilesvec blocks.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        pathway_entity, drug_entity = declared_blocks(self, data)
        model = self._build_model(
            pathway_entity.shape[1] + drug_entity.shape[1],
            self._hyperparameters["dropout"],
        )
        self._model = model

        train_loader = pair_loader(
            data, pathway_entity, drug=drug_entity, batch_size=self._hyperparameters["batch_size"], shuffle=True
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

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Run Precily inference on the given batch.

        :param data: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "Precily model not initialized."
            raise ValueError(msg)

        pathway_entity, drug_entity = declared_blocks(self, data)
        return predict_pairs(
            self._model,
            data,
            pathway_entity,
            drug_entity,
            batch_size=self._hyperparameters["batch_size"],
            device=self._device,
            forward=concatenated_forward(self._model),
        )
