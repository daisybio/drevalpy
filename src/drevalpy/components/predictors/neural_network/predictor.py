"""Dense feed-forward neural network predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.predictors._lightning_training import LightningRun, reload_best_weights, run_lightning_fit
from drevalpy.components.predictors._pair_loader import collect_predictions, pair_loader
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

# torch, pytorch_lightning and .network (which imports lightning) are imported
# inside the methods that need them. This module is imported eagerly by
# register_builtins(), so a module-scope import would put ~1.2s of
# lightning on the critical path of every `import drevalpy`. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork


#: Layer widths DrEvalPy 1.6.1 tuned over: the union of the ``SimpleNeuralNetwork`` and
#: ``MultiViewNeuralNetwork`` grids.
UNITS_PER_LAYER_CHOICES: list[list[int]] = [
    [32, 16, 8, 4],
    [128, 64, 32],
    [64, 64, 32],
    [1024, 128, 64, 16],
    [16, 8, 4],
]


@register(
    "neuralNetwork",
    description="Dense feed-forward network on concatenated sample and drug features.",
    accelerator=Accelerator.OPTIONAL,
)
class NeuralNetworkPredictor(Predictor):
    """Neural network predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.matrix()
    drug_inputs: ClassVar[Inputs] = Inputs.matrix()
    supports_early_stopping: ClassVar[bool] = True

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: Optional overrides of ``default_hyperparameters``.
        """
        super().__init__(hyperparameters)
        self._model: FeedForwardNetwork | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs for the layer widths (the 1.6.1 grids of ``SimpleNeuralNetwork`` and
            ``MultiViewNeuralNetwork``), dropout, epoch budget and mini-batch size; the defaults are the first
            grid point of DrEvalPy 1.6.1 ``SimpleNeuralNetwork``. The early-stopping patience is fixed at 5, as
            it was there.
        """
        return {
            "units_per_layer": {"type": "categorical", "choices": UNITS_PER_LAYER_CHOICES, "default": [32, 16, 8, 4]},
            "dropout_prob": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.3},
            "max_epochs": {"type": "int", "low": 10, "high": 100, "default": 100},
            "batch_size": {"type": "int", "low": 8, "high": 64, "default": 16},
            "patience": {"type": "fixed", "default": 5},
        }

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data using lazy pair-level lookup.

        :param data: Training batch with the response and optional early-stopping pairs.
        """
        from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork

        self._model = None
        if len(data) == 0:
            return
        model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": self._hyperparameters["units_per_layer"],
                "dropout_prob": self._hyperparameters["dropout_prob"],
            },
            input_dim=self._compute_input_dim(data),
        )
        self._train_with_optional_early_stopping(data, model)
        self._model = model

    @staticmethod
    def _compute_input_dim(data: PredictorInput) -> int:
        """Determine the concatenated feature width from entity matrices.

        :param data: Input batch.
        :returns: Total feature dimensionality per pair.
        """
        dim = 0
        if data.sample.matrix.size > 0 and data.sample.matrix.ndim == 2:
            dim += data.sample.matrix.shape[1]
        if data.drug is not None and data.drug.matrix.size > 0 and data.drug.matrix.ndim == 2:
            dim += data.drug.matrix.shape[1]
        return dim

    def _build_pair_loader(
        self, data: PredictorInput, *, shuffle: bool, drop_last: bool = False, response: bool = True
    ) -> DataLoader:
        """Build a lazy pair loader over the sample and drug matrices of *data*, leaving out empty ones.

        :param data: Pairs to iterate: the training pairs, their ``stopping`` pairs, or pairs to score.
        :param shuffle: Whether to shuffle.
        :param drop_last: Whether to drop the last incomplete batch.
        :param response: Whether the batches carry the response.
        :returns: DataLoader.
        """
        batch_size = max(1, min(int(self._hyperparameters["batch_size"]), len(data)))
        samples = (data.sample.matrix,) if data.sample.matrix.size > 0 else ()
        drug = data.drug.matrix if data.drug is not None and data.drug.matrix.size > 0 else None
        return pair_loader(
            data, *samples, drug=drug, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, response=response
        )

    def _validation_loader(self, data: PredictorInput) -> DataLoader | None:
        """Build the early-stopping loader, if the batch carries usable validation pairs.

        :param data: Training batch.
        :returns: A validation loader, or ``None`` when there are no early-stopping pairs.
        """
        stopping = data.stopping
        if stopping is None or len(stopping) == 0:
            return None
        return self._build_pair_loader(stopping, shuffle=False)

    def _train_with_optional_early_stopping(self, data: PredictorInput, model: FeedForwardNetwork) -> None:
        """Train *model*, early-stopping on validation loss when validation pairs exist.

        :param data: Training batch.
        :param model: The freshly initialised network.
        """
        batch_size = min(int(self._hyperparameters["batch_size"]), len(data))
        train_loader = self._build_pair_loader(data, shuffle=True, drop_last=batch_size < len(data))
        checkpoint = run_lightning_fit(
            model,
            train_loader,
            self._validation_loader(data),
            LightningRun(
                max_epochs=int(self._hyperparameters["max_epochs"]),
                patience=int(self._hyperparameters["patience"]),
                checkpoint_dir=data.training_context.checkpoint_dir,
                devices=1,
                logger=False,
            ),
        )
        reload_best_weights(model, checkpoint)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict using lazy pair-level lookup (no full feature expansion).

        :param data: Featurized pairs to score.
        :returns: Predicted responses.
        """
        import torch

        if self._model is None or len(data) == 0:
            return np.full(len(data), np.nan, dtype=np.float64)

        loader = self._build_pair_loader(data, shuffle=False, response=False)

        model = self._model
        return collect_predictions(model, loader, lambda tensors: model(torch.cat(tensors, dim=1)))
