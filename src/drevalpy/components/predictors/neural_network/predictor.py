"""Dense feed-forward neural network predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.predictors._lightning_training import LightningRun, reload_best_weights, run_lightning_fit
from drevalpy.components.predictors._pair_loader import collect_predictions, make_pair_loader
from drevalpy.components.predictors.abstract.base import Predictor, response_rows
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.accelerator import Accelerator

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
        self._is_fitted = False

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

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data using lazy pair-level lookup.

        :param batch: Training batch with the response and optional early-stopping pairs.
        """
        from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork

        self._model = FeedForwardNetwork(
            hyperparameters={
                "units_per_layer": self._hyperparameters["units_per_layer"],
                "dropout_prob": self._hyperparameters["dropout_prob"],
            },
            input_dim=self._compute_input_dim(batch),
        )
        self._is_fitted = False
        if batch.n_pairs == 0:
            return
        self._train_with_optional_early_stopping(batch, self._model)
        self._is_fitted = True

    @staticmethod
    def _compute_input_dim(batch: ModelInputBatch) -> int:
        """Determine the concatenated feature width from entity matrices.

        :param batch: Input batch.
        :returns: Total feature dimensionality per pair.
        """
        dim = 0
        if batch.sample.matrix.size > 0 and batch.sample.matrix.ndim == 2:
            dim += batch.sample.matrix.shape[1]
        if batch.drug is not None and batch.drug.matrix.size > 0 and batch.drug.matrix.ndim == 2:
            dim += batch.drug.matrix.shape[1]
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
    ) -> DataLoader:
        """Build a lazy pair loader from entity matrices and pair indices.

        :param batch: Batch providing entity-level feature matrices.
        :param cell_pair_idx: Sample row index per pair.
        :param drug_pair_idx: Drug pair indices (or None).
        :param response: Optional response array.
        :param shuffle: Whether to shuffle.
        :param drop_last: Whether to drop last incomplete batch.
        :returns: DataLoader.
        """
        batch_size = min(int(self._hyperparameters["batch_size"]), len(cell_pair_idx))
        if batch_size < 1:
            batch_size = 1

        specs: list[tuple[np.ndarray, np.ndarray]] = []
        if batch.sample.matrix.size > 0:
            specs.append((batch.sample.matrix, cell_pair_idx))
        if batch.drug is not None and batch.drug.matrix.size > 0 and drug_pair_idx is not None:
            specs.append((batch.drug.matrix, drug_pair_idx))

        return make_pair_loader(
            *specs,
            response=response,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def _validation_loader(self, batch: ModelInputBatch) -> DataLoader | None:
        """Build the early-stopping loader, if the batch carries usable validation pairs.

        :param batch: Training batch.
        :returns: A validation loader, or ``None`` when there are no early-stopping responses.
        """
        es_resp = batch.early_stopping_response
        if es_resp is None or len(es_resp) == 0 or es_resp.response is None:
            return None
        y_val = np.asarray(es_resp.response, dtype=np.float32).reshape(-1)
        if len(y_val) == 0:
            return None
        es_cell_idx, es_drug_idx = response_rows(batch, es_resp)
        return self._build_pair_loader(batch, es_cell_idx, es_drug_idx, y_val, shuffle=False, drop_last=False)

    def _train_with_optional_early_stopping(self, batch: ModelInputBatch, model: FeedForwardNetwork) -> None:
        """Train *model*, early-stopping on validation loss when validation pairs exist.

        :param batch: Training batch.
        :param model: The freshly initialised network.
        """
        y = np.asarray(batch.response, dtype=np.float32).reshape(-1)
        batch_size = min(int(self._hyperparameters["batch_size"]), batch.n_pairs)
        train_loader = self._build_pair_loader(
            batch,
            batch.sample_pair_idx,
            batch.drug_pair_idx,
            y,
            shuffle=True,
            drop_last=batch_size < batch.n_pairs,
        )
        checkpoint = run_lightning_fit(
            model,
            train_loader,
            self._validation_loader(batch),
            LightningRun(
                max_epochs=int(self._hyperparameters["max_epochs"]),
                patience=int(self._hyperparameters["patience"]),
                checkpoint_dir=batch.training_context.checkpoint_dir,
                devices=1,
                logger=False,
            ),
        )
        reload_best_weights(model, checkpoint)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict using lazy pair-level lookup (no full feature expansion).

        :param batch: Featurized pairs to score.
        :returns: Predicted responses.
        """
        import torch

        if not self._is_fitted or self._model is None or batch.n_pairs == 0:
            return np.full(batch.n_pairs, np.nan, dtype=np.float64)

        loader = self._build_pair_loader(
            batch,
            batch.sample_pair_idx,
            batch.drug_pair_idx,
            response=None,
            shuffle=False,
            drop_last=False,
        )

        model = self._model
        return collect_predictions(model, loader, lambda tensors: model(torch.cat(tensors, dim=1)))
