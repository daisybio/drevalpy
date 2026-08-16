"""PharmaFormer block literature predictor.

``torch`` and ``.model_utils`` are imported inside the functions that need them.
``drevalpy.registry`` imports this module to register the ``pharmaFormer``
predictor on ``import drevalpy``, so a module-scope ``import torch`` put ~0.35s on
the startup path of every CLI invocation. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    train_with_early_stopping,
)
from drevalpy.components.predictors.literature._metadata import PHARMAFORMER_REFERENCE
from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models.config import PredictionMode
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.tensor_data import make_pair_loader
from drevalpy.utils.torch_io import (
    load_state_dict,
    load_trusted_mapping,
    save_state_dict,
    save_trusted_mapping,
)

if TYPE_CHECKING:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    from .model_utils import CombinedModel


def _build_combined_model(gene_input_size: int, hyperparameters: dict[str, Any], device: torch.device) -> CombinedModel:
    from .model_utils import CombinedModel

    return CombinedModel(
        gene_input_size=gene_input_size,
        gene_hidden_size=hyperparameters["gene_hidden_size"],
        drug_hidden_size=hyperparameters["drug_hidden_size"],
        feature_dim=hyperparameters["feature_dim"],
        nhead=hyperparameters["nhead"],
        num_layers=hyperparameters.get("num_layers", 3),
        dim_feedforward=hyperparameters.get("dim_feedforward", 2048),
        dropout=hyperparameters.get("dropout", 0.1),
    ).to(device)


def _run_epoch(
    model: CombinedModel,
    loader: DataLoader,
    loss_func: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one train or validation epoch.

    :param model: Combined PharmaFormer model.
    :param loader: Training or validation data loader.
    :param loss_func: Loss function.
    :param optimizer: Optimizer for training; ``None`` for eval-only.
    :param device: Torch device.
    :returns: Mean epoch loss.
    """
    import torch

    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    batch_count = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for gene_inputs, smiles_inputs, batch_targets in loader:
            gene_inputs = gene_inputs.to(device)
            smiles_inputs = smiles_inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(gene_inputs, smiles_inputs)
            loss = loss_func(outputs.squeeze(), batch_targets)

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            else:
                epoch_loss += loss.item()

            batch_count += 1

    return epoch_loss / max(batch_count, 1)


@register(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PHARMAFORMER_REFERENCE,
)
class PharmaFormerPredictor(BlockPredictor):
    """Registered PharmaFormer predictor consuming ModelInputBatch directly."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),
    )
    supports_early_stopping: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the PharmaFormer predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        import torch

        super().__init__(hyperparameters)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: CombinedModel | None = None
        self._gene_input_size: int | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters for PharmaFormer.

        :returns: Default hyperparameter mapping.
        """
        return {
            "gene_hidden_size": 2048,
            "drug_hidden_size": 128,
            "feature_dim": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "batch_size": 64,
            "lr": 0.00001,
            "epochs": 100,
            "patience": 10,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {}

    def _entity_blocks(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return the entity-level gene expression and BPE SMILES matrices.

        :param batch: Batch to read the blocks from.
        :returns: Tuple of ``(gene_expression, bpe_smiles)`` matrices.
        """
        return (
            batch.cell_line_blocks["gene_expression"].values,
            batch.drug_blocks["bpe_smiles"].values,
        )

    def _build_loaders(self, batch: ModelInputBatch) -> tuple[DataLoader, DataLoader]:
        """Build the training and early-stopping loaders.

        :param batch: Training batch with an early-stopping response.
        :returns: Tuple of ``(train_loader, val_loader)``.
        :raises ValueError: If early stopping data is not provided.
        """
        if batch.early_stopping_response is None:
            msg = "PharmaFormer model requires early stopping data."
            raise ValueError(msg)

        gene_entity, drug_entity = self._entity_blocks(batch)
        batch_size = self._hyperparameters["batch_size"]

        train_loader = make_pair_loader(
            (gene_entity, batch.cell_line_pair_idx),
            (drug_entity, require_drug_pair_idx(batch.drug_pair_idx)),
            response=np.asarray(batch.response, dtype=np.float32),
            batch_size=batch_size,
            shuffle=True,
        )

        es_response = batch.early_stopping_response
        es_cell_pair_idx, es_drug_pair_idx = batch._pair_indices_for(es_response)
        val_loader = make_pair_loader(
            (gene_entity, es_cell_pair_idx),
            (drug_entity, require_drug_pair_idx(es_drug_pair_idx)),
            response=np.asarray(es_response.response, dtype=np.float32),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the PharmaFormer model with early stopping.

        :param batch: Training batch with gene_expression and bpe_smiles blocks.
        """
        import torch.nn as nn
        import torch.optim as optim

        train_loader, val_loader = self._build_loaders(batch)

        gene_entity, _ = self._entity_blocks(batch)
        self._gene_input_size = gene_entity.shape[1]
        model = _build_combined_model(self._gene_input_size, self._hyperparameters, self._device)
        self._model = model

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self._hyperparameters["lr"])

        train_with_early_stopping(
            model,
            EarlyStoppingRun(
                epochs=self._hyperparameters["epochs"],
                patience=self._hyperparameters.get("patience", 10),
                checkpoint_dir=batch.training_context.checkpoint_dir,
                model_name="PharmaFormer",
                verbose=True,
            ),
            train_epoch=lambda: _run_epoch(model, train_loader, loss_func, optimizer, self._device),
            val_epoch=lambda: _run_epoch(model, val_loader, loss_func, None, self._device),
            device=self._device,
        )

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Run PharmaFormer inference on the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises ValueError: If the model has not been trained yet.
        """
        if self._model is None:
            msg = "PharmaFormer model not initialized."
            raise ValueError(msg)

        gene_entity, drug_entity = self._entity_blocks(batch)
        return predict_pairs(
            self._model,
            batch,
            PairEvalSpec(
                cell_line_blocks=(gene_entity,),
                drug_blocks=(drug_entity,),
                batch_size=self._hyperparameters.get("batch_size", 64),
                device=self._device,
            ),
        )

    def is_fitted(self) -> bool:
        """Return whether the model has been trained.

        :returns: True when the model is initialized.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob when fitted, else empty.
        """
        if self._model is None:
            return {}
        payload: dict[str, Any] = {
            "predictor_hyperparameters": dict(self._hyperparameters),
            "gene_input_size": self._gene_input_size,
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
            msg = "PharmaFormerPredictor state requires a payload byte blob"
            raise PredictorStateError(msg)
        try:
            payload = load_trusted_mapping(bytes(blob))
        except Exception as exc:
            msg = "PharmaFormerPredictor payload could not be deserialized"
            raise PredictorStateError(msg) from exc
        hyperparameters = payload.get("predictor_hyperparameters")
        if not isinstance(hyperparameters, dict):
            msg = "PharmaFormerPredictor payload is missing predictor_hyperparameters"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(hyperparameters)
        gene_input_size = payload.get("gene_input_size")
        model_state = payload.get("model_state")
        if isinstance(gene_input_size, int) and isinstance(model_state, (bytes, bytearray)):
            self._gene_input_size = gene_input_size
            self._model = _build_combined_model(gene_input_size, self._hyperparameters, self._device)
            self._model.load_state_dict(load_state_dict(bytes(model_state)))
            self._model.to(self._device)
