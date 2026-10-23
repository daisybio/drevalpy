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
from drevalpy.components.predictors._state_helpers import load_payload_state
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._early_stopping import (
    EarlyStoppingRun,
    run_epoch,
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
from drevalpy.utils.torch_io import load_state_dict, save_state_dict, save_trusted_mapping

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from .model_utils import CombinedModel


def _build_combined_model(gene_input_size: int, hyperparameters: dict[str, Any], device: torch.device) -> CombinedModel:
    """Construct the combined PharmaFormer network on *device*.

    :param gene_input_size: Width of the gene expression block.
    :param hyperparameters: Hyperparameters holding the architecture keys.
    :param device: Device the network is moved to.
    :returns: Initialized network.
    """
    from .model_utils import CombinedModel

    return CombinedModel(
        gene_input_size=gene_input_size,
        gene_hidden_size=hyperparameters["gene_hidden_size"],
        drug_hidden_size=hyperparameters["drug_hidden_size"],
        feature_dim=hyperparameters["feature_dim"],
        nhead=hyperparameters["nhead"],
        num_layers=hyperparameters["num_layers"],
        dim_feedforward=hyperparameters["dim_feedforward"],
        dropout=hyperparameters["dropout"],
    ).to(device)


@register(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PHARMAFORMER_REFERENCE,
)
class PharmaFormerPredictor(BlockPredictor):
    """Registered PharmaFormer predictor consuming ModelInputBatch directly."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    required_sample_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),
    )
    supports_early_stopping: ClassVar[bool] = True
    requires_early_stopping: ClassVar[bool] = True
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

        :returns: Hyperparameter search space specs.
        """
        return {}

    def _entity_blocks(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return the entity-level gene expression and BPE SMILES matrices.

        :param batch: Batch to read the blocks from.
        :returns: Tuple of ``(gene_expression, bpe_smiles)`` matrices.
        """
        return (
            batch.sample.blocks["gene_expression"].values,
            batch.drug.blocks["bpe_smiles"].values,
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
            (gene_entity, batch.sample_pair_idx),
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

        def batch_loss(loader_batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            gene_inputs, smiles_inputs, targets = (tensor.to(self._device) for tensor in loader_batch)
            return loss_func(model(gene_inputs, smiles_inputs).squeeze(), targets)

        train_with_early_stopping(
            model,
            EarlyStoppingRun(
                epochs=self._hyperparameters["epochs"],
                patience=self._hyperparameters["patience"],
                checkpoint_dir=batch.training_context.checkpoint_dir,
                model_name="PharmaFormer",
                verbose=True,
            ),
            train_epoch=lambda: run_epoch(model, train_loader, batch_loss, optimizer),
            val_epoch=lambda: run_epoch(model, val_loader, batch_loss, None),
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
                sample_blocks=(gene_entity,),
                drug_blocks=(drug_entity,),
                batch_size=self._hyperparameters["batch_size"],
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
        payload = load_payload_state(state, "PharmaFormer")
        gene_input_size = payload.get("gene_input_size")
        model_state = payload.get("model_state")
        if not isinstance(gene_input_size, int) or not isinstance(model_state, (bytes, bytearray)):
            msg = "PharmaFormer payload is missing gene_input_size or model_state"
            raise PredictorStateError(msg)
        self._hyperparameters = dict(payload["predictor_hyperparameters"])
        self._gene_input_size = gene_input_size
        self._model = _build_combined_model(gene_input_size, self._hyperparameters, self._device)
        self._model.load_state_dict(load_state_dict(bytes(model_state)))
