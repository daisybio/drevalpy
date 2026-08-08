"""DrugGNN block predictor – GCN on molecular graphs with dense cell-line features."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset as PytorchDataset
from torch_geometric.loader import DataLoader

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.literature._metadata import DRUGGNN_REFERENCE
from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNNModule
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import PredictionMode


class _DrugGNNDataset(PytorchDataset):
    """PyTorch Dataset that yields (drug_graph, cell_features, response) tuples.

    Accepts pre-built index arrays mapping each pair to entity-level features.
    """

    def __init__(
        self,
        cell_line_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray,
        cell_line_matrix: np.ndarray,
        drug_graphs: np.ndarray,
        response: np.ndarray,
    ) -> None:
        """Initialize the DrugGNN dataset.

        :param cell_line_pair_idx: Pair-level indices into the cell-line matrix.
        :param drug_pair_idx: Pair-level indices into the drug graphs array.
        :param cell_line_matrix: Cell-line feature matrix.
        :param drug_graphs: Array of PyG Data objects for drugs.
        :param response: Response values for each pair.
        """
        self._cl_pair_idx = cell_line_pair_idx
        self._drug_pair_idx = drug_pair_idx
        self._cl_tensors = torch.as_tensor(cell_line_matrix, dtype=torch.float32)
        self._drug_graphs = drug_graphs
        self._response = torch.as_tensor(response, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples.

        :returns: Dataset length.
        """
        return len(self._response)

    def __getitem__(self, idx: int):
        """Return (drug_graph, cell_features, response) for the given index.

        :param idx: Sample index.
        :returns: Tuple of drug graph, cell tensor, and response scalar.
        """
        cl_idx = self._cl_pair_idx[idx]
        drug_idx = self._drug_pair_idx[idx]
        return self._drug_graphs[drug_idx], self._cl_tensors[cl_idx], self._response[idx]


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.GRAPH,
    reference=DRUGGNN_REFERENCE,
)
class DrugGNNPredictor(BlockPredictor):
    """Registered DrugGNN predictor consuming ModelInputBatch directly."""

    supports_early_stopping: ClassVar[bool] = True
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("drug_graph",)
    required_cell_line_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
    )
    required_drug_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("drug_graph", FeatureFormat.GRAPH),)
    validate_drug_graphs: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the DrugGNN predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: DrugGNNModule | None = None
        self._num_node_features: int = 0
        self._num_cell_features: int = 0

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "learning_rate": 0.001,
            "epochs": 2,
            "hidden_dim": 64,
            "dropout": 0.2,
            "batch_size": 8,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Ray Tune-style hyperparameter specs.
        """
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_drug_pair_idx(batch: ModelInputBatch) -> np.ndarray:
        """Return drug_pair_idx, computing it from entity IDs if absent.

        :param batch: Input batch.
        :returns: Array mapping each pair to a drug entity index.
        :raises ValueError: If neither drug_pair_idx nor drug_entity_ids are available.
        """
        if batch.drug_pair_idx is not None:
            return batch.drug_pair_idx
        if batch.drug_entity_ids is None:
            msg = "DrugGNN requires either drug_pair_idx or drug_entity_ids"
            raise ValueError(msg)
        entity_map = {str(eid): i for i, eid in enumerate(batch.drug_entity_ids)}
        return np.array([entity_map[str(did)] for did in batch.drug_ids], dtype=np.int64)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the DrugGNN model on the batch.

        :param batch: Training batch with gene_expression and drug_graph blocks.
        :raises ValueError: If response data or drug features are missing.
        """
        cell_line_matrix = batch.cell_line_blocks["gene_expression"].values
        drug_graphs = batch.drug_blocks["drug_graph"].values
        if batch.response is None:
            msg = "DrugGNN requires training response data"
            raise ValueError(msg)

        drug_pair_idx = self._resolve_drug_pair_idx(batch)

        self._num_node_features = int(drug_graphs[0].num_node_features)
        self._num_cell_features = int(cell_line_matrix.shape[1])

        self._model = DrugGNNModule(
            num_node_features=self._num_node_features,
            num_cell_features=self._num_cell_features,
            hidden_dim=int(self._hyperparameters.get("hidden_dim", 64)),
            dropout=float(self._hyperparameters.get("dropout", 0.2)),
            learning_rate=float(self._hyperparameters.get("learning_rate", 0.001)),
        )

        train_dataset = _DrugGNNDataset(
            cell_line_pair_idx=batch.cell_line_pair_idx,
            drug_pair_idx=drug_pair_idx,
            cell_line_matrix=cell_line_matrix,
            drug_graphs=drug_graphs,
            response=batch.response,
        )
        batch_size = int(self._hyperparameters.get("batch_size", 1024))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = self._build_val_loader(batch, cell_line_matrix, drug_graphs)

        callbacks: list[pl.callbacks.Callback] | None = None
        if val_loader is not None:
            callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)]

        trainer = pl.Trainer(
            max_epochs=int(self._hyperparameters.get("epochs", 100)),
            accelerator="auto",
            devices="auto",
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=True,
            log_every_n_steps=int(self._hyperparameters.get("log_every_n_steps", 50)),
            precision=self._hyperparameters.get("precision", 32),
        )
        trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def _build_val_loader(
        self,
        batch: ModelInputBatch,
        cell_line_matrix: np.ndarray,
        drug_graphs: np.ndarray,
    ) -> DataLoader | None:
        """Build a validation DataLoader from the early-stopping response if available.

        :param batch: Full training batch (for entity ID lookups).
        :param cell_line_matrix: Cell-line feature matrix.
        :param drug_graphs: Array of drug graph objects.
        :returns: Validation DataLoader or None if no early-stopping data.
        """
        es = batch.early_stopping_response
        if es is None or len(es) == 0:
            return None

        cl_entity_map = {str(eid): i for i, eid in enumerate(batch.cell_line_entity_ids)}
        drug_entity_map: dict[str, int] = {}
        if batch.drug_entity_ids is not None:
            drug_entity_map = {str(eid): i for i, eid in enumerate(batch.drug_entity_ids)}

        val_cl_idx = np.array([cl_entity_map[str(cid)] for cid in es.cell_line_ids], dtype=np.int64)
        val_drug_idx = np.array([drug_entity_map[str(did)] for did in es.drug_ids], dtype=np.int64)

        val_dataset = _DrugGNNDataset(
            cell_line_pair_idx=val_cl_idx,
            drug_pair_idx=val_drug_idx,
            cell_line_matrix=cell_line_matrix,
            drug_graphs=drug_graphs,
            response=es.response,
        )
        return DataLoader(
            val_dataset,
            batch_size=int(self._hyperparameters.get("batch_size", 32)),
            num_workers=0,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises RuntimeError: If the model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        if batch.n_pairs == 0:
            return np.array([])

        drug_pair_idx = self._resolve_drug_pair_idx(batch)

        cell_line_matrix = batch.cell_line_blocks["gene_expression"].values
        drug_graphs = batch.drug_blocks["drug_graph"].values

        predict_dataset = _DrugGNNDataset(
            cell_line_pair_idx=batch.cell_line_pair_idx,
            drug_pair_idx=drug_pair_idx,
            cell_line_matrix=cell_line_matrix,
            drug_graphs=drug_graphs,
            response=np.zeros(batch.n_pairs, dtype=np.float32),
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=int(self._hyperparameters.get("batch_size", 32)),
            num_workers=0,
            pin_memory=True,
        )

        trainer = pl.Trainer(accelerator="auto", devices="auto", enable_progress_bar=False, logger=False)
        predictions_list = trainer.predict(self._model, dataloaders=predict_loader)

        if not predictions_list:
            return np.array([])

        predictions_flat = [
            item for sublist in predictions_list for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        return torch.cat(predictions_flat).cpu().numpy()

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit.

        :returns: True when model has been trained.
        """
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        """Serialize fitted predictor state.

        :returns: Mapping with binary payload blob and architecture metadata.
        """
        if not self.is_fitted():
            return {}
        if self._model is None:
            return {}
        return {
            "model_state": save_state_dict(self._model.state_dict()),
            "architecture": {
                "num_node_features": self._num_node_features,
                "num_cell_features": self._num_cell_features,
                "hidden_dim": int(self._hyperparameters.get("hidden_dim", 64)),
                "dropout": float(self._hyperparameters.get("dropout", 0.2)),
                "learning_rate": float(self._hyperparameters.get("learning_rate", 0.001)),
            },
            "hyperparameters": dict(self._hyperparameters),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore predictor from get_state output.

        :param state: Serialized state previously returned by get_state.
        :raises PredictorStateError: If payload is missing or invalid.
        """
        model_state_blob = state.get("model_state")
        if not isinstance(model_state_blob, (bytes, bytearray)):
            msg = f"{self.__class__.__name__} state requires model_state bytes"
            raise PredictorStateError(msg)

        architecture = state.get("architecture")
        if not isinstance(architecture, dict):
            msg = f"{self.__class__.__name__} state requires architecture dict"
            raise PredictorStateError(msg)

        hyperparameters = state.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)

        self._num_node_features = int(architecture["num_node_features"])
        self._num_cell_features = int(architecture["num_cell_features"])

        self._model = DrugGNNModule(
            num_node_features=self._num_node_features,
            num_cell_features=self._num_cell_features,
            hidden_dim=int(architecture.get("hidden_dim", 64)),
            dropout=float(architecture.get("dropout", 0.2)),
            learning_rate=float(architecture.get("learning_rate", 0.001)),
        )
        self._model.load_state_dict(load_state_dict(bytes(model_state_blob)))
