"""DrugGNN block predictor – GCN on molecular graphs with dense sample features."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors._lightning_training import LightningRun, run_lightning_fit
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.literature._metadata import DRUGGNN_REFERENCE
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

# torch, pytorch_lightning and torch_geometric are imported inside the methods
# that need them, not at module scope: this module is imported eagerly by
# register_builtins(), so a module-scope import would put ~2s of the
# training stack on the critical path of every `import drevalpy`. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    import torch
    from torch_geometric.loader import DataLoader

    from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNNModule


class _DrugGNNDataset:
    """Map-style dataset that yields (drug_graph, cell_features, response) tuples.

    Accepts pre-built index arrays mapping each pair to entity-level features.

    Deliberately not a ``torch.utils.data.Dataset`` subclass: that base class
    contributes only ``__add__``, which nothing here uses, and inheriting from it
    would require ``torch`` at class-definition time.
    """

    def __init__(
        self,
        sample_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray,
        sample_matrix: np.ndarray,
        drug_graphs: np.ndarray,
        response: np.ndarray,
    ) -> None:
        """Initialize the DrugGNN dataset.

        :param sample_pair_idx: Pair-level indices into the sample matrix.
        :param drug_pair_idx: Pair-level indices into the drug graphs array.
        :param sample_matrix: Sample feature matrix.
        :param drug_graphs: Array of PyG Data objects for drugs.
        :param response: Response values for each pair.
        """
        import torch

        self._cl_pair_idx = sample_pair_idx
        self._drug_pair_idx = drug_pair_idx
        self._cl_tensors = torch.as_tensor(sample_matrix, dtype=torch.float32)
        self._drug_graphs = drug_graphs
        self._response = torch.as_tensor(response, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of pairs.

        :returns: Dataset length.
        """
        return len(self._response)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
        """Return (drug_graph, cell_features, response) for the given index.

        :param idx: Pair index.
        :returns: Tuple of drug graph, cell tensor, and response scalar.
        """
        cl_idx = self._cl_pair_idx[idx]
        drug_idx = self._drug_pair_idx[idx]
        return self._drug_graphs[drug_idx], self._cl_tensors[cl_idx], self._response[idx]


@register(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense sample features.",
    reference=DRUGGNN_REFERENCE,
)
class DrugGNNPredictor(Predictor):
    """Registered DrugGNN predictor consuming ModelInputBatch directly."""

    supports_early_stopping: ClassVar[bool] = True
    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("drug_graph", FeatureFormat.GRAPH))

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the DrugGNN predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: DrugGNNModule | None = None
        self._num_node_features: int = 0
        self._num_cell_features: int = 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Hyperparameter search space specs; the Lightning precision and logging interval are fixed.
        """
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 100, "default": 100},
            "batch_size": {"type": "int", "low": 4, "high": 1024, "log": True, "default": 1024},
            "precision": {"type": "fixed", "default": 32},
            "log_every_n_steps": {"type": "fixed", "default": 50},
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
        if batch.drug is None:
            msg = "DrugGNN requires either drug_pair_idx or drug_entity_ids"
            raise ValueError(msg)
        return batch.drug.rows_for(batch.drug_ids)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Train the DrugGNN model on the batch.

        :param batch: Training batch with gene_expression and drug_graph blocks.
        :raises ValueError: If response data or drug features are missing.
        """
        if batch.response is None:
            msg = "DrugGNN requires training response data"
            raise ValueError(msg)

        sample_matrix = batch.sample.blocks["gene_expression"].values
        drug_graphs = batch.drug.blocks["drug_graph"].values

        self._model = self._build_model(sample_matrix, drug_graphs)
        train_loader = self._graph_loader(
            batch.sample_pair_idx,
            self._resolve_drug_pair_idx(batch),
            sample_matrix,
            drug_graphs,
            batch.response,
            batch_size=int(self._hyperparameters["batch_size"]),
            shuffle=True,
        )
        val_loader = self._build_val_loader(batch, sample_matrix, drug_graphs)
        self._run_training(train_loader, val_loader)

    def _build_model(self, sample_matrix: np.ndarray, drug_graphs: np.ndarray) -> DrugGNNModule:
        """Instantiate the Lightning module and record the widths it was built for.

        :param sample_matrix: Sample feature matrix.
        :param drug_graphs: Array of drug graph objects.
        :returns: An untrained ``DrugGNNModule``.
        """
        return self._new_module(int(drug_graphs[0].num_node_features), int(sample_matrix.shape[1]))

    def _new_module(self, num_node_features: int, num_cell_features: int) -> DrugGNNModule:
        """Create an untrained module of the given widths from the current hyperparameters.

        :param num_node_features: Width of each drug graph node's features.
        :param num_cell_features: Width of the sample feature vector.
        :returns: An untrained ``DrugGNNModule``, whose widths are recorded on the predictor.
        """
        from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNNModule

        self._num_node_features = num_node_features
        self._num_cell_features = num_cell_features
        return DrugGNNModule(
            num_node_features=num_node_features,
            num_cell_features=num_cell_features,
            hidden_dim=int(self._hyperparameters["hidden_dim"]),
            dropout=float(self._hyperparameters["dropout"]),
            learning_rate=float(self._hyperparameters["learning_rate"]),
        )

    def _run_training(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        """Fit the model, early-stopping on the validation loss when one is available.

        :param train_loader: Training loader.
        :param val_loader: Validation loader, or ``None``, in which case nothing is checkpointed.
        """
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = run_lightning_fit(
                self._model,
                train_loader,
                val_loader,
                LightningRun(
                    max_epochs=int(self._hyperparameters["epochs"]),
                    checkpoint_dir=directory,
                    precision=self._hyperparameters["precision"],
                    log_every_n_steps=int(self._hyperparameters["log_every_n_steps"]),
                    logger=False,
                    enable_progress_bar=True,
                    checkpoint=val_loader is not None,
                ),
            )
            if checkpoint is not None and checkpoint.best_model_path:
                self._model.load_state_dict(load_state_dict(checkpoint.best_model_path)["state_dict"])

    @staticmethod
    def _graph_loader(
        sample_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray,
        sample_matrix: np.ndarray,
        drug_graphs: np.ndarray,
        response: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build one graph loader over the given pairs.

        :param sample_pair_idx: Sample row per pair.
        :param drug_pair_idx: Drug row per pair.
        :param sample_matrix: Sample feature matrix.
        :param drug_graphs: Array of drug graph objects.
        :param response: Pair-level response values.
        :param batch_size: Mini-batch size.
        :param shuffle: Whether to shuffle each epoch.
        :returns: A ``torch_geometric`` loader.
        """
        from torch_geometric.loader import DataLoader

        dataset = _DrugGNNDataset(
            sample_pair_idx=sample_pair_idx,
            drug_pair_idx=drug_pair_idx,
            sample_matrix=sample_matrix,
            drug_graphs=drug_graphs,
            response=response,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    def _build_val_loader(
        self,
        batch: ModelInputBatch,
        sample_matrix: np.ndarray,
        drug_graphs: np.ndarray,
    ) -> DataLoader | None:
        """Build a validation DataLoader from the early-stopping response if available.

        :param batch: Full training batch (for entity ID lookups).
        :param sample_matrix: Sample feature matrix.
        :param drug_graphs: Array of drug graph objects.
        :returns: Validation DataLoader or None if no early-stopping data.
        :raises ValueError: If the batch has early-stopping pairs but no drug features.
        """
        es = batch.early_stopping_response
        if es is None or len(es) == 0:
            return None

        if batch.drug is None:
            raise ValueError("DrugGNN requires drug features for early stopping")

        return self._graph_loader(
            batch.sample.rows_for(es.sample_ids),
            batch.drug.rows_for(es.drug_ids),
            sample_matrix,
            drug_graphs,
            es.response,
            batch_size=int(self._hyperparameters["batch_size"]),
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for the given batch.

        :param batch: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises RuntimeError: If the model has not been trained yet.
        """
        import pytorch_lightning as pl
        import torch

        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        if batch.n_pairs == 0:
            return np.array([])

        predict_loader = self._graph_loader(
            batch.sample_pair_idx,
            self._resolve_drug_pair_idx(batch),
            batch.sample.blocks["gene_expression"].values,
            batch.drug.blocks["drug_graph"].values,
            np.zeros(batch.n_pairs, dtype=np.float32),
            batch_size=int(self._hyperparameters["batch_size"]),
        )

        trainer = pl.Trainer(accelerator="auto", devices="auto", enable_progress_bar=False, logger=False)
        predictions_list = trainer.predict(self._model, dataloaders=predict_loader)

        if not predictions_list:
            return np.array([])

        predictions_flat = [
            item for sublist in predictions_list for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        return torch.cat(predictions_flat).cpu().numpy()
