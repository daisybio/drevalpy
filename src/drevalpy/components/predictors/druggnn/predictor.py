"""DrugGNN block predictor: a simple GCN baseline on molecular graphs with dense sample features.

DrugGNN is a native model inspired by GraphDRP (Nguyen et al., 2021, doi:10.1109/TCBB.2021.3060430), not a
reimplementation of it. Three graph convolutions with mean pooling encode the drug graph, a dense branch encodes the
landmark gene expression of the sample, and an unbounded regression head combines both. It trains with Adam on the
mean squared error, stops early on the validation loss and restores the weights of its best validation epoch.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from drevalpy._utils.accelerator import lightning_accelerator
from drevalpy.components.contracts.feature_format import BlockSpec, FeatureFormat
from drevalpy.components.predictors._lightning_training import LightningRun, reload_best_weights, run_lightning_fit
from drevalpy.components.predictors._pair_loader import require_drug_pair_idx
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.enums import Accelerator

# torch, pytorch_lightning and torch_geometric are imported inside the methods
# that need them, not at module scope: this module is imported eagerly by
# register_builtins(), so a module-scope import would put ~2s of the
# training stack on the critical path of every `import drevalpy`. Guarded by
# tests/test_import_cost_policy.py.
if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from drevalpy.components.predictors.druggnn.algorithm import DrugGNNModule


@register(
    "drugGNN",
    description="Simple GCN baseline on molecular graphs with dense sample features.",
    accelerator=Accelerator.OPTIONAL,
)
class DrugGNNPredictor(Predictor):
    """Simple GCN baseline inspired by GraphDRP (Nguyen et al., 2021, doi:10.1109/TCBB.2021.3060430)."""

    supports_early_stopping: ClassVar[bool] = True
    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("gene_expression"))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("drug_graph", FeatureFormat.GRAPH))

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize the DrugGNN predictor.

        :param hyperparameters: Optional hyperparameter overrides.
        """
        super().__init__(hyperparameters)
        self._model: DrugGNNModule | None = None

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

    def _fit(self, data: PredictorInput) -> None:
        """Train the DrugGNN model on the batch.

        :param data: Training batch with gene_expression and drug_graph blocks.
        """
        sample_matrix = data.sample.blocks["gene_expression"].values
        drug_graphs = data.drug.blocks["drug_graph"].values

        self._model = self._build_model(sample_matrix, drug_graphs)
        train_loader = self._graph_loader(data, batch_size=int(self._hyperparameters["batch_size"]), shuffle=True)
        val_loader = self._build_val_loader(data)
        self._run_training(train_loader, val_loader)

    def _build_model(self, sample_matrix: np.ndarray, drug_graphs: np.ndarray) -> DrugGNNModule:
        """Create an untrained module sized for *sample_matrix* and *drug_graphs* from the current hyperparameters.

        :param sample_matrix: Sample feature matrix.
        :param drug_graphs: Array of drug graph objects.
        :returns: An untrained ``DrugGNNModule``.
        """
        from drevalpy.components.predictors.druggnn.algorithm import DrugGNNModule

        return DrugGNNModule(
            num_node_features=int(drug_graphs[0].num_node_features),
            num_cell_features=int(sample_matrix.shape[1]),
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
            reload_best_weights(self._model, checkpoint)

    @staticmethod
    def _graph_loader(data: PredictorInput, *, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Build the loader over the pairs of *data*, batching each pair's drug graph in the collate function.

        The loader yields ``(drug graph batch, sample features, response)``; the response is ``NaN`` where unknown.

        :param data: Pairs to iterate, with the gene_expression and drug_graph blocks.
        :param batch_size: Mini-batch size.
        :param shuffle: Whether to shuffle the pairs each epoch.
        :returns: A ``DataLoader`` over the pairs of *data*.
        """
        import torch
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch

        sample_matrix = torch.as_tensor(data.sample.blocks["gene_expression"].values, dtype=torch.float32)
        drug_graphs = data.drug.blocks["drug_graph"].values
        drug_idx = require_drug_pair_idx(data.drug_pair_idx)
        response = torch.as_tensor(data.response, dtype=torch.float32)

        def gather(batch: list[int]) -> tuple[Any, torch.Tensor, torch.Tensor]:
            rows = torch.as_tensor(batch, dtype=torch.long)
            graphs = Batch.from_data_list(list(drug_graphs[drug_idx[batch]]))
            return graphs, sample_matrix[data.sample_pair_idx[batch]], response[rows]

        return DataLoader(
            cast("Any", range(len(data))),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=gather,
        )

    def _build_val_loader(self, data: PredictorInput) -> DataLoader | None:
        """Build a validation DataLoader over the early-stopping pairs, if there are any.

        :param data: Training batch.
        :returns: Validation DataLoader or None if no early-stopping data.
        """
        stopping = data.stopping
        if stopping is None or len(stopping) == 0:
            return None
        return self._graph_loader(stopping, batch_size=int(self._hyperparameters["batch_size"]))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for the given batch.

        :param data: Featurized pairs to score.
        :returns: One predicted response per pair.
        :raises RuntimeError: If the model has not been trained yet.
        """
        import pytorch_lightning as pl
        import torch

        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        if len(data) == 0:
            return np.array([])

        predict_loader = self._graph_loader(data, batch_size=int(self._hyperparameters["batch_size"]))

        trainer = pl.Trainer(
            accelerator=lightning_accelerator(), devices="auto", enable_progress_bar=False, logger=False
        )
        predictions_list = trainer.predict(self._model, dataloaders=predict_loader)

        if not predictions_list:
            return np.array([])

        predictions_flat = [
            item for sublist in predictions_list for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        return torch.cat(predictions_flat).cpu().numpy()
