"""DrugGNN structured literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.pair_batch import PairBatch
from drevalpy.components.predictors.literature._metadata import DRUGGNN_METADATA
from drevalpy.components.predictors.literature.impl.druggnn.drug_gnn import (
    DrugGNNModule,
    _DrugResponsePytorchDataset,
)
from drevalpy.components.predictors.structured import StructuredPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


@register_predictor(
    "drugGNN",
    description="DrugGNN: GCN on molecular graphs with dense cell-line features.",
    **DRUGGNN_METADATA,
)
class DrugGNNPredictor(StructuredPredictor):
    """Drug gnnpredictor component."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.GRAPH)

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._model: DrugGNNModule | None = None
        self._num_node_features: int | None = None
        self._num_cell_features: int | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "hidden_dim": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 2,
            "batch_size": 8,
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "default": 64},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "default": 0.2},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "default": 1e-3},
            "epochs": {"type": "int", "low": 1, "high": 10, "default": 2},
            "batch_size": {"type": "int", "low": 4, "high": 32, "default": 8},
        }

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        merged = {**self.get_default_hyperparameters(), **hyperparameters}
        self._hyperparameters = merged
        num_node = int(input_dims.get("drug", 0)) or 9
        num_cell = int(input_dims.get("cell_line", 0)) or 1
        self._model = DrugGNNModule(
            num_node_features=num_node,
            num_cell_features=num_cell,
            hidden_dim=int(merged["hidden_dim"]),
            dropout=float(merged["dropout"]),
            learning_rate=float(merged["learning_rate"]),
        )

    def _response_dataset(
        self,
        batch: PairBatch,
        *,
        output: DrugResponseDataset | None,
    ) -> DrugResponseDataset:
        if output is not None:
            return output
        if batch.response is None:
            msg = "DrugGNN requires response data"
            raise RuntimeError(msg)
        return DrugResponseDataset(
            response=batch.response,
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output: DrugResponseDataset | None = None,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        _ = batch, output_earlystopping
        if cell_line_input is None or drug_input is None:
            msg = "DrugGNN requires cell_line_input and drug_input"
            raise RuntimeError(msg)
        num_node_features = next(iter(drug_input.features.values()))["drug_graph"].num_node_features
        num_cell_features = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]
        self._num_node_features = int(num_node_features)
        self._num_cell_features = int(num_cell_features)
        merged = self._hyperparameters
        self._model = DrugGNNModule(
            num_node_features=num_node_features,
            num_cell_features=num_cell_features,
            hidden_dim=int(merged["hidden_dim"]),
            dropout=float(merged["dropout"]),
            learning_rate=float(merged["learning_rate"]),
        )
        response = self._response_dataset(batch, output=output)
        dataset = _DrugResponsePytorchDataset(
            response=response.response,
            cell_line_ids=response.cell_line_ids,
            drug_ids=response.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        loader = DataLoader(dataset, batch_size=int(self._hyperparameters["batch_size"]), shuffle=True)
        trainer = pl.Trainer(
            max_epochs=int(self._hyperparameters["epochs"]),
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(self._model, train_dataloaders=loader)

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._model is None or cell_line_input is None or drug_input is None:
            return np.full(len(batch.cell_line_ids), np.nan, dtype=np.float64)
        response = DrugResponseDataset(
            response=np.zeros(len(batch.cell_line_ids)),
            cell_line_ids=batch.cell_line_ids,
            drug_ids=batch.drug_ids,
        )
        dataset = _DrugResponsePytorchDataset(
            response=response.response,
            cell_line_ids=response.cell_line_ids,
            drug_ids=response.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        loader = DataLoader(dataset, batch_size=int(self._hyperparameters.get("batch_size", 8)))
        trainer = pl.Trainer(accelerator="cpu", devices=1, enable_progress_bar=False, logger=False)
        predictions = trainer.predict(self._model, dataloaders=loader)
        if predictions is None:
            return np.array([], dtype=np.float64)
        flat = [item for sublist in predictions for item in (sublist if isinstance(sublist, list) else [sublist])]
        import torch

        return torch.cat(flat).cpu().numpy().reshape(-1)

    def is_fitted(self) -> bool:
        return self._model is not None

    def get_state(self) -> dict[str, object]:
        if self._model is None:
            return {}
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(
            {
                "hyperparameters": dict(self._hyperparameters),
                "state_dict": self._model.state_dict(),
                "num_node_features": self._num_node_features,
                "num_cell_features": self._num_cell_features,
            },
            buffer,
        )
        return {"checkpoint": buffer.getvalue()}

    def set_state(self, state: dict[str, object]) -> None:
        checkpoint = state.get("checkpoint")
        if not isinstance(checkpoint, (bytes, bytearray)):
            return
        import io

        import torch

        data = torch.load(io.BytesIO(checkpoint), weights_only=False)  # noqa: S614
        if not isinstance(data, dict):
            return
        hyperparameters = data.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)
        num_node = data.get("num_node_features")
        num_cell = data.get("num_cell_features")
        if num_node is None or num_cell is None:
            return
        self._num_node_features = int(num_node)
        self._num_cell_features = int(num_cell)
        merged = self._hyperparameters
        self._model = DrugGNNModule(
            num_node_features=self._num_node_features,
            num_cell_features=self._num_cell_features,
            hidden_dim=int(merged.get("hidden_dim", 64)),
            dropout=float(merged.get("dropout", 0.2)),
            learning_rate=float(merged.get("learning_rate", 0.001)),
        )
        state_dict = data.get("state_dict")
        if state_dict is not None:
            self._model.load_state_dict(state_dict)
