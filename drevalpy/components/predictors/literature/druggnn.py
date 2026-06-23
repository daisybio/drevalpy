"""DrugGNN structured literature predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from drevalpy.components.config import PredictionMode
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
    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.GRAPH,
        view="drug_graph",
        backend="pyg",
    )

    def __init__(self) -> None:
        self._hyperparameters: dict[str, Any] = {}
        self._model: DrugGNNModule | None = None

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {
            "hidden_dim": 64,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 2,
            "batch_size": 8,
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
        flat = [item for sublist in predictions for item in (sublist if isinstance(sublist, list) else [sublist])]
        import torch

        return torch.cat(flat).cpu().numpy().reshape(-1)

    def is_fitted(self) -> bool:
        return self._model is not None
