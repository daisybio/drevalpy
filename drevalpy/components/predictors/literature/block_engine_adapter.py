"""Block-based literature engine adapter."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_mixin import LiteratureEngineMixin
from drevalpy.components.predictors.literature._feature_dataset_from_batch import feature_dataset_from_blocks
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.datasets.dataset import FeatureDataset


class BlockLiteratureEnginePredictor(LiteratureEngineMixin, BlockPredictor):
    """Train a literature engine only from declared featurizer blocks."""

    requires_drug_featurizer: ClassVar[bool] = True
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_drug_blocks: ClassVar[tuple[str, ...]] = ()

    def _dataset_from_blocks(
        self,
        entity_ids: np.ndarray | None,
        blocks: dict[str, np.ndarray],
        *,
        required: tuple[str, ...],
        side: str,
    ) -> FeatureDataset:
        if entity_ids is None:
            msg = f"{self.__class__.__name__} requires {side} entity ids"
            raise RuntimeError(msg)
        missing = [name for name in required if name not in blocks]
        if missing:
            msg = f"{self.__class__.__name__} missing {side} block(s) {missing}; required={list(required)}"
            raise ValueError(msg)
        selected = {name: blocks[name] for name in required}
        return feature_dataset_from_blocks(entity_ids, selected)

    def _materialize_inputs(self, batch: ModelInputBatch) -> tuple[FeatureDataset, FeatureDataset | None]:
        cell_lines = self._dataset_from_blocks(
            batch.cell_line_entity_ids,
            batch.cell_line_blocks,
            required=self.required_cell_line_blocks,
            side="cell_line",
        )
        if not self.requires_drug_featurizer:
            return cell_lines, None
        drugs = self._dataset_from_blocks(
            batch.drug_entity_ids,
            batch.drug_blocks,
            required=self.required_drug_blocks,
            side="drug",
        )
        return cell_lines, drugs

    def fit(self, batch: ModelInputBatch) -> None:
        cell_lines, drugs = self._materialize_inputs(batch)
        self._train_engine(batch, cell_lines, drugs)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_lines, drugs = self._materialize_inputs(batch)
        return self._predict_engine(batch, cell_lines, drugs)
