"""Canonical predictor input batch for component-based models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.featurizers._matrix import stack_pair_features
from drevalpy.components.pair_features import pair_cell_line_indices, pair_drug_indices
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset


@dataclass
class ModelInputBatch:
    """Featurized training or prediction batch handed to predictors."""

    cell_line_ids: np.ndarray
    drug_ids: np.ndarray
    response: np.ndarray | None
    cell_line_entity_ids: np.ndarray
    drug_entity_ids: np.ndarray | None
    cell_line_features: np.ndarray
    drug_features: np.ndarray | None
    cell_line_pair_idx: np.ndarray
    drug_pair_idx: np.ndarray | None
    cell_line_blocks: dict[str, FeatureBlock] = field(default_factory=dict)
    drug_blocks: dict[str, FeatureBlock] = field(default_factory=dict)
    early_stopping_response: DrugResponseDataset | None = None
    training_context: TrainingContext = field(default_factory=TrainingContext)

    @property
    def n_pairs(self) -> int:
        """Return the number of cell-line/drug pairs in the batch."""
        return len(self.cell_line_ids)

    @classmethod
    def from_response(
        cls,
        response: DrugResponseDataset,
        *,
        cell_line_entity_ids: np.ndarray,
        drug_entity_ids: np.ndarray | None,
        cell_line_features: np.ndarray,
        drug_features: np.ndarray | None,
        cell_line_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray | None,
        cell_line_blocks: dict[str, FeatureBlock] | None = None,
        drug_blocks: dict[str, FeatureBlock] | None = None,
        early_stopping_response: DrugResponseDataset | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        """Build a predictor input batch from a response dataset and featurizer outputs."""
        return cls(
            cell_line_ids=response.cell_line_ids,
            drug_ids=response.drug_ids,
            response=np.asarray(response.response, dtype=np.float64),
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids,
            cell_line_features=cell_line_features,
            drug_features=drug_features,
            cell_line_pair_idx=cell_line_pair_idx,
            drug_pair_idx=drug_pair_idx,
            cell_line_blocks=dict(cell_line_blocks or {}),
            drug_blocks=dict(drug_blocks or {}),
            early_stopping_response=early_stopping_response,
            training_context=training_context or TrainingContext(),
        )

    def _pair_indices_for(self, response: DrugResponseDataset) -> tuple[np.ndarray, np.ndarray | None]:
        if self.cell_line_entity_ids.size == 0:
            cell_line_pair_idx = np.zeros(len(response), dtype=np.int64)
        else:
            cell_line_map = {str(entity_id): row for row, entity_id in enumerate(self.cell_line_entity_ids)}
            cell_line_pair_idx = pair_cell_line_indices(response.cell_line_ids, cell_line_map)

        drug_pair_idx = None
        if self.drug_entity_ids is not None and self.drug_features is not None:
            if self.drug_entity_ids.size == 0:
                drug_pair_idx = np.zeros(len(response), dtype=np.int64)
            else:
                drug_map = {str(entity_id): row for row, entity_id in enumerate(self.drug_entity_ids)}
                drug_pair_idx = pair_drug_indices(response.drug_ids, drug_map)
        return cell_line_pair_idx, drug_pair_idx

    def feature_matrix_for(self, response: DrugResponseDataset) -> np.ndarray:
        """Return a dense design matrix for an alternate response dataset."""
        n_pairs = len(response)
        if n_pairs == 0:
            return np.empty((0, 0), dtype=np.float32)
        cell_line_pair_idx, drug_pair_idx = self._pair_indices_for(response)
        if self.drug_features is None or self.drug_features.size == 0:
            if self.cell_line_features.size == 0:
                return np.empty((n_pairs, 0), dtype=np.float32)
            return self.cell_line_features[cell_line_pair_idx]
        if self.cell_line_features.size == 0:
            if drug_pair_idx is None:
                msg = "drug_pair_idx is required when only drug features are present"
                raise ValueError(msg)
            return self.drug_features[drug_pair_idx]
        if drug_pair_idx is None:
            msg = "drug_pair_idx is required when drug features are present"
            raise ValueError(msg)
        return stack_pair_features(
            self.cell_line_features,
            self.drug_features,
            cell_line_pair_idx,
            drug_pair_idx,
        )

    def early_stopping_feature_matrix(self) -> np.ndarray | None:
        """Return validation features when early-stopping pairs are present."""
        if self.early_stopping_response is None or len(self.early_stopping_response) == 0:
            return None
        return self.feature_matrix_for(self.early_stopping_response)

    def to_feature_matrix(self) -> np.ndarray:
        """Return a dense design matrix with one row per response pair."""
        if self.response is None:
            msg = "ModelInputBatch.response is required to build a feature matrix"
            raise ValueError(msg)
        response = DrugResponseDataset(
            response=np.asarray(self.response, dtype=np.float64),
            cell_line_ids=self.cell_line_ids,
            drug_ids=self.drug_ids,
        )
        return self.feature_matrix_for(response)
