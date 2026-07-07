"""Canonical predictor input batch for component-based models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drevalpy.components.featurizers._matrix import stack_pair_features
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


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
    cell_line_blocks: dict[str, np.ndarray] = field(default_factory=dict)
    drug_blocks: dict[str, np.ndarray] = field(default_factory=dict)
    cell_line_input: FeatureDataset | None = None
    drug_input: FeatureDataset | None = None

    @property
    def n_pairs(self) -> int:
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
        cell_line_blocks: dict[str, np.ndarray] | None = None,
        drug_blocks: dict[str, np.ndarray] | None = None,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
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
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

    def to_feature_matrix(self) -> np.ndarray:
        """Return a dense design matrix with one row per response pair."""
        if self.n_pairs == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.drug_features is None or self.drug_features.size == 0:
            if self.cell_line_features.size == 0:
                return np.empty((self.n_pairs, 0), dtype=np.float32)
            return self.cell_line_features[self.cell_line_pair_idx]
        if self.cell_line_features.size == 0:
            if self.drug_pair_idx is None:
                msg = "drug_pair_idx is required when only drug features are present"
                raise ValueError(msg)
            return self.drug_features[self.drug_pair_idx]
        if self.drug_pair_idx is None:
            msg = "drug_pair_idx is required when drug features are present"
            raise ValueError(msg)
        return stack_pair_features(
            self.cell_line_features,
            self.drug_features,
            self.cell_line_pair_idx,
            self.drug_pair_idx,
        )
