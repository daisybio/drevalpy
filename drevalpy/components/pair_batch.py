"""Structured pair batches for literature predictors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drevalpy.components.pair_context import PairContext
from drevalpy.datasets.dataset import DrugResponseDataset


@dataclass
class PairBatch:
    """Featurized training or prediction batch for one set of response pairs."""

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
    pair_context: PairContext | None = None

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
        pair_context: PairContext | None = None,
    ) -> PairBatch:
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
            pair_context=pair_context,
        )
