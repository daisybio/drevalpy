"""Canonical predictor input batch for component-based models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeatureBlock
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _map_pair_indices(
    entity_ids: np.ndarray,
    id_to_row: dict[str, int],
    *,
    side: str,
) -> np.ndarray:
    """Map pair identifiers to featurizer row indices with contextual errors.

    :param entity_ids: Entity id per response pair.
    :param id_to_row: Mapping from entity id to featurizer row index.
    :param side: Human-readable side label used in error messages.
    :returns: Integer array of row indices aligned with *entity_ids*.
    :raises ValueError: If any pair id is missing from *id_to_row*.
    """
    missing: list[str] = []
    rows: list[int] = []
    for entity_id in entity_ids:
        key = str(entity_id)
        row = id_to_row.get(key)
        if row is None:
            missing.append(key)
        else:
            rows.append(row)
    if missing:
        preview = ", ".join(repr(item) for item in missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        msg = f"Missing {side} identifiers in featurizer rows: {preview}{suffix}"
        raise ValueError(msg)
    return np.asarray(rows, dtype=np.int64)


def pair_cell_line_indices(
    cell_line_ids: np.ndarray,
    cell_line_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair cell-line identifiers to featurizer row indices.

    :param cell_line_ids: Cell-line id per response pair.
    :param cell_line_id_to_row: Mapping from entity id to featurizer row index.
    :returns: Integer array of row indices aligned with *cell_line_ids*.
    """
    return _map_pair_indices(cell_line_ids, cell_line_id_to_row, side="cell-line")


def pair_drug_indices(
    drug_ids: np.ndarray,
    drug_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair drug identifiers to featurizer row indices.

    :param drug_ids: Drug id per response pair.
    :param drug_id_to_row: Mapping from entity id to featurizer row index.
    :returns: Integer array of row indices aligned with *drug_ids*.
    """
    return _map_pair_indices(drug_ids, drug_id_to_row, side="drug")


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
    early_stopping_response: ResponseBatch | None = None
    training_context: TrainingContext = field(default_factory=TrainingContext)

    def __post_init__(self) -> None:
        """Validate structural consistency of the batch.

        :raises ValueError: If array lengths are inconsistent with n_pairs.
        """
        n = len(self.cell_line_ids)
        if len(self.drug_ids) != n:
            msg = f"drug_ids length ({len(self.drug_ids)}) must match cell_line_ids length ({n})"
            raise ValueError(msg)
        if self.response is not None and len(self.response) != n:
            msg = f"response length ({len(self.response)}) must match n_pairs ({n})"
            raise ValueError(msg)
        if len(self.cell_line_pair_idx) != n:
            msg = f"cell_line_pair_idx length ({len(self.cell_line_pair_idx)}) must match n_pairs ({n})"
            raise ValueError(msg)
        if self.drug_pair_idx is not None and len(self.drug_pair_idx) != n:
            msg = f"drug_pair_idx length ({len(self.drug_pair_idx)}) must match n_pairs ({n})"
            raise ValueError(msg)
        if self.response is not None:
            self.response = np.asarray(self.response, dtype=np.float64)

    @property
    def n_pairs(self) -> int:
        """Return the number of cell-line/drug pairs in the batch.

        :returns: Result.
        """
        return len(self.cell_line_ids)

    @classmethod
    def from_response(
        cls,
        response: ResponseBatch,
        *,
        cell_line_entity_ids: np.ndarray,
        drug_entity_ids: np.ndarray | None,
        cell_line_features: np.ndarray,
        drug_features: np.ndarray | None,
        cell_line_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray | None,
        cell_line_blocks: dict[str, FeatureBlock] | None = None,
        drug_blocks: dict[str, FeatureBlock] | None = None,
        early_stopping_response: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        """Build a predictor input batch from a response dataset and featurizer outputs.

        :param response: Cell-line/drug pairs and optional response values.
        :param cell_line_entity_ids: Entity ids aligned with cell-line feature rows.
        :param drug_entity_ids: Entity ids aligned with drug feature rows, or ``None``.
        :param cell_line_features: Dense or object cell-line feature matrix.
        :param drug_features: Dense or object drug feature matrix, or ``None``.
        :param cell_line_pair_idx: Row index into cell-line features for each pair.
        :param drug_pair_idx: Row index into drug features for each pair, or ``None``.
        :param cell_line_blocks: Named cell-line feature blocks from featurizers.
        :param drug_blocks: Named drug feature blocks from featurizers.
        :param early_stopping_response: Optional validation pairs for early stopping.
        :param training_context: Runtime metadata for the training call.
        :returns: ``ModelInputBatch`` ready for predictor ``fit`` or ``predict``.
        """
        return cls(
            cell_line_ids=response.cell_line_ids,
            drug_ids=response.drug_ids,
            response=response.response,
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

    def _pair_indices_for(self, response: ResponseBatch) -> tuple[np.ndarray, np.ndarray | None]:
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

    def feature_matrix_for(self, response: ResponseBatch) -> np.ndarray:
        """Return a dense design matrix for an alternate response dataset.

        :param response: Pairs whose features should be materialized from stored

        :returns: Design matrix with one row per pair in *response*.

        :raises ValueError: If drug features are present but pair indices are missing.
        """
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
        from drevalpy.components.featurizers._matrix import stack_pair_features

        return stack_pair_features(
            self.cell_line_features,
            self.drug_features,
            cell_line_pair_idx,
            drug_pair_idx,
        )

    def early_stopping_feature_matrix(self) -> np.ndarray | None:
        """Return validation features when early-stopping pairs are present.

        :returns: Design matrix for ``early_stopping_response``, or ``None`` when early stopping is disabled.
        """
        if self.early_stopping_response is None or len(self.early_stopping_response) == 0:
            return None
        return self.feature_matrix_for(self.early_stopping_response)

    def to_feature_matrix(self) -> np.ndarray:
        """Return a dense design matrix with one row per response pair.

        :returns: Design matrix for the batch's primary ``response`` pairs.

        :raises ValueError: If ``response`` is ``None``.
        """
        if self.response is None:
            msg = "ModelInputBatch.response is required to build a feature matrix"
            raise ValueError(msg)
        response = ResponseBatch(
            response=self.response,
            cell_line_ids=self.cell_line_ids,
            drug_ids=self.drug_ids,
        )
        return self.feature_matrix_for(response)

    def subset_pairs(self, mask: np.ndarray) -> ModelInputBatch:
        """Return a batch containing only the selected response pairs.

        Early-stopping pairs are narrowed to the set of drugs that survive
        *mask*, so a multi-drug subset keeps a multi-drug validation set. A
        single-drug subset is the one-element case of that same rule, which is
        what the single-drug predictors rely on. ``PredictorBase.fit`` needs the
        multi-drug case because it filters NaN pairs across the whole batch.

        :param mask: One-dimensional boolean array with length ``n_pairs``.

        :returns: New batch referencing the same entity-level features.

        :raises ValueError: If *mask* is not a one-dimensional boolean array of length ``n_pairs``.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != self.n_pairs:
            msg = "subset mask must be a one-dimensional boolean array matching n_pairs"
            raise ValueError(msg)

        early_stopping = self.early_stopping_response
        if early_stopping is not None and np.any(mask):
            selected_drugs = np.unique(self.drug_ids[mask])
            es_mask = np.isin(early_stopping.drug_ids, selected_drugs)
            if not np.any(es_mask):
                early_stopping = None
            else:
                early_stopping = ResponseBatch(
                    response=early_stopping.response[es_mask],
                    cell_line_ids=early_stopping.cell_line_ids[es_mask],
                    drug_ids=early_stopping.drug_ids[es_mask],
                )

        drug_pair_idx = self.drug_pair_idx
        return ModelInputBatch(
            cell_line_ids=self.cell_line_ids[mask],
            drug_ids=self.drug_ids[mask],
            response=None if self.response is None else self.response[mask],
            cell_line_entity_ids=self.cell_line_entity_ids,
            drug_entity_ids=self.drug_entity_ids,
            cell_line_features=self.cell_line_features,
            drug_features=self.drug_features,
            cell_line_pair_idx=self.cell_line_pair_idx[mask],
            drug_pair_idx=None if drug_pair_idx is None else drug_pair_idx[mask],
            cell_line_blocks=dict(self.cell_line_blocks),
            drug_blocks=dict(self.drug_blocks),
            early_stopping_response=early_stopping,
            training_context=self.training_context,
        )
