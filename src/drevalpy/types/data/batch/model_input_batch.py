"""Canonical predictor input batch for component-based models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeaturizerResult
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


def pair_sample_indices(
    sample_ids: np.ndarray,
    sample_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair sample identifiers to featurizer row indices.

    :param sample_ids: Sample id per response pair.
    :param sample_id_to_row: Mapping from entity id to featurizer row index.
    :returns: Integer array of row indices aligned with *sample_ids*.
    """
    return _map_pair_indices(sample_ids, sample_id_to_row, side="sample")


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


def _validate_pair_indices(
    table: FeaturizerResult | None, indices: np.ndarray | None, pair_ids: np.ndarray, *, side: str
) -> None:
    if table is None:
        if indices is not None:
            raise ValueError(f"{side} pair indices require a feature result")
        return
    if indices is None:
        raise ValueError(f"{side} feature result requires pair indices")
    if np.asarray(indices).dtype.kind not in "iu":
        raise ValueError(f"{side} pair indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(table.entity_ids)):
        raise ValueError(f"{side} pair indices are outside the entity table")
    if len(np.unique(table.entity_ids)) != len(table.entity_ids):
        raise ValueError(f"{side} entity_ids must be unique in a predictor entity table")
    if not np.array_equal(table.entity_ids[indices], pair_ids):
        raise ValueError(f"{side} pair identifiers do not match their indexed entity rows")


@dataclass
class ModelInputBatch:
    """Response pairs indexed into canonical per-side featurizer results."""

    sample_ids: np.ndarray
    drug_ids: np.ndarray
    response: np.ndarray | None
    sample: FeaturizerResult
    drug: FeaturizerResult | None
    sample_pair_idx: np.ndarray
    drug_pair_idx: np.ndarray | None
    early_stopping_response: ResponseBatch | None = None
    training_context: TrainingContext = field(default_factory=TrainingContext)

    def __post_init__(self) -> None:
        """Validate pair lengths and indices against canonical entity tables."""
        n = len(self.sample_ids)
        for name in ("sample_ids", "drug_ids", "response", "sample_pair_idx", "drug_pair_idx"):
            value = getattr(self, name)
            if value is not None and (np.asarray(value).ndim != 1 or len(value) != n):
                raise ValueError(f"{name} must be one-dimensional with n_pairs ({n}) entries")
        if self.response is not None:
            self.response = np.asarray(self.response, dtype=np.float64)
        for side, table, indices in (
            ("sample", self.sample, self.sample_pair_idx),
            ("drug", self.drug, self.drug_pair_idx),
        ):
            _validate_pair_indices(table, indices, getattr(self, f"{side}_ids"), side=side)

    @property
    def n_pairs(self) -> int:
        """Number of requested sample/drug response pairs."""
        return len(self.sample_ids)

    @classmethod
    def from_response(
        cls,
        response: ResponseBatch,
        *,
        sample: FeaturizerResult,
        drug: FeaturizerResult | None = None,
        sample_pair_idx: np.ndarray,
        drug_pair_idx: np.ndarray | None = None,
        early_stopping_response: ResponseBatch | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        """Index response pairs into already computed entity-level results.

        :param response: Requested pairs and optional targets.
        :param sample: Canonical sample features.
        :param drug: Optional canonical drug features.
        :param sample_pair_idx: Entity row per response pair.
        :param drug_pair_idx: Drug row per response pair.
        :param early_stopping_response: Optional disjoint stopping pairs.
        :param training_context: Runtime checkpoint and logging context.
        :returns: Aligned predictor input.
        """
        return cls(
            response.sample_ids,
            response.drug_ids,
            response.response,
            sample,
            drug,
            sample_pair_idx,
            drug_pair_idx,
            early_stopping_response,
            training_context or TrainingContext(),
        )

    def _pair_indices_for(self, response: ResponseBatch) -> tuple[np.ndarray, np.ndarray | None]:
        def indices(table: FeaturizerResult, ids: np.ndarray, side: str) -> np.ndarray:
            mapping = {str(entity): row for row, entity in enumerate(table.entity_ids)}
            return _map_pair_indices(ids, mapping, side=side)

        return (
            indices(self.sample, response.sample_ids, "sample"),
            None if self.drug is None else indices(self.drug, response.drug_ids, "drug"),
        )

    def _feature_matrix(self, cell_indices: np.ndarray, drug_indices: np.ndarray | None, *, n_pairs: int) -> np.ndarray:
        arrays = []
        for table, indices in ((self.sample, cell_indices), (self.drug, drug_indices)):
            if table is None or not table.projection:
                continue
            matrix = table.matrix
            if matrix.ndim != 2 or matrix.dtype.kind not in "biufc":
                raise ValueError("Matrix predictors require two-dimensional numeric feature projections")
            arrays.append(matrix[indices])
        if not arrays:
            return np.empty((n_pairs, 0), dtype=np.float32)
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)

    def to_feature_matrix(self) -> np.ndarray:
        """Return the dense design matrix in the batch's requested pair order."""
        return self._feature_matrix(self.sample_pair_idx, self.drug_pair_idx, n_pairs=self.n_pairs)

    def subset_pairs(self, mask: np.ndarray) -> ModelInputBatch:
        """Subset pairs, sharing entity payloads and restricting stopping drugs.

        :param mask: One boolean value per pair.
        :returns: Independent pair selection over the same canonical features.
        :raises ValueError: If mask shape does not match n_pairs.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != self.n_pairs:
            raise ValueError("subset mask must be a one-dimensional boolean array matching n_pairs")
        stopping = self.early_stopping_response
        if stopping is not None and mask.any():
            keep = np.isin(stopping.drug_ids, np.unique(self.drug_ids[mask]))
            stopping = (
                ResponseBatch(stopping.response[keep], stopping.sample_ids[keep], stopping.drug_ids[keep])
                if keep.any()
                else None
            )
        return replace(
            self,
            sample_ids=self.sample_ids[mask],
            drug_ids=self.drug_ids[mask],
            response=None if self.response is None else self.response[mask],
            sample_pair_idx=self.sample_pair_idx[mask],
            drug_pair_idx=None if self.drug_pair_idx is None else self.drug_pair_idx[mask],
            early_stopping_response=stopping,
        )
