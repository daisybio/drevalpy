"""Shared helpers for naive predictor unit tests."""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def one_hot(labels: list[str] | np.ndarray, categories: list[str]) -> np.ndarray:
    """Build a dense float64 one-hot matrix for *labels* over *categories*.

    :param labels: Category label per row.
    :param categories: Ordered category vocabulary (column order).
    :returns: Dense one-hot matrix with shape ``(len(labels), len(categories))``.
    """
    index = {category: i for i, category in enumerate(categories)}
    matrix = np.zeros((len(labels), len(categories)), dtype=np.float64)
    for row, label in enumerate(labels):
        col = index.get(str(label))
        if col is not None:
            matrix[row, col] = 1.0
    return matrix


def _as_feature_blocks(blocks: dict[str, np.ndarray | FeatureBlock] | None) -> dict[str, FeatureBlock]:
    if not blocks:
        return {}
    wrapped: dict[str, FeatureBlock] = {}
    for name, value in blocks.items():
        if isinstance(value, FeatureBlock):
            wrapped[name] = value
        else:
            wrapped[name] = numeric_feature_block(np.asarray(value, dtype=np.float32))
    return wrapped


def _infer_n_pairs(
    *,
    n_pairs: int | None,
    response: np.ndarray | None,
    sample_pair_idx: np.ndarray | None,
    drug_pair_idx: np.ndarray | None,
    sample_features: np.ndarray | None,
    drug_features: np.ndarray | None,
) -> int:
    if n_pairs is not None:
        return n_pairs
    if response is not None:
        return len(response)
    if sample_pair_idx is not None:
        return len(sample_pair_idx)
    if drug_pair_idx is not None:
        return len(drug_pair_idx)
    if sample_features is not None and sample_features.ndim == 2:
        return int(sample_features.shape[0])
    if drug_features is not None and drug_features.ndim == 2:
        return int(drug_features.shape[0])
    msg = "n_pairs, response, or features are required"
    raise ValueError(msg)


def naive_batch(
    *,
    n_pairs: int | None = None,
    response: np.ndarray | None = None,
    sample_features: np.ndarray | None = None,
    drug_features: np.ndarray | None = None,
    sample_blocks: dict[str, np.ndarray | FeatureBlock] | None = None,
    drug_blocks: dict[str, np.ndarray | FeatureBlock] | None = None,
    sample_pair_idx: np.ndarray | None = None,
    drug_pair_idx: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    drug_ids: np.ndarray | None = None,
) -> ModelInputBatch:
    """Build a matrix-native batch for naive predictor tests.

    Pair IDs default to deliberate decoys so tests prove predictors ignore them.

    :param n_pairs: Number of pairs when not implied by other arguments.
    :param response: Optional response values aligned with the pairs.
    :param sample_features: Entity-level sample feature matrix.
    :param drug_features: Entity-level drug feature matrix.
    :param sample_blocks: Optional named sample feature blocks.
    :param drug_blocks: Optional named drug feature blocks.
    :param sample_pair_idx: Pair-to-sample row index.
    :param drug_pair_idx: Pair-to-drug row index.
    :param sample_ids: Decoy pair sample IDs (ignored by predictors).
    :param drug_ids: Decoy pair drug IDs (ignored by predictors).
    :returns: Minimal ``ModelInputBatch`` for naive predictor unit tests.
    """
    n_pairs = _infer_n_pairs(
        n_pairs=n_pairs,
        response=response,
        sample_pair_idx=sample_pair_idx,
        drug_pair_idx=drug_pair_idx,
        sample_features=sample_features,
        drug_features=drug_features,
    )

    if sample_features is None:
        sample_features = np.ones((n_pairs, 1), dtype=np.float64)
    if drug_features is None:
        drug_features = np.ones((n_pairs, 1), dtype=np.float64)
    if sample_pair_idx is None:
        sample_pair_idx = np.arange(n_pairs, dtype=np.int64)
    if drug_pair_idx is None:
        drug_pair_idx = np.arange(n_pairs, dtype=np.int64)
    cell_entities = np.array([f"entity_cl_{i}" for i in range(len(sample_features))], dtype=str)
    drug_entities = np.array([f"entity_d_{i}" for i in range(len(drug_features))], dtype=str)
    # Canonical batches validate pair IDs against their indexed entity rows. The
    # old decoy-ID fixture is intentionally replaced by this explicit mapping.
    sample_ids = cell_entities[np.asarray(sample_pair_idx, dtype=np.int64)]
    drug_ids = drug_entities[np.asarray(drug_pair_idx, dtype=np.int64)]

    cell_blocks = _as_feature_blocks(sample_blocks)
    drug_blocks = _as_feature_blocks(drug_blocks)
    if not cell_blocks:
        cell_blocks = {"matrix": numeric_feature_block(np.asarray(sample_features, dtype=np.float64))}
    if not drug_blocks:
        drug_blocks = {"matrix": numeric_feature_block(np.asarray(drug_features, dtype=np.float64))}
    return ModelInputBatch(
        sample_ids=sample_ids,
        drug_ids=drug_ids,
        response=response,
        sample=FeaturizerResult.from_blocks(cell_blocks, cell_entities),
        drug=FeaturizerResult.from_blocks(drug_blocks, drug_entities),
        sample_pair_idx=np.asarray(sample_pair_idx, dtype=np.int64),
        drug_pair_idx=np.asarray(drug_pair_idx, dtype=np.int64),
    )
