"""Shared helpers for naive predictor unit tests."""

from __future__ import annotations

import numpy as np

from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.model_input_batch import ModelInputBatch


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


def naive_batch(
    *,
    n_pairs: int | None = None,
    response: np.ndarray | None = None,
    cell_line_features: np.ndarray | None = None,
    drug_features: np.ndarray | None = None,
    cell_line_blocks: dict[str, np.ndarray | FeatureBlock] | None = None,
    drug_blocks: dict[str, np.ndarray | FeatureBlock] | None = None,
    cell_line_pair_idx: np.ndarray | None = None,
    drug_pair_idx: np.ndarray | None = None,
    cell_line_ids: np.ndarray | None = None,
    drug_ids: np.ndarray | None = None,
) -> ModelInputBatch:
    """Build a matrix-native batch for naive predictor tests.

    Pair IDs default to deliberate decoys so tests prove predictors ignore them.

    :param n_pairs: Number of pairs when not implied by other arguments.
    :param response: Optional response values aligned with the pairs.
    :param cell_line_features: Entity-level cell-line feature matrix.
    :param drug_features: Entity-level drug feature matrix.
    :param cell_line_blocks: Optional named cell-line feature blocks.
    :param drug_blocks: Optional named drug feature blocks.
    :param cell_line_pair_idx: Pair-to-cell-line row index.
    :param drug_pair_idx: Pair-to-drug row index.
    :param cell_line_ids: Decoy pair cell-line IDs (ignored by predictors).
    :param drug_ids: Decoy pair drug IDs (ignored by predictors).
    :returns: Minimal ``ModelInputBatch`` for naive predictor unit tests.
    :raises ValueError: If pair count cannot be inferred from the inputs.
    """
    if n_pairs is None:
        if response is not None:
            n_pairs = len(response)
        elif cell_line_pair_idx is not None:
            n_pairs = len(cell_line_pair_idx)
        elif drug_pair_idx is not None:
            n_pairs = len(drug_pair_idx)
        elif cell_line_features is not None and cell_line_features.ndim == 2:
            n_pairs = int(cell_line_features.shape[0])
        elif drug_features is not None and drug_features.ndim == 2:
            n_pairs = int(drug_features.shape[0])
        else:
            msg = "n_pairs, response, or features are required"
            raise ValueError(msg)

    if cell_line_features is None:
        cell_line_features = np.ones((n_pairs, 1), dtype=np.float64)
    if drug_features is None:
        drug_features = np.ones((n_pairs, 1), dtype=np.float64)
    if cell_line_pair_idx is None:
        cell_line_pair_idx = np.arange(n_pairs, dtype=np.int64)
    if drug_pair_idx is None:
        drug_pair_idx = np.arange(n_pairs, dtype=np.int64)
    if cell_line_ids is None:
        cell_line_ids = np.array([f"decoy_cl_{i}" for i in range(n_pairs)], dtype=str)
    if drug_ids is None:
        drug_ids = np.array([f"decoy_d_{i}" for i in range(n_pairs)], dtype=str)

    return ModelInputBatch(
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        response=response,
        cell_line_entity_ids=np.array([f"entity_cl_{i}" for i in range(len(cell_line_features))], dtype=str),
        drug_entity_ids=np.array([f"entity_d_{i}" for i in range(len(drug_features))], dtype=str),
        cell_line_features=np.asarray(cell_line_features, dtype=np.float64),
        drug_features=np.asarray(drug_features, dtype=np.float64),
        cell_line_pair_idx=np.asarray(cell_line_pair_idx, dtype=np.int64),
        drug_pair_idx=np.asarray(drug_pair_idx, dtype=np.int64),
        cell_line_blocks=_as_feature_blocks(cell_line_blocks),
        drug_blocks=_as_feature_blocks(drug_blocks),
    )
