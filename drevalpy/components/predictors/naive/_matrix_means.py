"""Matrix helpers for one-hot naive mean predictors."""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def pair_align(entity_matrix: np.ndarray, pair_idx: np.ndarray | None) -> np.ndarray:
    """Index an entity-level feature matrix to pair rows.

    :param entity_matrix: entity matrix.
    :param pair_idx: pair idx.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    matrix = np.asarray(entity_matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if pair_idx is None:
        msg = "pair index is required to align entity features"
        raise ValueError(msg)
    return matrix[np.asarray(pair_idx, dtype=np.int64)]


def require_pair_matrix(
    batch: ModelInputBatch,
    *,
    side: str,
) -> np.ndarray:
    """Return a pair-aligned dense matrix for the cell-line or drug side.

    :param batch: batch.
    :param side: side.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    if side == "cell_line":
        return pair_align(batch.cell_line_features, batch.cell_line_pair_idx)
    if side == "drug":
        if batch.drug_features is None:
            msg = "drug features are required"
            raise ValueError(msg)
        return pair_align(batch.drug_features, batch.drug_pair_idx)
    msg = f"Unknown feature side {side!r}"
    raise ValueError(msg)


def block_pair_matrix(batch: ModelInputBatch, block_name: str) -> np.ndarray:
    """Return a pair-aligned named cell-line block matrix.

    :param batch: batch.
    :param block_name: block name.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    if block_name not in batch.cell_line_blocks:
        msg = f"Required cell-line block {block_name!r} is missing"
        raise ValueError(msg)
    return pair_align(batch.cell_line_blocks[block_name].values, batch.cell_line_pair_idx)


def category_means(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return per-column means for a one-hot design matrix.

    :param design: design.
    :param y: y.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2:
        msg = "design matrix must be 2-dimensional"
        raise ValueError(msg)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if matrix.shape[0] != target.shape[0]:
        msg = "design rows must match response length"
        raise ValueError(msg)
    if matrix.shape[1] == 0:
        return np.empty((0,), dtype=np.float64)
    counts = matrix.sum(axis=0)
    sums = matrix.T @ target
    means = np.zeros(matrix.shape[1], dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def additive_effects(design: np.ndarray, y: np.ndarray, *, baseline: float) -> np.ndarray:
    """Return additive effects relative to *baseline* for observed one-hot columns.

    :param design: design.
    :param y: y.
    :param baseline: baseline.
    :returns: Result.
    """
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return category_means(design, y)
    counts = matrix.sum(axis=0)
    effects = category_means(design, y) - float(baseline)
    return np.where(counts > 0, effects, 0.0)


def predict_with_effects(design: np.ndarray, effects: np.ndarray, *, baseline: float) -> np.ndarray:
    """Predict baseline + design @ effects for a one-hot design matrix.

    :param design: design.
    :param effects: effects.
    :param baseline: baseline.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    matrix = np.asarray(design, dtype=np.float64)
    coeffs = np.asarray(effects, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        msg = "design matrix must be 2-dimensional"
        raise ValueError(msg)
    if matrix.shape[1] == 0:
        return np.full(matrix.shape[0], float(baseline), dtype=np.float64)
    if matrix.shape[1] != coeffs.shape[0]:
        msg = "effect vector length must match design columns"
        raise ValueError(msg)
    return float(baseline) + matrix @ coeffs


def state_float_vector(state: dict[str, object], key: str) -> np.ndarray | None:
    """Restore a 1D float vector stored under *key*.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)


def state_float_matrix(state: dict[str, object], key: str) -> np.ndarray | None:
    """Restore a 2D float matrix stored under *key*.

    :param state: state.
    :param key: key.
    :returns: Result.
    """
    value = state.get(key)
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    return matrix
