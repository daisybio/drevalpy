"""Matrix helpers for one-hot naive mean predictors."""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def pair_align(entity_matrix: np.ndarray, pair_idx: np.ndarray | None) -> np.ndarray:
    """Index an entity-level feature matrix to pair rows.

    :param entity_matrix: Feature matrix with one row per entity; a 1D vector is treated as a single column.
    :param pair_idx: For each pair, the row of ``entity_matrix`` that belongs to it.
    :returns: Matrix with one row per pair.
    :raises ValueError: If ``pair_idx`` is None.
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
    """Return a pair-aligned dense matrix for the sample or drug side.

    :param batch: Featurized batch whose sample or drug matrix is aligned.
    :param side: ``"sample"`` or ``"drug"``.
    :returns: Matrix with one row per pair.
    :raises ValueError: If ``side`` is unknown or the drug side is requested but the batch has no drug features.
    """
    if side == "sample":
        return pair_align(batch.sample.matrix, batch.sample_pair_idx)
    if side == "drug":
        if batch.drug is None:
            msg = "drug features are required"
            raise ValueError(msg)
        return pair_align(batch.drug.matrix, batch.drug_pair_idx)
    msg = f"Unknown feature side {side!r}"
    raise ValueError(msg)


def block_pair_matrix(batch: ModelInputBatch, block_name: str) -> np.ndarray:
    """Return a pair-aligned named sample block matrix.

    :param batch: Featurized batch.
    :param block_name: Name of the sample block to align.
    :returns: Matrix with one row per pair.
    :raises ValueError: If the sample side has no block named ``block_name``.
    """
    if block_name not in batch.sample.blocks:
        msg = f"Required sample block {block_name!r} is missing"
        raise ValueError(msg)
    return pair_align(batch.sample.blocks[block_name].values, batch.sample_pair_idx)


def category_means(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return per-column means for a one-hot design matrix.

    :param design: One-hot design matrix with one row per pair.
    :param y: Response of each pair.
    :returns: Mean response of the pairs in each column; 0 for a column without pairs.
    :raises ValueError: If ``design`` is not 2D or its row count differs from the length of ``y``.
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

    :param design: One-hot design matrix with one row per pair.
    :param y: Response of each pair.
    :param baseline: Value subtracted from the column means, normally the overall mean response.
    :returns: Mean offset per column; 0 for a column without pairs.
    """
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return category_means(design, y)
    counts = matrix.sum(axis=0)
    effects = category_means(design, y) - float(baseline)
    return np.where(counts > 0, effects, 0.0)


def predict_with_effects(design: np.ndarray, effects: np.ndarray, *, baseline: float) -> np.ndarray:
    """Predict baseline + design @ effects for a one-hot design matrix.

    :param design: One-hot design matrix with one row per pair.
    :param effects: Effect of each design column.
    :param baseline: Intercept added to every prediction.
    :returns: Predicted response per pair.
    :raises ValueError: If ``design`` is not 2D or the number of effects differs from its column count.
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

    :param state: Mapping produced by ``get_state``.
    :param key: Entry to read.
    :returns: The entry as a float64 vector, or None when ``key`` is absent.
    """
    value = state.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)


def state_float_matrix(state: dict[str, object], key: str) -> np.ndarray | None:
    """Restore a 2D float matrix stored under *key*.

    :param state: Mapping produced by ``get_state``.
    :param key: Entry to read.
    :returns: The entry as a float64 matrix (a 1D entry becomes one column), or None when ``key`` is absent.
    """
    value = state.get(key)
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    return matrix
