"""Category-code helpers for the naive mean predictors.

The naive predictors never multiply a one-hot design matrix. The featurizers emit one-hot rows per
*entity* (a few hundred to a few thousand rows), and :func:`pair_codes` reduces them to one integer
category code per pair. Group sums and counts are then ``np.bincount`` calls and a prediction is an
index into the per-category effects, so memory stays linear in the number of pairs instead of
pairs x categories.

A code of ``-1`` marks a pair without a category, an all-zero row such as a sample or tissue unseen in
training. Such a pair adds no effect, so it falls back to the baseline. A category without training
pairs has an effect of 0 as well.
"""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.feature_source import Side
from drevalpy.types.data.predictor_input import PredictorInput

_NO_CATEGORY = -1


def pair_codes(entity_matrix: np.ndarray, pair_idx: np.ndarray | None) -> tuple[np.ndarray, int]:
    """Turn an entity-level one-hot matrix into one category code per pair.

    :param entity_matrix: One-hot matrix with one row per entity; a 1D vector is treated as a single column.
    :param pair_idx: For each pair, the row of ``entity_matrix`` that belongs to it.
    :returns: The category code of each pair (``-1`` for an all-zero row) and the number of categories.
    :raises ValueError: If ``pair_idx`` is None or a row is not all zeros with at most one 1.
    """
    matrix = np.asarray(entity_matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if pair_idx is None:
        msg = "pair index is required to align entity features"
        raise ValueError(msg)
    if not (((matrix == 0) | (matrix == 1)).all() and (matrix.sum(axis=1) <= 1).all()):
        msg = "naive mean predictors need one-hot rows: zeros and at most one 1"
        raise ValueError(msg)
    if matrix.shape[1] == 0:
        codes = np.full(len(matrix), _NO_CATEGORY, dtype=np.int64)
    else:
        codes = np.where(matrix.any(axis=1), matrix.argmax(axis=1), _NO_CATEGORY)
    return codes[np.asarray(pair_idx, dtype=np.int64)], matrix.shape[1]


def require_pair_codes(data: PredictorInput, *, side: Side) -> tuple[np.ndarray, int]:
    """Return the category codes of the sample or drug side, one per pair.

    Raises ``ValueError`` (from :func:`pair_codes`) if the matrix of *side* is not one-hot.

    :param data: Featurized pairs whose sample or drug matrix is one-hot.
    :param side: The side to encode.
    :returns: The category code of each pair and the number of categories.
    :raises ValueError: If the drug side is requested but *data* has no drug features.
    """
    if side == Side.SAMPLE:
        return pair_codes(data.sample.matrix, data.sample_pair_idx)
    if data.drug is None:
        msg = "drug features are required"
        raise ValueError(msg)
    return pair_codes(data.drug.matrix, data.drug_pair_idx)


def block_pair_codes(data: PredictorInput, block_name: str) -> tuple[np.ndarray, int]:
    """Return the category codes of a named sample block, one per pair.

    Raises ``ValueError`` (from :func:`pair_codes`) if the block is not one-hot.

    :param data: Featurized pairs.
    :param block_name: Name of the one-hot sample block to encode.
    :returns: The category code of each pair and the number of categories.
    :raises ValueError: If the sample side has no block named ``block_name``.
    """
    if block_name not in data.sample.blocks:
        msg = f"Required sample block {block_name!r} is missing"
        raise ValueError(msg)
    return pair_codes(data.sample.blocks[block_name].values, data.sample_pair_idx)


def joint_codes(codes_a: np.ndarray, n_a: int, codes_b: np.ndarray, n_b: int) -> tuple[np.ndarray, int]:
    """Combine two category codes per pair into the code of their combination.

    :param codes_a: Code of the first category per pair; ``-1`` for none.
    :param n_a: Number of first categories.
    :param codes_b: Code of the second category per pair; ``-1`` for none.
    :param n_b: Number of second categories.
    :returns: The combination code of each pair (``-1`` if either category is missing) and the number of
        combinations. Combination ``(a, b)`` has code ``a * n_b + b``.
    """
    joint = np.where((codes_a >= 0) & (codes_b >= 0), codes_a * n_b + codes_b, _NO_CATEGORY)
    return joint, n_a * n_b


def additive_effects(codes: np.ndarray, n_categories: int, y: np.ndarray, *, baseline: float) -> np.ndarray:
    """Return additive effects relative to *baseline* for each category.

    Raises ``ValueError`` if ``codes`` and ``y`` differ in length.

    :param codes: Category code per pair; ``-1`` for none.
    :param n_categories: Number of categories.
    :param y: Response of each pair.
    :param baseline: Value subtracted from the category means, normally the overall mean response.
    :returns: Mean offset per category; 0 for a category without pairs.
    """
    means, counts = _means_and_counts(codes, n_categories, y)
    return np.where(counts > 0, means - float(baseline), 0.0)


def effects_of(codes: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """Look up the effect of each pair's category.

    :param codes: Category code per pair; ``-1`` for none.
    :param effects: Effect of each category.
    :returns: Effect per pair; 0 for a pair without a category.
    :raises ValueError: If a code is not a row of ``effects``.
    """
    if codes.size and codes.max() >= len(effects):
        msg = "effect vector length must match the number of categories"
        raise ValueError(msg)
    if len(effects) == 0:
        return np.zeros(len(codes), dtype=np.float64)
    return np.where(codes >= 0, effects[np.maximum(codes, 0)], 0.0)


def _means_and_counts(codes: np.ndarray, n_categories: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean response and the number of pairs per category."""
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if codes.shape != target.shape:
        msg = "codes must match response length"
        raise ValueError(msg)
    known = codes >= 0
    counts = np.bincount(codes[known], minlength=n_categories)
    sums = np.bincount(codes[known], weights=target[known], minlength=n_categories)
    means = np.zeros(n_categories, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts
