"""Shared identity-based routing for single-drug predictors."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch


def routing_keys(batch: ModelInputBatch) -> np.ndarray:
    """Decode per-pair drug IDs from identity blocks."""
    identity_block = batch.drug_blocks.get("identity")
    categories_block = batch.drug_blocks.get("identity_categories")
    if identity_block is None or categories_block is None or batch.drug_pair_idx is None:
        msg = "Single-drug predictors require drug identity features for per-drug routing"
        raise ValueError(msg)

    identity_matrix = np.asarray(identity_block.values)
    category_ids = np.asarray(categories_block.values, dtype=str).reshape(-1)
    if identity_matrix.ndim != 2 or identity_matrix.shape[1] != len(category_ids):
        msg = "Drug identity features and identity categories are misaligned"
        raise ValueError(msg)

    pair_identity = identity_matrix[batch.drug_pair_idx]
    known = np.isclose(pair_identity.sum(axis=1), 1.0)
    keys = np.full(batch.n_pairs, "", dtype=object)
    if np.any(known):
        keys[known] = category_ids[np.argmax(pair_identity[known], axis=1)]
    return np.asarray(keys, dtype=str)


def require_known_training_keys(keys: np.ndarray) -> None:
    """Reject unknown drug identities during training."""
    if np.any(keys == ""):
        msg = "Training pairs contain unknown drug identities"
        raise ValueError(msg)


def iter_drug_masks(batch: ModelInputBatch) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(drug_id, pair_mask)`` for each known drug in the batch.

    :yields: drug identifier and boolean mask over response pairs
    """
    keys = routing_keys(batch)
    for drug_id in np.unique(keys):
        if drug_id == "":
            continue
        yield str(drug_id), keys == drug_id
