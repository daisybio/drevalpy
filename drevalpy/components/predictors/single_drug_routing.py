"""Shared identity-based routing for single-drug predictors."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch

ROUTING_DRUG_FEATURIZER = "identity"
"""The drug featurizer per-drug routing is built on.

Not a choice: a single-drug predictor fits one estimator per drug, so it needs the drug's
identity to dispatch each pair to the right one. The identity featurizer emits exactly that,
and its output is used as a routing key rather than as features, which is why the design
matrix below is cell-line columns only.
"""


def routing_keys(batch: ModelInputBatch) -> np.ndarray:
    """Decode per-pair drug IDs from identity blocks.

    :param batch: Featurized batch with drug identity blocks.
    :returns: Drug id string per pair (empty string when unknown).
    :raises ValueError: If identity blocks are missing or misaligned.
    """
    identity_block = batch.drug_blocks.get(ROUTING_DRUG_FEATURIZER)
    categories_block = batch.drug_blocks.get(f"{ROUTING_DRUG_FEATURIZER}_categories")
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
    """Reject unknown drug identities during training.

    :param keys: Per-pair drug id strings from ``routing_keys``.
    :raises ValueError: If any entry is an empty string.
    """
    if np.any(keys == ""):
        msg = "Training pairs contain unknown drug identities"
        raise ValueError(msg)


def iter_drug_masks(batch: ModelInputBatch) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(drug_id, pair_mask)`` for each known drug in the batch.

    :param batch: Featurized batch with drug identity blocks.
    :yields: Drug identifier and boolean mask over response pairs.
    """
    keys = routing_keys(batch)
    for drug_id in np.unique(keys):
        if drug_id == "":
            continue
        yield str(drug_id), keys == drug_id
