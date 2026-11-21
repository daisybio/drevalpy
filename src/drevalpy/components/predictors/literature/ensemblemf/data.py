"""Batch-to-tensor helpers of the EnsembleMF predictor."""

from __future__ import annotations

import numpy as np
import torch

from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

from .training import PairTensors

#: Share of the training pairs held out for early stopping when the run supplies none.
FALLBACK_VALIDATION_FRACTION = 0.1


def sorted_entities(result: FeaturizerResult, block: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order the entity rows of *block* by identifier, as the 1.6.1 model did.

    :param result: Entity table holding *block*.
    :param block: Name of the numeric block to read.
    :returns: Sorted identifiers, the matrix in that order and, for every original row, its sorted position.
    """
    order = np.argsort(result.entity_ids.astype(str), kind="stable")
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    return result.entity_ids[order], result.blocks[block].values[order], position


def drug_table(batch: ModelInputBatch) -> FeaturizerResult:
    """Narrow a batch's drug table to non-``None``.

    :param batch: Batch to read.
    :returns: The drug entity table.
    :raises RuntimeError: If the batch carries no drug features.
    """
    if batch.drug is None or batch.drug_pair_idx is None:
        msg = "EnsembleMF requires drug features (fingerprints)."
        raise RuntimeError(msg)
    return batch.drug


def _rows(pairs: PairTensors, rows: torch.Tensor) -> PairTensors:
    """Select pairs by position.

    :param pairs: Pairs to select from.
    :param rows: Positions to keep.
    :returns: The selected pairs.
    """
    return PairTensors(pairs.sample_idx[rows], pairs.drug_idx[rows], pairs.response[rows])


def split_pairs(train: PairTensors, validation: PairTensors | None, seed: int) -> tuple[PairTensors, PairTensors]:
    """Return the training and early-stopping pairs, holding out a seeded 10% when none were supplied.

    :param train: All training pairs.
    :param validation: Early-stopping pairs of the run, or ``None``.
    :param seed: Seed of the hold-out permutation; a negative seed draws an unseeded one.
    :returns: ``(training pairs, early-stopping pairs)``.
    :raises ValueError: If a fallback split is needed but fewer than two pairs exist.
    """
    if validation is not None and len(validation) > 0:
        return train, validation
    if len(train) < 2:
        msg = "EnsembleMF needs an early-stopping set or at least two training pairs to hold one out."
        raise ValueError(msg)
    generator = torch.Generator()
    if seed >= 0:
        generator.manual_seed(seed)
    else:
        generator.seed()
    perm = torch.randperm(len(train), generator=generator).to(train.response.device)
    n_val = max(1, int(FALLBACK_VALIDATION_FRACTION * len(train)))
    return _rows(train, perm[n_val:]), _rows(train, perm[:n_val])
