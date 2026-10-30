"""Prepare entity features once and select usable response pairs for a fit.

:func:`prepare_features` owns everything around a featurizer's ``fit`` and ``transform``:
which entities are usable (``valid_mask``), the NaN-threshold warning, the stored-output
lookup, fitting, transforming, padding unusable entities back with missing values, and the
entity-order check. Featurizers therefore only ever see usable entity IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from drevalpy.components.featurizers.featurizer import (
    Featurizer,
    fit_unless_stored,
    transform_or_stored,
    usable,
)
from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.feature_source import FeatureSource

_logger = get_logger(__name__)


@dataclass(frozen=True)
class FitIds:
    """The pairs a featurizer is fitted on, as the entity ID of each pair."""

    #: Entity ID of every training pair, repeated per pair.
    train_pair_ids: np.ndarray
    #: Entity ID of every early-stopping pair, or ``None`` without an early-stopping holdout.
    es_pair_ids: np.ndarray | None = None


def unique_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Return unique entity IDs in first-seen order.

    :param entity_ids: Entity IDs, possibly repeated.
    :returns: Each ID once, in the order it first appears.
    """
    uniq, index = np.unique(entity_ids, return_index=True)
    return uniq[index.argsort()]


def _warn_if_above_threshold(featurizer: Featurizer, valid: np.ndarray, step: str) -> None:
    """Log a warning when the fraction of unusable entities exceeds the featurizer's threshold.

    :param featurizer: Featurizer whose ``nan_threshold`` applies.
    :param valid: Usable-entity mask.
    :param step: ``"fit"`` or ``"transform"``, named in the message.
    """
    if len(valid) == 0:
        return
    invalid_frac = 1.0 - valid.mean()
    if invalid_frac > featurizer.nan_threshold:
        _logger.warning(
            "%s: %.0f%% of inputs are invalid (threshold: %.0f%%)",
            f"{type(featurizer).__name__}.{step}",
            invalid_frac * 100,
            featurizer.nan_threshold * 100,
        )


def _check_side(featurizer: Featurizer, source: FeatureSource) -> None:
    """Raise when *featurizer* does not support the side of *source*.

    :param featurizer: Featurizer to run.
    :param source: Feature source of one side.
    :raises ValueError: If the side is not among the featurizer's ``sides``.
    """
    if source.side not in featurizer.sides:
        msg = (
            f"{type(featurizer).__name__} cannot be used on the {source.side} side; "
            f"it supports only: {', '.join(sorted(featurizer.sides))}"
        )
        raise ValueError(msg)


def _fit(featurizer: Featurizer, source: FeatureSource, fit: FitIds) -> None:
    """Fit *featurizer* on the usable entities of the training pairs, unless its output is stored.

    :param featurizer: Featurizer to fit.
    :param source: Feature source of the featurizer's side.
    :param fit: Training and early-stopping pairs.
    :raises ValueError: If the featurizer requires early stopping and no usable early-stopping pair remains.
    """
    train_ids = unique_entity_ids(fit.train_pair_ids)
    valid = usable(featurizer, source, train_ids)
    _warn_if_above_threshold(featurizer, valid, "fit")
    train_ids = train_ids if valid.all() else train_ids[valid]
    train_pair_ids = fit.train_pair_ids[np.isin(fit.train_pair_ids, train_ids)]
    es_pair_ids = fit.es_pair_ids
    if es_pair_ids is not None:
        es_pair_ids = es_pair_ids[usable(featurizer, source, es_pair_ids)]
    if featurizer.requires_early_stopping and (es_pair_ids is None or len(es_pair_ids) == 0):
        raise ValueError(f"{type(featurizer).__name__} requires a usable early-stopping holdout")
    fit_unless_stored(featurizer, source, train_ids, train_pair_ids=train_pair_ids, es_pair_ids=es_pair_ids)


def _padded_block(block: FeatureBlock, valid: np.ndarray) -> FeatureBlock:
    """Restore entity rows with dtype-appropriate missing values.

    :param block: Usable-only entity block.
    :param valid: Requested positions holding usable rows.
    :returns: Padded block retaining feature names and annotations.
    """
    numeric = block.values.dtype.kind in "biufc"
    full = np.full(
        (len(valid), *block.values.shape[1:]),
        np.nan if numeric else None,
        dtype=np.result_type(block.values.dtype, np.float32) if numeric else object,
    )
    full[valid] = block.values
    return replace(block, values=full)


def _padded(blocks: Mapping[str, FeatureBlock], valid: np.ndarray) -> dict[str, FeatureBlock]:
    """Pad every entity-aligned block back to all requested entities; pass the others through.

    :param blocks: Blocks computed on the usable entities only.
    :param valid: Requested positions holding usable rows.
    :returns: Blocks aligned to every requested entity.
    """
    return {name: _padded_block(block, valid) if block.entity_aligned else block for name, block in blocks.items()}


def prepare_features(
    featurizer: Featurizer,
    source: FeatureSource,
    ids: np.ndarray,
    *,
    fit: FitIds | None,
) -> FeaturizerResult:
    """Fit *featurizer* when *fit* is given, then compute the features of every entity in *ids* once.

    Unusable entities (``valid_mask``) are left out of ``fit`` and ``transform`` and get NaN
    rows (``None`` for object blocks). When the source stores the featurizer's output for its
    ``stored_params()``, the featurizer is neither fitted nor transformed.

    :param featurizer: Featurizer to run.
    :param source: Feature source of the featurizer's side.
    :param ids: Entities to compute, possibly repeated (one per pair).
    :param fit: Training pairs to fit on first; ``None`` only transforms.
    :returns: Features in stable unique entity order.
    :raises ValueError: If the featurizer does not support the source's side, or its transform
        does not keep the requested entity order.
    """
    _check_side(featurizer, source)
    if fit is not None:
        _fit(featurizer, source, fit)
    ids = unique_entity_ids(ids)
    valid = usable(featurizer, source, ids)
    _warn_if_above_threshold(featurizer, valid, "transform")
    features = transform_or_stored(featurizer, source, ids[valid])
    if not np.array_equal(features.entity_ids, ids[valid]):
        raise ValueError(f"{type(featurizer).__name__} must preserve requested entity order")
    if valid.all():
        return features
    return FeaturizerResult(ids, _padded(features.blocks, valid), features.projection)


def usable_responses(
    response: ResponseBatch,
    sample_input: FeatureSource,
    drug_input: FeatureSource | None,
    sample_featurizer: Featurizer | None,
    drug_featurizer: Featurizer | None,
) -> ResponseBatch:
    """Select observed pairs eligible under each component's missing-input policy.

    :param response: Requested targets and pair identifiers.
    :param sample_input: Raw sample features.
    :param drug_input: Optional raw drug features.
    :param sample_featurizer: Sample eligibility policy.
    :param drug_featurizer: Optional drug eligibility policy.
    :returns: Aligned observed, usable pairs in their original order.
    """
    keep = ~np.isnan(response.response)
    for ids, featurizer, source in (
        (response.sample_ids, sample_featurizer, sample_input),
        (response.drug_ids, drug_featurizer, drug_input),
    ):
        if featurizer is not None and source is not None:
            keep &= usable(featurizer, source, ids)
    return response.subset(keep)
