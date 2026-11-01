"""Prepare entity features once and select usable response pairs for a fit.

:func:`usable_pairs` asks each featurizer once per fit which entities are usable
(``valid_mask`` on the unique entities of the training and early-stopping pairs) and keeps the
pairs whose sample and drug both are. :func:`prepare_features` owns everything around a
featurizer's ``fit`` and ``transform``: fitting on usable entities, transforming every entity
once, padding unusable entities back with missing values at prediction, the NaN-threshold
warning and the entity-order check. Featurizers therefore only ever see usable entity IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from drevalpy.components.featurizers.featurizer import Featurizer
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


def usable_pairs(
    response: ResponseBatch,
    stopping: ResponseBatch | None,
    sides: tuple[tuple[str, Featurizer | None, FeatureSource | None], ...],
) -> tuple[ResponseBatch, ResponseBatch | None]:
    """Keep the observed pairs whose entities are usable under each featurizer's missing-input policy.

    Each featurizer's ``valid_mask`` runs once, on the unique entities of the training and
    early-stopping pairs together.

    :param response: Training pairs.
    :param stopping: Early-stopping pairs, or ``None``.
    :param sides: ``(pair attribute, featurizer, source)`` per side: ``"sample_ids"`` or ``"drug_ids"``,
        and ``None`` for a side without a featurizer.
    :returns: The usable training and early-stopping pairs, in their original order.
    """
    keep = ~np.isnan(response.response)
    keep_stopping = None if stopping is None else ~np.isnan(stopping.response)
    for attr, featurizer, source in sides:
        if featurizer is None or source is None:
            continue
        pair_ids = getattr(response, attr)
        stopping_ids = None if stopping is None else getattr(stopping, attr)
        ids = unique_entity_ids(pair_ids if stopping_ids is None else np.concatenate([pair_ids, stopping_ids]))
        valid = featurizer.valid_mask(source, ids)
        _warn_if_above_threshold(featurizer, valid, "fit")
        if valid.all():
            continue
        keep &= np.isin(pair_ids, ids[valid])
        if keep_stopping is not None:
            keep_stopping &= np.isin(stopping_ids, ids[valid])
    return response.subset(keep), None if stopping is None else stopping.subset(keep_stopping)


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

    With *fit*, every ID must be usable: the pairs come from :func:`usable_pairs`. Without it
    (prediction), unusable entities (``valid_mask``) are left out of ``transform`` and get NaN
    rows (``None`` for object blocks).

    :param featurizer: Featurizer to run.
    :param source: Feature source of the featurizer's side.
    :param ids: Entities to compute, possibly repeated (one per pair).
    :param fit: Usable training and early-stopping pairs to fit on first; ``None`` only transforms.
    :returns: Features in stable unique entity order.
    :raises ValueError: If the featurizer does not support the source's side, requires early stopping
        and no early-stopping pair is given, or its transform does not keep the requested entity order.
    """
    _check_side(featurizer, source)
    ids = unique_entity_ids(ids)
    if fit is None:
        valid = featurizer.valid_mask(source, ids)
        _warn_if_above_threshold(featurizer, valid, "transform")
    else:
        if featurizer.requires_early_stopping and (fit.es_pair_ids is None or len(fit.es_pair_ids) == 0):
            raise ValueError(f"{type(featurizer).__name__} requires a usable early-stopping holdout")
        featurizer.fit(
            source,
            unique_entity_ids(fit.train_pair_ids),
            train_pair_ids=fit.train_pair_ids,
            es_pair_ids=fit.es_pair_ids,
        )
        valid = np.ones(len(ids), dtype=bool)
    features = featurizer.transform(source, ids[valid])
    if not np.array_equal(features.entity_ids, ids[valid]):
        raise ValueError(f"{type(featurizer).__name__} must preserve requested entity order")
    if valid.all():
        return features
    return FeaturizerResult(ids, _padded(features.blocks, valid), features.projection)
