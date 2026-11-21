"""Prepare entity features once and select usable response pairs for a fit.

:func:`usable_pairs` asks each featurizer's ``valid_mask`` once per fit and keeps pairs whose
entities are usable. :func:`prepare_features` runs ``fit``/``transform`` on unique entities,
checks side and entity order, logs NaN-threshold warnings, and pads unusable rows back in at
prediction time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput

if TYPE_CHECKING:
    from drevalpy.components.featurizers.featurizer import Featurizer
    from drevalpy.types.data.feature_source import FeatureSource
    from drevalpy.types.data.response_table import ResponseTable

_logger = get_logger(__name__)


def unique_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Return unique entity IDs in first-seen order.

    :param entity_ids: Entity IDs, possibly repeated.
    :returns: Each ID once, in the order it first appears.
    """
    uniq, index = np.unique(entity_ids, return_index=True)
    return uniq[index.argsort()]


def _warn_if_above_threshold(featurizer: Featurizer, valid: np.ndarray, step: str) -> None:
    """Log a warning when the invalid-entity share exceeds ``featurizer.nan_threshold``."""
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


def usable_pairs(
    response: ResponseTable,
    stopping: ResponseTable | None,
    sides: tuple[tuple[str, Featurizer | None, FeatureSource | None], ...],
) -> tuple[ResponseTable, ResponseTable | None]:
    """Keep observed pairs whose entities are usable under each featurizer's ``valid_mask``.

    :param response: Training pairs.
    :param stopping: Early-stopping pairs, or ``None``.
    :param sides: ``(pair attribute, featurizer, source)`` per side.
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
    return response[keep], None if stopping is None else stopping[keep_stopping]


def _padded_block(block: FeatureBlock, valid: np.ndarray) -> FeatureBlock:
    """Restore entity rows with dtype-appropriate missing values (NaN or None)."""
    numeric = block.values.dtype.kind in "biufc"
    full = np.full(
        (len(valid), *block.values.shape[1:]),
        np.nan if numeric else None,
        dtype=np.result_type(block.values.dtype, np.float32) if numeric else object,
    )
    full[valid] = block.values
    return replace(block, values=full)


def prepare_features(
    featurizer: Featurizer,
    source: FeatureSource,
    ids: np.ndarray,
    *,
    train_pair_ids: np.ndarray | None = None,
    es_pair_ids: np.ndarray | None = None,
) -> FeaturizerOutput:
    """Fit *featurizer* when *train_pair_ids* is given, then transform every unique entity in *ids*.

    When *train_pair_ids* is given, every ID is assumed usable (from :func:`usable_pairs`).
    Otherwise (prediction), unusable entities are excluded from ``transform`` and padded back
    with NaN (or ``None`` for object blocks).

    :param featurizer: Featurizer to run.
    :param source: Feature source of the featurizer's side.
    :param ids: Entities to compute, possibly repeated (one per pair).
    :param train_pair_ids: Usable training pair entity IDs to fit on; ``None`` only transforms.
    :param es_pair_ids: Usable early-stopping pair entity IDs, or ``None``.
    :returns: Features in stable unique entity order.
    :raises ValueError: If the featurizer side mismatches *source*, requires early stopping
        without a holdout, or reorders entities.
    """
    side = getattr(featurizer, "side", None)
    if side is not None and side != source.side:
        msg = f"{type(featurizer).__name__} is a {side} featurizer and cannot be used on the {source.side} side"
        raise ValueError(msg)
    ids = unique_entity_ids(ids)
    if train_pair_ids is None:
        valid = featurizer.valid_mask(source, ids)
        _warn_if_above_threshold(featurizer, valid, "transform")
    else:
        if featurizer.requires_early_stopping and (es_pair_ids is None or len(es_pair_ids) == 0):
            raise ValueError(f"{type(featurizer).__name__} requires a usable early-stopping holdout")
        featurizer.fit(
            source,
            unique_entity_ids(train_pair_ids),
            train_pair_ids=train_pair_ids,
            es_pair_ids=es_pair_ids,
        )
        valid = np.ones(len(ids), dtype=bool)
    features = featurizer.transform(source, ids[valid])
    if not np.array_equal(features.entity_ids, ids[valid]):
        raise ValueError(f"{type(featurizer).__name__} must preserve requested entity order")
    if valid.all():
        return features
    padded = {
        name: _padded_block(block, valid) if block.entity_aligned else block for name, block in features.blocks.items()
    }
    return FeaturizerOutput(ids, padded)
