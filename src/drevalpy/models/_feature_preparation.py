"""Prepare entity features once and select usable response pairs for a fit."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.feature_source import FeatureSource


def _entity_id_only_featurizer(featurizer: Featurizer | None) -> bool:
    return getattr(featurizer, "entity_id_only", False)


def transform_entities(
    featurizer: Featurizer | None,
    source: FeatureSource | None,
    ids: np.ndarray,
    *,
    train_ids: np.ndarray | None = None,
    es_ids: np.ndarray | None = None,
) -> FeaturizerResult:
    """Fit on training entities, then transform the requested unique entities once.

    :param featurizer: Configured component, or None for an empty projection.
    :param source: Raw entity features.
    :param ids: Entities needed for this operation, including stopping entities.
    :param train_ids: Training pair identifiers; None skips fitting.
    :param es_ids: Optional stopping pair identifiers.
    :returns: Canonical features in stable unique entity order.
    :raises ValueError: If a featurizer is configured but *source* is ``None`` and the featurizer needs it.
    """
    if featurizer is None:
        return FeaturizerResult(unique_entity_ids(ids), {}, ())
    if source is None and not _entity_id_only_featurizer(featurizer):
        raise ValueError("Feature source is required when a featurizer is configured")
    if train_ids is not None:
        featurizer.fit(
            source,
            entity_ids=unique_entity_ids(train_ids),
            pair_expanded_ids=train_ids,
            pair_expanded_es_ids=es_ids,
        )
    ids = unique_entity_ids(ids)
    return featurizer.transform_features(source, ids)


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
        if featurizer is not None:
            keep &= featurizer._detect_valid(source, ids)
    return response.subset(keep)
