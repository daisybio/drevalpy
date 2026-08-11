"""Drug identity featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.core.batch.feature_block import (
    BlockSpec,
    FeatureBlock,
    metadata_feature_block,
    numeric_feature_block,
)
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "identity",
    description="One-hot encoding of drug entity identifiers.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class DrugIdentityFeaturizer(DrugFeaturizer):
    """Encode drug IDs as dense one-hot vectors."""

    entity_id_only: ClassVar[bool] = True
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("identity", FeatureFormat.NUMERIC_MATRIX),)

    def __init__(self) -> None:
        """Initialize instance state."""
        self._encoder = OneHotCategoryEncoder()

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> DrugIdentityFeaturizer:
        """Fit on training data.

        :param source: Feature source (unused; identity only needs IDs).
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._encoder.fit_categories(ids)
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source (unused).
        :param entity_ids: entity ids.
        :returns: Result.
        """
        _ = source
        return self._encoder.transform(entity_ids)

    def _transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source (unused).
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            "identity": numeric_feature_block(self._transform(source, entity_ids)),
            "identity_categories": metadata_feature_block(
                np.asarray(self._encoder.categories, dtype=str),
            ),
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        self._encoder.set_state(state)
