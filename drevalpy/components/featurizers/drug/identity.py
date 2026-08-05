"""Drug identity featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec, FeatureBlock, metadata_feature_block, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset


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

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> DrugIdentityFeaturizer:
        """Fit on training data.

        :param features: features.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        """
        _ = features, context
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()), dtype=str)
        self._encoder.fit_categories(ids)
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        _ = features
        return self._encoder.transform(entity_ids)

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            "identity": numeric_feature_block(self.transform(features, entity_ids)),
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
