"""Drug identity featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset


@register_drug_featurizer(
    "identity",
    description="One-hot encoding of drug entity identifiers.",
    category="native",
    contract=FeatureKind.DENSE,
)
class DrugIdentityFeaturizer(DrugFeaturizer):
    """Encode drug IDs as dense one-hot vectors."""

    entity_id_only: ClassVar[bool] = True

    def __init__(self) -> None:
        self._encoder = OneHotCategoryEncoder()

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> DrugIdentityFeaturizer:
        _ = features
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()), dtype=str)
        self._encoder.fit_categories(ids)
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        _ = features
        return self._encoder.transform(entity_ids)

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "identity": self.transform(features, entity_ids),
            "identity_categories": np.asarray(self._encoder.categories, dtype=str),
        }

    @property
    def output_dim(self) -> int:
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        self._encoder.set_state(state)
