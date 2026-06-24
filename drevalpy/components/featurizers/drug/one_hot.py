"""One-hot drug identifier featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "oneHot",
    description="One-hot encoding of drug identifiers.",
    category="native",
)
class OneHotDrugFeaturizer(DrugFeaturizer):
    """Fit a one-hot space over all drugs present in the feature dataset."""

    def __init__(self) -> None:
        self._drug_to_index: dict[str, int] = {}
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> OneHotDrugFeaturizer:
        drug_ids = sorted(features.features.keys())
        self._drug_to_index = {str(drug_id): index for index, drug_id in enumerate(drug_ids)}
        self._output_dim = len(self._drug_to_index)
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(entity_ids), self._output_dim), dtype=np.float32)
        for row, drug_id in enumerate(entity_ids):
            index = self._drug_to_index.get(str(drug_id))
            if index is None:
                msg = f"Unknown drug id {drug_id!r} for oneHot featurizer"
                raise KeyError(msg)
            matrix[row, index] = 1.0
        return matrix

    @property
    def output_dim(self) -> int:
        return self._output_dim
