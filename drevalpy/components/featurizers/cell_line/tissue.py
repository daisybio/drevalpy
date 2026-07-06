"""Tissue metadata featurizer for cell lines."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import TISSUE_IDENTIFIER


@register_cell_line_featurizer(
    "tissue",
    description="One-hot encoding of tissue or lineage labels for cell-line entities.",
    category="native",
)
class TissueFeaturizer(CellLineFeaturizer):
    """Map each cell line to a dense one-hot tissue vector."""

    output_contract = FeatureContract(kind=FeatureKind.DENSE, scope="tissue")

    def __init__(self) -> None:
        self._encoder = OneHotCategoryEncoder()

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> TissueFeaturizer:
        if not any(TISSUE_IDENTIFIER in views for views in features.features.values()):
            msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
            raise ValueError(msg)
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()), dtype=str)
        tissues = features.get_feature_matrix(view=TISSUE_IDENTIFIER, identifiers=ids)
        self._encoder.fit_categories(np.asarray(tissues).reshape(-1))
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        tissues = features.get_feature_matrix(view=TISSUE_IDENTIFIER, identifiers=entity_ids)
        return self._encoder.transform(np.asarray(tissues).reshape(-1))

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "tissue": self.transform(features, entity_ids),
            "tissue_categories": np.asarray(self._encoder.categories, dtype=str),
        }

    @property
    def output_dim(self) -> int:
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        self._encoder.set_state(state)
