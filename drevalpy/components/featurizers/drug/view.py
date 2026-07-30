"""Single-view drug featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "view",
    description="Pass through one dense drug view from a FeatureDataset.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ViewDrugFeaturizer(DrugFeaturizer):
    """Featurize one drug view without additional transformation."""

    def __init__(self, *, view: str = "fingerprints") -> None:
        self._view = view
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> ViewDrugFeaturizer:
        _ = context
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        return {
            self._view: numeric_feature_block(
                self.transform(features, entity_ids),
                feature_names=feature_names_for_view(features, self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        return self._output_dim
