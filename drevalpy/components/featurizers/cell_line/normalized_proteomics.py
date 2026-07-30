"""Normalized proteomics featurizer for cell lines."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.data.preprocessing import (
    ProteomicsMedianCenterAndImputeTransformer,
    prepare_proteomics,
)


@register_cell_line_featurizer(
    "normalizedProteomics",
    description="Proteomics view with log10 transform, median centering, and imputation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class NormalizedProteomicsCellLineFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    def __init__(
        self,
        *,
        view: str = "proteomics",
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        self._view = view
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> NormalizedProteomicsCellLineFeaturizer:
        _ = context
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        processed = prepare_proteomics(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(ids),
            training=True,
            transformer=self._transformer,
        )
        matrix = stack_view_matrix(processed, self._view, np.array(list(processed.features.keys())))
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        processed = prepare_proteomics(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(entity_ids),
            training=False,
            transformer=self._transformer,
        )
        return stack_view_matrix(processed, self._view, entity_ids).astype(np.float32)

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

    def get_state(self) -> dict[str, object]:
        return {
            "proteomics_transformer": self._transformer,
            "view": self._view,
            "output_dim": self._output_dim,
        }

    def set_state(self, state: dict[str, object]) -> None:
        transformer = state.get("proteomics_transformer")
        if isinstance(transformer, ProteomicsMedianCenterAndImputeTransformer):
            self._transformer = transformer
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
