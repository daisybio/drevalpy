"""Normalized proteomics featurizer for cell lines."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.feature_source import FeatureSource
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.preprocessing import (
    ProteomicsMedianCenterAndImputeTransformer,
    log10_and_set_na,
)
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "normalizedProteomics",
    description="Proteomics view with log10 transform, median centering, and imputation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class NormalizedProteomicsCellLineFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("proteomics",)

    def __init__(
        self,
        *,
        view: str = "proteomics",
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        """Initialize instance state.

        :param view: view.
        :param proteomics_feature_threshold: proteomics feature threshold.
        :param proteomics_n_features: proteomics n features.
        :param proteomics_normalization_width: proteomics normalization width.
        :param proteomics_normalization_downshift: proteomics normalization downshift.
        """
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
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> NormalizedProteomicsCellLineFeaturizer:
        """Fit on training data.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else source.identifiers
        matrix = log10_and_set_na(source.get_view_matrix(self._view, np.unique(ids)))
        self._transformer.fit(matrix)
        self._output_dim = len(self._transformer.protein_indices)
        return self

    def _transform_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Get log10-transformed matrix and apply the fitted transformer row-by-row."""
        matrix = log10_and_set_na(source.get_view_matrix(self._view, entity_ids))
        rows = []
        for row in matrix:
            rows.append(self._transformer.transform(row[None, :])[0])
        return np.vstack(rows).astype(np.float32)

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return self._transform_matrix(source, entity_ids)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self.transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {
            "proteomics_transformer": self._transformer,
            "view": self._view,
            "output_dim": self._output_dim,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        transformer = state.get("proteomics_transformer")
        if isinstance(transformer, ProteomicsMedianCenterAndImputeTransformer):
            self._transformer = transformer
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
