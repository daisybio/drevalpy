"""Normalized proteomics featurizer for samples.

``ProteomicsMedianCenterAndImputeTransformer`` lives in
``_proteomics_transformer`` because it subclasses ``sklearn.base.BaseEstimator``,
which cannot be deferred past the ``class`` statement, so importing it here at
registration time would import ``sklearn``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


def log10_and_set_na(x: np.ndarray) -> np.ndarray:
    """Log10 transform and set NaN for infinite values.

    :param x: Input array; modified in place.
    :returns: ``log10`` of *x* with NaN where the result is infinite
    """
    x = np.log10(x)
    x[np.isinf(x)] = np.nan
    return x


@register(
    "normalizedProteomics",
    description="Proteomics view with log10 transform, median centering, and imputation.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class NormalizedProteomicsSampleFeaturizer(DenseViewSampleFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("proteomics",)
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(
        self,
        *,
        view: str = "proteomics",
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        """Bind the proteomics view and configure the normalizing transformer.

        :param view: Name of the proteomics view to read.
        :param proteomics_feature_threshold: Minimum fraction of non-missing values a protein needs
            to be kept.
        :param proteomics_n_features: Number of proteins kept when thresholding leaves fewer.
        :param proteomics_normalization_width: Width factor for the standard deviation of the
            imputation distribution.
        :param proteomics_normalization_downshift: Downshift of the imputation distribution's mean,
            in standard deviations.
        """
        from drevalpy.components.featurizers.sample._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer,
        )

        super().__init__(view=view)
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "proteomics_feature_threshold": {"type": "float", "low": 0.3, "high": 0.9, "default": 0.7},
            "proteomics_n_features": {"type": "int", "low": 500, "high": 2000, "default": 1000},
            "proteomics_normalization_downshift": {"type": "float", "low": 1.0, "high": 3.0, "default": 1.8},
            "proteomics_normalization_width": {"type": "float", "low": 0.1, "high": 0.6, "default": 0.3},
        }

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the median-centering transformer on log10 training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated sample identifiers to fit on.
        :returns: Number of retained proteins.
        """
        self._transformer.fit(log10_and_set_na(self._raw_matrix(source, entity_ids)))
        return len(self._transformer.protein_indices)

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Log10-transform *matrix* and apply the fitted transformer row by row.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Normalized feature matrix.
        """
        _ = source
        rows = [self._transformer.transform(row[None, :])[0] for row in log10_and_set_na(matrix)]
        return np.vstack(rows)

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state."""
        return {
            "proteomics_transformer": self._transformer,
            **self._dense_state(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        from drevalpy.components.featurizers.sample._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer,
        )

        transformer = state.get("proteomics_transformer")
        if isinstance(transformer, ProteomicsMedianCenterAndImputeTransformer):
            self._transformer = transformer
        self._restore_dense_state(state)
